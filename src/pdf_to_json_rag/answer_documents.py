"""Document selection, semantic synthesis, and comparison answers."""

from __future__ import annotations

import re
from pathlib import Path

from .answer_contracts import build_answer_contract
from .document_inventory import build_inventory_summary, get_inventory_entry
from .document_semantics import (
    interpret_document_semantics,
    query_semantic_preferences,
    relationship_signal,
)
from .intent_config import (
    get_document_profile,
    matching_source_doc_ids,
)
from .query_planning import plan_query
from .schemas import ChunkRecord, DocumentRecord


from .answer_evidence import (
    DOCUMENT_SELECTION_STRATEGIES,
    SOURCE_ANCHORED_HINTS,
    UNSUPPORTED_ENTITY_TERMS,
    _normalize_text,
    _preferred_source_doc_id,
    _query_terms,
    _selection_limit,
    _specific_query_terms,
    _split_sentences,
)

from .answer_models import (
    DocumentSelection,
    DocumentSynthesis,
    EvidenceSentence,
)


def _humanize_source_label(source_pdf: str) -> str:
    stem = Path(source_pdf).stem
    label = stem.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def _display_label_for_chunk(chunk: ChunkRecord) -> str:
    profile = get_document_profile(chunk.doc_id)
    if profile:
        return profile.label
    return _humanize_source_label(chunk.doc_id)


def _is_low_quality_document_title(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", title).strip()
    lowered = normalized.lower()
    if lowered.startswith("doi:"):
        return True
    if ".indd" in lowered:
        return True
    if lowered.startswith("since january 2020 elsevier has created"):
        return True
    return False


def _document_label(doc_id: str, chunk_root: Path) -> str:
    record = _load_document_record(doc_id, chunk_root)
    profile = get_document_profile(doc_id)
    if record and record.title:
        normalized_title = re.sub(r"\s+", " ", record.title).strip()
        if not _is_low_quality_document_title(normalized_title):
            return normalized_title
    if profile:
        return profile.label
    return _humanize_source_label(doc_id)


def _structured_source_summary(chunks: list[ChunkRecord]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    summary: list[tuple[str, str]] = []
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        summary.append((chunk.doc_id, _display_label_for_chunk(chunk)))
    return summary


def _load_document_record(doc_id: str, chunk_root: Path) -> DocumentRecord | None:
    document_path = (
        chunk_root.expanduser().resolve().parent
        / "documents"
        / f"{doc_id}.document.json"
    )
    if not document_path.exists():
        return None
    try:
        return DocumentRecord.model_validate_json(
            document_path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def _extract_document_topics(chunks: list[ChunkRecord], doc_id: str) -> list[str]:
    topics: list[str] = []
    seen: set[str] = set()
    chapter_pattern = re.compile(r"Chapter\s+\d+\s+([A-Za-z][A-Za-z0-9 \-]{2,80})")
    for chunk in chunks:
        if chunk.doc_id != doc_id:
            continue
        section_title = (chunk.section_title or "").strip()
        section_upper = section_title.upper()
        if (
            section_title
            and not section_upper.startswith(
                ("CONTENTS", "INDEX", "BIBLIOGRAPHY", "FOREWORD")
            )
            and not (section_upper == section_title and len(section_title) <= 5)
            and len(section_title) >= 6
            and section_title not in seen
        ):
            seen.add(section_title)
            topics.append(section_title)
        for match in chapter_pattern.findall(chunk.text):
            topic = match.strip()
            if topic.upper().startswith(("CONTENTS", "INDEX", "BIBLIOGRAPHY")):
                continue
            if topic.upper() == topic and len(topic) <= 5:
                continue
            if topic not in seen:
                seen.add(topic)
                topics.append(topic)
        if len(topics) >= 5:
            break
    return topics[:5]


def _clean_topic_label(topic: str) -> str:
    cleaned = re.sub(r"^Chapter\s+\d+\s+", "", topic).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or topic


def _document_summary_cues(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> list[str]:
    record = _load_document_record(doc_id, chunk_root)
    profile = get_document_profile(doc_id)
    cues: list[str] = []
    seen: set[str] = set()
    doc_titles = {
        item.lower()
        for item in (
            record.title if record and record.title else None,
            profile.label if profile else None,
        )
        if item
    }

    def maybe_add(value: str) -> None:
        cue = _clean_topic_label(value).strip()
        cue_upper = cue.upper()
        if not cue:
            return
        if cue.lower() in doc_titles:
            return
        if cue_upper in {
            "THE CENTRE FOR HUMANITARIAN DATA",
            "OCHA CENTRE FOR HUMANITARIAN DATA",
            "GUIDANCE NOTE SERIES",
        }:
            return
        if cue_upper.startswith(("CONTENTS", "INDEX", "BIBLIOGRAPHY", "FOREWORD")):
            return
        if cue in seen:
            return
        seen.add(cue)
        cues.append(cue)

    if record:
        for cue in record.summary_cues:
            maybe_add(cue)
        if not cues:
            for cue in record.toc[:12]:
                maybe_add(cue)
    if not cues:
        for cue in _extract_document_topics(top_k_hits, doc_id):
            maybe_add(cue)
    return cues[:6]


def _document_discovery_terms(doc_id: str, chunk_root: Path) -> list[str]:
    record = _load_document_record(doc_id, chunk_root)
    if record and record.discovery_terms:
        return record.discovery_terms[:20]
    profile = get_document_profile(doc_id)
    if profile:
        return sorted(profile.topical_terms)
    return []


def _document_semantics(doc_id: str, top_k_hits: list[ChunkRecord], chunk_root: Path):
    record = _load_document_record(doc_id, chunk_root)
    entry = get_inventory_entry(doc_id)
    title = (
        (record.title if record and record.title else None)
        or (entry.title if entry else None)
        or _document_label(doc_id, chunk_root)
    )
    toc = record.toc if record else []
    summary_cues = (
        list(record.summary_cues)
        if record and record.summary_cues
        else list(entry.summary_cues if entry else ())
    )
    discovery_terms = (
        list(record.discovery_terms)
        if record and record.discovery_terms
        else list(entry.discovery_terms if entry else ())
    )
    if not summary_cues:
        summary_cues = _document_summary_cues(doc_id, top_k_hits, chunk_root)
    if not discovery_terms:
        discovery_terms = _document_discovery_terms(doc_id, chunk_root)
    return interpret_document_semantics(
        source_pdf=record.source_pdf if record else "",
        title=title,
        toc=toc,
        summary_cues=summary_cues,
        discovery_terms=discovery_terms,
        leading_block_lines=[],
        metadata_values=[],
        page_count=record.page_count if record else 0,
        document_type=(record.document_type if record else None)
        or (entry.document_type if entry else None),
        document_purpose=(record.document_purpose if record else None)
        or (entry.document_purpose if entry else None),
        audience=(record.audience if record else None)
        or (entry.audience if entry else None),
        evidence_style=(record.evidence_style if record else None)
        or (entry.evidence_style if entry else None),
        structure_style=(record.structure_style if record else None)
        or (entry.structure_style if entry else None),
        facet_terms=(record.facet_terms if record and record.facet_terms else None)
        or (list(entry.facet_terms) if entry else None),
        inventory_summary=(record.inventory_summary if record else None)
        or (entry.inventory_summary if entry else None),
        document_family=(record.document_family if record else None)
        or (entry.document_family if entry else None),
        coverage_terms=(
            record.coverage_terms if record and record.coverage_terms else None
        )
        or (list(entry.coverage_terms) if entry else None),
        coverage_summary=(record.coverage_summary if record else None)
        or (entry.coverage_summary if entry else None),
    )


def _document_facets(doc_id: str, chunk_root: Path) -> dict[str, str | list[str]]:
    semantics = _document_semantics(doc_id, [], chunk_root)
    return {
        "document_type": semantics.document_type,
        "document_purpose": semantics.document_purpose,
        "audience": semantics.audience,
        "evidence_style": semantics.evidence_style,
        "structure_style": semantics.structure_style,
        "facet_terms": list(semantics.facet_terms[:10]),
    }


def _document_facet_tokens(doc_id: str, chunk_root: Path) -> set[str]:
    facets = _document_facets(doc_id, chunk_root)
    tokens: set[str] = set()
    for key in (
        "document_type",
        "document_purpose",
        "audience",
        "evidence_style",
        "structure_style",
    ):
        tokens.update(re.findall(r"[a-zA-Z]{3,}", str(facets.get(key, "")).lower()))
    for item in facets.get("facet_terms", []):
        if isinstance(item, str):
            tokens.update(re.findall(r"[a-zA-Z]{3,}", item.lower()))
    return tokens


def _document_overview_fragments(
    doc_id: str, top_k_hits: list[ChunkRecord], chunk_root: Path
) -> list[str]:
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    fragments: list[str] = []
    topics = list(semantics.coverage_terms) or _document_summary_cues(
        doc_id, top_k_hits, chunk_root
    )

    if semantics.document_type:
        fragments.append(f"it is {_indefinite_phrase(semantics.document_type)}")
    if semantics.document_family:
        fragments.append(
            f"it belongs to the {semantics.document_family.replace('_', ' ')} family"
        )
    if semantics.document_purpose:
        fragments.append(
            f"its main purpose is {semantics.document_purpose.replace('_', ' ')}"
        )
    if semantics.audience and semantics.audience != "general_professional":
        fragments.append(f"it appears aimed at {semantics.audience.replace('_', ' ')}")
    if topics:
        fragments.append("it covers topics such as " + ", ".join(topics[:4]))
    return fragments


def _document_profile_summary(doc_id: str, chunk_root: Path) -> str:
    semantics = _document_semantics(doc_id, [], chunk_root)
    parts: list[str] = []
    if semantics.document_purpose:
        parts.append(semantics.document_purpose.replace("_", " "))
    if semantics.evidence_style:
        parts.append(semantics.evidence_style.replace("_", " "))
    if semantics.audience and semantics.audience != "general_professional":
        parts.append(f"for {semantics.audience.replace('_', ' ')}")
    if semantics.structure_style:
        parts.append(semantics.structure_style.replace("_", " "))
    if semantics.document_family:
        parts.append(semantics.document_family.replace("_", " "))
    if semantics.coverage_terms:
        parts.append("covering " + ", ".join(semantics.coverage_terms[:3]))
    if parts:
        return "; ".join(parts)
    return "general reference"


def _document_difference_summary(doc_ids: list[str], chunk_root: Path) -> str | None:
    if len(doc_ids) < 2:
        return None
    first = _document_semantics(doc_ids[0], [], chunk_root)
    second = _document_semantics(doc_ids[1], [], chunk_root)
    first_purpose = first.document_purpose.replace("_", " ")
    second_purpose = second.document_purpose.replace("_", " ")
    first_style = first.evidence_style.replace("_", " ")
    second_style = second.evidence_style.replace("_", " ")
    first_audience = first.audience.replace("_", " ")
    second_audience = second.audience.replace("_", " ")
    first_structure = first.structure_style.replace("_", " ")
    second_structure = second.structure_style.replace("_", " ")
    first_family = first.document_family.replace("_", " ")
    second_family = second.document_family.replace("_", " ")

    differences: list[str] = []
    if first_purpose and second_purpose and first_purpose != second_purpose:
        differences.append(
            f"their purposes differ ({first_purpose} vs {second_purpose})"
        )
    if first_style and second_style and first_style != second_style:
        differences.append(
            f"their evidence styles differ ({first_style} vs {second_style})"
        )
    if (
        first_audience
        and second_audience
        and first_audience != second_audience
        and "general professional" not in {first_audience, second_audience}
    ):
        differences.append(
            f"their audiences differ ({first_audience} vs {second_audience})"
        )
    if first_structure and second_structure and first_structure != second_structure:
        differences.append(
            f"their structures differ ({first_structure} vs {second_structure})"
        )
    if first_family and second_family and first_family != second_family:
        differences.append(
            f"they come from different document families ({first_family} vs {second_family})"
        )
    if not differences:
        return None
    return "Both are relevant, but " + " and ".join(differences) + "."


def _document_relationship_signal(
    doc_ids: list[str], chunk_root: Path
) -> tuple[str | None, str | None]:
    if len(doc_ids) < 2:
        return None, None
    first_entry = get_inventory_entry(doc_ids[0])
    second_entry = get_inventory_entry(doc_ids[1])
    first = _document_semantics(doc_ids[0], [], chunk_root)
    second = _document_semantics(doc_ids[1], [], chunk_root)
    label, sentence, _ = relationship_signal(
        first=first,
        second=second,
        first_topical_terms=set(first_entry.topical_terms) if first_entry else set(),
        second_topical_terms=set(second_entry.topical_terms) if second_entry else set(),
    )
    return label, sentence


def _document_inventory_summary(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
    *,
    include_label: bool = False,
) -> str:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    summary = (
        semantics.inventory_summary.strip()
        if semantics.inventory_summary
        else build_inventory_summary(
            title=label,
            document_type=semantics.document_type,
            document_purpose=semantics.document_purpose,
            audience=semantics.audience,
            evidence_style=semantics.evidence_style,
            structure_style=semantics.structure_style,
            summary_cues=semantics.summary_cues,
        )
    )
    prefix = f"{label} | "
    if not include_label and summary.startswith(prefix):
        return summary[len(prefix) :]
    if include_label:
        return summary if summary.startswith(prefix) else f"{label} | {summary}"
    return summary


def _document_confidence(
    doc_id: str, chunk_root: Path
) -> tuple[float | None, float | None]:
    record = _load_document_record(doc_id, chunk_root)
    if not record:
        return None, None
    return record.structure_confidence, record.layout_confidence


def _classification_assessment(
    doc_id: str, top_k_hits: list[ChunkRecord], chunk_root: Path
) -> dict[str, object]:
    record = _load_document_record(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    structure_confidence = getattr(record, "structure_confidence", None)
    layout_confidence = getattr(record, "layout_confidence", None)
    semantic_confidence = getattr(record, "semantic_confidence", None)
    if semantic_confidence is None:
        semantic_confidence = semantics.semantic_confidence
    weighted_total = 0.65 * float(semantic_confidence)
    weight_sum = 0.65
    if structure_confidence is not None:
        weighted_total += 0.2 * float(structure_confidence)
        weight_sum += 0.2
    if layout_confidence is not None:
        weighted_total += 0.15 * float(layout_confidence)
        weight_sum += 0.15
    classification_confidence = round(
        min(0.95, weighted_total / max(weight_sum, 0.01)), 3
    )
    if classification_confidence >= 0.8:
        label = "high"
        status = "well_supported"
    elif classification_confidence >= 0.62:
        label = "moderate"
        status = "provisional"
    else:
        label = "low"
        status = "uncertain"
    semantic_warnings = list(
        getattr(record, "semantic_warnings", []) or semantics.semantic_warnings
    )
    trust_policy = "stable_semantic_classification"
    if status == "uncertain":
        trust_policy = "heuristic_semantic_guess"
    elif status == "provisional" or semantic_warnings:
        trust_policy = "confidence_aware_provisional_classification"
    return {
        "classification_confidence": classification_confidence,
        "classification_confidence_label": label,
        "classification_status": status,
        "trust_policy": trust_policy,
        "semantic_confidence": round(float(semantic_confidence), 3),
        "semantic_confidence_label": getattr(record, "semantic_confidence_label", None)
        or semantics.semantic_confidence_label,
        "structure_confidence": structure_confidence,
        "layout_confidence": layout_confidence,
        "semantic_rationale": list(
            getattr(record, "semantic_rationale", []) or semantics.semantic_rationale
        ),
        "semantic_warnings": semantic_warnings,
    }


def _semantic_phrase(value: str | None, fallback: str = "document") -> str:
    if not value:
        return fallback
    return value.replace("_", " ")


def _indefinite_phrase(value: str | None, fallback: str = "document") -> str:
    phrase = _semantic_phrase(value, fallback).strip()
    if not phrase:
        return fallback
    article = "an" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {phrase}"


def _confidence_aware_language(assessment: dict[str, object]) -> dict[str, str]:
    label = assessment.get("classification_confidence_label")
    if label == "high":
        return {
            "type_verb": "is",
            "purpose_verb": "is",
            "audience_verb": "is intended for",
            "overview_type_verb": "is",
            "overview_purpose_verb": "its main purpose is",
            "overview_audience_verb": "it is aimed at",
        }
    if label == "moderate":
        return {
            "type_verb": "appears to be",
            "purpose_verb": "appears to be",
            "audience_verb": "appears intended for",
            "overview_type_verb": "appears to be",
            "overview_purpose_verb": "its likely purpose is",
            "overview_audience_verb": "it likely targets",
        }
    return {
        "type_verb": "is likely",
        "purpose_verb": "is provisionally",
        "audience_verb": "is provisionally intended for",
        "overview_type_verb": "is likely",
        "overview_purpose_verb": "its apparent purpose is",
        "overview_audience_verb": "it may be aimed at",
    }


def _render_document_overview(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str, list[str], dict[str, object]]:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    fragments: list[str] = []
    cues: list[str] = []
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    low_confidence = assessment["classification_confidence_label"] == "low"
    language = _confidence_aware_language(assessment)

    type_label = (
        semantics.document_type.replace("_", " ")
        if semantics.document_type
        else "document"
    )
    fragments.append(
        f"{label} {language['overview_type_verb']} {_indefinite_phrase(semantics.document_type)}"
    )
    cues.append(type_label)

    if semantics.document_purpose:
        purpose = semantics.document_purpose.replace("_", " ")
        fragments.append(f"{language['overview_purpose_verb']} {purpose}")
        cues.append(purpose)
    if semantics.audience and semantics.audience != "general_professional":
        audience = semantics.audience.replace("_", " ")
        fragments.append(f"{language['overview_audience_verb']} {audience}")
        cues.append(audience)

    coverage_terms = list(semantics.coverage_terms[:4])
    if coverage_terms:
        fragments.append("it covers " + ", ".join(coverage_terms))
        cues.extend(coverage_terms)
    elif semantics.coverage_summary:
        coverage_summary = semantics.coverage_summary.removeprefix(
            "covers topics such as "
        ).strip()
        if coverage_summary:
            fragments.append(f"it covers {coverage_summary}")
            cues.append(coverage_summary)

    section_titles = _document_summary_cues(doc_id, top_k_hits, chunk_root)[:3]
    if section_titles:
        fragments.append("key sections include " + ", ".join(section_titles))
        cues.extend(section_titles)
    elif low_confidence:
        fragments.append(
            "the recovered structure is limited, so this overview should be treated as provisional"
        )

    answer = (
        ". ".join(
            part[:1].upper() + part[1:] if index else part
            for index, part in enumerate(fragments)
        )
        + "."
    )
    contract = build_answer_contract(
        mode="document_overview",
        primary_doc_ids=[doc_id],
        document_families=[
            get_inventory_entry(doc_id).document_family
            if get_inventory_entry(doc_id)
            else "general_reference"
        ],
        summary_type="section_aware_overview",
        coverage_terms=coverage_terms,
    )
    return answer, cues, contract


def _render_document_type(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str, list[str], dict[str, object]]:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    language = _confidence_aware_language(assessment)
    type_label = _semantic_phrase(semantics.document_type)
    answer = f"{label} {language['type_verb']} {_indefinite_phrase(semantics.document_type)}."
    if semantics.document_purpose:
        answer += (
            f" Its main purpose is {_semantic_phrase(semantics.document_purpose)}."
        )
    contract = build_answer_contract(
        mode="document_type",
        primary_doc_ids=[doc_id],
        document_families=[
            get_inventory_entry(doc_id).document_family
            if get_inventory_entry(doc_id)
            else "general_reference"
        ],
        summary_type="type_focused",
    )
    return (
        answer,
        [type_label, _semantic_phrase(semantics.document_purpose, "")],
        contract,
    )


def _render_document_purpose(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str, list[str], dict[str, object]]:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    purpose = _semantic_phrase(semantics.document_purpose, "reference lookup")
    type_label = _semantic_phrase(semantics.document_type)
    confidence_label = assessment["classification_confidence_label"]
    purpose_verb = (
        "is"
        if confidence_label == "high"
        else "appears to be"
        if confidence_label == "moderate"
        else "is provisionally"
    )
    answer = f"The main purpose of {label} {purpose_verb} {purpose}."
    answer += f" It is structured as {_indefinite_phrase(semantics.document_type)}."
    if semantics.audience and semantics.audience != "general_professional":
        answer += f" It is aimed at {_semantic_phrase(semantics.audience)}."
    contract = build_answer_contract(
        mode="document_purpose",
        primary_doc_ids=[doc_id],
        document_families=[
            get_inventory_entry(doc_id).document_family
            if get_inventory_entry(doc_id)
            else "general_reference"
        ],
        summary_type="purpose_focused",
    )
    return (
        answer,
        [purpose, type_label, _semantic_phrase(semantics.audience, "")],
        contract,
    )


def _render_document_audience(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str, list[str], dict[str, object]]:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    language = _confidence_aware_language(assessment)
    audience = _semantic_phrase(semantics.audience, "general professionals")
    purpose = _semantic_phrase(semantics.document_purpose, "reference lookup")
    answer = f"{label} {language['audience_verb']} {audience}."
    answer += f" Its main purpose is {purpose}."
    contract = build_answer_contract(
        mode="document_audience",
        primary_doc_ids=[doc_id],
        document_families=[
            get_inventory_entry(doc_id).document_family
            if get_inventory_entry(doc_id)
            else "general_reference"
        ],
        summary_type="audience_focused",
    )
    return answer, [audience, purpose], contract


def _render_document_confidence(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str, list[str], dict[str, object]]:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    status = assessment["classification_status"]
    confidence_label = str(assessment["classification_confidence_label"])
    type_label = _semantic_phrase(semantics.document_type)
    purpose = _semantic_phrase(semantics.document_purpose, "reference lookup")
    warnings = [
        item.replace("_", " ") for item in assessment.get("semantic_warnings", [])
    ]
    rationale = [
        item.replace("_", " ") for item in assessment.get("semantic_rationale", [])
    ]
    if status == "well_supported":
        answer = (
            f"The current classification of {label} is well-supported. "
            f"It is being treated as {_indefinite_phrase(semantics.document_type)} with {confidence_label} confidence, and its main purpose is {purpose}."
        )
    elif status == "provisional":
        answer = (
            f"The current classification of {label} is provisional. "
            f"It is being treated as {_indefinite_phrase(semantics.document_type)} with {confidence_label} confidence, and its main purpose appears to be {purpose}."
        )
    else:
        answer = (
            f"The current classification of {label} is uncertain. "
            f"It is tentatively being treated as {_indefinite_phrase(semantics.document_type)} with {confidence_label} confidence, and this should be treated as a heuristic guess."
        )
    if rationale:
        answer += " Supporting cues include " + ", ".join(rationale[:2]) + "."
    if warnings:
        answer += " Current limits include " + ", ".join(warnings[:2]) + "."
    contract = build_answer_contract(
        mode="document_confidence",
        primary_doc_ids=[doc_id],
        document_families=[
            get_inventory_entry(doc_id).document_family
            if get_inventory_entry(doc_id)
            else "general_reference"
        ],
        summary_type="confidence_focused",
    )
    cues = [confidence_label, type_label, purpose, *rationale[:2]]
    return answer, cues, contract


def _render_document_classification_rationale(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str, list[str], dict[str, object]]:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    type_label = _semantic_phrase(semantics.document_type)
    purpose = _semantic_phrase(semantics.document_purpose, "reference lookup")
    rationale = [
        item.replace("_", " ") for item in assessment.get("semantic_rationale", [])
    ]
    if rationale:
        cue_text = ", ".join(rationale[:3])
    else:
        cue_text = "recovered title, section structure, and document metadata"
    answer = (
        f"{label} is currently classified as {_indefinite_phrase(semantics.document_type)} because the strongest supporting cues point in that direction. "
        f"Its main purpose is being interpreted as {purpose}. "
        f"The main supporting cues are {cue_text}."
    )
    contract = build_answer_contract(
        mode="document_classification_rationale",
        primary_doc_ids=[doc_id],
        document_families=[
            get_inventory_entry(doc_id).document_family
            if get_inventory_entry(doc_id)
            else "general_reference"
        ],
        summary_type="classification_rationale",
    )
    cues = [type_label, purpose, *rationale[:3]]
    return answer, cues, contract


def _render_document_classification_limits(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str, list[str], dict[str, object]]:
    label = _document_label(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    type_label = _semantic_phrase(semantics.document_type)
    warnings = [
        item.replace("_", " ") for item in assessment.get("semantic_warnings", [])
    ]
    answer = (
        f"The current classification of {label} still has limits. "
        f"It is being treated as {_indefinite_phrase(semantics.document_type)} with {assessment['classification_confidence_label']} confidence, "
        f"and that judgment still depends on recovered structure, layout, and semantic cues."
    )
    if warnings:
        answer += " The main current limits are " + ", ".join(warnings[:3]) + "."
    else:
        answer += " No major semantic warnings are present, but the result is still heuristic rather than authoritative."
    contract = build_answer_contract(
        mode="document_classification_limits",
        primary_doc_ids=[doc_id],
        document_families=[
            get_inventory_entry(doc_id).document_family
            if get_inventory_entry(doc_id)
            else "general_reference"
        ],
        summary_type="classification_limits",
    )
    cues = [
        type_label,
        str(assessment["classification_confidence_label"]),
        *warnings[:3],
    ]
    return answer, cues, contract


def _document_support_trace_item(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
    *,
    matched_terms: list[str] | None = None,
    support_sentences: list[str] | None = None,
) -> dict[str, object]:
    record = _load_document_record(doc_id, chunk_root)
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    section_titles = _document_summary_cues(doc_id, top_k_hits, chunk_root)[:3]
    section_paths = [
        " > ".join(section.section_path)
        for section in (record.sections if record else [])
        if section.section_path
    ][:3]
    section_summaries = [
        section.summary.strip()
        for section in (record.sections if record else [])
        if section.summary and section.summary.strip()
    ][:3]
    section_kinds = list(
        dict.fromkeys(
            section.section_kind
            for section in (record.sections if record else [])
            if section.section_kind
        )
    )[:4]
    section_content_hints = list(
        dict.fromkeys(
            hint
            for section in (record.sections if record else [])
            for hint in section.content_hints
            if hint
        )
    )[:4]
    coverage_terms = list(semantics.coverage_terms[:4])
    summary_cues = list(semantics.summary_cues[:4])
    inventory_summary = _document_inventory_summary(
        doc_id, top_k_hits, chunk_root, include_label=True
    )
    assessment = _classification_assessment(doc_id, top_k_hits, chunk_root)
    structure_confidence = assessment["structure_confidence"]
    layout_confidence = assessment["layout_confidence"]

    support_fragments: list[str] = []
    if semantics.document_type:
        support_fragments.append(
            f"document type: {semantics.document_type.replace('_', ' ')}"
        )
        support_fragments.append(
            f"{_document_label(doc_id, chunk_root)} is {_indefinite_phrase(semantics.document_type)}."
        )
    if semantics.document_purpose:
        support_fragments.append(
            f"purpose: {semantics.document_purpose.replace('_', ' ')}"
        )
        support_fragments.append(
            f"Its main purpose is {semantics.document_purpose.replace('_', ' ')}."
        )
    if semantics.audience and semantics.audience != "general_professional":
        support_fragments.append(f"audience: {semantics.audience.replace('_', ' ')}")
        support_fragments.append(
            f"It is aimed at {semantics.audience.replace('_', ' ')}."
        )
    if coverage_terms:
        support_fragments.append("coverage: " + ", ".join(coverage_terms))
    if section_titles:
        support_fragments.append("sections: " + ", ".join(section_titles))
    if section_paths:
        support_fragments.append("section paths: " + " | ".join(section_paths))
    if section_summaries:
        support_fragments.append("section summaries: " + " | ".join(section_summaries))
    if section_kinds:
        support_fragments.append("section kinds: " + ", ".join(section_kinds))
    if section_content_hints:
        support_fragments.append("section hints: " + ", ".join(section_content_hints))
    if structure_confidence is not None:
        support_fragments.append(f"structure confidence: {structure_confidence:.3f}")
    if layout_confidence is not None:
        support_fragments.append(f"layout confidence: {layout_confidence:.3f}")
    support_fragments.append(
        "classification confidence: "
        f"{assessment['classification_confidence']:.3f} ({assessment['classification_confidence_label']})"
    )
    support_fragments.append(
        "trust policy: " + str(assessment["trust_policy"]).replace("_", " ")
    )
    if assessment["semantic_rationale"]:
        support_fragments.append(
            "semantic rationale: "
            + ", ".join(
                item.replace("_", " ") for item in assessment["semantic_rationale"][:3]
            )
        )
    if assessment["semantic_warnings"]:
        support_fragments.append(
            "semantic warnings: "
            + ", ".join(
                item.replace("_", " ") for item in assessment["semantic_warnings"][:3]
            )
        )
    else:
        support_fragments.append(
            "No major semantic warnings are present, but the result is still heuristic rather than authoritative."
        )
    if matched_terms:
        support_fragments.append("matched terms: " + ", ".join(matched_terms[:4]))
    if support_sentences:
        support_fragments.extend(support_sentences[:3])

    return {
        "doc_id": doc_id,
        "label": _document_label(doc_id, chunk_root),
        "inventory_summary": inventory_summary,
        "document_family": semantics.document_family,
        "document_type": semantics.document_type,
        "document_purpose": semantics.document_purpose,
        "audience": semantics.audience,
        "structure_confidence": structure_confidence,
        "layout_confidence": layout_confidence,
        "semantic_confidence": assessment["semantic_confidence"],
        "semantic_confidence_label": assessment["semantic_confidence_label"],
        "classification_confidence": assessment["classification_confidence"],
        "classification_confidence_label": assessment[
            "classification_confidence_label"
        ],
        "classification_status": assessment["classification_status"],
        "trust_policy": assessment["trust_policy"],
        "semantic_rationale": list(assessment["semantic_rationale"]),
        "semantic_warnings": list(assessment["semantic_warnings"]),
        "coverage_terms": coverage_terms,
        "summary_cues": summary_cues,
        "section_titles": section_titles,
        "section_paths": section_paths,
        "section_summaries": section_summaries,
        "section_kinds": section_kinds,
        "section_content_hints": section_content_hints,
        "matched_terms": list(matched_terms[:4]) if matched_terms else [],
        "support_sentences": list(support_sentences[:3]) if support_sentences else [],
        "support_fragments": support_fragments,
    }


def _rank_document_candidates(
    doc_ids: list[str],
    query: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> list[str]:
    query_terms = _specific_query_terms(_query_terms(query))
    semantic_preferences = query_semantic_preferences(query)
    hit_rank: dict[str, int] = {}
    for idx, chunk in enumerate(top_k_hits):
        hit_rank.setdefault(chunk.doc_id, idx)
    ranked: list[tuple[float, str]] = []
    for doc_id in doc_ids:
        profile = get_document_profile(doc_id)
        label_terms = set(
            re.findall(r"[a-zA-Z]{3,}", (profile.label if profile else doc_id).lower())
        )
        profile_terms = set(profile.topical_terms) if profile else set()
        discovery_terms = set(_document_discovery_terms(doc_id, chunk_root))
        semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
        facet_terms = _document_facet_tokens(doc_id, chunk_root)
        summary_terms = {
            token
            for cue in semantics.summary_cues
            for token in re.findall(r"[a-zA-Z]{3,}", cue.lower())
        }
        coverage_terms = {
            token
            for cue in semantics.coverage_terms
            for token in re.findall(r"[a-zA-Z]{3,}", cue.lower())
        }
        overlap = len(
            query_terms
            & (
                label_terms
                | profile_terms
                | discovery_terms
                | summary_terms
                | coverage_terms
                | facet_terms
            )
        )
        score = overlap * 3.0
        score += len(query_terms & facet_terms) * 2.0
        score += len(query_terms & coverage_terms) * 2.0
        if profile:
            score += len(query_terms & profile_terms) * 0.75
        if semantic_preferences["families"]:
            if semantics.document_family in semantic_preferences["families"]:
                score += 4.0
            else:
                score -= 2.5
        if semantic_preferences["purposes"]:
            if semantics.document_purpose in semantic_preferences["purposes"]:
                score += 3.0
            else:
                score -= 1.5
        if doc_id in hit_rank:
            score += max(0.0, 3.0 - min(hit_rank[doc_id], 6) * 0.4)
        ranked.append((score, doc_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in ranked]


def _document_semantic_match_terms(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> set[str]:
    semantics = _document_semantics(doc_id, top_k_hits, chunk_root)
    profile = get_document_profile(doc_id)
    terms: set[str] = set()
    for value in (
        semantics.document_type,
        semantics.document_purpose,
        semantics.audience,
        semantics.evidence_style,
        semantics.structure_style,
        semantics.document_family,
        semantics.inventory_summary,
        semantics.coverage_summary,
        _document_label(doc_id, chunk_root),
    ):
        if value:
            terms.update(re.findall(r"[a-zA-Z]{3,}", str(value).lower()))
    for collection in (
        semantics.summary_cues,
        semantics.discovery_terms,
        semantics.coverage_terms,
        semantics.facet_terms,
    ):
        for item in collection:
            terms.update(re.findall(r"[a-zA-Z]{3,}", str(item).lower()))
    if profile:
        terms.update(profile.topical_terms)
    return terms


def _sentence_overlap_score(
    sentence: str, query_terms: set[str]
) -> tuple[float, list[str]]:
    sentence_terms = set(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))
    matched_terms = sorted(sentence_terms & query_terms)
    return float(len(matched_terms)), matched_terms[:4]


def _best_support_sentences_for_doc(
    query: str,
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    context_chunks: list[ChunkRecord],
    evidence: list[EvidenceSentence],
    chunk_root: Path,
) -> list[str]:
    query_terms = _specific_query_terms(_query_terms(query))
    chunk_lookup = {chunk.chunk_id: chunk for chunk in context_chunks}
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()

    for item in evidence:
        chunk = chunk_lookup.get(item.chunk_id)
        if not chunk or chunk.doc_id != doc_id:
            continue
        normalized = _normalize_text(item.sentence)
        if not normalized or normalized in seen:
            continue
        overlap_score, _ = _sentence_overlap_score(normalized, query_terms)
        score = overlap_score + float(item.score)
        scored.append((score, normalized))
        seen.add(normalized)

    if not scored:
        record = _load_document_record(doc_id, chunk_root)
        if record:
            for section in record.sections:
                if not section.summary:
                    continue
                normalized = _normalize_text(section.summary)
                if not normalized or normalized in seen:
                    continue
                overlap_score, _ = _sentence_overlap_score(normalized, query_terms)
                section_bonus = (
                    0.75
                    if section.title
                    and any(term in section.title.lower() for term in query_terms)
                    else 0.25
                )
                scored.append((overlap_score + section_bonus, normalized))
                seen.add(normalized)
                if len(scored) >= 4:
                    break

    if not scored:
        for chunk in context_chunks:
            if chunk.doc_id != doc_id:
                continue
            for sentence in _split_sentences(chunk.text):
                normalized = _normalize_text(sentence)
                if not normalized or normalized in seen:
                    continue
                overlap_score, _ = _sentence_overlap_score(normalized, query_terms)
                section_bonus = (
                    0.5
                    if chunk.section_title
                    and any(term in chunk.section_title.lower() for term in query_terms)
                    else 0.0
                )
                scored.append((overlap_score + section_bonus, normalized))
                seen.add(normalized)
                if len(scored) >= 8:
                    break
            if len(scored) >= 8:
                break

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sentence.rstrip(".") for _, sentence in scored[:2]]


def _matched_terms_for_doc(
    query: str,
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
    support_sentences: list[str],
) -> list[str]:
    def _normalize_query_phrases(matched_terms: list[str]) -> list[str]:
        if len(matched_terms) < 2:
            return matched_terms[:4]

        normalized_terms = list(matched_terms)
        lowered = {term.lower() for term in normalized_terms}
        query_tokens = re.findall(r"[a-zA-Z]{3,}", query.lower())
        query_bigrams = list(zip(query_tokens, query_tokens[1:]))
        combined: list[str] = []
        consumed: set[str] = set()

        for left, right in query_bigrams:
            if left in lowered and right in lowered:
                phrase = f"{left} {right}"
                combined.append(phrase)
                consumed.update({left, right})

        if not combined:
            return normalized_terms[:4]

        collapsed = [term for term in normalized_terms if term.lower() not in consumed]
        for phrase in reversed(combined):
            collapsed.insert(0, phrase)
        return collapsed[:4]

    query_terms = _specific_query_terms(_query_terms(query))
    semantic_terms = _document_semantic_match_terms(doc_id, top_k_hits, chunk_root)
    matched_terms = sorted(semantic_terms & query_terms)
    if matched_terms:
        return _normalize_query_phrases(matched_terms)
    sentence_terms: set[str] = set()
    for sentence in support_sentences:
        sentence_terms.update(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))
    return _normalize_query_phrases(sorted(sentence_terms & query_terms))


def _build_document_support_entry(
    query: str,
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    context_chunks: list[ChunkRecord],
    evidence: list[EvidenceSentence],
    chunk_root: Path,
) -> dict[str, object]:
    support_sentences = _best_support_sentences_for_doc(
        query=query,
        doc_id=doc_id,
        top_k_hits=top_k_hits,
        context_chunks=context_chunks,
        evidence=evidence,
        chunk_root=chunk_root,
    )
    matched_terms = _matched_terms_for_doc(
        query=query,
        doc_id=doc_id,
        top_k_hits=top_k_hits,
        chunk_root=chunk_root,
        support_sentences=support_sentences,
    )
    return _document_support_trace_item(
        doc_id,
        top_k_hits,
        chunk_root,
        matched_terms=matched_terms,
        support_sentences=support_sentences,
    )


def _build_document_support_entries(
    query: str,
    doc_ids: list[str],
    top_k_hits: list[ChunkRecord],
    context_chunks: list[ChunkRecord],
    evidence: list[EvidenceSentence],
    chunk_root: Path,
) -> list[dict[str, object]]:
    return [
        _build_document_support_entry(
            query=query,
            doc_id=doc_id,
            top_k_hits=top_k_hits,
            context_chunks=context_chunks,
            evidence=evidence,
            chunk_root=chunk_root,
        )
        for doc_id in doc_ids
    ]


def _document_families(doc_ids: list[str]) -> list[str]:
    return [
        get_inventory_entry(doc_id).document_family
        if get_inventory_entry(doc_id)
        else "general_reference"
        for doc_id in doc_ids
    ]


def _trace_payload(
    *,
    template_id: str,
    matched_pattern: str,
    matched_cues: list[str],
    mode: str,
    primary_doc_ids: list[str],
    summary_type: str | None = None,
    coverage_terms: list[str] | None = None,
    matched_terms: list[str] | None = None,
    relationship: str | None = None,
    support_trace: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    contract_kwargs: dict[str, object] = {
        "mode": mode,
        "primary_doc_ids": primary_doc_ids,
        "document_families": _document_families(primary_doc_ids),
    }
    if summary_type:
        contract_kwargs["summary_type"] = summary_type
    if coverage_terms:
        contract_kwargs["coverage_terms"] = coverage_terms
    if matched_terms:
        contract_kwargs["matched_terms"] = matched_terms
    if relationship:
        contract_kwargs["relationship"] = relationship
    return {
        "template_id": template_id,
        "matched_pattern": matched_pattern,
        "matched_cues": matched_cues,
        "answer_contract": build_answer_contract(**contract_kwargs),
        "support_trace": support_trace or [],
    }


def _comparison_support_trace_item(
    difference_summary: str | None,
    relation_summary: str | None,
) -> dict[str, object] | None:
    if not difference_summary and not relation_summary:
        return None
    return {
        "doc_id": "__comparison__",
        "label": "Cross-document relationship",
        "inventory_summary": "",
        "document_family": "comparison",
        "document_type": "relationship",
        "document_purpose": "comparison",
        "audience": "",
        "coverage_terms": [],
        "summary_cues": [],
        "section_titles": [],
        "matched_terms": [],
        "support_sentences": [
            item for item in (difference_summary, relation_summary) if item
        ],
        "support_fragments": [
            item for item in (difference_summary, relation_summary) if item
        ],
    }


def _base_answer_trace(
    query: str,
    query_intent: str,
    evidence: list[EvidenceSentence],
    template_id: str | None = None,
    matched_pattern: str | None = None,
    matched_cues: list[str] | None = None,
    answer_contract: dict[str, object] | None = None,
    support_trace: list[dict[str, object]] | None = None,
    document_selection: dict[str, object] | None = None,
    document_synthesis: dict[str, object] | None = None,
    retrieval_contract: dict[str, object] | None = None,
    synthesis_prompt_contract: dict[str, object] | None = None,
    synthesis_runtime: dict[str, object] | None = None,
    claim_alignment: dict[str, object] | None = None,
) -> dict[str, object]:
    plan = plan_query(query)
    return {
        "query_class": plan.query_class,
        "answer_mode": plan.answer_mode,
        "query_intent": query_intent,
        "source_doc_id": _preferred_source_doc_id(query),
        "inventory_doc_ids": list(plan.inventory_doc_ids),
        "matched_doc_ids": list(plan.matched_doc_ids),
        "candidate_doc_ids": list(plan.candidate_doc_ids),
        "query_features": dict(plan.query_features),
        "mode_scores": dict(plan.mode_scores),
        "chosen_rationale": list(plan.chosen_rationale),
        "template_id": template_id,
        "matched_pattern": matched_pattern,
        "matched_cues": matched_cues or [],
        "answer_contract": answer_contract or {},
        "retrieval_contract": retrieval_contract or {},
        "document_selection": document_selection or {},
        "document_synthesis": document_synthesis or {},
        "synthesis_prompt_contract": synthesis_prompt_contract or {},
        "synthesis_runtime": synthesis_runtime or {},
        "claim_alignment": claim_alignment or {},
        "support_trace": support_trace or [],
        "evidence_chunk_ids": [item.chunk_id for item in evidence],
    }


def _document_selection_payload(selection: DocumentSelection) -> dict[str, object]:
    return {
        "strategy": selection.strategy,
        "candidate_doc_ids": list(selection.candidate_doc_ids),
        "ranked_doc_ids": list(selection.ranked_doc_ids),
        "selected_doc_ids": list(selection.selected_doc_ids),
        "primary_doc_id": selection.primary_doc_id,
        "shortlist_breakdown": list(selection.shortlist_breakdown),
    }


def _document_synthesis_payload(synthesis: DocumentSynthesis) -> dict[str, object]:
    return {
        "support_scope": synthesis.support_scope,
        "support_doc_ids": list(synthesis.support_doc_ids),
        "answer_chunk_doc_ids": list(synthesis.answer_chunk_doc_ids),
        "selected_chunk_count": synthesis.selected_chunk_count,
    }


def _resolve_candidate_doc_ids(
    plan,
    shortlist_breakdown: list[dict[str, object]],
    top_k_hits: list[ChunkRecord],
) -> list[str]:
    candidate_doc_ids = list(plan.candidate_doc_ids) or [
        candidate["doc_id"] for candidate in shortlist_breakdown
    ]
    if candidate_doc_ids:
        return candidate_doc_ids

    seen: set[str] = set()
    for chunk in top_k_hits:
        if chunk.doc_id not in seen:
            candidate_doc_ids.append(chunk.doc_id)
            seen.add(chunk.doc_id)
    return candidate_doc_ids


def _ranked_candidate_doc_ids(
    answer_mode: str,
    candidate_doc_ids: list[str],
    query: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> list[str]:
    if answer_mode in {"document_routing", "source_listing", "cross_document_compare"}:
        return _rank_document_candidates(
            candidate_doc_ids, query, top_k_hits, chunk_root
        )
    return list(candidate_doc_ids)


def _selected_doc_ids_for_mode(
    plan,
    ranked_doc_ids: list[str],
    primary_doc_id: str | None,
) -> tuple[list[str], str | None, str]:
    answer_mode = plan.answer_mode
    strategy = DOCUMENT_SELECTION_STRATEGIES.get(answer_mode, "preferred_doc")
    limit = _selection_limit(
        answer_mode,
        plural_routing=bool(plan.query_features.get("plural_routing")),
    )
    if answer_mode in {
        "document_overview",
        "source_justification",
        "grounded_evidence",
    }:
        selected_doc_ids = [primary_doc_id] if primary_doc_id else []
        return selected_doc_ids, primary_doc_id, strategy

    selected_doc_ids = ranked_doc_ids[:limit]
    if selected_doc_ids:
        primary_doc_id = selected_doc_ids[0]
    return selected_doc_ids, primary_doc_id, strategy


def _support_doc_ids_for_mode(
    plan,
    selection: DocumentSelection,
) -> list[str]:
    if plan.answer_mode in {"source_listing", "cross_document_compare"}:
        return list(selection.selected_doc_ids)
    if plan.answer_mode == "document_routing" and plan.query_features.get(
        "plural_routing"
    ):
        return list(selection.selected_doc_ids)
    if selection.primary_doc_id:
        return [selection.primary_doc_id]
    return list(selection.selected_doc_ids[:1])


def _build_document_selection(
    plan,
    query: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> DocumentSelection:
    shortlist_breakdown = [
        {
            "doc_id": candidate.entry.doc_id,
            "label": candidate.entry.label,
            "score": round(candidate.breakdown.total, 3),
            "matched_terms": list(candidate.matched_terms),
            "rationale": list(candidate.rationale),
        }
        for candidate in plan.shortlist[:6]
    ]
    answer_mode = plan.answer_mode
    candidate_doc_ids = _resolve_candidate_doc_ids(
        plan, shortlist_breakdown, top_k_hits
    )
    ranked_doc_ids = _ranked_candidate_doc_ids(
        answer_mode,
        candidate_doc_ids,
        query,
        top_k_hits,
        chunk_root,
    )
    primary_doc_id = plan.preferred_doc_id or (
        ranked_doc_ids[0] if ranked_doc_ids else None
    )
    selected_doc_ids, primary_doc_id, strategy = _selected_doc_ids_for_mode(
        plan,
        ranked_doc_ids,
        primary_doc_id,
    )

    return DocumentSelection(
        answer_mode=answer_mode,
        candidate_doc_ids=candidate_doc_ids,
        ranked_doc_ids=ranked_doc_ids,
        selected_doc_ids=selected_doc_ids,
        primary_doc_id=primary_doc_id,
        strategy=strategy,
        shortlist_breakdown=shortlist_breakdown,
    )


def _build_document_synthesis(
    *,
    plan,
    query: str,
    query_intent: str,
    top_k_hits: list[ChunkRecord],
    expanded_hits: list[ChunkRecord],
    evidence: list[EvidenceSentence],
    chunk_root: Path,
    retrieval_contract: dict[str, object],
) -> DocumentSynthesis:
    selection = _build_document_selection(
        plan=plan,
        query=query,
        top_k_hits=top_k_hits,
        chunk_root=chunk_root,
    )
    answer_chunks = expanded_hits
    support_scope = "expanded_hits"
    selected_doc_ids = set(selection.selected_doc_ids)
    if selected_doc_ids and plan.answer_mode != "grounded_evidence":
        filtered = [
            chunk for chunk in expanded_hits if chunk.doc_id in selected_doc_ids
        ]
        if filtered:
            answer_chunks = filtered
            support_scope = "selected_docs"

    retrieval_path = str(retrieval_contract.get("retrieval_path") or "")
    contract_preferred_doc_id = retrieval_contract.get("preferred_doc_id")
    if isinstance(contract_preferred_doc_id, str) and contract_preferred_doc_id:
        preferred_hits = [
            chunk
            for chunk in answer_chunks
            if chunk.doc_id == contract_preferred_doc_id
        ]
        if preferred_hits and retrieval_path == "single_document_qa":
            answer_chunks = preferred_hits
            support_scope = "preferred_doc"

    query_terms = _query_terms(query)
    source_anchor_preferred_doc_id = _preferred_source_doc_id(query)
    if (
        source_anchor_preferred_doc_id
        and plan.answer_mode == "grounded_evidence"
        and query_terms.intersection(SOURCE_ANCHORED_HINTS)
    ):
        preferred_hits = []
        seen_preferred_chunk_ids: set[str] = set()
        for chunk in [*top_k_hits, *expanded_hits]:
            if chunk.doc_id != source_anchor_preferred_doc_id:
                continue
            if chunk.chunk_id in seen_preferred_chunk_ids:
                continue
            seen_preferred_chunk_ids.add(chunk.chunk_id)
            preferred_hits.append(chunk)
        if preferred_hits:
            answer_chunks = preferred_hits
            support_scope = "source_anchor_preferred_doc"

    if (
        top_k_hits
        and query_terms.intersection(SOURCE_ANCHORED_HINTS)
        and support_scope != "source_anchor_preferred_doc"
        and plan.answer_mode
        not in {"document_routing", "source_listing", "cross_document_compare"}
    ):
        doc_counts: dict[str, int] = {}
        for chunk in top_k_hits:
            doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
        dominant_doc_id, dominant_count = max(
            doc_counts.items(), key=lambda item: item[1]
        )
        if dominant_count >= max(2, (len(top_k_hits) // 2) + 1):
            filtered = [
                chunk for chunk in expanded_hits if chunk.doc_id == dominant_doc_id
            ]
            if filtered:
                answer_chunks = filtered
                support_scope = "dominant_hit_doc"

    answer_chunk_doc_ids = list(dict.fromkeys(chunk.doc_id for chunk in answer_chunks))
    support_doc_ids = _support_doc_ids_for_mode(plan, selection)
    if (
        support_scope == "source_anchor_preferred_doc"
        and source_anchor_preferred_doc_id
    ):
        support_doc_ids = [source_anchor_preferred_doc_id]
    support_entries = _build_document_support_entries(
        query=query,
        doc_ids=support_doc_ids,
        top_k_hits=top_k_hits,
        context_chunks=answer_chunks,
        evidence=evidence,
        chunk_root=chunk_root,
    )
    return DocumentSynthesis(
        selection=selection,
        support_scope=support_scope,
        support_doc_ids=support_doc_ids,
        answer_chunk_doc_ids=answer_chunk_doc_ids,
        selected_chunk_count=len(answer_chunks),
        answer_chunks=answer_chunks,
        support_entries=support_entries,
    )


def _document_overview_mode_answer(
    synthesis: DocumentSynthesis,
    query: str,
    query_intent: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str | None, dict[str, object] | None]:
    selection = synthesis.selection
    primary_doc_id = selection.primary_doc_id or (
        selection.selected_doc_ids[0] if selection.selected_doc_ids else None
    )
    if primary_doc_id is None and top_k_hits:
        primary_doc_id = top_k_hits[0].doc_id
    if primary_doc_id is None:
        return None, None
    support_entry = (
        synthesis.support_entries[0]
        if synthesis.support_entries
        else _document_support_trace_item(
            primary_doc_id,
            top_k_hits,
            chunk_root,
        )
    )
    if query_intent == "document_type":
        answer, cues, answer_contract = _render_document_type(
            primary_doc_id, top_k_hits, chunk_root
        )
        template_id = "document_level.type"
        matched_pattern = "document-type-facet-summary"
    elif query_intent == "document_purpose":
        answer, cues, answer_contract = _render_document_purpose(
            primary_doc_id, top_k_hits, chunk_root
        )
        template_id = "document_level.purpose"
        matched_pattern = "document-purpose-facet-summary"
    elif query_intent == "document_audience":
        answer, cues, answer_contract = _render_document_audience(
            primary_doc_id, top_k_hits, chunk_root
        )
        template_id = "document_level.audience"
        matched_pattern = "document-audience-facet-summary"
    elif query_intent == "document_confidence":
        answer, cues, answer_contract = _render_document_confidence(
            primary_doc_id, top_k_hits, chunk_root
        )
        template_id = "document_level.confidence"
        matched_pattern = "document-confidence-facet-summary"
    elif query_intent == "document_classification_rationale":
        answer, cues, answer_contract = _render_document_classification_rationale(
            primary_doc_id, top_k_hits, chunk_root
        )
        template_id = "document_level.classification_rationale"
        matched_pattern = "document-classification-rationale-summary"
    elif query_intent == "document_classification_limits":
        answer, cues, answer_contract = _render_document_classification_limits(
            primary_doc_id, top_k_hits, chunk_root
        )
        template_id = "document_level.classification_limits"
        matched_pattern = "document-classification-limits-summary"
    else:
        answer, cues, answer_contract = _render_document_overview(
            primary_doc_id, top_k_hits, chunk_root
        )
        template_id = "document_level.overview"
        matched_pattern = "section-aware-document-summary"
    if answer:
        return (
            answer,
            {
                "template_id": template_id,
                "matched_pattern": matched_pattern,
                "matched_cues": cues,
                "answer_contract": answer_contract,
                "support_trace": [support_entry],
            },
        )
    primary_label = _document_label(primary_doc_id, chunk_root)
    overview_fragments = _document_overview_fragments(
        primary_doc_id, top_k_hits, chunk_root
    )
    if overview_fragments:
        return (
            f"{primary_label}: " + "; ".join(overview_fragments) + ".",
            {
                "template_id": "document_level.overview",
                "matched_pattern": "document-facets-and-summary-cues",
                "matched_cues": overview_fragments,
                "support_trace": [support_entry],
            },
        )
    return None, None


def _document_routing_mode_answer(
    plan,
    synthesis: DocumentSynthesis,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str | None, dict[str, object] | None]:
    selection = synthesis.selection
    if not top_k_hits or not selection.selected_doc_ids:
        return None, None
    if plan.query_features.get("plural_routing"):
        chosen_doc_ids = list(selection.selected_doc_ids)
        if len(chosen_doc_ids) >= 2:
            support_trace = synthesis.support_entries
            fragments: list[str] = []
            cues: list[str] = []
            for support_entry in support_trace:
                label = str(support_entry["label"])
                inventory_summary = str(support_entry["inventory_summary"])
                fragments.append(f"{label} ({inventory_summary})")
                cues.extend([label, inventory_summary])
            return (
                "The most relevant files are " + "; ".join(fragments) + ".",
                _trace_payload(
                    template_id="document_level.routing_multi",
                    matched_pattern="inventory-ranked-multi-doc-summary",
                    matched_cues=cues,
                    mode="document_routing",
                    primary_doc_ids=chosen_doc_ids,
                    summary_type="inventory_summary",
                    support_trace=support_trace,
                ),
            )

    primary_doc_id = selection.primary_doc_id
    if not primary_doc_id:
        return None, None
    support_entry = (
        synthesis.support_entries[0]
        if synthesis.support_entries
        else _document_support_trace_item(
            primary_doc_id,
            top_k_hits,
            chunk_root,
        )
    )
    primary_label = str(support_entry["label"])
    inventory_summary = str(support_entry["inventory_summary"])
    topical_terms = list(support_entry.get("matched_terms", []))[:4]
    topics = list(support_entry.get("section_titles", []))[:3]
    detail_parts: list[str] = []
    if topical_terms:
        detail_parts.append("grounded material on " + ", ".join(topical_terms))
    if topics:
        detail_parts.append("sections such as " + ", ".join(topics))
    if not inventory_summary and not detail_parts:
        return None, None
    answer_sentences = [f"The most relevant file is {primary_label}."]
    if inventory_summary:
        answer_sentences.append(inventory_summary + ".")
    if detail_parts:
        answer_sentences.append("It includes " + " and ".join(detail_parts) + ".")
    rendered_answer = " ".join(answer_sentences)
    support_trace = [
        _document_support_trace_item(
            primary_doc_id,
            top_k_hits,
            chunk_root,
            matched_terms=topical_terms[:4],
            support_sentences=[rendered_answer],
        )
    ]
    return (
        rendered_answer,
        _trace_payload(
            template_id="document_level.routing",
            matched_pattern="top-doc-topical-summary",
            matched_cues=[
                primary_label,
                inventory_summary,
                *topical_terms[:4],
                *topics[:3],
            ],
            mode="document_routing",
            primary_doc_ids=[primary_doc_id],
            summary_type="inventory_summary_plus_topics",
            coverage_terms=list(
                _document_semantics(
                    primary_doc_id, top_k_hits, chunk_root
                ).coverage_terms[:4]
            ),
            matched_terms=topical_terms[:4],
            support_trace=support_trace,
        ),
    )


def _source_justification_mode_answer(
    synthesis: DocumentSynthesis,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str | None, dict[str, object] | None]:
    selection = synthesis.selection
    if not top_k_hits:
        return None, None
    primary_doc_id = selection.primary_doc_id or top_k_hits[0].doc_id
    support_entry = (
        synthesis.support_entries[0]
        if synthesis.support_entries
        else _document_support_trace_item(
            primary_doc_id,
            top_k_hits,
            chunk_root,
        )
    )
    primary_label = str(support_entry["label"])
    matched_terms = list(support_entry.get("matched_terms", []))[:4]
    summary_cues = list(support_entry.get("section_titles", []))[:3]
    facets = _document_facets(primary_doc_id, chunk_root)
    parts: list[str] = []
    inventory_summary = str(support_entry["inventory_summary"])
    if inventory_summary:
        parts.append(inventory_summary)
    if matched_terms:
        parts.append("it matches cues such as " + ", ".join(matched_terms))
    if facets.get("document_type") or facets.get("document_purpose"):
        facet_parts = [
            str(value).replace("_", " ")
            for value in (facets.get("document_type"), facets.get("document_purpose"))
            if value
        ]
        if facet_parts:
            parts.append("it is classified as " + " / ".join(facet_parts))
    if summary_cues:
        parts.append("it includes sections such as " + ", ".join(summary_cues[:3]))
    if not parts:
        parts.append(
            "it surfaced the strongest grounded support in the current benchmark"
        )
    rendered_answer = (
        f"{primary_label} is the best match because " + "; ".join(parts) + "."
    )
    support_trace = [
        _document_support_trace_item(
            primary_doc_id,
            top_k_hits,
            chunk_root,
            matched_terms=matched_terms[:4],
            support_sentences=[rendered_answer],
        )
    ]
    return (
        rendered_answer,
        _trace_payload(
            template_id="document_level.justification",
            matched_pattern="doc-metadata-justification",
            matched_cues=[
                primary_label,
                inventory_summary,
                *matched_terms,
                *summary_cues[:3],
            ],
            mode="source_justification",
            primary_doc_ids=[primary_doc_id],
            summary_type="inventory_summary_plus_cues",
            coverage_terms=list(
                _document_semantics(
                    primary_doc_id, top_k_hits, chunk_root
                ).coverage_terms[:4]
            ),
            matched_terms=matched_terms[:4],
            support_trace=support_trace,
        ),
    )


def _source_listing_mode_answer(
    synthesis: DocumentSynthesis,
) -> tuple[str | None, dict[str, object] | None]:
    selection = synthesis.selection
    if not selection.selected_doc_ids:
        return None, None
    support_trace = synthesis.support_entries
    labels = [
        f"{support_entry['label']} ({support_entry['inventory_summary']})"
        for support_entry in support_trace
    ]
    return (
        "Relevant sources include: " + "; ".join(labels) + ".",
        _trace_payload(
            template_id="cross_doc.source_listing",
            matched_pattern="inventory-ranked-doc-diverse-topk",
            matched_cues=labels,
            mode="source_listing",
            primary_doc_ids=list(selection.selected_doc_ids),
            summary_type="inventory_summary",
            support_trace=support_trace,
        ),
    )


def _cross_document_compare_mode_answer(
    synthesis: DocumentSynthesis,
    chunk_root: Path,
) -> tuple[str | None, dict[str, object] | None]:
    selection = synthesis.selection
    doc_ids = list(selection.selected_doc_ids[:3])
    if len(doc_ids) < 2:
        return None, None
    support_entries = synthesis.support_entries
    fragments = [
        f"{item['label']}: {item['inventory_summary']}. Evidence: {item['support_sentences'][0]}."
        for item in support_entries
        if item.get("support_sentences")
    ]
    support_trace = list(support_entries)
    difference_summary = _document_difference_summary(doc_ids, chunk_root)
    relation_label, relation_summary = _document_relationship_signal(
        doc_ids, chunk_root
    )
    if difference_summary:
        fragments.append(difference_summary)
    if relation_summary:
        fragments.append(relation_summary)
    comparison_item = _comparison_support_trace_item(
        difference_summary, relation_summary
    )
    if comparison_item:
        support_trace.append(comparison_item)
    return (
        " ".join(fragments),
        _trace_payload(
            template_id="cross_doc.compare",
            matched_pattern="doc-diverse-evidence-plus-facets",
            matched_cues=[str(item["label"]) for item in support_entries]
            + [_document_profile_summary(doc_id, chunk_root) for doc_id in doc_ids],
            mode="cross_document_compare",
            primary_doc_ids=doc_ids,
            relationship=relation_label,
            support_trace=support_trace,
        ),
    )


def _document_mode_answer(
    plan,
    synthesis: DocumentSynthesis,
    query: str,
    query_intent: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> tuple[str | None, dict[str, object] | None]:
    answer_mode = plan.answer_mode
    query_terms = _query_terms(query)
    unsupported_entities = query_terms.intersection(UNSUPPORTED_ENTITY_TERMS)
    explicit_source_matches = matching_source_doc_ids(query, allow_topical=False)
    if (
        answer_mode in {"document_routing", "source_listing"}
        and unsupported_entities
        and not explicit_source_matches
    ):
        return None, None
    if answer_mode == "document_overview":
        return _document_overview_mode_answer(
            synthesis,
            query,
            query_intent,
            top_k_hits,
            chunk_root,
        )
    if answer_mode == "document_routing":
        return _document_routing_mode_answer(
            plan,
            synthesis,
            top_k_hits,
            chunk_root,
        )
    if answer_mode == "source_justification":
        return _source_justification_mode_answer(
            synthesis,
            top_k_hits,
            chunk_root,
        )
    if answer_mode == "source_listing":
        return _source_listing_mode_answer(synthesis)
    if answer_mode == "cross_document_compare":
        return _cross_document_compare_mode_answer(synthesis, chunk_root)
    return None, None
