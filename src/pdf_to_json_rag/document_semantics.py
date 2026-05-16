"""Shared document-semantics interpretation for inventory, routing, and comparison."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .document_facets import derive_document_facets, facet_token_terms
from .document_family import classify_document_family


SEMANTIC_STOPWORDS = {
    "about",
    "also",
    "an",
    "and",
    "are",
    "book",
    "books",
    "chapter",
    "chapters",
    "covers",
    "data",
    "document",
    "documents",
    "file",
    "files",
    "for",
    "from",
    "guidance",
    "how",
    "in",
    "introduction",
    "is",
    "manual",
    "model",
    "note",
    "notes",
    "of",
    "on",
    "or",
    "overview",
    "report",
    "reports",
    "review",
    "section",
    "sections",
    "source",
    "sources",
    "summary",
    "technical",
    "that",
    "the",
    "this",
    "what",
    "which",
    "why",
    "with",
}

LOW_SIGNAL_PHRASES = {
    "guidance note series",
    "the centre for humanitarian data",
    "ocha centre for humanitarian data",
    "model report:",
}

QUERY_FAMILY_HINTS = {
    "structured_form": ("questionnaire", "checklist", "appendix", "form", "grid"),
    "technical_manual": ("manual", "technical", "field manual"),
    "humanitarian_guidance": (
        "guidance note",
        "guidance",
        "donor",
        "cyber threats",
        "data incident",
    ),
    "humanitarian_model_report": ("model report", "forecast", "trigger", "anticipatory action"),
    "educational_book": ("book", "chapter", "learning", "introduction"),
    "clinical_review": ("review", "meta-analysis", "literature"),
}

QUERY_PURPOSE_HINTS = {
    "teaching_reference": ("learn", "learning", "teach", "chapter"),
    "procedural_guidance": ("guidance", "procedure", "should", "policy"),
    "structured_data_capture": ("questionnaire", "survey", "capture", "form"),
    "operational_checklist": ("checklist", "screening", "before"),
    "risk_or_trigger_assessment": ("forecast", "trigger", "risk", "scenario"),
    "evidence_summary": ("review", "evidence", "meta-analysis", "compare"),
}


@dataclass(frozen=True)
class DocumentSemantics:
    document_type: str
    document_purpose: str
    audience: str
    evidence_style: str
    structure_style: str
    document_family: str
    facet_terms: tuple[str, ...]
    summary_cues: tuple[str, ...]
    discovery_terms: tuple[str, ...]
    inventory_summary: str
    coverage_terms: tuple[str, ...]
    coverage_summary: str


def _normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").strip())


def _tokenize(text: str, min_len: int = 3) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{%d,}" % min_len, text.lower())
        if token not in SEMANTIC_STOPWORDS
    }


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(value)
    return ordered


def _clean_signal_values(values: list[str] | tuple[str, ...], title: str) -> tuple[str, ...]:
    normalized_title = _normalize_phrase(title).lower()
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalize_phrase(raw)
        lowered = value.lower()
        if not value or lowered == normalized_title or lowered in LOW_SIGNAL_PHRASES:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(value)
    return tuple(cleaned)


def _candidate_coverage_phrases(
    *,
    discovery_terms: list[str] | tuple[str, ...],
    summary_cues: list[str] | tuple[str, ...],
    toc: list[str] | tuple[str, ...],
    title: str,
) -> list[str]:
    candidates: list[str] = []
    normalized_title = _normalize_phrase(title).lower()
    for item in discovery_terms:
        clean = _normalize_phrase(item)
        if clean and len(clean.split()) <= 5 and clean.lower() != normalized_title:
            candidates.append(clean)
    for item in summary_cues:
        clean = _normalize_phrase(item)
        if clean and len(clean.split()) <= 6 and clean.lower() != normalized_title:
            candidates.append(clean)
    for item in toc[:8]:
        clean = _normalize_phrase(item)
        if clean and len(clean.split()) <= 6 and clean.lower() != normalized_title:
            candidates.append(clean)
    if title:
        title_clean = _normalize_phrase(title)
        title_terms = _tokenize(title_clean)
        if 1 <= len(title_terms) <= 4:
            candidates.append(title_clean)
    return _ordered_unique(candidates)


def build_inventory_summary(
    *,
    title: str,
    document_type: str,
    document_purpose: str,
    audience: str,
    evidence_style: str,
    structure_style: str,
    summary_cues: list[str] | tuple[str, ...],
) -> str:
    label = title.strip() if isinstance(title, str) else ""
    parts: list[str] = []
    if document_type:
        parts.append(document_type.replace("_", " "))
    if document_purpose:
        parts.append(document_purpose.replace("_", " "))
    if audience and audience != "general_professional":
        parts.append(f"for {audience.replace('_', ' ')}")
    if evidence_style:
        parts.append(evidence_style.replace("_", " "))
    if structure_style:
        parts.append(structure_style.replace("_", " "))
    if summary_cues:
        parts.append("topics: " + ", ".join(summary_cues[:3]))
    if not parts:
        return label
    if label:
        return f"{label} | " + "; ".join(parts)
    return "; ".join(parts)


def derive_coverage_terms(
    *,
    title: str,
    discovery_terms: list[str] | tuple[str, ...],
    summary_cues: list[str] | tuple[str, ...],
    toc: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    phrases = _candidate_coverage_phrases(
        discovery_terms=discovery_terms,
        summary_cues=summary_cues,
        toc=toc,
        title=title,
    )
    filtered: list[str] = []
    for phrase in phrases:
        terms = _tokenize(phrase, min_len=2)
        if not terms:
            continue
        if phrase.lower() in LOW_SIGNAL_PHRASES:
            continue
        if phrase.lower() in SEMANTIC_STOPWORDS:
            continue
        filtered.append(phrase)
    return tuple(filtered[:8])


def build_coverage_summary(
    *,
    coverage_terms: list[str] | tuple[str, ...],
    document_family: str,
    document_purpose: str,
) -> str:
    if coverage_terms:
        return "covers topics such as " + ", ".join(_normalize_phrase(item) for item in coverage_terms[:4])
    family = _normalize_phrase(document_family)
    purpose = _normalize_phrase(document_purpose)
    if family and purpose:
        return f"{family} focused on {purpose}"
    if purpose:
        return purpose
    return family or "general reference"


def interpret_document_semantics(
    *,
    source_pdf: str,
    title: str,
    toc: list[str] | tuple[str, ...],
    summary_cues: list[str] | tuple[str, ...],
    discovery_terms: list[str] | tuple[str, ...],
    leading_block_lines: list[str],
    metadata_values: list[str] | None = None,
    page_count: int = 0,
    document_type: str | None = None,
    document_purpose: str | None = None,
    audience: str | None = None,
    evidence_style: str | None = None,
    structure_style: str | None = None,
    facet_terms: list[str] | tuple[str, ...] | None = None,
    inventory_summary: str | None = None,
    document_family: str | None = None,
    coverage_terms: list[str] | tuple[str, ...] | None = None,
    coverage_summary: str | None = None,
) -> DocumentSemantics:
    derived_facets = derive_document_facets(
        source_pdf=source_pdf,
        title=title,
        toc=list(toc),
        summary_cues=list(summary_cues),
        leading_block_lines=leading_block_lines,
        metadata_values=metadata_values or [],
        page_count=page_count,
    )
    resolved_summary_cues = _clean_signal_values(summary_cues, title)
    resolved_discovery_terms = _clean_signal_values(discovery_terms, title)
    resolved_document_type = document_type or str(derived_facets["document_type"])
    resolved_document_purpose = document_purpose or str(derived_facets["document_purpose"])
    resolved_audience = audience or str(derived_facets["audience"])
    resolved_evidence_style = evidence_style or str(derived_facets["evidence_style"])
    resolved_structure_style = structure_style or str(derived_facets["structure_style"])
    resolved_facet_terms = tuple(
        item
        for item in (facet_terms if facet_terms else derived_facets["facet_terms"])
        if isinstance(item, str)
    )
    resolved_document_family = document_family or classify_document_family(
        document_type=resolved_document_type,
        document_purpose=resolved_document_purpose,
        audience=resolved_audience,
        evidence_style=resolved_evidence_style,
        structure_style=resolved_structure_style,
    )
    resolved_inventory_summary = inventory_summary or build_inventory_summary(
        title=title,
        document_type=resolved_document_type,
        document_purpose=resolved_document_purpose,
        audience=resolved_audience,
        evidence_style=resolved_evidence_style,
        structure_style=resolved_structure_style,
        summary_cues=resolved_summary_cues,
    )
    resolved_coverage_terms = tuple(
        item
        for item in (coverage_terms if coverage_terms else derive_coverage_terms(
            title=title,
            discovery_terms=resolved_discovery_terms,
            summary_cues=resolved_summary_cues,
            toc=toc,
        ))
        if isinstance(item, str)
    )
    resolved_coverage_summary = coverage_summary or build_coverage_summary(
        coverage_terms=resolved_coverage_terms,
        document_family=resolved_document_family,
        document_purpose=resolved_document_purpose,
    )
    return DocumentSemantics(
        document_type=resolved_document_type,
        document_purpose=resolved_document_purpose,
        audience=resolved_audience,
        evidence_style=resolved_evidence_style,
        structure_style=resolved_structure_style,
        document_family=resolved_document_family,
        facet_terms=resolved_facet_terms,
        summary_cues=resolved_summary_cues,
        discovery_terms=resolved_discovery_terms,
        inventory_summary=resolved_inventory_summary,
        coverage_terms=resolved_coverage_terms,
        coverage_summary=resolved_coverage_summary,
    )


def semantic_match_terms(semantics: DocumentSemantics) -> set[str]:
    tokens = set()
    tokens |= facet_token_terms(
        {
            "document_type": semantics.document_type,
            "document_purpose": semantics.document_purpose,
            "audience": semantics.audience,
            "evidence_style": semantics.evidence_style,
            "structure_style": semantics.structure_style,
            "facet_terms": list(semantics.facet_terms),
        }
    )
    for value in (
        list(semantics.discovery_terms)
        + list(semantics.summary_cues)
        + list(semantics.coverage_terms)
        + [semantics.inventory_summary, semantics.coverage_summary, semantics.document_family]
    ):
        tokens |= _tokenize(value, min_len=2)
    return tokens


def query_semantic_preferences(query: str) -> dict[str, set[str]]:
    lowered = query.lower()
    terms = _tokenize(lowered, min_len=2)
    family_preferences = {
        family
        for family, hints in QUERY_FAMILY_HINTS.items()
        if any(hint in lowered for hint in hints)
    }
    purpose_preferences = {
        purpose
        for purpose, hints in QUERY_PURPOSE_HINTS.items()
        if any(hint in lowered for hint in hints)
    }
    return {
        "families": family_preferences,
        "purposes": purpose_preferences,
        "terms": terms,
    }


def relationship_signal(
    *,
    first: DocumentSemantics,
    second: DocumentSemantics,
    first_topical_terms: set[str] | None = None,
    second_topical_terms: set[str] | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    first_coverage = {_normalize_phrase(item).lower() for item in first.coverage_terms}
    second_coverage = {_normalize_phrase(item).lower() for item in second.coverage_terms}
    shared_coverage = sorted(first_coverage & second_coverage)
    first_topics = set(first_topical_terms or set())
    second_topics = set(second_topical_terms or set())
    shared_topics = sorted(first_topics & second_topics)
    same_family = first.document_family == second.document_family
    same_purpose = first.document_purpose == second.document_purpose
    same_audience = first.audience == second.audience

    if same_family and same_purpose and (shared_coverage or shared_topics):
        details = shared_coverage[:3] or shared_topics[:3]
        summary = "The sources overlap in family, purpose, and core coverage"
        if details:
            summary += " around " + ", ".join(details)
        return "overlap", summary + ".", tuple(details)

    if same_purpose or same_audience or shared_coverage or shared_topics:
        details = shared_coverage[:3] or shared_topics[:3]
        summary = "The sources are complementary: they overlap in topic or purpose but differ in framing"
        if details:
            summary += " around " + ", ".join(details)
        return "complement", summary + ".", tuple(details)

    summary = (
        "The sources diverge in document family or purpose and should be treated as distinct perspectives."
    )
    return "diverge", summary, tuple()
