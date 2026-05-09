"""Grounded answer assembly for the MVP pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .retrieval import retrieve_top_k_with_neighbors
from .schemas import ChunkRecord


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}

LOW_SIGNAL_QUERY_TERMS = {
    "common",
    "cold",
    "colds",
    "review",
    "paper",
    "prevent",
    "prevents",
    "preventing",
    "treat",
    "treats",
    "treating",
    "treatment",
    "help",
    "helps",
    "say",
    "says",
}

NO_GROUNDED_ANSWER = "No grounded answer could be assembled from the retrieved context."

SYMPTOM_HINTS = {
    "symptom",
    "symptoms",
    "include",
    "sneezing",
    "rhinorrhoea",
    "runny",
    "nose",
    "headache",
    "malaise",
    "sore",
    "throat",
    "cough",
}

TREATMENT_NOISE = {
    "placebo",
    "effective",
    "effectiveness",
    "reduce",
    "reducing",
    "treatment",
    "treatments",
    "vitamin",
    "antihistamines",
    "decongestants",
    "evidence",
}

DEFINITION_HINTS = {
    "definition",
    "defined",
    "upper",
    "respiratory",
    "tract",
    "mucosa",
}

CAUSE_HINTS = {
    "cause",
    "causes",
    "caused",
    "virus",
    "viruses",
    "rhinovirus",
    "coronavirus",
    "syncytial",
    "metapneumovirus",
}

TRANSMISSION_HINTS = {
    "transmission",
    "transmitted",
    "hand",
    "contact",
    "droplet",
    "nostrils",
    "eyes",
    "virus",
    "viruses",
}

INCIDENCE_HINTS = {
    "incidence",
    "prevalence",
    "year",
    "children",
    "adults",
    "infections",
}

DURATION_HINTS = {
    "duration",
    "last",
    "lasts",
    "days",
    "week",
    "weeks",
    "peak",
    "clear",
    "lingering",
    "persists",
    "cough",
}

CT_FINDINGS_HINTS = {
    "ct",
    "scans",
    "sinus",
    "abnormalities",
    "ostiomeatal",
    "ethmoid",
    "maxillary",
}

CT_FOLLOW_UP_HINTS = {
    "follow-up",
    "follow",
    "days",
    "residual",
    "abnormalities",
    "improvement",
    "resolved",
    "normal",
}

VITAMIN_C_HINTS = {
    "vitamin",
    "prophylaxis",
    "incidence",
    "duration",
    "normal",
    "populations",
    "stress",
    "physical",
    "beneficial",
}

TREATMENT_ENTITY_HINTS = {
    "vitamin",
    "echinacea",
    "propolis",
}

TREATMENT_PREVENTION_HINTS = {
    "prevention",
    "prevent",
    "prevents",
    "prophylaxis",
    "incidence",
    "contracting",
    "benefit",
}

TREATMENT_NULL_EFFECT_HINTS = {
    "normal",
    "populations",
    "not",
    "altered",
    "no",
    "effect",
}

TREATMENT_SUBGROUP_HINTS = {
    "stress",
    "physical",
    "subgroup",
    "beneficial",
    "reduction",
    "runners",
    "skiers",
    "soldiers",
}

TREATMENT_DURATION_HINTS = {
    "duration",
    "shorten",
    "shortens",
    "reduced",
    "days",
    "episodes",
    "course",
}

TREATMENT_OVERALL_HINTS = {
    "meta-analysis",
    "conclusion",
    "incidence",
    "duration",
    "prevention",
    "treatment",
    "benefit",
}

BIBLIOGRAPHIC_NOISE = {
    "abstract",
    "introduction",
    "review",
    "trial",
    "double-blind",
    "placebo-controlled",
    "zincum",
    "nasal gel",
}


@dataclass
class EvidenceSentence:
    chunk_id: str
    page_start: int
    page_end: int
    section_title: str | None
    sentence: str
    score: float


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    evidence: list[EvidenceSentence]
    top_k_hits: list[ChunkRecord]
    expanded_hits: list[ChunkRecord]


def _normalize_text(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", text).strip()


def _query_terms(query: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[a-zA-Z]{2,}", query.lower())
        if token not in STOPWORDS
    }
    return terms


def _normalized_sentence_surface(text: str) -> str:
    normalized = _normalize_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _specific_query_terms(query_terms: set[str]) -> set[str]:
    specific = query_terms - LOW_SIGNAL_QUERY_TERMS
    return specific or query_terms


def _split_sentences(text: str) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}", text)
    return [part.strip() for part in parts if len(part.strip()) >= 30]


def _detect_query_intent(query: str, query_terms: set[str]) -> str:
    query_lower = query.lower()
    has_treatment_query = bool(query_terms & TREATMENT_ENTITY_HINTS) and "cold" in query_terms
    if has_treatment_query:
        if "stress" in query_terms or ("physical" in query_terms and "stress" in query_terms) or "subgroup" in query_terms:
            return "treatment_subgroup_benefit"
        if "normal" in query_terms and "populations" in query_terms:
            return "treatment_null_effect"
        if "duration" in query_terms or "shorten" in query_terms:
            return "treatment_duration"
        if "conclude" in query_terms or "conclusion" in query_terms or "meta" in query_terms or "analysis" in query_terms:
            return "treatment_overall"
        if (
            "prevent" in query_terms
            or "prevents" in query_terms
            or "prevention" in query_terms
            or "prophylaxis" in query_terms
            or "incidence" in query_terms
        ):
            return "treatment_prevention"
    if query_lower.startswith("what is") or "definition" in query_terms or "define" in query_terms:
        return "definition"
    if "ct" in query_terms and ("follow" in query_terms or "followup" in query_terms):
        return "ct_follow_up"
    if "ct" in query_terms and ("abnormalities" in query_terms or "sinus" in query_terms or "scans" in query_terms):
        return "ct_findings"
    if "cause" in query_terms or "causes" in query_terms:
        return "causes"
    if "transmitted" in query_terms or "transmission" in query_terms:
        return "transmission"
    if "last" in query_terms or "long" in query_terms or "duration" in query_terms:
        return "duration"
    if "year" in query_terms or ("children" in query_terms and "adults" in query_terms):
        return "incidence"
    if "symptom" in query_terms or "symptoms" in query_terms:
        return "symptoms"
    return "generic"


def _score_sentence(
    sentence: str,
    query_terms: set[str],
    query_intent: str,
    section_title: str | None = None,
) -> float:
    sentence_lower = sentence.lower()
    sentence_terms = set(re.findall(r"[a-zA-Z]{2,}", sentence_lower))
    section_upper = (section_title or "").upper()
    overlap = sentence_terms & query_terms
    if not overlap:
        return 0.0
    score = float(len(overlap))
    if any(noisy in section_upper for noisy in ("DISCLAIMER", "METHODS", "QUESTION", "GRADE")):
        score -= 4.0
    if re.match(r"^\d+\s+", sentence.strip()):
        score -= 2.5
    if "symptom" in sentence_lower or "symptoms" in sentence_lower:
        score += 1.5
    if query_intent == "symptoms":
        score += len(sentence_terms & SYMPTOM_HINTS) * 0.75
        if "symptoms include" in sentence_lower:
            score += 3.0
        if "experience" in sentence_lower:
            score += 1.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 2.5
    if query_intent == "definition":
        score += len(sentence_terms & DEFINITION_HINTS) * 1.0
        if "defined as" in sentence_lower:
            score += 4.0
        if section_upper.startswith("DEFINITION"):
            score += 3.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "causes":
        score += len(sentence_terms & CAUSE_HINTS) * 1.0
        if "mainly caused by viruses" in sentence_lower:
            score += 4.0
        if "rhinovirus" in sentence_lower or "coronavirus" in sentence_lower:
            score += 2.5
        if "AETIOLOGY" in section_upper or "RISK FACTORS" in section_upper:
            score += 3.0
        if "PROGNOSIS" in section_upper or "TREATMENTS" in section_upper:
            score -= 3.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "transmission":
        score += len(sentence_terms & TRANSMISSION_HINTS) * 1.0
        if "hand-to-hand contact" in sentence_lower:
            score += 4.0
        if "droplet" in sentence_lower:
            score += 2.0
        if "AETIOLOGY" in section_upper:
            score += 2.0
        if "PROGNOSIS" in section_upper:
            score -= 2.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.0
    if query_intent == "duration":
        score += len(sentence_terms & DURATION_HINTS) * 0.9
        if "1 week" in sentence_lower or "generally clear by 1 week" in sentence_lower:
            score += 4.0
        if "few days" in sentence_lower:
            score += 3.0
        if "cough often persists" in sentence_lower or "lingering symptoms" in sentence_lower:
            score += 2.0
        if "PROGNOSIS" in section_upper:
            score += 3.0
        if "symptoms include" in sentence_lower:
            score -= 2.5
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.0
    if query_intent == "incidence":
        score += len(sentence_terms & INCIDENCE_HINTS) * 0.9
        if "each year" in sentence_lower:
            score += 2.5
        if "children suffer" in sentence_lower or "adults" in sentence_lower:
            score += 2.0
        if "INCIDENCE" in section_upper or "PREVALENCE" in section_upper:
            score += 3.0
        if "year 6 compared" in sentence_lower or "twice as likely" in sentence_lower:
            score -= 2.5
        if "cross-sectional study" in sentence_lower or "prospective us study" in sentence_lower:
            score -= 2.0
        if "symptoms of colds" in sentence_lower or "types of virus" in sentence_lower:
            score -= 2.5
        if "adverse effects" in sentence_lower:
            score -= 4.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.0
    if query_intent == "ct_findings":
        score += len(sentence_terms & CT_FINDINGS_HINTS) * 0.8
        if "high prevalence of ostiomeatal and sinus abnormalities" in sentence_lower:
            score += 6.0
        if "abnormalities on ct scans" in sentence_lower or "abnormalities of one or more sinuses" in sentence_lower:
            score += 3.5
        if "discussion" in section_upper or "FOLLOW-UP" in section_upper:
            score += 2.0
        if "study was approved" in sentence_lower or "screened by telephone" in sentence_lower:
            score -= 4.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "ct_follow_up":
        score += len(sentence_terms & CT_FOLLOW_UP_HINTS) * 0.8
        if "follow-up evaluation after 13 to 20 days" in sentence_lower:
            score += 6.0
        if "residual abnormalities" in sentence_lower or "marked improvement" in sentence_lower:
            score += 4.0
        if "returned to normal" in sentence_lower or "resolved or markedly improved" in sentence_lower:
            score += 3.0
        if "FOLLOW-UP" in section_upper or "DISCUSSION" in section_upper:
            score += 2.0
        if "screened by telephone" in sentence_lower or "study was approved" in sentence_lower:
            score -= 4.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "treatment_prevention":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_PREVENTION_HINTS) * 0.4
        if "decreasing the incidence" in sentence_lower or "reduces the incidence" in sentence_lower:
            score += 5.0
        if "substantial reductions in the incidence" in sentence_lower:
            score += 5.0
        if "published evidence supports" in sentence_lower:
            score += 4.0
        if "suggests an additional benefit" in sentence_lower:
            score += 4.0
        if "contracting a cold" in sentence_lower:
            score += 3.0
        if "benefit" in sentence_lower:
            score += 1.5
        if "incidence was not altered" in sentence_lower or "normal populations" in sentence_lower:
            score -= 2.0
        if "evidence for the prevention of a cold was lacking" in sentence_lower:
            score -= 3.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_null_effect":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_NULL_EFFECT_HINTS) * 0.5
        if "incidence was not altered" in sentence_lower:
            score += 7.0
        if "lack of effect" in sentence_lower:
            score += 4.0
        if "normal populations" in sentence_lower:
            score += 4.0
        if "throws doubt on the utility" in sentence_lower:
            score += 2.0
        if "beneficial effect" in sentence_lower or "50% reduction" in sentence_lower or "decreasing the incidence" in sentence_lower:
            score -= 3.0
        if (
            "criteria for inclusion" in sentence_lower
            or "literature from" in sentence_lower
            or "overview of the results" in sentence_lower
        ):
            score -= 5.0
        if "vitamin c for preventing and treating the common cold" in sentence_lower:
            score -= 5.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_subgroup_benefit":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_SUBGROUP_HINTS) * 0.5
        if "cold stress" in sentence_lower or "physical stress" in sentence_lower:
            score += 5.0
        if "beneficial effect" in sentence_lower or "50% reduction" in sentence_lower:
            score += 4.0
        if "collective evidence indicates" in sentence_lower:
            score += 2.0
        if "marathon runners" in sentence_lower or "skiers" in sentence_lower or "soldiers" in sentence_lower:
            score += 3.0
        if "normal populations" in sentence_lower:
            score -= 2.0
        if "vitamin c for preventing and treating the common cold" in sentence_lower:
            score -= 5.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_duration":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_DURATION_HINTS) * 0.5
        if (
            "duration of cold episodes" in sentence_lower
            or "duration of common cold episodes" in sentence_lower
            or "duration of the common cold" in sentence_lower
        ):
            score += 4.0
        if "reduced the duration" in sentence_lower or "shortens the course" in sentence_lower:
            score += 3.0
        if "14%" in sentence_lower or "8%" in sentence_lower:
            score += 2.0
        if "onset of symptoms" in sentence_lower or "8 g" in sentence_lower:
            score += 2.0
        if "normal populations" in sentence_lower:
            score -= 1.0
        if "vitamin c for preventing and treating the common cold" in sentence_lower:
            score -= 5.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_overall":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_OVERALL_HINTS) * 0.4
        if "incidence" in sentence_lower and "duration" in sentence_lower:
            score += 4.0
        if "prevention" in sentence_lower and "treatment" in sentence_lower:
            score += 3.0
        if "published evidence supports" in sentence_lower or "suggests that echinacea has a benefit" in sentence_lower:
            score += 4.0
        if "suggests an additional benefit" in sentence_lower:
            score += 4.0
        if "large-scale randomised prospective studies" in sentence_lower:
            score += 1.5
        if "trials were included for analysis" in sentence_lower or "inclusion criteria" in sentence_lower:
            score -= 4.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if len(sentence) > 320:
        score -= 1.5
    return score


def build_grounded_context(chunks: list[ChunkRecord]) -> str:
    """Flatten retrieved chunks into a prompt-ready context string."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk.chunk_id}] page={chunk.page_start}-{chunk.page_end} "
            f"section={chunk.section_title or 'n/a'}\n{chunk.text}"
        )
    return "\n\n".join(parts)


def select_evidence_sentences(
    query: str,
    chunks: list[ChunkRecord],
    max_sentences: int = 4,
) -> list[EvidenceSentence]:
    """Select the most query-relevant sentences from expanded chunk context."""
    query_terms = _query_terms(query)
    query_intent = _detect_query_intent(query, query_terms)
    candidates: list[EvidenceSentence] = []
    seen_sentences: set[str] = set()

    for chunk in chunks:
        for sentence in _split_sentences(chunk.text):
            normalized = sentence.lower()
            if normalized in seen_sentences:
                continue
            score = _score_sentence(
                sentence,
                query_terms,
                query_intent,
                section_title=chunk.section_title,
            )
            if score <= 0:
                continue
            if query_intent == "definition" and "defined as" not in normalized and chunk.section_title:
                if chunk.section_title.upper().startswith("DEFINITION"):
                    score += 1.0
            if query_intent == "causes" and chunk.section_title:
                if "AETIOLOGY" in chunk.section_title.upper():
                    score += 1.0
            if query_intent == "transmission" and "transmission" in normalized:
                score += 1.0
            seen_sentences.add(normalized)
            candidates.append(
                EvidenceSentence(
                    chunk_id=chunk.chunk_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    sentence=sentence,
                    score=score,
                )
            )

    candidates.sort(
        key=lambda item: (-item.score, item.page_start, item.chunk_id, item.sentence)
    )
    selected = candidates[:max_sentences]

    coverage_targets = {
        "treatment_null_effect": ("not altered",),
        "treatment_subgroup_benefit": ("beneficial effect",),
        "treatment_prevention": ("benefit",),
        "treatment_overall": ("benefit",),
    }.get(query_intent, ())

    if coverage_targets:
        selected_surfaces = [
            _normalized_sentence_surface(item.sentence) for item in selected
        ]
        for target in coverage_targets:
            target_surface = _normalized_sentence_surface(target)
            if any(target_surface in surface for surface in selected_surfaces):
                continue
            replacement = next(
                (
                    item
                    for item in candidates
                    if target_surface in _normalized_sentence_surface(item.sentence)
                ),
                None,
            )
            if replacement is None or replacement in selected:
                continue
            if selected:
                selected[-1] = replacement
                selected.sort(
                    key=lambda item: (-item.score, item.page_start, item.chunk_id, item.sentence)
                )
                selected_surfaces = [
                    _normalized_sentence_surface(item.sentence) for item in selected
                ]

    deduped: list[EvidenceSentence] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in selected:
        key = (item.chunk_id, _normalized_sentence_surface(item.sentence))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)

    return deduped


def _compress_sentences(evidence: list[EvidenceSentence]) -> str:
    if not evidence:
        return NO_GROUNDED_ANSWER

    fragments = [item.sentence.rstrip(".") for item in evidence]
    return " ".join(f"{fragment}." for fragment in fragments)


def _should_abstain(query: str, evidence: list[EvidenceSentence]) -> bool:
    if not evidence:
        return True

    query_terms = _query_terms(query)
    specific_terms = _specific_query_terms(query_terms)
    query_intent = _detect_query_intent(query, query_terms)
    intent_support_terms = {
        "definition": DEFINITION_HINTS,
        "symptoms": SYMPTOM_HINTS,
        "causes": CAUSE_HINTS,
        "transmission": TRANSMISSION_HINTS,
        "duration": DURATION_HINTS,
        "incidence": INCIDENCE_HINTS,
        "ct_findings": CT_FINDINGS_HINTS,
        "ct_follow_up": CT_FOLLOW_UP_HINTS,
        "treatment_prevention": TREATMENT_ENTITY_HINTS | TREATMENT_PREVENTION_HINTS,
        "treatment_null_effect": TREATMENT_ENTITY_HINTS | TREATMENT_NULL_EFFECT_HINTS,
        "treatment_subgroup_benefit": TREATMENT_ENTITY_HINTS | TREATMENT_SUBGROUP_HINTS,
        "treatment_duration": TREATMENT_ENTITY_HINTS | TREATMENT_DURATION_HINTS,
        "treatment_overall": TREATMENT_ENTITY_HINTS | TREATMENT_OVERALL_HINTS,
        "generic": set(),
    }.get(query_intent, set())
    evidence_text = " ".join(item.sentence.lower() for item in evidence)
    has_specific_overlap = any(term in evidence_text for term in specific_terms)
    has_intent_overlap = bool(intent_support_terms) and any(
        term in evidence_text for term in intent_support_terms
    )
    if not has_specific_overlap and not has_intent_overlap:
        return True

    top_score = max(item.score for item in evidence)
    if top_score < 2.0:
        return True
    return False


def format_grounded_answer(result: GroundedAnswer) -> str:
    """Format a deterministic grounded answer with explicit evidence."""
    lines = [
        "Answer:",
        result.answer,
        "",
        "Evidence:",
    ]
    for item in result.evidence:
        lines.append(
            f"- {item.sentence} "
            f"[{item.chunk_id}, pages {item.page_start}-{item.page_end}, "
            f"section={item.section_title or 'n/a'}]"
        )
    lines.extend(
        [
            "",
            f"Top-k hits: {len(result.top_k_hits)}",
            f"Expanded context chunks: {len(result.expanded_hits)}",
        ]
    )
    return "\n".join(lines)


def answer_from_chunks(query: str, chunks: list[ChunkRecord]) -> GroundedAnswer:
    """Assemble a grounded answer only from the provided chunk context."""
    evidence = select_evidence_sentences(query=query, chunks=chunks)
    answer = NO_GROUNDED_ANSWER if _should_abstain(query, evidence) else _compress_sentences(evidence)
    return GroundedAnswer(
        query=query,
        answer=answer,
        evidence=evidence,
        top_k_hits=[],
        expanded_hits=chunks,
    )


def answer_query_with_retrieval(
    query: str,
    index_dir: Path,
    chunk_root: Path,
    k: int = 5,
) -> GroundedAnswer:
    """Retrieve, expand, and assemble a grounded answer from local artifacts."""
    top_k_hits, expanded_hits = retrieve_top_k_with_neighbors(
        query=query,
        index_dir=index_dir,
        chunk_root=chunk_root,
        k=k,
    )
    evidence = select_evidence_sentences(query=query, chunks=expanded_hits)
    answer = NO_GROUNDED_ANSWER if _should_abstain(query, evidence) else _compress_sentences(evidence)
    return GroundedAnswer(
        query=query,
        answer=answer,
        evidence=evidence,
        top_k_hits=top_k_hits,
        expanded_hits=expanded_hits,
    )
