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
    return re.sub(r"\s+", " ", text).strip()


def _query_terms(query: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[a-zA-Z]{2,}", query.lower())
        if token not in STOPWORDS
    }
    return terms


def _split_sentences(text: str) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}", text)
    return [part.strip() for part in parts if len(part.strip()) >= 30]


def _detect_query_intent(query: str, query_terms: set[str]) -> str:
    query_lower = query.lower()
    if query_lower.startswith("what is") or "definition" in query_terms or "define" in query_terms:
        return "definition"
    if "transmitted" in query_terms or "transmission" in query_terms:
        return "transmission"
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
    return candidates[:max_sentences]


def _compress_sentences(evidence: list[EvidenceSentence]) -> str:
    if not evidence:
        return "No grounded answer could be assembled from the retrieved context."

    fragments = [item.sentence.rstrip(".") for item in evidence]
    return " ".join(f"{fragment}." for fragment in fragments)


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
    answer = _compress_sentences(evidence)
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
    answer = _compress_sentences(evidence)
    return GroundedAnswer(
        query=query,
        answer=answer,
        evidence=evidence,
        top_k_hits=top_k_hits,
        expanded_hits=expanded_hits,
    )
