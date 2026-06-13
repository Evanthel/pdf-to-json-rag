"""Lightweight answer-claim to evidence alignment helpers."""

from __future__ import annotations

import re
from typing import Any


ALIGNMENT_STATUSES = ("exact", "fragment", "weak", "unsupported")


def split_claim_sentences(answer: str) -> list[str]:
    normalized = " ".join((answer or "").split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _surface(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]{3,}", value.lower())
        if token
        not in {
            "and",
            "are",
            "but",
            "for",
            "from",
            "has",
            "have",
            "into",
            "not",
            "the",
            "this",
            "that",
            "with",
        }
    }


def _fragment_text(fragment: Any) -> str:
    if isinstance(fragment, str):
        return fragment
    sentence = getattr(fragment, "sentence", None)
    if isinstance(sentence, str):
        return sentence
    text = getattr(fragment, "text", None)
    if isinstance(text, str):
        return text
    return str(fragment)


def _fragment_id(fragment: Any) -> str | None:
    return getattr(fragment, "chunk_id", None) if not isinstance(fragment, str) else None


def _alignment_status(claim: str, support: str) -> tuple[str, float]:
    claim_surface = _surface(claim)
    support_surface = _surface(support)
    if not claim_surface or not support_surface:
        return "unsupported", 0.0
    if claim_surface in support_surface:
        return "exact", 1.0

    claim_terms = _terms(claim_surface)
    support_terms = _terms(support_surface)
    if not claim_terms:
        return "unsupported", 0.0
    overlap = len(claim_terms & support_terms) / max(1, len(claim_terms))
    if overlap >= 0.72:
        return "fragment", round(overlap, 3)
    if overlap >= 0.4:
        return "weak", round(overlap, 3)
    return "unsupported", round(overlap, 3)


def align_answer_claims(
    *,
    answer: str,
    evidence_fragments: list[Any],
    context_fragments: list[Any] | None = None,
) -> dict[str, object]:
    claims = split_claim_sentences(answer)
    fragments = list(evidence_fragments)
    if context_fragments:
        fragments.extend(context_fragments)

    aligned_claims: list[dict[str, object]] = []
    for claim in claims:
        best_status = "unsupported"
        best_score = 0.0
        best_fragment = ""
        best_chunk_id = None
        for fragment in fragments:
            fragment_text = _fragment_text(fragment)
            status, score = _alignment_status(claim, fragment_text)
            if ALIGNMENT_STATUSES.index(status) < ALIGNMENT_STATUSES.index(best_status) or (
                status == best_status and score > best_score
            ):
                best_status = status
                best_score = score
                best_fragment = fragment_text
                best_chunk_id = _fragment_id(fragment)
        aligned_claims.append(
            {
                "claim": claim,
                "status": best_status,
                "score": best_score,
                "chunk_id": best_chunk_id,
                "support_preview": best_fragment[:240],
            }
        )

    supported_count = sum(1 for item in aligned_claims if item["status"] in {"exact", "fragment"})
    weak_count = sum(1 for item in aligned_claims if item["status"] == "weak")
    unsupported_count = sum(1 for item in aligned_claims if item["status"] == "unsupported")
    claim_count = len(aligned_claims)
    return {
        "claim_count": claim_count,
        "supported_claim_count": supported_count,
        "weak_claim_count": weak_count,
        "unsupported_claim_count": unsupported_count,
        "supported_claim_ratio": round(supported_count / claim_count, 3) if claim_count else 0.0,
        "alignment_status": "pass" if unsupported_count == 0 else "needs_review",
        "claims": aligned_claims,
    }
