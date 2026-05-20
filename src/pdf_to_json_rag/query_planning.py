"""Lightweight query planning for discovery vs evidence paths."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .document_inventory import ShortlistCandidate, shortlist_document_candidates
from .intent_config import (
    detect_structured_intent,
    matching_source_doc_ids as configured_matching_source_doc_ids,
    preferred_source_doc_id as configured_source_doc_id,
)


MULTI_DOC_COMPARE_TERMS = {"compare", "versus", "vs"}
TREATMENT_ENTITY_TERMS = {
    "vitamin",
    "echinacea",
    "propolis",
    "zinc",
    "honey",
    "ginseng",
    "handwashing",
}
UNSUPPORTED_DISCOVERY_TERMS = {
    "insulin",
    "gadolinium",
    "monoclonal",
    "monoclonals",
    "lease",
    "leases",
    "clause",
    "clauses",
    "rent",
    "escalation",
}


@dataclass(frozen=True)
class QueryPlan:
    query: str
    query_terms: frozenset[str]
    query_class: str
    query_intent: str
    answer_mode: str
    inventory_doc_ids: tuple[str, ...]
    matched_doc_ids: tuple[str, ...]
    candidate_doc_ids: tuple[str, ...]
    preferred_doc_id: str | None
    query_features: dict[str, bool] = field(default_factory=dict)
    mode_scores: dict[str, float] = field(default_factory=dict)
    chosen_rationale: tuple[str, ...] = ()
    shortlist: tuple[ShortlistCandidate, ...] = ()


def _query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{2,}", query.lower()))


def _is_explicit_plural_routing(query_lower: str) -> bool:
    return "which file or files" in query_lower or "which files" in query_lower


def has_overview_cue(query_lower: str, query_terms: set[str]) -> bool:
    return (
        ("what does" in query_lower and "cover" in query_lower)
        or ("what is" in query_lower and "about" in query_lower)
        or ("what kind of" in query_lower and bool({"document", "file", "source"} & query_terms))
        or ("what type of" in query_lower and bool({"document", "file", "source"} & query_terms))
        or ("what is the purpose of" in query_lower)
    )


def has_routing_cue(query_lower: str, query_terms: set[str]) -> bool:
    return (
        bool({"file", "document", "source"} & query_terms)
        and (
            "which file" in query_lower
            or "which document" in query_lower
            or "which source" in query_lower
        )
        and (
            "most relevant" in query_lower
            or "best source" in query_lower
            or "best document" in query_lower
            or "best file" in query_lower
        )
    )


def has_compare_cue(query_lower: str, query_terms: set[str]) -> bool:
    return bool(query_terms.intersection(MULTI_DOC_COMPARE_TERMS)) or "how do they differ" in query_lower


def has_plural_source_cue(query_lower: str, query_terms: set[str]) -> bool:
    return (
        "which sources" in query_lower
        or "which documents" in query_lower
        or ("which files" in query_lower and not has_routing_cue(query_lower, query_terms))
    )


def has_unsupported_domain_cue(query_terms: set[str]) -> bool:
    return bool(query_terms.intersection(UNSUPPORTED_DISCOVERY_TERMS))


def _build_query_features(
    query_lower: str,
    query_terms: set[str],
    *,
    explicit_source_doc_id: str | None,
    metadata_matches: list[str],
    shortlist: list[ShortlistCandidate],
) -> dict[str, bool]:
    return {
        "overview_cue": has_overview_cue(query_lower, query_terms),
        "routing_cue": has_routing_cue(query_lower, query_terms),
        "compare_cue": has_compare_cue(query_lower, query_terms),
        "plural_source_cue": has_plural_source_cue(query_lower, query_terms),
        "justification_cue": query_lower.startswith("why") and any(
            phrase in query_lower
            for phrase in (
                "most relevant source",
                "most relevant file",
                "best source",
                "best file",
                "best document",
                "best match",
            )
        ),
        "unsupported_domain_cue": has_unsupported_domain_cue(query_terms),
        "explicit_source_anchor": bool(explicit_source_doc_id),
        "metadata_matches": bool(metadata_matches),
        "shortlist_available": bool(shortlist),
        "multi_doc_candidates": len(metadata_matches) >= 2 or len(shortlist) >= 2,
        "plural_routing": _is_explicit_plural_routing(query_lower),
        "treatment_compare_cue": bool(query_terms.intersection(TREATMENT_ENTITY_TERMS)) and has_compare_cue(query_lower, query_terms),
    }


def _score_answer_modes(
    features: dict[str, bool],
    *,
    metadata_match_count: int,
    shortlist_count: int,
) -> dict[str, float]:
    scores = {
        "grounded_evidence": 0.5,
        "document_overview": 0.0,
        "document_routing": 0.0,
        "source_justification": 0.0,
        "source_listing": 0.0,
        "cross_document_compare": 0.0,
    }
    if features["overview_cue"]:
        scores["document_overview"] += 4.0
        if shortlist_count:
            scores["document_overview"] += 1.0
        if features["explicit_source_anchor"]:
            scores["document_overview"] += 0.5
    if features["routing_cue"]:
        scores["document_routing"] += 4.0
        if shortlist_count:
            scores["document_routing"] += 1.0
        if metadata_match_count:
            scores["document_routing"] += 1.0
    if features["justification_cue"]:
        scores["source_justification"] += 4.0
        if shortlist_count:
            scores["source_justification"] += 1.0
        if metadata_match_count:
            scores["source_justification"] += 1.0
    if features["plural_source_cue"]:
        scores["source_listing"] += 4.0
        if shortlist_count:
            scores["source_listing"] += 0.5
        if metadata_match_count:
            scores["source_listing"] += 1.0
        if features["multi_doc_candidates"]:
            scores["source_listing"] += 1.0
    if features["compare_cue"]:
        scores["cross_document_compare"] += 4.0
        if features["multi_doc_candidates"]:
            scores["cross_document_compare"] += 1.5
        if features["treatment_compare_cue"]:
            scores["cross_document_compare"] += 1.0

    if features["unsupported_domain_cue"] and not features["explicit_source_anchor"] and not features["metadata_matches"]:
        scores["document_routing"] = 0.0
        scores["source_justification"] = 0.0
        scores["source_listing"] = 0.0
        scores["cross_document_compare"] = 0.0
    if not features["multi_doc_candidates"]:
        scores["cross_document_compare"] = min(scores["cross_document_compare"], 0.5)
    return scores


def _rationale_for_mode(answer_mode: str, features: dict[str, bool], shortlist: list[ShortlistCandidate]) -> tuple[str, ...]:
    rationale: list[str] = []
    if answer_mode == "document_overview":
        if features["overview_cue"]:
            rationale.append("overview_cue")
        if shortlist:
            rationale.append("shortlist_available")
    elif answer_mode == "document_routing":
        if features["routing_cue"]:
            rationale.append("routing_cue")
        if features["metadata_matches"]:
            rationale.append("metadata_match")
        if shortlist:
            rationale.append("inventory_shortlist")
    elif answer_mode == "source_justification":
        if features["justification_cue"]:
            rationale.append("justification_cue")
        if features["metadata_matches"]:
            rationale.append("metadata_match")
        if shortlist:
            rationale.append("inventory_shortlist")
    elif answer_mode == "source_listing":
        if features["plural_source_cue"]:
            rationale.append("plural_source_cue")
        if features["multi_doc_candidates"]:
            rationale.append("multi_doc_candidates")
    elif answer_mode == "cross_document_compare":
        if features["compare_cue"]:
            rationale.append("compare_cue")
        if features["multi_doc_candidates"]:
            rationale.append("multi_doc_candidates")
    else:
        rationale.append("fallback_evidence_lookup")
    return tuple(rationale)


def _answer_mode_for(query_class: str, query_intent: str) -> str:
    if query_class == "structured_form":
        return "structured_form"
    if query_intent == "document_overview":
        return "document_overview"
    if query_intent == "document_routing":
        return "document_routing"
    if query_intent == "source_listing":
        return "source_listing"
    if query_intent == "source_justification":
        return "source_justification"
    if query_intent == "cross_document_compare":
        return "cross_document_compare"
    return "grounded_evidence"


def plan_query(query: str) -> QueryPlan:
    query_lower = query.lower()
    query_terms = _query_terms(query)
    structured_intent = detect_structured_intent(query, query_terms)
    if structured_intent:
        source_doc_id = configured_source_doc_id(query)
        matched = (source_doc_id,) if source_doc_id else tuple()
        return QueryPlan(
            query=query,
            query_terms=frozenset(query_terms),
            query_class="structured_form",
            query_intent=structured_intent,
            answer_mode=_answer_mode_for("structured_form", structured_intent),
            inventory_doc_ids=matched,
            matched_doc_ids=matched,
            candidate_doc_ids=matched,
            preferred_doc_id=source_doc_id,
            chosen_rationale=("structured_intent",),
        )

    explicit_source_doc_id = configured_source_doc_id(query)
    metadata_matches = list(configured_matching_source_doc_ids(query, allow_topical=True))
    shortlist = shortlist_document_candidates(query, limit=6)
    inventory_doc_ids = tuple(candidate.entry.doc_id for candidate in shortlist)
    features = _build_query_features(
        query_lower,
        query_terms,
        explicit_source_doc_id=explicit_source_doc_id,
        metadata_matches=metadata_matches,
        shortlist=shortlist,
    )
    mode_scores = _score_answer_modes(
        features,
        metadata_match_count=len(metadata_matches),
        shortlist_count=len(shortlist),
    )
    answer_mode = max(mode_scores.items(), key=lambda item: (item[1], item[0]))[0]
    if mode_scores[answer_mode] <= mode_scores["grounded_evidence"]:
        answer_mode = "grounded_evidence"

    query_class = "evidence_lookup"
    query_intent = "generic"
    matched_doc_ids: tuple[str, ...] = tuple()
    candidate_doc_ids: tuple[str, ...] = tuple()
    preferred_doc_id: str | None = None

    if answer_mode == "document_overview":
        query_class = "document_facet"
        query_intent = "document_overview"
        preferred_doc_id = explicit_source_doc_id or (inventory_doc_ids[0] if inventory_doc_ids else None)
        matched_doc_ids = (preferred_doc_id,) if preferred_doc_id else tuple()
        candidate_doc_ids = matched_doc_ids or inventory_doc_ids[:3]
    elif answer_mode == "document_routing":
        query_class = "document_discovery"
        query_intent = "document_routing"
        matched_doc_ids = tuple(metadata_matches) if metadata_matches else inventory_doc_ids[:4]
        if matched_doc_ids and not features["plural_routing"]:
            preferred_doc_id = matched_doc_ids[0]
        candidate_doc_ids = matched_doc_ids
    elif answer_mode == "source_justification":
        query_class = "document_discovery"
        query_intent = "source_justification"
        preferred_doc_id = explicit_source_doc_id or (metadata_matches[0] if metadata_matches else (inventory_doc_ids[0] if inventory_doc_ids else None))
        matched_doc_ids = (preferred_doc_id,) if preferred_doc_id else tuple()
        candidate_doc_ids = matched_doc_ids or inventory_doc_ids[:3]
    elif answer_mode == "source_listing":
        query_class = "cross_document"
        query_intent = "source_listing"
        matched_doc_ids = tuple(metadata_matches[:4]) if metadata_matches else inventory_doc_ids[:4]
        candidate_doc_ids = matched_doc_ids
    elif answer_mode == "cross_document_compare":
        query_class = "cross_document"
        query_intent = "cross_document_compare"
        matched_doc_ids = tuple(metadata_matches[:4]) if len(metadata_matches) >= 2 else inventory_doc_ids[:4]
        if len(matched_doc_ids) < 2:
            query_class = "evidence_lookup"
            query_intent = "generic"
            answer_mode = "grounded_evidence"
        else:
            candidate_doc_ids = matched_doc_ids

    return QueryPlan(
        query=query,
        query_terms=frozenset(query_terms),
        query_class=query_class,
        query_intent=query_intent,
        answer_mode=answer_mode,
        inventory_doc_ids=inventory_doc_ids,
        matched_doc_ids=matched_doc_ids,
        candidate_doc_ids=candidate_doc_ids,
        preferred_doc_id=preferred_doc_id,
        query_features=features,
        mode_scores=mode_scores,
        chosen_rationale=_rationale_for_mode(answer_mode, features, shortlist),
        shortlist=tuple(shortlist),
    )
