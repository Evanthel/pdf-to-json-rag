"""Lightweight query planning for discovery vs evidence paths."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .document_inventory import ShortlistCandidate, shortlist_document_candidates
from .intent_config import (
    detect_structured_intent,
    matching_source_doc_ids as configured_matching_source_doc_ids,
    preferred_source_doc_id as configured_source_doc_id,
    resolve_preferred_source_doc_id,
)


MULTI_DOC_COMPARE_TERMS = {"compare", "versus", "vs"}
SOURCE_ENTITY_TERMS = {
    "book",
    "document",
    "file",
    "form",
    "guide",
    "manual",
    "note",
    "report",
    "source",
    "statement",
}
TREATMENT_ENTITY_TERMS = {
    "vitamin",
    "echinacea",
    "propolis",
    "zinc",
    "honey",
    "ginseng",
    "handwashing",
}
COMMON_COLD_TERMS = {"cold", "colds"}
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
    )


def has_type_cue(query_lower: str, query_terms: set[str], *, has_source_anchor: bool = False) -> bool:
    return (
        ("what kind of" in query_lower and bool(SOURCE_ENTITY_TERMS & query_terms))
        or ("what type of" in query_lower and bool(SOURCE_ENTITY_TERMS & query_terms))
        or (has_source_anchor and query_lower.startswith("what kind of"))
        or (has_source_anchor and query_lower.startswith("what type of"))
    )


def has_purpose_cue(query_lower: str, query_terms: set[str], *, has_source_anchor: bool = False) -> bool:
    return (
        "what is the purpose of" in query_lower
        or ("purpose of this" in query_lower and bool(SOURCE_ENTITY_TERMS & query_terms))
        or ("what is this document for" in query_lower)
        or ("what is this file for" in query_lower)
        or (has_source_anchor and query_lower.startswith("what is ") and query_lower.endswith(" for"))
    )


def has_audience_cue(query_lower: str, query_terms: set[str], *, has_source_anchor: bool = False) -> bool:
    return (
        "who is this document for" in query_lower
        or "who is this file for" in query_lower
        or "who is the audience" in query_lower
        or ("who is it for" in query_lower and bool(SOURCE_ENTITY_TERMS & query_terms))
        or ("intended audience" in query_lower)
        or (
            query_lower.startswith("who is ")
            and " for" in query_lower
            and (has_source_anchor or bool(SOURCE_ENTITY_TERMS & query_terms))
        )
    )


def has_confidence_cue(query_lower: str, query_terms: set[str], *, has_source_anchor: bool = False) -> bool:
    return (
        "how confident" in query_lower
        or "how certain" in query_lower
        or "how sure" in query_lower
        or "confidence in this classification" in query_lower
        or (
            has_source_anchor
            and (
                "how confident is" in query_lower
                or "how certain is" in query_lower
                or "how sure is" in query_lower
            )
        )
        or (
            ("classification" in query_terms or "overview" in query_terms)
            and bool(SOURCE_ENTITY_TERMS & query_terms)
            and ("confidence" in query_terms or "certain" in query_terms or "sure" in query_terms)
        )
    )


def has_rationale_cue(query_lower: str, query_terms: set[str], *, has_source_anchor: bool = False) -> bool:
    return (
        "why is this document classified" in query_lower
        or "why is this file classified" in query_lower
        or "what supports this classification" in query_lower
        or "why is this classified" in query_lower
        or "why is it classified" in query_lower
        or (
            has_source_anchor
            and (
                "why is" in query_lower
                and ("classified" in query_lower or "treated as" in query_lower)
            )
        )
        or (
            ("classification" in query_terms or "classified" in query_terms)
            and ("why" in query_terms or "supports" in query_terms or "support" in query_terms)
        )
    )


def has_uncertainty_cue(query_lower: str, query_terms: set[str], *, has_source_anchor: bool = False) -> bool:
    return (
        "what are the main limits of this document classification" in query_lower
        or "what are the limits of this classification" in query_lower
        or "what is uncertain about this document classification" in query_lower
        or "what are you unsure about" in query_lower
        or "what are the limitations of this classification" in query_lower
        or (
            has_source_anchor
            and (
                ("limits" in query_terms or "limitations" in query_terms or "uncertain" in query_terms)
                and "classification" in query_terms
            )
        )
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
    has_source_anchor = bool(explicit_source_doc_id or metadata_matches)
    return {
        "overview_cue": has_overview_cue(query_lower, query_terms),
        "type_cue": has_type_cue(query_lower, query_terms, has_source_anchor=has_source_anchor),
        "purpose_cue": has_purpose_cue(query_lower, query_terms, has_source_anchor=has_source_anchor),
        "audience_cue": has_audience_cue(query_lower, query_terms, has_source_anchor=has_source_anchor),
        "confidence_cue": has_confidence_cue(query_lower, query_terms, has_source_anchor=has_source_anchor),
        "rationale_cue": has_rationale_cue(query_lower, query_terms, has_source_anchor=has_source_anchor),
        "uncertainty_cue": has_uncertainty_cue(query_lower, query_terms, has_source_anchor=has_source_anchor),
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
    if features["type_cue"]:
        scores["document_overview"] += 4.5
        if shortlist_count:
            scores["document_overview"] += 1.0
    if features["purpose_cue"]:
        scores["document_overview"] += 4.5
        if shortlist_count:
            scores["document_overview"] += 1.0
    if features["audience_cue"]:
        scores["document_overview"] += 4.5
        if shortlist_count:
            scores["document_overview"] += 1.0
    if features["confidence_cue"]:
        scores["document_overview"] += 4.5
        if shortlist_count:
            scores["document_overview"] += 0.75
        if features["explicit_source_anchor"] or features["metadata_matches"]:
            scores["document_overview"] += 0.5
    if features["rationale_cue"]:
        scores["document_overview"] += 4.25
        if shortlist_count:
            scores["document_overview"] += 0.75
        if features["explicit_source_anchor"] or features["metadata_matches"]:
            scores["document_overview"] += 0.5
    if features["uncertainty_cue"]:
        scores["document_overview"] += 4.25
        if shortlist_count:
            scores["document_overview"] += 0.75
        if features["explicit_source_anchor"] or features["metadata_matches"]:
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
        if features["type_cue"]:
            rationale.append("type_cue")
        if features["purpose_cue"]:
            rationale.append("purpose_cue")
        if features["audience_cue"]:
            rationale.append("audience_cue")
        if features["confidence_cue"]:
            rationale.append("confidence_cue")
        if features["rationale_cue"]:
            rationale.append("rationale_cue")
        if features["uncertainty_cue"]:
            rationale.append("uncertainty_cue")
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


def _detect_evidence_query_intent(query: str, query_terms: set[str]) -> str:
    query_lower = query.lower()
    has_common_cold = bool(query_terms.intersection(COMMON_COLD_TERMS))
    has_treatment_entity = bool(query_terms.intersection(TREATMENT_ENTITY_TERMS))
    if "antibiotic" in query_terms or "antibiotics" in query_terms:
        return "antibiotics"
    if "ct" in query_terms and ("follow" in query_terms or "followup" in query_terms):
        return "ct_follow_up"
    if "ct" in query_terms and ("abnormalities" in query_terms or "sinus" in query_terms or "scans" in query_terms):
        return "ct_findings"
    if (
        ("preventive" in query_terms and "interventions" in query_terms)
        or ("handwashing" in query_terms and "prevent" in query_terms)
        or ("best" in query_terms and "evidence" in query_terms and ("prevent" in query_terms or "preventive" in query_terms))
    ):
        return "review_prevention"
    if has_treatment_entity and has_common_cold:
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
            or "preventing" in query_terms
            or "prevention" in query_terms
            or "prophylaxis" in query_terms
            or "incidence" in query_terms
        ):
            return "treatment_prevention"
    if "cause" in query_terms or "causes" in query_terms:
        return "causes"
    if "transmission" in query_terms or "spread" in query_terms:
        return "transmission"
    if "duration" in query_terms or "long" in query_terms:
        return "duration"
    if "symptom" in query_terms or "symptoms" in query_terms:
        return "symptoms"
    if query_lower.startswith("what is") or "definition" in query_terms or "define" in query_terms:
        return "definition"
    return "generic"


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
        if features["type_cue"]:
            query_intent = "document_type"
        elif features["purpose_cue"]:
            query_intent = "document_purpose"
        elif features["audience_cue"]:
            query_intent = "document_audience"
        elif features["confidence_cue"]:
            query_intent = "document_confidence"
        elif features["rationale_cue"]:
            query_intent = "document_classification_rationale"
        elif features["uncertainty_cue"]:
            query_intent = "document_classification_limits"
        else:
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
        matched_doc_ids = tuple(metadata_matches[:4]) if len(metadata_matches) >= 2 else inventory_doc_ids[:4]
        candidate_doc_ids = matched_doc_ids
    elif answer_mode == "cross_document_compare":
        query_class = "cross_document"
        query_intent = "cross_document_compare"
        matched_doc_ids = tuple(metadata_matches[:4]) if len(metadata_matches) >= 2 else inventory_doc_ids[:4]
        if len(matched_doc_ids) < 2:
            query_class = "evidence_lookup"
            query_intent = _detect_evidence_query_intent(query, query_terms)
            answer_mode = "grounded_evidence"
        else:
            candidate_doc_ids = matched_doc_ids
    else:
        query_intent = _detect_evidence_query_intent(query, query_terms)
        if query_intent != "generic":
            preferred_doc_id = resolve_preferred_source_doc_id(
                query,
                query_class=query_class,
                query_intent=query_intent,
                planned_preferred_doc_id=explicit_source_doc_id,
            )
            matched_doc_ids = (preferred_doc_id,) if preferred_doc_id else tuple()
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
