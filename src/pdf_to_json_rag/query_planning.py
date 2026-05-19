"""Lightweight query planning for discovery vs evidence paths."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .document_inventory import shortlist_documents
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
    preferred_doc_id: str | None


def _query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{2,}", query.lower()))


def _is_explicit_plural_routing(query_lower: str) -> bool:
    return "which file or files" in query_lower or "which files" in query_lower


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
    unsupported_discovery = bool(query_terms.intersection(UNSUPPORTED_DISCOVERY_TERMS))
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
            preferred_doc_id=source_doc_id,
        )

    explicit_source_doc_id = configured_source_doc_id(query)
    metadata_matches = list(configured_matching_source_doc_ids(query, allow_topical=True))
    inventory = shortlist_documents(query, limit=6)
    inventory_doc_ids = tuple(entry.doc_id for entry in inventory)

    if (
        ("what does" in query_lower and "cover" in query_lower)
        or ("what is" in query_lower and "about" in query_lower)
        or ("what kind of" in query_lower and ("document" in query_lower or "file" in query_lower or "source" in query_lower))
        or ("what type of" in query_lower and ("document" in query_lower or "file" in query_lower or "source" in query_lower))
        or ("what is the purpose of" in query_lower)
    ):
        preferred = explicit_source_doc_id or (inventory_doc_ids[0] if inventory_doc_ids else None)
        matched = (preferred,) if preferred else tuple()
        return QueryPlan(
            query=query,
            query_terms=frozenset(query_terms),
            query_class="document_facet",
            query_intent="document_overview",
            answer_mode=_answer_mode_for("document_facet", "document_overview"),
            inventory_doc_ids=inventory_doc_ids,
            matched_doc_ids=matched,
            preferred_doc_id=preferred,
        )

    if (
        ("which file" in query_lower or "which document" in query_lower or "which source" in query_lower)
        and (
            "most relevant" in query_lower
            or "best source" in query_lower
            or "best document" in query_lower
        )
    ):
        if unsupported_discovery and not explicit_source_doc_id and not metadata_matches:
            return QueryPlan(
                query=query,
                query_terms=frozenset(query_terms),
                query_class="document_discovery",
                query_intent="document_routing",
                answer_mode=_answer_mode_for("document_discovery", "document_routing"),
                inventory_doc_ids=tuple(),
                matched_doc_ids=tuple(),
                preferred_doc_id=None,
            )
        matched = tuple(metadata_matches) if explicit_source_doc_id and metadata_matches else inventory_doc_ids
        if matched and not _is_explicit_plural_routing(query_lower):
            matched = matched[:1]
        preferred = matched[0] if matched and not _is_explicit_plural_routing(query_lower) else None
        return QueryPlan(
            query=query,
            query_terms=frozenset(query_terms),
            query_class="document_discovery",
            query_intent="document_routing",
            answer_mode=_answer_mode_for("document_discovery", "document_routing"),
            inventory_doc_ids=inventory_doc_ids,
            matched_doc_ids=matched,
            preferred_doc_id=preferred,
        )

    if query_lower.startswith("why") and (
        "most relevant source" in query_lower
        or "most relevant file" in query_lower
        or "best source" in query_lower
        or "best file" in query_lower
        or "best document" in query_lower
        or "best match" in query_lower
    ):
        preferred = explicit_source_doc_id or (inventory_doc_ids[0] if inventory_doc_ids else None)
        matched = (preferred,) if preferred else tuple()
        return QueryPlan(
            query=query,
            query_terms=frozenset(query_terms),
            query_class="document_discovery",
            query_intent="source_justification",
            answer_mode=_answer_mode_for("document_discovery", "source_justification"),
            inventory_doc_ids=inventory_doc_ids,
            matched_doc_ids=matched,
            preferred_doc_id=preferred,
        )

    if "which sources" in query_lower or "which documents" in query_lower:
        if unsupported_discovery and not metadata_matches:
            return QueryPlan(
                query=query,
                query_terms=frozenset(query_terms),
                query_class="cross_document",
                query_intent="source_listing",
                answer_mode=_answer_mode_for("cross_document", "source_listing"),
                inventory_doc_ids=tuple(),
                matched_doc_ids=tuple(),
                preferred_doc_id=None,
            )
        matched = tuple(metadata_matches) if metadata_matches else inventory_doc_ids[:4]
        return QueryPlan(
            query=query,
            query_terms=frozenset(query_terms),
            query_class="cross_document",
            query_intent="source_listing",
            answer_mode=_answer_mode_for("cross_document", "source_listing"),
            inventory_doc_ids=inventory_doc_ids,
            matched_doc_ids=matched,
            preferred_doc_id=None,
        )

    if (
        query_terms.intersection(MULTI_DOC_COMPARE_TERMS)
        and (len(metadata_matches) >= 2 or len(inventory_doc_ids) >= 2)
    ) or (
        query_terms.intersection(MULTI_DOC_COMPARE_TERMS)
        and len(query_terms.intersection(TREATMENT_ENTITY_TERMS)) >= 2
    ):
        matched = tuple(metadata_matches[:4]) if len(metadata_matches) >= 2 else inventory_doc_ids[:4]
        return QueryPlan(
            query=query,
            query_terms=frozenset(query_terms),
            query_class="cross_document",
            query_intent="cross_document_compare",
            answer_mode=_answer_mode_for("cross_document", "cross_document_compare"),
            inventory_doc_ids=inventory_doc_ids,
            matched_doc_ids=matched,
            preferred_doc_id=None,
        )

    return QueryPlan(
        query=query,
        query_terms=frozenset(query_terms),
        query_class="evidence_lookup",
        query_intent="generic",
        answer_mode=_answer_mode_for("evidence_lookup", "generic"),
        inventory_doc_ids=tuple(),
        matched_doc_ids=tuple(),
        preferred_doc_id=None,
    )
