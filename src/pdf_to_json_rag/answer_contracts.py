"""Shared answer-contract helpers for inspectable answer modes."""

from __future__ import annotations


def build_answer_contract(
    *,
    mode: str,
    primary_doc_ids: list[str] | tuple[str, ...],
    document_families: list[str] | tuple[str, ...],
    summary_type: str | None = None,
    relationship: str | None = None,
    coverage_terms: list[str] | tuple[str, ...] | None = None,
    matched_terms: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    contract: dict[str, object] = {
        "mode": mode,
        "primary_doc_ids": list(primary_doc_ids),
        "document_families": list(document_families),
    }
    if summary_type:
        contract["summary_type"] = summary_type
    if relationship:
        contract["relationship"] = relationship
    if coverage_terms:
        contract["coverage_terms"] = list(coverage_terms)
    if matched_terms:
        contract["matched_terms"] = list(matched_terms)
    return contract
