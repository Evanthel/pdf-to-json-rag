"""Result serialization and quality summaries for the command-line interface."""

import json
from pathlib import Path

from . import __version__


from .cli_shared import (
    CliError,
)


def _human_status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _release_channel_recommendation(
    overall_pass: bool,
    *,
    public_surface_all_pass: bool,
    maintainer_checks_available: bool,
    maintainer_surface_all_pass: bool,
    benchmark_assets_available: bool,
    regression_all_pass: bool,
) -> dict[str, object]:
    if overall_pass:
        reasons = [
            "public CLI smoke checks pass",
            "packaged install verification passes",
        ]
        if maintainer_checks_available and maintainer_surface_all_pass:
            reasons.append("maintainer package and CLI test gates pass")
        if benchmark_assets_available and regression_all_pass:
            reasons.append("internal benchmark regression shards pass")
        else:
            reasons.append(
                "internal benchmark regressions were skipped because benchmark assets were not present in the active data root"
            )
        reasons.append(
            "known limitations are documented and do not block the current public release path"
        )
        return {
            "release_ready": True,
            "suggested_tag": "v0.2.0",
            "why": reasons,
        }
    reasons: list[str] = []
    if not public_surface_all_pass:
        reasons.append("at least one public-surface gate is failing")
    if maintainer_checks_available and not maintainer_surface_all_pass:
        reasons.append("at least one maintainer package or CLI test gate is failing")
    if benchmark_assets_available and not regression_all_pass:
        reasons.append("at least one internal benchmark regression shard is failing")
    if not reasons:
        reasons.append("release gating requirements are not fully satisfied")
    return {
        "release_ready": False,
        "suggested_tag": None,
        "why": reasons,
    }


def _chunk_payload(chunk) -> dict[str, object]:
    retrieval_signals = dict(chunk.retrieval_signals)
    backend_code = retrieval_signals.get("rerank_backend_code")
    backend_label = {
        0.0: "heuristic",
        1.0: "lightweight",
        2.0: "cross_encoder",
    }.get(backend_code, "unknown")
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "section_path": list(chunk.section_path),
        "section_kind": chunk.section_kind,
        "section_role": getattr(chunk, "section_role", None),
        "section_summary": chunk.section_summary,
        "section_coverage_terms": list(chunk.section_coverage_terms),
        "section_content_hints": list(chunk.section_content_hints),
        "structure_confidence": chunk.structure_confidence,
        "layout_confidence": chunk.layout_confidence,
        "chunk_type": chunk.chunk_type,
        "chunk_strategy": getattr(chunk, "chunk_strategy", None),
        "layout_signals": list(getattr(chunk, "layout_signals", []) or []),
        "text_quality_score": getattr(chunk, "text_quality_score", None),
        "preceding_chunk_id": chunk.preceding_chunk_id,
        "following_chunk_id": chunk.following_chunk_id,
        "extraction_method": chunk.extraction_method,
        "quality_score": chunk.quality_score,
        "confidence": chunk.confidence,
        "rerank_backend": backend_label,
        "retrieval_signals": retrieval_signals,
        "noise_labels": list(chunk.noise_labels),
        "preview": chunk.text.replace("\n", " ").strip()[:220],
    }


def _evidence_payload(item) -> dict[str, object]:
    return {
        "chunk_id": item.chunk_id,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "section_title": item.section_title,
        "score": item.score,
        "sentence": item.sentence,
    }


def _document_payload(entry) -> dict[str, object]:
    return {
        "doc_id": entry.doc_id,
        "label": entry.label,
        "title": entry.title,
        "document_family": entry.document_family,
        "document_type": entry.document_type,
        "document_purpose": entry.document_purpose,
        "audience": entry.audience,
        "evidence_style": entry.evidence_style,
        "structure_style": entry.structure_style,
        "structure_confidence": None,
        "layout_confidence": None,
        "semantic_confidence": None,
        "semantic_confidence_label": None,
        "semantic_rationale": [],
        "semantic_warnings": [],
        "inventory_summary": entry.inventory_summary,
        "coverage_summary": entry.coverage_summary,
        "coverage_terms": list(entry.coverage_terms),
        "discovery_terms": list(entry.discovery_terms),
    }


def _section_payload(section) -> dict[str, object]:
    return {
        "section_id": section.section_id,
        "title": section.title,
        "level": section.level,
        "section_kind": section.section_kind,
        "section_role": getattr(section, "section_role", None),
        "page_start": section.page_start,
        "page_end": section.page_end,
        "reading_order_start": section.reading_order_start,
        "reading_order_end": section.reading_order_end,
        "summary": section.summary,
        "coverage_terms": list(section.coverage_terms),
        "content_hints": list(section.content_hints),
        "block_count": getattr(section, "block_count", 0),
        "text_source_profile": list(getattr(section, "text_source_profile", []) or []),
        "layout_signals": list(getattr(section, "layout_signals", []) or []),
        "source_block_count": len(getattr(section, "source_block_ids", []) or []),
        "source_block_roles": list(getattr(section, "source_block_roles", []) or []),
        "structure_confidence": section.structure_confidence,
    }


def _shortlist_candidate_payload(candidate) -> dict[str, object]:
    return {
        "doc_id": candidate.entry.doc_id,
        "label": candidate.entry.label,
        "total_score": round(candidate.breakdown.total, 3),
        "matched_terms": list(candidate.matched_terms),
        "rationale": list(candidate.rationale),
        "breakdown": {
            "title_label_score": round(candidate.breakdown.title_label_score, 3),
            "semantic_discovery_score": round(
                candidate.breakdown.semantic_discovery_score, 3
            ),
            "facet_fit_score": round(candidate.breakdown.facet_fit_score, 3),
            "rarity_distinctive_score": round(
                candidate.breakdown.rarity_distinctive_score, 3
            ),
        },
    }


def _plan_payload(plan, *, verbose: bool = False) -> dict[str, object]:
    payload = {
        "query": plan.query,
        "query_class": plan.query_class,
        "query_intent": plan.query_intent,
        "answer_mode": plan.answer_mode,
        "inventory_doc_ids": list(plan.inventory_doc_ids),
        "matched_doc_ids": list(plan.matched_doc_ids),
        "candidate_doc_ids": list(plan.candidate_doc_ids),
        "preferred_doc_id": plan.preferred_doc_id,
        "chosen_rationale": list(plan.chosen_rationale),
    }
    if verbose:
        payload["query_features"] = dict(plan.query_features)
        payload["mode_scores"] = {
            key: round(value, 3) for key, value in plan.mode_scores.items()
        }
        payload["shortlist"] = [
            _shortlist_candidate_payload(candidate) for candidate in plan.shortlist
        ]
    return payload


def _compact_answer_trace(answer_trace: dict[str, object]) -> dict[str, object]:
    return {
        "query_class": answer_trace.get("query_class"),
        "answer_mode": answer_trace.get("answer_mode"),
        "query_intent": answer_trace.get("query_intent"),
        "candidate_doc_ids": answer_trace.get("candidate_doc_ids", []),
        "retrieval_contract": answer_trace.get("retrieval_contract", {}),
        "document_selection": answer_trace.get("document_selection", {}),
        "document_synthesis": answer_trace.get("document_synthesis", {}),
        "synthesis_prompt_contract": answer_trace.get("synthesis_prompt_contract", {}),
        "synthesis_runtime": answer_trace.get("synthesis_runtime", {}),
        "claim_alignment": answer_trace.get("claim_alignment", {}),
        "template_id": answer_trace.get("template_id"),
        "matched_pattern": answer_trace.get("matched_pattern"),
        "matched_cues": answer_trace.get("matched_cues", []),
        "chosen_rationale": answer_trace.get("chosen_rationale", []),
        "answer_contract": answer_trace.get("answer_contract", {}),
        "support_trace": answer_trace.get("support_trace", []),
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _support_trace_doc_ids(support_trace: object) -> list[str]:
    if not isinstance(support_trace, list):
        return []
    doc_ids: list[str] = []
    for item in support_trace:
        if (
            isinstance(item, dict)
            and item.get("doc_id")
            and item.get("doc_id") != "__comparison__"
        ):
            doc_ids.append(str(item["doc_id"]))
    return list(dict.fromkeys(doc_ids))


def _retrieval_contract_status(
    answer_trace: dict[str, object], *, evidence_count: int = 0
) -> dict[str, object]:
    retrieval_contract = answer_trace.get("retrieval_contract", {})
    document_selection = answer_trace.get("document_selection", {})
    document_synthesis = answer_trace.get("document_synthesis", {})
    answer_contract = answer_trace.get("answer_contract", {})
    support_trace = answer_trace.get("support_trace", [])
    answer_mode = str(answer_trace.get("answer_mode") or "")
    grounded_mode = answer_mode == "grounded_evidence"

    candidate_doc_ids = _string_list(answer_trace.get("candidate_doc_ids"))
    if isinstance(document_selection, dict):
        candidate_doc_ids.extend(
            _string_list(document_selection.get("candidate_doc_ids"))
        )
    candidate_doc_ids = list(dict.fromkeys(candidate_doc_ids))

    selected_doc_ids: list[str] = []
    if isinstance(document_selection, dict):
        selected_doc_ids.extend(
            _string_list(document_selection.get("selected_doc_ids"))
        )
    if isinstance(answer_contract, dict):
        selected_doc_ids.extend(_string_list(answer_contract.get("primary_doc_ids")))
    selected_doc_ids = list(dict.fromkeys(selected_doc_ids))

    support_doc_ids: list[str] = []
    answer_chunk_doc_ids: list[str] = []
    if isinstance(document_synthesis, dict):
        support_doc_ids.extend(_string_list(document_synthesis.get("support_doc_ids")))
        answer_chunk_doc_ids.extend(
            _string_list(document_synthesis.get("answer_chunk_doc_ids"))
        )
    support_doc_ids.extend(_support_trace_doc_ids(support_trace))
    support_doc_ids = list(dict.fromkeys(support_doc_ids))
    answer_chunk_doc_ids = list(dict.fromkeys(answer_chunk_doc_ids))

    support_scope = (
        document_synthesis.get("support_scope")
        if isinstance(document_synthesis, dict)
        else None
    )
    retrieval_path = (
        retrieval_contract.get("retrieval_path")
        if isinstance(retrieval_contract, dict)
        else None
    )
    support_available = bool(evidence_count) if grounded_mode else bool(support_doc_ids)
    allowed_support_docs = set(selected_doc_ids or candidate_doc_ids)
    support_subset_ok = (
        not support_doc_ids
        or not allowed_support_docs
        or set(support_doc_ids).issubset(allowed_support_docs)
    )
    chunk_subset_ok = (
        not answer_chunk_doc_ids
        or not support_doc_ids
        or set(answer_chunk_doc_ids).issubset(set(support_doc_ids))
    )
    selected_from_candidates_ok = (
        not selected_doc_ids
        or not candidate_doc_ids
        or set(selected_doc_ids).issubset(set(candidate_doc_ids))
    )
    checks = [
        {"name": "retrieval_contract_present", "passed": bool(retrieval_contract)},
        {"name": "retrieval_path_present", "passed": bool(retrieval_path)},
        {
            "name": "candidate_docs_present",
            "passed": bool(candidate_doc_ids or selected_doc_ids) or grounded_mode,
        },
        {
            "name": "selected_docs_present",
            "passed": bool(selected_doc_ids) or grounded_mode,
        },
        {
            "name": "support_scope_present",
            "passed": bool(support_scope) or grounded_mode,
        },
        {"name": "support_available", "passed": support_available},
        {
            "name": "selected_docs_from_candidates",
            "passed": selected_from_candidates_ok,
        },
        {"name": "support_docs_match_selection", "passed": support_subset_ok},
        {"name": "answer_chunks_match_support_docs", "passed": chunk_subset_ok},
    ]
    failed = _failed_check_reasons(checks)
    if not retrieval_contract:
        status = "fail"
    elif failed:
        status = "warn"
    else:
        status = "pass"
    return {
        "status": status,
        "checks": checks,
        "reasons": failed,
        "retrieval_path": retrieval_path,
        "support_scope": support_scope,
        "candidate_doc_ids": candidate_doc_ids,
        "selected_doc_ids": selected_doc_ids,
        "support_doc_ids": support_doc_ids,
        "answer_chunk_doc_ids": answer_chunk_doc_ids,
    }


def _support_coverage(
    answer_trace: dict[str, object], *, evidence_count: int = 0
) -> dict[str, object]:
    claim_alignment = answer_trace.get("claim_alignment", {})
    claims = (
        claim_alignment.get("claims", []) if isinstance(claim_alignment, dict) else []
    )
    support_trace_doc_ids = _support_trace_doc_ids(
        answer_trace.get("support_trace", [])
    )
    supported = weak = unsupported = chunk_evidence = document_semantics = 0
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            status = str(claim.get("status") or "")
            has_chunk = bool(claim.get("chunk_id"))
            if status in {"exact", "fragment"}:
                supported += 1
                if has_chunk:
                    chunk_evidence += 1
                elif support_trace_doc_ids:
                    document_semantics += 1
            elif status == "weak":
                weak += 1
                if has_chunk:
                    chunk_evidence += 1
                elif support_trace_doc_ids:
                    document_semantics += 1
            elif status == "unsupported":
                unsupported += 1
    claim_count = (
        len(claims)
        if isinstance(claims, list)
        else int(claim_alignment.get("claim_count", 0) or 0)
        if isinstance(claim_alignment, dict)
        else 0
    )
    if not claim_count and isinstance(claim_alignment, dict):
        supported = int(claim_alignment.get("supported_claim_count", 0) or 0)
        weak = int(claim_alignment.get("weak_claim_count", 0) or 0)
        unsupported = int(claim_alignment.get("unsupported_claim_count", 0) or 0)
        claim_count = int(claim_alignment.get("claim_count", 0) or 0)
    return {
        "claim_count": claim_count,
        "supported_claim_count": supported,
        "weak_claim_count": weak,
        "unsupported_claim_count": unsupported,
        "chunk_evidence_claim_count": chunk_evidence,
        "document_semantics_claim_count": document_semantics,
        "support_trace_doc_ids": support_trace_doc_ids,
        "evidence_count": evidence_count,
    }


def _answer_source_mix(
    answer_trace: dict[str, object], *, evidence_count: int = 0
) -> dict[str, object]:
    support_trace = answer_trace.get("support_trace", [])
    synthesis_runtime = answer_trace.get("synthesis_runtime", {})
    support_trace_count = len(support_trace) if isinstance(support_trace, list) else 0
    support_trace_doc_ids = _support_trace_doc_ids(support_trace)
    llm_invoked = (
        bool(synthesis_runtime.get("invoked"))
        if isinstance(synthesis_runtime, dict)
        else False
    )
    evidence_chunk_ids = _string_list(answer_trace.get("evidence_chunk_ids"))
    return {
        "chunk_evidence": {
            "present": bool(evidence_count or evidence_chunk_ids),
            "evidence_count": evidence_count,
            "evidence_chunk_count": len(evidence_chunk_ids),
        },
        "document_semantics": {
            "present": bool(support_trace_doc_ids),
            "doc_ids": support_trace_doc_ids,
        },
        "support_trace": {
            "present": bool(support_trace_count),
            "item_count": support_trace_count,
        },
        "llm_runtime": {
            "present": llm_invoked,
            "runtime": synthesis_runtime.get("runtime")
            if isinstance(synthesis_runtime, dict)
            else None,
        },
    }


def _answer_contract_health(
    answer_trace: dict[str, object], *, evidence_count: int = 0
) -> dict[str, object]:
    retrieval_contract = answer_trace.get("retrieval_contract", {})
    document_synthesis = answer_trace.get("document_synthesis", {})
    answer_contract = answer_trace.get("answer_contract", {})
    support_trace = answer_trace.get("support_trace", [])
    claim_alignment = answer_trace.get("claim_alignment", {})
    answer_mode = str(answer_trace.get("answer_mode") or "")
    support_doc_ids = (
        document_synthesis.get("support_doc_ids", [])
        if isinstance(document_synthesis, dict)
        else []
    )
    selected_doc_ids = (
        answer_contract.get("primary_doc_ids", [])
        if isinstance(answer_contract, dict)
        else []
    )
    support_scope = (
        document_synthesis.get("support_scope")
        if isinstance(document_synthesis, dict)
        else None
    )
    retrieval_path = (
        retrieval_contract.get("retrieval_path")
        if isinstance(retrieval_contract, dict)
        else None
    )
    has_document_support = bool(support_trace) or bool(support_doc_ids)
    grounded_mode = answer_mode == "grounded_evidence"
    support_available = bool(evidence_count) if grounded_mode else has_document_support
    checks = [
        {"name": "retrieval_contract_present", "passed": bool(retrieval_contract)},
        {"name": "retrieval_path_present", "passed": bool(retrieval_path)},
        {"name": "answer_contract_present", "passed": bool(answer_contract)},
        {"name": "document_synthesis_present", "passed": bool(document_synthesis)},
        {"name": "support_scope_present", "passed": bool(support_scope)},
        {"name": "support_available_for_mode", "passed": support_available},
        {"name": "claim_alignment_present", "passed": bool(claim_alignment)},
    ]
    retrieval_status = _retrieval_contract_status(
        answer_trace, evidence_count=evidence_count
    )
    support_coverage = _support_coverage(answer_trace, evidence_count=evidence_count)
    source_mix = _answer_source_mix(answer_trace, evidence_count=evidence_count)
    return {
        "all_pass": all(item["passed"] for item in checks),
        "checks": checks,
        "retrieval_path": retrieval_path,
        "support_scope": support_scope,
        "selected_doc_ids": list(selected_doc_ids)
        if isinstance(selected_doc_ids, list)
        else [],
        "support_doc_ids": list(support_doc_ids)
        if isinstance(support_doc_ids, list)
        else [],
        "support_trace_count": len(support_trace)
        if isinstance(support_trace, list)
        else 0,
        "evidence_count": evidence_count,
        "retrieval_contract_status": retrieval_status,
        "support_coverage": support_coverage,
        "answer_source_mix": source_mix,
    }


def _grounded_answer_payload(result, *, verbose: bool = False) -> dict[str, object]:
    answer_trace = (
        result.answer_trace if verbose else _compact_answer_trace(result.answer_trace)
    )
    evidence_payload = (
        [_evidence_payload(item) for item in result.evidence] if verbose else []
    )
    contract_health = _answer_contract_health(
        answer_trace, evidence_count=len(result.evidence)
    )
    return {
        "query": result.query,
        "query_intent": result.query_intent,
        "answer": result.answer,
        "answer_trace": answer_trace,
        "contract_health": contract_health,
        "retrieval_contract_status": contract_health["retrieval_contract_status"],
        "support_coverage": contract_health["support_coverage"],
        "answer_source_mix": contract_health["answer_source_mix"],
        **(
            {
                "top_k_hits": [_chunk_payload(chunk) for chunk in result.top_k_hits],
                "expanded_hits": [
                    _chunk_payload(chunk) for chunk in result.expanded_hits
                ],
                "evidence": evidence_payload,
            }
            if verbose
            else {}
        ),
    }


def _write_json_output(
    payload: dict[str, object], output_path: Path | None = None
) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _emit_json(
    command: str, payload: dict[str, object], output_path: Path | None = None
) -> None:
    _write_json_output(
        {
            "command": command,
            "version": __version__,
            "ok": True,
            "result": payload,
        },
        output_path=output_path,
    )


def _emit_error_json(
    command: str | None, error: CliError, output_path: Path | None = None
) -> None:
    _write_json_output(
        {
            "command": command,
            "version": __version__,
            "ok": False,
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
            },
        },
        output_path=output_path,
    )


def _quality_status(
    score: float | int | None, *, warn_at: float = 0.55, pass_at: float = 0.7
) -> str:
    if score is None:
        return "unknown"
    numeric_score = float(score)
    if numeric_score >= pass_at:
        return "pass"
    if numeric_score >= warn_at:
        return "warn"
    return "fail"


QUALITY_PROFILE_THRESHOLDS: dict[str, object] = {
    "processing_quality": {
        "required": [
            "chunks_created",
            "structure_confidence_present",
            "layout_confidence_present",
        ],
        "warn_at": 0.55,
        "pass_at": 0.7,
    },
    "semantic_confidence": {
        "provisional_at": 0.55,
        "well_supported_at": 0.85,
    },
    "retrieval_readiness": {
        "required": [
            "contract_health_available",
            "retrieval_contract_present",
            "support_scope_present",
            "support_available",
            "selected_or_support_docs_present",
        ],
    },
    "answer_trust": {
        "pass_requires": [
            "answer_present",
            "answer_contract_present",
            "claim_alignment_present",
            "no_weak_claims",
            "no_unsupported_claims",
        ],
        "review_when": [
            "weak_claims_present",
            "unsupported_claims_present",
            "claim_support_incomplete",
        ],
    },
}


def _failed_check_reasons(checks: list[dict[str, object]]) -> list[str]:
    return [str(item.get("name")) for item in checks if not bool(item.get("passed"))]


def _processing_drilldown(
    document: dict[str, object], index: dict[str, object]
) -> dict[str, object]:
    extraction_summary = (
        document.get("extraction_summary", {})
        if isinstance(document.get("extraction_summary"), dict)
        else {}
    )
    page_count = document.get("page_count")
    pages_requiring_ocr = extraction_summary.get("pages_requiring_ocr")
    pages_processed_with_ocr = extraction_summary.get("pages_processed_with_ocr")
    section_count = document.get("section_count")
    native_blocks = extraction_summary.get("native_blocks")
    block_role_counts = extraction_summary.get("block_role_counts", {})
    text_source_counts = extraction_summary.get("text_source_counts", {})
    layout_signal_counts = extraction_summary.get("layout_signal_counts", {})
    pdf_inspector = extraction_summary.get("pdf_inspector", {})
    ocr_used = bool(extraction_summary.get("ocr_used") or pages_processed_with_ocr)
    table_or_form_signal_count = 0
    if isinstance(block_role_counts, dict):
        table_or_form_signal_count += int(block_role_counts.get("table_like", 0) or 0)
        table_or_form_signal_count += int(block_role_counts.get("form_field", 0) or 0)
        table_or_form_signal_count += int(block_role_counts.get("key_value", 0) or 0)
    if isinstance(layout_signal_counts, dict):
        table_or_form_signal_count += int(
            layout_signal_counts.get("table_like", 0) or 0
        )
        table_or_form_signal_count += int(layout_signal_counts.get("form_like", 0) or 0)
    text_extraction_coverage = None
    if (
        isinstance(page_count, int)
        and page_count > 0
        and isinstance(pages_requiring_ocr, int)
    ):
        text_extraction_coverage = round(
            (page_count - pages_requiring_ocr) / page_count, 3
        )
    return {
        "page_count": page_count,
        "chunk_count": index.get("chunk_count"),
        "section_count": section_count,
        "native_block_count": native_blocks,
        "ocr_used": ocr_used,
        "pages_requiring_ocr": pages_requiring_ocr,
        "pages_processed_with_ocr": pages_processed_with_ocr,
        "text_extraction_coverage": text_extraction_coverage,
        "text_source_counts": text_source_counts
        if isinstance(text_source_counts, dict)
        else {},
        "block_role_counts": block_role_counts
        if isinstance(block_role_counts, dict)
        else {},
        "layout_signal_counts": layout_signal_counts
        if isinstance(layout_signal_counts, dict)
        else {},
        "table_or_form_signal_count": table_or_form_signal_count,
        "pdf_inspector": pdf_inspector if isinstance(pdf_inspector, dict) else {},
    }


def _processing_diagnostics(
    document: dict[str, object], index: dict[str, object]
) -> dict[str, object]:
    drilldown = _processing_drilldown(document, index)
    chunk_count = int(drilldown.get("chunk_count") or 0)
    section_count = int(drilldown.get("section_count") or 0)
    native_block_count = drilldown.get("native_block_count")
    pages_requiring_ocr = int(drilldown.get("pages_requiring_ocr") or 0)
    pages_processed_with_ocr = int(drilldown.get("pages_processed_with_ocr") or 0)
    table_or_form_signal_count = int(drilldown.get("table_or_form_signal_count") or 0)
    text_extraction_coverage = drilldown.get("text_extraction_coverage")
    structure_confidence = document.get("structure_confidence")
    layout_confidence = document.get("layout_confidence")
    pdf_inspector = (
        drilldown.get("pdf_inspector", {})
        if isinstance(drilldown.get("pdf_inspector"), dict)
        else {}
    )
    taxonomy: list[str] = []
    if native_block_count is None or int(native_block_count or 0) <= 0:
        taxonomy.append("native_text_low")
    if (
        bool(drilldown.get("ocr_used"))
        or pages_requiring_ocr > 0
        or pages_processed_with_ocr > 0
    ):
        taxonomy.append("ocr_required")
    if section_count <= 0 or (
        isinstance(structure_confidence, (int, float))
        and float(structure_confidence) < 0.55
    ):
        taxonomy.append("weak_sections")
    if table_or_form_signal_count >= 3:
        taxonomy.append("table_or_form_heavy")
    if layout_confidence is None or (
        isinstance(layout_confidence, (int, float)) and float(layout_confidence) < 0.55
    ):
        taxonomy.append("layout_uncertain")
    if bool(pdf_inspector.get("has_encoding_issues")):
        taxonomy.append("encoding_uncertain")
    disagreements = pdf_inspector.get("disagreements", [])
    if isinstance(disagreements, list) and disagreements:
        taxonomy.append("extraction_engine_disagreement")
    if chunk_count <= 0 or (
        isinstance(text_extraction_coverage, (int, float))
        and float(text_extraction_coverage) < 0.8
    ):
        taxonomy.append("low_text_coverage")
    technical_processed = chunk_count > 0 and (
        native_block_count is not None
        or bool(drilldown.get("ocr_used"))
        or pages_processed_with_ocr > 0
    )
    structurally_reliable = technical_processed and not any(
        item in taxonomy
        for item in {"weak_sections", "layout_uncertain", "low_text_coverage"}
    )
    if not technical_processed:
        status = "fail"
        recommended_next_action = "inspect_document_or_try_ocr"
    elif not structurally_reliable:
        status = "review"
        recommended_next_action = "inspect_document_structure"
    elif any(
        item in taxonomy
        for item in {
            "ocr_required",
            "table_or_form_heavy",
            "encoding_uncertain",
            "extraction_engine_disagreement",
        }
    ):
        status = "warn"
        recommended_next_action = "review_processing_diagnostics"
    else:
        status = "pass"
        recommended_next_action = "none"
    return {
        "status": status,
        "taxonomy": taxonomy,
        "technical_processed": technical_processed,
        "structurally_reliable": structurally_reliable,
        "recommended_next_action": recommended_next_action,
        "summary": {
            "chunk_count": chunk_count,
            "section_count": section_count,
            "native_block_count": native_block_count,
            "ocr_used": bool(drilldown.get("ocr_used")),
            "pages_requiring_ocr": pages_requiring_ocr,
            "text_extraction_coverage": text_extraction_coverage,
            "structure_confidence": structure_confidence,
            "layout_confidence": layout_confidence,
            "table_or_form_signal_count": table_or_form_signal_count,
            "pdf_inspector": pdf_inspector,
        },
    }


def _claim_alignment_status(claim_alignment: dict[str, object]) -> dict[str, object]:
    unsupported_count = int(claim_alignment.get("unsupported_claim_count", 0) or 0)
    weak_count = int(claim_alignment.get("weak_claim_count", 0) or 0)
    claim_count = int(claim_alignment.get("claim_count", 0) or 0)
    supported_ratio = float(claim_alignment.get("supported_claim_ratio", 0.0) or 0.0)
    alignment_status = str(claim_alignment.get("alignment_status") or "")
    if not claim_alignment:
        status = "warn"
        reasons = ["claim_alignment_missing"]
    elif unsupported_count > 0:
        status = "review"
        reasons = ["unsupported_claims_present"]
    elif weak_count > 0:
        status = "review"
        reasons = ["weak_claims_present"]
    elif claim_count and supported_ratio >= 1.0:
        status = "pass"
        reasons = []
    else:
        status = "review"
        reasons = ["claim_support_incomplete"]
    return {
        "status": status,
        "reasons": reasons,
        "alignment_status": alignment_status,
        "claim_count": claim_count,
        "supported_claim_ratio": supported_ratio,
        "weak_claim_count": weak_count,
        "unsupported_claim_count": unsupported_count,
    }


def _quality_overall_status(statuses: dict[str, object]) -> str:
    if any(status == "fail" for status in statuses.values()):
        return "fail"
    if any(status == "review" for status in statuses.values()):
        return "review"
    if any(status == "warn" for status in statuses.values()):
        return "warn"
    if any(status == "skip" for status in statuses.values()):
        return "skip"
    if statuses and all(status == "pass" for status in statuses.values()):
        return "pass"
    return "unknown"


def _quality_recommended_next_action(
    statuses: dict[str, object], reasons: list[str]
) -> str:
    reason_set = set(reasons)
    if statuses.get("processing_quality") == "fail":
        if "text_or_ocr_path_known" in reason_set:
            return "inspect_document_or_try_ocr"
        return "inspect_document"
    if statuses.get("processing_quality") in {"review", "warn"}:
        return "inspect_processing_diagnostics"
    if statuses.get("semantic_confidence") == "fail":
        return "inspect_document_semantics"
    if statuses.get("retrieval_readiness") in {"warn", "fail"}:
        return "review_retrieval_contract"
    if statuses.get("answer_trust") == "review":
        return "review_claim_alignment"
    if statuses and all(status == "pass" for status in statuses.values()):
        return "none"
    return "review_quality_profile"


def _workflow_quality_profile(payload: dict[str, object]) -> dict[str, object]:
    document = (
        payload.get("document", {}) if isinstance(payload.get("document"), dict) else {}
    )
    index = payload.get("index", {}) if isinstance(payload.get("index"), dict) else {}
    answer = (
        payload.get("answer", {}) if isinstance(payload.get("answer"), dict) else {}
    )
    answer_trace = (
        answer.get("answer_trace", {})
        if isinstance(answer.get("answer_trace"), dict)
        else {}
    )
    contract_health = (
        answer.get("contract_health", {})
        if isinstance(answer.get("contract_health"), dict)
        else {}
    )
    retrieval_contract_status = answer.get("retrieval_contract_status", {})
    if not isinstance(retrieval_contract_status, dict):
        retrieval_contract_status = {}
    if not retrieval_contract_status and isinstance(
        contract_health.get("retrieval_contract_status"), dict
    ):
        retrieval_contract_status = contract_health["retrieval_contract_status"]
    support_coverage = answer.get("support_coverage", {})
    if not isinstance(support_coverage, dict):
        support_coverage = {}
    if not support_coverage and isinstance(
        contract_health.get("support_coverage"), dict
    ):
        support_coverage = contract_health["support_coverage"]
    answer_source_mix = answer.get("answer_source_mix", {})
    if not isinstance(answer_source_mix, dict):
        answer_source_mix = {}
    if not answer_source_mix and isinstance(
        contract_health.get("answer_source_mix"), dict
    ):
        answer_source_mix = contract_health["answer_source_mix"]
    structure_confidence = document.get("structure_confidence")
    layout_confidence = document.get("layout_confidence")
    semantic_confidence = document.get("semantic_confidence")
    processing_drilldown = _processing_drilldown(document, index)
    processing_diagnostics = _processing_diagnostics(document, index)
    processing_checks = [
        {"name": "chunks_created", "passed": bool(index.get("chunk_count", 0) > 0)},
        {
            "name": "structure_confidence_present",
            "passed": structure_confidence is not None,
        },
        {"name": "layout_confidence_present", "passed": layout_confidence is not None},
        {
            "name": "text_or_ocr_path_known",
            "passed": bool(
                processing_drilldown.get("native_block_count") is not None
                or processing_drilldown.get("ocr_used")
            ),
        },
        {
            "name": "sections_or_chunks_present",
            "passed": bool(
                processing_drilldown.get("section_count") or index.get("chunk_count", 0)
            ),
        },
    ]
    semantic_checks = [
        {
            "name": "semantic_confidence_present",
            "passed": semantic_confidence is not None,
        },
        {
            "name": "document_type_present",
            "passed": bool(document.get("document_type")),
        },
        {
            "name": "document_purpose_present",
            "passed": bool(document.get("document_purpose")),
        },
        {
            "name": "inventory_summary_present",
            "passed": bool(document.get("inventory_summary")),
        },
    ]
    retrieval_checks = [
        {"name": "contract_health_available", "passed": bool(contract_health)},
        {
            "name": "retrieval_contract_present",
            "passed": bool(answer_trace.get("retrieval_contract")),
        },
        {
            "name": "support_scope_present",
            "passed": bool(contract_health.get("support_scope")),
        },
        {
            "name": "support_available",
            "passed": bool(
                contract_health.get("support_trace_count")
                or contract_health.get("evidence_count")
            ),
        },
        {
            "name": "selected_or_support_docs_present",
            "passed": bool(
                contract_health.get("selected_doc_ids")
                or contract_health.get("support_doc_ids")
            ),
        },
        {
            "name": "retrieval_contract_status_pass",
            "passed": retrieval_contract_status.get("status") == "pass",
        },
    ]
    claim_status = _claim_alignment_status(
        answer_trace.get("claim_alignment", {})
        if isinstance(answer_trace.get("claim_alignment"), dict)
        else {}
    )
    answer_checks = [
        {"name": "answer_present", "passed": bool(answer.get("answer"))},
        {
            "name": "answer_contract_present",
            "passed": bool(answer_trace.get("answer_contract")),
        },
        {
            "name": "claim_alignment_present",
            "passed": bool(answer_trace.get("claim_alignment")),
        },
        {
            "name": "no_unsupported_claims",
            "passed": claim_status["unsupported_claim_count"] == 0,
        },
        {"name": "no_weak_claims", "passed": claim_status["weak_claim_count"] == 0},
    ]
    answer_trust_reasons = [
        *([] if answer.get("answer") else ["answer_missing"]),
        *([] if answer_trace.get("answer_contract") else ["answer_contract_missing"]),
        *([] if answer_trace.get("claim_alignment") else ["claim_alignment_missing"]),
        *claim_status["reasons"],
    ]
    semantic_classification = (
        "well_supported"
        if isinstance(semantic_confidence, (int, float)) and semantic_confidence >= 0.85
        else "provisional"
        if isinstance(semantic_confidence, (int, float)) and semantic_confidence >= 0.55
        else "unknown"
    )
    sections = {
        "processing_quality": {
            "status": "pass"
            if all(item["passed"] for item in processing_checks)
            and processing_diagnostics["status"] == "pass"
            else processing_diagnostics["status"],
            "thresholds": QUALITY_PROFILE_THRESHOLDS["processing_quality"],
            "checks": processing_checks,
            "reasons": sorted(
                set(
                    [
                        *_failed_check_reasons(processing_checks),
                        *processing_diagnostics["taxonomy"],
                    ]
                )
            ),
            "structure_confidence": structure_confidence,
            "layout_confidence": layout_confidence,
            "structure_status": _quality_status(structure_confidence),
            "layout_status": _quality_status(layout_confidence),
            "drilldown": processing_drilldown,
            "diagnostics": processing_diagnostics,
        },
        "semantic_confidence": {
            "status": "pass"
            if all(item["passed"] for item in semantic_checks)
            else "fail",
            "thresholds": QUALITY_PROFILE_THRESHOLDS["semantic_confidence"],
            "checks": semantic_checks,
            "score": semantic_confidence,
            "label": document.get("semantic_confidence_label"),
            "classification_status": semantic_classification,
            "trust_policy": (
                "stable_semantic_classification"
                if semantic_classification == "well_supported"
                else "confidence_aware_provisional_classification"
            ),
        },
        "retrieval_readiness": {
            "status": "pass"
            if all(item["passed"] for item in retrieval_checks)
            else "warn",
            "thresholds": QUALITY_PROFILE_THRESHOLDS["retrieval_readiness"],
            "checks": retrieval_checks,
            "reasons": sorted(
                set(
                    [
                        *_failed_check_reasons(retrieval_checks),
                        *retrieval_contract_status.get("reasons", []),
                    ]
                )
            ),
            "retrieval_path": contract_health.get("retrieval_path"),
            "support_scope": contract_health.get("support_scope"),
            "support_doc_ids": contract_health.get("support_doc_ids", []),
            "selected_doc_ids": contract_health.get("selected_doc_ids", []),
            "retrieval_contract_status": retrieval_contract_status,
            "support_coverage": support_coverage,
            "answer_source_mix": answer_source_mix,
        },
        "answer_trust": {
            "status": "pass"
            if all(item["passed"] for item in answer_checks)
            else claim_status["status"],
            "thresholds": QUALITY_PROFILE_THRESHOLDS["answer_trust"],
            "checks": answer_checks,
            "reasons": sorted(set(answer_trust_reasons)),
            "claim_alignment_status": claim_status,
            "contract_health": contract_health,
        },
    }
    statuses = {
        key: value.get("status")
        for key, value in sections.items()
        if isinstance(value, dict) and value.get("status")
    }
    reasons: list[str] = []
    for value in sections.values():
        if isinstance(value, dict) and isinstance(value.get("reasons"), list):
            reasons.extend(str(reason) for reason in value["reasons"])
    return {
        "overall_status": _quality_overall_status(statuses),
        "recommended_next_action": _quality_recommended_next_action(statuses, reasons),
        "statuses": statuses,
        "reasons": sorted(set(reasons)),
        **sections,
    }


def _quality_profile_summary(
    quality_profile: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(quality_profile, dict):
        return {
            "available": False,
            "overall_status": "unknown",
            "statuses": {},
            "reasons": ["quality_profile_missing"],
        }
    statuses = (
        dict(quality_profile.get("statuses", {}))
        if isinstance(quality_profile.get("statuses"), dict)
        else {
            key: value.get("status")
            for key, value in quality_profile.items()
            if isinstance(value, dict) and value.get("status")
        }
    )
    reasons: list[str] = []
    if isinstance(quality_profile.get("reasons"), list):
        reasons.extend(str(reason) for reason in quality_profile.get("reasons", []))
    else:
        for value in quality_profile.values():
            if isinstance(value, dict) and isinstance(value.get("reasons"), list):
                reasons.extend(str(reason) for reason in value.get("reasons", []))
    overall_status = str(
        quality_profile.get("overall_status") or _quality_overall_status(statuses)
    )
    return {
        "available": True,
        "overall_status": overall_status,
        "statuses": statuses,
        "reasons": sorted(set(reasons)),
        "recommended_next_action": quality_profile.get("recommended_next_action")
        or _quality_recommended_next_action(statuses, reasons),
    }


PUBLIC_COMPACT_WORKFLOW_KEYS = (
    "pdf",
    "doc_id",
    "document",
    "plan",
    "index",
    "answer",
    "processing_diagnostics",
    "quality_profile_summary",
)


PUBLIC_COMPACT_SMOKE_KEYS = (*PUBLIC_COMPACT_WORKFLOW_KEYS, "checks", "all_pass")


PUBLIC_ASSESS_PDF_KEYS = (
    "pdf",
    "doc_id",
    "overall_status",
    "processing_status",
    "semantic_status",
    "retrieval_status",
    "answer_trust",
    "recommended_next_action",
    "acceptance_profile",
    "structure_support",
    "messages",
)


PUBLIC_COMPACT_DOCUMENT_KEYS = (
    "doc_id",
    "label",
    "title",
    "document_family",
    "document_type",
    "document_purpose",
    "audience",
    "inventory_summary",
    "coverage_terms",
    "structure_confidence",
    "layout_confidence",
    "semantic_confidence",
    "semantic_confidence_label",
    "page_count",
    "section_count",
)


PUBLIC_COMPACT_INDEX_KEYS = ("doc_ids", "chunk_count", "embedding")


PUBLIC_COMPACT_ANSWER_KEYS = (
    "query",
    "query_intent",
    "answer",
    "answer_trace",
    "contract_health",
    "retrieval_contract_status",
    "support_coverage",
    "answer_source_mix",
)


def _compact_workflow_payload(payload: dict[str, object]) -> dict[str, object]:
    document = (
        payload.get("document", {}) if isinstance(payload.get("document"), dict) else {}
    )
    index = payload.get("index", {}) if isinstance(payload.get("index"), dict) else {}
    return {
        "pdf": payload.get("pdf"),
        "doc_id": payload.get("doc_id"),
        "document": {
            key: document.get(key)
            for key in PUBLIC_COMPACT_DOCUMENT_KEYS
            if key in document
        },
        "plan": payload.get("plan", {}),
        "index": {
            "doc_ids": index.get("doc_ids", []),
            "chunk_count": index.get("chunk_count"),
            "embedding": index.get("embedding", {}),
        },
        "answer": payload.get("answer", {}),
        "processing_diagnostics": payload.get("processing_diagnostics", {}),
        "quality_profile_summary": _quality_profile_summary(
            payload.get("quality_profile")
            if isinstance(payload.get("quality_profile"), dict)
            else None
        ),
    }


def _assessment_profile(
    document: dict[str, object], processing_diagnostics: dict[str, object]
) -> str:
    extraction_summary = (
        document.get("extraction_summary", {})
        if isinstance(document.get("extraction_summary"), dict)
        else {}
    )
    block_role_counts = extraction_summary.get("block_role_counts", {})
    layout_signal_counts = extraction_summary.get("layout_signal_counts", {})
    taxonomy = set(_string_list(processing_diagnostics.get("taxonomy")))
    page_count = document.get("page_count")
    document_type = str(document.get("document_type") or "")
    form_signal_count = 0
    table_signal_count = 0
    if isinstance(block_role_counts, dict):
        form_signal_count += int(block_role_counts.get("form_field", 0) or 0)
        form_signal_count += int(block_role_counts.get("key_value", 0) or 0)
        table_signal_count += int(block_role_counts.get("table_like", 0) or 0)
    if isinstance(layout_signal_counts, dict):
        form_signal_count += int(layout_signal_counts.get("form_like", 0) or 0)
        table_signal_count += int(layout_signal_counts.get("table_like", 0) or 0)
    if "ocr_required" in taxonomy or "native_text_low" in taxonomy:
        return "scanned_pdf"
    if document_type in {"financial_statement", "statistical_table"}:
        return "table_heavy_pdf"
    if form_signal_count >= max(3, table_signal_count):
        return "form_heavy_pdf"
    if table_signal_count >= 3:
        return "table_heavy_pdf"
    if isinstance(page_count, int) and page_count <= 2:
        return "short_document"
    if isinstance(page_count, int) and page_count >= 15:
        return "long_document"
    return "medium_document"


def _assessment_messages(
    *,
    processing_diagnostics: dict[str, object],
    quality_summary: dict[str, object],
    answer: dict[str, object],
    document: dict[str, object],
) -> list[str]:
    messages: list[str] = []
    statuses = (
        quality_summary.get("statuses", {})
        if isinstance(quality_summary.get("statuses"), dict)
        else {}
    )
    processing_status = str(statuses.get("processing_quality") or "unknown")
    semantic_status = str(statuses.get("semantic_confidence") or "unknown")
    answer_trust = str(statuses.get("answer_trust") or "unknown")
    taxonomy = set(_string_list(processing_diagnostics.get("taxonomy")))
    source_mix = (
        answer.get("answer_source_mix", {})
        if isinstance(answer.get("answer_source_mix"), dict)
        else {}
    )
    chunk_evidence = (
        source_mix.get("chunk_evidence", {})
        if isinstance(source_mix.get("chunk_evidence"), dict)
        else {}
    )
    document_semantics = (
        source_mix.get("document_semantics", {})
        if isinstance(source_mix.get("document_semantics"), dict)
        else {}
    )

    if processing_status == "pass":
        messages.append("processed_with_reliable_structure")
    elif processing_status == "warn":
        messages.append("processed_with_processing_caveats")
    elif processing_status == "review":
        messages.append("processed_but_structurally_weak")
    elif processing_status == "fail":
        messages.append("processing_failed_or_no_reliable_text")

    if "ocr_required" in taxonomy:
        messages.append("ocr_or_scan_path_used")
    if "table_or_form_heavy" in taxonomy:
        messages.append("table_or_form_heavy_layout")

    semantic_label = str(document.get("semantic_confidence_label") or "")
    if semantic_status == "pass" and semantic_label != "low":
        messages.append("semantic_classification_supported")
    else:
        messages.append("semantic_guess_review_recommended")

    if bool(document_semantics.get("present")) and not bool(
        chunk_evidence.get("present")
    ):
        messages.append("answer_supported_by_document_semantics_only")
    elif bool(chunk_evidence.get("present")):
        messages.append("answer_supported_by_chunk_evidence")

    if answer_trust == "review":
        messages.append("answer_claims_need_review")
    elif answer_trust == "pass":
        messages.append("answer_claims_supported")
    return list(dict.fromkeys(messages))


def _assess_pdf_payload(workflow_payload: dict[str, object]) -> dict[str, object]:
    quality_summary = _quality_profile_summary(
        workflow_payload.get("quality_profile")
        if isinstance(workflow_payload.get("quality_profile"), dict)
        else None
    )
    statuses = (
        quality_summary.get("statuses", {})
        if isinstance(quality_summary.get("statuses"), dict)
        else {}
    )
    document = (
        workflow_payload.get("document", {})
        if isinstance(workflow_payload.get("document"), dict)
        else {}
    )
    answer = (
        workflow_payload.get("answer", {})
        if isinstance(workflow_payload.get("answer"), dict)
        else {}
    )
    processing_diagnostics = (
        workflow_payload.get("processing_diagnostics", {})
        if isinstance(workflow_payload.get("processing_diagnostics"), dict)
        else {}
    )
    assessment = {
        "pdf": workflow_payload.get("pdf"),
        "doc_id": workflow_payload.get("doc_id"),
        "overall_status": quality_summary.get("overall_status"),
        "processing_status": statuses.get("processing_quality", "unknown"),
        "semantic_status": statuses.get("semantic_confidence", "unknown"),
        "retrieval_status": statuses.get("retrieval_readiness", "unknown"),
        "answer_trust": statuses.get("answer_trust", "unknown"),
        "recommended_next_action": quality_summary.get("recommended_next_action"),
        "acceptance_profile": _assessment_profile(document, processing_diagnostics),
        "structure_support": _structure_support_summary(
            document, processing_diagnostics
        ),
        "messages": _assessment_messages(
            processing_diagnostics=processing_diagnostics,
            quality_summary=quality_summary,
            answer=answer,
            document=document,
        ),
    }
    return assessment


def _structure_support_summary(
    document: dict[str, object],
    processing_diagnostics: dict[str, object],
) -> dict[str, object]:
    extraction_summary = (
        document.get("extraction_summary", {})
        if isinstance(document.get("extraction_summary"), dict)
        else {}
    )
    block_role_counts = extraction_summary.get("block_role_counts", {})
    layout_signal_counts = extraction_summary.get("layout_signal_counts", {})
    taxonomy = set(_string_list(processing_diagnostics.get("taxonomy")))
    document_type = str(document.get("document_type") or "")
    form_signal_count = 0
    table_signal_count = 0
    if document_type in {"financial_statement", "statistical_table"}:
        table_signal_count += 3
    if isinstance(block_role_counts, dict):
        form_signal_count += int(block_role_counts.get("form_field", 0) or 0)
        form_signal_count += int(block_role_counts.get("key_value", 0) or 0)
        table_signal_count += int(block_role_counts.get("table_like", 0) or 0)
    if isinstance(layout_signal_counts, dict):
        form_signal_count += int(layout_signal_counts.get("form_like", 0) or 0)
        table_signal_count += int(layout_signal_counts.get("table_like", 0) or 0)
    if "low_text_coverage" in taxonomy or "native_text_low" in taxonomy:
        status = "weak"
    elif "weak_sections" in taxonomy or "layout_uncertain" in taxonomy:
        status = "review"
    elif form_signal_count >= 3 or table_signal_count >= 3:
        status = "structured"
    else:
        status = "basic"
    return {
        "status": status,
        "form_signal_count": form_signal_count,
        "table_signal_count": table_signal_count,
        "table_or_form_signal_count": processing_diagnostics.get("drilldown", {}).get(
            "table_or_form_signal_count"
        )
        if isinstance(processing_diagnostics.get("drilldown"), dict)
        else None,
        "section_count": document.get("section_count"),
        "chunk_count": processing_diagnostics.get("drilldown", {}).get("chunk_count")
        if isinstance(processing_diagnostics.get("drilldown"), dict)
        else None,
        "taxonomy": sorted(taxonomy),
    }
