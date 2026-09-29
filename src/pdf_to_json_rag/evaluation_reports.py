"""MVP and regression evaluation runners, reports, and release gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answering import GroundedAnswer, answer_query_with_retrieval
from .retrieval import retrieve_top_k, retrieve_top_k_with_neighbors


from .evaluation_cases import (
    DEFAULT_REGRESSION_CASE_IDS,
    DEFAULT_REGRESSION_REPORT_FILENAME,
    DEFAULT_REGRESSION_SHARDS,
    DEFAULT_REPORT_FILENAME,
    LAYER_STABILITY_THRESHOLDS,
    SLICE_STABILITY_THRESHOLDS,
)

from .evaluation_metrics import (
    _answer_faithfulness_layer_record,
    _average,
    _debug_case_record,
    _faithfulness_audit_record,
    _keyword_matches,
    _layer_summary,
    _ordered_unique,
    _processing_layer_record,
    _retrieval_layer_record,
    _run_faithfulness_audit,
    _summarize_retrieval_results,
    ensure_default_eval_cases,
    ensure_default_faithfulness_audit,
    load_eval_cases,
    load_faithfulness_audit_case_ids,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def evaluate_retrieval_case(
    case: dict,
    index_dir: Path,
    chunk_root: Path | None,
    k: int,
    use_lightweight_rerank: bool = True,
) -> dict:
    """Evaluate retrieval metrics for a single query."""
    if chunk_root is not None:
        hits, _ = retrieve_top_k_with_neighbors(
            query=case["query"],
            index_dir=index_dir,
            chunk_root=chunk_root,
            k=k,
            use_lightweight_rerank=use_lightweight_rerank,
        )
    else:
        hits = retrieve_top_k(
            query=case["query"],
            index_dir=index_dir,
            k=k,
            use_lightweight_rerank=use_lightweight_rerank,
        )
    retrieved_ids = [chunk.chunk_id for chunk in hits]
    retrieved_doc_ids = _ordered_unique([chunk.doc_id for chunk in hits])
    relevant = set(case.get("relevant_chunk_ids", []))
    relevant_doc_ids = set(case.get("relevant_doc_ids", []))
    case_type = case.get("case_type", "grounded")
    evaluation_level = "document" if relevant_doc_ids else "chunk"
    if case_type == "negative":
        return {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "evaluation_level": evaluation_level,
            "retrieved_ids": retrieved_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
        }
    if relevant_doc_ids:
        return {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "evaluation_level": "document",
            "retrieved_ids": retrieved_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "precision_at_k": precision_at_k(retrieved_doc_ids, relevant_doc_ids, k),
            "recall_at_k": recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
            "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
        }
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
        "query": case["query"],
        "evaluation_level": "chunk",
        "retrieved_ids": retrieved_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
    }


def evaluate_answer_case(case: dict, index_dir: Path, chunk_root: Path, k: int) -> dict:
    """Evaluate the grounded answer path for a single query."""
    result: GroundedAnswer = answer_query_with_retrieval(
        query=case["query"],
        index_dir=index_dir,
        chunk_root=chunk_root,
        k=k,
    )
    keyword_eval = _keyword_matches(result.answer, case.get("expected_keywords", []))
    case_type = case.get("case_type", "grounded")
    abstained = result.answer.startswith("No grounded answer")
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
        "query": case["query"],
        "answer": result.answer,
        "top_k_hit_ids": [chunk.chunk_id for chunk in result.top_k_hits],
        "expanded_hit_ids": [chunk.chunk_id for chunk in result.expanded_hits],
        "evidence_chunk_ids": [item.chunk_id for item in result.evidence],
        "evidence_sentences": [item.sentence for item in result.evidence],
        "answer_trace": result.answer_trace,
        "abstained": abstained,
        "negative_success": abstained if case_type == "negative" else None,
        **keyword_eval,
    }


def _slice_summary(label: str, debug_cases: list[dict]) -> dict[str, Any]:
    slice_cases = [
        item for item in debug_cases if label in item.get("slice_labels", [])
    ]
    grounded_cases = [
        item for item in slice_cases if item.get("case_type") != "negative"
    ]
    negative_cases = [
        item for item in slice_cases if item.get("case_type") == "negative"
    ]
    warning_case_ids = [
        item["case_id"] for item in slice_cases if item.get("status") != "pass"
    ]

    return {
        "case_count": len(slice_cases),
        "grounded_case_count": len(grounded_cases),
        "negative_case_count": len(negative_cases),
        "avg_precision_at_k": _average(
            [item["retrieval"]["precision_at_k"] for item in grounded_cases]
        ),
        "avg_recall_at_k": _average(
            [item["retrieval"]["recall_at_k"] for item in grounded_cases]
        ),
        "mrr": _average(
            [item["retrieval"]["reciprocal_rank"] for item in grounded_cases]
        ),
        "avg_keyword_coverage": _average(
            [item["answer"]["keyword_coverage"] for item in grounded_cases]
        ),
        "negative_success_rate": _average(
            [
                1.0 if item["answer"].get("negative_success") else 0.0
                for item in negative_cases
            ]
        ),
        "warning_case_count": len(warning_case_ids),
        "warning_case_ids": warning_case_ids,
    }


def _evaluate_slice_stability(
    slices: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failed_labels: list[str] = []

    for label, thresholds in SLICE_STABILITY_THRESHOLDS.items():
        slice_summary = slices.get(label)
        if slice_summary is None:
            checks[label] = {
                "present": False,
                "pass": False,
                "reason": "slice missing from current benchmark",
                "thresholds": thresholds,
            }
            failed_labels.append(label)
            continue

        failed_metrics: dict[str, dict[str, float]] = {}
        for metric_name, min_value in thresholds.items():
            if (
                metric_name == "negative_success_rate"
                and int(slice_summary.get("negative_case_count", 0)) == 0
            ):
                continue
            actual_value = float(slice_summary.get(metric_name, 0.0))
            if actual_value < min_value:
                failed_metrics[metric_name] = {
                    "actual": actual_value,
                    "required_min": min_value,
                }

        passed = not failed_metrics
        checks[label] = {
            "present": True,
            "pass": passed,
            "thresholds": thresholds,
            "failed_metrics": failed_metrics,
        }
        if not passed:
            failed_labels.append(label)

    return {
        "all_pass": not failed_labels,
        "failed_labels": failed_labels,
        "checks": checks,
    }


def _evaluate_layer_stability(layer_summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failed_layers: list[str] = []

    for layer_name, thresholds in LAYER_STABILITY_THRESHOLDS.items():
        summary = layer_summary.get(layer_name)
        if summary is None:
            checks[layer_name] = {
                "present": False,
                "pass": False,
                "reason": "layer missing from current evaluation report",
                "thresholds": thresholds,
            }
            failed_layers.append(layer_name)
            continue

        failed_metrics: dict[str, dict[str, float]] = {}
        for metric_name, min_value in thresholds.items():
            actual_value = float(summary.get(metric_name, 0.0) or 0.0)
            if actual_value < min_value:
                failed_metrics[metric_name] = {
                    "actual": actual_value,
                    "required_min": min_value,
                }

        passed = not failed_metrics
        checks[layer_name] = {
            "present": True,
            "pass": passed,
            "thresholds": thresholds,
            "failed_metrics": failed_metrics,
        }
        if not passed:
            failed_layers.append(layer_name)

    return {
        "all_pass": not failed_layers,
        "failed_layers": failed_layers,
        "checks": checks,
    }


def _architecture_gates(
    *,
    summary: dict[str, Any],
    layer_stability: dict[str, Any],
    slice_stability: dict[str, Any],
    faithfulness_audit: dict[str, Any],
    is_default_eval_suite: bool,
) -> dict[str, Any]:
    faithfulness_pass = not bool(faithfulness_audit.get("recommend_llm_judge"))
    warning_free = int(summary.get("warning_case_count", 0)) == 0
    layer_pass = bool(layer_stability.get("all_pass"))
    slice_pass = (
        bool(slice_stability.get("all_pass")) if is_default_eval_suite else None
    )

    reasons: list[str] = []
    if not warning_free:
        reasons.append("benchmark warnings present")
    if not layer_pass:
        reasons.append("layer stability thresholds not met")
    if is_default_eval_suite and not bool(slice_stability.get("all_pass")):
        reasons.append("slice stability thresholds not met")
    if not faithfulness_pass:
        reasons.append("faithfulness audit recommends deeper review")

    all_pass = warning_free and layer_pass and faithfulness_pass
    if is_default_eval_suite:
        all_pass = all_pass and bool(slice_stability.get("all_pass"))

    return {
        "all_pass": all_pass,
        "warning_free": warning_free,
        "layer_stability_pass": layer_pass,
        "slice_stability_pass": slice_pass,
        "faithfulness_pass": faithfulness_pass,
        "is_default_eval_suite": is_default_eval_suite,
        "reasons": reasons,
    }


def _deferred_feature_decisions(
    summary: dict[str, Any],
    slices: dict[str, Any],
    faithfulness_audit: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    table_adjacent_warnings = slices.get("table_adjacent", {}).get(
        "warning_case_count", 0
    )
    table_heavy_warnings = slices.get("table_heavy", {}).get("warning_case_count", 0)
    source_review_warnings = slices.get("source_anchored_review", {}).get(
        "warning_case_count", 0
    )
    source_technical_warnings = slices.get("source_anchored_technical", {}).get(
        "warning_case_count", 0
    )
    return {
        "pdfplumber_probe": {
            "recommended": bool(table_heavy_warnings and table_adjacent_warnings),
            "reason": (
                "keep deferred: the current table-heavy benchmark is hitting extracted table content, "
                "and remaining issues are source-locking or answer selection rather than table extraction misses"
                if not (table_heavy_warnings and table_adjacent_warnings)
                else "table-heavy cases still show table-adjacent warnings after source-locking and chunk-quality filtering"
            ),
        },
        "cross_encoder_reranking": {
            "recommended": summary.get("warning_case_count", 0) > 0
            and (source_review_warnings > 0 or source_technical_warnings > 0)
            and baseline_summary.get("mrr", 0.0) == summary.get("mrr", 0.0),
            "reason": (
                "keep deferred: lightweight rerank plus source-aware heuristics are sufficient on the current benchmark"
                if not (
                    summary.get("warning_case_count", 0) > 0
                    and (source_review_warnings > 0 or source_technical_warnings > 0)
                    and baseline_summary.get("mrr", 0.0) == summary.get("mrr", 0.0)
                )
                else "remaining source-anchored warnings survive the current lightweight rerank without MRR improvement"
            ),
        },
        "llm_as_judge": {
            "recommended": bool(faithfulness_audit.get("recommend_llm_judge")),
            "reason": (
                "keep deferred: sampled extractive faithfulness audit does not show unsupported-answer drift"
                if not faithfulness_audit.get("recommend_llm_judge")
                else "sampled faithfulness audit found unsupported answer sentences"
            ),
        },
    }


def run_mvp_evaluation(
    index_dir: Path,
    chunk_root: Path,
    eval_dir: Path,
    k: int = 5,
    eval_path: Path | None = None,
) -> tuple[dict, Path]:
    """Run the small local MVP evaluation workflow and save a report."""
    eval_dir = eval_dir.expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    default_eval_path = ensure_default_eval_cases(eval_dir)
    if eval_path is None:
        eval_path = default_eval_path
    else:
        eval_path = eval_path.expanduser().resolve()
    is_default_eval_suite = eval_path == default_eval_path

    cases = load_eval_cases(eval_path)
    audit_path = ensure_default_faithfulness_audit(eval_dir)
    audit_case_ids = load_faithfulness_audit_case_ids(audit_path)
    retrieval_results = []
    answer_results = []
    debug_cases = []
    baseline_retrieval_results = []

    for case in cases:
        grounded_answer = answer_query_with_retrieval(
            query=case["query"],
            index_dir=index_dir,
            chunk_root=chunk_root,
            k=k,
            use_lightweight_rerank=True,
        )
        retrieved_ids = [chunk.chunk_id for chunk in grounded_answer.top_k_hits]
        relevant = set(case.get("relevant_chunk_ids", []))
        case_type = case.get("case_type", "grounded")
        baseline_retrieval_results.append(
            evaluate_retrieval_case(
                case=case,
                index_dir=index_dir,
                chunk_root=chunk_root,
                k=k,
                use_lightweight_rerank=False,
            )
        )
        if case_type == "negative":
            retrieval_result = {
                "case_id": case["case_id"],
                "case_type": case_type,
                "query": case["query"],
                "evaluation_level": "document"
                if case.get("relevant_doc_ids")
                else "chunk",
                "retrieved_ids": retrieved_ids,
                "retrieved_doc_ids": _ordered_unique(
                    [chunk.doc_id for chunk in grounded_answer.top_k_hits]
                ),
                "precision_at_k": None,
                "recall_at_k": None,
                "reciprocal_rank": None,
            }
        else:
            relevant_doc_ids = set(case.get("relevant_doc_ids", []))
            retrieved_doc_ids = _ordered_unique(
                [chunk.doc_id for chunk in grounded_answer.top_k_hits]
            )
            if relevant_doc_ids:
                retrieval_result = {
                    "case_id": case["case_id"],
                    "case_type": case_type,
                    "query": case["query"],
                    "evaluation_level": "document",
                    "retrieved_ids": retrieved_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "precision_at_k": precision_at_k(
                        retrieved_doc_ids, relevant_doc_ids, k
                    ),
                    "recall_at_k": recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
                    "reciprocal_rank": reciprocal_rank(
                        retrieved_doc_ids, relevant_doc_ids
                    ),
                }
            else:
                retrieval_result = {
                    "case_id": case["case_id"],
                    "case_type": case_type,
                    "query": case["query"],
                    "evaluation_level": "chunk",
                    "retrieved_ids": retrieved_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
                    "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
                    "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
                }

        keyword_eval = _keyword_matches(
            grounded_answer.answer,
            case.get("expected_keywords", []),
        )
        abstained = grounded_answer.answer.startswith("No grounded answer")
        answer_result = {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "answer": grounded_answer.answer,
            "top_k_hit_ids": [chunk.chunk_id for chunk in grounded_answer.top_k_hits],
            "expanded_hit_ids": [
                chunk.chunk_id for chunk in grounded_answer.expanded_hits
            ],
            "evidence_chunk_ids": [item.chunk_id for item in grounded_answer.evidence],
            "evidence_sentences": [item.sentence for item in grounded_answer.evidence],
            "abstained": abstained,
            "negative_success": abstained if case_type == "negative" else None,
            **keyword_eval,
        }

        retrieval_results.append(retrieval_result)
        answer_results.append(answer_result)
        debug_cases.append(
            _debug_case_record(
                case=case,
                retrieval_result=retrieval_result,
                answer_result=answer_result,
                grounded_answer=grounded_answer,
            )
        )

    all_slice_labels = sorted(
        {label for item in debug_cases for label in item.get("slice_labels", [])}
    )
    slices = {label: _slice_summary(label, debug_cases) for label in all_slice_labels}
    slice_stability = _evaluate_slice_stability(slices)
    faithfulness_audit = _run_faithfulness_audit(debug_cases, audit_case_ids)
    full_faithfulness = {
        item["case_id"]: _faithfulness_audit_record(item)
        for item in debug_cases
        if item.get("case_type") != "negative"
    }
    for debug_case in debug_cases:
        case_id = debug_case["case_id"]
        processing_chunks = (
            debug_case["retrieval"]["top_k_snapshots"]
            + debug_case["retrieval"]["expanded_snapshots"]
        )
        debug_case["layers"] = {
            "processing": _processing_layer_record(processing_chunks),
            "retrieval": _retrieval_layer_record(
                next(item for item in retrieval_results if item["case_id"] == case_id),
                next(item for item in answer_results if item["case_id"] == case_id),
            ),
            "answer_faithfulness": _answer_faithfulness_layer_record(
                debug_case,
                full_faithfulness.get(case_id),
            ),
        }
    layer_summary = _layer_summary(debug_cases)
    layer_stability = _evaluate_layer_stability(layer_summary)
    summary = _summarize_retrieval_results(retrieval_results, answer_results)
    baseline_summary = _summarize_retrieval_results(
        baseline_retrieval_results, answer_results
    )
    architecture_gates = _architecture_gates(
        summary=summary,
        layer_stability=layer_stability,
        slice_stability=slice_stability,
        faithfulness_audit=faithfulness_audit,
        is_default_eval_suite=is_default_eval_suite,
    )
    deferred_feature_decisions = _deferred_feature_decisions(
        summary=summary,
        slices=slices,
        faithfulness_audit=faithfulness_audit,
        baseline_summary=baseline_summary,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "faithfulness_audit_file": str(audit_path),
        "case_count": len(cases),
        "summary": summary,
        "layer_summary": layer_summary,
        "layer_stability": layer_stability,
        "slices": slices,
        "slice_stability": slice_stability,
        "architecture_gates": architecture_gates,
        "retrieval_strategy_comparison": {
            "baseline_chunking_only": {
                "avg_precision_at_k": baseline_summary["avg_precision_at_k"],
                "avg_recall_at_k": baseline_summary["avg_recall_at_k"],
                "mrr": baseline_summary["mrr"],
            },
            "lightweight_rerank": {
                "avg_precision_at_k": summary["avg_precision_at_k"],
                "avg_recall_at_k": summary["avg_recall_at_k"],
                "mrr": summary["mrr"],
            },
        },
        "faithfulness_audit": faithfulness_audit,
        "deferred_feature_decisions": deferred_feature_decisions,
        "retrieval_results": retrieval_results,
        "baseline_retrieval_results": baseline_retrieval_results,
        "answer_results": answer_results,
        "debug_cases": debug_cases,
    }

    report_path = eval_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report, report_path


def _regression_case_status(
    case_type: str,
    retrieval_result: dict[str, Any],
    answer_result: dict[str, Any],
) -> str:
    if case_type == "negative":
        return "pass" if answer_result.get("negative_success") else "negative_fail"
    if float(retrieval_result.get("reciprocal_rank") or 0.0) <= 0.0:
        return "retrieval_fail"
    if float(answer_result.get("keyword_coverage") or 0.0) < 0.9:
        return "answer_fail"
    return "pass"


def run_regression_suite(
    index_dir: Path,
    chunk_root: Path,
    eval_dir: Path,
    k: int = 5,
    eval_path: Path | None = None,
    case_ids: list[str] | None = None,
    shard: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run a deterministic high-risk regression subset before full benchmark reruns."""
    eval_dir = eval_dir.expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_path is None:
        eval_path = ensure_default_eval_cases(eval_dir)
    else:
        eval_path = eval_path.expanduser().resolve()

    all_cases = load_eval_cases(eval_path)
    case_map = {item["case_id"]: item for item in all_cases}
    selected_case_ids = case_ids or DEFAULT_REGRESSION_SHARDS.get(
        shard or "", DEFAULT_REGRESSION_CASE_IDS
    )

    missing_case_ids = [
        case_id for case_id in selected_case_ids if case_id not in case_map
    ]
    selected_cases = [
        case_map[case_id] for case_id in selected_case_ids if case_id in case_map
    ]

    case_results: list[dict[str, Any]] = []
    failed_case_ids: list[str] = []
    for case in selected_cases:
        retrieval_result = evaluate_retrieval_case(
            case=case, index_dir=index_dir, chunk_root=chunk_root, k=k
        )
        answer_result = evaluate_answer_case(
            case=case, index_dir=index_dir, chunk_root=chunk_root, k=k
        )
        status = _regression_case_status(
            case.get("case_type", "grounded"), retrieval_result, answer_result
        )
        if status != "pass":
            failed_case_ids.append(case["case_id"])
        case_results.append(
            {
                "case_id": case["case_id"],
                "case_type": case.get("case_type", "grounded"),
                "status": status,
                "retrieval": retrieval_result,
                "answer": answer_result,
            }
        )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "selected_shard": shard,
        "selected_case_ids": selected_case_ids,
        "missing_case_ids": missing_case_ids,
        "case_count": len(selected_cases),
        "pass_count": len(selected_cases) - len(failed_case_ids),
        "fail_count": len(failed_case_ids),
        "failed_case_ids": failed_case_ids,
        "all_pass": len(failed_case_ids) == 0 and not missing_case_ids,
        "case_results": case_results,
    }

    report_path = eval_dir / DEFAULT_REGRESSION_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report, report_path
