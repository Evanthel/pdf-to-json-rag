"""Runtime-mode benchmark execution and promotion gates."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any

from .answering import GroundedAnswer, answer_query_with_retrieval
from .indexing import build_local_index
from .schemas import ChunkRecord


from .evaluation_cases import (
    DEFAULT_REGRESSION_SHARDS,
    DEFAULT_RUNTIME_COMPARISON_CASE_IDS,
    DEFAULT_RUNTIME_COMPARISON_REPORT_FILENAME,
    RUNTIME_COMPARISON_MODES,
)

from .evaluation_metrics import (
    _keyword_matches,
    _ordered_unique,
    _preview_text,
    _summarize_retrieval_results,
    ensure_default_eval_cases,
    load_eval_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

from .evaluation_reports import (
    _regression_case_status,
)


def _load_all_chunk_records(chunk_root: Path) -> list[ChunkRecord]:
    chunk_root = chunk_root.expanduser().resolve()
    chunks: list[ChunkRecord] = []
    for chunk_path in sorted(chunk_root.glob("*/*.json")):
        data = json.loads(chunk_path.read_text(encoding="utf-8"))
        chunks.append(ChunkRecord.model_validate(data))
    return chunks


def _with_runtime_env(updates: dict[str, str | None]):
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_runtime_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _retrieval_result_from_grounded_answer(
    *,
    case: dict[str, Any],
    grounded_answer: GroundedAnswer,
    k: int,
) -> dict[str, Any]:
    retrieved_ids = [chunk.chunk_id for chunk in grounded_answer.top_k_hits]
    retrieved_doc_ids = _ordered_unique(
        [chunk.doc_id for chunk in grounded_answer.top_k_hits]
    )
    case_type = case.get("case_type", "grounded")
    if case_type == "negative":
        return {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "evaluation_level": "document" if case.get("relevant_doc_ids") else "chunk",
            "retrieved_ids": retrieved_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
        }

    relevant_doc_ids = set(case.get("relevant_doc_ids", []))
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

    relevant = set(case.get("relevant_chunk_ids", []))
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


def _answer_result_from_grounded_answer(
    *,
    case: dict[str, Any],
    grounded_answer: GroundedAnswer,
) -> dict[str, Any]:
    case_type = case.get("case_type", "grounded")
    keyword_eval = _keyword_matches(
        grounded_answer.answer,
        case.get("expected_keywords", []),
    )
    abstained = grounded_answer.answer.startswith("No grounded answer")
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
        "query": case["query"],
        "answer": grounded_answer.answer,
        "top_k_hit_ids": [chunk.chunk_id for chunk in grounded_answer.top_k_hits],
        "expanded_hit_ids": [chunk.chunk_id for chunk in grounded_answer.expanded_hits],
        "evidence_chunk_ids": [item.chunk_id for item in grounded_answer.evidence],
        "evidence_sentences": [item.sentence for item in grounded_answer.evidence],
        "abstained": abstained,
        "negative_success": abstained if case_type == "negative" else None,
        **keyword_eval,
    }


def _runtime_signal_summary(answers: list[GroundedAnswer]) -> dict[str, Any]:
    backend_counts: dict[str, int] = {}
    cross_encoder_fallback_count = 0
    llm_configured_count = 0
    llm_invoked_count = 0
    llm_used_count = 0
    for answer in answers:
        for chunk in answer.top_k_hits + answer.expanded_hits:
            backend_code = chunk.retrieval_signals.get("rerank_backend_code")
            backend = {
                0.0: "heuristic",
                1.0: "lightweight",
                2.0: "cross_encoder",
            }.get(backend_code, "unknown")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if chunk.retrieval_signals.get("cross_encoder_fallback"):
                cross_encoder_fallback_count += 1
        synthesis_runtime = answer.answer_trace.get("synthesis_runtime", {})
        if synthesis_runtime.get("configured"):
            llm_configured_count += 1
        if synthesis_runtime.get("invoked"):
            llm_invoked_count += 1
        if synthesis_runtime.get("used_for_final_answer"):
            llm_used_count += 1
    return {
        "rerank_backend_counts": backend_counts,
        "cross_encoder_fallback_chunk_count": cross_encoder_fallback_count,
        "llm_configured_case_count": llm_configured_count,
        "llm_invoked_case_count": llm_invoked_count,
        "llm_used_case_count": llm_used_count,
    }


def _evaluate_runtime_mode(
    *,
    mode: str,
    cases: list[dict[str, Any]],
    index_dir: Path,
    chunk_root: Path,
    k: int,
    index_manifest: dict[str, Any],
    env_updates: dict[str, str | None],
) -> dict[str, Any]:
    previous_env = _with_runtime_env(env_updates)
    try:
        mode_started = time.perf_counter()
        retrieval_results: list[dict[str, Any]] = []
        answer_results: list[dict[str, Any]] = []
        case_results: list[dict[str, Any]] = []
        grounded_answers: list[GroundedAnswer] = []
        failed_case_ids: list[str] = []
        query_latencies_ms: list[float] = []
        for case in cases:
            query_started = time.perf_counter()
            grounded_answer = answer_query_with_retrieval(
                query=case["query"],
                index_dir=index_dir,
                chunk_root=chunk_root,
                k=k,
                use_lightweight_rerank=True,
            )
            query_latency_ms = (time.perf_counter() - query_started) * 1000.0
            query_latencies_ms.append(query_latency_ms)
            grounded_answers.append(grounded_answer)
            retrieval_result = _retrieval_result_from_grounded_answer(
                case=case,
                grounded_answer=grounded_answer,
                k=k,
            )
            answer_result = _answer_result_from_grounded_answer(
                case=case,
                grounded_answer=grounded_answer,
            )
            status = _regression_case_status(
                case.get("case_type", "grounded"),
                retrieval_result,
                answer_result,
            )
            if status != "pass":
                failed_case_ids.append(case["case_id"])
            retrieval_results.append(retrieval_result)
            answer_results.append(answer_result)
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case.get("case_type", "grounded"),
                    "status": status,
                    "retrieval": retrieval_result,
                    "answer": {
                        "keyword_coverage": answer_result.get("keyword_coverage"),
                        "abstained": answer_result.get("abstained"),
                        "negative_success": answer_result.get("negative_success"),
                        "answer_preview": _preview_text(
                            str(answer_result.get("answer", "")), limit=240
                        ),
                    },
                    "runtime": {
                        "query_latency_ms": round(query_latency_ms, 3),
                        "synthesis_runtime": grounded_answer.answer_trace.get(
                            "synthesis_runtime", {}
                        ),
                        "claim_alignment": grounded_answer.answer_trace.get(
                            "claim_alignment", {}
                        ),
                    },
                }
            )

        summary = _summarize_retrieval_results(retrieval_results, answer_results)
        runtime_signals = _runtime_signal_summary(grounded_answers)
        sorted_latencies = sorted(query_latencies_ms)
        p95_index = max(
            0,
            min(len(sorted_latencies) - 1, math.ceil(len(sorted_latencies) * 0.95) - 1),
        )
        runtime_signals.update(
            {
                "latency_scope": "answer_query_with_retrieval; includes first-use model load in this process",
                "mode_wall_seconds": round(time.perf_counter() - mode_started, 3),
                "total_query_latency_ms": round(sum(query_latencies_ms), 3),
                "avg_query_latency_ms": round(statistics.fmean(query_latencies_ms), 3)
                if query_latencies_ms
                else None,
                "median_query_latency_ms": round(
                    statistics.median(query_latencies_ms), 3
                )
                if query_latencies_ms
                else None,
                "p95_query_latency_ms": round(sorted_latencies[p95_index], 3)
                if sorted_latencies
                else None,
            }
        )
        return {
            "mode": mode,
            "case_count": len(cases),
            "pass_count": len(cases) - len(failed_case_ids),
            "fail_count": len(failed_case_ids),
            "failed_case_ids": failed_case_ids,
            "all_pass": len(failed_case_ids) == 0,
            "summary": summary,
            "index_manifest": {
                "embedding_backend": index_manifest.get("embedding_backend"),
                "embedding_model": index_manifest.get("embedding_model"),
                "embedding_fallback_reason": index_manifest.get(
                    "embedding_fallback_reason"
                ),
                "chunk_count": index_manifest.get("chunk_count"),
                "benchmark_index_build_seconds": index_manifest.get(
                    "benchmark_index_build_seconds"
                ),
            },
            "runtime_signals": runtime_signals,
            "case_results": case_results,
        }
    finally:
        _restore_runtime_env(previous_env)


def _runtime_mode_deltas(
    baseline: dict[str, Any],
    mode_result: dict[str, Any],
) -> dict[str, float | int]:
    base_summary = baseline.get("summary", {})
    summary = mode_result.get("summary", {})
    return {
        "pass_count_delta": int(mode_result.get("pass_count", 0))
        - int(baseline.get("pass_count", 0)),
        "avg_precision_at_k_delta": round(
            float(summary.get("avg_precision_at_k") or 0.0)
            - float(base_summary.get("avg_precision_at_k") or 0.0),
            4,
        ),
        "avg_recall_at_k_delta": round(
            float(summary.get("avg_recall_at_k") or 0.0)
            - float(base_summary.get("avg_recall_at_k") or 0.0),
            4,
        ),
        "mrr_delta": round(
            float(summary.get("mrr") or 0.0) - float(base_summary.get("mrr") or 0.0),
            4,
        ),
        "avg_keyword_coverage_delta": round(
            float(summary.get("avg_keyword_coverage") or 0.0)
            - float(base_summary.get("avg_keyword_coverage") or 0.0),
            4,
        ),
    }


def _runtime_promotion_gate(
    *,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "candidate_mode": candidate.get("mode") if candidate else None,
            "promotable": False,
            "checks": [],
            "reasons": ["baseline or candidate result is missing"],
        }

    baseline_summary = baseline.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    candidate_manifest = candidate.get("index_manifest", {})
    checks = [
        {
            "name": "candidate_is_active",
            "passed": candidate_manifest.get("embedding_backend")
            == "sentence-transformers",
            "details": {
                "embedding_backend": candidate_manifest.get("embedding_backend"),
                "embedding_model": candidate_manifest.get("embedding_model"),
                "embedding_fallback_reason": candidate_manifest.get(
                    "embedding_fallback_reason"
                ),
            },
        },
        {
            "name": "no_pass_count_regression",
            "passed": int(candidate.get("pass_count", 0))
            >= int(baseline.get("pass_count", 0)),
            "details": {
                "baseline_pass_count": baseline.get("pass_count", 0),
                "candidate_pass_count": candidate.get("pass_count", 0),
            },
        },
        {
            "name": "recall_not_lower",
            "passed": float(candidate_summary.get("avg_recall_at_k") or 0.0)
            >= float(baseline_summary.get("avg_recall_at_k") or 0.0),
            "details": {
                "baseline_avg_recall_at_k": baseline_summary.get("avg_recall_at_k"),
                "candidate_avg_recall_at_k": candidate_summary.get("avg_recall_at_k"),
            },
        },
        {
            "name": "mrr_not_lower",
            "passed": float(candidate_summary.get("mrr") or 0.0)
            >= float(baseline_summary.get("mrr") or 0.0),
            "details": {
                "baseline_mrr": baseline_summary.get("mrr"),
                "candidate_mrr": candidate_summary.get("mrr"),
            },
        },
        {
            "name": "warnings_not_higher",
            "passed": int(candidate_summary.get("warning_case_count") or 0)
            <= int(baseline_summary.get("warning_case_count") or 0),
            "details": {
                "baseline_warning_case_count": baseline_summary.get(
                    "warning_case_count"
                ),
                "candidate_warning_case_count": candidate_summary.get(
                    "warning_case_count"
                ),
            },
        },
    ]
    failed = [item for item in checks if not item["passed"]]
    return {
        "candidate_mode": candidate.get("mode"),
        "promotable": not failed,
        "checks": checks,
        "reasons": [item["name"] for item in failed],
    }


def run_runtime_mode_comparison(
    *,
    index_dir: Path,
    chunk_root: Path,
    eval_dir: Path,
    k: int = 5,
    eval_path: Path | None = None,
    case_ids: list[str] | None = None,
    shard: str | None = None,
    modes: list[str] | None = None,
    all_cases: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Compare default retrieval against optional local model/runtime modes."""
    eval_dir = eval_dir.expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_path is None:
        eval_path = ensure_default_eval_cases(eval_dir)
    else:
        eval_path = eval_path.expanduser().resolve()

    eval_cases = load_eval_cases(eval_path)
    case_map = {item["case_id"]: item for item in eval_cases}
    if all_cases:
        selected_case_ids = [item["case_id"] for item in eval_cases]
    else:
        selected_case_ids = case_ids or DEFAULT_REGRESSION_SHARDS.get(
            shard or "", DEFAULT_RUNTIME_COMPARISON_CASE_IDS
        )
    selected_modes = modes or list(RUNTIME_COMPARISON_MODES)
    unknown_modes = [
        mode for mode in selected_modes if mode not in RUNTIME_COMPARISON_MODES
    ]
    missing_case_ids = [
        case_id for case_id in selected_case_ids if case_id not in case_map
    ]
    selected_cases = [
        case_map[case_id] for case_id in selected_case_ids if case_id in case_map
    ]

    chunk_root = chunk_root.expanduser().resolve()
    with tempfile.TemporaryDirectory(
        prefix="pdf-to-json-rag-runtime-compare-"
    ) as workspace:
        workspace_path = Path(workspace)
        chunks = _load_all_chunk_records(chunk_root)
        baseline_index_dir = workspace_path / "hash_baseline_index"
        baseline_build_started = time.perf_counter()
        previous_env = _with_runtime_env({"PDF_TO_JSON_RAG_EMBEDDING_BACKEND": "hash"})
        try:
            baseline_manifest = build_local_index(
                chunks=chunks,
                index_dir=baseline_index_dir,
            )
        finally:
            _restore_runtime_env(previous_env)
        baseline_manifest["benchmark_index_build_seconds"] = round(
            time.perf_counter() - baseline_build_started,
            3,
        )
        mode_index_dirs: dict[str, Path] = {
            "baseline": baseline_index_dir,
            "cross-encoder": baseline_index_dir,
            "llm-synthesis": baseline_index_dir,
        }
        mode_manifests: dict[str, dict[str, Any]] = {}
        mode_manifests["baseline"] = baseline_manifest
        mode_manifests["cross-encoder"] = mode_manifests["baseline"]
        mode_manifests["llm-synthesis"] = mode_manifests["baseline"]

        if "sentence-transformers" in selected_modes:
            sentence_index_dir = workspace_path / "sentence_transformers_index"
            sentence_build_started = time.perf_counter()
            previous_env = _with_runtime_env(
                {"PDF_TO_JSON_RAG_EMBEDDING_BACKEND": "sentence-transformers"}
            )
            try:
                mode_manifests["sentence-transformers"] = build_local_index(
                    chunks=chunks,
                    index_dir=sentence_index_dir,
                )
            finally:
                _restore_runtime_env(previous_env)
            mode_manifests["sentence-transformers"]["benchmark_index_build_seconds"] = (
                round(
                    time.perf_counter() - sentence_build_started,
                    3,
                )
            )
            mode_index_dirs["sentence-transformers"] = sentence_index_dir

        mode_envs = {
            "baseline": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
                "PDF_TO_JSON_RAG_LLM_COMMAND": None,
            },
            "sentence-transformers": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
                "PDF_TO_JSON_RAG_LLM_COMMAND": None,
            },
            "cross-encoder": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": "1",
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
                "PDF_TO_JSON_RAG_LLM_COMMAND": None,
            },
            "llm-synthesis": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
            },
        }

        mode_results: list[dict[str, Any]] = []
        for mode in selected_modes:
            if mode in unknown_modes:
                continue
            if mode == "sentence-transformers" and mode not in mode_index_dirs:
                continue
            mode_results.append(
                _evaluate_runtime_mode(
                    mode=mode,
                    cases=selected_cases,
                    index_dir=mode_index_dirs[mode],
                    chunk_root=chunk_root,
                    k=k,
                    index_manifest=mode_manifests[mode],
                    env_updates=mode_envs[mode],
                )
            )

    baseline_result = next(
        (item for item in mode_results if item["mode"] == "baseline"), None
    )
    sentence_transformers_result = next(
        (item for item in mode_results if item["mode"] == "sentence-transformers"),
        None,
    )
    deltas = {
        item["mode"]: _runtime_mode_deltas(baseline_result, item)
        for item in mode_results
        if baseline_result is not None and item["mode"] != "baseline"
    }
    promotion_gates = {}
    if sentence_transformers_result is not None:
        promotion_gates["sentence-transformers"] = _runtime_promotion_gate(
            baseline=baseline_result,
            candidate=sentence_transformers_result,
        )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "selected_shard": shard,
        "all_cases": all_cases,
        "selected_case_ids": selected_case_ids,
        "missing_case_ids": missing_case_ids,
        "unknown_modes": unknown_modes,
        "available_modes": list(RUNTIME_COMPARISON_MODES),
        "case_count": len(selected_cases),
        "comparison_index_policy": "fresh hash and sentence-transformer indexes built from the same chunks",
        "comparison_chunk_count": len(chunks),
        "mode_results": mode_results,
        "baseline_deltas": deltas,
        "promotion_gates": promotion_gates,
        "all_pass": (
            not missing_case_ids
            and not unknown_modes
            and all(item.get("all_pass") for item in mode_results)
        ),
    }

    report_path = eval_dir / DEFAULT_RUNTIME_COMPARISON_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report, report_path
