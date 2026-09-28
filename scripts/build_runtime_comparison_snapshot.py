#!/usr/bin/env python3
"""Build a compact, reviewable snapshot from a full runtime comparison report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any


MODE_REQUIREMENTS = {
    "baseline": {
        "pipeline": "hash-384 retrieval + lightweight reranking",
        "cpu": "required",
        "gpu": "not required",
        "local_model": "none",
        "fit": "lowest-dependency offline fallback",
    },
    "sentence-transformers": {
        "pipeline": "all-MiniLM-L6-v2 retrieval + lightweight reranking",
        "cpu": "supported",
        "gpu": "optional",
        "local_model": "embedding model required",
        "fit": "preferred local semantic retrieval when the model is cached",
    },
    "cross-encoder": {
        "pipeline": "hash-384 retrieval + ms-marco-MiniLM-L-6-v2 reranking",
        "cpu": "supported",
        "gpu": "optional; useful at higher query volume",
        "local_model": "reranker model required",
        "fit": "opt-in reranking experiment",
    },
}


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _weight_record(path: Path | None, root: Path) -> dict[str, Any]:
    if path is None:
        return {
            "primary_weight_path": None,
            "primary_weight_bytes": 0,
            "primary_weight_mib": 0.0,
        }
    if not path.is_file():
        raise FileNotFoundError(f"Model weight file not found: {path}")
    size = path.stat().st_size
    return {
        "primary_weight_path": _relative_path(path, root),
        "primary_weight_bytes": size,
        "primary_weight_mib": round(size / (1024 * 1024), 1),
    }


def build_snapshot(
    *,
    report_path: Path,
    output_path: Path,
    sentence_weight: Path,
    cross_encoder_weight: Path,
) -> dict[str, Any]:
    root = Path.cwd().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mode_results = {item["mode"]: item for item in report.get("mode_results", [])}
    required_modes = tuple(MODE_REQUIREMENTS)
    missing = [mode for mode in required_modes if mode not in mode_results]
    if missing:
        raise ValueError(f"Runtime report is missing required modes: {', '.join(missing)}")

    weights = {
        "baseline": _weight_record(None, root),
        "sentence-transformers": _weight_record(sentence_weight, root),
        "cross-encoder": _weight_record(cross_encoder_weight, root),
    }
    modes: list[dict[str, Any]] = []
    for mode in required_modes:
        result = mode_results[mode]
        summary = result.get("summary", {})
        runtime = result.get("runtime_signals", {})
        modes.append(
            {
                "mode": mode,
                **MODE_REQUIREMENTS[mode],
                "active": (
                    result.get("index_manifest", {}).get("embedding_backend") == "sentence-transformers"
                    if mode == "sentence-transformers"
                    else runtime.get("rerank_backend_counts", {}).get("cross_encoder", 0) > 0
                    if mode == "cross-encoder"
                    else True
                ),
                "quality": {
                    "pass_count": result.get("pass_count"),
                    "case_count": result.get("case_count"),
                    "avg_precision_at_k": summary.get("avg_precision_at_k"),
                    "avg_recall_at_k": summary.get("avg_recall_at_k"),
                    "mrr": summary.get("mrr"),
                    "avg_keyword_coverage": summary.get("avg_keyword_coverage"),
                    "warning_case_count": summary.get("warning_case_count"),
                },
                "latency": {
                    "avg_query_latency_ms": runtime.get("avg_query_latency_ms"),
                    "median_query_latency_ms": runtime.get("median_query_latency_ms"),
                    "p95_query_latency_ms": runtime.get("p95_query_latency_ms"),
                    "mode_wall_seconds": runtime.get("mode_wall_seconds"),
                    "scope": runtime.get("latency_scope"),
                },
                "runtime": {
                    "embedding_backend": result.get("index_manifest", {}).get("embedding_backend"),
                    "embedding_model": result.get("index_manifest", {}).get("embedding_model"),
                    "index_build_seconds": result.get("index_manifest", {}).get(
                        "benchmark_index_build_seconds"
                    ),
                    "rerank_backend_counts": runtime.get("rerank_backend_counts", {}),
                    "cross_encoder_fallback_chunk_count": runtime.get(
                        "cross_encoder_fallback_chunk_count",
                        0,
                    ),
                },
                "model_artifact": weights[mode],
            }
        )

    inactive = [item["mode"] for item in modes if not item["active"]]
    if inactive:
        raise ValueError(f"Required runtime modes fell back instead of running: {', '.join(inactive)}")

    snapshot = {
        "schema_version": 1,
        "snapshot_kind": "runtime_backend_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "report_generated_at_utc": report.get("generated_at_utc"),
            "report_path": _relative_path(report_path, root),
            "command": "pdf-to-json-rag compare-runtime-modes --modes baseline,sentence-transformers,cross-encoder --all-cases --json",
            "case_count": report.get("case_count"),
            "k": report.get("k"),
            "all_pass": report.get("all_pass"),
            "comparison_index_policy": report.get("comparison_index_policy"),
            "comparison_chunk_count": report.get("comparison_chunk_count"),
        },
        "benchmark_environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "latency_note": "Host-specific single-process measurement; includes first-use model loading and should be compared directionally.",
        },
        "modes": modes,
        "decision": {
            "default_backend": "auto",
            "preferred_when_cached": "sentence-transformers",
            "offline_fallback": "baseline",
            "experimental_opt_in": "cross-encoder",
            "default_change_allowed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentence-weight", type=Path, required=True)
    parser.add_argument("--cross-encoder-weight", type=Path, required=True)
    args = parser.parse_args()
    snapshot = build_snapshot(
        report_path=args.report,
        output_path=args.output,
        sentence_weight=args.sentence_weight,
        cross_encoder_weight=args.cross_encoder_weight,
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
