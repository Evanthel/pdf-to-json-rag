"""CLI entry points for the local PDF-to-JSON RAG tool."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata as importlib_metadata
from importlib import resources as importlib_resources
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse

import fitz

from . import __version__
from .answering import answer_query_with_retrieval, format_grounded_answer
from .chunking import load_document_record, process_saved_document_to_chunks
from .config import PATHS
from .document_inventory import (
    get_inventory_entry,
    load_document_inventory,
    shortlist_document_candidates,
    shortlist_documents,
)
from .evaluation import (
    DEFAULT_EVAL_FILENAME,
    DEFAULT_RUNTIME_COMPARISON_REPORT_FILENAME,
    DEFAULT_RUNTIME_PROMOTION_SNAPSHOT_FILENAME,
    ensure_default_eval_cases,
    run_mvp_evaluation,
    run_regression_suite,
    run_runtime_mode_comparison,
)
from .extraction import process_native_pdf_to_json
from .indexing import build_local_index, embedding_runtime_diagnostics, load_chunk_records
from .query_planning import plan_query
from .retrieval import retrieve_top_k, retrieve_top_k_with_neighbors


class CliError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliError("invalid_arguments", message)


COMMAND_ALIASES = {
    "extract": "extract-native",
    "chunk": "chunk-document",
    "index": "build-index",
    "workflow": "run-workflow",
    "create-demo": "create-demo-pdf",
    "list": "list-documents",
    "inspect": "inspect-document",
    "plan": "plan-query",
    "answer": "answer-query",
    "demo": "demo-profile",
    "self-check": "doctor",
    "layout-check": "layout-sanity-check",
    "corpus-check": "corpus-sanity-check",
    "compare-modes": "compare-runtime-modes",
    "runtime": "runtime-check",
    "promotion-report": "runtime-promotion-report",
    "readme-smoke": "readme-smoke-check",
    "beta-check": "public-beta-check",
}

COMMAND_HELP: dict[str, dict[str, object]] = {
    "init": {
        "summary": "Create local data directories under the configured data root.",
        "example": "pdf-to-json-rag init --json",
    },
    "create-demo-pdf": {
        "summary": "Create a small public-safe demo PDF for first-run workflow checks.",
        "example": "pdf-to-json-rag create-demo-pdf --path /tmp/demo.pdf --json",
    },
    "extract-native": {
        "summary": "Extract a PDF into document-level JSON artifacts.",
        "example": "pdf-to-json-rag extract-native --pdf /path/to/file.pdf --json",
    },
    "chunk-document": {
        "summary": "Turn one saved document JSON into chunk JSON files.",
        "example": "pdf-to-json-rag chunk-document --doc-id your-doc-id --json",
    },
    "build-index": {
        "summary": "Build the local vector index from one or more chunked documents.",
        "example": "pdf-to-json-rag build-index --doc-ids doc-a,doc-b --json",
    },
    "run-workflow": {
        "summary": "Run extract -> chunk -> index -> plan -> answer in one command.",
        "example": "pdf-to-json-rag run-workflow --pdf /path/to/file.pdf --query \"What does this file cover?\" --json",
    },
    "smoke-check": {
        "summary": "Validate the packaged workflow path and return pass/fail checks.",
        "example": "pdf-to-json-rag smoke-check --pdf /path/to/file.pdf --query \"What does this file cover?\" --json",
    },
    "list-documents": {
        "summary": "List indexed document inventory entries, optionally filtered by a query.",
        "example": "pdf-to-json-rag list-documents --json",
    },
    "inspect-document": {
        "summary": "Inspect one document inventory entry and its metadata contract.",
        "example": "pdf-to-json-rag inspect-document --doc-id common-cold-clinincal-evidence --json",
    },
    "plan-query": {
        "summary": "Classify a query before retrieval and show its answer mode.",
        "example": "pdf-to-json-rag plan-query --query \"Which file is most relevant for drought triggers?\" --json",
    },
    "retrieve": {
        "summary": "Return top-k chunks without answer assembly.",
        "example": "pdf-to-json-rag retrieve --query \"What are common cold symptoms?\" --top-k 5 --json",
    },
    "retrieve-expanded": {
        "summary": "Return top-k chunks plus adjacent expansion.",
        "example": "pdf-to-json-rag retrieve-expanded --query \"What are common cold symptoms?\" --top-k 5 --json",
    },
    "answer-query": {
        "summary": "Answer a query using the current local index and answer-mode logic.",
        "example": "pdf-to-json-rag answer-query --query \"What are common cold symptoms?\" --json",
    },
    "evaluate-mvp": {
        "summary": "Run the full local benchmark and write the evaluation report.",
        "example": "pdf-to-json-rag evaluate-mvp --top-k 5 --json",
    },
    "evaluate-regression": {
        "summary": "Run a smaller regression shard or explicit case subset.",
        "example": "pdf-to-json-rag evaluate-regression --shard query_planning_core --top-k 5 --json",
    },
    "compare-runtime-modes": {
        "summary": "Compare baseline, sentence-transformers, cross-encoder, and opt-in LLM synthesis modes on the same eval cases.",
        "example": "pdf-to-json-rag compare-runtime-modes --shard evidence_anchor_core --json",
    },
    "runtime-check": {
        "summary": "Report effective embedding/runtime backend selection and local optional-model readiness.",
        "example": "pdf-to-json-rag runtime-check --json",
    },
    "runtime-promotion-report": {
        "summary": "Summarize the latest runtime-mode comparison and promotion gate decision.",
        "example": "pdf-to-json-rag runtime-promotion-report --json",
    },
    "readme-smoke-check": {
        "summary": "Maintainer check: install the package into a temporary environment and replay the public README smoke workflow.",
        "example": "pdf-to-json-rag readme-smoke-check --json",
    },
    "public-beta-check": {
        "summary": "Maintainer check: aggregate public README smoke, runtime decision, corpus quick gate, and compact release summary.",
        "example": "pdf-to-json-rag public-beta-check --json",
    },
    "demo-profile": {
        "summary": "Show a public-safe demo profile with stable example commands and queries.",
        "example": "pdf-to-json-rag demo-profile --json",
    },
    "doctor": {
        "summary": "Check install/runtime readiness without touching private benchmark inputs.",
        "example": "pdf-to-json-rag doctor --json",
    },
    "package-check": {
        "summary": "Maintainer check: build a wheel and verify the packaged CLI from a clean temporary install root.",
        "example": "pdf-to-json-rag package-check --json",
    },
    "release-check": {
        "summary": "Maintainer check: run public-surface smoke checks plus package/test/regression release gates.",
        "example": "pdf-to-json-rag release-check --json",
    },
    "layout-sanity-check": {
        "summary": "Maintainer check: run isolated local sanity workflows on one or more external PDFs without adding them to the benchmark.",
        "example": "pdf-to-json-rag layout-sanity-check --pdfs /path/a.pdf,/path/b.pdf --json",
    },
    "corpus-sanity-check": {
        "summary": "Maintainer check: sample the local pdf/ corpus and run compact semantic sanity workflows on unfamiliar PDFs.",
        "example": "pdf-to-json-rag corpus-sanity-check --profile quick --json",
    },
    "help": {
        "summary": "Show command summaries or detailed help for one command.",
        "example": "pdf-to-json-rag help --topic answer-query",
    },
}

CANONICAL_COMMANDS = list(COMMAND_HELP.keys())
CLI_EPILOG = """Common first-run commands:
  python -m pip install .
  pdf-to-json-rag init --json
  pdf-to-json-rag doctor --json
  pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
  pdf-to-json-rag smoke-check --pdf /path/to/file.pdf --query "What does this file cover?" --json

Use `pdf-to-json-rag help --topic <command>` for a focused command summary.
"""

EXPECTED_EXAMPLE_FILES = (
    "public_demo_profile.json",
    "public_workflow.json",
    "public_demo_queries.json",
    "inspect_document.example.json",
    "plan_query.example.json",
    "answer_query.example.json",
)
CORPUS_SAMPLE_PROFILES = {
    "quick": 4,
    "balanced": 12,
    "stress": 24,
}
CORPUS_BUCKET_ORDER = ("scan_like", "form_like", "short_doc", "medium_doc", "long_doc")


@dataclass(frozen=True)
class LocalPdfCorpusEntry:
    digest: str
    pdf_path: Path
    urlkey: str
    original: str
    pages: int
    file_size: int
    creator_tool: str
    producer: str
    bucket: str


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
            reasons.append("internal benchmark regressions were skipped because benchmark assets were not present in the active data root")
        reasons.append("known limitations are documented and do not block the current public release path")
        return {
            "release_ready": True,
            "suggested_tag": "v0.1.0-beta",
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
            "semantic_discovery_score": round(candidate.breakdown.semantic_discovery_score, 3),
            "facet_fit_score": round(candidate.breakdown.facet_fit_score, 3),
            "rarity_distinctive_score": round(candidate.breakdown.rarity_distinctive_score, 3),
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
        payload["mode_scores"] = {key: round(value, 3) for key, value in plan.mode_scores.items()}
        payload["shortlist"] = [_shortlist_candidate_payload(candidate) for candidate in plan.shortlist]
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


def _answer_contract_health(answer_trace: dict[str, object], *, evidence_count: int = 0) -> dict[str, object]:
    retrieval_contract = answer_trace.get("retrieval_contract", {})
    document_synthesis = answer_trace.get("document_synthesis", {})
    answer_contract = answer_trace.get("answer_contract", {})
    support_trace = answer_trace.get("support_trace", [])
    claim_alignment = answer_trace.get("claim_alignment", {})
    answer_mode = str(answer_trace.get("answer_mode") or "")
    support_doc_ids = document_synthesis.get("support_doc_ids", []) if isinstance(document_synthesis, dict) else []
    selected_doc_ids = answer_contract.get("primary_doc_ids", []) if isinstance(answer_contract, dict) else []
    support_scope = document_synthesis.get("support_scope") if isinstance(document_synthesis, dict) else None
    retrieval_path = retrieval_contract.get("retrieval_path") if isinstance(retrieval_contract, dict) else None
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
    return {
        "all_pass": all(item["passed"] for item in checks),
        "checks": checks,
        "retrieval_path": retrieval_path,
        "support_scope": support_scope,
        "selected_doc_ids": list(selected_doc_ids) if isinstance(selected_doc_ids, list) else [],
        "support_doc_ids": list(support_doc_ids) if isinstance(support_doc_ids, list) else [],
        "support_trace_count": len(support_trace) if isinstance(support_trace, list) else 0,
        "evidence_count": evidence_count,
    }


def _grounded_answer_payload(result, *, verbose: bool = False) -> dict[str, object]:
    answer_trace = result.answer_trace if verbose else _compact_answer_trace(result.answer_trace)
    evidence_payload = [_evidence_payload(item) for item in result.evidence] if verbose else []
    return {
        "query": result.query,
        "query_intent": result.query_intent,
        "answer": result.answer,
        "answer_trace": answer_trace,
        "contract_health": _answer_contract_health(answer_trace, evidence_count=len(result.evidence)),
        **(
            {
                "top_k_hits": [_chunk_payload(chunk) for chunk in result.top_k_hits],
                "expanded_hits": [_chunk_payload(chunk) for chunk in result.expanded_hits],
                "evidence": evidence_payload,
            }
            if verbose
            else {}
        ),
    }


def _write_json_output(payload: dict[str, object], output_path: Path | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _emit_json(command: str, payload: dict[str, object], output_path: Path | None = None) -> None:
    _write_json_output(
        {
            "command": command,
            "version": __version__,
            "ok": True,
            "result": payload,
        },
        output_path=output_path,
    )


def _emit_error_json(command: str | None, error: CliError, output_path: Path | None = None) -> None:
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


def _quality_status(score: float | int | None, *, warn_at: float = 0.55, pass_at: float = 0.7) -> str:
    if score is None:
        return "unknown"
    numeric_score = float(score)
    if numeric_score >= pass_at:
        return "pass"
    if numeric_score >= warn_at:
        return "warn"
    return "fail"


def _workflow_quality_profile(payload: dict[str, object]) -> dict[str, object]:
    document = payload.get("document", {}) if isinstance(payload.get("document"), dict) else {}
    index = payload.get("index", {}) if isinstance(payload.get("index"), dict) else {}
    answer = payload.get("answer", {}) if isinstance(payload.get("answer"), dict) else {}
    answer_trace = answer.get("answer_trace", {}) if isinstance(answer.get("answer_trace"), dict) else {}
    contract_health = answer.get("contract_health", {}) if isinstance(answer.get("contract_health"), dict) else {}
    structure_confidence = document.get("structure_confidence")
    layout_confidence = document.get("layout_confidence")
    semantic_confidence = document.get("semantic_confidence")
    processing_checks = [
        {"name": "chunks_created", "passed": bool(index.get("chunk_count", 0) > 0)},
        {"name": "structure_confidence_present", "passed": structure_confidence is not None},
        {"name": "layout_confidence_present", "passed": layout_confidence is not None},
    ]
    semantic_checks = [
        {"name": "semantic_confidence_present", "passed": semantic_confidence is not None},
        {"name": "document_type_present", "passed": bool(document.get("document_type"))},
        {"name": "document_purpose_present", "passed": bool(document.get("document_purpose"))},
        {"name": "inventory_summary_present", "passed": bool(document.get("inventory_summary"))},
    ]
    retrieval_checks = [
        {"name": "contract_health_available", "passed": bool(contract_health)},
        {"name": "retrieval_contract_present", "passed": bool(answer_trace.get("retrieval_contract"))},
        {"name": "support_scope_present", "passed": bool(contract_health.get("support_scope"))},
        {"name": "support_available", "passed": bool(contract_health.get("support_trace_count") or contract_health.get("evidence_count"))},
    ]
    answer_checks = [
        {"name": "answer_present", "passed": bool(answer.get("answer"))},
        {"name": "answer_contract_present", "passed": bool(answer_trace.get("answer_contract"))},
        {"name": "claim_alignment_present", "passed": bool(answer_trace.get("claim_alignment"))},
    ]
    semantic_classification = (
        "well_supported"
        if isinstance(semantic_confidence, (int, float)) and semantic_confidence >= 0.85
        else "provisional"
        if isinstance(semantic_confidence, (int, float)) and semantic_confidence >= 0.55
        else "unknown"
    )
    return {
        "processing_quality": {
            "status": "pass" if all(item["passed"] for item in processing_checks) else "fail",
            "checks": processing_checks,
            "structure_confidence": structure_confidence,
            "layout_confidence": layout_confidence,
            "structure_status": _quality_status(structure_confidence),
            "layout_status": _quality_status(layout_confidence),
        },
        "semantic_confidence": {
            "status": "pass" if all(item["passed"] for item in semantic_checks) else "fail",
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
            "status": "pass" if all(item["passed"] for item in retrieval_checks) else "warn",
            "checks": retrieval_checks,
            "retrieval_path": contract_health.get("retrieval_path"),
            "support_scope": contract_health.get("support_scope"),
            "support_doc_ids": contract_health.get("support_doc_ids", []),
            "selected_doc_ids": contract_health.get("selected_doc_ids", []),
        },
        "answer_trust": {
            "status": "pass" if all(item["passed"] for item in answer_checks) else "warn",
            "checks": answer_checks,
            "contract_health": contract_health,
        },
    }


def _wants_json(argv: list[str]) -> bool:
    if "--json" in argv:
        return True
    for index, token in enumerate(argv[:-1]):
        if token == "--format" and argv[index + 1] == "json":
            return True
    return False


def _resolve_output_path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _require_arg(value: str | None, flag: str, command: str) -> str:
    if value:
        return value
    raise CliError(
        "missing_argument",
        f"{flag} is required for {command}",
        {"command": command, "flag": flag},
    )


def _resolve_pdf_path(value: str) -> Path:
    pdf_path = Path(value).expanduser().resolve()
    if not pdf_path.exists():
        raise CliError(
            "missing_pdf",
            f"PDF file does not exist: {pdf_path}",
            {"pdf": str(pdf_path)},
        )
    if not pdf_path.is_file():
        raise CliError(
            "invalid_pdf_path",
            f"PDF path is not a file: {pdf_path}",
            {"pdf": str(pdf_path)},
        )
    return pdf_path


def _resolve_pdf_paths(value: str) -> list[Path]:
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        raise CliError(
            "missing_pdf",
            "At least one PDF path is required",
            {"pdfs": value},
        )
    return [_resolve_pdf_path(item) for item in raw_items]


def _local_pdf_corpus_paths(corpus_dir: Path | None = None) -> tuple[Path, Path] | None:
    if corpus_dir is not None:
        metadata_path = corpus_dir / "lcwa_gov_pdf_metadata.csv"
        return (corpus_dir, metadata_path) if corpus_dir.exists() and metadata_path.exists() else None
    project_root = _discover_project_root(PATHS.root) or _discover_project_root(Path.cwd())
    if project_root is None:
        return None
    pdf_dir = project_root / "pdf"
    metadata_path = pdf_dir / "lcwa_gov_pdf_metadata.csv"
    if pdf_dir.exists() and metadata_path.exists():
        return pdf_dir, metadata_path
    return None


def _safe_int(value: str | None) -> int:
    try:
        return int(str(value or "0").strip())
    except ValueError:
        return 0


def _pdf_corpus_bucket(url_text: str, *, pages: int, creator_tool: str, producer: str) -> str:
    lowered = " ".join((url_text, creator_tool, producer)).lower()
    if any(term in lowered for term in ("scan", "scanning", "capture", "ocr")):
        return "scan_like"
    if any(term in lowered for term in ("form", "statement", "application", "questionnaire", "checklist", "appendix")):
        return "form_like"
    if pages <= 2:
        return "short_doc"
    if pages >= 20:
        return "long_doc"
    return "medium_doc"


def _corpus_alias_name(entry: LocalPdfCorpusEntry) -> str:
    source = entry.original or entry.urlkey or entry.digest
    parsed = urlparse(source if "://" in source else f"https://{source}")
    candidate = Path(unquote(parsed.path)).name or Path(unquote(source)).name
    candidate_stem = Path(candidate).stem or entry.digest
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate_stem).strip("-._")
    if not sanitized:
        sanitized = entry.digest.lower()
    return f"{sanitized}.pdf"


def _load_local_pdf_corpus(corpus_dir: Path | None = None) -> list[LocalPdfCorpusEntry]:
    corpus_paths = _local_pdf_corpus_paths(corpus_dir)
    if corpus_paths is None:
        location = str(corpus_dir) if corpus_dir else "repo-local pdf/"
        raise CliError(
            "missing_local_pdf_corpus",
            f"Local PDF corpus was not found: {location}",
            {"corpus_dir": location},
        )
    pdf_dir, metadata_path = corpus_paths
    entries: list[LocalPdfCorpusEntry] = []
    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        metadata_text = metadata_path.read_text(encoding="latin-1")
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", newline="", delete=True) as handle:
        handle.write(metadata_text)
        handle.flush()
        handle.seek(0)
        reader = csv.DictReader(handle)
        for row in reader:
            digest = str(row.get("digest", "")).strip()
            if not digest:
                continue
            pdf_path = pdf_dir / f"{digest}.pdf"
            if not pdf_path.exists():
                continue
            pages = _safe_int(row.get("pages"))
            file_size = _safe_int(row.get("file_size"))
            if pages <= 0 or file_size <= 0:
                continue
            creator_tool = str(row.get("creator_tool", "") or "").strip()
            producer = str(row.get("producer", "") or "").strip()
            urlkey = str(row.get("urlkey", "") or "").strip()
            original = str(row.get("original", "") or "").strip()
            entries.append(
                LocalPdfCorpusEntry(
                    digest=digest,
                    pdf_path=pdf_path.resolve(),
                    urlkey=urlkey,
                    original=original,
                    pages=pages,
                    file_size=file_size,
                    creator_tool=creator_tool,
                    producer=producer,
                    bucket=_pdf_corpus_bucket(
                        f"{urlkey} {original}",
                        pages=pages,
                        creator_tool=creator_tool,
                        producer=producer,
                    ),
                )
            )
    entries.sort(key=lambda item: (item.bucket, item.pages, item.file_size, item.digest))
    return entries


def _sample_local_pdf_corpus(entries: list[LocalPdfCorpusEntry], sample_size: int) -> list[LocalPdfCorpusEntry]:
    if sample_size <= 0:
        raise CliError(
            "invalid_sample_size",
            "Sample size must be a positive integer",
            {"sample_size": sample_size},
        )
    grouped: dict[str, list[LocalPdfCorpusEntry]] = {bucket: [] for bucket in CORPUS_BUCKET_ORDER}
    for entry in entries:
        grouped.setdefault(entry.bucket, []).append(entry)
    for bucket_entries in grouped.values():
        bucket_entries.sort(key=lambda item: (item.pages, item.file_size, item.digest))

    sampled: list[LocalPdfCorpusEntry] = []
    while len(sampled) < sample_size:
        progressed = False
        for bucket in CORPUS_BUCKET_ORDER:
            bucket_entries = grouped.get(bucket, [])
            if not bucket_entries:
                continue
            sampled.append(bucket_entries.pop(0))
            progressed = True
            if len(sampled) >= sample_size:
                break
        if not progressed:
            break
    return sampled


def _corpus_sampling_manifest(
    entries: list[LocalPdfCorpusEntry],
    sampled_entries: list[LocalPdfCorpusEntry],
    *,
    sample_profile: str,
    requested_sample_size: int,
) -> dict[str, object]:
    selected_digests = [entry.digest for entry in sampled_entries]
    checksum_input = "\n".join(selected_digests).encode("utf-8")
    return {
        "sampling_algorithm": "bucket_round_robin_v1",
        "bucket_order": list(CORPUS_BUCKET_ORDER),
        "sample_profile": sample_profile,
        "requested_sample_size": requested_sample_size,
        "available_pdf_count": len(entries),
        "available_bucket_counts": _count_values([entry.bucket for entry in entries]),
        "selected_pdf_count": len(sampled_entries),
        "selected_bucket_counts": _count_values([entry.bucket for entry in sampled_entries]),
        "selected_digest_checksum": hashlib.sha256(checksum_input).hexdigest(),
        "selected_digests": selected_digests,
    }


def _resolve_corpus_sample_size(sample_profile: str, sample_size: int | None) -> int:
    if sample_size is not None:
        return sample_size
    return CORPUS_SAMPLE_PROFILES[sample_profile]


def _resolve_index_dir(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default


def _default_public_index_dir() -> Path:
    workflow_smoke_dir = PATHS.data_index / "workflow_smoke"
    if not (PATHS.data_index / "index_manifest.json").exists() and (
        workflow_smoke_dir / "index_manifest.json"
    ).exists():
        return workflow_smoke_dir
    return PATHS.data_index


def _resolve_optional_path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default


def _create_demo_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Demo Safety Guide\n\n"
        "Purpose: provide procedural guidance for basic safety checks, incident response, and follow-up.\n"
        "Audience: operations staff and team leads.\n"
        "Section 1: Preparation\n"
        "Use the checklist before field work.\n"
        "Section 2: Response\n"
        "Report incidents, document evidence, and notify the supervisor.\n"
        "Section 3: Follow-up\n"
        "Review the event, record lessons learned, and close the ticket.\n",
    )
    doc.save(path)
    doc.close()
    return path


def _subprocess_env(data_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PATHS.root / "src")
    maintainer_root = _discover_project_root(PATHS.root) or _discover_project_root(Path.cwd())
    if maintainer_root is not None:
        env["PDF_TO_JSON_RAG_PROJECT_ROOT"] = str(maintainer_root)
    if data_dir is not None:
        env["PDF_TO_JSON_RAG_DATA_DIR"] = str(data_dir)
    return env


def _run_cli_subprocess(args: list[str], data_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pdf_to_json_rag", *args],
        cwd=PATHS.root,
        env=_subprocess_env(data_dir=data_dir),
        capture_output=True,
        text=True,
    )


def _process_output_tail(process: subprocess.CompletedProcess[str] | None) -> str:
    if process is None:
        return ""
    return "\n".join(
        part.strip()
        for part in (process.stdout, process.stderr)
        if part and part.strip()
    )[-1200:]


def _json_payload_from_process(process: subprocess.CompletedProcess[str] | None) -> dict[str, object]:
    if process is None or not process.stdout.strip():
        return {}
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_installed_readme_flow(script_path: Path, workspace: Path, data_dir: Path) -> dict[str, object]:
    package_env = os.environ.copy()
    package_env["PDF_TO_JSON_RAG_DATA_DIR"] = str(data_dir)
    demo_pdf = workspace / "readme-demo.pdf"

    steps: list[dict[str, object]] = []

    def run_step(name: str, args: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        process = subprocess.run(
            [str(script_path), *args],
            cwd=workspace,
            env=package_env,
            capture_output=True,
            text=True,
        )
        payload = _json_payload_from_process(process)
        steps.append(
            {
                "name": name,
                "command": "pdf-to-json-rag " + " ".join(args),
                "returncode": process.returncode,
                "ok": bool(payload.get("ok")),
                "output_tail": _process_output_tail(process) if process.returncode != 0 or not payload.get("ok") else "",
            }
        )
        return process, payload

    init_process, init_payload = run_step("init", ["init", "--json"])
    doctor_process, doctor_payload = run_step("doctor", ["doctor", "--json"])
    create_process, create_payload = run_step(
        "create-demo-pdf",
        ["create-demo-pdf", "--path", str(demo_pdf), "--json"],
    )
    smoke_process, smoke_payload = run_step(
        "smoke-check",
        [
            "smoke-check",
            "--pdf",
            str(demo_pdf),
            "--query",
            "What does this file cover?",
            "--json",
        ],
    )
    runtime_process, runtime_payload = run_step("runtime-check", ["runtime-check", "--json"])

    return {
        "workspace": str(workspace),
        "data_dir": str(data_dir),
        "demo_pdf": str(demo_pdf),
        "script_path": str(script_path),
        "path_type": "installed_console_script",
        "public_path": True,
        "maintainer_benchmark_path": False,
        "steps": steps,
        "init_returncode": init_process.returncode,
        "doctor_returncode": doctor_process.returncode,
        "create_demo_returncode": create_process.returncode,
        "smoke_returncode": smoke_process.returncode,
        "runtime_returncode": runtime_process.returncode,
        "init_ok": bool(init_payload.get("ok")),
        "doctor_ok": bool(doctor_payload.get("ok")),
        "doctor_ready_for_public_cli": bool(doctor_payload.get("result", {}).get("ready_for_public_cli")),
        "create_demo_ok": bool(create_payload.get("ok")),
        "smoke_ok": bool(smoke_payload.get("ok")),
        "smoke_all_pass": bool(smoke_payload.get("result", {}).get("all_pass")),
        "runtime_ok": bool(runtime_payload.get("ok")),
        "runtime_decision": runtime_payload.get("result", {}).get("runtime_decision", {}),
        "all_pass": (
            bool(init_payload.get("ok"))
            and bool(doctor_payload.get("ok"))
            and bool(doctor_payload.get("result", {}).get("ready_for_public_cli"))
            and bool(create_payload.get("ok"))
            and bool(smoke_payload.get("ok"))
            and bool(smoke_payload.get("result", {}).get("all_pass"))
            and bool(runtime_payload.get("ok"))
        ),
    }


def _run_public_surface_release_smoke() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir_name:
        workspace = Path(temp_dir_name)
        data_dir = workspace / "data"
        demo_pdf = _create_demo_pdf(workspace / "public-release-demo.pdf")
        init_process = _run_cli_subprocess(["init", "--json"], data_dir=data_dir)
        smoke_process = _run_cli_subprocess(
            [
                "smoke-check",
                "--pdf",
                str(demo_pdf),
                "--query",
                "What does this file cover?",
                "--json",
            ],
            data_dir=data_dir,
        )
        init_payload = json.loads(init_process.stdout) if init_process.stdout.strip() else {}
        smoke_payload = json.loads(smoke_process.stdout) if smoke_process.stdout.strip() else {}
        return {
            "workspace": str(workspace),
            "data_dir": str(data_dir),
            "demo_pdf": str(demo_pdf),
            "init_returncode": init_process.returncode,
            "smoke_returncode": smoke_process.returncode,
            "init_ok": bool(init_payload.get("ok")),
            "smoke_ok": bool(smoke_payload.get("ok")),
            "smoke_all_pass": bool(smoke_payload.get("result", {}).get("all_pass")),
            "smoke_checks": smoke_payload.get("result", {}).get("checks", []),
        }


def _run_layout_sanity_check(
    pdf_paths: list[Path],
    k: int = 5,
    *,
    display_pdf_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    results: list[dict[str, object]] = []

    for pdf_path in pdf_paths:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            workspace = Path(temp_dir_name)
            data_dir = workspace / "data"
            smoke_process = _run_cli_subprocess(
                [
                    "smoke-check",
                    "--pdf",
                    str(pdf_path),
                    "--query",
                    "What does this file cover?",
                    "--json",
                ],
                data_dir=data_dir,
            )
            smoke_payload = json.loads(smoke_process.stdout) if smoke_process.stdout.strip() else {}

            result: dict[str, object] = {
                "pdf": (display_pdf_paths or {}).get(str(pdf_path), str(pdf_path)),
                "workspace": str(workspace),
                "data_dir": str(data_dir),
                "smoke_returncode": smoke_process.returncode,
                "smoke_ok": bool(smoke_payload.get("ok")),
                "smoke_all_pass": bool(smoke_payload.get("result", {}).get("all_pass")),
            }

            if not smoke_payload.get("ok"):
                result["error"] = smoke_payload.get("error", {})
                result["checks"] = [
                    {"name": "smoke_ok", "passed": False},
                    {"name": "smoke_all_pass", "passed": False},
                ]
                results.append(result)
                continue

            smoke_result = smoke_payload.get("result", {})
            doc_id = str(smoke_result.get("doc_id", ""))
            inspect_process = _run_cli_subprocess(
                ["inspect-document", "--doc-id", doc_id, "--json"],
                data_dir=data_dir,
            )
            inspect_payload = json.loads(inspect_process.stdout) if inspect_process.stdout.strip() else {}
            type_process = _run_cli_subprocess(
                ["answer-query", "--query", "What kind of document is this?", "--json"],
                data_dir=data_dir,
            )
            type_payload = json.loads(type_process.stdout) if type_process.stdout.strip() else {}
            purpose_process = _run_cli_subprocess(
                ["answer-query", "--query", "What is the purpose of this document?", "--json"],
                data_dir=data_dir,
            )
            purpose_payload = json.loads(purpose_process.stdout) if purpose_process.stdout.strip() else {}
            audience_process = _run_cli_subprocess(
                ["answer-query", "--query", "Who is this document for?", "--json"],
                data_dir=data_dir,
            )
            audience_payload = json.loads(audience_process.stdout) if audience_process.stdout.strip() else {}
            confidence_process = _run_cli_subprocess(
                ["answer-query", "--query", "How confident is this document classification?", "--json"],
                data_dir=data_dir,
            )
            confidence_payload = json.loads(confidence_process.stdout) if confidence_process.stdout.strip() else {}
            rationale_process = _run_cli_subprocess(
                ["answer-query", "--query", "Why is this document classified this way?", "--json"],
                data_dir=data_dir,
            )
            rationale_payload = json.loads(rationale_process.stdout) if rationale_process.stdout.strip() else {}
            limits_process = _run_cli_subprocess(
                ["answer-query", "--query", "What are the main limits of this document classification?", "--json"],
                data_dir=data_dir,
            )
            limits_payload = json.loads(limits_process.stdout) if limits_process.stdout.strip() else {}

            inspect_result = inspect_payload.get("result", {}) if inspect_payload.get("ok") else {}
            overview_answer = smoke_result.get("answer", {}).get("answer", "")
            type_answer = type_payload.get("result", {}).get("answer", "") if type_payload.get("ok") else ""
            purpose_answer = purpose_payload.get("result", {}).get("answer", "") if purpose_payload.get("ok") else ""
            audience_answer = audience_payload.get("result", {}).get("answer", "") if audience_payload.get("ok") else ""
            confidence_answer = confidence_payload.get("result", {}).get("answer", "") if confidence_payload.get("ok") else ""
            rationale_answer = rationale_payload.get("result", {}).get("answer", "") if rationale_payload.get("ok") else ""
            limits_answer = limits_payload.get("result", {}).get("answer", "") if limits_payload.get("ok") else ""
            confidence_support = confidence_payload.get("result", {}).get("answer_trace", {}).get("support_trace", [])
            confidence_support_item = confidence_support[0] if confidence_support else {}
            structure_confidence = inspect_result.get("structure_confidence")
            layout_confidence = inspect_result.get("layout_confidence")
            semantic_confidence = inspect_result.get("semantic_confidence")
            semantic_confidence_label = inspect_result.get("semantic_confidence_label")

            checks = [
                {"name": "smoke_ok", "passed": bool(smoke_payload.get("ok"))},
                {"name": "smoke_all_pass", "passed": bool(smoke_result.get("all_pass"))},
                {"name": "inspect_ok", "passed": bool(inspect_payload.get("ok"))},
                {"name": "structure_confidence_present", "passed": structure_confidence is not None},
                {"name": "layout_confidence_present", "passed": layout_confidence is not None},
                {"name": "semantic_confidence_present", "passed": semantic_confidence is not None},
                {"name": "overview_answer_present", "passed": bool(overview_answer)},
                {"name": "type_answer_present", "passed": bool(type_answer)},
                {"name": "purpose_answer_present", "passed": bool(purpose_answer)},
                {"name": "audience_answer_present", "passed": bool(audience_answer)},
                {"name": "confidence_answer_present", "passed": bool(confidence_answer)},
                {"name": "rationale_answer_present", "passed": bool(rationale_answer)},
                {"name": "limits_answer_present", "passed": bool(limits_answer)},
                {"name": "type_vs_purpose_distinct", "passed": bool(type_answer and purpose_answer and type_answer != purpose_answer)},
            ]

            result.update(
                {
                    "doc_id": doc_id,
                    "document_type": inspect_result.get("document_type") or smoke_result.get("document", {}).get("document_type"),
                    "document_purpose": inspect_result.get("document_purpose") or smoke_result.get("document", {}).get("document_purpose"),
                    "document_family": inspect_result.get("document_family") or smoke_result.get("document", {}).get("document_family"),
                    "structure_confidence": structure_confidence,
                    "layout_confidence": layout_confidence,
                    "semantic_confidence": semantic_confidence,
                    "semantic_confidence_label": semantic_confidence_label,
                    "classification_status": confidence_support_item.get("classification_status"),
                    "trust_policy": confidence_support_item.get("trust_policy"),
                    "semantic_specificity": (
                        _is_specific_document_type(
                            inspect_result.get("document_type")
                            or smoke_result.get("document", {}).get("document_type")
                        )
                        or _is_specific_document_purpose(
                            inspect_result.get("document_purpose")
                            or smoke_result.get("document", {}).get("document_purpose")
                        )
                    ),
                    "semantic_rationale": inspect_result.get("semantic_rationale", []),
                    "semantic_warnings": inspect_result.get("semantic_warnings", []),
                    "section_count": inspect_result.get("section_count"),
                    "chunk_count": smoke_result.get("index", {}).get("chunk_count"),
                    "overview_answer": overview_answer,
                    "type_answer": type_answer,
                    "purpose_answer": purpose_answer,
                    "audience_answer": audience_answer,
                    "confidence_answer": confidence_answer,
                    "rationale_answer": rationale_answer,
                    "limits_answer": limits_answer,
                    "smoke_checks": smoke_result.get("checks", []),
                    "checks": checks,
                    "all_pass": all(item["passed"] for item in checks),
                }
            )
            result["trust_limited"] = _is_trust_limited(result)
            result["semantic_pass"] = _semantic_pass(result)
            results.append(result)

    return {
        "pdf_count": len(pdf_paths),
        "results": results,
        "all_pass": all(bool(item.get("all_pass")) for item in results),
    }


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _is_specific_document_type(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value != "document"


def _is_specific_document_purpose(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value != "reference_lookup"


def _is_trust_limited(item: dict[str, object]) -> bool:
    return (
        str(item.get("classification_status", "")) == "uncertain"
        or str(item.get("trust_policy", "")) == "heuristic_semantic_guess"
        or str(item.get("semantic_confidence_label", "")) == "low"
    )


def _semantic_pass(item: dict[str, object]) -> bool:
    if not bool(item.get("all_pass")):
        return False
    has_specific_signal = _is_specific_document_type(item.get("document_type")) or _is_specific_document_purpose(
        item.get("document_purpose")
    )
    semantic_confidence = item.get("semantic_confidence")
    semantic_confidence_value = float(semantic_confidence) if semantic_confidence is not None else 0.0
    return (
        has_specific_signal
        and semantic_confidence_value >= 0.56
        and str(item.get("classification_status", "")) != "uncertain"
    )


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 3) if total else None


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _corpus_failure_reasons(item: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if not bool(item.get("all_pass")):
        reasons.append("technical_failure")
    if not _is_specific_document_type(item.get("document_type")):
        reasons.append("generic_document_type")
    if not _is_specific_document_purpose(item.get("document_purpose")):
        reasons.append("generic_document_purpose")
    if str(item.get("classification_status", "")) == "uncertain":
        reasons.append("uncertain_classification")
    if str(item.get("semantic_confidence_label", "")) == "low":
        reasons.append("low_semantic_confidence")
    if bool(item.get("trust_limited")):
        reasons.append("trust_limited")
    if not bool(item.get("semantic_pass")):
        reasons.append("semantic_gate_failed")
    return sorted(set(reasons))


def _corpus_failure_example(item: dict[str, object], reasons: list[str]) -> dict[str, object]:
    return {
        "pdf": str(item.get("pdf")),
        "bucket": item.get("bucket"),
        "reasons": reasons,
        "document_type": item.get("document_type"),
        "document_purpose": item.get("document_purpose"),
        "semantic_confidence": item.get("semantic_confidence"),
        "semantic_confidence_label": item.get("semantic_confidence_label"),
        "classification_status": item.get("classification_status"),
        "trust_policy": item.get("trust_policy"),
    }


def _corpus_bucket_diagnostics(sampled_results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in sampled_results:
        grouped.setdefault(str(item.get("bucket") or "unknown"), []).append(item)

    diagnostics: dict[str, dict[str, object]] = {}
    for bucket, items in sorted(grouped.items()):
        total = len(items)
        structure_values = [
            float(item["structure_confidence"]) for item in items if item.get("structure_confidence") is not None
        ]
        layout_values = [
            float(item["layout_confidence"]) for item in items if item.get("layout_confidence") is not None
        ]
        semantic_values = [
            float(item["semantic_confidence"]) for item in items if item.get("semantic_confidence") is not None
        ]
        reason_values: list[str] = []
        failing_pdfs: list[str] = []
        failure_examples: list[dict[str, object]] = []
        for item in items:
            reasons = _corpus_failure_reasons(item)
            if reasons:
                reason_values.extend(reasons)
                failing_pdfs.append(str(item.get("pdf")))
                if len(failure_examples) < 5:
                    failure_examples.append(_corpus_failure_example(item, reasons))
        diagnostics[bucket] = {
            "sample_count": total,
            "technical_pass_rate": _rate(sum(1 for item in items if bool(item.get("all_pass"))), total),
            "semantic_pass_rate": _rate(sum(1 for item in items if bool(item.get("semantic_pass"))), total),
            "specific_document_rate": _rate(
                sum(1 for item in items if _is_specific_document_type(item.get("document_type"))),
                total,
            ),
            "specific_purpose_rate": _rate(
                sum(1 for item in items if _is_specific_document_purpose(item.get("document_purpose"))),
                total,
            ),
            "low_confidence_rate": _rate(
                sum(1 for item in items if item.get("semantic_confidence_label") == "low"),
                total,
            ),
            "trust_limited_rate": _rate(sum(1 for item in items if bool(item.get("trust_limited"))), total),
            "avg_structure_confidence": _avg(structure_values),
            "avg_layout_confidence": _avg(layout_values),
            "avg_semantic_confidence": _avg(semantic_values),
            "dominant_failure_reasons": _count_values(reason_values),
            "failing_pdf_count": len(failing_pdfs),
            "failing_pdfs": failing_pdfs,
            "failure_examples": failure_examples,
        }
    return diagnostics


def _corpus_follow_up_actions(bucket_diagnostics: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for bucket, summary in bucket_diagnostics.items():
        failure_reasons = summary.get("dominant_failure_reasons", {})
        failure_examples = summary.get("failure_examples", [])
        if float(summary.get("technical_pass_rate") or 0.0) < 1.0:
            actions.append(
                {
                    "bucket": bucket,
                    "priority": "high",
                    "focus": "processing_layer",
                    "reason": "at least one sampled PDF did not complete the technical smoke path",
                    "dominant_failure_reasons": failure_reasons,
                    "failure_examples": failure_examples,
                }
            )
            continue
        if float(summary.get("semantic_pass_rate") or 0.0) < 0.66:
            actions.append(
                {
                    "bucket": bucket,
                    "priority": "high",
                    "focus": "document_semantics",
                    "reason": "semantic pass rate is below the corpus-layer threshold",
                    "dominant_failure_reasons": failure_reasons,
                    "failure_examples": failure_examples,
                }
            )
            continue
        if (
            float(summary.get("avg_structure_confidence") or 1.0) < 0.55
            or float(summary.get("avg_layout_confidence") or 1.0) < 0.55
        ):
            actions.append(
                {
                    "bucket": bucket,
                    "priority": "medium",
                    "focus": "layout_processing",
                    "reason": "structure or layout confidence is below the processing threshold",
                    "dominant_failure_reasons": failure_reasons,
                    "failure_examples": failure_examples,
                }
            )
            continue
        if float(summary.get("trust_limited_rate") or 0.0) > 0.34:
            actions.append(
                {
                    "bucket": bucket,
                    "priority": "medium",
                    "focus": "trust_policy",
                    "reason": "too many documents are classified as trust-limited",
                    "dominant_failure_reasons": failure_reasons,
                    "failure_examples": failure_examples,
                }
            )
            continue
        if float(summary.get("low_confidence_rate") or 0.0) > 0.34:
            actions.append(
                {
                    "bucket": bucket,
                    "priority": "medium",
                    "focus": "semantic_confidence",
                    "reason": "too many documents have low semantic confidence",
                    "dominant_failure_reasons": failure_reasons,
                    "failure_examples": failure_examples,
                }
            )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        actions,
        key=lambda item: (priority_order.get(str(item["priority"]), 99), str(item["bucket"])),
    )


def _corpus_contract_checks(
    bucket_diagnostics: dict[str, dict[str, object]],
    follow_up_actions: list[dict[str, object]],
    architecture_gates: dict[str, object],
) -> dict[str, object]:
    checks = [
        {
            "name": "bucket_diagnostics_present",
            "passed": bool(bucket_diagnostics),
        },
        {
            "name": "bucket_diagnostics_have_required_rates",
            "passed": all(
                all(
                    key in item
                    for key in (
                        "sample_count",
                        "technical_pass_rate",
                        "semantic_pass_rate",
                        "dominant_failure_reasons",
                        "failure_examples",
                    )
                )
                for item in bucket_diagnostics.values()
            ),
        },
        {
            "name": "architecture_gate_has_bucket_contract",
            "passed": all(key in architecture_gates for key in ("bucket_gate_pass", "bucket_follow_up_count")),
        },
        {
            "name": "follow_up_actions_have_required_fields",
            "passed": all(
                all(key in item for key in ("bucket", "priority", "focus", "reason", "dominant_failure_reasons"))
                for item in follow_up_actions
            ),
        },
        {
            "name": "follow_up_count_matches_gate",
            "passed": architecture_gates.get("bucket_follow_up_count") == len(follow_up_actions),
        },
    ]
    return {
        "all_pass": all(bool(item["passed"]) for item in checks),
        "checks": checks,
    }


def _write_corpus_sanity_snapshot(payload: dict[str, object]) -> Path:
    PATHS.data_eval.mkdir(parents=True, exist_ok=True)
    snapshot_path = PATHS.data_eval / "corpus_sanity_snapshot.json"
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_path


def _run_corpus_sanity_check(
    sample_size: int,
    *,
    corpus_dir: Path | None = None,
    k: int = 5,
    sample_profile: str = "custom",
    save_snapshot: bool = False,
) -> dict[str, object]:
    corpus_entries = _load_local_pdf_corpus(corpus_dir)
    sampled_entries = _sample_local_pdf_corpus(corpus_entries, sample_size)
    sample_manifest = _corpus_sampling_manifest(
        corpus_entries,
        sampled_entries,
        sample_profile=sample_profile,
        requested_sample_size=sample_size,
    )
    with tempfile.TemporaryDirectory() as alias_dir_name:
        alias_dir = Path(alias_dir_name)
        alias_paths: list[Path] = []
        display_paths: dict[str, str] = {}
        for entry in sampled_entries:
            alias_path = alias_dir / _corpus_alias_name(entry)
            if alias_path.exists():
                alias_path = alias_dir / f"{alias_path.stem}-{entry.digest.lower()[:8]}{alias_path.suffix}"
            shutil.copyfile(entry.pdf_path, alias_path)
            alias_paths.append(alias_path)
            display_paths[str(alias_path)] = str(entry.pdf_path)

        layout_payload = _run_layout_sanity_check(alias_paths, k=k, display_pdf_paths=display_paths)
    by_pdf = {str(item["pdf"]): item for item in layout_payload["results"]}

    sampled_results: list[dict[str, object]] = []
    for entry in sampled_entries:
        payload = dict(by_pdf.get(str(entry.pdf_path), {}))
        payload.update(
            {
                "digest": entry.digest,
                "bucket": entry.bucket,
                "pages": entry.pages,
                "file_size": entry.file_size,
                "creator_tool": entry.creator_tool,
                "producer": entry.producer,
                "urlkey": entry.urlkey,
                "original": entry.original,
            }
        )
        sampled_results.append(payload)

    structure_values = [float(item["structure_confidence"]) for item in sampled_results if item.get("structure_confidence") is not None]
    layout_values = [float(item["layout_confidence"]) for item in sampled_results if item.get("layout_confidence") is not None]
    semantic_values = [float(item["semantic_confidence"]) for item in sampled_results if item.get("semantic_confidence") is not None]
    low_confidence_count = sum(1 for item in sampled_results if item.get("semantic_confidence_label") == "low")
    trust_limited_count = sum(1 for item in sampled_results if bool(item.get("trust_limited")))
    semantic_pass_count = sum(1 for item in sampled_results if bool(item.get("semantic_pass")))
    specific_document_count = sum(1 for item in sampled_results if _is_specific_document_type(item.get("document_type")))
    specific_purpose_count = sum(1 for item in sampled_results if _is_specific_document_purpose(item.get("document_purpose")))
    generic_warning_count = sum(
        1
        for item in sampled_results
        if any(
            warning in {"generic_document_type", "generic_document_purpose", "generic_audience"}
            for warning in item.get("semantic_warnings", [])
        )
    )
    summary = {
        "technical_pass_rate": _rate(
            sum(1 for item in sampled_results if bool(item.get("all_pass"))),
            len(sampled_results),
        ),
        "semantic_pass_rate": _rate(semantic_pass_count, len(sampled_results)),
        "avg_structure_confidence": _avg(structure_values),
        "avg_layout_confidence": _avg(layout_values),
        "avg_semantic_confidence": _avg(semantic_values),
        "specific_document_rate": _rate(specific_document_count, len(sampled_results)),
        "specific_purpose_rate": _rate(specific_purpose_count, len(sampled_results)),
        "low_confidence_rate": _rate(low_confidence_count, len(sampled_results)),
        "trust_limited_rate": _rate(trust_limited_count, len(sampled_results)),
        "bucket_counts": _count_values([str(item.get("bucket", "")) for item in sampled_results]),
        "document_type_counts": _count_values([str(item.get("document_type", "")) for item in sampled_results]),
        "document_purpose_counts": _count_values([str(item.get("document_purpose", "")) for item in sampled_results]),
        "classification_status_counts": _count_values([str(item.get("classification_status", "")) for item in sampled_results]),
        "trust_policy_counts": _count_values([str(item.get("trust_policy", "")) for item in sampled_results]),
        "semantic_confidence_label_counts": _count_values(
            [str(item.get("semantic_confidence_label", "")) for item in sampled_results]
        ),
        "generic_warning_count": generic_warning_count,
    }
    bucket_diagnostics = _corpus_bucket_diagnostics(sampled_results)
    follow_up_actions = _corpus_follow_up_actions(bucket_diagnostics)

    processing_failed = [
        str(item.get("pdf"))
        for item in sampled_results
        if not bool(item.get("all_pass"))
    ]
    semantic_failed = [
        str(item.get("pdf"))
        for item in sampled_results
        if not bool(item.get("semantic_pass"))
    ]
    trust_failed = [
        str(item.get("pdf"))
        for item in sampled_results
        if bool(item.get("trust_limited"))
    ]
    layer_summary = {
        "processing": {
            "sample_count": len(sampled_results),
            "pass_rate": summary["technical_pass_rate"],
            "technical_pass_rate": summary["technical_pass_rate"],
            "avg_structure_confidence": summary["avg_structure_confidence"],
            "avg_layout_confidence": summary["avg_layout_confidence"],
            "failing_pdf_count": len(processing_failed),
            "failing_pdfs": processing_failed,
        },
        "semantics": {
            "sample_count": len(sampled_results),
            "pass_rate": summary["semantic_pass_rate"],
            "semantic_pass_rate": summary["semantic_pass_rate"],
            "specific_document_rate": summary["specific_document_rate"],
            "specific_purpose_rate": summary["specific_purpose_rate"],
            "failing_pdf_count": len(semantic_failed),
            "failing_pdfs": semantic_failed,
        },
        "trust": {
            "sample_count": len(sampled_results),
            "low_confidence_rate": summary["low_confidence_rate"],
            "trust_limited_rate": summary["trust_limited_rate"],
            "generic_warning_count": summary["generic_warning_count"],
            "failing_pdf_count": len(trust_failed),
            "failing_pdfs": trust_failed,
        },
    }

    layer_stability_checks: dict[str, object] = {}
    failed_layers: list[str] = []
    for layer_name, thresholds in CORPUS_LAYER_THRESHOLDS.items():
        current = layer_summary[layer_name]
        failed_metrics: dict[str, dict[str, float]] = {}
        if layer_name == "trust":
            for metric_name, max_value in thresholds.items():
                actual_key = metric_name.replace("max_", "")
                actual_value = float(current.get(actual_key, 0.0) or 0.0)
                if actual_value > max_value:
                    failed_metrics[actual_key] = {
                        "actual": actual_value,
                        "required_max": max_value,
                    }
        else:
            for metric_name, min_value in thresholds.items():
                actual_value = float(current.get(metric_name, 0.0) or 0.0)
                if actual_value < min_value:
                    failed_metrics[metric_name] = {
                        "actual": actual_value,
                        "required_min": min_value,
                    }
        passed = not failed_metrics
        layer_stability_checks[layer_name] = {
            "pass": passed,
            "thresholds": thresholds,
            "failed_metrics": failed_metrics,
        }
        if not passed:
            failed_layers.append(layer_name)
    layer_stability = {
        "all_pass": not failed_layers,
        "failed_layers": failed_layers,
        "checks": layer_stability_checks,
    }
    bucket_gate_pass = not any(str(action.get("priority")) == "high" for action in follow_up_actions)

    corpus_architecture_gates = {
        "all_pass": bool(layer_stability["all_pass"]) and bool(layout_payload["all_pass"]) and bucket_gate_pass,
        "technical_gate_pass": bool(layout_payload["all_pass"]),
        "layer_stability_pass": bool(layer_stability["all_pass"]),
        "semantic_gate_pass": bool(sampled_results) and semantic_pass_count == len(sampled_results),
        "bucket_gate_pass": bucket_gate_pass,
        "bucket_follow_up_count": len(follow_up_actions),
        "reasons": [
            *([] if layout_payload["all_pass"] else ["technical corpus smoke failures present"]),
            *([] if layer_stability["all_pass"] else ["corpus layer thresholds not met"]),
            *([] if sampled_results and semantic_pass_count == len(sampled_results) else ["not every sampled PDF reached semantic pass"]),
            *([] if bucket_gate_pass else ["high-priority bucket-level follow-up actions are present"]),
        ],
    }
    corpus_paths = _local_pdf_corpus_paths(corpus_dir)
    contract_gate = _corpus_contract_checks(
        bucket_diagnostics=bucket_diagnostics,
        follow_up_actions=follow_up_actions,
        architecture_gates=corpus_architecture_gates,
    )
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_paths[0]) if corpus_paths else (str(corpus_dir) if corpus_dir else None),
        "metadata_path": str(corpus_paths[1]) if corpus_paths else None,
        "corpus_pdf_count": len(corpus_entries),
        "sample_profile": sample_profile,
        "requested_sample_size": sample_size,
        "sample_size": len(sampled_entries),
        "sample_manifest": sample_manifest,
        "results": sampled_results,
        "summary": summary,
        "bucket_diagnostics": bucket_diagnostics,
        "follow_up_actions": follow_up_actions,
        "contract_gate": contract_gate,
        "layer_summary": layer_summary,
        "layer_stability": layer_stability,
        "architecture_gates": corpus_architecture_gates,
        "technical_all_pass": bool(layout_payload["all_pass"]),
        "semantic_all_pass": bool(sampled_results) and semantic_pass_count == len(sampled_results),
        "all_pass": bool(layout_payload["all_pass"]),
    }
    if save_snapshot:
        snapshot_path = _write_corpus_sanity_snapshot(payload)
        payload["snapshot_path"] = str(snapshot_path)
        snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_public_surface_unittests() -> dict[str, object]:
    project_root = _discover_project_root(PATHS.root) or _discover_project_root(Path.cwd())
    if project_root is None:
        return {
            "returncode": None,
            "passed": False,
            "skipped": True,
            "reason": "project_root_not_available",
            "output_tail": "",
        }
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_cli_public_surface.py",
        ],
        cwd=project_root,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    output_text = "\n".join(
        part.strip()
        for part in (process.stdout, process.stderr)
        if part and part.strip()
    )
    return {
        "returncode": process.returncode,
        "passed": process.returncode == 0,
        "skipped": False,
        "output_tail": output_text[-1200:],
    }


def _run_package_check() -> dict[str, object]:
    project_root = _discover_project_root(PATHS.root) or _discover_project_root(Path.cwd())
    if project_root is None:
        return {
            "all_pass": False,
            "workspace": None,
            "wheel_path": None,
            "venv_path": None,
            "script_path": None,
            "build_returncode": None,
            "venv_returncode": None,
            "install_returncode": None,
            "doctor_returncode": None,
            "smoke_returncode": None,
            "doctor_ok": False,
            "smoke_ok": False,
            "smoke_all_pass": False,
            "runtime_returncode": None,
            "runtime_ok": False,
            "readme_flow": {
                "all_pass": False,
                "public_path": True,
                "maintainer_benchmark_path": False,
                "steps": [],
                "reason": "project_root_not_available",
            },
            "skipped": True,
            "reason": "project_root_not_available",
            "build_output_tail": "",
            "install_output_tail": "",
        }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        workspace = Path(temp_dir_name)
        wheel_dir = workspace / "wheelhouse"
        venv_dir = workspace / "venv"
        data_dir = workspace / "data"
        wheel_dir.mkdir(parents=True, exist_ok=True)

        build_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        wheels = sorted(wheel_dir.glob("*.whl"))
        wheel_path = wheels[0] if wheels else None

        venv_process = None
        install_process = None
        venv_python = venv_dir / "bin" / "python"
        script_path = venv_dir / "bin" / "pdf-to-json-rag"
        readme_flow: dict[str, object] = {
            "all_pass": False,
            "public_path": True,
            "maintainer_benchmark_path": False,
            "steps": [],
        }

        if build_process.returncode == 0 and wheel_path is not None:
            venv_process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(venv_dir),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if venv_process.returncode == 0 and venv_python.exists():
                install_process = subprocess.run(
                    [
                        str(venv_python),
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--force-reinstall",
                        str(wheel_path),
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                )

            if install_process is not None and install_process.returncode == 0 and script_path.exists():
                readme_flow = _run_installed_readme_flow(script_path, workspace, data_dir)

        all_pass = (
            build_process.returncode == 0
            and wheel_path is not None
            and venv_process is not None
            and venv_process.returncode == 0
            and install_process is not None
            and install_process.returncode == 0
            and script_path.exists()
            and bool(readme_flow.get("all_pass"))
        )

        return {
            "all_pass": all_pass,
            "workspace": str(workspace),
            "wheel_path": str(wheel_path) if wheel_path else None,
            "venv_path": str(venv_dir),
            "script_path": str(script_path),
            "build_returncode": build_process.returncode,
            "venv_returncode": venv_process.returncode if venv_process else None,
            "install_returncode": install_process.returncode if install_process else None,
            "doctor_returncode": readme_flow.get("doctor_returncode"),
            "smoke_returncode": readme_flow.get("smoke_returncode"),
            "runtime_returncode": readme_flow.get("runtime_returncode"),
            "doctor_ok": bool(readme_flow.get("doctor_ok")),
            "smoke_ok": bool(readme_flow.get("smoke_ok")),
            "smoke_all_pass": bool(readme_flow.get("smoke_all_pass")),
            "runtime_ok": bool(readme_flow.get("runtime_ok")),
            "readme_flow": readme_flow,
            "skipped": False,
            "build_output_tail": _process_output_tail(build_process),
            "install_output_tail": _process_output_tail(install_process),
        }


RELEASE_CHECK_SHARDS = [
    "query_planning_core",
    "answer_modes_core",
    "document_pipeline_core",
    "structure_chunking_core",
    "section_reconstruction_core",
    "document_selection_core",
    "retrieval_contract_core",
    "retrieval_synthesis_core",
    "semantic_document_understanding_core",
    "confidence_aware_document_core",
    "trust_policy_document_core",
    "document_maintenance_core",
    "structured_form_maintenance_core",
    "processing_layer_core",
    "processing_strategy_core",
    "layout_robustness_core",
    "single_doc_random_pdf_core",
    "table_layout_robustness_core",
    "form_layout_robustness_core",
    "evidence_anchor_core",
    "source_anchor_contract_core",
    "document_family_core",
    "inventory_coverage_core",
    "relationship_core",
]
CORPUS_LAYER_THRESHOLDS: dict[str, dict[str, float]] = {
    "processing": {
        "technical_pass_rate": 1.0,
        "avg_structure_confidence": 0.55,
        "avg_layout_confidence": 0.55,
    },
    "semantics": {
        "semantic_pass_rate": 0.66,
        "specific_document_rate": 0.66,
        "specific_purpose_rate": 0.66,
    },
    "trust": {
        "max_low_confidence_rate": 0.34,
        "max_trust_limited_rate": 0.34,
    },
}


def _run_release_check(k: int) -> dict[str, object]:
    doctor = _doctor_checks()
    public_smoke = _run_public_surface_release_smoke()
    public_unittests = _run_public_surface_unittests()
    package_check = _run_package_check()
    maintainer_root = _discover_project_root(PATHS.root) or _discover_project_root(Path.cwd())

    regressions: list[dict[str, object]] = []
    inventory = load_document_inventory()
    benchmark_eval_path = PATHS.data_eval / DEFAULT_EVAL_FILENAME
    root_manifest_path = PATHS.data_index / "index_manifest.json"
    benchmark_assets_available = (
        root_manifest_path.exists()
        and benchmark_eval_path.exists()
        and len(inventory) >= 5
    )
    local_corpus_paths = _local_pdf_corpus_paths(None)
    local_corpus_available = bool(local_corpus_paths)
    local_corpus_sanity = (
        _run_corpus_sanity_check(
            sample_size=CORPUS_SAMPLE_PROFILES["quick"],
            corpus_dir=local_corpus_paths[0],
            k=k,
            sample_profile="quick",
            save_snapshot=True,
        )
        if local_corpus_available
        else None
    )
    regression_all_pass = True

    if benchmark_assets_available:
        eval_path = benchmark_eval_path
        for shard in RELEASE_CHECK_SHARDS:
            report, report_path = run_regression_suite(
                index_dir=PATHS.data_index,
                chunk_root=PATHS.data_chunks,
                eval_dir=PATHS.data_eval,
                k=k,
                eval_path=eval_path,
                shard=shard,
            )
            regressions.append(
                {
                    "shard": shard,
                    "all_pass": report["all_pass"],
                    "pass_count": report["pass_count"],
                    "fail_count": report["fail_count"],
                    "failed_case_ids": report["failed_case_ids"],
                    "report_path": str(report_path),
                }
            )
        regression_all_pass = all(item["all_pass"] for item in regressions)
    else:
        regression_all_pass = False

    public_surface_all_pass = (
        public_smoke["init_ok"]
        and public_smoke["smoke_ok"]
        and public_smoke["smoke_all_pass"]
        and doctor["ready_for_public_cli"]
    )

    maintainer_checks_available = maintainer_root is not None
    maintainer_surface_all_pass = (
        maintainer_checks_available
        and not package_check.get("skipped", False)
        and bool(package_check["all_pass"])
        and not public_unittests.get("skipped", False)
        and bool(public_unittests["passed"])
    )
    maintainer_release_all_pass = maintainer_surface_all_pass and (
        regression_all_pass if benchmark_assets_available else True
    )

    overall_pass = public_surface_all_pass and (
        maintainer_release_all_pass if maintainer_checks_available else True
    )
    recommendation = _release_channel_recommendation(
        overall_pass,
        public_surface_all_pass=public_surface_all_pass,
        maintainer_checks_available=maintainer_checks_available,
        maintainer_surface_all_pass=maintainer_surface_all_pass,
        benchmark_assets_available=benchmark_assets_available,
        regression_all_pass=regression_all_pass,
    )
    if local_corpus_sanity is not None and not bool(local_corpus_sanity.get("architecture_gates", {}).get("all_pass")):
        recommendation["why"].append("local unknown-document corpus gate is failing or trust-limited")

    return {
        "doctor": doctor,
        "public_surface": {
            "all_pass": public_surface_all_pass,
            "smoke": public_smoke,
        },
        "maintainer_checks": {
            "available": maintainer_checks_available,
            "project_root": str(maintainer_root) if maintainer_root is not None else None,
            "all_pass": maintainer_release_all_pass if maintainer_checks_available else None,
            "unittests": public_unittests,
            "package_check": package_check,
        },
        "internal_regressions": {
            "benchmark_assets_available": benchmark_assets_available,
            "benchmark_asset_details": {
                "inventory_count": len(inventory),
                "required_min_inventory_count": 5,
                "root_manifest_path": str(root_manifest_path),
                "root_manifest_present": root_manifest_path.exists(),
                "eval_file_path": str(benchmark_eval_path),
                "eval_file_present": benchmark_eval_path.exists(),
            },
            "selected_shards": RELEASE_CHECK_SHARDS,
            "skipped": not benchmark_assets_available,
            "all_pass": regression_all_pass if benchmark_assets_available else None,
            "results": regressions,
        },
        "local_corpus_sanity": {
            "available": local_corpus_available,
            "result": local_corpus_sanity,
        },
        "overall_pass": overall_pass,
        "recommendation": recommendation,
    }


def _gate_record(name: str, passed: bool | None, *, skipped: bool = False, reason: str | None = None) -> dict[str, object]:
    if skipped:
        status = "skip"
    elif passed is True:
        status = "pass"
    else:
        status = "fail"
    return {
        "name": name,
        "status": status,
        "passed": passed,
        "skipped": skipped,
        "reason": reason,
    }


def _release_check_compact_payload(payload: dict[str, object]) -> dict[str, object]:
    doctor = payload.get("doctor", {})
    maintainer = payload.get("maintainer_checks", {})
    package_check = maintainer.get("package_check", {})
    unittests = maintainer.get("unittests", {})
    regressions = payload.get("internal_regressions", {})
    local_corpus = payload.get("local_corpus_sanity", {})
    local_corpus_result = local_corpus.get("result") if isinstance(local_corpus, dict) else None
    corpus_gates = (
        local_corpus_result.get("architecture_gates", {})
        if isinstance(local_corpus_result, dict)
        else {}
    )
    runtime = doctor.get("runtime", {}) if isinstance(doctor, dict) else {}
    regression_results = regressions.get("results", []) if isinstance(regressions, dict) else []
    shard_records = [
        {
            **_gate_record(str(item.get("shard")), bool(item.get("all_pass"))),
            "pass_count": item.get("pass_count"),
            "fail_count": item.get("fail_count"),
            "failed_case_ids": item.get("failed_case_ids", []),
        }
        for item in regression_results
    ]
    return {
        "overall": _gate_record("overall", bool(payload.get("overall_pass"))),
        "recommendation": payload.get("recommendation", {}),
        "runtime_decision": runtime.get("runtime_decision", {}),
        "public_path": {
            "all_pass": bool(payload.get("public_surface", {}).get("all_pass")),
            "gates": [
                _gate_record("doctor_public_cli", bool(doctor.get("ready_for_public_cli"))),
                _gate_record(
                    "public_smoke",
                    bool(payload.get("public_surface", {}).get("smoke", {}).get("smoke_all_pass")),
                ),
            ],
        },
        "maintainer_path": {
            "available": bool(maintainer.get("available")),
            "all_pass": maintainer.get("all_pass"),
            "gates": [
                _gate_record(
                    "package_check",
                    bool(package_check.get("all_pass")) if package_check else None,
                    skipped=bool(package_check.get("skipped")) if package_check else True,
                    reason=package_check.get("reason") if isinstance(package_check, dict) else "package_check_missing",
                ),
                _gate_record(
                    "unit_tests",
                    bool(unittests.get("passed")) if unittests else None,
                    skipped=bool(unittests.get("skipped")) if unittests else True,
                    reason=unittests.get("reason") if isinstance(unittests, dict) else "unit_tests_missing",
                ),
                _gate_record(
                    "internal_regressions",
                    bool(regressions.get("all_pass")) if not regressions.get("skipped") else None,
                    skipped=bool(regressions.get("skipped")),
                    reason=(
                        "benchmark assets not present in the active data root"
                        if regressions.get("skipped")
                        else None
                    ),
                ),
            ],
        },
        "internal_regressions": {
            "benchmark_assets_available": regressions.get("benchmark_assets_available"),
            "selected_shard_count": len(regressions.get("selected_shards", [])),
            "all_pass": regressions.get("all_pass"),
            "shards": shard_records,
        },
        "local_corpus_sanity": {
            "available": local_corpus.get("available") if isinstance(local_corpus, dict) else False,
            "gate": _gate_record(
                "local_corpus_architecture",
                bool(corpus_gates.get("all_pass")) if corpus_gates else None,
                skipped=not bool(local_corpus.get("available")) if isinstance(local_corpus, dict) else True,
                reason=(
                    None
                    if corpus_gates
                    else "repo-local pdf/ corpus is not available or was not sampled"
                ),
            ),
            "sample_manifest": (
                local_corpus_result.get("sample_manifest", {})
                if isinstance(local_corpus_result, dict)
                else {}
            ),
            "follow_up_count": (
                len(local_corpus_result.get("follow_up_actions", []))
                if isinstance(local_corpus_result, dict)
                else None
            ),
        },
    }


def _public_beta_check_compact_payload(release_payload: dict[str, object]) -> dict[str, object]:
    release_summary = _release_check_compact_payload(release_payload)
    package_check = (
        release_payload.get("maintainer_checks", {}).get("package_check", {})
        if isinstance(release_payload.get("maintainer_checks"), dict)
        else {}
    )
    readme_flow = package_check.get("readme_flow", {}) if isinstance(package_check, dict) else {}
    runtime_decision = release_summary.get("runtime_decision", {})
    corpus_summary = release_summary.get("local_corpus_sanity", {})
    gates = [
        _gate_record(
            "installed_readme_flow",
            bool(readme_flow.get("all_pass")) if readme_flow else None,
            skipped=not bool(readme_flow),
            reason=None if readme_flow else "installed README flow was not available",
        ),
        _gate_record(
            "runtime_default_policy",
            runtime_decision.get("default_backend") == "hash",
            reason=runtime_decision.get("not_default_reason"),
        ),
        corpus_summary.get("gate", _gate_record("local_corpus_architecture", None, skipped=True)),
        release_summary.get("overall", _gate_record("release_summary", None, skipped=True)),
    ]
    return {
        "all_pass": all(gate.get("status") == "pass" for gate in gates),
        "gates": gates,
        "public_readme_flow": {
            "all_pass": bool(readme_flow.get("all_pass")),
            "steps": [
                {
                    "name": step.get("name"),
                    "status": "pass" if step.get("ok") else "fail",
                    "returncode": step.get("returncode"),
                }
                for step in readme_flow.get("steps", [])
            ],
        },
        "runtime_decision": runtime_decision,
        "corpus_quick": corpus_summary,
        "release_summary": release_summary,
        "scope": {
            "default_backend": "hash",
            "sentence_transformers": "recommended_opt_in_only",
            "cross_encoder": "experimental_opt_in_only",
            "llm_synthesis": "opt_in_only",
        },
    }


def _run_public_beta_check(k: int) -> dict[str, object]:
    return _public_beta_check_compact_payload(_run_release_check(k))


def _resolve_document_paths(doc_id: str) -> tuple[Path, Path]:
    native_path = PATHS.data_documents / f"{doc_id}.native.json"
    document_path = PATHS.data_documents / f"{doc_id}.document.json"
    missing: list[str] = []
    if not native_path.exists():
        missing.append(str(native_path))
    if not document_path.exists():
        missing.append(str(document_path))
    if missing:
        raise CliError(
            "missing_document_artifacts",
            f"Missing saved extraction artifacts for doc_id '{doc_id}'",
            {"doc_id": doc_id, "missing_paths": missing},
        )
    return native_path, document_path


def _existing_chunk_doc_ids() -> list[str]:
    if not PATHS.data_chunks.exists():
        return []
    return sorted(path.name for path in PATHS.data_chunks.iterdir() if path.is_dir())


def _load_doc_ids_with_chunks(doc_id_arg: str | None) -> list[str]:
    if doc_id_arg:
        return [item.strip() for item in doc_id_arg.split(",") if item.strip()]
    doc_ids = _existing_chunk_doc_ids()
    if not doc_ids:
        raise CliError(
            "missing_chunks",
            "No chunk directories were found. Run chunk-document first.",
            {"chunk_root": str(PATHS.data_chunks)},
        )
    return doc_ids


def _validate_index_dir(index_dir: Path) -> None:
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise CliError(
            "missing_index",
            "Index manifest was not found. Run build-index first or pass --index-dir.",
            {"index_dir": str(index_dir), "expected_manifest": str(manifest_path)},
        )


def _smoke_checks(payload: dict[str, object]) -> list[dict[str, object]]:
    answer = payload["answer"]
    checks = [
        {
            "name": "doc_id_present",
            "passed": bool(payload.get("doc_id")),
        },
        {
            "name": "chunks_created",
            "passed": bool(payload["index"].get("chunk_count", 0) > 0),
        },
        {
            "name": "inventory_summary_present",
            "passed": bool(payload["document"].get("inventory_summary")),
        },
        {
            "name": "semantic_confidence_present",
            "passed": payload["document"].get("semantic_confidence") is not None,
        },
        {
            "name": "plan_classified",
            "passed": bool(payload["plan"].get("query_class") and payload["plan"].get("answer_mode")),
        },
        {
            "name": "answer_present",
            "passed": bool(answer.get("answer")),
        },
        {
            "name": "evidence_or_document_answer",
            "passed": bool(answer.get("evidence") or answer.get("answer_trace", {}).get("answer_mode") != "grounded_evidence"),
        },
        {
            "name": "answer_contract_health_present",
            "passed": bool(answer.get("contract_health")),
        },
        {
            "name": "quality_profile_present",
            "passed": bool(payload.get("quality_profile")),
        },
    ]
    return checks


def _canonical_command(command: str) -> str:
    return COMMAND_ALIASES.get(command, command)


def _discover_project_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _packaged_examples_dir() -> Path:
    package_root = importlib_resources.files("pdf_to_json_rag")
    return Path(str(package_root / "assets" / "examples"))


def _project_examples_dir() -> Path | None:
    project_root = _discover_project_root(PATHS.root)
    if project_root is None:
        project_root = _discover_project_root(Path.cwd())
    if project_root is None:
        return None
    examples_dir = project_root / "examples"
    if not examples_dir.exists():
        return None
    if not all((examples_dir / name).exists() for name in EXPECTED_EXAMPLE_FILES):
        return None
    return examples_dir


def _available_examples_dir() -> Path:
    project_examples = _project_examples_dir()
    if project_examples is not None and project_examples.exists():
        return project_examples
    return _packaged_examples_dir()


def _project_metadata_available() -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {}
    project_examples = _project_examples_dir()
    if project_examples is not None:
        pyproject_path = project_examples.parent / "pyproject.toml"
        if pyproject_path.exists():
            details["pyproject_path"] = str(pyproject_path)
            return True, details
    try:
        installed_version = importlib_metadata.version("pdf-to-json-rag")
    except importlib_metadata.PackageNotFoundError:
        return False, details
    details["installed_version"] = installed_version
    return True, details


def _load_example_json(filename: str) -> dict[str, object] | list[object]:
    path = _available_examples_dir() / filename
    if not path.exists():
        raise CliError(
            "missing_example_asset",
            f"Example asset was not found: {path}",
            {"filename": filename, "path": str(path)},
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _render_help(topic: str | None = None) -> str:
    if not topic:
        lines = ["Available commands:"]
        for command in CANONICAL_COMMANDS:
            summary = str(COMMAND_HELP[command]["summary"])
            lines.append(f"- {command}: {summary}")
        lines.append("")
        lines.append("Run `pdf-to-json-rag help --topic <command>` for a concrete example.")
        return "\n".join(lines)

    command = _canonical_command(topic)
    spec = COMMAND_HELP.get(command)
    if not spec:
        raise CliError(
            "unknown_help_topic",
            f"Unknown help topic: {topic}",
            {"topic": topic, "known_commands": CANONICAL_COMMANDS},
        )
    lines = [command, str(spec["summary"])]
    aliases = sorted(alias for alias, target in COMMAND_ALIASES.items() if target == command)
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")
    lines.append(f"Example: {spec['example']}")
    return "\n".join(lines)


def _runtime_promotion_snapshot_status() -> dict[str, object]:
    snapshot_path = PATHS.data_eval / DEFAULT_RUNTIME_PROMOTION_SNAPSHOT_FILENAME
    if not snapshot_path.exists():
        return {
            "available": False,
            "path": str(snapshot_path),
            "promotion_ready": False,
            "candidate_mode": None,
        }
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "available": True,
            "path": str(snapshot_path),
            "promotion_ready": False,
            "candidate_mode": None,
            "reason": "snapshot_json_invalid",
        }
    gate = snapshot.get("promotion_gate", {})
    return {
        "available": True,
        "path": str(snapshot_path),
        "promotion_ready": bool(gate.get("promotable")),
        "candidate_mode": snapshot.get("candidate_mode"),
        "case_count": snapshot.get("case_count"),
        "candidate_pass_count": snapshot.get("candidate_pass_count"),
        "recommended_default_change": bool(snapshot.get("recommended_default_change")),
    }


def _runtime_decision_payload(embedding: dict[str, object]) -> dict[str, object]:
    promotion = _runtime_promotion_snapshot_status()
    recommended_backend = (
        "sentence-transformers"
        if promotion.get("promotion_ready") and promotion.get("candidate_mode") == "sentence-transformers"
        else None
    )
    return {
        "default_backend": "hash",
        "effective_backend": embedding.get("effective_backend"),
        "requested_backend": embedding.get("requested_backend"),
        "recommended_opt_in_backend": recommended_backend,
        "recommended_opt_in_source": "runtime_promotion_snapshot" if recommended_backend else None,
        "promotion_snapshot": promotion,
        "not_default_reason": (
            "The deterministic hash backend remains the public default because it is offline-safe, reproducible, "
            "and the sentence-transformer backend has only been promoted as an explicit opt-in path."
        ),
        "cross_encoder_default": "disabled",
        "llm_synthesis_default": "disabled",
    }


def _runtime_check_payload() -> dict[str, object]:
    embedding = embedding_runtime_diagnostics()
    return {
        "install_context": {
            "version": __version__,
            "python": sys.executable,
            "module_path": str(Path(__file__).resolve()),
            "project_root": str((_discover_project_root(PATHS.root) or _discover_project_root(Path.cwd()) or PATHS.root)),
        },
        "embedding": embedding,
        "runtime_decision": _runtime_decision_payload(embedding),
        "llm_synthesis": {
            "env_var": "PDF_TO_JSON_RAG_LLM_COMMAND",
            "configured": bool(os.environ.get("PDF_TO_JSON_RAG_LLM_COMMAND")),
            "provider": "local_command" if os.environ.get("PDF_TO_JSON_RAG_LLM_COMMAND") else None,
            "default_enabled": False,
        },
        "cross_encoder": {
            "env_var": "PDF_TO_JSON_RAG_USE_CROSS_ENCODER",
            "configured": os.environ.get("PDF_TO_JSON_RAG_USE_CROSS_ENCODER") == "1",
            "default_enabled": False,
        },
        "default_policy": {
            "embedding_backend": "hash",
            "sentence_transformers": "opt_in",
            "cross_encoder": "opt_in",
            "llm_synthesis": "opt_in",
        },
    }


def _embedding_manifest_payload(manifest: dict[str, object]) -> dict[str, object]:
    runtime = embedding_runtime_diagnostics()
    return {
        "requested_backend": manifest.get("embedding_requested_backend")
        or runtime["requested_backend"],
        "effective_backend": manifest.get("embedding_backend"),
        "effective_model": manifest.get("embedding_model"),
        "fallback_reason": manifest.get("embedding_fallback_reason"),
        "runtime_check": runtime,
    }


def _portable_snapshot_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _portable_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_snapshot_value(item) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
        except (OSError, ValueError):
            return value
        if path.is_absolute():
            try:
                return path.resolve().relative_to(PATHS.root.resolve()).as_posix()
            except ValueError:
                return path.name
    return value


def _write_runtime_promotion_snapshot(report: dict[str, object], report_path: Path) -> Path | None:
    gate = report.get("promotion_gates", {}).get("sentence-transformers", {})
    if not (report.get("all_cases") and gate.get("promotable")):
        return None
    mode_results = {item.get("mode"): item for item in report.get("mode_results", [])}
    baseline = mode_results.get("baseline", {})
    candidate = mode_results.get("sentence-transformers", {})
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_path": _portable_snapshot_value(str(report_path)),
        "case_count": report.get("case_count", 0),
        "baseline_pass_count": baseline.get("pass_count"),
        "candidate_mode": "sentence-transformers",
        "candidate_pass_count": candidate.get("pass_count"),
        "candidate_index_manifest": _portable_snapshot_value(candidate.get("index_manifest", {})),
        "baseline_deltas": report.get("baseline_deltas", {}).get("sentence-transformers", {}),
        "promotion_gate": _portable_snapshot_value(gate),
        "recommended_default_change": False,
        "recommendation": "Sentence-transformers is validated as an opt-in backend; keep hash as default until an explicit default-change decision is made.",
    }
    snapshot_path = PATHS.data_eval / DEFAULT_RUNTIME_PROMOTION_SNAPSHOT_FILENAME
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_path


def _runtime_promotion_report_payload(report_path: Path | None = None) -> dict[str, object]:
    path = report_path or (PATHS.data_eval / DEFAULT_RUNTIME_COMPARISON_REPORT_FILENAME)
    path = path.expanduser().resolve()
    if not path.exists():
        return {
            "available": False,
            "report_path": str(path),
            "promotion_ready": False,
            "recommendation": "Run `pdf-to-json-rag compare-runtime-modes --modes baseline,sentence-transformers --all-cases --json` first.",
        }

    report = json.loads(path.read_text(encoding="utf-8"))
    mode_results = {item.get("mode"): item for item in report.get("mode_results", [])}
    baseline = mode_results.get("baseline", {})
    candidate = mode_results.get("sentence-transformers", {})
    gate = report.get("promotion_gates", {}).get("sentence-transformers", {})
    promotion_ready = bool(gate.get("promotable"))
    recommendation = (
        "Optional sentence-transformer embeddings are promotion-ready; default remains hash until an explicit default-change decision is made."
        if promotion_ready
        else "Keep sentence-transformers opt-in until the promotion gate is green on the full suite."
    )
    snapshot_path = _write_runtime_promotion_snapshot(report, path)
    return {
        "available": True,
        "report_path": str(path),
        "all_cases": bool(report.get("all_cases")),
        "case_count": report.get("case_count", 0),
        "baseline": {
            "pass_count": baseline.get("pass_count"),
            "fail_count": baseline.get("fail_count"),
            "summary": baseline.get("summary", {}),
            "index_manifest": baseline.get("index_manifest", {}),
        },
        "candidate": {
            "mode": "sentence-transformers",
            "pass_count": candidate.get("pass_count"),
            "fail_count": candidate.get("fail_count"),
            "summary": candidate.get("summary", {}),
            "index_manifest": candidate.get("index_manifest", {}),
        },
        "deltas": report.get("baseline_deltas", {}).get("sentence-transformers", {}),
        "promotion_gate": gate,
        "promotion_ready": promotion_ready,
        "promotion_snapshot_path": str(snapshot_path) if snapshot_path else None,
        "recommendation": recommendation,
    }


def _doctor_checks() -> dict[str, object]:
    package_metadata_present, package_metadata_details = _project_metadata_available()
    examples_dir = _available_examples_dir()
    runtime = _runtime_check_payload()
    manifest_candidates = [
        PATHS.data_index / "index_manifest.json",
        PATHS.data_index / "workflow_smoke" / "index_manifest.json",
    ]
    selected_manifest = next((path for path in manifest_candidates if path.exists()), manifest_candidates[0])
    inventory = load_document_inventory()
    checks: list[dict[str, object]] = [
        {
            "name": "package_metadata_present",
            "passed": package_metadata_present,
            "category": "required_public_tool",
            "details": package_metadata_details,
        },
        {
            "name": "data_root_configured",
            "passed": bool(PATHS.data_dir),
            "category": "required_public_tool",
            "details": {"data_dir": str(PATHS.data_dir)},
        },
        {
            "name": "data_dirs_exist",
            "passed": all(path.exists() for path in (PATHS.data_input, PATHS.data_documents, PATHS.data_chunks, PATHS.data_index, PATHS.data_eval)),
            "category": "required_public_tool",
            "details": {
                "data_input": str(PATHS.data_input),
                "data_documents": str(PATHS.data_documents),
                "data_chunks": str(PATHS.data_chunks),
                "data_index": str(PATHS.data_index),
                "data_eval": str(PATHS.data_eval),
            },
        },
        {
            "name": "tesseract_available",
            "passed": shutil.which("tesseract") is not None,
            "category": "optional_capability",
            "details": {"which": shutil.which("tesseract")},
        },
        {
            "name": "pdfplumber_available",
            "passed": importlib.util.find_spec("pdfplumber") is not None,
            "category": "optional_capability",
            "details": {
                "install": "python -m pip install 'pdf-to-json-rag[tables]'",
            },
        },
        {
            "name": "embedding_backend_configured",
            "passed": bool(runtime["embedding"].get("effective_backend")),
            "category": "optional_capability",
            "details": runtime["embedding"],
        },
        {
            "name": "example_assets_present",
            "passed": examples_dir.exists() and all((examples_dir / name).exists() for name in EXPECTED_EXAMPLE_FILES),
            "category": "required_public_tool",
            "details": {
                "examples_dir": str(examples_dir),
                "expected_files": list(EXPECTED_EXAMPLE_FILES),
            },
        },
        {
            "name": "demo_pdf_generation_available",
            "passed": True,
            "category": "required_public_tool",
            "details": {"engine": "PyMuPDF"},
        },
        {
            "name": "document_inventory_available",
            "passed": len(inventory) > 0,
            "category": "internal_benchmark",
            "details": {"document_count": len(inventory)},
        },
        {
            "name": "index_manifest_available",
            "passed": any(path.exists() for path in manifest_candidates),
            "category": "internal_benchmark",
            "details": {
                "manifest_path": str(selected_manifest),
                "manifest_candidates": [str(path) for path in manifest_candidates],
            },
        },
    ]
    ready_for_public_cli = all(
        check["passed"]
        for check in checks
        if check["category"] == "required_public_tool"
    )
    ready_for_retrieval = ready_for_public_cli and all(
        check["passed"]
        for check in checks
        if check["name"] in {"document_inventory_available", "index_manifest_available"}
    )
    ready_for_internal_benchmark = all(
        check["passed"]
        for check in checks
        if check["category"] in {"required_public_tool", "internal_benchmark"}
    )
    next_steps: list[str] = []
    if not ready_for_public_cli:
        next_steps.extend(
            [
                "Run `pdf-to-json-rag init --json` to create the local data directories.",
                "Re-run `pdf-to-json-rag doctor --json` after the required public-tool checks pass.",
            ]
        )
    elif not ready_for_retrieval:
        next_steps.extend(
            [
                "Run `pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json` for a self-contained sample input.",
                "Run `pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query \"What does this file cover?\" --json` to build a demo index and validate retrieval.",
            ]
        )
    else:
        next_steps.extend(
            [
                "Run `pdf-to-json-rag inspect-document --doc-id <doc_id> --json` to inspect extracted document metadata.",
                "Run `pdf-to-json-rag answer-query --query \"What does this file cover?\" --json` against the current local index.",
            ]
        )
    return {
        "checks": checks,
        "ready_for_public_cli": ready_for_public_cli,
        "ready_for_retrieval": ready_for_retrieval,
        "ready_for_internal_benchmark": ready_for_internal_benchmark,
        "data_root": str(PATHS.data_dir),
        "project_root": str((_discover_project_root(PATHS.root) or _discover_project_root(Path.cwd()) or PATHS.root)),
        "runtime": runtime,
        "next_steps": next_steps,
    }


def main() -> None:
    argv = sys.argv[1:]
    parser = CliArgumentParser(
        description="PDF-to-JSON RAG local-first CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=CLI_EPILOG,
    )
    parser.add_argument(
        "command",
        choices=sorted(set(CANONICAL_COMMANDS + list(COMMAND_ALIASES.keys()))),
        help="Command to run. Use `help` or `help --topic <command>` for focused guidance.",
    )
    parser.add_argument(
        "--pdf",
        help="Path to a local PDF file.",
    )
    parser.add_argument(
        "--pdfs",
        help="Comma-separated local PDF paths for layout-sanity-check.",
    )
    parser.add_argument(
        "--corpus-dir",
        help="Optional local corpus directory override for corpus-sanity-check.",
    )
    parser.add_argument(
        "--path",
        help="Optional output path for generated local assets such as create-demo-pdf.",
    )
    parser.add_argument(
        "--doc-id",
        "--doc-ids",
        dest="doc_id",
        help="Document ID or comma-separated document IDs, depending on the command.",
    )
    parser.add_argument(
        "--query",
        help="Natural-language query to plan, retrieve, or answer.",
    )
    parser.add_argument(
        "--k",
        "--top-k",
        type=int,
        dest="k",
        default=5,
        help="Number of retrieval hits to return.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override the number of local corpus PDFs to sample for corpus-sanity-check.",
    )
    parser.add_argument(
        "--sample-profile",
        "--profile",
        dest="sample_profile",
        choices=tuple(CORPUS_SAMPLE_PROFILES),
        default="balanced",
        help="Corpus sample profile for corpus-sanity-check: quick=4, balanced=12, stress=24.",
    )
    parser.add_argument(
        "--index-dir",
        help="Optional custom index directory.",
    )
    parser.add_argument(
        "--eval-file",
        help="Optional path to a custom evaluation JSON file.",
    )
    parser.add_argument(
        "--case-ids",
        help="Optional comma-separated case IDs for evaluate-regression.",
    )
    parser.add_argument(
        "--shard",
        help="Optional regression shard for evaluate-regression or compare-runtime-modes.",
    )
    parser.add_argument(
        "--modes",
        help="Optional comma-separated runtime modes for compare-runtime-modes.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Use every evaluation case for compare-runtime-modes instead of the default comparison subset.",
    )
    parser.add_argument(
        "--topic",
        help="Optional command topic for the `help` command.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        help="Optional output format. `json` is equivalent to `--json`.",
    )
    parser.add_argument(
        "--output",
        help="Optional file path for JSON output. Requires JSON output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include fuller debug payloads for planner and answer JSON output.",
    )
    try:
        args = parser.parse_args(argv)
        command = _canonical_command(args.command)
        output_path = _resolve_output_path(args.output)
        if args.json and args.format == "text":
            raise CliError(
                "conflicting_output_format",
                "--json cannot be combined with --format text",
                {"format": args.format},
            )
        json_output = args.json or args.format == "json"
        if output_path and not json_output:
            raise CliError(
                "output_requires_json",
                "--output can only be used together with JSON output",
                {"output": str(output_path)},
            )

        if command == "help":
            help_text = _render_help(args.topic)
            if json_output:
                _emit_json(
                    "help",
                    {
                        "topic": args.topic,
                        "help_text": help_text,
                    },
                    output_path=output_path,
                )
                return
            print(help_text)
            return

        if command == "demo-profile":
            payload = _load_example_json("public_demo_profile.json")
            if json_output:
                _emit_json("demo-profile", {"profile": payload}, output_path=output_path)
                return
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if command == "runtime-check":
            payload = _runtime_check_payload()
            if json_output:
                _emit_json("runtime-check", payload, output_path=output_path)
                return
            embedding = payload["embedding"]
            print(f"Requested embedding backend: {embedding.get('requested_backend')}")
            print(f"Effective embedding backend: {embedding.get('effective_backend')}")
            print(f"Effective embedding model: {embedding.get('effective_model')}")
            if embedding.get("fallback_reason"):
                print(f"Fallback reason: {embedding.get('fallback_reason')}")
            decision = payload["runtime_decision"]
            print(f"Default backend: {decision.get('default_backend')}")
            if decision.get("recommended_opt_in_backend"):
                print(f"Recommended opt-in backend: {decision.get('recommended_opt_in_backend')}")
            print(f"Not default reason: {decision.get('not_default_reason')}")
            print(f"Sentence-transformers package: {_human_status(embedding.get('sentence_transformers_package_available'))}")
            print(f"Sentence-transformers model cached: {_human_status(embedding.get('sentence_transformers_model_cached'))}")
            print(f"Cross-encoder opt-in configured: {_human_status(payload['cross_encoder']['configured'])}")
            print(f"LLM synthesis opt-in configured: {_human_status(payload['llm_synthesis']['configured'])}")
            return

        if command == "runtime-promotion-report":
            payload = _runtime_promotion_report_payload()
            if json_output:
                _emit_json("runtime-promotion-report", payload, output_path=output_path)
                return
            print(f"Runtime comparison report: {payload['report_path']}")
            if not payload["available"]:
                print("Available: no")
                print(payload["recommendation"])
                return
            print(f"Available: yes")
            print(f"Cases: {payload.get('case_count', 0)}")
            baseline = payload.get("baseline", {})
            candidate = payload.get("candidate", {})
            print(f"Baseline pass: {baseline.get('pass_count')}")
            print(f"Sentence-transformers pass: {candidate.get('pass_count')}")
            print(f"Promotion ready: {_human_status(payload.get('promotion_ready'))}")
            reasons = payload.get("promotion_gate", {}).get("reasons", [])
            if reasons:
                print(f"Promotion reasons: {', '.join(reasons)}")
            print(payload["recommendation"])
            return

        if command == "doctor":
            payload = _doctor_checks()
            if json_output:
                _emit_json("doctor", payload, output_path=output_path)
                return
            print(f"Public CLI readiness: {_human_status(payload['ready_for_public_cli'])}")
            print(f"Retrieval workflow readiness: {_human_status(payload['ready_for_retrieval'])}")
            print(f"Internal benchmark readiness: {_human_status(payload['ready_for_internal_benchmark'])}")
            grouped_checks = {
                "required_public_tool": "Required public-tool checks",
                "optional_capability": "Optional capabilities",
                "internal_benchmark": "Internal benchmark assets",
            }
            for category, label in grouped_checks.items():
                category_checks = [check for check in payload["checks"] if check["category"] == category]
                if not category_checks:
                    continue
                print("")
                print(f"{label}:")
                for check in category_checks:
                    print(f"- {_human_status(check['passed'])} {check['name']}")
            if payload["next_steps"]:
                print("")
                print("Suggested next steps:")
                for step in payload["next_steps"]:
                    print(f"- {step}")
            return

        if command == "create-demo-pdf":
            PATHS.ensure_dirs()
            output_demo_path = _resolve_optional_path(
                args.path,
                PATHS.data_input / "public_demo.pdf",
            )
            created_path = _create_demo_pdf(output_demo_path)
            payload = {
                "pdf": str(created_path),
                "suggested_queries": _load_example_json("public_demo_queries.json"),
            }
            if json_output:
                _emit_json("create-demo-pdf", payload, output_path=output_path)
                return
            print(f"Created demo PDF: {created_path}")
            print(
                "Next: run `pdf-to-json-rag smoke-check --pdf "
                f"{created_path} --query \"What does this file cover?\" --json`"
            )
            return

        if command == "package-check":
            PATHS.ensure_dirs()
            payload = _run_package_check()
            if json_output:
                _emit_json("package-check", payload, output_path=output_path)
                return
            print(f"Package check: {_human_status(payload['all_pass'])}")
            if payload["wheel_path"]:
                print(f"Built wheel: {payload['wheel_path']}")
            print(f"Installed CLI path: {payload['script_path']}")
            print(f"Build step: {_human_status(payload['build_returncode'] == 0)}")
            print(f"Install step: {_human_status(payload['install_returncode'] == 0)}")
            print(f"Packaged doctor: {_human_status(payload['doctor_ok'])}")
            print(f"Packaged smoke-check: {_human_status(payload['smoke_all_pass'])}")
            print(f"Packaged runtime-check: {_human_status(payload['runtime_ok'])}")
            print(f"Installed README flow: {_human_status(payload.get('readme_flow', {}).get('all_pass'))}")
            return

        if command == "readme-smoke-check":
            PATHS.ensure_dirs()
            package_payload = _run_package_check()
            payload = {
                "all_pass": bool(package_payload.get("all_pass")),
                "install": {
                    "wheel_path": package_payload.get("wheel_path"),
                    "venv_path": package_payload.get("venv_path"),
                    "script_path": package_payload.get("script_path"),
                    "build_returncode": package_payload.get("build_returncode"),
                    "venv_returncode": package_payload.get("venv_returncode"),
                    "install_returncode": package_payload.get("install_returncode"),
                    "build_output_tail": package_payload.get("build_output_tail"),
                    "install_output_tail": package_payload.get("install_output_tail"),
                },
                "public_readme_flow": package_payload.get("readme_flow", {}),
                "maintainer_benchmark_path": {
                    "included": False,
                    "reason": "readme-smoke-check validates only the public installed README flow; use release-check for maintainer benchmark regressions.",
                },
            }
            if json_output:
                _emit_json("readme-smoke-check", payload, output_path=output_path)
                return
            print(f"Installed README smoke: {_human_status(payload['all_pass'])}")
            print(f"Installed CLI path: {payload['install']['script_path']}")
            flow = payload.get("public_readme_flow", {})
            for step in flow.get("steps", []):
                print(f"- {_human_status(bool(step.get('ok')))} {step.get('name')}")
            return

        if command == "public-beta-check":
            PATHS.ensure_dirs()
            payload = _run_public_beta_check(args.k)
            if json_output:
                _emit_json("public-beta-check", payload, output_path=output_path)
                return
            print(f"Public beta check: {_human_status(payload['all_pass'])}")
            for gate in payload.get("gates", []):
                print(f"- {gate.get('status', 'unknown').upper()} {gate.get('name')}")
            runtime_decision = payload.get("runtime_decision", {})
            if runtime_decision:
                print(f"Default backend: {runtime_decision.get('default_backend')}")
                if runtime_decision.get("recommended_opt_in_backend"):
                    print(f"Recommended opt-in backend: {runtime_decision.get('recommended_opt_in_backend')}")
            return

        if command == "release-check":
            PATHS.ensure_dirs()
            payload = _run_release_check(args.k)
            if json_output:
                result_payload = payload if args.verbose else _release_check_compact_payload(payload)
                _emit_json("release-check", result_payload, output_path=output_path)
                return
            print(f"Release check: {_human_status(payload['overall_pass'])}")
            recommendation = payload["recommendation"]
            if recommendation["suggested_tag"]:
                print(f"Suggested release tag: {recommendation['suggested_tag']}")
            print(f"Public CLI surface: {_human_status(payload['public_surface']['all_pass'])}")
            maintainer_checks = payload["maintainer_checks"]
            if maintainer_checks["available"]:
                print(f"Packaged CLI gate: {_human_status(maintainer_checks['package_check']['all_pass'])}")
                print(f"Maintainer unit-test gate: {_human_status(maintainer_checks['unittests']['passed'])}")
                if payload["internal_regressions"]["skipped"]:
                    print("Maintainer regression gate: SKIPPED (benchmark assets not present in the active data root)")
                else:
                    print(f"Maintainer regression gate: {_human_status(bool(payload['internal_regressions']['all_pass']))}")
                local_corpus = payload.get("local_corpus_sanity", {})
                if local_corpus.get("available") and local_corpus.get("result"):
                    corpus_result = local_corpus["result"]
                    print(
                        "Local corpus gate: "
                        f"{_human_status(bool(corpus_result.get('architecture_gates', {}).get('all_pass')))}"
                    )
                    follow_up_count = len(corpus_result.get("follow_up_actions", []))
                    if follow_up_count:
                        print(f"Local corpus follow-up actions: {follow_up_count}")
            else:
                print("Maintainer gates: SKIPPED (run from a source checkout to include package and regression checks)")
            print("")
            print("Why:")
            for reason in recommendation["why"]:
                print(f"- {reason}")
            return

        if command == "layout-sanity-check":
            pdf_values = _require_arg(args.pdfs, "--pdfs", "layout-sanity-check")
            PATHS.ensure_dirs()
            pdf_paths = _resolve_pdf_paths(pdf_values)
            payload = _run_layout_sanity_check(pdf_paths, k=args.k)
            if json_output:
                _emit_json("layout-sanity-check", payload, output_path=output_path)
                return
            print(f"Layout sanity check: {_human_status(payload['all_pass'])}")
            for item in payload["results"]:
                print("")
                print(f"{Path(str(item['pdf'])).name}: {_human_status(bool(item.get('all_pass')))}")
                print(
                    f"  type={item.get('document_type')} | purpose={item.get('document_purpose')} | "
                    f"structure_confidence={item.get('structure_confidence')} | "
                    f"layout_confidence={item.get('layout_confidence')} | "
                    f"semantic_confidence={item.get('semantic_confidence')} ({item.get('semantic_confidence_label')})"
                )
                if item.get("classification_status") or item.get("trust_policy"):
                    print(
                        f"  trust: status={item.get('classification_status')} | "
                        f"policy={item.get('trust_policy')}"
                    )
                print(
                    f"  semantic: specificity={item.get('semantic_specificity')} | "
                    f"semantic_pass={item.get('semantic_pass')} | trust_limited={item.get('trust_limited')}"
                )
                if item.get("audience_answer"):
                    print(f"  audience: {item.get('audience_answer')}")
                if item.get("confidence_answer"):
                    print(f"  confidence: {item.get('confidence_answer')}")
                if item.get("rationale_answer"):
                    print(f"  rationale: {item.get('rationale_answer')}")
                if item.get("limits_answer"):
                    print(f"  limits: {item.get('limits_answer')}")
            return

        if command == "corpus-sanity-check":
            PATHS.ensure_dirs()
            corpus_dir = Path(args.corpus_dir).expanduser().resolve() if args.corpus_dir else None
            sample_size = _resolve_corpus_sample_size(args.sample_profile, args.sample_size)
            payload = _run_corpus_sanity_check(
                sample_size,
                corpus_dir=corpus_dir,
                k=args.k,
                sample_profile=args.sample_profile if args.sample_size is None else "custom",
                save_snapshot=True,
            )
            if json_output:
                _emit_json("corpus-sanity-check", payload, output_path=output_path)
                return
            print(f"Corpus sanity check: {_human_status(payload['all_pass'])}")
            print(
                "Corpus semantic gate: "
                f"technical={_human_status(payload.get('technical_all_pass'))} | "
                f"semantic={_human_status(payload.get('semantic_all_pass'))}"
            )
            architecture_gates = payload.get("architecture_gates", {})
            if architecture_gates:
                print(
                    "Corpus architecture gate: "
                    f"{_human_status(architecture_gates.get('all_pass'))}"
                )
            print(f"Sample profile: {payload.get('sample_profile')}")
            print(f"Corpus PDFs available: {payload['corpus_pdf_count']}")
            print(f"Sampled PDFs: {payload['sample_size']}")
            sample_manifest = payload.get("sample_manifest", {})
            if sample_manifest:
                print(f"Sample algorithm: {sample_manifest.get('sampling_algorithm')}")
                print(f"Sample checksum: {sample_manifest.get('selected_digest_checksum')}")
                print(f"Selected bucket counts: {sample_manifest.get('selected_bucket_counts')}")
            if payload.get("snapshot_path"):
                print(f"Corpus snapshot: {payload['snapshot_path']}")
            summary = payload["summary"]
            print(
                "Average confidences: "
                f"structure={summary.get('avg_structure_confidence')} | "
                f"layout={summary.get('avg_layout_confidence')} | "
                f"semantic={summary.get('avg_semantic_confidence')}"
            )
            print(
                "Semantic rates: "
                f"pass={summary.get('semantic_pass_rate')} | "
                f"specific_type={summary.get('specific_document_rate')} | "
                f"specific_purpose={summary.get('specific_purpose_rate')} | "
                f"low_confidence={summary.get('low_confidence_rate')} | "
                f"trust_limited={summary.get('trust_limited_rate')}"
            )
            print(f"Classification status counts: {summary.get('classification_status_counts')}")
            print(f"Trust policy counts: {summary.get('trust_policy_counts')}")
            print(f"Generic warning count: {summary.get('generic_warning_count')}")
            layer_stability = payload.get("layer_stability", {})
            if layer_stability:
                print(f"Corpus layer stability: {_human_status(layer_stability.get('all_pass'))}")
                failed_layers = layer_stability.get("failed_layers", [])
                if failed_layers:
                    print(f"Corpus failed layers: {', '.join(failed_layers)}")
            reasons = architecture_gates.get("reasons", []) if architecture_gates else []
            if reasons:
                print(f"Corpus gate reasons: {', '.join(reasons)}")
            contract_gate = payload.get("contract_gate", {})
            if contract_gate:
                print(f"Corpus contract gate: {_human_status(contract_gate.get('all_pass'))}")
            bucket_diagnostics = payload.get("bucket_diagnostics", {})
            if bucket_diagnostics:
                print("Bucket diagnostics:")
                for bucket, item in bucket_diagnostics.items():
                    reasons_count = item.get("dominant_failure_reasons", {})
                    reason_text = (
                        ", ".join(list(reasons_count.keys())[:3])
                        if isinstance(reasons_count, dict)
                        else ""
                    )
                    print(
                        f"- {bucket}: n={item.get('sample_count')} | "
                        f"technical={item.get('technical_pass_rate')} | "
                        f"semantic={item.get('semantic_pass_rate')} | "
                        f"trust_limited={item.get('trust_limited_rate')} | "
                        f"reasons={reason_text or 'none'}"
                    )
            follow_up_actions = payload.get("follow_up_actions", [])
            if follow_up_actions:
                print("Corpus follow-up:")
                for action in follow_up_actions:
                    print(
                        f"- {action.get('priority')} | {action.get('bucket')} | "
                        f"{action.get('focus')}: {action.get('reason')}"
                    )
                    examples = action.get("failure_examples", [])
                    if isinstance(examples, list):
                        for example in examples[:3]:
                            if isinstance(example, dict):
                                print(
                                    f"  example: {Path(str(example.get('pdf'))).name} | "
                                    f"reasons={', '.join(str(reason) for reason in example.get('reasons', []))}"
                                )
            return

        if command == "init":
            PATHS.ensure_dirs()
            if json_output:
                _emit_json(
                    "init",
                    {
                        "data_root": str(PATHS.data_dir),
                        "created_dirs": {
                            "data_input": str(PATHS.data_input),
                            "data_documents": str(PATHS.data_documents),
                            "data_chunks": str(PATHS.data_chunks),
                            "data_index": str(PATHS.data_index),
                            "data_eval": str(PATHS.data_eval),
                        }
                    },
                    output_path=output_path,
                )
                return
            print("Created local data directories.")
            print("Next: run `pdf-to-json-rag doctor --json`")
            return

        if command == "extract-native":
            pdf_value = _require_arg(args.pdf, "--pdf", "extract-native")
            PATHS.ensure_dirs()
            pdf_path = _resolve_pdf_path(pdf_value)
            extraction, document_record, native_path, document_path = process_native_pdf_to_json(
                pdf_path=pdf_path,
                output_dir=PATHS.data_documents,
            )
            if json_output:
                _emit_json(
                    "extract-native",
                {
                    "pdf": str(pdf_path),
                    "doc_id": extraction.doc_id,
                        "page_count": extraction.page_count,
                        "native_block_count": len(extraction.blocks),
                        "pages_requiring_ocr": document_record.extraction_summary["pages_requiring_ocr"],
                        "document_type": document_record.document_type,
                        "document_purpose": document_record.document_purpose,
                        "document_family": document_record.document_family,
                        "structure_style": document_record.structure_style,
                        "inventory_summary": document_record.inventory_summary,
                        "coverage_summary": document_record.coverage_summary,
                        "saved_native_json": str(native_path),
                        "saved_document_json": str(document_path),
                    },
                    output_path=output_path,
                )
                return
            print(f"Processed: {pdf_path.name}")
            print(f"doc_id: {extraction.doc_id}")
            print(f"pages: {extraction.page_count}")
            print(f"native_blocks: {len(extraction.blocks)}")
            print(f"pages_requiring_ocr: {document_record.extraction_summary['pages_requiring_ocr']}")
            print(f"document_type: {document_record.document_type}")
            print(f"document_purpose: {document_record.document_purpose}")
            print(f"structure_style: {document_record.structure_style}")
            print(f"saved_native_json: {native_path}")
            print(f"saved_document_json: {document_path}")
            print(
                "Next: run "
                f"`pdf-to-json-rag chunk-document --doc-id {extraction.doc_id} --json`"
            )
            return

        if command == "chunk-document":
            doc_id = _require_arg(args.doc_id, "--doc-id", "chunk-document")
            PATHS.ensure_dirs()
            native_path, document_path = _resolve_document_paths(doc_id)
            document, chunks, saved_paths = process_saved_document_to_chunks(
                native_path=native_path,
                document_path=document_path,
                output_dir=PATHS.data_chunks,
            )
            if json_output:
                _emit_json(
                    "chunk-document",
                    {
                        "doc_id": document.doc_id,
                        "source_pdf": document.source_pdf,
                        "chunks_created": len(chunks),
                        "chunk_output_dir": str(PATHS.data_chunks / document.doc_id),
                        "first_chunk_file": str(saved_paths[0]) if saved_paths else None,
                        "last_chunk_file": str(saved_paths[-1]) if saved_paths else None,
                    },
                    output_path=output_path,
                )
                return
            print(f"Chunked document: {document.source_pdf}")
            print(f"doc_id: {document.doc_id}")
            print(f"chunks_created: {len(chunks)}")
            print(f"chunk_output_dir: {PATHS.data_chunks / document.doc_id}")
            if saved_paths:
                print(f"first_chunk_file: {saved_paths[0]}")
                print(f"last_chunk_file: {saved_paths[-1]}")
            print(
                "Next: run "
                f"`pdf-to-json-rag build-index --doc-id {document.doc_id} --json`"
            )
            return

        if command == "build-index":
            PATHS.ensure_dirs()
            doc_ids = _load_doc_ids_with_chunks(args.doc_id)
            chunks = []
            for doc_id in doc_ids:
                chunk_dir = PATHS.data_chunks / doc_id
                if not chunk_dir.exists():
                    raise CliError(
                        "missing_chunk_directory",
                        f"Chunk directory does not exist for doc_id '{doc_id}'",
                        {"doc_id": doc_id, "chunk_dir": str(chunk_dir)},
                    )
                chunks.extend(load_chunk_records(chunk_dir))
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            manifest = build_local_index(chunks=chunks, index_dir=index_dir)
            embedding_payload = _embedding_manifest_payload(manifest)
            if json_output:
                _emit_json(
                    "build-index",
                    {
                        "doc_ids": doc_ids,
                        "chunk_count": manifest["chunk_count"],
                        "collection_name": manifest["collection_name"],
                        "embedding_backend": manifest["embedding_backend"],
                        "embedding_model": manifest["embedding_model"],
                        "embedding": embedding_payload,
                        "index_dir": str(index_dir),
                    },
                    output_path=output_path,
                )
                return
            print(f"Indexed doc IDs: {', '.join(doc_ids)}")
            print(f"chunks_indexed: {manifest['chunk_count']}")
            print(f"collection_name: {manifest['collection_name']}")
            print(f"requested_embedding_backend: {embedding_payload['requested_backend']}")
            print(f"effective_embedding_backend: {embedding_payload['effective_backend']}")
            print(f"effective_embedding_model: {embedding_payload['effective_model']}")
            if embedding_payload.get("fallback_reason"):
                print(f"embedding_fallback_reason: {embedding_payload['fallback_reason']}")
            print(f"index_dir: {index_dir}")
            print(
                'Next: run `pdf-to-json-rag answer-query --query "What does this file cover?" --json`'
            )
            return

        if command in {"run-workflow", "smoke-check"}:
            pdf_value = _require_arg(args.pdf, "--pdf", command)
            query = _require_arg(args.query, "--query", command)
            PATHS.ensure_dirs()
            pdf_path = _resolve_pdf_path(pdf_value)
            workflow_index_dir = (
                _resolve_index_dir(args.index_dir, PATHS.data_index / "workflow_smoke")
            )
            extraction, document_record, native_path, document_path = process_native_pdf_to_json(
                pdf_path=pdf_path,
                output_dir=PATHS.data_documents,
            )
            document, chunks, saved_paths = process_saved_document_to_chunks(
                native_path=native_path,
                document_path=document_path,
                output_dir=PATHS.data_chunks,
            )
            manifest = build_local_index(chunks=chunks, index_dir=workflow_index_dir)
            embedding_payload = _embedding_manifest_payload(manifest)
            inventory_entry = get_inventory_entry(document.doc_id)
            plan = plan_query(query)
            answer = answer_query_with_retrieval(
                query=query,
                index_dir=workflow_index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            payload = {
                "pdf": str(pdf_path),
                "doc_id": extraction.doc_id,
                "artifacts": {
                    "native_json": str(native_path),
                    "document_json": str(document_path),
                    "chunk_dir": str(PATHS.data_chunks / document.doc_id),
                    "index_dir": str(workflow_index_dir),
                    "first_chunk_file": str(saved_paths[0]) if saved_paths else None,
                },
                "document": {
                    **(_document_payload(inventory_entry) if inventory_entry else {
                        "doc_id": document.doc_id,
                        "label": document.title or document.doc_id,
                        "document_family": document.document_family,
                        "document_type": document.document_type,
                        "document_purpose": document.document_purpose,
                        "audience": document.audience,
                        "evidence_style": document.evidence_style,
                        "structure_style": document.structure_style,
                        "inventory_summary": document.inventory_summary,
                        "coverage_summary": document.coverage_summary,
                        "coverage_terms": list(document.coverage_terms),
                        "discovery_terms": list(document.discovery_terms),
                    }),
                    "structure_confidence": document.structure_confidence,
                    "layout_confidence": document.layout_confidence,
                    "semantic_confidence": document.semantic_confidence,
                    "semantic_confidence_label": document.semantic_confidence_label,
                    "semantic_rationale": list(document.semantic_rationale),
                    "semantic_warnings": list(document.semantic_warnings),
                },
                "plan": {
                    **_plan_payload(plan, verbose=args.verbose),
                },
                "index": {
                    "doc_ids": [document.doc_id],
                    "chunk_count": manifest["chunk_count"],
                    "collection_name": manifest["collection_name"],
                    "embedding_backend": manifest["embedding_backend"],
                    "embedding_model": manifest["embedding_model"],
                    "embedding": embedding_payload,
                },
                "answer": _grounded_answer_payload(answer, verbose=args.verbose),
            }
            payload["quality_profile"] = _workflow_quality_profile(payload)
            if command == "smoke-check":
                checks = _smoke_checks(payload)
                smoke_payload = {
                    **payload,
                    "checks": checks,
                    "all_pass": all(item["passed"] for item in checks),
                }
                if json_output:
                    _emit_json("smoke-check", smoke_payload, output_path=output_path)
                    return
                print(f"Smoke check for: {pdf_path.name}")
                for item in checks:
                    print(f"- {item['name']}: {'PASS' if item['passed'] else 'FAIL'}")
                print(f"all_pass: {all(item['passed'] for item in checks)}")
                print(f"requested_embedding_backend: {embedding_payload['requested_backend']}")
                print(f"effective_embedding_backend: {embedding_payload['effective_backend']}")
                if embedding_payload.get("fallback_reason"):
                    print(f"embedding_fallback_reason: {embedding_payload['fallback_reason']}")
                return
            if json_output:
                _emit_json("run-workflow", payload, output_path=output_path)
                return
            print(f"Workflow complete for: {pdf_path.name}")
            print(f"doc_id: {document.doc_id}")
            print(f"chunks_created: {len(chunks)}")
            print(f"index_dir: {workflow_index_dir}")
            print(f"requested_embedding_backend: {embedding_payload['requested_backend']}")
            print(f"effective_embedding_backend: {embedding_payload['effective_backend']}")
            if embedding_payload.get("fallback_reason"):
                print(f"embedding_fallback_reason: {embedding_payload['fallback_reason']}")
            print(format_grounded_answer(answer))
            return

        if command == "list-documents":
            PATHS.ensure_dirs()
            entries = shortlist_documents(args.query, limit=args.k if args.query else 20) if args.query else list(load_document_inventory())[:20]
            if json_output:
                shortlist = shortlist_document_candidates(args.query, limit=args.k) if args.query else []
                _emit_json(
                    "list-documents",
                    {
                        "query": args.query,
                        "count": len(entries),
                        "documents": [_document_payload(entry) for entry in entries],
                        **({"shortlist": [_shortlist_candidate_payload(candidate) for candidate in shortlist]} if args.query and args.verbose else {}),
                    },
                    output_path=output_path,
                )
                return
            print(f"documents: {len(entries)}")
            for entry in entries:
                print(
                    f"- {entry.doc_id} | {entry.document_family} | {entry.document_purpose} | "
                    f"{entry.coverage_summary}"
                )
            return

        if command == "inspect-document":
            doc_id = _require_arg(args.doc_id, "--doc-id", "inspect-document")
            PATHS.ensure_dirs()
            entry = get_inventory_entry(doc_id)
            if not entry:
                raise CliError(
                    "unknown_doc_id",
                    f"Unknown doc_id: {doc_id}",
                    {"doc_id": doc_id},
                )
            section_payloads: list[dict[str, object]] = []
            document_record = None
            try:
                _, document_path = _resolve_document_paths(doc_id)
                document_record = load_document_record(document_path)
                section_payloads = [_section_payload(section) for section in document_record.sections[:12]]
            except CliError:
                section_payloads = []
            payload = {
                **_document_payload(entry),
                "structure_confidence": getattr(document_record, "structure_confidence", None),
                "layout_confidence": getattr(document_record, "layout_confidence", None),
                "semantic_confidence": getattr(document_record, "semantic_confidence", None),
                "semantic_confidence_label": getattr(document_record, "semantic_confidence_label", None),
                "semantic_rationale": list(getattr(document_record, "semantic_rationale", []) or []),
                "semantic_warnings": list(getattr(document_record, "semantic_warnings", []) or []),
                "extraction_summary": dict(getattr(document_record, "extraction_summary", {}) or {}),
                "section_count": len(section_payloads),
                "sections": section_payloads,
            }
            if json_output:
                _emit_json("inspect-document", payload, output_path=output_path)
                return
            for key, value in payload.items():
                print(f"{key}: {value}")
            return

        if command == "plan-query":
            query = _require_arg(args.query, "--query", "plan-query")
            PATHS.ensure_dirs()
            plan = plan_query(query)
            payload = _plan_payload(plan, verbose=args.verbose)
            if json_output:
                _emit_json("plan-query", payload, output_path=output_path)
                return
            for key, value in payload.items():
                print(f"{key}: {value}")
            return

        if command == "retrieve":
            query = _require_arg(args.query, "--query", "retrieve")
            PATHS.ensure_dirs()
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            _validate_index_dir(index_dir)
            hits = retrieve_top_k(query=query, index_dir=index_dir, k=args.k)
            if json_output:
                _emit_json(
                    "retrieve",
                    {
                        "query": query,
                        "k": args.k,
                        "index_dir": str(index_dir),
                        "hit_count": len(hits),
                        "hits": [_chunk_payload(chunk) for chunk in hits],
                    },
                    output_path=output_path,
                )
                return
            print(f"query: {query}")
            print(f"hits: {len(hits)}")
            for index, chunk in enumerate(hits, start=1):
                preview = chunk.text.replace("\n", " ").strip()[:220]
                print(
                    f"{index}. {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
                    f"section={chunk.section_title!r}"
                )
                print(f"   {preview}")
            return

        if command == "retrieve-expanded":
            query = _require_arg(args.query, "--query", "retrieve-expanded")
            PATHS.ensure_dirs()
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            _validate_index_dir(index_dir)
            hits, expanded = retrieve_top_k_with_neighbors(
                query=query,
                index_dir=index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            if json_output:
                _emit_json(
                    "retrieve-expanded",
                    {
                        "query": query,
                        "k": args.k,
                        "index_dir": str(index_dir),
                        "top_k_count": len(hits),
                        "expanded_count": len(expanded),
                        "top_k_hits": [_chunk_payload(chunk) for chunk in hits],
                        "expanded_hits": [_chunk_payload(chunk) for chunk in expanded],
                    },
                    output_path=output_path,
                )
                return
            print(f"query: {query}")
            print(f"top_k_hits: {len(hits)}")
            print(f"expanded_hits: {len(expanded)}")
            print("-- top-k --")
            for index, chunk in enumerate(hits, start=1):
                preview = chunk.text.replace("\n", " ").strip()[:180]
                print(
                    f"{index}. {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
                    f"prev={chunk.preceding_chunk_id} | next={chunk.following_chunk_id}"
                )
                print(f"   {preview}")
            print("-- expanded --")
            for index, chunk in enumerate(expanded, start=1):
                preview = chunk.text.replace("\n", " ").strip()[:180]
                print(
                    f"{index}. {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
                    f"section={chunk.section_title!r}"
                )
                print(f"   {preview}")
            return

        if command == "answer-query":
            query = _require_arg(args.query, "--query", "answer-query")
            PATHS.ensure_dirs()
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            _validate_index_dir(index_dir)
            result = answer_query_with_retrieval(
                query=query,
                index_dir=index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            if json_output:
                _emit_json("answer-query", _grounded_answer_payload(result, verbose=args.verbose), output_path=output_path)
                return
            print(format_grounded_answer(result))
            return

        if command == "evaluate-mvp":
            PATHS.ensure_dirs()
            eval_path = Path(args.eval_file).expanduser().resolve() if args.eval_file else None
            if eval_path is None:
                eval_path = ensure_default_eval_cases(PATHS.data_eval)
            report, report_path = run_mvp_evaluation(
                index_dir=PATHS.data_index,
                chunk_root=PATHS.data_chunks,
                eval_dir=PATHS.data_eval,
                k=args.k,
                eval_path=eval_path,
            )
            if json_output:
                _emit_json(
                    "evaluate-mvp",
                    {
                        "eval_file": str(eval_path),
                        "report_path": str(report_path),
                        "case_count": report["case_count"],
                        "summary": report["summary"],
                        "layer_summary": report.get("layer_summary", {}),
                        "layer_stability": report.get("layer_stability", {}),
                        "architecture_gates": report.get("architecture_gates", {}),
                        "faithfulness_audit": report["faithfulness_audit"],
                        "retrieval_strategy_comparison": report.get("retrieval_strategy_comparison", {}),
                        "deferred_feature_decisions": report.get("deferred_feature_decisions", {}),
                        "slice_stability": report.get("slice_stability", {}),
                    },
                    output_path=output_path,
                )
                return
            print(f"Evaluation file: {eval_path}")
            print(f"Report saved to: {report_path}")
            print(f"Cases: {report['case_count']}")
            print(f"avg_precision@{args.k}: {report['summary']['avg_precision_at_k']:.3f}")
            print(f"avg_recall@{args.k}: {report['summary']['avg_recall_at_k']:.3f}")
            print(f"MRR: {report['summary']['mrr']:.3f}")
            print(f"avg_keyword_coverage: {report['summary']['avg_keyword_coverage']:.3f}")
            print(f"negative_case_count: {report['summary']['negative_case_count']}")
            print(f"negative_success_rate: {report['summary']['negative_success_rate']:.3f}")
            print(f"warning_case_count: {report['summary']['warning_case_count']}")
            layer_summary = report.get("layer_summary", {})
            if layer_summary:
                processing = layer_summary.get("processing", {})
                retrieval = layer_summary.get("retrieval", {})
                answer_faithfulness = layer_summary.get("answer_faithfulness", {})
                print(f"layer_all_pass: {layer_summary.get('all_pass', False)}")
                print(f"processing_layer_pass_rate: {processing.get('pass_rate', 0.0):.3f}")
                print(f"retrieval_layer_pass_rate: {retrieval.get('pass_rate', 0.0):.3f}")
                print(
                    "answer_faithfulness_pass_rate: "
                    f"{answer_faithfulness.get('pass_rate', 0.0):.3f}"
                )
            layer_stability = report.get("layer_stability", {})
            if layer_stability:
                print(f"layer_stability_all_pass: {layer_stability.get('all_pass', False)}")
                failed_layers = layer_stability.get("failed_layers", [])
                if failed_layers:
                    print(f"layer_stability_failed_layers: {', '.join(failed_layers)}")
            print(
                "faithfulness_supported_sentence_ratio: "
                f"{report['faithfulness_audit']['avg_supported_sentence_ratio']:.3f}"
            )
            print(
                "recommend_llm_judge: "
                f"{report['faithfulness_audit']['recommend_llm_judge']}"
            )
            rerank = report.get("retrieval_strategy_comparison", {})
            if rerank:
                baseline = rerank.get("baseline_chunking_only", {})
                lightweight = rerank.get("lightweight_rerank", {})
                print(
                    "baseline_vs_rerank_mrr: "
                    f"{baseline.get('mrr', 0.0):.3f} -> {lightweight.get('mrr', 0.0):.3f}"
                )
            deferred = report.get("deferred_feature_decisions", {})
            if deferred:
                print(
                    "recommend_pdfplumber_probe: "
                    f"{deferred.get('pdfplumber_probe', {}).get('recommended', False)}"
                )
                print(
                    "recommend_cross_encoder: "
                    f"{deferred.get('cross_encoder_reranking', {}).get('recommended', False)}"
                )
            stability = report.get("slice_stability", {})
            if stability:
                print(f"slice_stability_all_pass: {stability.get('all_pass', False)}")
                failed = stability.get("failed_labels", [])
                if failed:
                    print(f"slice_stability_failed_labels: {', '.join(failed)}")
            architecture_gates = report.get("architecture_gates", {})
            if architecture_gates:
                print(f"architecture_gates_all_pass: {architecture_gates.get('all_pass', False)}")
                reasons = architecture_gates.get("reasons", [])
                if reasons:
                    print(f"architecture_gate_reasons: {', '.join(reasons)}")
            return

        if command == "evaluate-regression":
            PATHS.ensure_dirs()
            eval_path = Path(args.eval_file).expanduser().resolve() if args.eval_file else None
            case_ids = None
            if args.case_ids:
                case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
            report, report_path = run_regression_suite(
                index_dir=PATHS.data_index,
                chunk_root=PATHS.data_chunks,
                eval_dir=PATHS.data_eval,
                k=args.k,
                eval_path=eval_path,
                case_ids=case_ids,
                shard=args.shard,
            )
            if json_output:
                _emit_json(
                    "evaluate-regression",
                    {
                        "report_path": str(report_path),
                        "selected_shard": report.get("selected_shard"),
                        "case_count": report["case_count"],
                        "pass_count": report["pass_count"],
                        "fail_count": report["fail_count"],
                        "all_pass": report["all_pass"],
                        "missing_case_ids": report.get("missing_case_ids", []),
                        "failed_case_ids": report.get("failed_case_ids", []),
                    },
                    output_path=output_path,
                )
                return
            print(f"Regression report saved to: {report_path}")
            if report.get("selected_shard"):
                print(f"Selected shard: {report['selected_shard']}")
            print(f"Selected cases: {report['case_count']}")
            print(f"Pass count: {report['pass_count']}")
            print(f"Fail count: {report['fail_count']}")
            print(f"All pass: {report['all_pass']}")
            missing = report.get("missing_case_ids", [])
            if missing:
                print(f"Missing case IDs: {', '.join(missing)}")
            failed = report.get("failed_case_ids", [])
            if failed:
                print(f"Failed case IDs: {', '.join(failed)}")
            return

        if command == "compare-runtime-modes":
            PATHS.ensure_dirs()
            eval_path = Path(args.eval_file).expanduser().resolve() if args.eval_file else None
            case_ids = None
            if args.case_ids:
                case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
            modes = None
            if args.modes:
                modes = [item.strip() for item in args.modes.split(",") if item.strip()]
            report, report_path = run_runtime_mode_comparison(
                index_dir=PATHS.data_index,
                chunk_root=PATHS.data_chunks,
                eval_dir=PATHS.data_eval,
                k=args.k,
                eval_path=eval_path,
                case_ids=case_ids,
                shard=args.shard,
                modes=modes,
                all_cases=args.all_cases,
            )
            promotion_snapshot_path = _write_runtime_promotion_snapshot(report, report_path)
            if json_output:
                _emit_json(
                    "compare-runtime-modes",
                    {
                        "report_path": str(report_path),
                        "selected_shard": report.get("selected_shard"),
                        "all_cases": report.get("all_cases", False),
                        "case_count": report.get("case_count", 0),
                        "selected_case_ids": report.get("selected_case_ids", []),
                        "missing_case_ids": report.get("missing_case_ids", []),
                        "unknown_modes": report.get("unknown_modes", []),
                        "available_modes": report.get("available_modes", []),
                        "mode_results": [
                            {
                                "mode": item["mode"],
                                "case_count": item["case_count"],
                                "pass_count": item["pass_count"],
                                "fail_count": item["fail_count"],
                                "all_pass": item["all_pass"],
                                "failed_case_ids": item["failed_case_ids"],
                                "summary": item["summary"],
                                "index_manifest": item["index_manifest"],
                                "runtime_signals": item["runtime_signals"],
                            }
                            for item in report.get("mode_results", [])
                        ],
                        "baseline_deltas": report.get("baseline_deltas", {}),
                        "promotion_gates": report.get("promotion_gates", {}),
                        "promotion_snapshot_path": str(promotion_snapshot_path) if promotion_snapshot_path else None,
                        "all_pass": report.get("all_pass", False),
                    },
                    output_path=output_path,
                )
                return
            print(f"Runtime mode comparison saved to: {report_path}")
            if promotion_snapshot_path:
                print(f"Runtime promotion snapshot saved to: {promotion_snapshot_path}")
            print(f"Cases: {report.get('case_count', 0)}")
            for item in report.get("mode_results", []):
                summary = item.get("summary", {})
                manifest = item.get("index_manifest", {})
                signals = item.get("runtime_signals", {})
                print(
                    f"{item['mode']}: pass={item['pass_count']}/{item['case_count']} "
                    f"mrr={summary.get('mrr', 0.0):.3f} "
                    f"recall={summary.get('avg_recall_at_k', 0.0):.3f} "
                    f"keywords={summary.get('avg_keyword_coverage', 0.0):.3f} "
                    f"embedding={manifest.get('embedding_backend')}/{manifest.get('embedding_model')} "
                    f"llm_used={signals.get('llm_used_case_count', 0)}"
                )
            deltas = report.get("baseline_deltas", {})
            if deltas:
                print("Deltas vs baseline:")
                for mode, delta in deltas.items():
                    print(
                        f"{mode}: mrr_delta={delta.get('mrr_delta', 0.0):+.3f} "
                        f"recall_delta={delta.get('avg_recall_at_k_delta', 0.0):+.3f} "
                        f"keyword_delta={delta.get('avg_keyword_coverage_delta', 0.0):+.3f}"
                    )
            promotion_gates = report.get("promotion_gates", {})
            sentence_gate = promotion_gates.get("sentence-transformers")
            if sentence_gate:
                print(
                    "sentence_transformers_promotable: "
                    f"{sentence_gate.get('promotable', False)}"
                )
                reasons = sentence_gate.get("reasons", [])
                if reasons:
                    print(f"sentence_transformers_promotion_reasons: {', '.join(reasons)}")
            missing = report.get("missing_case_ids", [])
            if missing:
                print(f"Missing case IDs: {', '.join(missing)}")
            unknown = report.get("unknown_modes", [])
            if unknown:
                print(f"Unknown modes: {', '.join(unknown)}")
            return

        raise CliError("unknown_command", f"Unknown command: {command}", {"command": command})
    except CliError as error:
        if _wants_json(argv) or ("args" in locals() and args.format == "json"):
            _emit_error_json(locals().get("command"), error, output_path=locals().get("output_path"))
        else:
            print(f"Error [{error.code}]: {error.message}", file=sys.stderr)
            if error.details:
                print(json.dumps(error.details, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
if __name__ == "__main__":
    main()
