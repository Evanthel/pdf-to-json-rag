"""CLI entry points for the local PDF-to-JSON RAG tool."""

import argparse
from importlib import metadata as importlib_metadata
from importlib import resources as importlib_resources
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

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
from .evaluation import DEFAULT_EVAL_FILENAME, ensure_default_eval_cases, run_mvp_evaluation, run_regression_suite
from .extraction import process_native_pdf_to_json
from .indexing import build_local_index, load_chunk_records
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
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "section_summary": chunk.section_summary,
        "section_coverage_terms": list(chunk.section_coverage_terms),
        "section_content_hints": list(chunk.section_content_hints),
        "chunk_type": chunk.chunk_type,
        "preceding_chunk_id": chunk.preceding_chunk_id,
        "following_chunk_id": chunk.following_chunk_id,
        "extraction_method": chunk.extraction_method,
        "quality_score": chunk.quality_score,
        "confidence": chunk.confidence,
        "retrieval_signals": dict(chunk.retrieval_signals),
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
        "page_start": section.page_start,
        "page_end": section.page_end,
        "reading_order_start": section.reading_order_start,
        "reading_order_end": section.reading_order_end,
        "summary": section.summary,
        "coverage_terms": list(section.coverage_terms),
        "content_hints": list(section.content_hints),
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
        "document_selection": answer_trace.get("document_selection", {}),
        "template_id": answer_trace.get("template_id"),
        "matched_pattern": answer_trace.get("matched_pattern"),
        "matched_cues": answer_trace.get("matched_cues", []),
        "chosen_rationale": answer_trace.get("chosen_rationale", []),
        "answer_contract": answer_trace.get("answer_contract", {}),
        "support_trace": answer_trace.get("support_trace", []),
    }


def _grounded_answer_payload(result, *, verbose: bool = False) -> dict[str, object]:
    return {
        "query": result.query,
        "query_intent": result.query_intent,
        "answer": result.answer,
        "answer_trace": result.answer_trace if verbose else _compact_answer_trace(result.answer_trace),
        **(
            {
                "top_k_hits": [_chunk_payload(chunk) for chunk in result.top_k_hits],
                "expanded_hits": [_chunk_payload(chunk) for chunk in result.expanded_hits],
                "evidence": [_evidence_payload(item) for item in result.evidence],
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
        demo_pdf = workspace / "package-demo.pdf"
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
        doctor_process = None
        smoke_process = None
        doctor_payload: dict[str, object] = {}
        smoke_payload: dict[str, object] = {}

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
                package_env = os.environ.copy()
                package_env["PDF_TO_JSON_RAG_DATA_DIR"] = str(data_dir)
                create_demo_process = subprocess.run(
                    [str(script_path), "create-demo-pdf", "--path", str(demo_pdf), "--json"],
                    cwd=workspace,
                    env=package_env,
                    capture_output=True,
                    text=True,
                )
                if create_demo_process.returncode == 0:
                    doctor_process = subprocess.run(
                        [str(script_path), "doctor", "--json"],
                        cwd=workspace,
                        env=package_env,
                        capture_output=True,
                        text=True,
                    )
                    smoke_process = subprocess.run(
                        [
                            str(script_path),
                            "smoke-check",
                            "--pdf",
                            str(demo_pdf),
                            "--query",
                            "What does this file cover?",
                            "--json",
                        ],
                        cwd=workspace,
                        env=package_env,
                        capture_output=True,
                        text=True,
                    )
                    if doctor_process.stdout.strip():
                        doctor_payload = json.loads(doctor_process.stdout)
                    if smoke_process.stdout.strip():
                        smoke_payload = json.loads(smoke_process.stdout)
                else:
                    smoke_payload = {
                        "ok": False,
                        "error": {
                            "code": "create_demo_failed",
                            "message": create_demo_process.stderr.strip() or create_demo_process.stdout.strip(),
                        },
                    }

        all_pass = (
            build_process.returncode == 0
            and wheel_path is not None
            and venv_process is not None
            and venv_process.returncode == 0
            and install_process is not None
            and install_process.returncode == 0
            and script_path.exists()
            and doctor_process is not None
            and doctor_process.returncode == 0
            and bool(doctor_payload.get("ok"))
            and smoke_process is not None
            and smoke_process.returncode == 0
            and bool(smoke_payload.get("ok"))
            and bool(smoke_payload.get("result", {}).get("all_pass"))
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
            "doctor_returncode": doctor_process.returncode if doctor_process else None,
            "smoke_returncode": smoke_process.returncode if smoke_process else None,
            "doctor_ok": bool(doctor_payload.get("ok")),
            "smoke_ok": bool(smoke_payload.get("ok")),
            "smoke_all_pass": bool(smoke_payload.get("result", {}).get("all_pass")),
            "skipped": False,
            "build_output_tail": "\n".join(
                part.strip()
                for part in (build_process.stdout, build_process.stderr)
                if part and part.strip()
            )[-1200:],
            "install_output_tail": (
                "\n".join(
                    part.strip()
                    for part in ((install_process.stdout if install_process else ""), (install_process.stderr if install_process else ""))
                    if part and part.strip()
                )[-1200:]
                if install_process is not None
                else ""
            ),
        }


RELEASE_CHECK_SHARDS = [
    "query_planning_core",
    "answer_modes_core",
    "document_pipeline_core",
    "structure_chunking_core",
    "evidence_anchor_core",
    "document_family_core",
    "inventory_coverage_core",
    "relationship_core",
]


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
        "overall_pass": overall_pass,
        "recommendation": recommendation,
    }


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


def _doctor_checks() -> dict[str, object]:
    package_metadata_present, package_metadata_details = _project_metadata_available()
    examples_dir = _available_examples_dir()
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
        help="Optional regression shard for evaluate-regression.",
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
            return

        if command == "release-check":
            PATHS.ensure_dirs()
            payload = _run_release_check(args.k)
            if json_output:
                _emit_json("release-check", payload, output_path=output_path)
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
            else:
                print("Maintainer gates: SKIPPED (run from a source checkout to include package and regression checks)")
            print("")
            print("Why:")
            for reason in recommendation["why"]:
                print(f"- {reason}")
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
            if json_output:
                _emit_json(
                    "build-index",
                    {
                        "doc_ids": doc_ids,
                        "chunk_count": manifest["chunk_count"],
                        "collection_name": manifest["collection_name"],
                        "embedding_backend": manifest["embedding_backend"],
                        "embedding_model": manifest["embedding_model"],
                        "index_dir": str(index_dir),
                    },
                    output_path=output_path,
                )
                return
            print(f"Indexed doc IDs: {', '.join(doc_ids)}")
            print(f"chunks_indexed: {manifest['chunk_count']}")
            print(f"collection_name: {manifest['collection_name']}")
            print(f"embedding_backend: {manifest['embedding_backend']}")
            print(f"embedding_model: {manifest['embedding_model']}")
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
                "document": _document_payload(inventory_entry) if inventory_entry else {
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
                },
                "answer": _grounded_answer_payload(answer, verbose=args.verbose),
            }
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
                return
            if json_output:
                _emit_json("run-workflow", payload, output_path=output_path)
                return
            print(f"Workflow complete for: {pdf_path.name}")
            print(f"doc_id: {document.doc_id}")
            print(f"chunks_created: {len(chunks)}")
            print(f"index_dir: {workflow_index_dir}")
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
            try:
                _, document_path = _resolve_document_paths(doc_id)
                document_record = load_document_record(document_path)
                section_payloads = [_section_payload(section) for section in document_record.sections[:12]]
            except CliError:
                section_payloads = []
            payload = {
                **_document_payload(entry),
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
