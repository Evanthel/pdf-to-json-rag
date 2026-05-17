"""CLI entry points for the local PDF-to-JSON RAG tool."""

import argparse
import json
from pathlib import Path
import shutil
import sys

from . import __version__
from .answering import answer_query_with_retrieval, format_grounded_answer
from .chunking import process_saved_document_to_chunks
from .config import PATHS
from .document_inventory import get_inventory_entry, load_document_inventory, shortlist_documents
from .evaluation import ensure_default_eval_cases, run_mvp_evaluation, run_regression_suite
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
    "help": {
        "summary": "Show command summaries or detailed help for one command.",
        "example": "pdf-to-json-rag help --topic answer-query",
    },
}

CANONICAL_COMMANDS = list(COMMAND_HELP.keys())
CLI_EPILOG = """Common first-run commands:
  pdf-to-json-rag init --json
  pdf-to-json-rag doctor --json
  pdf-to-json-rag demo-profile --json
  pdf-to-json-rag smoke-check --pdf /path/to/file.pdf --query "What does this file cover?" --json

Use `pdf-to-json-rag help --topic <command>` for a focused command summary.
"""


def _chunk_payload(chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_title": chunk.section_title,
        "preceding_chunk_id": chunk.preceding_chunk_id,
        "following_chunk_id": chunk.following_chunk_id,
        "extraction_method": chunk.extraction_method,
        "quality_score": chunk.quality_score,
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


def _grounded_answer_payload(result) -> dict[str, object]:
    return {
        "query": result.query,
        "query_intent": result.query_intent,
        "answer": result.answer,
        "top_k_hits": [_chunk_payload(chunk) for chunk in result.top_k_hits],
        "expanded_hits": [_chunk_payload(chunk) for chunk in result.expanded_hits],
        "evidence": [_evidence_payload(item) for item in result.evidence],
        "answer_trace": result.answer_trace,
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
    return "--json" in argv


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


def _project_examples_dir() -> Path:
    return PATHS.root / "examples"


def _load_example_json(filename: str) -> dict[str, object] | list[object]:
    path = _project_examples_dir() / filename
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
    pyproject_path = PATHS.root / "pyproject.toml"
    examples_dir = _project_examples_dir()
    example_files = [
        "public_demo_profile.json",
        "public_workflow.json",
        "public_demo_queries.json",
        "inspect_document.example.json",
        "plan_query.example.json",
        "answer_query.example.json",
    ]
    manifest_path = PATHS.data_index / "index_manifest.json"
    inventory = load_document_inventory()
    checks: list[dict[str, object]] = [
        {
            "name": "package_metadata_present",
            "passed": pyproject_path.exists(),
            "details": {"path": str(pyproject_path)},
        },
        {
            "name": "data_root_configured",
            "passed": bool(PATHS.data_dir),
            "details": {"data_dir": str(PATHS.data_dir)},
        },
        {
            "name": "data_dirs_exist",
            "passed": all(path.exists() for path in (PATHS.data_input, PATHS.data_documents, PATHS.data_chunks, PATHS.data_index, PATHS.data_eval)),
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
            "details": {"which": shutil.which("tesseract")},
        },
        {
            "name": "example_assets_present",
            "passed": examples_dir.exists() and all((examples_dir / name).exists() for name in example_files),
            "details": {
                "examples_dir": str(examples_dir),
                "expected_files": example_files,
            },
        },
        {
            "name": "document_inventory_available",
            "passed": len(inventory) > 0,
            "details": {"document_count": len(inventory)},
        },
        {
            "name": "index_manifest_available",
            "passed": manifest_path.exists(),
            "details": {"manifest_path": str(manifest_path)},
        },
    ]
    ready_for_cli = all(
        check["passed"]
        for check in checks
        if check["name"] in {"package_metadata_present", "data_root_configured", "data_dirs_exist", "example_assets_present"}
    )
    ready_for_retrieval = ready_for_cli and all(
        check["passed"]
        for check in checks
        if check["name"] in {"document_inventory_available", "index_manifest_available"}
    )
    return {
        "checks": checks,
        "ready_for_cli": ready_for_cli,
        "ready_for_retrieval": ready_for_retrieval,
        "data_root": str(PATHS.data_dir),
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
        "--output",
        help="Optional file path for JSON output. Requires --json.",
    )
    try:
        args = parser.parse_args(argv)
        command = _canonical_command(args.command)
        output_path = _resolve_output_path(args.output)
        if output_path and not args.json:
            raise CliError(
                "output_requires_json",
                "--output can only be used together with --json",
                {"output": str(output_path)},
            )

        if command == "help":
            help_text = _render_help(args.topic)
            if args.json:
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
            if args.json:
                _emit_json("demo-profile", {"profile": payload}, output_path=output_path)
                return
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if command == "doctor":
            payload = _doctor_checks()
            if args.json:
                _emit_json("doctor", payload, output_path=output_path)
                return
            print(f"ready_for_cli: {payload['ready_for_cli']}")
            print(f"ready_for_retrieval: {payload['ready_for_retrieval']}")
            for check in payload["checks"]:
                print(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
            return

        if command == "init":
            PATHS.ensure_dirs()
            if args.json:
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
            print("Created MVP data directories.")
            return

        if command == "extract-native":
            pdf_value = _require_arg(args.pdf, "--pdf", "extract-native")
            PATHS.ensure_dirs()
            pdf_path = _resolve_pdf_path(pdf_value)
            extraction, document_record, native_path, document_path = process_native_pdf_to_json(
                pdf_path=pdf_path,
                output_dir=PATHS.data_documents,
            )
            if args.json:
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
            if args.json:
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
            index_dir = _resolve_index_dir(args.index_dir, PATHS.data_index)
            manifest = build_local_index(chunks=chunks, index_dir=index_dir)
            if args.json:
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
                    "query": plan.query,
                    "query_class": plan.query_class,
                    "query_intent": plan.query_intent,
                    "answer_mode": plan.answer_mode,
                    "inventory_doc_ids": list(plan.inventory_doc_ids),
                    "matched_doc_ids": list(plan.matched_doc_ids),
                    "preferred_doc_id": plan.preferred_doc_id,
                },
                "index": {
                    "doc_ids": [document.doc_id],
                    "chunk_count": manifest["chunk_count"],
                    "collection_name": manifest["collection_name"],
                    "embedding_backend": manifest["embedding_backend"],
                    "embedding_model": manifest["embedding_model"],
                },
                "answer": _grounded_answer_payload(answer),
            }
            if command == "smoke-check":
                checks = _smoke_checks(payload)
                smoke_payload = {
                    **payload,
                    "checks": checks,
                    "all_pass": all(item["passed"] for item in checks),
                }
                if args.json:
                    _emit_json("smoke-check", smoke_payload, output_path=output_path)
                    return
                print(f"Smoke check for: {pdf_path.name}")
                for item in checks:
                    print(f"- {item['name']}: {'PASS' if item['passed'] else 'FAIL'}")
                print(f"all_pass: {all(item['passed'] for item in checks)}")
                return
            if args.json:
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
            if args.json:
                _emit_json(
                    "list-documents",
                    {
                        "query": args.query,
                        "count": len(entries),
                        "documents": [_document_payload(entry) for entry in entries],
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
            payload = {
                **_document_payload(entry),
            }
            if args.json:
                _emit_json("inspect-document", payload, output_path=output_path)
                return
            for key, value in payload.items():
                print(f"{key}: {value}")
            return

        if command == "plan-query":
            query = _require_arg(args.query, "--query", "plan-query")
            PATHS.ensure_dirs()
            plan = plan_query(query)
            payload = {
                "query": plan.query,
                "query_class": plan.query_class,
                "query_intent": plan.query_intent,
                "answer_mode": plan.answer_mode,
                "inventory_doc_ids": list(plan.inventory_doc_ids),
                "matched_doc_ids": list(plan.matched_doc_ids),
                "preferred_doc_id": plan.preferred_doc_id,
            }
            if args.json:
                _emit_json("plan-query", payload, output_path=output_path)
                return
            for key, value in payload.items():
                print(f"{key}: {value}")
            return

        if command == "retrieve":
            query = _require_arg(args.query, "--query", "retrieve")
            PATHS.ensure_dirs()
            index_dir = _resolve_index_dir(args.index_dir, PATHS.data_index)
            _validate_index_dir(index_dir)
            hits = retrieve_top_k(query=query, index_dir=index_dir, k=args.k)
            if args.json:
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
            index_dir = _resolve_index_dir(args.index_dir, PATHS.data_index)
            _validate_index_dir(index_dir)
            hits, expanded = retrieve_top_k_with_neighbors(
                query=query,
                index_dir=index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            if args.json:
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
            index_dir = _resolve_index_dir(args.index_dir, PATHS.data_index)
            _validate_index_dir(index_dir)
            result = answer_query_with_retrieval(
                query=query,
                index_dir=index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            if args.json:
                _emit_json("answer-query", _grounded_answer_payload(result), output_path=output_path)
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
            if args.json:
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
            if args.json:
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
        if _wants_json(argv):
            _emit_error_json(locals().get("command"), error, output_path=locals().get("output_path"))
        else:
            print(f"Error [{error.code}]: {error.message}", file=sys.stderr)
            if error.details:
                print(json.dumps(error.details, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
if __name__ == "__main__":
    main()
