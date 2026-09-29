"""Shared types, constants, and path discovery for the command-line interface."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from .config import PATHS


class CliError(Exception):
    def __init__(
        self, code: str, message: str, details: dict[str, object] | None = None
    ):
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
    "assess": "assess-pdf",
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
    "corpus-compare": "corpus-profile-compare",
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
        "example": 'pdf-to-json-rag run-workflow --pdf /path/to/file.pdf --query "What does this file cover?" --json',
    },
    "smoke-check": {
        "summary": "Validate the packaged workflow path and return pass/fail checks.",
        "example": 'pdf-to-json-rag smoke-check --pdf /path/to/file.pdf --query "What does this file cover?" --json',
    },
    "assess-pdf": {
        "summary": "Assess whether an unfamiliar PDF processed reliably and how much to trust the answer.",
        "example": "pdf-to-json-rag assess-pdf --pdf /path/to/file.pdf --json",
    },
    "inspect-pdf-quality": {
        "summary": "Inspect processing quality, table/form signals, and readiness for an unfamiliar PDF.",
        "example": "pdf-to-json-rag inspect-pdf-quality --pdf /path/to/file.pdf --json",
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
        "example": 'pdf-to-json-rag plan-query --query "Which file is most relevant for drought triggers?" --json',
    },
    "retrieve": {
        "summary": "Return top-k chunks without answer assembly.",
        "example": 'pdf-to-json-rag retrieve --query "What are common cold symptoms?" --top-k 5 --json',
    },
    "retrieve-expanded": {
        "summary": "Return top-k chunks plus adjacent expansion.",
        "example": 'pdf-to-json-rag retrieve-expanded --query "What are common cold symptoms?" --top-k 5 --json',
    },
    "answer-query": {
        "summary": "Answer a query using the current local index and answer-mode logic.",
        "example": 'pdf-to-json-rag answer-query --query "What are common cold symptoms?" --json',
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
    "real-ground-truth-check": {
        "summary": "Product gate: evaluate real local PDFs against a small hand-built ground-truth retrieval/evidence set.",
        "example": "pdf-to-json-rag real-ground-truth-check --json",
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
    "corpus-profile-compare": {
        "summary": "Compare saved local corpus sanity snapshots without reprocessing PDFs.",
        "example": "pdf-to-json-rag corpus-profile-compare --baseline-profile quick --candidate-profile balanced --json",
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
  pdf-to-json-rag assess-pdf --pdf /path/to/file.pdf --json

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


DEFAULT_REAL_PDF_GROUND_TRUTH_FILENAME = "real_pdf_ground_truth_cases.json"


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


def _discover_project_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _local_pdf_corpus_paths(corpus_dir: Path | None = None) -> tuple[Path, Path] | None:
    if corpus_dir is not None:
        metadata_path = corpus_dir / "lcwa_gov_pdf_metadata.csv"
        return (
            (corpus_dir, metadata_path)
            if corpus_dir.exists() and metadata_path.exists()
            else None
        )
    project_root = _discover_project_root(PATHS.root) or _discover_project_root(
        Path.cwd()
    )
    if project_root is None:
        return None
    pdf_dir = project_root / "pdf"
    metadata_path = pdf_dir / "lcwa_gov_pdf_metadata.csv"
    if pdf_dir.exists() and metadata_path.exists():
        return pdf_dir, metadata_path
    return None
