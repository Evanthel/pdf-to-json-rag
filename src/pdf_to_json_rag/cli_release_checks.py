"""Maintainer release, package, layout, and corpus checks."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF releases
    import fitz

from .config import PATHS
from .document_inventory import (
    load_document_inventory,
)
from .evaluation import (
    DEFAULT_EVAL_FILENAME,
    run_regression_suite,
)


from .cli_diagnostics import (
    _doctor_checks,
)

from .cli_formatting import (
    _quality_profile_summary,
    _release_channel_recommendation,
)

from .cli_shared import (
    CORPUS_BUCKET_ORDER,
    CORPUS_SAMPLE_PROFILES,
    CliError,
    LocalPdfCorpusEntry,
    _discover_project_root,
    _local_pdf_corpus_paths,
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


def _resolve_pdf_paths(value: str) -> list[Path]:
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        raise CliError(
            "missing_pdf",
            "At least one PDF path is required",
            {"pdfs": value},
        )
    return [_resolve_pdf_path(item) for item in raw_items]


def _safe_int(value: str | None) -> int:
    try:
        return int(str(value or "0").strip())
    except ValueError:
        return 0


def _pdf_corpus_bucket(
    url_text: str, *, pages: int, creator_tool: str, producer: str
) -> str:
    lowered = " ".join((url_text, creator_tool, producer)).lower()
    if any(term in lowered for term in ("scan", "scanning", "capture", "ocr")):
        return "scan_like"
    if any(
        term in lowered
        for term in (
            "form",
            "statement",
            "application",
            "questionnaire",
            "checklist",
            "appendix",
        )
    ):
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
    with tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8", newline="", delete=True
    ) as handle:
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
    entries.sort(
        key=lambda item: (item.bucket, item.pages, item.file_size, item.digest)
    )
    return entries


def _sample_local_pdf_corpus(
    entries: list[LocalPdfCorpusEntry], sample_size: int
) -> list[LocalPdfCorpusEntry]:
    if sample_size <= 0:
        raise CliError(
            "invalid_sample_size",
            "Sample size must be a positive integer",
            {"sample_size": sample_size},
        )
    grouped: dict[str, list[LocalPdfCorpusEntry]] = {
        bucket: [] for bucket in CORPUS_BUCKET_ORDER
    }
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
        "selected_bucket_counts": _count_values(
            [entry.bucket for entry in sampled_entries]
        ),
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
    if (
        not (PATHS.data_index / "index_manifest.json").exists()
        and (workflow_smoke_dir / "index_manifest.json").exists()
    ):
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
    maintainer_root = _discover_project_root(PATHS.root) or _discover_project_root(
        Path.cwd()
    )
    if maintainer_root is not None:
        env["PDF_TO_JSON_RAG_PROJECT_ROOT"] = str(maintainer_root)
    if data_dir is not None:
        env["PDF_TO_JSON_RAG_DATA_DIR"] = str(data_dir)
    return env


def _run_cli_subprocess(
    args: list[str], data_dir: Path | None = None
) -> subprocess.CompletedProcess[str]:
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


def _json_payload_from_process(
    process: subprocess.CompletedProcess[str] | None,
) -> dict[str, object]:
    if process is None or not process.stdout.strip():
        return {}
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_installed_readme_flow(
    script_path: Path, workspace: Path, data_dir: Path
) -> dict[str, object]:
    package_env = os.environ.copy()
    package_env["PDF_TO_JSON_RAG_DATA_DIR"] = str(data_dir)
    demo_pdf = workspace / "readme-demo.pdf"

    steps: list[dict[str, object]] = []

    def run_step(
        name: str, args: list[str]
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
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
                "output_tail": _process_output_tail(process)
                if process.returncode != 0 or not payload.get("ok")
                else "",
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
    runtime_process, runtime_payload = run_step(
        "runtime-check", ["runtime-check", "--json"]
    )

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
        "doctor_ready_for_public_cli": bool(
            doctor_payload.get("result", {}).get("ready_for_public_cli")
        ),
        "create_demo_ok": bool(create_payload.get("ok")),
        "smoke_ok": bool(smoke_payload.get("ok")),
        "smoke_all_pass": bool(smoke_payload.get("result", {}).get("all_pass")),
        "runtime_ok": bool(runtime_payload.get("ok")),
        "runtime_decision": runtime_payload.get("result", {}).get(
            "runtime_decision", {}
        ),
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
        init_payload = (
            json.loads(init_process.stdout) if init_process.stdout.strip() else {}
        )
        smoke_payload = (
            json.loads(smoke_process.stdout) if smoke_process.stdout.strip() else {}
        )
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
            "quality_profile_summary": smoke_payload.get("result", {}).get(
                "quality_profile_summary"
            )
            or _quality_profile_summary(
                smoke_payload.get("result", {}).get("quality_profile")
            ),
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
            smoke_payload = (
                json.loads(smoke_process.stdout) if smoke_process.stdout.strip() else {}
            )

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
            inspect_payload = (
                json.loads(inspect_process.stdout)
                if inspect_process.stdout.strip()
                else {}
            )
            type_process = _run_cli_subprocess(
                ["answer-query", "--query", "What kind of document is this?", "--json"],
                data_dir=data_dir,
            )
            type_payload = (
                json.loads(type_process.stdout) if type_process.stdout.strip() else {}
            )
            purpose_process = _run_cli_subprocess(
                [
                    "answer-query",
                    "--query",
                    "What is the purpose of this document?",
                    "--json",
                ],
                data_dir=data_dir,
            )
            purpose_payload = (
                json.loads(purpose_process.stdout)
                if purpose_process.stdout.strip()
                else {}
            )
            audience_process = _run_cli_subprocess(
                ["answer-query", "--query", "Who is this document for?", "--json"],
                data_dir=data_dir,
            )
            audience_payload = (
                json.loads(audience_process.stdout)
                if audience_process.stdout.strip()
                else {}
            )
            confidence_process = _run_cli_subprocess(
                [
                    "answer-query",
                    "--query",
                    "How confident is this document classification?",
                    "--json",
                ],
                data_dir=data_dir,
            )
            confidence_payload = (
                json.loads(confidence_process.stdout)
                if confidence_process.stdout.strip()
                else {}
            )
            rationale_process = _run_cli_subprocess(
                [
                    "answer-query",
                    "--query",
                    "Why is this document classified this way?",
                    "--json",
                ],
                data_dir=data_dir,
            )
            rationale_payload = (
                json.loads(rationale_process.stdout)
                if rationale_process.stdout.strip()
                else {}
            )
            limits_process = _run_cli_subprocess(
                [
                    "answer-query",
                    "--query",
                    "What are the main limits of this document classification?",
                    "--json",
                ],
                data_dir=data_dir,
            )
            limits_payload = (
                json.loads(limits_process.stdout)
                if limits_process.stdout.strip()
                else {}
            )

            inspect_result = (
                inspect_payload.get("result", {}) if inspect_payload.get("ok") else {}
            )
            overview_answer = smoke_result.get("answer", {}).get("answer", "")
            type_answer = (
                type_payload.get("result", {}).get("answer", "")
                if type_payload.get("ok")
                else ""
            )
            purpose_answer = (
                purpose_payload.get("result", {}).get("answer", "")
                if purpose_payload.get("ok")
                else ""
            )
            audience_answer = (
                audience_payload.get("result", {}).get("answer", "")
                if audience_payload.get("ok")
                else ""
            )
            confidence_answer = (
                confidence_payload.get("result", {}).get("answer", "")
                if confidence_payload.get("ok")
                else ""
            )
            rationale_answer = (
                rationale_payload.get("result", {}).get("answer", "")
                if rationale_payload.get("ok")
                else ""
            )
            limits_answer = (
                limits_payload.get("result", {}).get("answer", "")
                if limits_payload.get("ok")
                else ""
            )
            confidence_support = (
                confidence_payload.get("result", {})
                .get("answer_trace", {})
                .get("support_trace", [])
            )
            confidence_support_item = (
                confidence_support[0] if confidence_support else {}
            )
            structure_confidence = inspect_result.get("structure_confidence")
            layout_confidence = inspect_result.get("layout_confidence")
            semantic_confidence = inspect_result.get("semantic_confidence")
            semantic_confidence_label = inspect_result.get("semantic_confidence_label")

            checks = [
                {"name": "smoke_ok", "passed": bool(smoke_payload.get("ok"))},
                {
                    "name": "smoke_all_pass",
                    "passed": bool(smoke_result.get("all_pass")),
                },
                {"name": "inspect_ok", "passed": bool(inspect_payload.get("ok"))},
                {
                    "name": "structure_confidence_present",
                    "passed": structure_confidence is not None,
                },
                {
                    "name": "layout_confidence_present",
                    "passed": layout_confidence is not None,
                },
                {
                    "name": "semantic_confidence_present",
                    "passed": semantic_confidence is not None,
                },
                {"name": "overview_answer_present", "passed": bool(overview_answer)},
                {"name": "type_answer_present", "passed": bool(type_answer)},
                {"name": "purpose_answer_present", "passed": bool(purpose_answer)},
                {"name": "audience_answer_present", "passed": bool(audience_answer)},
                {
                    "name": "confidence_answer_present",
                    "passed": bool(confidence_answer),
                },
                {"name": "rationale_answer_present", "passed": bool(rationale_answer)},
                {"name": "limits_answer_present", "passed": bool(limits_answer)},
                {
                    "name": "type_vs_purpose_distinct",
                    "passed": bool(
                        type_answer and purpose_answer and type_answer != purpose_answer
                    ),
                },
            ]

            result.update(
                {
                    "doc_id": doc_id,
                    "document_type": inspect_result.get("document_type")
                    or smoke_result.get("document", {}).get("document_type"),
                    "document_purpose": inspect_result.get("document_purpose")
                    or smoke_result.get("document", {}).get("document_purpose"),
                    "document_family": inspect_result.get("document_family")
                    or smoke_result.get("document", {}).get("document_family"),
                    "structure_confidence": structure_confidence,
                    "layout_confidence": layout_confidence,
                    "semantic_confidence": semantic_confidence,
                    "semantic_confidence_label": semantic_confidence_label,
                    "classification_status": confidence_support_item.get(
                        "classification_status"
                    ),
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
    has_specific_signal = _is_specific_document_type(
        item.get("document_type")
    ) or _is_specific_document_purpose(item.get("document_purpose"))
    semantic_confidence = item.get("semantic_confidence")
    semantic_confidence_value = (
        float(semantic_confidence) if semantic_confidence is not None else 0.0
    )
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


def _corpus_failure_example(
    item: dict[str, object], reasons: list[str]
) -> dict[str, object]:
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


def _corpus_bucket_diagnostics(
    sampled_results: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in sampled_results:
        grouped.setdefault(str(item.get("bucket") or "unknown"), []).append(item)

    diagnostics: dict[str, dict[str, object]] = {}
    for bucket, items in sorted(grouped.items()):
        total = len(items)
        structure_values = [
            float(item["structure_confidence"])
            for item in items
            if item.get("structure_confidence") is not None
        ]
        layout_values = [
            float(item["layout_confidence"])
            for item in items
            if item.get("layout_confidence") is not None
        ]
        semantic_values = [
            float(item["semantic_confidence"])
            for item in items
            if item.get("semantic_confidence") is not None
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
            "technical_pass_rate": _rate(
                sum(1 for item in items if bool(item.get("all_pass"))), total
            ),
            "semantic_pass_rate": _rate(
                sum(1 for item in items if bool(item.get("semantic_pass"))), total
            ),
            "specific_document_rate": _rate(
                sum(
                    1
                    for item in items
                    if _is_specific_document_type(item.get("document_type"))
                ),
                total,
            ),
            "specific_purpose_rate": _rate(
                sum(
                    1
                    for item in items
                    if _is_specific_document_purpose(item.get("document_purpose"))
                ),
                total,
            ),
            "low_confidence_rate": _rate(
                sum(
                    1
                    for item in items
                    if item.get("semantic_confidence_label") == "low"
                ),
                total,
            ),
            "trust_limited_rate": _rate(
                sum(1 for item in items if bool(item.get("trust_limited"))), total
            ),
            "avg_structure_confidence": _avg(structure_values),
            "avg_layout_confidence": _avg(layout_values),
            "avg_semantic_confidence": _avg(semantic_values),
            "dominant_failure_reasons": _count_values(reason_values),
            "failing_pdf_count": len(failing_pdfs),
            "failing_pdfs": failing_pdfs,
            "failure_examples": failure_examples,
        }
    return diagnostics


def _corpus_follow_up_actions(
    bucket_diagnostics: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
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
        key=lambda item: (
            priority_order.get(str(item["priority"]), 99),
            str(item["bucket"]),
        ),
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
            "passed": all(
                key in architecture_gates
                for key in ("bucket_gate_pass", "bucket_follow_up_count")
            ),
        },
        {
            "name": "follow_up_actions_have_required_fields",
            "passed": all(
                all(
                    key in item
                    for key in (
                        "bucket",
                        "priority",
                        "focus",
                        "reason",
                        "dominant_failure_reasons",
                    )
                )
                for item in follow_up_actions
            ),
        },
        {
            "name": "follow_up_count_matches_gate",
            "passed": architecture_gates.get("bucket_follow_up_count")
            == len(follow_up_actions),
        },
    ]
    return {
        "all_pass": all(bool(item["passed"]) for item in checks),
        "checks": checks,
    }


def _write_corpus_sanity_snapshot(payload: dict[str, object]) -> Path:
    PATHS.data_eval.mkdir(parents=True, exist_ok=True)
    snapshot_path = PATHS.data_eval / "corpus_sanity_snapshot.json"
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact_payload = _compact_corpus_sanity_snapshot(payload)
    compact_path = PATHS.data_eval / "corpus_sanity_compact_snapshot.json"
    compact_path.write_text(
        json.dumps(compact_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profile = str(payload.get("sample_profile") or "").strip().lower()
    if profile and re.fullmatch(r"[a-z0-9_-]+", profile):
        profile_path = PATHS.data_eval / f"corpus_sanity_{profile}_snapshot.json"
        profile_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        profile_compact_path = (
            PATHS.data_eval / f"corpus_sanity_{profile}_compact_snapshot.json"
        )
        profile_compact_path.write_text(
            json.dumps(compact_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return snapshot_path


def _compact_corpus_sanity_snapshot(payload: dict[str, object]) -> dict[str, object]:
    return {
        "sample_profile": payload.get("sample_profile"),
        "sample_size": payload.get("sample_size"),
        "summary": payload.get("summary", {}),
        "sample_manifest": payload.get("sample_manifest", {}),
        "bucket_diagnostics": payload.get("bucket_diagnostics", {}),
        "architecture_gates": payload.get("architecture_gates", {}),
        "corpus_contract": payload.get("corpus_contract", {}),
        "follow_up_actions": payload.get("follow_up_actions", []),
    }


def _compact_corpus_failure_examples(
    follow_up_actions: object, *, limit: int = 5
) -> list[dict[str, object]]:
    if not isinstance(follow_up_actions, list):
        return []
    examples: list[dict[str, object]] = []
    for action in follow_up_actions:
        if not isinstance(action, dict):
            continue
        bucket = action.get("bucket")
        focus = action.get("focus")
        priority = action.get("priority")
        for example in action.get("failure_examples", []):
            if not isinstance(example, dict):
                continue
            examples.append(
                {
                    "bucket": bucket,
                    "focus": focus,
                    "priority": priority,
                    "pdf": example.get("pdf"),
                    "doc_id": example.get("doc_id"),
                    "reasons": example.get("reasons", []),
                    "document_type": example.get("document_type"),
                    "document_purpose": example.get("document_purpose"),
                    "semantic_confidence": example.get("semantic_confidence"),
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def _corpus_snapshot_path_for_profile(profile: str) -> Path:
    safe_profile = profile.strip().lower()
    if safe_profile in {"latest", "default-latest", "default"}:
        return PATHS.data_eval / "corpus_sanity_snapshot.json"
    if safe_profile.endswith("-latest"):
        safe_profile = safe_profile.removesuffix("-latest")
    if not re.fullmatch(r"[a-z0-9_-]+", safe_profile):
        raise CliError(
            "invalid_profile",
            f"Invalid corpus profile name: {profile}",
            {"profile": profile},
        )
    return PATHS.data_eval / f"corpus_sanity_{safe_profile}_snapshot.json"


def _load_corpus_snapshot(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(
            "invalid_corpus_snapshot",
            f"Corpus snapshot is not valid JSON: {path}",
            {"path": str(path), "error": str(exc)},
        ) from exc
    return payload if isinstance(payload, dict) else None


def _corpus_snapshot_metric_summary(snapshot: dict[str, object]) -> dict[str, object]:
    summary = (
        snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
    )
    architecture_gates = (
        snapshot.get("architecture_gates", {})
        if isinstance(snapshot.get("architecture_gates"), dict)
        else {}
    )
    manifest = (
        snapshot.get("sample_manifest", {})
        if isinstance(snapshot.get("sample_manifest"), dict)
        else {}
    )
    return {
        "sample_profile": snapshot.get("sample_profile"),
        "sample_size": snapshot.get("sample_size"),
        "sample_checksum": manifest.get("selected_digest_checksum"),
        "technical_pass_rate": summary.get("technical_pass_rate"),
        "semantic_pass_rate": summary.get("semantic_pass_rate"),
        "avg_structure_confidence": summary.get("avg_structure_confidence"),
        "avg_layout_confidence": summary.get("avg_layout_confidence"),
        "avg_semantic_confidence": summary.get("avg_semantic_confidence"),
        "specific_document_rate": summary.get("specific_document_rate"),
        "specific_purpose_rate": summary.get("specific_purpose_rate"),
        "low_confidence_rate": summary.get("low_confidence_rate"),
        "trust_limited_rate": summary.get("trust_limited_rate"),
        "architecture_gate_pass": architecture_gates.get("all_pass"),
        "follow_up_count": len(snapshot.get("follow_up_actions", []))
        if isinstance(snapshot.get("follow_up_actions"), list)
        else None,
    }


def _numeric_delta(candidate: object, baseline: object) -> float | None:
    if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
        return round(float(candidate) - float(baseline), 4)
    return None


def _corpus_diff_summary(
    baseline_snapshot: dict[str, object],
    candidate_snapshot: dict[str, object],
    deltas: dict[str, float | None],
    regressions: list[str],
) -> dict[str, object]:
    baseline_buckets = baseline_snapshot.get("bucket_diagnostics", {})
    candidate_buckets = candidate_snapshot.get("bucket_diagnostics", {})
    baseline_bucket_names = (
        set(baseline_buckets) if isinstance(baseline_buckets, dict) else set()
    )
    candidate_bucket_names = (
        set(candidate_buckets) if isinstance(candidate_buckets, dict) else set()
    )
    sample_changed = _corpus_snapshot_metric_summary(baseline_snapshot).get(
        "sample_checksum"
    ) != _corpus_snapshot_metric_summary(candidate_snapshot).get("sample_checksum")
    checks = [
        {
            "name": "snapshots_loaded",
            "status": "pass",
            "reason": "both saved corpus snapshots were loaded",
        },
        {
            "name": "sample_checksum",
            "status": "skip" if sample_changed else "pass",
            "reason": "sample set changed; metric deltas may include corpus composition effects"
            if sample_changed
            else "sample checksum is unchanged",
        },
        {
            "name": "technical_pass_rate",
            "status": "fail" if "technical_pass_rate" in regressions else "pass",
            "reason": f"delta={deltas.get('technical_pass_rate')}",
        },
        {
            "name": "semantic_pass_rate",
            "status": "fail" if "semantic_pass_rate" in regressions else "pass",
            "reason": f"delta={deltas.get('semantic_pass_rate')}",
        },
        {
            "name": "trust_limited_rate",
            "status": "fail" if "trust_limited_rate" in regressions else "pass",
            "reason": f"delta={deltas.get('trust_limited_rate')}",
        },
        {
            "name": "follow_up_count",
            "status": "fail" if "follow_up_count" in regressions else "pass",
            "reason": f"delta={deltas.get('follow_up_count')}",
        },
        {
            "name": "bucket_set",
            "status": "skip"
            if baseline_bucket_names != candidate_bucket_names
            else "pass",
            "reason": "bucket set changed"
            if baseline_bucket_names != candidate_bucket_names
            else "bucket set is unchanged",
        },
    ]
    return {
        "all_pass": not any(item["status"] == "fail" for item in checks),
        "checks": checks,
        "bucket_changes": {
            "baseline_buckets": sorted(baseline_bucket_names),
            "candidate_buckets": sorted(candidate_bucket_names),
            "added": sorted(candidate_bucket_names - baseline_bucket_names),
            "removed": sorted(baseline_bucket_names - candidate_bucket_names),
        },
        "regression_metrics": regressions,
    }


def _corpus_review_status(
    *,
    candidate_summary: dict[str, object],
    regressions: list[str],
    diff_summary: dict[str, object],
) -> str:
    if not bool(candidate_summary.get("architecture_gate_pass")):
        return "fail"
    hard_regressions = {
        "technical_pass_rate",
        "semantic_pass_rate",
        "trust_limited_rate",
        "follow_up_count",
    }
    if any(metric in hard_regressions for metric in regressions):
        return "fail"
    if regressions or not bool(diff_summary.get("all_pass")):
        return "review"
    return "pass"


def _top_corpus_review_metrics(
    deltas: dict[str, float | None], regressions: list[str]
) -> list[dict[str, object]]:
    metric_priority = [
        "technical_pass_rate",
        "semantic_pass_rate",
        "trust_limited_rate",
        "follow_up_count",
        "avg_structure_confidence",
        "avg_layout_confidence",
        "avg_semantic_confidence",
        "specific_document_rate",
        "specific_purpose_rate",
        "low_confidence_rate",
    ]
    regression_set = set(regressions)
    scored: list[tuple[int, float, str, float | None]] = []
    for index, metric in enumerate(metric_priority):
        delta = deltas.get(metric)
        if delta is None:
            continue
        severity = 0 if metric in regression_set else 1
        scored.append((severity, -abs(float(delta)), metric, delta))
    return [
        {
            "metric": metric,
            "delta": delta,
            "status": "review" if metric in regression_set else "pass",
        }
        for _, _, metric, delta in sorted(scored)[:3]
    ]


def _corpus_model_experiment_scope(
    review_status: str, review_metrics: list[dict[str, object]]
) -> dict[str, object]:
    review_metric_names = [
        str(item.get("metric"))
        for item in review_metrics
        if item.get("status") == "review"
    ]
    semantics_metrics = {
        "semantic_pass_rate",
        "avg_semantic_confidence",
        "specific_document_rate",
        "specific_purpose_rate",
        "trust_limited_rate",
    }
    retrieval_or_structure_metrics = {
        "avg_structure_confidence",
        "avg_layout_confidence",
        "technical_pass_rate",
        "follow_up_count",
    }
    if review_status == "pass":
        return {
            "worth_running": False,
            "reason": "corpus profiles are stable; no opt-in model experiment is justified by this comparison",
            "candidate_backends": [],
            "target_metrics": [],
            "default_change_allowed": False,
        }
    candidate_backends: list[str] = []
    if any(metric in semantics_metrics for metric in review_metric_names):
        candidate_backends.append("sentence-transformers")
    if any(metric in retrieval_or_structure_metrics for metric in review_metric_names):
        candidate_backends.append("cross-encoder")
    if "trust_limited_rate" in review_metric_names:
        candidate_backends.append("llm-synthesis")
    if not candidate_backends and review_metric_names:
        candidate_backends.append("sentence-transformers")
    return {
        "worth_running": bool(candidate_backends),
        "reason": "run opt-in model experiments only against the listed review metrics",
        "candidate_backends": list(dict.fromkeys(candidate_backends)),
        "target_metrics": review_metric_names,
        "default_change_allowed": False,
    }


def _corpus_profile_compare_payload(
    *,
    baseline_profile: str = "quick",
    candidate_profile: str = "balanced",
    baseline_path: Path | None = None,
    candidate_path: Path | None = None,
) -> dict[str, object]:
    baseline_snapshot_path = baseline_path or _corpus_snapshot_path_for_profile(
        baseline_profile
    )
    candidate_snapshot_path = candidate_path or _corpus_snapshot_path_for_profile(
        candidate_profile
    )
    baseline_snapshot = _load_corpus_snapshot(baseline_snapshot_path)
    candidate_snapshot = _load_corpus_snapshot(candidate_snapshot_path)
    missing = [
        str(path)
        for path, snapshot in (
            (baseline_snapshot_path, baseline_snapshot),
            (candidate_snapshot_path, candidate_snapshot),
        )
        if snapshot is None
    ]
    if missing:
        return {
            "available": False,
            "all_pass": False,
            "missing_snapshots": missing,
            "baseline_path": str(baseline_snapshot_path),
            "candidate_path": str(candidate_snapshot_path),
            "recommendation": "Run corpus-sanity-check for both profiles before comparing saved snapshots.",
        }
    baseline_summary = _corpus_snapshot_metric_summary(baseline_snapshot or {})
    candidate_summary = _corpus_snapshot_metric_summary(candidate_snapshot or {})
    metric_names = [
        "technical_pass_rate",
        "semantic_pass_rate",
        "avg_structure_confidence",
        "avg_layout_confidence",
        "avg_semantic_confidence",
        "specific_document_rate",
        "specific_purpose_rate",
        "low_confidence_rate",
        "trust_limited_rate",
        "follow_up_count",
    ]
    deltas = {
        name: _numeric_delta(candidate_summary.get(name), baseline_summary.get(name))
        for name in metric_names
    }
    regressions = [
        name
        for name, delta in deltas.items()
        if delta is not None
        and (
            (
                name in {"low_confidence_rate", "trust_limited_rate", "follow_up_count"}
                and delta > 0
            )
            or (
                name
                not in {"low_confidence_rate", "trust_limited_rate", "follow_up_count"}
                and delta < 0
            )
        )
    ]
    checksum_changed = baseline_summary.get("sample_checksum") != candidate_summary.get(
        "sample_checksum"
    )
    diff_summary = _corpus_diff_summary(
        baseline_snapshot or {},
        candidate_snapshot or {},
        deltas,
        regressions,
    )
    review_status = _corpus_review_status(
        candidate_summary=candidate_summary,
        regressions=regressions,
        diff_summary=diff_summary,
    )
    review_metrics = _top_corpus_review_metrics(deltas, regressions)
    return {
        "available": True,
        "all_pass": not regressions
        and bool(candidate_summary.get("architecture_gate_pass")),
        "review_status": review_status,
        "baseline_path": str(baseline_snapshot_path),
        "candidate_path": str(candidate_snapshot_path),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "deltas": deltas,
        "regressions": regressions,
        "sample_changed": checksum_changed,
        "corpus_diff_summary": diff_summary,
        "corpus_review": {
            "status": review_status,
            "top_metrics": review_metrics,
            "sample_changed": checksum_changed,
            "bucket_changes": diff_summary.get("bucket_changes", {}),
            "model_experiment_scope": _corpus_model_experiment_scope(
                review_status, review_metrics
            ),
        },
        "recommendation": (
            "Candidate corpus profile is stable against baseline snapshot."
            if not regressions
            else "Inspect regression metrics before treating the candidate corpus profile as stable."
        ),
    }


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
                alias_path = (
                    alias_dir
                    / f"{alias_path.stem}-{entry.digest.lower()[:8]}{alias_path.suffix}"
                )
            shutil.copyfile(entry.pdf_path, alias_path)
            alias_paths.append(alias_path)
            display_paths[str(alias_path)] = str(entry.pdf_path)

        layout_payload = _run_layout_sanity_check(
            alias_paths, k=k, display_pdf_paths=display_paths
        )
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

    structure_values = [
        float(item["structure_confidence"])
        for item in sampled_results
        if item.get("structure_confidence") is not None
    ]
    layout_values = [
        float(item["layout_confidence"])
        for item in sampled_results
        if item.get("layout_confidence") is not None
    ]
    semantic_values = [
        float(item["semantic_confidence"])
        for item in sampled_results
        if item.get("semantic_confidence") is not None
    ]
    low_confidence_count = sum(
        1 for item in sampled_results if item.get("semantic_confidence_label") == "low"
    )
    trust_limited_count = sum(
        1 for item in sampled_results if bool(item.get("trust_limited"))
    )
    semantic_pass_count = sum(
        1 for item in sampled_results if bool(item.get("semantic_pass"))
    )
    specific_document_count = sum(
        1
        for item in sampled_results
        if _is_specific_document_type(item.get("document_type"))
    )
    specific_purpose_count = sum(
        1
        for item in sampled_results
        if _is_specific_document_purpose(item.get("document_purpose"))
    )
    generic_warning_count = sum(
        1
        for item in sampled_results
        if any(
            warning
            in {"generic_document_type", "generic_document_purpose", "generic_audience"}
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
        "bucket_counts": _count_values(
            [str(item.get("bucket", "")) for item in sampled_results]
        ),
        "document_type_counts": _count_values(
            [str(item.get("document_type", "")) for item in sampled_results]
        ),
        "document_purpose_counts": _count_values(
            [str(item.get("document_purpose", "")) for item in sampled_results]
        ),
        "classification_status_counts": _count_values(
            [str(item.get("classification_status", "")) for item in sampled_results]
        ),
        "trust_policy_counts": _count_values(
            [str(item.get("trust_policy", "")) for item in sampled_results]
        ),
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
    bucket_gate_pass = not any(
        str(action.get("priority")) == "high" for action in follow_up_actions
    )

    corpus_architecture_gates = {
        "all_pass": bool(layer_stability["all_pass"])
        and bool(layout_payload["all_pass"])
        and bucket_gate_pass,
        "technical_gate_pass": bool(layout_payload["all_pass"]),
        "layer_stability_pass": bool(layer_stability["all_pass"]),
        "semantic_gate_pass": bool(sampled_results)
        and semantic_pass_count == len(sampled_results),
        "bucket_gate_pass": bucket_gate_pass,
        "bucket_follow_up_count": len(follow_up_actions),
        "reasons": [
            *(
                []
                if layout_payload["all_pass"]
                else ["technical corpus smoke failures present"]
            ),
            *(
                []
                if layer_stability["all_pass"]
                else ["corpus layer thresholds not met"]
            ),
            *(
                []
                if sampled_results and semantic_pass_count == len(sampled_results)
                else ["not every sampled PDF reached semantic pass"]
            ),
            *(
                []
                if bucket_gate_pass
                else ["high-priority bucket-level follow-up actions are present"]
            ),
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
        "corpus_dir": str(corpus_paths[0])
        if corpus_paths
        else (str(corpus_dir) if corpus_dir else None),
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
        "semantic_all_pass": bool(sampled_results)
        and semantic_pass_count == len(sampled_results),
        "all_pass": bool(layout_payload["all_pass"]),
    }
    if save_snapshot:
        snapshot_path = _write_corpus_sanity_snapshot(payload)
        payload["snapshot_path"] = str(snapshot_path)
        snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload


def _run_public_surface_unittests() -> dict[str, object]:
    project_root = _discover_project_root(PATHS.root) or _discover_project_root(
        Path.cwd()
    )
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
    project_root = _discover_project_root(PATHS.root) or _discover_project_root(
        Path.cwd()
    )
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

            if (
                install_process is not None
                and install_process.returncode == 0
                and script_path.exists()
            ):
                readme_flow = _run_installed_readme_flow(
                    script_path, workspace, data_dir
                )

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
            "install_returncode": install_process.returncode
            if install_process
            else None,
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
    "unknown_document_semantics_core",
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
    maintainer_root = _discover_project_root(PATHS.root) or _discover_project_root(
        Path.cwd()
    )

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
    if local_corpus_sanity is not None and not bool(
        local_corpus_sanity.get("architecture_gates", {}).get("all_pass")
    ):
        recommendation["why"].append(
            "local unknown-document corpus gate is failing or trust-limited"
        )

    return {
        "doctor": doctor,
        "public_surface": {
            "all_pass": public_surface_all_pass,
            "smoke": public_smoke,
        },
        "maintainer_checks": {
            "available": maintainer_checks_available,
            "project_root": str(maintainer_root)
            if maintainer_root is not None
            else None,
            "all_pass": maintainer_release_all_pass
            if maintainer_checks_available
            else None,
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


def _gate_record(
    name: str, passed: bool | None, *, skipped: bool = False, reason: str | None = None
) -> dict[str, object]:
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
    local_corpus_result = (
        local_corpus.get("result") if isinstance(local_corpus, dict) else None
    )
    corpus_gates = (
        local_corpus_result.get("architecture_gates", {})
        if isinstance(local_corpus_result, dict)
        else {}
    )
    runtime = doctor.get("runtime", {}) if isinstance(doctor, dict) else {}
    regression_results = (
        regressions.get("results", []) if isinstance(regressions, dict) else []
    )
    shard_records = [
        {
            **_gate_record(str(item.get("shard")), bool(item.get("all_pass"))),
            "pass_count": item.get("pass_count"),
            "fail_count": item.get("fail_count"),
            "failed_case_ids": item.get("failed_case_ids", []),
        }
        for item in regression_results
    ]
    public_path_pass = bool(payload.get("public_surface", {}).get("all_pass"))
    benchmark_skipped = bool(regressions.get("skipped"))
    benchmark_pass = (
        bool(regressions.get("all_pass")) if not benchmark_skipped else None
    )
    corpus_available = (
        bool(local_corpus.get("available")) if isinstance(local_corpus, dict) else False
    )
    corpus_pass = bool(corpus_gates.get("all_pass")) if corpus_gates else None
    follow_up_actions = (
        local_corpus_result.get("follow_up_actions", [])
        if isinstance(local_corpus_result, dict)
        else []
    )
    product_gate = {
        "all_pass": public_path_pass
        and benchmark_pass is True
        and (corpus_pass is True or not corpus_available),
        "public_path": {
            "status": "pass" if public_path_pass else "fail",
            "reason": None if public_path_pass else "public CLI path is failing",
        },
        "benchmark": {
            "status": "skip"
            if benchmark_skipped
            else "pass"
            if benchmark_pass
            else "fail",
            "reason": "benchmark assets not present in the active data root"
            if benchmark_skipped
            else None,
        },
        "corpus": {
            "status": (
                "skip" if not corpus_available else "pass" if corpus_pass else "review"
            ),
            "reason": (
                "repo-local pdf/ corpus is not available or was not sampled"
                if not corpus_available
                else None
                if corpus_pass
                else "local unknown-document corpus gate needs review"
            ),
            "follow_up_count": len(follow_up_actions)
            if isinstance(follow_up_actions, list)
            else None,
            "failure_examples": _compact_corpus_failure_examples(follow_up_actions),
        },
    }
    return {
        "overall": _gate_record("overall", bool(payload.get("overall_pass"))),
        "recommendation": payload.get("recommendation", {}),
        "runtime_decision": runtime.get("runtime_decision", {}),
        "product_gate": product_gate,
        "public_path": {
            "all_pass": public_path_pass,
            "gates": [
                _gate_record(
                    "doctor_public_cli", bool(doctor.get("ready_for_public_cli"))
                ),
                _gate_record(
                    "public_smoke",
                    bool(
                        payload.get("public_surface", {})
                        .get("smoke", {})
                        .get("smoke_all_pass")
                    ),
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
                    skipped=bool(package_check.get("skipped"))
                    if package_check
                    else True,
                    reason=package_check.get("reason")
                    if isinstance(package_check, dict)
                    else "package_check_missing",
                ),
                _gate_record(
                    "unit_tests",
                    bool(unittests.get("passed")) if unittests else None,
                    skipped=bool(unittests.get("skipped")) if unittests else True,
                    reason=unittests.get("reason")
                    if isinstance(unittests, dict)
                    else "unit_tests_missing",
                ),
                _gate_record(
                    "internal_regressions",
                    bool(regressions.get("all_pass"))
                    if not regressions.get("skipped")
                    else None,
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
            "available": corpus_available,
            "gate": _gate_record(
                "local_corpus_architecture",
                bool(corpus_gates.get("all_pass")) if corpus_gates else None,
                skipped=not corpus_available,
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
            "failure_examples": _compact_corpus_failure_examples(follow_up_actions),
        },
    }


def _public_beta_check_compact_payload(
    release_payload: dict[str, object],
) -> dict[str, object]:
    release_summary = _release_check_compact_payload(release_payload)
    public_smoke = (
        release_payload.get("public_surface", {}).get("smoke", {})
        if isinstance(release_payload.get("public_surface"), dict)
        else {}
    )
    package_check = (
        release_payload.get("maintainer_checks", {}).get("package_check", {})
        if isinstance(release_payload.get("maintainer_checks"), dict)
        else {}
    )
    readme_flow = (
        package_check.get("readme_flow", {}) if isinstance(package_check, dict) else {}
    )
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
            runtime_decision.get("default_backend") == "auto",
            reason=runtime_decision.get("not_default_reason"),
        ),
        corpus_summary.get(
            "gate", _gate_record("local_corpus_architecture", None, skipped=True)
        ),
        release_summary.get(
            "overall", _gate_record("release_summary", None, skipped=True)
        ),
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
        "public_smoke_quality": public_smoke.get("quality_profile_summary", {}),
        "runtime_decision": runtime_decision,
        "corpus_quick": corpus_summary,
        "release_summary": release_summary,
        "scope": {
            "default_backend": "auto",
            "sentence_transformers": "default_when_cached",
            "cross_encoder": "experimental_opt_in_only",
            "llm_synthesis": "opt_in_only",
        },
    }


def _run_public_beta_check(k: int) -> dict[str, object]:
    return _public_beta_check_compact_payload(_run_release_check(k))
