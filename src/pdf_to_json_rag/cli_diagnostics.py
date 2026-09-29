"""Runtime, installation, and real-PDF diagnostic checks."""

from datetime import datetime, timezone
import importlib.util
from importlib import metadata as importlib_metadata
from importlib import resources as importlib_resources
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata

from . import __version__
from .answering import answer_query_with_retrieval
from .chunking import process_saved_document_to_chunks
from .config import PATHS
from .document_inventory import (
    load_document_inventory,
)
from .evaluation import (
    DEFAULT_RUNTIME_COMPARISON_REPORT_FILENAME,
    DEFAULT_RUNTIME_PROMOTION_SNAPSHOT_FILENAME,
)
from .extraction import process_native_pdf_to_json
from .indexing import build_local_index, embedding_runtime_diagnostics
from .pdf_inspector_adapter import pdf_inspector_runtime_status


from .cli_shared import (
    CANONICAL_COMMANDS,
    COMMAND_ALIASES,
    COMMAND_HELP,
    CliError,
    DEFAULT_REAL_PDF_GROUND_TRUTH_FILENAME,
    EXPECTED_EXAMPLE_FILES,
    _discover_project_root,
    _local_pdf_corpus_paths,
)


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
            "passed": bool(
                payload["plan"].get("query_class")
                and payload["plan"].get("answer_mode")
            ),
        },
        {
            "name": "answer_present",
            "passed": bool(answer.get("answer")),
        },
        {
            "name": "evidence_or_document_answer",
            "passed": bool(
                answer.get("evidence")
                or answer.get("answer_trace", {}).get("answer_mode")
                != "grounded_evidence"
            ),
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
        lines.append(
            "Run `pdf-to-json-rag help --topic <command>` for a concrete example."
        )
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
    aliases = sorted(
        alias for alias, target in COMMAND_ALIASES.items() if target == command
    )
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
        if promotion.get("promotion_ready")
        and promotion.get("candidate_mode") == "sentence-transformers"
        else None
    )
    not_default_reason = (
        "The public default is auto: use a locally cached sentence-transformer model when available, "
        "otherwise fall back to deterministic hash embeddings without downloading models."
    )
    return {
        "default_backend": "auto",
        "effective_backend": embedding.get("effective_backend"),
        "requested_backend": embedding.get("requested_backend"),
        "recommended_opt_in_backend": recommended_backend,
        "recommended_opt_in_source": "runtime_promotion_snapshot"
        if recommended_backend
        else None,
        "promotion_snapshot": promotion,
        "not_default_reason": not_default_reason,
        "cross_encoder_default": "disabled",
        "llm_synthesis_default": "disabled",
        "backend_policy": {
            "default_backend": {
                "name": "auto",
                "status": "default",
                "reason": "local real embeddings when cached; deterministic hash fallback otherwise",
                "fallback_backend": "hash",
                "preferred_backend": "sentence-transformers",
            },
            "recommended_opt_in_backend": {
                "name": recommended_backend,
                "status": "recommended_opt_in"
                if recommended_backend
                else "not_available",
                "reason": "promotion snapshot is green"
                if recommended_backend
                else "no green promotion snapshot found",
            },
            "experimental_backends": [
                {
                    "name": "cross-encoder",
                    "status": "experimental_opt_in",
                    "default_enabled": False,
                    "reason": "reranking comparison/fallback only; not part of default path",
                }
            ],
            "llm_synthesis": {
                "status": "opt_in_only",
                "default_enabled": False,
                "reason": "answer synthesis stays extractive unless a local command is explicitly configured",
            },
            "why_not_default": not_default_reason,
        },
    }


def _runtime_check_payload() -> dict[str, object]:
    embedding = embedding_runtime_diagnostics()
    return {
        "install_context": {
            "version": __version__,
            "python": sys.executable,
            "module_path": str(Path(__file__).with_name("cli.py").resolve()),
            "project_root": str(
                (
                    _discover_project_root(PATHS.root)
                    or _discover_project_root(Path.cwd())
                    or PATHS.root
                )
            ),
        },
        "embedding": embedding,
        "runtime_decision": _runtime_decision_payload(embedding),
        "llm_synthesis": {
            "env_var": "PDF_TO_JSON_RAG_LLM_COMMAND",
            "configured": bool(os.environ.get("PDF_TO_JSON_RAG_LLM_COMMAND")),
            "provider": "local_command"
            if os.environ.get("PDF_TO_JSON_RAG_LLM_COMMAND")
            else None,
            "default_enabled": False,
        },
        "cross_encoder": {
            "env_var": "PDF_TO_JSON_RAG_USE_CROSS_ENCODER",
            "configured": os.environ.get("PDF_TO_JSON_RAG_USE_CROSS_ENCODER") == "1",
            "default_enabled": False,
        },
        "default_policy": {
            "embedding_backend": "auto",
            "sentence_transformers": "default_when_cached",
            "cross_encoder": "opt_in",
            "llm_synthesis": "opt_in",
        },
    }


def _temporary_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_real_pdf_ground_truth_cases(
    eval_path: Path | None = None,
) -> list[dict[str, object]]:
    path = eval_path or (
        PATHS.root / "data" / "eval" / DEFAULT_REAL_PDF_GROUND_TRUTH_FILENAME
    )
    if not path.exists():
        path = PATHS.data_eval / DEFAULT_REAL_PDF_GROUND_TRUTH_FILENAME
    if not path.exists():
        raise CliError(
            "missing_real_pdf_ground_truth_cases",
            f"Real-PDF ground-truth case file was not found: {path}",
            {"path": str(path)},
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise CliError(
            "invalid_real_pdf_ground_truth_cases",
            "Real-PDF ground-truth file must contain a list of cases.",
            {"path": str(path)},
        )
    return [case for case in data if isinstance(case, dict)]


def _keyword_coverage(text: str, keywords: list[str]) -> dict[str, object]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    compact_text = re.sub(r"[^a-z0-9]+", "", normalized)
    matches = []
    for keyword in keywords:
        normalized_keyword = unicodedata.normalize("NFKC", keyword).casefold()
        compact_keyword = re.sub(r"[^a-z0-9]+", "", normalized_keyword)
        if normalized_keyword in normalized or (
            len(compact_keyword) >= 4 and compact_keyword in compact_text
        ):
            matches.append(keyword)
    return {
        "coverage": round(len(matches) / max(len(keywords), 1), 3),
        "matched": matches,
        "missing": [keyword for keyword in keywords if keyword not in matches],
    }


def _rank_score(
    retrieved_doc_ids: list[str], expected_doc_id: str
) -> dict[str, object]:
    if expected_doc_id not in retrieved_doc_ids:
        return {"recall_at_k": 0.0, "reciprocal_rank": 0.0, "rank": None}
    rank = retrieved_doc_ids.index(expected_doc_id) + 1
    return {"recall_at_k": 1.0, "reciprocal_rank": round(1.0 / rank, 3), "rank": rank}


def _prepare_real_pdf_ground_truth_workspace(
    *,
    cases: list[dict[str, object]],
    workspace: Path,
    env_updates: dict[str, str | None],
    corpus_dir: Path | None = None,
) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
    if corpus_dir is not None:
        pdf_dir = corpus_dir.expanduser().resolve()
        if not pdf_dir.is_dir():
            raise CliError(
                "missing_local_pdf_corpus",
                f"The requested real-PDF corpus directory was not found: {pdf_dir}",
                {"corpus_dir": str(pdf_dir)},
            )
    else:
        corpus_paths = _local_pdf_corpus_paths(None)
        if corpus_paths is None:
            raise CliError(
                "missing_local_pdf_corpus",
                "The repo-local pdf/ corpus is required for real-ground-truth-check.",
                {"corpus_dir": "pdf"},
            )
        pdf_dir, _metadata_path = corpus_paths
    document_dir = workspace / "documents"
    chunk_root = workspace / "chunks"
    index_dir = workspace / "index"
    document_dir.mkdir(parents=True, exist_ok=True)
    chunk_root.mkdir(parents=True, exist_ok=True)
    digests = sorted(
        {
            str(case.get("pdf_digest", "")).strip()
            for case in cases
            if case.get("pdf_digest")
        }
    )
    if not digests:
        raise CliError(
            "empty_real_pdf_ground_truth_cases",
            "Real-PDF ground-truth cases must include at least one pdf_digest.",
            {},
        )

    digest_to_doc_id: dict[str, str] = {}
    processing_docs: list[dict[str, object]] = []
    all_chunks = []
    previous = _temporary_env(env_updates)
    try:
        for digest in digests:
            pdf_path = pdf_dir / f"{digest}.pdf"
            if not pdf_path.exists():
                raise CliError(
                    "missing_real_pdf_ground_truth_pdf",
                    f"Ground-truth PDF was not found: {pdf_path}",
                    {"pdf_digest": digest, "path": str(pdf_path)},
                )
            extraction, document, native_path, document_path = (
                process_native_pdf_to_json(
                    pdf_path=pdf_path,
                    output_dir=document_dir,
                )
            )
            chunked_document, chunks, _saved_paths = process_saved_document_to_chunks(
                native_path=native_path,
                document_path=document_path,
                output_dir=chunk_root,
            )
            all_chunks.extend(chunks)
            digest_to_doc_id[digest] = chunked_document.doc_id
            covered_pages = sorted(
                {
                    page
                    for chunk in chunks
                    for page in range(chunk.page_start, chunk.page_end + 1)
                }
            )
            processing_docs.append(
                {
                    "pdf_digest": digest,
                    "source_pdf": pdf_path.name,
                    "doc_id": chunked_document.doc_id,
                    "title": chunked_document.title,
                    "document_type": chunked_document.document_type,
                    "document_purpose": chunked_document.document_purpose,
                    "document_family": chunked_document.document_family,
                    "page_count": chunked_document.page_count,
                    "chunk_count": len(chunks),
                    "covered_pages": covered_pages,
                    "page_coverage": round(
                        len(covered_pages) / max(chunked_document.page_count, 1), 3
                    ),
                    "structure_confidence": chunked_document.structure_confidence,
                    "layout_confidence": chunked_document.layout_confidence,
                    "semantic_confidence": chunked_document.semantic_confidence,
                    "semantic_warnings": list(chunked_document.semantic_warnings),
                }
            )
        manifest = build_local_index(chunks=all_chunks, index_dir=index_dir)
    finally:
        _restore_env(previous)

    return (
        digest_to_doc_id,
        manifest,
        {
            "documents": processing_docs,
            "chunk_root": str(chunk_root),
            "index_dir": str(index_dir),
        },
    )


def _evaluate_real_pdf_mode(
    *,
    mode: str,
    cases: list[dict[str, object]],
    digest_to_doc_id: dict[str, str],
    index_dir: Path,
    chunk_root: Path,
    k: int,
    env_updates: dict[str, str | None],
) -> dict[str, object]:
    previous = _temporary_env(env_updates)
    try:
        case_results: list[dict[str, object]] = []
        for case in cases:
            digest = str(case.get("pdf_digest", ""))
            expected_doc_id = digest_to_doc_id.get(digest)
            expected_keywords = [
                str(item) for item in case.get("expected_keywords", [])
            ]
            evidence_keywords = [
                str(item) for item in case.get("evidence_keywords", [])
            ]
            answer = answer_query_with_retrieval(
                query=str(case["query"]),
                index_dir=index_dir,
                chunk_root=chunk_root,
                k=k,
                use_lightweight_rerank=True,
            )
            retrieved_doc_ids = []
            for chunk in answer.top_k_hits:
                if chunk.doc_id not in retrieved_doc_ids:
                    retrieved_doc_ids.append(chunk.doc_id)
            selected_chunk_ids = [chunk.chunk_id for chunk in answer.top_k_hits[:k]]
            rank = _rank_score(retrieved_doc_ids, expected_doc_id or "")
            support_text = "\n".join(
                [
                    answer.answer,
                    *[
                        "\n".join(
                            [
                                chunk.section_title or "",
                                " > ".join(chunk.section_path),
                                chunk.section_summary or "",
                                " ".join(chunk.section_coverage_terms),
                                chunk.text,
                            ]
                        )
                        for chunk in answer.top_k_hits
                    ],
                    *[
                        "\n".join(
                            [
                                chunk.section_title or "",
                                " > ".join(chunk.section_path),
                                chunk.section_summary or "",
                                " ".join(chunk.section_coverage_terms),
                                chunk.text,
                            ]
                        )
                        for chunk in answer.expanded_hits
                    ],
                    *[item.sentence for item in answer.evidence],
                ]
            )
            answer_keyword_eval = _keyword_coverage(answer.answer, expected_keywords)
            evidence_keyword_eval = _keyword_coverage(support_text, evidence_keywords)
            synthesis_runtime = answer.answer_trace.get("synthesis_runtime", {})
            cross_encoder_fallback = any(
                bool(chunk.retrieval_signals.get("cross_encoder_fallback"))
                for chunk in answer.top_k_hits + answer.expanded_hits
            )
            passed = (
                bool(rank["recall_at_k"])
                and float(evidence_keyword_eval["coverage"]) >= 0.67
            )
            case_results.append(
                {
                    "case_id": case.get("case_id"),
                    "bucket": case.get("bucket"),
                    "query": case.get("query"),
                    "expected_doc_id": expected_doc_id,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "selected_chunk_ids": selected_chunk_ids,
                    "rank": rank["rank"],
                    "recall_at_k": rank["recall_at_k"],
                    "reciprocal_rank": rank["reciprocal_rank"],
                    "answer_keyword_coverage": answer_keyword_eval,
                    "evidence_keyword_coverage": evidence_keyword_eval,
                    "passed": passed,
                    "answer_preview": answer.answer[:240],
                    "runtime": {
                        "synthesis_runtime": synthesis_runtime,
                        "cross_encoder_fallback": cross_encoder_fallback,
                    },
                }
            )
    finally:
        _restore_env(previous)

    grounded = case_results
    pass_count = sum(1 for item in grounded if item["passed"])
    answer_quality = _real_pdf_answer_quality_summary(grounded)
    weak_case_workbench = _real_pdf_weak_case_workbench(grounded)
    return {
        "mode": mode,
        "case_count": len(grounded),
        "pass_count": pass_count,
        "fail_count": len(grounded) - pass_count,
        "all_pass": pass_count == len(grounded),
        "summary": {
            "avg_recall_at_k": round(
                sum(float(item["recall_at_k"]) for item in grounded)
                / max(len(grounded), 1),
                3,
            ),
            "mrr": round(
                sum(float(item["reciprocal_rank"]) for item in grounded)
                / max(len(grounded), 1),
                3,
            ),
            "avg_answer_keyword_coverage": round(
                sum(
                    float(item["answer_keyword_coverage"]["coverage"])
                    for item in grounded
                )
                / max(len(grounded), 1),
                3,
            ),
            "avg_evidence_keyword_coverage": round(
                sum(
                    float(item["evidence_keyword_coverage"]["coverage"])
                    for item in grounded
                )
                / max(len(grounded), 1),
                3,
            ),
            "cross_encoder_fallback_case_count": sum(
                1 for item in grounded if item["runtime"]["cross_encoder_fallback"]
            ),
            "llm_used_case_count": sum(
                1
                for item in grounded
                if item["runtime"]["synthesis_runtime"].get("used_for_final_answer")
            ),
        },
        "answer_quality": answer_quality,
        "weak_case_workbench": weak_case_workbench,
        "failed_case_ids": [
            str(item["case_id"]) for item in grounded if not item["passed"]
        ],
        "case_results": grounded,
    }


def _real_pdf_answer_quality_summary(
    case_results: list[dict[str, object]],
    *,
    threshold: float = 0.67,
) -> dict[str, object]:
    weak_cases = [
        item
        for item in case_results
        if float(item.get("answer_keyword_coverage", {}).get("coverage") or 0.0)
        < threshold
    ]
    bucket_counts: dict[str, int] = {}
    for item in weak_cases:
        bucket = str(item.get("bucket") or "unknown")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    abstained_case_ids = [
        str(item.get("case_id"))
        for item in case_results
        if str(item.get("answer_preview") or "").startswith("No grounded answer")
    ]
    if not weak_cases:
        recommended_next_action = (
            "answer coverage is healthy for the current real-PDF set"
        )
    elif len(weak_cases) <= 3:
        recommended_next_action = "review the remaining weak answer cases directly"
    else:
        recommended_next_action = (
            "improve extractive synthesis for buckets with weak answer coverage"
        )
    return {
        "threshold": threshold,
        "weak_case_count": len(weak_cases),
        "weak_case_ids": [str(item.get("case_id")) for item in weak_cases],
        "abstained_case_count": len(abstained_case_ids),
        "abstained_case_ids": abstained_case_ids,
        "weak_buckets": [
            {"bucket": bucket, "weak_case_count": count}
            for bucket, count in sorted(
                bucket_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ],
        "recommended_next_action": recommended_next_action,
    }


def _real_pdf_weak_case_workbench(
    case_results: list[dict[str, object]],
    *,
    threshold: float = 0.67,
    limit: int = 12,
) -> dict[str, object]:
    weak_cases = [
        item
        for item in case_results
        if float(item.get("answer_keyword_coverage", {}).get("coverage") or 0.0)
        < threshold
    ]
    weak_cases.sort(
        key=lambda item: (
            float(item.get("answer_keyword_coverage", {}).get("coverage") or 0.0),
            str(item.get("bucket") or ""),
            str(item.get("case_id") or ""),
        )
    )
    items = []
    for item in weak_cases[:limit]:
        answer_eval = item.get("answer_keyword_coverage", {})
        evidence_eval = item.get("evidence_keyword_coverage", {})
        selected_docs = [str(doc_id) for doc_id in item.get("retrieved_doc_ids", [])]
        selected_chunks = [
            str(chunk_id) for chunk_id in item.get("selected_chunk_ids", [])
        ]
        missing = [str(value) for value in answer_eval.get("missing", [])]
        evidence_missing = [str(value) for value in evidence_eval.get("missing", [])]
        if evidence_missing:
            action = "fix retrieval/evidence support first"
        elif str(item.get("answer_preview") or "").startswith("No grounded answer"):
            action = "add or tune extractive fallback for this query/layout"
        else:
            action = "improve answer extraction around missing keywords"
        items.append(
            {
                "case_id": str(item.get("case_id")),
                "bucket": str(item.get("bucket")),
                "query": str(item.get("query")),
                "answer_coverage": answer_eval.get("coverage"),
                "evidence_coverage": evidence_eval.get("coverage"),
                "expected_keywords": [*answer_eval.get("matched", []), *missing],
                "missing_keywords": missing,
                "selected_docs": selected_docs,
                "selected_chunks": selected_chunks,
                "expected_doc_rank": item.get("rank"),
                "answer_preview": str(item.get("answer_preview") or "")[:300],
                "suggested_action": action,
            }
        )
    return {
        "threshold": threshold,
        "limit": limit,
        "weak_case_count": len(weak_cases),
        "reported_case_count": len(items),
        "cases": items,
    }


def _real_pdf_runtime_decisions(
    default_result: dict[str, object], mode_results: list[dict[str, object]]
) -> dict[str, object]:
    by_mode = {str(item.get("mode")): item for item in mode_results}
    default_summary = default_result.get("summary", {})
    decisions: list[dict[str, object]] = []
    cross = by_mode.get("cross-encoder")
    if cross:
        cross_summary = cross.get("summary", {})
        fallback_count = int(
            cross_summary.get("cross_encoder_fallback_case_count") or 0
        )
        improved = int(cross.get("pass_count", 0)) > int(
            default_result.get("pass_count", 0)
        ) or float(cross_summary.get("mrr") or 0.0) > float(
            default_summary.get("mrr") or 0.0
        )
        degraded = int(cross.get("pass_count", 0)) < int(
            default_result.get("pass_count", 0)
        ) or float(cross_summary.get("mrr") or 0.0) < float(
            default_summary.get("mrr") or 0.0
        )
        if fallback_count:
            status = "not_ready"
            reason = "cross-encoder fallback occurred on the real-PDF eval"
        elif improved:
            status = "recommended_opt_in"
            reason = "cross-encoder improved real-PDF ranking without fallback"
        elif degraded:
            status = "do_not_promote"
            reason = "cross-encoder regressed real-PDF ranking"
        else:
            status = "keep_experimental"
            reason = "cross-encoder did not improve the real-PDF eval"
        decisions.append(
            {
                "backend": "cross-encoder",
                "status": status,
                "default_change_allowed": False,
                "reason": reason,
            }
        )
    llm = by_mode.get("llm-synthesis")
    if llm:
        llm_summary = llm.get("summary", {})
        configured = bool(os.environ.get("PDF_TO_JSON_RAG_LLM_COMMAND"))
        used = int(llm_summary.get("llm_used_case_count") or 0)
        improved_answer = float(
            llm_summary.get("avg_answer_keyword_coverage") or 0.0
        ) > float(default_summary.get("avg_answer_keyword_coverage") or 0.0)
        if not configured:
            status = "skipped_not_configured"
            reason = "PDF_TO_JSON_RAG_LLM_COMMAND is not configured"
        elif used and improved_answer:
            status = "valuable_opt_in"
            reason = (
                "LLM synthesis improved answer keyword coverage on the real-PDF eval"
            )
        else:
            status = "no_value_added"
            reason = "LLM synthesis did not improve the real-PDF eval"
        decisions.append(
            {
                "backend": "llm-synthesis",
                "status": status,
                "default_change_allowed": False,
                "reason": reason,
            }
        )
    return {
        "default_backend": "auto",
        "default_change_allowed": False,
        "decisions": decisions,
    }


def _real_pdf_modes_from_arg(value: str | None) -> list[str]:
    if not value:
        return ["default-auto"]
    raw_modes = [item.strip() for item in value.split(",") if item.strip()]
    if "all" in raw_modes:
        return ["default-auto", "hash-baseline", "cross-encoder", "llm-synthesis"]
    allowed = {"default-auto", "hash-baseline", "cross-encoder", "llm-synthesis"}
    modes = []
    for mode in raw_modes:
        if mode not in allowed:
            raise CliError(
                "invalid_real_pdf_mode",
                f"Unsupported real-ground-truth-check mode: {mode}",
                {"mode": mode, "allowed": sorted(allowed | {"all"})},
            )
        modes.append(mode)
    return modes or ["default-auto", "hash-baseline"]


def _run_real_pdf_ground_truth_check(
    k: int,
    eval_path: Path | None = None,
    modes: list[str] | None = None,
    corpus_dir: Path | None = None,
) -> dict[str, object]:
    cases = _load_real_pdf_ground_truth_cases(eval_path)
    selected_modes = modes or [
        "default-auto",
        "hash-baseline",
        "cross-encoder",
        "llm-synthesis",
    ]
    default_env = {
        "PDF_TO_JSON_RAG_EMBEDDING_BACKEND": None,
        "PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS": None,
        "PDF_TO_JSON_RAG_ALLOW_MODEL_DOWNLOAD": None,
    }
    hash_env = {"PDF_TO_JSON_RAG_EMBEDDING_BACKEND": "hash"}
    with tempfile.TemporaryDirectory(prefix="pdf-to-json-rag-real-gt-") as workspace:
        workspace_path = Path(workspace)
        digest_to_doc_id, default_manifest, processing = (
            _prepare_real_pdf_ground_truth_workspace(
                cases=cases,
                workspace=workspace_path / "default_auto",
                env_updates=default_env,
                corpus_dir=corpus_dir,
            )
        )
        default_index_dir = Path(str(processing["index_dir"]))
        default_chunk_root = Path(str(processing["chunk_root"]))
        hash_manifest: dict[str, object] = {}
        hash_index_dir: Path | None = None
        hash_chunk_root: Path | None = None
        if "hash-baseline" in selected_modes:
            _, hash_manifest, hash_processing = (
                _prepare_real_pdf_ground_truth_workspace(
                    cases=cases,
                    workspace=workspace_path / "hash",
                    env_updates=hash_env,
                    corpus_dir=corpus_dir,
                )
            )
            hash_index_dir = Path(str(hash_processing["index_dir"]))
            hash_chunk_root = Path(str(hash_processing["chunk_root"]))
        mode_results = []
        if "default-auto" in selected_modes:
            mode_results.append(
                _evaluate_real_pdf_mode(
                    mode="default-auto",
                    cases=cases,
                    digest_to_doc_id=digest_to_doc_id,
                    index_dir=default_index_dir,
                    chunk_root=default_chunk_root,
                    k=k,
                    env_updates={
                        "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
                        "PDF_TO_JSON_RAG_LLM_COMMAND": None,
                    },
                )
            )
        if "hash-baseline" in selected_modes:
            if hash_index_dir is None or hash_chunk_root is None:
                raise CliError(
                    "missing_hash_ground_truth_workspace",
                    "Hash-baseline mode was requested but its workspace was not prepared.",
                    {},
                )
            mode_results.append(
                _evaluate_real_pdf_mode(
                    mode="hash-baseline",
                    cases=cases,
                    digest_to_doc_id=digest_to_doc_id,
                    index_dir=hash_index_dir,
                    chunk_root=hash_chunk_root,
                    k=k,
                    env_updates={
                        "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
                        "PDF_TO_JSON_RAG_LLM_COMMAND": None,
                    },
                )
            )
        if "cross-encoder" in selected_modes:
            mode_results.append(
                _evaluate_real_pdf_mode(
                    mode="cross-encoder",
                    cases=cases,
                    digest_to_doc_id=digest_to_doc_id,
                    index_dir=default_index_dir,
                    chunk_root=default_chunk_root,
                    k=k,
                    env_updates={
                        "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": "1",
                        "PDF_TO_JSON_RAG_LLM_COMMAND": None,
                        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                        "TRANSFORMERS_OFFLINE": os.environ.get(
                            "TRANSFORMERS_OFFLINE", "1"
                        ),
                    },
                )
            )
        if "llm-synthesis" in selected_modes:
            mode_results.append(
                _evaluate_real_pdf_mode(
                    mode="llm-synthesis",
                    cases=cases,
                    digest_to_doc_id=digest_to_doc_id,
                    index_dir=default_index_dir,
                    chunk_root=default_chunk_root,
                    k=k,
                    env_updates={"PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None},
                )
            )

    default_result = next(
        (item for item in mode_results if item.get("mode") == "default-auto"),
        mode_results[0],
    )
    processing_docs = (
        list(processing["documents"])
        if isinstance(processing.get("documents"), list)
        else []
    )
    processing_gate = {
        "all_pass": all(
            float(doc.get("page_coverage") or 0.0) >= 1.0
            and int(doc.get("chunk_count") or 0) > 0
            for doc in processing_docs
        ),
        "documents": processing_docs,
    }
    hash_result = next(
        (item for item in mode_results if item.get("mode") == "hash-baseline"),
        default_result,
    )
    hash_summary = hash_result["summary"]
    default_summary = default_result["summary"]
    quality_gate = {
        "passed": (
            int(default_result.get("pass_count") or 0)
            / max(int(default_result.get("case_count") or 0), 1)
            >= 0.75
            and float(default_summary.get("mrr") or 0.0) >= 0.75
            and float(default_summary.get("avg_evidence_keyword_coverage") or 0.0)
            >= 0.75
        ),
        "min_pass_rate": 0.75,
        "min_mrr": 0.75,
        "min_evidence_keyword_coverage": 0.75,
    }
    return {
        "case_file": str(
            eval_path
            or (PATHS.root / "data" / "eval" / DEFAULT_REAL_PDF_GROUND_TRUTH_FILENAME)
        ),
        "k": k,
        "modes": selected_modes,
        "case_count": len(cases),
        "pdf_count": len(processing_docs),
        "default_index_manifest": {
            "embedding_backend": default_manifest.get("embedding_backend"),
            "embedding_model": default_manifest.get("embedding_model"),
            "embedding_requested_backend": default_manifest.get(
                "embedding_requested_backend"
            ),
            "embedding_fallback_reason": default_manifest.get(
                "embedding_fallback_reason"
            ),
            "chunk_count": default_manifest.get("chunk_count"),
        },
        "hash_index_manifest": {
            "embedding_backend": hash_manifest.get("embedding_backend"),
            "embedding_model": hash_manifest.get("embedding_model"),
            "embedding_requested_backend": hash_manifest.get(
                "embedding_requested_backend"
            ),
            "chunk_count": hash_manifest.get("chunk_count"),
        },
        "processing_quality": processing_gate,
        "mode_results": mode_results,
        "default_vs_hash": {
            "mrr_delta": round(
                float(default_summary.get("mrr") or 0.0)
                - float(hash_summary.get("mrr") or 0.0),
                3,
            ),
            "recall_delta": round(
                float(default_summary.get("avg_recall_at_k") or 0.0)
                - float(hash_summary.get("avg_recall_at_k") or 0.0),
                3,
            ),
            "answer_keyword_coverage_delta": round(
                float(default_summary.get("avg_answer_keyword_coverage") or 0.0)
                - float(hash_summary.get("avg_answer_keyword_coverage") or 0.0),
                3,
            ),
        },
        "runtime_decisions": _real_pdf_runtime_decisions(default_result, mode_results),
        "quality_gate": quality_gate,
        "all_pass": bool(default_result.get("all_pass"))
        and bool(processing_gate["all_pass"]),
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


def _write_runtime_promotion_snapshot(
    report: dict[str, object], report_path: Path
) -> Path | None:
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
        "candidate_index_manifest": _portable_snapshot_value(
            candidate.get("index_manifest", {})
        ),
        "baseline_deltas": report.get("baseline_deltas", {}).get(
            "sentence-transformers", {}
        ),
        "promotion_gate": _portable_snapshot_value(gate),
        "recommended_default_change": False,
        "recommendation": "Sentence-transformers is validated as the preferred auto backend when cached locally; deterministic hash remains the offline fallback.",
    }
    snapshot_path = PATHS.data_eval / DEFAULT_RUNTIME_PROMOTION_SNAPSHOT_FILENAME
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return snapshot_path


def _model_decision_gate_from_runtime_report(
    report: dict[str, object],
) -> dict[str, object]:
    mode_results = {
        str(item.get("mode")): item
        for item in report.get("mode_results", [])
        if isinstance(item, dict)
    }
    promotion_gates = (
        report.get("promotion_gates", {})
        if isinstance(report.get("promotion_gates"), dict)
        else {}
    )
    baseline = mode_results.get("baseline", {})
    decisions: list[dict[str, object]] = []

    sentence_gate = (
        promotion_gates.get("sentence-transformers", {})
        if isinstance(promotion_gates, dict)
        else {}
    )
    sentence_result = mode_results.get("sentence-transformers", {})
    if sentence_result:
        decisions.append(
            {
                "backend": "sentence-transformers",
                "status": "recommended_opt_in"
                if sentence_gate.get("promotable")
                else "opt_in_review",
                "model_helped": bool(sentence_gate.get("promotable")),
                "default_change_allowed": False,
                "reason": (
                    "green full-suite promotion gate; keep as explicit opt-in"
                    if sentence_gate.get("promotable")
                    else "promotion gate is not green"
                ),
            }
        )

    cross_result = mode_results.get("cross-encoder", {})
    if cross_result:
        baseline_pass = int(baseline.get("pass_count", 0) or 0)
        cross_pass = int(cross_result.get("pass_count", 0) or 0)
        fallback_count = int(
            cross_result.get("runtime_signals", {}).get(
                "cross_encoder_fallback_count", 0
            )
            or 0
        )
        decisions.append(
            {
                "backend": "cross-encoder",
                "status": "experimental_opt_in",
                "model_helped": cross_pass > baseline_pass and fallback_count == 0,
                "default_change_allowed": False,
                "reason": "only promote for concrete failure buckets; fallback/reporting path otherwise",
            }
        )

    llm_result = mode_results.get("llm-synthesis", {})
    if llm_result:
        llm_used = int(
            llm_result.get("runtime_signals", {}).get("llm_used_case_count", 0) or 0
        )
        decisions.append(
            {
                "backend": "llm-synthesis",
                "status": "opt_in_only",
                "model_helped": False,
                "default_change_allowed": False,
                "reason": (
                    "local command was invoked for comparison; keep synthesis opt-in"
                    if llm_used
                    else "LLM synthesis not configured or not used"
                ),
            }
        )

    return {
        "default_backend": "auto",
        "default_change_allowed": False,
        "decisions": decisions,
        "next_step": (
            "keep auto default; run cross-encoder or LLM experiments only for measured corpus review buckets"
            if decisions
            else "run compare-runtime-modes before making model decisions"
        ),
    }


def _runtime_promotion_report_payload(
    report_path: Path | None = None,
) -> dict[str, object]:
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
        "Sentence-transformer embeddings are promotion-ready and are used by the auto default when cached locally."
        if promotion_ready
        else "Auto default will use deterministic hash fallback until the sentence-transformer gate is green and the model is cached locally."
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
        "default_decision": {
            "default_backend": "auto",
            "preferred_backend_when_cached": "sentence-transformers"
            if promotion_ready
            else None,
            "fallback_backend": "hash",
            "cross_encoder": "experimental_opt_in_only",
            "llm_synthesis": "opt_in_only",
            "why_not_default": (
                "The auto default uses sentence-transformers only when the model is cached locally; "
                "hash remains the offline-safe fallback."
                if promotion_ready
                else "No green full-suite promotion gate is available, so auto falls back to hash unless explicitly configured."
            ),
        },
        "model_decision_gate": _model_decision_gate_from_runtime_report(report),
        "recommendation": recommendation,
    }


def _doctor_checks() -> dict[str, object]:
    package_metadata_present, package_metadata_details = _project_metadata_available()
    examples_dir = _available_examples_dir()
    runtime = _runtime_check_payload()
    pdf_inspector_runtime = pdf_inspector_runtime_status()
    manifest_candidates = [
        PATHS.data_index / "index_manifest.json",
        PATHS.data_index / "workflow_smoke" / "index_manifest.json",
    ]
    selected_manifest = next(
        (path for path in manifest_candidates if path.exists()), manifest_candidates[0]
    )
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
            "passed": all(
                path.exists()
                for path in (
                    PATHS.data_input,
                    PATHS.data_documents,
                    PATHS.data_chunks,
                    PATHS.data_index,
                    PATHS.data_eval,
                )
            ),
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
            "name": "pdf_inspector_available",
            "passed": bool(pdf_inspector_runtime["available"]),
            "category": "required_public_tool",
            "details": pdf_inspector_runtime,
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
            "passed": examples_dir.exists()
            and all((examples_dir / name).exists() for name in EXPECTED_EXAMPLE_FILES),
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
                'Run `pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json` to build a demo index and validate retrieval.',
            ]
        )
    else:
        next_steps.extend(
            [
                "Run `pdf-to-json-rag inspect-document --doc-id <doc_id> --json` to inspect extracted document metadata.",
                'Run `pdf-to-json-rag answer-query --query "What does this file cover?" --json` against the current local index.',
            ]
        )
    return {
        "checks": checks,
        "ready_for_public_cli": ready_for_public_cli,
        "ready_for_retrieval": ready_for_retrieval,
        "ready_for_internal_benchmark": ready_for_internal_benchmark,
        "data_root": str(PATHS.data_dir),
        "project_root": str(
            (
                _discover_project_root(PATHS.root)
                or _discover_project_root(Path.cwd())
                or PATHS.root
            )
        ),
        "runtime": runtime,
        "pdf_inspector": pdf_inspector_runtime,
        "next_steps": next_steps,
    }
