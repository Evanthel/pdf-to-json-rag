"""Application service used by the lightweight local web interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import RLock
from typing import Any

from .answering import GroundedAnswer, answer_query_with_retrieval
from .chunking import process_saved_document_to_chunks
from .config import PATHS, ProjectPaths
from .extraction import process_native_pdf_to_json
from .indexing import build_local_index, load_chunk_records


DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,199}$")


class WebServiceError(Exception):
    """A safe, user-facing failure raised by the local web service."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


def _clean_filename(filename: str) -> str:
    name = Path(filename or "document.pdf").name.strip()
    stem = Path(name).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9._ -]+", "-", stem)
    safe_stem = re.sub(r"\s+", "-", safe_stem).strip("-._")[:120]
    return f"{safe_stem or 'document'}.pdf"


def _excerpt(text: str, limit: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _json_dict(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WebServiceError(
            "invalid_document_artifact",
            "The saved document could not be read.",
            status=500,
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise WebServiceError(
            "invalid_document_artifact",
            "The saved document has an invalid format.",
            status=500,
            details={"path": str(path)},
        )
    return payload


@dataclass
class RagWebService:
    """Coordinate the existing pipeline for one local web client."""

    paths: ProjectPaths = PATHS
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    _pipeline_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.paths.ensure_dirs()

    @property
    def web_index_root(self) -> Path:
        return self.paths.data_index / "web"

    def _validate_doc_id(self, doc_id: str) -> str:
        normalized = doc_id.strip().lower()
        if not DOC_ID_RE.fullmatch(normalized):
            raise WebServiceError(
                "invalid_document_id",
                "Invalid document ID.",
                status=400,
            )
        return normalized

    def _document_path(self, doc_id: str) -> Path:
        return self.paths.data_documents / f"{self._validate_doc_id(doc_id)}.document.json"

    def _native_path(self, doc_id: str) -> Path:
        return self.paths.data_documents / f"{self._validate_doc_id(doc_id)}.native.json"

    def _index_dir(self, doc_id: str) -> Path:
        return self.web_index_root / self._validate_doc_id(doc_id)

    def _input_path(self, filename: str, content: bytes) -> Path:
        safe_name = _clean_filename(filename)
        candidate = self.paths.data_input / safe_name
        if not candidate.exists():
            return candidate
        try:
            if hashlib.sha256(candidate.read_bytes()).digest() == hashlib.sha256(content).digest():
                return candidate
        except OSError:
            pass
        digest = hashlib.sha256(content).hexdigest()[:10]
        return candidate.with_name(f"{candidate.stem}-{digest}.pdf")

    def ingest_pdf(self, filename: str, content: bytes) -> dict[str, object]:
        """Save and process a PDF through the canonical local pipeline."""
        if not content:
            raise WebServiceError("empty_upload", "The selected file is empty.")
        if len(content) > self.max_upload_bytes:
            raise WebServiceError(
                "upload_too_large",
                f"The file exceeds the {self.max_upload_bytes // (1024 * 1024)} MB limit.",
                status=413,
            )
        if not content.lstrip().startswith(b"%PDF-"):
            raise WebServiceError(
                "not_a_pdf",
                "The selected file does not appear to be a valid PDF.",
                status=415,
            )

        with self._pipeline_lock:
            pdf_path = self._input_path(filename, content)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            if not pdf_path.exists():
                pdf_path.write_bytes(content)

            try:
                extraction, _document_record, native_path, document_path = (
                    process_native_pdf_to_json(
                        pdf_path=pdf_path,
                        output_dir=self.paths.data_documents,
                    )
                )
                document, chunks, _saved_paths = process_saved_document_to_chunks(
                    native_path=native_path,
                    document_path=document_path,
                    output_dir=self.paths.data_chunks,
                )
                if not chunks:
                    raise WebServiceError(
                        "no_chunks_created",
                        "No searchable chunks could be created from this document.",
                        status=422,
                    )
                build_local_index(chunks=chunks, index_dir=self._index_dir(document.doc_id))
            except WebServiceError:
                raise
            except Exception as exc:
                raise WebServiceError(
                    "processing_failed",
                    "This PDF could not be processed.",
                    status=422,
                ) from exc

        return self.get_document(extraction.doc_id)

    def _chunk_count(self, doc_id: str, document: dict[str, object]) -> int:
        embedded = document.get("chunks")
        if isinstance(embedded, list) and embedded:
            return len(embedded)
        chunk_dir = self.paths.data_chunks / doc_id
        return len(list(chunk_dir.glob("*.json"))) if chunk_dir.exists() else 0

    def _diagnostics(
        self,
        document: dict[str, object],
        *,
        chunk_count: int,
    ) -> dict[str, object]:
        extraction = document.get("extraction_summary")
        extraction = extraction if isinstance(extraction, dict) else {}
        inspector = extraction.get("pdf_inspector")
        inspector = inspector if isinstance(inspector, dict) else {}
        pages_requiring_ocr = int(extraction.get("pages_requiring_ocr") or 0)
        pages_processed_with_ocr = int(extraction.get("pages_processed_with_ocr") or 0)
        disagreements = inspector.get("disagreements")
        disagreements = disagreements if isinstance(disagreements, list) else []
        table_pages = inspector.get("pages_with_tables")
        table_pages = table_pages if isinstance(table_pages, list) else []
        added_tables = int(inspector.get("tables_added") or 0)

        flags: list[str] = []
        if pages_requiring_ocr or pages_processed_with_ocr:
            flags.append("ocr_used")
        if inspector.get("has_encoding_issues"):
            flags.append("encoding_uncertain")
        if disagreements:
            flags.append("engine_disagreement")
        if table_pages or added_tables:
            flags.append("tables_detected")
        if not chunk_count:
            flags.append("not_indexed")
        structure_confidence = document.get("structure_confidence")
        layout_confidence = document.get("layout_confidence")
        if isinstance(structure_confidence, (int, float)) and structure_confidence < 0.55:
            flags.append("structure_uncertain")
        if isinstance(layout_confidence, (int, float)) and layout_confidence < 0.55:
            flags.append("layout_uncertain")
        if pages_requiring_ocr > pages_processed_with_ocr:
            flags.append("ocr_incomplete")

        status = "ready"
        if not chunk_count:
            status = "incomplete"
        elif any(
            flag in flags
            for flag in (
                "encoding_uncertain",
                "engine_disagreement",
                "structure_uncertain",
                "layout_uncertain",
                "ocr_incomplete",
            )
        ):
            status = "review"

        return {
            "status": status,
            "flags": flags,
            "ocr": {
                "used": bool(extraction.get("ocr_used") or pages_processed_with_ocr),
                "pages_requiring": pages_requiring_ocr,
                "pages_processed": pages_processed_with_ocr,
            },
            "tables": {
                "pages": table_pages,
                "added": added_tables,
            },
            "inspector": {
                "status": inspector.get("status"),
                "requested_mode": inspector.get("requested_mode"),
                "effective_mode": inspector.get("effective_mode"),
                "version": inspector.get("version"),
                "confidence": inspector.get("confidence"),
                "encoding_issues": bool(inspector.get("has_encoding_issues")),
                "disagreements": disagreements,
                "fallback_reason": inspector.get("fallback_reason"),
            },
        }

    def _document_view(self, path: Path) -> dict[str, object]:
        document = _json_dict(path)
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not DOC_ID_RE.fullmatch(doc_id):
            raise WebServiceError(
                "invalid_document_artifact",
                "The saved document does not have a valid ID.",
                status=500,
                details={"path": str(path)},
            )
        chunk_count = self._chunk_count(doc_id, document)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        title = document.get("title")
        source_pdf = str(document.get("source_pdf") or f"{doc_id}.pdf")
        label = str(title).strip() if isinstance(title, str) and title.strip() else Path(source_pdf).stem
        return {
            "doc_id": doc_id,
            "label": label,
            "title": title,
            "source_pdf": source_pdf,
            "page_count": int(document.get("page_count") or 0),
            "section_count": len(document.get("sections") or []),
            "chunk_count": chunk_count,
            "document_type": document.get("document_type"),
            "document_purpose": document.get("document_purpose"),
            "audience": document.get("audience"),
            "detected_language": document.get("detected_language"),
            "summary": document.get("inventory_summary") or document.get("coverage_summary"),
            "structure_confidence": document.get("structure_confidence"),
            "layout_confidence": document.get("layout_confidence"),
            "semantic_confidence": document.get("semantic_confidence"),
            "updated_at": updated_at,
            "diagnostics": self._diagnostics(document, chunk_count=chunk_count),
        }

    def list_documents(self) -> list[dict[str, object]]:
        """Return readable document summaries, newest first."""
        if not self.paths.data_documents.exists():
            return []
        documents: list[dict[str, object]] = []
        for path in self.paths.data_documents.glob("*.document.json"):
            try:
                documents.append(self._document_view(path))
            except WebServiceError:
                continue
        documents.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return documents

    def get_document(self, doc_id: str) -> dict[str, object]:
        path = self._document_path(doc_id)
        if not path.exists():
            raise WebServiceError(
                "document_not_found",
                "Document not found.",
                status=404,
            )
        return self._document_view(path)

    def _ensure_index(self, doc_id: str) -> Path:
        index_dir = self._index_dir(doc_id)
        if (index_dir / "index_manifest.json").exists():
            return index_dir
        chunk_dir = self.paths.data_chunks / doc_id
        if not chunk_dir.exists():
            raise WebServiceError(
                "document_not_ready",
                "This document does not have searchable chunks yet.",
                status=409,
            )
        chunks = load_chunk_records(chunk_dir)
        if not chunks:
            raise WebServiceError(
                "document_not_ready",
                "This document does not have searchable chunks yet.",
                status=409,
            )
        build_local_index(chunks=chunks, index_dir=index_dir)
        return index_dir

    def ask(self, doc_id: str, query: str, *, k: int = 5) -> dict[str, object]:
        """Answer one question against a document-scoped local index."""
        normalized_doc_id = self._validate_doc_id(doc_id)
        self.get_document(normalized_doc_id)
        clean_query = re.sub(r"\s+", " ", query).strip()
        if not clean_query:
            raise WebServiceError("empty_query", "Enter a question for this document.")
        if len(clean_query) > 4000:
            raise WebServiceError("query_too_long", "The question is too long.")
        if not 1 <= k <= 12:
            raise WebServiceError("invalid_top_k", "The number of sources must be between 1 and 12.")

        with self._pipeline_lock:
            try:
                index_dir = self._ensure_index(normalized_doc_id)
                result = answer_query_with_retrieval(
                    query=clean_query,
                    index_dir=index_dir,
                    chunk_root=self.paths.data_chunks,
                    k=k,
                )
            except WebServiceError:
                raise
            except Exception as exc:
                raise WebServiceError(
                    "answer_failed",
                    "An answer could not be prepared.",
                    status=422,
                ) from exc
        return answer_view(result)


def answer_view(result: GroundedAnswer | Any) -> dict[str, object]:
    """Serialize a grounded answer into a compact UI-facing contract."""
    evidence = [
        {
            "chunk_id": item.chunk_id,
            "page_start": int(item.page_start),
            "page_end": int(item.page_end),
            "section_title": item.section_title,
            "sentence": item.sentence,
            "score": round(float(item.score), 4),
            "matched_terms": list(item.matched_terms),
        }
        for item in result.evidence
    ]
    sources = [
        {
            "chunk_id": chunk.chunk_id,
            "page_start": int(chunk.page_start),
            "page_end": int(chunk.page_end),
            "section_title": chunk.section_title,
            "chunk_type": chunk.chunk_type,
            "extraction_method": chunk.extraction_method,
            "quality_score": round(float(chunk.quality_score), 3),
            "excerpt": _excerpt(chunk.text),
        }
        for chunk in result.top_k_hits[:8]
    ]
    trace = result.answer_trace if isinstance(result.answer_trace, dict) else {}
    claim_alignment = trace.get("claim_alignment")
    claim_alignment = claim_alignment if isinstance(claim_alignment, dict) else {}
    unsupported = int(claim_alignment.get("unsupported_claim_count") or 0)
    weak = int(claim_alignment.get("weak_claim_count") or 0)
    trust = "supported"
    if unsupported:
        trust = "review"
    elif weak:
        trust = "partial"
    elif not evidence and not sources:
        trust = "limited"
    return {
        "query": result.query,
        "query_intent": result.query_intent,
        "answer": result.answer,
        "trust": trust,
        "evidence": evidence,
        "sources": sources,
        "meta": {
            "evidence_count": len(evidence),
            "source_count": len(sources),
            "weak_claim_count": weak,
            "unsupported_claim_count": unsupported,
        },
    }
