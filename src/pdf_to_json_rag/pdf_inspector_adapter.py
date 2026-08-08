"""Fail-open adapter around the optional-at-runtime ``pdf-inspector`` engine.

The package is a required distribution dependency, but the extraction pipeline deliberately
keeps working when a local installation is incomplete or the native extension rejects a PDF.
All page numbers exposed by this module are zero-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata as importlib_metadata
import os
from pathlib import Path
from typing import Literal


PDF_INSPECTOR_MODE_ENV = "PDF_TO_JSON_RAG_PDF_INSPECTOR_MODE"
PDF_INSPECTOR_DEFAULT_MODE = "assist"
PDF_INSPECTOR_DISTRIBUTION = "pdf-inspector"
PdfInspectorMode = Literal["assist", "shadow", "off"]


@dataclass
class PdfInspectorResult:
    """Normalized, stable result independent of the upstream PyO3 classes."""

    requested_mode: str
    effective_mode: PdfInspectorMode
    status: str
    version: str | None = None
    pdf_type: str | None = None
    confidence: float | None = None
    processing_time_ms: int | None = None
    page_count: int | None = None
    pages_needing_ocr: list[int] = field(default_factory=list)
    pages_with_tables: list[int] = field(default_factory=list)
    pages_with_columns: list[int] = field(default_factory=list)
    ocr_reasons_by_page: dict[int, list[str]] = field(default_factory=dict)
    has_encoding_issues: bool = False
    disagreements: list[dict[str, object]] = field(default_factory=list)
    tables_added: int = 0
    table_markdown_status: str = "not_requested"
    fallback_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.status == "ok" and self.effective_mode in {"assist", "shadow"}

    def to_summary(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "version": self.version,
            "status": self.status,
            "pdf_type": self.pdf_type,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "page_count": self.page_count,
            "pages_needing_ocr": [page + 1 for page in self.pages_needing_ocr],
            "pages_with_tables": [page + 1 for page in self.pages_with_tables],
            "pages_with_columns": [page + 1 for page in self.pages_with_columns],
            "ocr_reasons_by_page": [
                {"page_num": page + 1, "reasons": list(reasons)}
                for page, reasons in sorted(self.ocr_reasons_by_page.items())
            ],
            "has_encoding_issues": self.has_encoding_issues,
            "disagreements": list(self.disagreements),
            "tables_added": self.tables_added,
            "table_markdown_status": self.table_markdown_status,
            "fallback_reason": self.fallback_reason,
        }


def resolve_pdf_inspector_mode(value: str | None = None) -> tuple[str, PdfInspectorMode, str | None]:
    """Resolve the configured mode and fail closed to ``off`` for invalid values."""
    requested = value if value is not None else os.getenv(
        PDF_INSPECTOR_MODE_ENV,
        PDF_INSPECTOR_DEFAULT_MODE,
    )
    normalized = str(requested).strip().lower()
    if normalized in {"assist", "shadow", "off"}:
        return normalized, normalized, None  # type: ignore[return-value]
    return str(requested), "off", f"invalid_mode:{requested}"


def _installed_version() -> str | None:
    try:
        return importlib_metadata.version(PDF_INSPECTOR_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _load_module() -> object:
    return import_module("pdf_inspector")


def pdf_inspector_runtime_status(value: str | None = None) -> dict[str, object]:
    """Return installation and mode details without processing a document."""
    requested, effective, mode_error = resolve_pdf_inspector_mode(value)
    version = _installed_version()
    try:
        module = _load_module()
        available = callable(getattr(module, "process_pdf", None))
    except Exception:
        available = False
    fallback_reason = mode_error
    if not available:
        fallback_reason = fallback_reason or "package_unavailable"
    return {
        "available": available,
        "version": version,
        "requested_mode": requested,
        "effective_mode": effective if available and not mode_error else "off",
        "fallback_reason": fallback_reason,
    }


def _safe_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if result != result:  # NaN
        return None
    return max(0.0, min(1.0, result))


def _zero_based_pages(value: object, *, page_count: int) -> list[int]:
    pages: set[int] = set()
    if not isinstance(value, (list, tuple, set)):
        return []
    for raw_page in value:
        page = _safe_int(raw_page)
        if page is None:
            continue
        normalized = page - 1
        if 0 <= normalized < page_count:
            pages.add(normalized)
    return sorted(pages)


def _normalize_ocr_reasons(value: object, *, page_count: int) -> dict[int, list[str]]:
    normalized: dict[int, list[str]] = {}
    if not isinstance(value, (list, tuple)):
        return normalized
    for item in value:
        page = _safe_int(getattr(item, "page", None))
        if page is None or not 1 <= page <= page_count:
            continue
        raw_reasons = getattr(item, "reasons", [])
        reasons = sorted(
            {
                str(reason).strip()
                for reason in raw_reasons
                if str(reason).strip()
            }
        ) if isinstance(raw_reasons, (list, tuple, set)) else []
        normalized[page - 1] = reasons
    return normalized


def inspect_pdf_with_pdf_inspector(
    pdf_path: Path,
    *,
    expected_page_count: int,
    mode: str | None = None,
) -> PdfInspectorResult:
    """Run ``process_pdf`` and normalize its output, falling back without raising."""
    requested, effective, mode_error = resolve_pdf_inspector_mode(mode)
    result = PdfInspectorResult(
        requested_mode=requested,
        effective_mode=effective,
        status=("fallback" if mode_error else "skipped") if effective == "off" else "pending",
        version=_installed_version(),
        fallback_reason=mode_error,
    )
    if effective == "off":
        return result

    try:
        module = _load_module()
    except Exception as exc:
        result.effective_mode = "off"
        result.status = "fallback"
        result.fallback_reason = f"import_failed:{type(exc).__name__}"
        return result

    process_pdf = getattr(module, "process_pdf", None)
    if not callable(process_pdf):
        result.effective_mode = "off"
        result.status = "fallback"
        result.fallback_reason = "process_pdf_unavailable"
        return result

    try:
        raw = process_pdf(str(pdf_path))
    except Exception as exc:
        result.effective_mode = "off"
        result.status = "fallback"
        result.fallback_reason = f"process_failed:{type(exc).__name__}"
        return result

    page_count = _safe_int(getattr(raw, "page_count", None))
    if page_count != expected_page_count:
        result.effective_mode = "off"
        result.status = "fallback"
        result.page_count = page_count
        result.fallback_reason = f"page_count_mismatch:{page_count}:{expected_page_count}"
        return result

    result.status = "ok"
    result.page_count = page_count
    result.pdf_type = str(getattr(raw, "pdf_type", "") or "") or None
    result.confidence = _safe_float(getattr(raw, "confidence", None))
    result.processing_time_ms = _safe_int(getattr(raw, "processing_time_ms", None))
    result.pages_needing_ocr = _zero_based_pages(
        getattr(raw, "pages_needing_ocr", []),
        page_count=expected_page_count,
    )
    result.pages_with_tables = _zero_based_pages(
        getattr(raw, "pages_with_tables", []),
        page_count=expected_page_count,
    )
    result.pages_with_columns = _zero_based_pages(
        getattr(raw, "pages_with_columns", []),
        page_count=expected_page_count,
    )
    result.ocr_reasons_by_page = _normalize_ocr_reasons(
        getattr(raw, "ocr_reasons_by_page", []),
        page_count=expected_page_count,
    )
    result.has_encoding_issues = bool(getattr(raw, "has_encoding_issues", False))
    return result


def extract_candidate_page_markdown(
    pdf_path: Path,
    pages: list[int],
) -> tuple[dict[int, str], str | None]:
    """Extract Markdown only for selected zero-based pages, preserving fail-open behavior."""
    selected_pages = sorted(set(page for page in pages if page >= 0))
    if not selected_pages:
        return {}, None
    try:
        module = _load_module()
        extract_pages = getattr(module, "extract_pages_markdown", None)
        if not callable(extract_pages):
            return {}, "extract_pages_markdown_unavailable"
        raw = extract_pages(str(pdf_path), pages=selected_pages)
        page_results = getattr(raw, "pages", [])
    except Exception as exc:
        return {}, f"markdown_failed:{type(exc).__name__}"

    markdown_by_page: dict[int, str] = {}
    if not isinstance(page_results, (list, tuple)):
        return {}, "invalid_markdown_result"
    for page_result in page_results:
        page = _safe_int(getattr(page_result, "page", None))
        markdown = getattr(page_result, "markdown", None)
        if page in selected_pages and isinstance(markdown, str) and markdown.strip():
            markdown_by_page[page] = markdown
    return markdown_by_page, None
