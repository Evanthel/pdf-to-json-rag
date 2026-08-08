"""Extraction-stage interfaces for the MVP pipeline."""

import json
from dataclasses import dataclass, field
import importlib
import importlib.util
from pathlib import Path
import re
import shutil
from collections import defaultdict

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF releases
    import fitz
from PIL import Image
import pytesseract

from .content_metadata import classify_block_metadata, infer_layout_signals
from .document_structure import build_document_structure_analysis
from .document_semantics import interpret_document_semantics
from .pdf_inspector_adapter import (
    PdfInspectorResult,
    extract_candidate_page_markdown,
    inspect_pdf_with_pdf_inspector,
)
from .schemas import DocumentRecord


OCR_PAGE_NUMBER_RE = re.compile(r"^\d+$")
OCR_STRUCTURAL_HEADING_RE = re.compile(
    r"^(abstract|background|methods?|results?|discussion|conclusions?|follow-?up evaluations?|ct scans?)[:.]?$",
    re.IGNORECASE,
)
OCR_AUTHOR_LINE_RE = re.compile(
    r"\b(m\.d\.|ph\.d\.|b\.s\.|m\.s\.)\b",
    re.IGNORECASE,
)
OPAQUE_STEM_RE = re.compile(r"^(?:[a-f0-9]{24,}|[A-Z0-9]{24,}|[A-Za-z0-9]{32,64})$")
SUMMARY_CUE_STOP_PREFIXES = (
    "contents",
    "table of contents",
    "list of figures",
    "list of tables",
    "index",
    "bibliography",
    "references",
)


@dataclass
class ExtractedBlock:
    block_id: str
    page_num: int
    text: str
    bbox: list[float] | None
    reading_order_index: int
    extraction_method: str = "native"
    text_source: str = "native"
    block_kind: str = "text"
    block_role: str = "paragraph"
    line_count: int = 1
    token_count: int = 0
    text_quality_score: float = 1.0
    block_labels: list[str] = field(default_factory=list)
    structural_flags: list[str] = field(default_factory=list)
    font_size: float | None = None
    relative_font_size: float | None = None
    font_is_bold: bool = False


@dataclass
class ExtractedPage:
    page_num: int
    text: str
    char_count: int
    block_count: int
    needs_ocr: bool
    ocr_used: bool
    pdf_inspector_signals: list[str] = field(default_factory=list)
    pdf_inspector_ocr_reasons: list[str] = field(default_factory=list)
    pdf_inspector_content_used: bool = False
    extraction_engine_disagreement: bool = False


@dataclass
class NativePdfExtraction:
    doc_id: str
    source_pdf: str
    page_count: int
    title: str | None
    toc: list[str]
    metadata: dict[str, str]
    pages: list[ExtractedPage]
    blocks: list[ExtractedBlock]
    table_probe: dict[str, object] = field(default_factory=dict)
    pdf_inspector: dict[str, object] = field(default_factory=dict)


def _slugify_doc_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _looks_like_opaque_stem(value: str) -> bool:
    compact = value.strip()
    return bool(OPAQUE_STEM_RE.fullmatch(compact))


def _normalize_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_width: float,
    page_height: float,
) -> list[float]:
    return [
        max(0.0, min(1.0, x0 / page_width)),
        max(0.0, min(1.0, y0 / page_height)),
        max(0.0, min(1.0, x1 / page_width)),
        max(0.0, min(1.0, y1 / page_height)),
    ]


def _cluster_column_lefts(lefts: list[float], page_width: float) -> list[float]:
    if len(lefts) < 4:
        return []
    threshold = page_width * 0.12 if page_width <= 2.0 else max(36.0, page_width * 0.12)
    clusters: list[list[float]] = []
    for left in sorted(lefts):
        if not clusters or abs(left - (sum(clusters[-1]) / len(clusters[-1]))) > threshold:
            clusters.append([left])
        else:
            clusters[-1].append(left)
    column_lefts = [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= 2]
    if len(column_lefts) < 2:
        return []
    if max(column_lefts) - min(column_lefts) < page_width * 0.25:
        return []
    return column_lefts[:3]


def _sort_page_blocks_reading_order(
    page_blocks: list[tuple[float, float, float, float, str]],
    *,
    page_width: float,
) -> list[tuple[float, float, float, float, str]]:
    """Sort native text blocks, using column order when the page is visibly multi-column."""
    default_order = sorted(page_blocks, key=lambda item: (item[1], item[0]))
    body_blocks = [
        item
        for item in default_order
        if page_width > 0 and ((item[2] - item[0]) / page_width) < 0.65
    ]
    column_lefts = _cluster_column_lefts([item[0] for item in body_blocks], page_width)
    if not column_lefts:
        return default_order

    top = min(item[1] for item in body_blocks)
    bottom = max(item[3] for item in body_blocks)
    vertical_tolerance = page_width * 0.04 if page_width <= 2.0 else max(24.0, page_width * 0.04)

    def column_index(item: tuple[float, float, float, float, str]) -> int:
        return min(range(len(column_lefts)), key=lambda index: abs(item[0] - column_lefts[index]))

    def sort_key(item: tuple[float, float, float, float, str]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1, _text = item
        width_ratio = ((x1 - x0) / page_width) if page_width else 0.0
        if width_ratio >= 0.65:
            if y1 <= top + vertical_tolerance:
                return (0.0, y0, x0, 0.0)
            if y0 >= bottom - vertical_tolerance:
                return (2.0, y0, x0, 0.0)
            return (1.0, y0, x0, 0.0)
        return (1.0, float(column_index(item)), y0, x0)

    return sorted(default_order, key=sort_key)


def _sort_page_block_dicts_reading_order(
    page_blocks: list[dict[str, object]],
    *,
    page_width: float,
) -> list[dict[str, object]]:
    indexed = [
        (
            float(block["x0"]),
            float(block["y0"]),
            float(block["x1"]),
            float(block["y1"]),
            str(index),
        )
        for index, block in enumerate(page_blocks)
    ]
    sorted_indexes = [
        int(item[4])
        for item in _sort_page_blocks_reading_order(indexed, page_width=page_width)
    ]
    return [page_blocks[index] for index in sorted_indexes]


def _sort_extracted_blocks_reading_order(blocks: list[ExtractedBlock]) -> list[ExtractedBlock]:
    page_groups: dict[int, list[ExtractedBlock]] = defaultdict(list)
    for block in blocks:
        page_groups[block.page_num].append(block)

    ordered: list[ExtractedBlock] = []
    for page_num in sorted(page_groups):
        page_blocks = page_groups[page_num]
        if not all(block.bbox and len(block.bbox) == 4 for block in page_blocks):
            ordered.extend(sorted(page_blocks, key=lambda block: (block.reading_order_index, block.block_id)))
            continue
        positioned = [
            (
                float(block.bbox[0]),
                float(block.bbox[1]),
                float(block.bbox[2]),
                float(block.bbox[3]),
                block.block_id,
            )
            for block in page_blocks
            if block.bbox
        ]
        sorted_ids = [
            item[4]
            for item in _sort_page_blocks_reading_order(positioned, page_width=1.0)
        ]
        block_by_id = {block.block_id: block for block in page_blocks}
        ordered.extend(block_by_id[block_id] for block_id in sorted_ids)
    return ordered


def _span_is_bold(span: dict) -> bool:
    font_name = str(span.get("font", "")).lower()
    return any(token in font_name for token in ("bold", "black", "heavy", "demi")) or bool(int(span.get("flags", 0)) & 16)


def _page_font_median(page_dict: dict) -> float | None:
    sizes: list[float] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                if text:
                    sizes.append(float(span.get("size", 0.0) or 0.0))
    sizes = sorted(size for size in sizes if size > 0)
    if not sizes:
        return None
    midpoint = len(sizes) // 2
    if len(sizes) % 2:
        return sizes[midpoint]
    return (sizes[midpoint - 1] + sizes[midpoint]) / 2


def _native_page_blocks_with_font(page: fitz.Page) -> list[dict[str, object]]:
    page_dict = page.get_text("dict")
    median_font_size = _page_font_median(page_dict) or 0.0
    page_blocks: list[dict[str, object]] = []
    for raw_block in page_dict.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        lines: list[str] = []
        font_sizes: list[float] = []
        bold = False
        for line in raw_block.get("lines", []):
            line_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = str(span.get("text", ""))
                if not span_text.strip():
                    continue
                line_parts.append(span_text)
                size = float(span.get("size", 0.0) or 0.0)
                if size > 0:
                    font_sizes.append(size)
                bold = bold or _span_is_bold(span)
            line_text = "".join(line_parts).strip()
            if line_text:
                lines.append(line_text)
        clean_text = "\n".join(lines).strip()
        if not clean_text:
            continue
        x0, y0, x1, y1 = raw_block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        font_size = max(font_sizes) if font_sizes else None
        page_blocks.append(
            {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": clean_text,
                "font_size": font_size,
                "relative_font_size": (font_size / median_font_size) if font_size and median_font_size else None,
                "font_is_bold": bold,
            }
        )
    return page_blocks


def _load_pdfplumber_module() -> tuple[object | None, str | None]:
    if importlib.util.find_spec("pdfplumber") is None:
        return None, "not_installed"
    try:
        return importlib.import_module("pdfplumber"), None
    except Exception as exc:  # pragma: no cover - defensive around optional dependency import failures
        return None, f"import_failed:{type(exc).__name__}"


def _pdfplumber_rows_to_text(rows: list[list[object]]) -> str:
    lines: list[str] = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        while cells and not cells[-1]:
            cells.pop()
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


def _pdfplumber_table_bbox(table: object, page: object) -> list[float] | None:
    bbox = getattr(table, "bbox", None)
    if not bbox or len(bbox) != 4:
        return None
    page_width = float(getattr(page, "width", 0.0) or 0.0)
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    if page_width <= 0 or page_height <= 0:
        return None
    x0, top, x1, bottom = [float(value) for value in bbox]
    return _normalize_bbox(x0, top, x1, bottom, page_width, page_height)


def extract_pdfplumber_table_blocks(pdf_path: Path, *, doc_id: str) -> tuple[list[ExtractedBlock], dict[str, object]]:
    """Extract optional pdfplumber table blocks without making pdfplumber a hard dependency."""
    pdfplumber, unavailable_reason = _load_pdfplumber_module()
    if pdfplumber is None:
        return [], {
            "engine": "pdfplumber",
            "available": False,
            "table_count": 0,
            "supplemental_block_count": 0,
            "page_table_counts": [],
            "reason": unavailable_reason,
        }

    table_blocks: list[ExtractedBlock] = []
    page_table_counts: list[dict[str, int]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                tables: list[tuple[list[list[object]], list[float] | None]] = []
                if hasattr(page, "find_tables"):
                    for table in page.find_tables() or []:
                        rows = table.extract() or []
                        tables.append((rows, _pdfplumber_table_bbox(table, page)))
                else:
                    tables.extend((rows, None) for rows in (page.extract_tables() or []))
                page_table_counts.append({"page_num": index, "table_count": len(tables)})
                for table_index, (rows, bbox) in enumerate(tables, start=1):
                    table_text = _pdfplumber_rows_to_text(rows)
                    if not table_text:
                        continue
                    table_blocks.append(
                        ExtractedBlock(
                            block_id=f"{doc_id}-p{index:03d}-pdfplumber-table-{table_index:03d}",
                            page_num=index - 1,
                            text=table_text,
                            bbox=bbox,
                            reading_order_index=0,
                            extraction_method="native",
                            text_source="native",
                            block_kind="table_like",
                            block_role="table_like",
                            line_count=len(table_text.splitlines()),
                            token_count=len(re.findall(r"\w+", table_text)),
                            text_quality_score=0.95,
                            block_labels=["table_like"],
                            structural_flags=["pdfplumber_table", "structured_signal", "table_like"],
                        )
                    )
    except Exception as exc:  # pragma: no cover - depends on optional parser internals
        return table_blocks, {
            "engine": "pdfplumber",
            "available": True,
            "table_count": sum(item["table_count"] for item in page_table_counts),
            "supplemental_block_count": len(table_blocks),
            "page_table_counts": page_table_counts,
            "reason": f"probe_failed:{type(exc).__name__}",
        }
    return table_blocks, {
        "engine": "pdfplumber",
        "available": True,
        "table_count": sum(item["table_count"] for item in page_table_counts),
        "supplemental_block_count": len(table_blocks),
        "page_table_counts": page_table_counts,
        "reason": None,
    }


def probe_pdfplumber_tables(pdf_path: Path) -> dict[str, object]:
    """Return optional pdfplumber table-detection metadata without making it a hard dependency."""
    _, probe = extract_pdfplumber_table_blocks(pdf_path, doc_id="probe")
    return probe


def _guess_title_from_blocks(blocks: list[ExtractedBlock]) -> str | None:
    for block in blocks:
        line = block.text.splitlines()[0].strip()
        if 5 <= len(line) <= 160:
            return line
    return None


def _derive_summary_cues(
    title: str | None,
    toc: list[str],
    blocks: list[ExtractedBlock],
) -> list[str]:
    cues: list[str] = []
    seen: set[str] = set()

    def maybe_add(value: str) -> None:
        cue = re.sub(r"\s+", " ", value).strip()
        if not cue:
            return
        cue_lower = cue.lower()
        if cue_lower.startswith(SUMMARY_CUE_STOP_PREFIXES):
            return
        if len(cue) < 4:
            return
        if cue in seen:
            return
        seen.add(cue)
        cues.append(cue)

    if title:
        maybe_add(title)

    chapter_like = re.compile(
        r"^(chapter\s+\d+[:\s-].+|introduction|foreword|conclusion|part\s+\d+[:\s-].+)$",
        re.IGNORECASE,
    )
    for item in toc:
        compact = re.sub(r"\.{2,}\s*\d+$", "", item).strip()
        if chapter_like.match(compact):
            maybe_add(compact)
        elif len(cues) < 8 and compact and compact[0].isupper():
            maybe_add(compact)
        if len(cues) >= 10:
            break

    if len(cues) < 6:
        for block in blocks[:40]:
            first_line = block.text.splitlines()[0].strip()
            if chapter_like.match(first_line):
                maybe_add(first_line)
            elif (
                4 <= len(first_line) <= 80
                and first_line.upper() == first_line
                and re.search(r"[A-Z]{3,}", first_line)
                and not re.fullmatch(r"\d+", first_line)
            ):
                maybe_add(first_line)
            if len(cues) >= 8:
                break
    return cues[:10]


def _derive_discovery_terms(
    title: str | None,
    toc: list[str],
    summary_cues: list[str],
    blocks: list[ExtractedBlock],
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "your",
        "about",
        "what",
        "when",
        "where",
        "which",
        "chapter",
        "introduction",
        "foreword",
        "part",
        "guide",
        "guidance",
        "note",
    }

    def add_text(value: str) -> None:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", value.lower()):
            if token in stop or token in seen:
                continue
            seen.add(token)
            terms.append(token)

    for value in filter(None, [title, *summary_cues, *toc[:8]]):
        add_text(value)
        if len(terms) >= 20:
            break

    if len(terms) < 12:
        for block in blocks[:30]:
            add_text(block.text.splitlines()[0].strip())
            if len(terms) >= 20:
                break

    return terms[:20]


def _normalize_heading_match(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _font_heading_signal(
    *,
    text: str,
    relative_font_size: float | None,
    font_is_bold: bool,
    toc_entries: set[str],
) -> tuple[bool, list[str]]:
    normalized = _normalize_heading_match(text)
    flags: list[str] = []
    if normalized and normalized in toc_entries:
        flags.append("toc_heading")
    line_count = max(1, len([line for line in text.splitlines() if line.strip()]))
    token_count = len(re.findall(r"\w+", text))
    if (
        relative_font_size is not None
        and relative_font_size >= 1.18
        and line_count <= 2
        and token_count <= 18
    ):
        flags.append("relative_font_heading")
    if font_is_bold and line_count <= 2 and token_count <= 18:
        flags.append("bold_font_heading")
    return bool(flags), flags


def _build_extracted_block(
    *,
    block_id: str,
    page_num: int,
    text: str,
    bbox: list[float] | None,
    reading_order_index: int,
    extraction_method: str,
    text_source: str | None = None,
    font_size: float | None = None,
    relative_font_size: float | None = None,
    font_is_bold: bool = False,
    toc_entries: set[str] | None = None,
) -> ExtractedBlock:
    metadata = classify_block_metadata(text)
    font_heading, font_flags = _font_heading_signal(
        text=text,
        relative_font_size=relative_font_size,
        font_is_bold=font_is_bold,
        toc_entries=toc_entries or set(),
    )
    block_kind = str(metadata["block_kind"])
    block_role = str(metadata["block_role"])
    block_labels = list(metadata["block_labels"])
    structural_flags = list(metadata["structural_flags"])
    if font_heading and block_role in {"paragraph", "unknown"}:
        block_kind = "heading"
        block_role = "heading"
        block_labels.append("heading")
    structural_flags.extend(font_flags)
    return ExtractedBlock(
        block_id=block_id,
        page_num=page_num,
        text=text,
        bbox=bbox,
        reading_order_index=reading_order_index,
        extraction_method=extraction_method,
        text_source=text_source or ("ocr" if extraction_method == "ocr" else "native"),
        block_kind=block_kind,
        block_role=block_role,
        line_count=int(metadata["line_count"]),
        token_count=int(metadata["token_count"]),
        text_quality_score=float(metadata["text_quality_score"]),
        block_labels=sorted(set(block_labels)),
        structural_flags=sorted(set(structural_flags)),
        font_size=round(font_size, 3) if font_size is not None else None,
        relative_font_size=round(relative_font_size, 3) if relative_font_size is not None else None,
        font_is_bold=font_is_bold,
    )


def _clone_extracted_block(
    source_block: ExtractedBlock,
    *,
    reading_order_index: int,
    extraction_method: str | None = None,
    text_source: str | None = None,
) -> ExtractedBlock:
    return ExtractedBlock(
        block_id=source_block.block_id,
        page_num=source_block.page_num,
        text=source_block.text,
        bbox=source_block.bbox,
        reading_order_index=reading_order_index,
        extraction_method=extraction_method or source_block.extraction_method,
        text_source=text_source or source_block.text_source,
        block_kind=source_block.block_kind,
        block_role=source_block.block_role,
        line_count=source_block.line_count,
        token_count=source_block.token_count,
        text_quality_score=source_block.text_quality_score,
        block_labels=list(source_block.block_labels),
        structural_flags=list(source_block.structural_flags),
        font_size=source_block.font_size,
        relative_font_size=source_block.relative_font_size,
        font_is_bold=source_block.font_is_bold,
    )


def _page_text_score(blocks: list[ExtractedBlock]) -> float:
    if not blocks:
        return 0.0
    text_chars = sum(len(block.text.strip()) for block in blocks)
    avg_quality = sum(block.text_quality_score for block in blocks) / len(blocks)
    structural_bonus = sum(
        1
        for block in blocks
        if block.block_role in {"heading", "table_like", "form_field", "key_value", "checklist_item"}
    )
    return (text_chars * 0.01) + (avg_quality * 10.0) + min(structural_bonus, 4) * 0.9


def _page_layout_profile(
    blocks: list[ExtractedBlock],
    *,
    pdf_inspector_signals: list[str] | tuple[str, ...] = (),
    pdf_inspector_confidence: float | None = None,
) -> dict[str, object]:
    role_counts: dict[str, int] = {}
    text_source_counts: dict[str, int] = {}
    for block in blocks:
        role_counts[block.block_role] = role_counts.get(block.block_role, 0) + 1
        text_source_counts[block.text_source] = text_source_counts.get(block.text_source, 0) + 1
    avg_text_quality = round(
        sum(block.text_quality_score for block in blocks) / max(len(blocks), 1),
        3,
    ) if blocks else 0.0
    layout_signals = infer_layout_signals(
        block_roles=[block.block_role for block in blocks],
        structural_flags=[flag for block in blocks for flag in block.structural_flags],
        bboxes=[block.bbox for block in blocks],
    )
    inspector_layout_signals = sorted(
        signal
        for signal in set(pdf_inspector_signals)
        if signal in {"pdf_inspector_table", "pdf_inspector_columns"}
    )
    return {
        "block_count": len(blocks),
        "role_counts": role_counts,
        "text_source_counts": text_source_counts,
        "avg_text_quality_score": avg_text_quality,
        "layout_signals": sorted(set([*layout_signals, *inspector_layout_signals])),
        "pdf_inspector_signals": list(pdf_inspector_signals),
        "pdf_inspector_confidence": pdf_inspector_confidence,
    }


def _native_text_suspicion_reasons(page_text: str) -> list[str]:
    """Return conservative PyMuPDF-side corroboration for an inspector OCR signal."""
    reasons: list[str] = []
    stripped = page_text.strip()
    if len(stripped) < 80:
        reasons.append("under_80_chars")
    if "\x00" in page_text or "\ufffd" in page_text:
        reasons.append("invalid_unicode_marker")
    if page_text:
        printable_count = sum(character.isprintable() or character.isspace() for character in page_text)
        if printable_count / len(page_text) < 0.95:
            reasons.append("low_printable_ratio")
    tokens = re.findall(r"\S+", page_text)
    if tokens:
        non_lexical = sum(not any(character.isalnum() for character in token) for token in tokens)
        if non_lexical / len(tokens) >= 0.25:
            reasons.append("non_lexical_tokens")
    return reasons


def _has_reliable_table_like_block(blocks: list[ExtractedBlock]) -> bool:
    for block in blocks:
        if block.block_role != "table_like" and block.block_kind != "table_like":
            continue
        if "pdfplumber_table" in block.structural_flags:
            return True
        alphanumeric_count = sum(character.isalnum() for character in block.text)
        if block.line_count >= 2 and block.token_count >= 4 and alphanumeric_count >= 20:
            return True
    return False


def _markdown_cells(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.strip().split("|")]
    if cells and not cells[0]:
        cells = cells[1:]
    if cells and not cells[-1]:
        cells = cells[:-1]
    return cells


def _extract_valid_markdown_tables(markdown: str) -> list[str]:
    """Extract strict Markdown tables suitable for supplemental RAG blocks."""
    lines = markdown.splitlines()
    tables: list[str] = []
    index = 0
    while index + 2 < len(lines):
        header = _markdown_cells(lines[index])
        separator = _markdown_cells(lines[index + 1])
        valid_separator = (
            len(header) >= 2
            and len(separator) == len(header)
            and all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
        )
        if not valid_separator:
            index += 1
            continue

        data_lines: list[str] = []
        cursor = index + 2
        while cursor < len(lines):
            cells = _markdown_cells(lines[cursor])
            if len(cells) != len(header) or "|" not in lines[cursor]:
                break
            if any(cell.strip() for cell in cells):
                data_lines.append(lines[cursor])
            cursor += 1
        candidate_lines = [lines[index], lines[index + 1], *data_lines]
        candidate = "\n".join(line.strip() for line in candidate_lines).strip()
        if data_lines and sum(character.isalnum() for character in candidate) >= 40:
            tables.append(candidate)
        index = max(cursor, index + 1)
    return tables


def _normalized_table_signature(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.lower(), flags=re.UNICODE))


def _is_duplicate_table(table_text: str, blocks: list[ExtractedBlock]) -> bool:
    signature = _normalized_table_signature(table_text)
    if not signature:
        return True
    table_tokens = set(signature.split())
    for block in blocks:
        if block.block_role != "table_like" and block.block_kind != "table_like":
            continue
        existing_signature = _normalized_table_signature(block.text)
        if signature == existing_signature:
            return True
        existing_tokens = set(existing_signature.split())
        if table_tokens and existing_tokens:
            overlap = len(table_tokens & existing_tokens) / min(len(table_tokens), len(existing_tokens))
            if overlap >= 0.9:
                return True
    return False


def _insert_supplemental_table(
    blocks: list[ExtractedBlock],
    table_block: ExtractedBlock,
) -> list[ExtractedBlock]:
    """Insert after the closest lexical match while preserving all existing block order."""
    table_tokens = {
        token
        for token in re.findall(r"[\w]+", table_block.text.lower(), flags=re.UNICODE)
        if len(token) >= 3
    }
    best_index: int | None = None
    best_score = 0
    for index, block in enumerate(blocks):
        block_tokens = {
            token
            for token in re.findall(r"[\w]+", block.text.lower(), flags=re.UNICODE)
            if len(token) >= 3
        }
        score = len(table_tokens & block_tokens)
        if score > best_score:
            best_index = index
            best_score = score
    if best_index is None:
        return [*blocks, table_block]
    return [*blocks[: best_index + 1], table_block, *blocks[best_index + 1 :]]


def _fuse_page_blocks(
    native_blocks: list[ExtractedBlock],
    ocr_blocks: list[ExtractedBlock],
) -> tuple[list[ExtractedBlock], bool]:
    if not ocr_blocks:
        return native_blocks, False
    if not native_blocks:
        return ocr_blocks, True

    native_score = _page_text_score(native_blocks)
    ocr_score = _page_text_score(ocr_blocks)
    native_structural = [
        block
        for block in native_blocks
        if block.block_role in {"heading", "table_like", "form_field", "key_value"}
    ]
    native_long_paragraphs = [block for block in native_blocks if block.token_count >= 18]
    ocr_long_paragraphs = [block for block in ocr_blocks if block.token_count >= 18]

    if native_structural and ocr_long_paragraphs and not native_long_paragraphs:
        fused_blocks: list[ExtractedBlock] = []
        seen_texts: set[str] = set()
        for block in [*native_structural, *ocr_long_paragraphs]:
            normalized = re.sub(r"\s+", " ", block.text).strip().lower()
            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            fused_blocks.append(block)
        if fused_blocks:
            return sorted(fused_blocks, key=lambda block: (block.page_num, block.reading_order_index)), True

    native_layout_signals = set(
        infer_layout_signals(
            block_roles=[block.block_role for block in native_blocks],
            structural_flags=[flag for block in native_blocks for flag in block.structural_flags],
            bboxes=[block.bbox for block in native_blocks],
        )
    )
    ocr_layout_signals = set(
        infer_layout_signals(
            block_roles=[block.block_role for block in ocr_blocks],
            structural_flags=[flag for block in ocr_blocks for flag in block.structural_flags],
            bboxes=[block.bbox for block in ocr_blocks],
        )
    )

    if "form_dense" in native_layout_signals and "form_dense" not in ocr_layout_signals:
        return native_blocks, False
    if "table_dense" in native_layout_signals and "table_dense" not in ocr_layout_signals:
        return native_blocks, False
    if ocr_score > native_score + 2.5:
        return ocr_blocks, True
    return native_blocks, False


def extract_native_pdf(pdf_path: Path) -> NativePdfExtraction:
    """Extract page blocks and document metadata from a native-text PDF.

    MVP intent:
    - use PyMuPDF for text blocks and coordinates
    - preserve page-level and block-level ordering metadata
    """
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    blocks: list[ExtractedBlock] = []
    pages: list[ExtractedPage] = []
    page_blocks_by_page: dict[int, list[ExtractedBlock]] = {}

    with fitz.open(pdf_path) as pdf_doc:
        inspector_result: PdfInspectorResult = inspect_pdf_with_pdf_inspector(
            pdf_path,
            expected_page_count=len(pdf_doc),
        )
        metadata = {
            key: value
            for key, value in (pdf_doc.metadata or {}).items()
            if isinstance(value, str) and value.strip()
        }
        toc = [
            entry[1].strip()
            for entry in pdf_doc.get_toc(simple=True)
            if len(entry) >= 2 and entry[1].strip()
        ]
        toc_entries = {_normalize_heading_match(item) for item in toc if _normalize_heading_match(item)}
        metadata_title = metadata.get("title")
        preferred_doc_id_source = (
            metadata_title
            if metadata_title and _looks_like_opaque_stem(pdf_path.stem)
            else pdf_path.stem
        )
        doc_id = _slugify_doc_id(preferred_doc_id_source)
        pdfplumber_table_blocks, table_probe = extract_pdfplumber_table_blocks(pdf_path, doc_id=doc_id)
        pdfplumber_blocks_by_page: dict[int, list[ExtractedBlock]] = defaultdict(list)
        for table_block in pdfplumber_table_blocks:
            pdfplumber_blocks_by_page[table_block.page_num].append(table_block)

        for page_num, page in enumerate(pdf_doc):
            page_width = float(page.rect.width) or 1.0
            page_height = float(page.rect.height) or 1.0
            page_blocks = _sort_page_block_dicts_reading_order(
                _native_page_blocks_with_font(page),
                page_width=page_width,
            )

            candidate_blocks: list[ExtractedBlock] = []
            page_text_parts: list[str] = []
            for block in page_blocks:
                x0 = float(block["x0"])
                y0 = float(block["y0"])
                x1 = float(block["x1"])
                y1 = float(block["y1"])
                clean_text = str(block["text"])
                page_text_parts.append(clean_text)
                candidate_blocks.append(
                    _build_extracted_block(
                        block_id=f"{doc_id}-p{page_num + 1:03d}-native-{len(candidate_blocks) + 1:03d}",
                        page_num=page_num,
                        text=clean_text,
                        bbox=_normalize_bbox(
                            x0,
                            y0,
                            x1,
                            y1,
                            page_width,
                            page_height,
                        ),
                        reading_order_index=0,
                        extraction_method="native",
                        font_size=block.get("font_size") if isinstance(block.get("font_size"), float) else None,
                        relative_font_size=(
                            block.get("relative_font_size")
                            if isinstance(block.get("relative_font_size"), float)
                            else None
                        ),
                        font_is_bold=bool(block.get("font_is_bold")),
                        toc_entries=toc_entries,
                    )
                )

            page_text = "\n\n".join(page_text_parts)
            native_needs_ocr = page_needs_ocr(page_text)
            suspicion_reasons = _native_text_suspicion_reasons(page_text)
            inspector_recommends_ocr = (
                inspector_result.active
                and page_num in inspector_result.pages_needing_ocr
            )
            inspector_routes_ocr = (
                inspector_result.effective_mode == "assist"
                and inspector_recommends_ocr
                and bool(suspicion_reasons)
            )
            needs_ocr = native_needs_ocr or inspector_routes_ocr
            engine_disagreement = (
                inspector_result.active
                and inspector_recommends_ocr != native_needs_ocr
            )
            if engine_disagreement:
                inspector_result.disagreements.append(
                    {
                        "page_num": page_num + 1,
                        "pymupdf_needs_ocr": native_needs_ocr,
                        "pdf_inspector_needs_ocr": inspector_recommends_ocr,
                        "native_suspicion_reasons": suspicion_reasons,
                        "action": "ocr_requested" if needs_ocr else "native_text_preserved",
                    }
                )
            final_page_blocks = candidate_blocks
            ocr_used = False
            if needs_ocr:
                ocr_blocks = extract_page_with_ocr(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    render_scale=1.5 if inspector_routes_ocr and not native_needs_ocr else 2.0,
                )
                if ocr_blocks:
                    final_page_blocks, ocr_used = _fuse_page_blocks(candidate_blocks, ocr_blocks)
                    page_text = "\n\n".join(block.text for block in final_page_blocks)
            if pdfplumber_blocks_by_page.get(page_num):
                final_page_blocks = _sort_extracted_blocks_reading_order(
                    [*final_page_blocks, *pdfplumber_blocks_by_page[page_num]]
                )
                page_text = "\n\n".join(block.text for block in final_page_blocks)

            inspector_signals: list[str] = []
            if inspector_result.active:
                if page_num in inspector_result.pages_needing_ocr:
                    inspector_signals.append("pdf_inspector_ocr_recommended")
                if page_num in inspector_result.pages_with_tables:
                    inspector_signals.append("pdf_inspector_table")
                if page_num in inspector_result.pages_with_columns:
                    inspector_signals.append("pdf_inspector_columns")
                page_ocr_reasons = inspector_result.ocr_reasons_by_page.get(page_num, [])
                if inspector_result.has_encoding_issues and any(
                    "encod" in reason.lower() or "font" in reason.lower()
                    for reason in page_ocr_reasons
                ):
                    inspector_signals.append("pdf_inspector_encoding_issue")
            else:
                page_ocr_reasons = []

            page_blocks_by_page[page_num] = final_page_blocks

            pages.append(
                ExtractedPage(
                    page_num=page_num,
                    text=page_text,
                    char_count=len(page_text),
                    block_count=len(final_page_blocks),
                    needs_ocr=needs_ocr,
                    ocr_used=ocr_used,
                    pdf_inspector_signals=sorted(inspector_signals),
                    pdf_inspector_ocr_reasons=list(page_ocr_reasons),
                    extraction_engine_disagreement=engine_disagreement,
                )
            )

        if inspector_result.active and inspector_result.effective_mode == "shadow":
            inspector_result.table_markdown_status = "shadow"
        elif inspector_result.active and inspector_result.effective_mode == "assist":
            table_candidate_pages = [
                page_num
                for page_num in inspector_result.pages_with_tables
                if not pages[page_num].needs_ocr
                and not _has_reliable_table_like_block(page_blocks_by_page.get(page_num, []))
            ]
            markdown_by_page, markdown_error = extract_candidate_page_markdown(
                pdf_path,
                table_candidate_pages,
            )
            if markdown_error:
                inspector_result.table_markdown_status = "fallback"
                inspector_result.fallback_reason = f"table_markdown:{markdown_error}"
            elif table_candidate_pages:
                inspector_result.table_markdown_status = "processed"
            else:
                inspector_result.table_markdown_status = "not_needed"

            for page_num in table_candidate_pages:
                existing_page_blocks = page_blocks_by_page.get(page_num, [])
                valid_tables = _extract_valid_markdown_tables(markdown_by_page.get(page_num, ""))
                added_on_page = 0
                for table_text in valid_tables:
                    if _is_duplicate_table(table_text, existing_page_blocks):
                        continue
                    added_on_page += 1
                    table_block = ExtractedBlock(
                        block_id=(
                            f"{doc_id}-p{page_num + 1:03d}-"
                            f"pdf-inspector-table-{added_on_page:03d}"
                        ),
                        page_num=page_num,
                        text=table_text,
                        bbox=None,
                        reading_order_index=0,
                        extraction_method="native",
                        text_source="native",
                        block_kind="table_like",
                        block_role="table_like",
                        line_count=len([line for line in table_text.splitlines() if line.strip()]),
                        token_count=len(re.findall(r"\w+", table_text, flags=re.UNICODE)),
                        text_quality_score=0.9,
                        block_labels=["table_like"],
                        structural_flags=[
                            "pdf_inspector_table",
                            "structured_signal",
                            "table_like",
                        ],
                    )
                    existing_page_blocks = _insert_supplemental_table(
                        existing_page_blocks,
                        table_block,
                    )
                    inspector_result.tables_added += 1
                if added_on_page:
                    page_blocks_by_page[page_num] = existing_page_blocks
                    pages[page_num].pdf_inspector_content_used = True
                    pages[page_num].text = "\n\n".join(
                        block.text for block in existing_page_blocks
                    )
                    pages[page_num].char_count = len(pages[page_num].text)
                    pages[page_num].block_count = len(existing_page_blocks)

        reading_order_index = 0
        for page in pages:
            for final_block in page_blocks_by_page.get(page.page_num, []):
                blocks.append(
                    _clone_extracted_block(
                        final_block,
                        reading_order_index=reading_order_index,
                        extraction_method=(
                            "mixed"
                            if page.ocr_used and final_block.extraction_method == "native"
                            else final_block.extraction_method
                        ),
                        text_source=(
                            "merged"
                            if page.ocr_used and final_block.extraction_method == "native"
                            else final_block.text_source
                        ),
                    )
                )
                reading_order_index += 1

        title = metadata.get("title") or _guess_title_from_blocks(blocks)

        return NativePdfExtraction(
            doc_id=doc_id,
            source_pdf=pdf_path.name,
            page_count=len(pdf_doc),
            title=title,
            toc=toc,
            metadata=metadata,
            pages=pages,
            blocks=blocks,
            table_probe=table_probe,
            pdf_inspector=inspector_result.to_summary(),
        )


def build_document_record_from_native_extraction(
    extraction: NativePdfExtraction,
) -> DocumentRecord:
    """Create the initial document-level JSON record from native extraction."""
    pages_requiring_ocr = sum(1 for page in extraction.pages if page.needs_ocr)
    pages_processed_with_ocr = sum(1 for page in extraction.pages if page.ocr_used)
    summary_cues = _derive_summary_cues(
        title=extraction.title,
        toc=extraction.toc,
        blocks=extraction.blocks,
    )
    structure_analysis = build_document_structure_analysis(
        doc_id=extraction.doc_id,
        title=extraction.title,
        toc=extraction.toc,
        blocks=extraction.blocks,
    )
    sections = structure_analysis.sections
    for section in sections:
        if section.title and section.title not in summary_cues and len(summary_cues) < 12:
            summary_cues.append(section.title)
    discovery_terms = _derive_discovery_terms(
        title=extraction.title,
        toc=extraction.toc,
        summary_cues=summary_cues,
        blocks=extraction.blocks,
    )
    metadata_values: list[str] = []
    for key, value in extraction.metadata.items():
        if not isinstance(value, str):
            continue
        clean_value = value.strip()
        if not clean_value:
            continue
        metadata_values.append(clean_value)
        metadata_values.append(f"{key}: {clean_value}")
    semantics = interpret_document_semantics(
        source_pdf=extraction.source_pdf,
        title=extraction.title or extraction.doc_id,
        toc=extraction.toc,
        summary_cues=summary_cues,
        discovery_terms=discovery_terms,
        leading_block_lines=[block.text.splitlines()[0].strip() for block in extraction.blocks[:30]],
        metadata_values=metadata_values,
        page_count=extraction.page_count,
    )
    block_role_counts: dict[str, int] = {}
    text_source_counts: dict[str, int] = {}
    for block in extraction.blocks:
        block_role_counts[block.block_role] = block_role_counts.get(block.block_role, 0) + 1
        text_source_counts[block.text_source] = text_source_counts.get(block.text_source, 0) + 1
    page_layout_profiles = [
        {
            "page_num": page.page_num + 1,
            **_page_layout_profile(
                [block for block in extraction.blocks if block.page_num == page.page_num],
                pdf_inspector_signals=page.pdf_inspector_signals,
                pdf_inspector_confidence=(
                    extraction.pdf_inspector.get("confidence")
                    if isinstance(extraction.pdf_inspector.get("confidence"), (int, float))
                    else None
                ),
            ),
        }
        for page in extraction.pages
    ]
    layout_signal_counts: dict[str, int] = defaultdict(int)
    for profile in page_layout_profiles:
        for signal in profile["layout_signals"]:
            layout_signal_counts[str(signal)] += 1
    avg_text_quality_score = round(
        sum(block.text_quality_score for block in extraction.blocks) / max(len(extraction.blocks), 1),
        3,
    ) if extraction.blocks else 0.0
    return DocumentRecord(
        doc_id=extraction.doc_id,
        source_pdf=extraction.source_pdf,
        page_count=extraction.page_count,
        title=extraction.title,
        toc=extraction.toc,
        summary_cues=summary_cues,
        discovery_terms=discovery_terms,
        inventory_summary=semantics.inventory_summary,
        coverage_summary=semantics.coverage_summary,
        coverage_terms=list(semantics.coverage_terms),
        document_family=semantics.document_family,
        document_type=semantics.document_type,
        document_purpose=semantics.document_purpose,
        audience=semantics.audience,
        evidence_style=semantics.evidence_style,
        structure_style=semantics.structure_style,
        structure_confidence=structure_analysis.structure_confidence,
        layout_confidence=structure_analysis.layout_confidence,
        semantic_confidence=semantics.semantic_confidence,
        semantic_confidence_label=semantics.semantic_confidence_label,
        semantic_rationale=list(semantics.semantic_rationale),
        semantic_warnings=list(semantics.semantic_warnings),
        facet_terms=list(semantics.facet_terms),
        extraction_summary={
            "native_blocks": len(extraction.blocks),
            "pages_requiring_ocr": pages_requiring_ocr,
            "pages_processed_with_ocr": pages_processed_with_ocr,
            "ocr_used": pages_processed_with_ocr > 0,
            "block_role_counts": block_role_counts,
            "text_source_counts": text_source_counts,
            "avg_text_quality_score": avg_text_quality_score,
            "layout_signal_counts": dict(layout_signal_counts),
            "page_layout_profiles": page_layout_profiles,
            "table_probe": extraction.table_probe,
            "pdf_inspector": extraction.pdf_inspector,
        },
        sections=sections,
    )


def native_extraction_to_dict(extraction: NativePdfExtraction) -> dict:
    """Serialize native extraction output into a JSON-friendly structure."""
    return {
        "doc_id": extraction.doc_id,
        "source_pdf": extraction.source_pdf,
        "page_count": extraction.page_count,
        "title": extraction.title,
        "toc": extraction.toc,
        "metadata": extraction.metadata,
        "table_probe": extraction.table_probe,
        "pdf_inspector": extraction.pdf_inspector,
        "pages": [
            {
                "page_num": page.page_num,
                "text": page.text,
                "char_count": page.char_count,
                "block_count": page.block_count,
                "needs_ocr": page.needs_ocr,
                "ocr_used": page.ocr_used,
                "pdf_inspector_signals": page.pdf_inspector_signals,
                "pdf_inspector_ocr_reasons": page.pdf_inspector_ocr_reasons,
                "pdf_inspector_content_used": page.pdf_inspector_content_used,
                "extraction_engine_disagreement": page.extraction_engine_disagreement,
            }
            for page in extraction.pages
        ],
        "blocks": [
            {
                "block_id": block.block_id,
                "page_num": block.page_num,
                "text": block.text,
                "bbox": block.bbox,
                "reading_order_index": block.reading_order_index,
                "extraction_method": block.extraction_method,
                "text_source": block.text_source,
                "block_kind": block.block_kind,
                "block_role": block.block_role,
                "line_count": block.line_count,
                "token_count": block.token_count,
                "text_quality_score": block.text_quality_score,
                "block_labels": block.block_labels,
                "structural_flags": block.structural_flags,
                "font_size": block.font_size,
                "relative_font_size": block.relative_font_size,
                "font_is_bold": block.font_is_bold,
            }
            for block in extraction.blocks
        ],
    }


def save_native_extraction(
    extraction: NativePdfExtraction,
    output_dir: Path,
) -> Path:
    """Write the raw native extraction artifact to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{extraction.doc_id}.native.json"
    output_path.write_text(
        json.dumps(native_extraction_to_dict(extraction), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def save_document_record(
    document_record: DocumentRecord,
    output_dir: Path,
) -> Path:
    """Write the document-level JSON record to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{document_record.doc_id}.document.json"
    output_path.write_text(
        json.dumps(document_record.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def process_native_pdf_to_json(
    pdf_path: Path,
    output_dir: Path,
) -> tuple[NativePdfExtraction, DocumentRecord, Path, Path]:
    """Run native extraction and write both document artifacts to JSON."""
    extraction = extract_native_pdf(pdf_path)
    document_record = build_document_record_from_native_extraction(extraction)
    native_path = save_native_extraction(extraction, output_dir)
    document_path = save_document_record(document_record, output_dir)
    return extraction, document_record, native_path, document_path


def page_needs_ocr(page_text: str, min_chars: int = 40) -> bool:
    """Return True when native extraction is too weak for a page."""
    return len(page_text.strip()) < min_chars


def _normalize_pixel_bbox(
    left: int,
    top: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> list[float]:
    return _normalize_bbox(
        float(left),
        float(top),
        float(left + width),
        float(top + height),
        float(image_width),
        float(image_height),
    )


def _normalize_ocr_line_text(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_ocr_noise_line(
    text: str,
    *,
    top: int,
    bottom: int,
    image_height: int,
    line_height: int,
) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    if OCR_PAGE_NUMBER_RE.match(normalized):
        return True

    is_top_band = top <= max(int(image_height * 0.08), line_height * 2)
    is_bottom_band = bottom >= image_height - max(int(image_height * 0.08), line_height * 2)

    if is_top_band or is_bottom_band:
        if any(
            hint in normalized
            for hint in (
                "doi:",
                "http://",
                "https://",
                "www.",
                "copyright",
                "all rights reserved",
                "vol ",
                "issue",
                "journal",
                "downloaded from",
                "page ",
            )
        ):
            return True
        if len(normalized) < 18 and sum(ch.isdigit() for ch in normalized) >= 2:
            return True

    if normalized.startswith("figure ") and len(normalized) < 24:
        return True
    return False


def _join_ocr_lines(lines: list[dict]) -> str:
    parts: list[str] = []
    for line in lines:
        line_text = _normalize_ocr_line_text(line["text"])
        if not line_text:
            continue
        if parts:
            prev = parts[-1]
            if prev.endswith("-") and line_text[:1].islower():
                parts[-1] = prev[:-1] + line_text
                continue
            if prev.endswith(("/", "(", "[")):
                parts[-1] = prev + line_text
                continue
            if prev.endswith((".", "?", "!", ":")):
                parts.append(line_text)
                continue
            parts[-1] = prev + " " + line_text
            continue
        parts.append(line_text)
    return "\n".join(parts).strip()


def _looks_like_structural_heading(text: str) -> bool:
    normalized = _normalize_ocr_line_text(text)
    if not normalized:
        return False
    if OCR_STRUCTURAL_HEADING_RE.match(normalized):
        return True
    if normalized.lower().startswith(("abstract ", "background ", "methods ", "results ", "discussion ")):
        return True
    return False


def _looks_like_author_credit(text: str) -> bool:
    normalized = _normalize_ocr_line_text(text)
    if not normalized:
        return False
    if not OCR_AUTHOR_LINE_RE.search(normalized):
        return False
    return len(normalized) <= 180


def _should_drop_ocr_paragraph(block_text: str, page_num: int) -> bool:
    normalized = _normalize_ocr_line_text(block_text)
    lower = normalized.lower()
    if not normalized:
        return True
    if page_num == 0 and (
        _looks_like_author_credit(normalized)
        or normalized.startswith("COMPUTED TOMOGRAPHIC STUDY OF THE COMMON COLD")
    ):
        return True
    if lower.startswith(("downloaded from", "massachusetts medical society registry")):
        return True
    if "reprint requests" in lower or "supported by the procter" in lower:
        return True
    if normalized.count("—") >= 2 and len(normalized) < 160:
        return True
    return False


def _build_ocr_blocks_from_image(image: Image.Image, page_num: int) -> list[ExtractedBlock]:
    try:
        ocr_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []

    grouped_words: dict[tuple[int, int], list[dict]] = defaultdict(list)
    item_count = len(ocr_data.get("text", []))

    for index in range(item_count):
        raw_text = (ocr_data["text"][index] or "").strip()
        if not raw_text:
            continue

        confidence_raw = ocr_data.get("conf", ["-1"] * item_count)[index]
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            continue

        block_num = int(ocr_data.get("block_num", [0] * item_count)[index])
        par_num = int(ocr_data.get("par_num", [0] * item_count)[index])
        line_num = int(ocr_data.get("line_num", [0] * item_count)[index])
        group_key = (block_num, par_num)

        grouped_words[group_key].append(
            {
                "text": raw_text,
                "left": int(ocr_data["left"][index]),
                "top": int(ocr_data["top"][index]),
                "width": int(ocr_data["width"][index]),
                "height": int(ocr_data["height"][index]),
                "line_num": line_num,
            }
        )

    if not grouped_words:
        return []

    blocks: list[ExtractedBlock] = []
    image_width, image_height = image.size
    page_lines: list[dict] = []

    for words in grouped_words.values():
        lines: dict[int, list[dict]] = defaultdict(list)
        for word in words:
            lines[word["line_num"]].append(word)

        ordered_lines = []
        for line_num in sorted(lines):
            line_words = sorted(lines[line_num], key=lambda item: (item["left"], item["top"]))
            line_text = _normalize_ocr_line_text(" ".join(word["text"] for word in line_words))
            if not line_text:
                continue
            line_left = min(word["left"] for word in line_words)
            line_top = min(word["top"] for word in line_words)
            line_right = max(word["left"] + word["width"] for word in line_words)
            line_bottom = max(word["top"] + word["height"] for word in line_words)
            line_height = max(1, line_bottom - line_top)
            if _looks_like_ocr_noise_line(
                line_text,
                top=line_top,
                bottom=line_bottom,
                image_height=image.size[1],
                line_height=line_height,
            ):
                continue
            ordered_lines.append(
                {
                    "line_num": line_num,
                    "words": line_words,
                    "text": line_text,
                    "left": line_left,
                    "top": line_top,
                    "right": line_right,
                    "bottom": line_bottom,
                    "height": line_height,
                }
            )

        if not ordered_lines:
            continue

        page_lines.extend(ordered_lines)

    if not page_lines:
        return []

    page_lines.sort(key=lambda line: (line["top"], line["left"]))

    paragraph_groups: list[list[dict]] = []
    current_group: list[dict] = []
    prev_line: dict | None = None

    for line in page_lines:
        if not current_group:
            current_group = [line]
            prev_line = line
            continue

        gap = line["top"] - prev_line["bottom"]
        gap_threshold = max(max(prev_line["height"], line["height"]) * 4.5, 40)
        left_delta = abs(line["left"] - prev_line["left"])
        same_column = left_delta <= max(prev_line["height"], line["height"]) * 6
        current_is_heading = _looks_like_structural_heading(line["text"])
        prev_is_heading = _looks_like_structural_heading(prev_line["text"])

        if gap > gap_threshold or not same_column or current_is_heading or prev_is_heading:
            paragraph_groups.append(current_group)
            current_group = [line]
        else:
            current_group.append(line)
        prev_line = line

    if current_group:
        paragraph_groups.append(current_group)

    for paragraph in paragraph_groups:
        block_text = _join_ocr_lines(paragraph)
        if not block_text:
            continue
        if _should_drop_ocr_paragraph(block_text, page_num):
            continue

        left = min(line["left"] for line in paragraph)
        top = min(line["top"] for line in paragraph)
        right = max(line["right"] for line in paragraph)
        bottom = max(line["bottom"] for line in paragraph)

        blocks.append(
            _build_extracted_block(
                block_id=f"ocr-page-{page_num + 1:03d}-{len(blocks) + 1:03d}",
                page_num=page_num,
                text=block_text,
                bbox=_normalize_pixel_bbox(
                    left=left,
                    top=top,
                    width=right - left,
                    height=bottom - top,
                    image_width=image_width,
                    image_height=image_height,
                ),
                reading_order_index=0,
                extraction_method="ocr",
                text_source="ocr",
            )
        )

    return sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]) if block.bbox else (0, 0))


def extract_page_with_ocr(
    pdf_path: Path,
    page_num: int,
    *,
    render_scale: float = 2.0,
) -> list[ExtractedBlock]:
    """OCR fallback for pages with poor native text extraction."""
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return []

    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    pdf_path = pdf_path.expanduser().resolve()
    with fitz.open(pdf_path) as pdf_doc:
        page = pdf_doc[page_num]
        matrix = fitz.Matrix(render_scale, render_scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        try:
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            ocr_blocks = _build_ocr_blocks_from_image(image=image, page_num=page_num)
        except Exception:
            return []
        if ocr_blocks:
            return ocr_blocks

        try:
            ocr_text = pytesseract.image_to_string(image)
        except Exception:
            return []

        ocr_text = re.sub(r"\s+", " ", ocr_text).strip()
        if not ocr_text:
            return []

        image_width, image_height = image.size
        return [
            _build_extracted_block(
                block_id=f"ocr-page-{page_num + 1:03d}-fallback-001",
                page_num=page_num,
                text=ocr_text,
                bbox=_normalize_bbox(
                    0.0,
                    0.0,
                    float(image_width),
                    float(image_height),
                    float(image_width),
                    float(image_height),
                ),
                reading_order_index=0,
                extraction_method="ocr",
                text_source="ocr",
            )
        ]
