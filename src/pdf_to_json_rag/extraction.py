"""Extraction-stage interfaces for the MVP pipeline."""

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import tempfile
from collections import defaultdict

import fitz
from PIL import Image
import pytesseract

from .content_metadata import classify_block_metadata
from .document_structure import build_document_structure_analysis
from .document_semantics import interpret_document_semantics
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
    page_num: int
    text: str
    bbox: list[float] | None
    reading_order_index: int
    extraction_method: str = "native"
    block_kind: str = "text"
    line_count: int = 1
    token_count: int = 0
    structural_flags: list[str] = field(default_factory=list)


@dataclass
class ExtractedPage:
    page_num: int
    text: str
    char_count: int
    block_count: int
    needs_ocr: bool
    ocr_used: bool


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


def _build_extracted_block(
    *,
    page_num: int,
    text: str,
    bbox: list[float] | None,
    reading_order_index: int,
    extraction_method: str,
) -> ExtractedBlock:
    metadata = classify_block_metadata(text)
    return ExtractedBlock(
        page_num=page_num,
        text=text,
        bbox=bbox,
        reading_order_index=reading_order_index,
        extraction_method=extraction_method,
        block_kind=str(metadata["block_kind"]),
        line_count=int(metadata["line_count"]),
        token_count=int(metadata["token_count"]),
        structural_flags=list(metadata["structural_flags"]),
    )


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
    reading_order_index = 0

    with fitz.open(pdf_path) as pdf_doc:
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
        metadata_title = metadata.get("title")
        preferred_doc_id_source = (
            metadata_title
            if metadata_title and _looks_like_opaque_stem(pdf_path.stem)
            else pdf_path.stem
        )
        doc_id = _slugify_doc_id(preferred_doc_id_source)

        for page_num, page in enumerate(pdf_doc):
            page_width = float(page.rect.width) or 1.0
            page_height = float(page.rect.height) or 1.0
            raw_blocks = page.get_text("blocks")

            page_blocks = []
            for raw_block in raw_blocks:
                x0, y0, x1, y1, text, _block_no, block_type = raw_block
                if block_type != 0:
                    continue
                clean_text = text.strip()
                if not clean_text:
                    continue
                page_blocks.append((x0, y0, x1, y1, clean_text))

            # Keep a deterministic order even before dedicated multi-column logic lands.
            page_blocks.sort(key=lambda item: (item[1], item[0]))

            candidate_blocks: list[ExtractedBlock] = []
            page_text_parts: list[str] = []
            for x0, y0, x1, y1, clean_text in page_blocks:
                page_text_parts.append(clean_text)
                candidate_blocks.append(
                    _build_extracted_block(
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
                    )
                )

            page_text = "\n\n".join(page_text_parts)
            needs_ocr = page_needs_ocr(page_text)
            final_page_blocks = candidate_blocks
            ocr_used = False
            if needs_ocr:
                ocr_blocks = extract_page_with_ocr(pdf_path=pdf_path, page_num=page_num)
                if ocr_blocks:
                    final_page_blocks = ocr_blocks
                    page_text = "\n\n".join(block.text for block in ocr_blocks)
                    ocr_used = True

            for final_block in final_page_blocks:
                blocks.append(
                    _build_extracted_block(
                        page_num=final_block.page_num,
                        text=final_block.text,
                        bbox=final_block.bbox,
                        reading_order_index=reading_order_index,
                        extraction_method=final_block.extraction_method,
                    )
                )
                reading_order_index += 1

            pages.append(
                ExtractedPage(
                    page_num=page_num,
                    text=page_text,
                    char_count=len(page_text),
                    block_count=len(final_page_blocks),
                    needs_ocr=needs_ocr,
                    ocr_used=ocr_used,
                )
            )

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
    semantics = interpret_document_semantics(
        source_pdf=extraction.source_pdf,
        title=extraction.title or extraction.doc_id,
        toc=extraction.toc,
        summary_cues=summary_cues,
        discovery_terms=discovery_terms,
        leading_block_lines=[block.text.splitlines()[0].strip() for block in extraction.blocks[:30]],
        metadata_values=[value for value in extraction.metadata.values() if isinstance(value, str)],
        page_count=extraction.page_count,
    )
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
        "pages": [
            {
                "page_num": page.page_num,
                "text": page.text,
                "char_count": page.char_count,
                "block_count": page.block_count,
                "needs_ocr": page.needs_ocr,
                "ocr_used": page.ocr_used,
            }
            for page in extraction.pages
        ],
        "blocks": [
            {
                "page_num": block.page_num,
                "text": block.text,
                "bbox": block.bbox,
                "reading_order_index": block.reading_order_index,
                "extraction_method": block.extraction_method,
                "block_kind": block.block_kind,
                "line_count": block.line_count,
                "token_count": block.token_count,
                "structural_flags": block.structural_flags,
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
            )
        )

    return sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]) if block.bbox else (0, 0))


def extract_page_with_ocr(pdf_path: Path, page_num: int) -> list[ExtractedBlock]:
    """OCR fallback for pages with poor native text extraction."""
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return []

    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    pdf_path = pdf_path.expanduser().resolve()
    with fitz.open(pdf_path) as pdf_doc:
        page = pdf_doc[page_num]
        matrix = fitz.Matrix(2, 2)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / f"page-{page_num + 1}.png"
            pixmap.save(str(image_path))
            try:
                image = Image.open(image_path)
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
                )
            ]
