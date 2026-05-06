"""Extraction-stage interfaces for the MVP pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
import re

import fitz

from .schemas import DocumentRecord


@dataclass
class ExtractedBlock:
    page_num: int
    text: str
    bbox: list[float] | None
    reading_order_index: int


@dataclass
class ExtractedPage:
    page_num: int
    text: str
    char_count: int
    block_count: int
    needs_ocr: bool


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


def extract_native_pdf(pdf_path: Path) -> NativePdfExtraction:
    """Extract page blocks and document metadata from a native-text PDF.

    MVP intent:
    - use PyMuPDF for text blocks and coordinates
    - preserve page-level and block-level ordering metadata
    """
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = _slugify_doc_id(pdf_path.stem)
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

            page_text_parts: list[str] = []
            for x0, y0, x1, y1, clean_text in page_blocks:
                page_text_parts.append(clean_text)
                blocks.append(
                    ExtractedBlock(
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
                        reading_order_index=reading_order_index,
                    )
                )
                reading_order_index += 1

            page_text = "\n\n".join(page_text_parts)
            pages.append(
                ExtractedPage(
                    page_num=page_num,
                    text=page_text,
                    char_count=len(page_text),
                    block_count=len(page_blocks),
                    needs_ocr=page_needs_ocr(page_text),
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
    return DocumentRecord(
        doc_id=extraction.doc_id,
        source_pdf=extraction.source_pdf,
        page_count=extraction.page_count,
        title=extraction.title,
        toc=extraction.toc,
        extraction_summary={
            "native_blocks": len(extraction.blocks),
            "pages_requiring_ocr": pages_requiring_ocr,
            "ocr_used": pages_requiring_ocr > 0,
        },
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
            }
            for page in extraction.pages
        ],
        "blocks": [
            {
                "page_num": block.page_num,
                "text": block.text,
                "bbox": block.bbox,
                "reading_order_index": block.reading_order_index,
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


def extract_page_with_ocr(pdf_path: Path, page_num: int) -> list[ExtractedBlock]:
    """OCR fallback for pages with poor native text extraction."""
    raise NotImplementedError("OCR fallback extraction is not implemented yet.")
