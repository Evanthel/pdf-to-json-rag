"""Chunking interfaces for the MVP pipeline."""

import json
import re
from pathlib import Path

from .extraction import ExtractedBlock
from .quality import TOC_LEADER_RE, PAGE_NUMBER_ONLY_RE, classify_chunk_quality
from .schemas import ChunkRecord, DocumentRecord

INLINE_SECTION_RE = re.compile(
    r"^(?P<label>[A-Z][A-Z/\-\s]{2,50})\s+(?P<body>.*[a-z].*)$"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BOILERPLATE_PATTERNS = (
    "© bmj publishing group ltd",
    "all rights reserved",
    "clinical evidence 2011",
    "respiratory disorders (acute)",
    "common cold search date",
    "the information contained in this publication is intended for medical professionals",
    "what are the effects of treatments for common cold",
    "favours effect size results and statistical analysis outcome",
    "no data from the following reference",
)
REFERENCE_LIKE_RE = re.compile(r"^\d+\.\s+[A-Z][^[]+\[PubMed\]", re.MULTILINE)
SHORT_TOC_NOISE = {
    "likely to be ineffective or harmful",
    "acute sinusitis",
    "acute bronchitis",
    "sore throat",
    "interventions to prevent common cold",
}


def normalize_reading_order(blocks: list[ExtractedBlock]) -> list[ExtractedBlock]:
    """Return blocks in deterministic reading order.

    MVP target:
    - preserve current order for single-column pages
    - later add explicit multi-column normalization
    """
    return sorted(blocks, key=lambda block: (block.page_num, block.reading_order_index))


def build_document_record(
    doc_id: str,
    source_pdf: str,
    page_count: int,
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        source_pdf=source_pdf,
        page_count=page_count,
    )


def _clean_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n|\n(?=[A-Z0-9•-])", text)
    return [part.strip() for part in parts if part.strip()]


def _is_noise_paragraph(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return True
    if PAGE_NUMBER_ONLY_RE.match(normalized):
        return True
    if TOC_LEADER_RE.search(text):
        return True
    if any(pattern in normalized for pattern in BOILERPLATE_PATTERNS):
        return True
    if REFERENCE_LIKE_RE.search(text):
        return True
    if normalized in SHORT_TOC_NOISE:
        return True
    if normalized in {"common cold", "respiratory disorders (acute)", "-", "benefits and harms"}:
        return True
    if normalized.startswith("question ") and "common cold" in normalized:
        return True
    if normalized.startswith("methods clinical evidence search and appraisal"):
        return True
    if normalized.startswith("grade evaluation of interventions for common cold"):
        return True
    if normalized.startswith("search date january"):
        return True
    if "clinical evidence search and appraisal" in normalized:
        return True
    if "to be covered in future updates" in normalized:
        return True
    if "covered elsewhere in clinical evidence" in normalized:
        return True
    if "likely to be beneficial" in normalized or "unlikely to be beneficial" in normalized:
        return True
    if "unknown effectiveness" in normalized:
        return True
    if text.count(". .") >= 3 and (
        "antihistamines" in normalized
        or "decongestants" in normalized
        or "vitamin c" in normalized
        or "steam inhalation" in normalized
    ):
        return True
    if text.count(". .") >= 6:
        return True
    if normalized.count("[pubmed]") >= 2:
        return True
    if normalized.count("search date") >= 2:
        return True
    if normalized.count("95% ci") >= 2:
        return True
    return False


def _sentence_aware_split(text: str, max_chars: int) -> list[str]:
    sentences = [item.strip() for item in SENTENCE_SPLIT_RE.split(text) if item.strip()]
    if not sentences or len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_chars = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if buffer and buffer_chars + sentence_len > max_chars:
            chunks.append(" ".join(buffer).strip())
            buffer = [sentence]
            buffer_chars = sentence_len
            continue
        buffer.append(sentence)
        buffer_chars += sentence_len + 1

    if buffer:
        chunks.append(" ".join(buffer).strip())

    return [chunk for chunk in chunks if chunk]


def _normalize_block_segments(text: str, max_segment_chars: int = 650) -> list[str]:
    lower_text = text.lower()
    if "key points" in lower_text:
        text = text[lower_text.index("key points") + len("key points") :]
    paragraphs = _split_paragraphs(text)
    cleaned: list[str] = []
    for paragraph in paragraphs:
        paragraph = _clean_text(paragraph)
        if not paragraph or _is_noise_paragraph(paragraph):
            continue
        if len(paragraph) > max_segment_chars:
            cleaned.extend(_sentence_aware_split(paragraph, max_segment_chars))
        else:
            cleaned.append(paragraph)
    first_bullet_index = next(
        (index for index, paragraph in enumerate(cleaned) if paragraph.lstrip().startswith("•")),
        None,
    )
    if first_bullet_index is not None:
        cleaned = cleaned[first_bullet_index:]
    return cleaned


def _looks_like_title_case(text: str) -> bool:
    words = [word for word in re.split(r"\s+", text) if word]
    if not words:
        return False
    titled = sum(1 for word in words if word[:1].isupper())
    return titled / len(words) >= 0.7


def _is_probable_header(text: str, toc_entries: set[str]) -> bool:
    normalized = _normalize_for_match(text)
    words = text.split()
    if not normalized:
        return False
    if normalized in toc_entries:
        return True
    if len(text) > 120 or len(words) > 12:
        return False
    if text.endswith((".", "?", "!")):
        return False
    if text.isupper():
        return True
    return _looks_like_title_case(text)


def _merge_bboxes(blocks: list[ExtractedBlock]) -> list[float] | None:
    valid = [block.bbox for block in blocks if block.bbox is not None]
    if not valid:
        return None
    x0 = min(bbox[0] for bbox in valid)
    y0 = min(bbox[1] for bbox in valid)
    x1 = max(bbox[2] for bbox in valid)
    y1 = max(bbox[3] for bbox in valid)
    return [x0, y0, x1, y1]


def _extract_inline_section_label(text: str) -> str | None:
    match = INLINE_SECTION_RE.match(text)
    if not match:
        return None
    label = re.sub(r"\s+", " ", match.group("label")).strip(" -/")
    words = [word for word in label.split() if word]
    if not words or len(words) > 6:
        return None
    uppercase_ratio = sum(1 for word in words if word.upper() == word) / len(words)
    if uppercase_ratio < 0.8:
        return None
    return label


def _infer_extraction_method(blocks: list[ExtractedBlock]) -> tuple[str, bool]:
    methods = {block.extraction_method for block in blocks}
    if methods == {"ocr"}:
        return "ocr", True
    if "ocr" in methods:
        return "mixed", True
    return "native", False


def _make_chunk_record(
    document: DocumentRecord,
    chunk_number: int,
    blocks: list[ExtractedBlock],
    section_title: str | None,
    section_level: int | None,
) -> ChunkRecord:
    text = "\n\n".join(block.text for block in blocks)
    inferred_title = _extract_inline_section_label(blocks[0].text)
    extraction_method, ocr_used = _infer_extraction_method(blocks)
    resolved_section_title = inferred_title or section_title
    noise_labels, quality_score = classify_chunk_quality(
        text=text,
        section_title=resolved_section_title,
        extraction_method=extraction_method,
    )
    return ChunkRecord(
        doc_id=document.doc_id,
        chunk_id=f"{document.doc_id}-chunk-{chunk_number:04d}",
        source_pdf=document.source_pdf,
        text=text,
        page_start=blocks[0].page_num + 1,
        page_end=blocks[-1].page_num + 1,
        bbox=_merge_bboxes(blocks),
        section_title=resolved_section_title,
        section_level=section_level,
        chunk_type="text",
        reading_order_index=blocks[0].reading_order_index,
        language=document.detected_language,
        extraction_method=extraction_method,
        ocr_used=ocr_used,
        noise_labels=noise_labels,
        quality_score=quality_score,
        confidence=None,
    )


def _link_adjacent_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    for index, chunk in enumerate(chunks):
        preceding = chunks[index - 1].chunk_id if index > 0 else None
        following = chunks[index + 1].chunk_id if index < len(chunks) - 1 else None
        chunk.preceding_chunk_id = preceding
        chunk.following_chunk_id = following
    return chunks


def chunk_document(
    document: DocumentRecord,
    blocks: list[ExtractedBlock],
    target_chars: int = 1200,
    min_chunk_chars: int = 350,
) -> list[ChunkRecord]:
    """Convert extracted blocks into chunk-level JSON records."""
    ordered_blocks = normalize_reading_order(blocks)
    toc_entries = {_normalize_for_match(entry) for entry in document.toc}

    chunks: list[ChunkRecord] = []
    buffer: list[ExtractedBlock] = []
    buffer_chars = 0
    chunk_number = 1
    current_section_title: str | None = document.title
    current_section_level: int | None = 1 if document.title else None
    in_key_points_summary = False
    last_buffer_page_num: int | None = None

    def flush_buffer() -> None:
        nonlocal buffer, buffer_chars, chunk_number, last_buffer_page_num
        if not buffer:
            return
        chunks.append(
            _make_chunk_record(
                document=document,
                chunk_number=chunk_number,
                blocks=buffer,
                section_title=current_section_title,
                section_level=current_section_level,
            )
        )
        chunk_number += 1
        buffer = []
        buffer_chars = 0
        last_buffer_page_num = None

    for block in ordered_blocks:
        raw_block_text = _clean_text(block.text)
        if _normalize_for_match(raw_block_text) == "key points":
            flush_buffer()
            in_key_points_summary = True
            continue

        if in_key_points_summary and last_buffer_page_num is not None and block.page_num != last_buffer_page_num:
            flush_buffer()
            in_key_points_summary = False

        normalized_segments = _normalize_block_segments(block.text)
        if not normalized_segments:
            continue

        for segment in normalized_segments:
            inline_section_title = _extract_inline_section_label(segment)
            if inline_section_title:
                flush_buffer()
                current_section_title = inline_section_title
                current_section_level = None

            if _is_probable_header(segment, toc_entries):
                flush_buffer()
                current_section_title = segment
                current_section_level = (
                    1 if _normalize_for_match(segment) in toc_entries else None
                )
                in_key_points_summary = False
                continue

            is_bullet_summary = in_key_points_summary and segment.lstrip().startswith("•")
            if is_bullet_summary and buffer:
                flush_buffer()

            prospective_chars = buffer_chars + len(segment)
            should_split = (
                buffer and prospective_chars > target_chars and buffer_chars >= min_chunk_chars
            )
            if should_split:
                flush_buffer()

            buffer.append(
                ExtractedBlock(
                    page_num=block.page_num,
                    text=segment,
                    bbox=block.bbox,
                    reading_order_index=block.reading_order_index,
                    extraction_method=block.extraction_method,
                )
            )
            buffer_chars += len(segment)
            last_buffer_page_num = block.page_num

    flush_buffer()
    return _link_adjacent_chunks(chunks)


def load_document_record(document_path: Path) -> DocumentRecord:
    """Load a saved document-level JSON artifact."""
    data = json.loads(document_path.read_text(encoding="utf-8"))
    return DocumentRecord.model_validate(data)


def load_blocks_from_native_json(native_path: Path) -> list[ExtractedBlock]:
    """Load extracted blocks from a saved native JSON artifact."""
    data = json.loads(native_path.read_text(encoding="utf-8"))
    return [
        ExtractedBlock(
            page_num=block["page_num"],
            text=block["text"],
            bbox=block.get("bbox"),
            reading_order_index=block["reading_order_index"],
            extraction_method=block.get("extraction_method", "native"),
        )
        for block in data.get("blocks", [])
    ]


def save_chunk_records(
    chunks: list[ChunkRecord],
    output_dir: Path,
    doc_id: str,
) -> list[Path]:
    """Write chunk JSON files to a per-document output folder."""
    doc_dir = output_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    for existing_chunk_path in doc_dir.glob("*.json"):
        existing_chunk_path.unlink()
    saved_paths = []
    for chunk in chunks:
        output_path = doc_dir / f"{chunk.chunk_id}.json"
        output_path.write_text(
            json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_paths.append(output_path)
    return saved_paths


def save_document_with_chunks(document: DocumentRecord, document_path: Path) -> Path:
    """Rewrite the document-level JSON with embedded chunk metadata."""
    document_path.write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return document_path


def process_saved_document_to_chunks(
    native_path: Path,
    document_path: Path,
    output_dir: Path,
) -> tuple[DocumentRecord, list[ChunkRecord], list[Path]]:
    """Load saved extraction artifacts, generate chunks, and persist them."""
    document = load_document_record(document_path)
    blocks = load_blocks_from_native_json(native_path)
    chunks = chunk_document(document, blocks)
    document.chunks = chunks
    saved_paths = save_chunk_records(chunks, output_dir, document.doc_id)
    save_document_with_chunks(document, document_path)
    return document, chunks, saved_paths
