"""Extraction-time document structure helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .content_metadata import derive_chunk_semantics
from .schemas import DocumentSectionRecord

if TYPE_CHECKING:
    from .extraction import ExtractedBlock


HEADING_PREFIX_RE = re.compile(
    r"^(chapter|part|appendix|section)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_structural_heading(text: str, toc_entries: set[str]) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    normalized_key = normalized.lower()
    if normalized_key in toc_entries:
        return True
    if HEADING_PREFIX_RE.match(normalized):
        return True
    words = normalized.split()
    if len(words) > 14:
        return False
    if normalized.endswith((".", "?", "!")):
        return False
    if normalized.isupper():
        return True
    title_case_words = sum(1 for word in words if word[:1].isupper())
    if words and title_case_words / len(words) >= 0.75 and len(normalized) <= 90:
        return True
    return False


def _section_summary_and_terms(
    blocks: list["ExtractedBlock"],
    title: str,
) -> tuple[str | None, list[str], list[str]]:
    body_parts = [
        _normalize(block.text)
        for block in blocks
        if _normalize(block.text) and _normalize(block.text) != _normalize(title)
    ]
    body_text = " ".join(body_parts).strip()
    if not body_text:
        return None, [], []
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(body_text) if part.strip()]
    summary_sentences: list[str] = []
    summary_len = 0
    for sentence in sentences:
        if summary_sentences and summary_len + len(sentence) > 220:
            break
        summary_sentences.append(sentence)
        summary_len += len(sentence) + 1
        if len(summary_sentences) >= 2:
            break
    summary = " ".join(summary_sentences).strip() or None
    coverage_terms, content_hints, _ = derive_chunk_semantics(
        text=body_text,
        section_title=title,
        limit=8,
    )
    return summary, coverage_terms[:8], content_hints


def build_document_sections(
    *,
    doc_id: str,
    title: str | None,
    toc: list[str],
    blocks: list["ExtractedBlock"],
) -> list[DocumentSectionRecord]:
    ordered_blocks = sorted(blocks, key=lambda block: (block.page_num, block.reading_order_index))
    if not ordered_blocks:
        return []

    toc_entries = {_normalize(item).lower() for item in toc if _normalize(item)}
    section_ranges: list[tuple[str, int | None, int, int]] = []

    current_title = title or "Document overview"
    current_level = 1 if title else None
    current_start = 0

    for index, block in enumerate(ordered_blocks):
        block_text = _normalize(block.text)
        if not block_text:
            continue
        if not _looks_like_structural_heading(block_text, toc_entries):
            continue
        if index == current_start and block_text == _normalize(current_title):
            continue
        if index > current_start:
            section_ranges.append((current_title, current_level, current_start, index - 1))
        current_title = block_text
        current_level = 1 if block_text.lower() in toc_entries or HEADING_PREFIX_RE.match(block_text) else None
        current_start = index

    section_ranges.append((current_title, current_level, current_start, len(ordered_blocks) - 1))

    sections: list[DocumentSectionRecord] = []
    for section_number, (section_title, level, start_index, end_index) in enumerate(section_ranges, start=1):
        section_blocks = ordered_blocks[start_index : end_index + 1]
        summary, coverage_terms, content_hints = _section_summary_and_terms(
            section_blocks,
            section_title,
        )
        sections.append(
            DocumentSectionRecord(
                section_id=f"{doc_id}-section-{section_number:03d}",
                title=section_title,
                level=level,
                page_start=section_blocks[0].page_num + 1,
                page_end=section_blocks[-1].page_num + 1,
                reading_order_start=section_blocks[0].reading_order_index,
                reading_order_end=section_blocks[-1].reading_order_index,
                summary=summary,
                coverage_terms=coverage_terms,
                content_hints=content_hints,
            )
        )
    return sections


__all__ = ["build_document_sections"]
