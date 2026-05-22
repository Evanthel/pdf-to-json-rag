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
NUMBERED_HEADING_RE = re.compile(r"^(?P<number>\d+(?:\.\d+){0,3})(?:[\).:-]|\s)\s*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
QUESTION_HEADING_RE = re.compile(r"^(question\s+\d+\b|section\s+\d+\b)", re.IGNORECASE)


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
    if NUMBERED_HEADING_RE.match(normalized):
        return True
    if QUESTION_HEADING_RE.match(normalized):
        return True
    title_case_words = sum(1 for word in words if word[:1].isupper())
    if words and title_case_words / len(words) >= 0.75 and len(normalized) <= 90:
        return True
    return False


def _heading_level(text: str, toc_entries: set[str]) -> int | None:
    normalized = _normalize(text)
    if not normalized:
        return None
    if normalized.lower() in toc_entries:
        return 1
    prefix_match = HEADING_PREFIX_RE.match(normalized)
    if prefix_match:
        keyword = prefix_match.group(1).lower()
        if keyword in {"chapter", "part", "appendix"}:
            return 1
        if keyword == "section":
            return 2
    numbered_match = NUMBERED_HEADING_RE.match(normalized)
    if numbered_match:
        number = numbered_match.group("number")
        return min(number.count(".") + 1, 4)
    if normalized.isupper():
        return 1
    return None


def _section_kind(
    title: str,
    body_text: str,
    content_hints: list[str],
    blocks: list["ExtractedBlock"],
) -> str | None:
    title_lower = _normalize(title).lower()
    body_lower = body_text.lower()
    block_kinds = {block.block_kind for block in blocks if block.block_kind}
    hint_set = set(content_hints)

    if "appendix" in title_lower:
        return "appendix"
    if "table" in title_lower or "table_like" in block_kinds:
        return "table_section"
    if QUESTION_HEADING_RE.match(title_lower) or "questionnaire_like" in hint_set:
        return "questionnaire_section"
    if "checklist" in title_lower or "checklist_like" in hint_set:
        return "checklist_section"
    if "procedural_like" in hint_set:
        return "procedural_section"
    if "evidence_summary" in hint_set or "comparative_evidence" in hint_set:
        return "summary_section"
    if "introduction" in title_lower or "background" in title_lower:
        return "intro_section"
    if body_lower.count(":") >= 3 and "table_like" not in block_kinds:
        return "checklist_section"
    return "report_section"


def _section_summary_and_terms(
    blocks: list["ExtractedBlock"],
    title: str,
) -> tuple[str | None, list[str], list[str], str | None]:
    body_parts = [
        _normalize(block.text)
        for block in blocks
        if _normalize(block.text) and _normalize(block.text) != _normalize(title)
    ]
    body_text = " ".join(body_parts).strip()
    if not body_text:
        return None, [], [], None
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
    return (
        summary,
        coverage_terms[:8],
        content_hints,
        _section_kind(title, body_text, content_hints, blocks),
    )


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
        current_level = _heading_level(block_text, toc_entries)
        current_start = index

    section_ranges.append((current_title, current_level, current_start, len(ordered_blocks) - 1))

    sections: list[DocumentSectionRecord] = []
    for section_number, (section_title, level, start_index, end_index) in enumerate(section_ranges, start=1):
        section_blocks = ordered_blocks[start_index : end_index + 1]
        summary, coverage_terms, content_hints, section_kind = _section_summary_and_terms(
            section_blocks,
            section_title,
        )
        parent_section_id: str | None = None
        section_path: list[str] = [section_title]
        if level is not None:
            for previous_section in reversed(sections):
                previous_level = previous_section.level or 1
                if previous_level < level:
                    parent_section_id = previous_section.section_id
                    section_path = [*previous_section.section_path, section_title]
                    break
        if not sections and title and section_title != title and section_path == [section_title]:
            section_path = [title, section_title]
        sections.append(
            DocumentSectionRecord(
                section_id=f"{doc_id}-section-{section_number:03d}",
                title=section_title,
                level=level,
                parent_section_id=parent_section_id,
                section_path=section_path,
                section_kind=section_kind,
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
