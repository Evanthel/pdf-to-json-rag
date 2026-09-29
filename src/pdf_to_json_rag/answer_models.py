"""Data contracts shared by evidence selection and answer assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import ChunkRecord


@dataclass
class EvidenceSentence:
    chunk_id: str
    page_start: int
    page_end: int
    section_title: str | None
    sentence: str
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    evidence: list[EvidenceSentence]
    top_k_hits: list[ChunkRecord]
    expanded_hits: list[ChunkRecord]
    query_intent: str
    answer_trace: dict[str, object]


@dataclass
class DocumentSelection:
    answer_mode: str
    candidate_doc_ids: list[str]
    ranked_doc_ids: list[str]
    selected_doc_ids: list[str]
    primary_doc_id: str | None
    strategy: str
    shortlist_breakdown: list[dict[str, object]] = field(default_factory=list)


@dataclass
class DocumentSynthesis:
    selection: DocumentSelection
    support_scope: str
    support_doc_ids: list[str]
    answer_chunk_doc_ids: list[str]
    selected_chunk_count: int
    answer_chunks: list[ChunkRecord] = field(default_factory=list)
    support_entries: list[dict[str, object]] = field(default_factory=list)
