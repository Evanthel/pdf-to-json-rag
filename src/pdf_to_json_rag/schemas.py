"""Core JSON schemas for the MVP pipeline."""

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


ChunkType = Literal["text", "table", "figure", "header", "footer", "unknown"]
ExtractionMethod = Literal["native", "ocr", "mixed"]


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    chunk_id: str
    source_pdf: str
    text: str
    page_start: int
    page_end: int
    bbox: list[float] | None = None
    section_title: str | None = None
    section_level: int | None = None
    chunk_type: ChunkType = "text"
    reading_order_index: int
    preceding_chunk_id: str | None = None
    following_chunk_id: str | None = None
    language: str | None = None
    extraction_method: ExtractionMethod = "native"
    ocr_used: bool = False
    subtopic_cues: list[str] = Field(default_factory=list)
    noise_labels: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source_pdf: str
    page_count: int
    title: str | None = None
    toc: list[str] = Field(default_factory=list)
    summary_cues: list[str] = Field(default_factory=list)
    discovery_terms: list[str] = Field(default_factory=list)
    detected_language: str | None = None
    extraction_summary: dict[str, str | int | bool | None] = Field(default_factory=dict)
    chunks: list[ChunkRecord] = Field(default_factory=list)
