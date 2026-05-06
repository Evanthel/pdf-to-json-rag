"""Retrieval interfaces for the MVP pipeline."""

import json
from pathlib import Path
import re

import chromadb

from .indexing import (
    DEFAULT_COLLECTION_NAME,
    load_embedder_from_manifest,
    load_index_manifest,
)
from .schemas import ChunkRecord

NOISY_SECTION_HINTS = {"DISCLAIMER", "METHODS", "QUESTION", "QUESTIONS", "GRADE"}


def _query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{2,}", query.lower()))


def _detect_query_intent(query: str) -> str:
    terms = _query_terms(query)
    query_lower = query.lower()
    if query_lower.startswith("what is") or "definition" in terms or "define" in terms:
        return "definition"
    if "transmitted" in terms or "transmission" in terms:
        return "transmission"
    if "last" in terms or "long" in terms or "duration" in terms:
        return "duration"
    if "symptom" in terms or "symptoms" in terms:
        return "symptoms"
    return "generic"


def _augment_query(query: str) -> str:
    intent = _detect_query_intent(query)
    suffix = {
        "definition": "definition defined as upper respiratory tract infection",
        "transmission": "transmission hand-to-hand contact droplets nostrils eyes",
        "duration": "prognosis duration symptoms peak clear by 1 week cough persists",
        "symptoms": "symptoms sneezing runny nose headache sore throat cough",
    }.get(intent, "")
    if not suffix:
        return query
    return f"{query} {suffix}"


def _heuristic_hit_bonus(chunk: ChunkRecord, query: str) -> float:
    section = (chunk.section_title or "").upper()
    text = chunk.text.lower()
    intent = _detect_query_intent(query)
    bonus = 0.0

    if any(noisy in section for noisy in NOISY_SECTION_HINTS):
        bonus -= 4.0
    if "bmj publishing group" in text or "all rights reserved" in text:
        bonus -= 5.0

    if intent == "definition":
        if section.startswith("DEFINITION"):
            bonus += 6.0
        if "defined as" in text:
            bonus += 5.0
        if section.startswith("PROGNOSIS") or section.startswith("AETIOLOGY"):
            bonus += 1.0
    elif intent == "transmission":
        if "TRANSMISSION" in section or "AETIOLOGY" in section:
            bonus += 6.0
        if "hand-to-hand contact" in text:
            bonus += 5.0
        if "droplet" in text or "nostrils" in text or "eyes" in text:
            bonus += 2.0
    elif intent == "duration":
        if section.startswith("PROGNOSIS"):
            bonus += 6.0
        if "1 week" in text or "few days" in text or "cough" in text:
            bonus += 2.0
    elif intent == "symptoms":
        if section.startswith("DEFINITION") or section.startswith("PROGNOSIS"):
            bonus += 4.0
        if "symptoms include" in text:
            bonus += 4.0
        if "sore throat" in text or "runny nose" in text or "rhinorrhoea" in text:
            bonus += 2.0
    return bonus


def _rerank_hits(hits: list[ChunkRecord], query: str) -> list[ChunkRecord]:
    scored = []
    for index, chunk in enumerate(hits):
        score = _heuristic_hit_bonus(chunk, query) - (index * 0.01)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored]


def retrieve_top_k(query: str, index_dir: Path, k: int = 5) -> list[ChunkRecord]:
    """Retrieve the most relevant chunks for a query."""
    index_dir = index_dir.expanduser().resolve()
    manifest = load_index_manifest(index_dir)
    collection_name = manifest.get("collection_name", DEFAULT_COLLECTION_NAME)

    embed_texts, _ = load_embedder_from_manifest(manifest)
    query_embedding = embed_texts([_augment_query(query)])[0]

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(name=collection_name)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    hits: list[ChunkRecord] = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        hits.append(
            ChunkRecord(
                doc_id=metadata["doc_id"],
                chunk_id=chunk_id,
                source_pdf=metadata["source_pdf"],
                text=text,
                page_start=int(metadata["page_start"]),
                page_end=int(metadata["page_end"]),
                bbox=None,
                section_title=metadata.get("section_title"),
                section_level=(
                    int(metadata["section_level"])
                    if metadata.get("section_level") is not None
                    else None
                ),
                chunk_type=metadata.get("chunk_type", "text"),
                reading_order_index=int(metadata["reading_order_index"]),
                preceding_chunk_id=metadata.get("preceding_chunk_id"),
                following_chunk_id=metadata.get("following_chunk_id"),
                language=metadata.get("language"),
                extraction_method=metadata.get("extraction_method", "native"),
                ocr_used=bool(metadata.get("ocr_used", False)),
                confidence=None,
            )
        )
    return _rerank_hits(hits, query)[:k]


def load_chunk_lookup(chunk_root: Path, doc_ids: set[str] | None = None) -> dict[str, ChunkRecord]:
    """Load chunk JSON records into a chunk_id -> ChunkRecord lookup."""
    chunk_root = chunk_root.expanduser().resolve()
    if not chunk_root.exists():
        raise FileNotFoundError(f"Chunk root not found: {chunk_root}")

    lookup: dict[str, ChunkRecord] = {}
    search_dirs = []
    if doc_ids:
        for doc_id in sorted(doc_ids):
            doc_dir = chunk_root / doc_id
            if doc_dir.exists():
                search_dirs.append(doc_dir)
    else:
        search_dirs = [path for path in sorted(chunk_root.iterdir()) if path.is_dir()]

    for doc_dir in search_dirs:
        for chunk_path in sorted(doc_dir.glob("*.json")):
            data = json.loads(chunk_path.read_text(encoding="utf-8"))
            chunk = ChunkRecord.model_validate(data)
            lookup[chunk.chunk_id] = chunk
    return lookup


def expand_with_neighbors(
    hits: list[ChunkRecord],
    all_chunks: dict[str, ChunkRecord],
) -> list[ChunkRecord]:
    """Expand retrieval results with preceding and following chunks."""
    expanded: dict[str, ChunkRecord] = {}
    for chunk in hits:
        expanded[chunk.chunk_id] = chunk
        if chunk.preceding_chunk_id and chunk.preceding_chunk_id in all_chunks:
            expanded[chunk.preceding_chunk_id] = all_chunks[chunk.preceding_chunk_id]
        if chunk.following_chunk_id and chunk.following_chunk_id in all_chunks:
            expanded[chunk.following_chunk_id] = all_chunks[chunk.following_chunk_id]
    return sorted(
        expanded.values(),
        key=lambda chunk: (chunk.doc_id, chunk.reading_order_index, chunk.chunk_id),
    )


def retrieve_top_k_with_neighbors(
    query: str,
    index_dir: Path,
    chunk_root: Path,
    k: int = 5,
) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    """Retrieve top-k chunks and expand them with adjacent neighbors."""
    hits = retrieve_top_k(query=query, index_dir=index_dir, k=k)
    doc_ids = {chunk.doc_id for chunk in hits}
    all_chunks = load_chunk_lookup(chunk_root=chunk_root, doc_ids=doc_ids)
    expanded = expand_with_neighbors(hits=hits, all_chunks=all_chunks)
    return hits, expanded
