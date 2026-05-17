"""Embedding and local vector index interfaces for the MVP pipeline."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np

from .content_metadata import derive_chunk_semantics
from .schemas import ChunkRecord


DEFAULT_COLLECTION_NAME = "pdf_to_json_rag_mvp"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_DIM = 384


def _hash_embedding(text: str, dim: int = FALLBACK_EMBEDDING_DIM) -> list[float]:
    """Deterministic local fallback embedding when no model is available."""
    vector = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return vector.tolist()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.tolist()


def _load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return an embedding callable plus backend metadata."""
    try:
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            model = SentenceTransformer(model_name)

        def embed_texts(texts: list[str]) -> list[list[float]]:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        return embed_texts, {
            "embedding_backend": "sentence-transformers",
            "embedding_model": model_name,
        }
    except Exception:
        def embed_texts(texts: list[str]) -> list[list[float]]:
            return [_hash_embedding(text) for text in texts]

        return embed_texts, {
            "embedding_backend": "hash-fallback",
            "embedding_model": f"hash-{FALLBACK_EMBEDDING_DIM}",
        }


def load_embedder_from_manifest(manifest: dict):
    """Load an embedder matching the saved index manifest."""
    backend = manifest.get("embedding_backend")
    if backend == "hash-fallback":
        def embed_texts(texts: list[str]) -> list[list[float]]:
            return [_hash_embedding(text) for text in texts]

        return embed_texts, {
            "embedding_backend": "hash-fallback",
            "embedding_model": manifest.get("embedding_model", f"hash-{FALLBACK_EMBEDDING_DIM}"),
        }
    model_name = manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    return _load_embedder(model_name=model_name)


def load_chunk_records(chunk_dir: Path) -> list[ChunkRecord]:
    """Load chunk JSON artifacts from disk."""
    chunk_dir = chunk_dir.expanduser().resolve()
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    chunks = []
    for chunk_path in sorted(chunk_dir.glob("*.json")):
        data = json.loads(chunk_path.read_text(encoding="utf-8"))
        chunks.append(ChunkRecord.model_validate(data))
    return chunks


def _chunk_metadata(chunk: ChunkRecord) -> dict[str, str | int | bool | None]:
    semantic_terms = list(chunk.semantic_terms)
    content_hints = list(chunk.content_hints)
    structural_flags = list(chunk.structural_flags)
    if not semantic_terms or not content_hints:
        fallback_terms, fallback_hints, fallback_flags = derive_chunk_semantics(
            text=chunk.text,
            section_title=chunk.section_title,
            source_block_kinds=chunk.source_block_kinds,
            source_structural_flags=chunk.structural_flags,
        )
        if not semantic_terms:
            semantic_terms = fallback_terms
        if not content_hints:
            content_hints = fallback_hints
        if not structural_flags:
            structural_flags = fallback_flags
        chunk.semantic_terms = semantic_terms
        chunk.content_hints = content_hints
        chunk.structural_flags = structural_flags
    metadata = {
        "doc_id": chunk.doc_id,
        "source_pdf": chunk.source_pdf,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_title": chunk.section_title,
        "section_level": chunk.section_level,
        "chunk_type": chunk.chunk_type,
        "reading_order_index": chunk.reading_order_index,
        "preceding_chunk_id": chunk.preceding_chunk_id,
        "following_chunk_id": chunk.following_chunk_id,
        "language": chunk.language,
        "extraction_method": chunk.extraction_method,
        "ocr_used": chunk.ocr_used,
        "subtopic_cues": "|".join(chunk.subtopic_cues) if chunk.subtopic_cues else None,
        "semantic_terms": "|".join(semantic_terms) if semantic_terms else None,
        "content_hints": "|".join(content_hints) if content_hints else None,
        "structural_flags": "|".join(structural_flags) if structural_flags else None,
        "source_block_kinds": "|".join(chunk.source_block_kinds) if chunk.source_block_kinds else None,
        "noise_labels": "|".join(chunk.noise_labels) if chunk.noise_labels else None,
        "quality_score": chunk.quality_score,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _save_index_manifest(index_dir: Path, manifest: dict) -> Path:
    manifest_path = index_dir / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def load_index_manifest(index_dir: Path) -> dict:
    """Load the saved index manifest."""
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def cleanup_unused_segment_dirs(index_dir: Path) -> list[Path]:
    """Remove orphaned Chroma vector segment folders left by previous rebuilds."""
    sqlite_path = index_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return []

    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            "select id from segments where scope = 'VECTOR'"
        ).fetchall()

    keep_ids = {row[0] for row in rows}
    removed_paths: list[Path] = []
    for path in index_dir.iterdir():
        if not path.is_dir():
            continue
        if path.name in keep_ids:
            continue
        # Chroma's persisted vector segments use UUID-style directory names.
        if len(path.name) == 36 and path.name.count("-") == 4:
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
            removed_paths.append(path)
    return removed_paths


def build_local_index(
    chunks: list[ChunkRecord],
    index_dir: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = True,
) -> dict[str, str | int | list[str]]:
    """Create a local vector index from chunk text and metadata."""
    index_dir = index_dir.expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    embed_texts, embedder_info = _load_embedder()
    client = chromadb.PersistentClient(path=str(index_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    ids = [chunk.chunk_id for chunk in chunks]
    metadatas = [_chunk_metadata(chunk) for chunk in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    manifest = {
        "collection_name": collection_name,
        "chunk_count": len(chunks),
        "doc_ids": sorted({chunk.doc_id for chunk in chunks}),
        "source_pdfs": sorted({chunk.source_pdf for chunk in chunks}),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        **embedder_info,
    }
    _save_index_manifest(index_dir, manifest)
    removed_paths = cleanup_unused_segment_dirs(index_dir)
    if removed_paths:
        manifest["removed_stale_segment_dirs"] = [str(path.name) for path in removed_paths]
        _save_index_manifest(index_dir, manifest)
    return manifest
