"""Embedding and local vector index interfaces for the MVP pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
import numpy as np
from overrides import override

from .content_metadata import derive_chunk_semantics
from .schemas import ChunkRecord


DEFAULT_COLLECTION_NAME = "pdf_to_json_rag_mvp"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_DIM = 384
EMBEDDING_BACKEND_ENV = "PDF_TO_JSON_RAG_EMBEDDING_BACKEND"
SENTENCE_TRANSFORMERS_MODEL_ENV = "PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL"
USE_SENTENCE_TRANSFORMERS_ENV = "PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS"
ALLOW_MODEL_DOWNLOAD_ENV = "PDF_TO_JSON_RAG_ALLOW_MODEL_DOWNLOAD"
SUPPORTED_EMBEDDING_BACKENDS = ("hash", "sentence-transformers", "auto")


class NoopProductTelemetry(ProductTelemetryClient):
    """Disable Chroma product telemetry for local embedded indexes."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None


def local_chroma_client(index_dir: Path):
    """Create an offline local Chroma client without telemetry noise."""
    return chromadb.PersistentClient(
        path=str(index_dir),
        settings=Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl="pdf_to_json_rag.indexing.NoopProductTelemetry",
            chroma_telemetry_impl="pdf_to_json_rag.indexing.NoopProductTelemetry",
        ),
    )


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


def _model_looks_locally_available(model_name: str) -> bool:
    if Path(model_name).expanduser().exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.utils import _CACHED_NO_EXIST

        cached = try_to_load_from_cache(model_name, "config.json")
        return cached not in {None, _CACHED_NO_EXIST}
    except Exception:
        return False


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _requested_embedding_backend() -> str:
    requested = os.environ.get(EMBEDDING_BACKEND_ENV, "").strip().lower()
    if not requested:
        if os.environ.get(USE_SENTENCE_TRANSFORMERS_ENV) == "1":
            return "sentence-transformers"
        if os.environ.get(ALLOW_MODEL_DOWNLOAD_ENV) == "1":
            return "sentence-transformers"
        return "auto"
    if requested in SUPPORTED_EMBEDDING_BACKENDS:
        return requested
    return "hash"


def embedding_runtime_diagnostics(model_name: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, object]:
    """Return public-safe embedding backend configuration and availability."""
    resolved_model = os.environ.get(SENTENCE_TRANSFORMERS_MODEL_ENV, model_name).strip() or model_name
    requested_backend = _requested_embedding_backend()
    legacy_sentence_transformers = os.environ.get(USE_SENTENCE_TRANSFORMERS_ENV) == "1"
    allow_download = os.environ.get(ALLOW_MODEL_DOWNLOAD_ENV) == "1"
    package_available = _sentence_transformers_available()
    model_cached = _model_looks_locally_available(resolved_model)

    fallback_reason = None
    effective_backend = "hash-fallback"
    effective_model = f"hash-{FALLBACK_EMBEDDING_DIM}"
    if requested_backend == "sentence-transformers":
        if package_available and (model_cached or allow_download):
            effective_backend = "sentence-transformers"
            effective_model = resolved_model
        elif not package_available:
            fallback_reason = "sentence-transformers package is not installed"
        else:
            fallback_reason = f"sentence-transformers model is not cached locally: {resolved_model}"
    elif requested_backend == "auto":
        if package_available and model_cached:
            effective_backend = "sentence-transformers"
            effective_model = resolved_model
        else:
            fallback_reason = "auto selected hash fallback because no local sentence-transformer model is ready"

    return {
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
        "effective_model": effective_model,
        "sentence_transformers_package_available": package_available,
        "sentence_transformers_model": resolved_model,
        "sentence_transformers_model_cached": model_cached,
        "allow_model_download": allow_download,
        "legacy_use_sentence_transformers": legacy_sentence_transformers,
        "fallback_reason": fallback_reason,
        "env": {
            EMBEDDING_BACKEND_ENV: os.environ.get(EMBEDDING_BACKEND_ENV),
            USE_SENTENCE_TRANSFORMERS_ENV: os.environ.get(USE_SENTENCE_TRANSFORMERS_ENV),
            SENTENCE_TRANSFORMERS_MODEL_ENV: os.environ.get(SENTENCE_TRANSFORMERS_MODEL_ENV),
            ALLOW_MODEL_DOWNLOAD_ENV: os.environ.get(ALLOW_MODEL_DOWNLOAD_ENV),
        },
    }


def _load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return an embedding callable plus backend metadata."""
    model_name = os.environ.get(SENTENCE_TRANSFORMERS_MODEL_ENV, model_name).strip() or model_name
    diagnostics = embedding_runtime_diagnostics(model_name)
    use_sentence_transformers = diagnostics["effective_backend"] == "sentence-transformers"
    if not use_sentence_transformers:
        def embed_texts(texts: list[str]) -> list[list[float]]:
            return [_hash_embedding(text) for text in texts]

        info = {
            "embedding_backend": "hash-fallback",
            "embedding_model": f"hash-{FALLBACK_EMBEDDING_DIM}",
        }
        if diagnostics.get("fallback_reason"):
            info["embedding_fallback_reason"] = str(diagnostics["fallback_reason"])
        info["embedding_requested_backend"] = str(diagnostics["requested_backend"])
        return embed_texts, info

    allow_download = os.environ.get(ALLOW_MODEL_DOWNLOAD_ENV) == "1"
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not _model_looks_locally_available(model_name):
            def embed_texts(texts: list[str]) -> list[list[float]]:
                return [_hash_embedding(text) for text in texts]

            return embed_texts, {
                "embedding_backend": "hash-fallback",
                "embedding_model": f"hash-{FALLBACK_EMBEDDING_DIM}",
                "embedding_fallback_reason": f"sentence-transformers model is not cached locally: {model_name}",
            }

    try:
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            if allow_download:
                model = SentenceTransformer(model_name)
            else:
                raise

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
            "embedding_requested_backend": str(diagnostics["requested_backend"]),
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
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "section_level": chunk.section_level,
        "section_parent_id": chunk.section_parent_id,
        "section_path": "|".join(chunk.section_path) if chunk.section_path else None,
        "section_kind": chunk.section_kind,
        "section_role": chunk.section_role,
        "section_summary": chunk.section_summary,
        "section_coverage_terms": "|".join(chunk.section_coverage_terms) if chunk.section_coverage_terms else None,
        "section_content_hints": "|".join(chunk.section_content_hints) if chunk.section_content_hints else None,
        "structure_confidence": chunk.structure_confidence,
        "layout_confidence": chunk.layout_confidence,
        "chunk_type": chunk.chunk_type,
        "chunk_strategy": chunk.chunk_strategy,
        "reading_order_index": chunk.reading_order_index,
        "preceding_chunk_id": chunk.preceding_chunk_id,
        "following_chunk_id": chunk.following_chunk_id,
        "language": chunk.language,
        "extraction_method": chunk.extraction_method,
        "text_source": chunk.text_source,
        "ocr_used": chunk.ocr_used,
        "subtopic_cues": "|".join(chunk.subtopic_cues) if chunk.subtopic_cues else None,
        "semantic_terms": "|".join(semantic_terms) if semantic_terms else None,
        "content_hints": "|".join(content_hints) if content_hints else None,
        "structural_flags": "|".join(structural_flags) if structural_flags else None,
        "layout_signals": "|".join(chunk.layout_signals) if chunk.layout_signals else None,
        "source_block_kinds": "|".join(chunk.source_block_kinds) if chunk.source_block_kinds else None,
        "source_block_roles": "|".join(chunk.source_block_roles) if chunk.source_block_roles else None,
        "noise_labels": "|".join(chunk.noise_labels) if chunk.noise_labels else None,
        "text_quality_score": chunk.text_quality_score,
        "quality_score": chunk.quality_score,
        "chunk_text": chunk.text,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _chunk_retrieval_text(chunk: ChunkRecord) -> str:
    """Build the text surface used for embeddings and vector search."""
    parts = [
        chunk.section_title or "",
        chunk.section_summary or "",
        " ".join(chunk.section_coverage_terms),
        " ".join(chunk.section_content_hints),
        chunk.text,
    ]
    return "\n".join(part for part in parts if part.strip())


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
    client = local_chroma_client(index_dir)

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [_chunk_retrieval_text(chunk) for chunk in chunks]
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
        "retrieval_text_surface": "section_metadata_plus_chunk_text",
        **embedder_info,
    }
    _save_index_manifest(index_dir, manifest)
    removed_paths = cleanup_unused_segment_dirs(index_dir)
    if removed_paths:
        manifest["removed_stale_segment_dirs"] = [str(path.name) for path in removed_paths]
        _save_index_manifest(index_dir, manifest)
    return manifest
