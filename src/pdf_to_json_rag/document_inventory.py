"""Document inventory built from extraction-time metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
from pathlib import Path
import re

from .config import PATHS
from .document_semantics import (
    build_inventory_summary,
    interpret_document_semantics,
    query_semantic_preferences,
    semantic_match_terms,
)


INVENTORY_STOPWORDS = {
    "about",
    "and",
    "are",
    "benchmark",
    "book",
    "books",
    "compare",
    "cover",
    "covers",
    "data",
    "document",
    "documents",
    "file",
    "files",
    "for",
    "from",
    "guidance",
    "humanitarian",
    "in",
    "is",
    "most",
    "note",
    "notes",
    "or",
    "relevant",
    "report",
    "reports",
    "review",
    "source",
    "sources",
    "the",
    "this",
    "what",
    "which",
    "why",
    "with",
}
GENERIC_DOC_TERMS = {
    "book",
    "document",
    "guidance",
    "manual",
    "model",
    "note",
    "report",
    "review",
    "source",
}


@dataclass(frozen=True)
class DocumentInventoryEntry:
    doc_id: str
    label: str
    title: str
    discovery_terms: tuple[str, ...]
    summary_cues: tuple[str, ...]
    inventory_summary: str
    coverage_summary: str
    coverage_terms: tuple[str, ...]
    document_family: str
    document_type: str
    document_purpose: str
    audience: str
    evidence_style: str
    structure_style: str
    facet_terms: tuple[str, ...]
    topical_terms: tuple[str, ...]


def _tokenize(text: str, min_len: int = 3) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{%d,}" % min_len, text.lower())
        if token not in INVENTORY_STOPWORDS
    }


def _humanize_doc_id(doc_id: str) -> str:
    return re.sub(r"\s+", " ", doc_id.replace("-", " ")).strip().title()


def _summary_terms(summary: str) -> set[str]:
    return _tokenize(summary, min_len=2)


def _entry_semantic_terms(entry: DocumentInventoryEntry) -> set[str]:
    semantics = interpret_document_semantics(
        source_pdf="",
        title=entry.title,
        toc=(),
        summary_cues=entry.summary_cues,
        discovery_terms=entry.discovery_terms,
        leading_block_lines=[],
        document_type=entry.document_type,
        document_purpose=entry.document_purpose,
        audience=entry.audience,
        evidence_style=entry.evidence_style,
        structure_style=entry.structure_style,
        facet_terms=entry.facet_terms,
        inventory_summary=entry.inventory_summary,
        document_family=entry.document_family,
        coverage_terms=entry.coverage_terms,
        coverage_summary=entry.coverage_summary,
    )
    return semantic_match_terms(semantics)


@lru_cache(maxsize=1)
def load_document_inventory() -> tuple[DocumentInventoryEntry, ...]:
    from .intent_config import get_document_profile

    documents_dir = PATHS.data_documents
    entries: list[DocumentInventoryEntry] = []
    if not documents_dir.exists():
        return tuple(entries)

    for path in sorted(documents_dir.glob("*.document.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_id = payload.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue

        profile = get_document_profile(doc_id)
        title = payload.get("title") if isinstance(payload.get("title"), str) else ""
        toc = [item for item in payload.get("toc", []) if isinstance(item, str)]
        summary_cues = [item for item in payload.get("summary_cues", []) if isinstance(item, str)]
        discovery_terms = [item for item in payload.get("discovery_terms", []) if isinstance(item, str)]
        label = profile.label if profile else title or _humanize_doc_id(doc_id)
        topical_terms = tuple(sorted(profile.topical_terms)) if profile else tuple()

        semantics = interpret_document_semantics(
            source_pdf=payload.get("source_pdf", ""),
            title=title or label,
            toc=toc,
            summary_cues=summary_cues,
            discovery_terms=discovery_terms,
            leading_block_lines=[],
            metadata_values=[],
            page_count=payload.get("page_count", 0) if isinstance(payload.get("page_count"), int) else 0,
            document_type=payload.get("document_type") if isinstance(payload.get("document_type"), str) else None,
            document_purpose=payload.get("document_purpose") if isinstance(payload.get("document_purpose"), str) else None,
            audience=payload.get("audience") if isinstance(payload.get("audience"), str) else None,
            evidence_style=payload.get("evidence_style") if isinstance(payload.get("evidence_style"), str) else None,
            structure_style=payload.get("structure_style") if isinstance(payload.get("structure_style"), str) else None,
            facet_terms=payload.get("facet_terms") if isinstance(payload.get("facet_terms"), list) else None,
            inventory_summary=payload.get("inventory_summary") if isinstance(payload.get("inventory_summary"), str) and payload.get("inventory_summary") else None,
            document_family=payload.get("document_family") if isinstance(payload.get("document_family"), str) and payload.get("document_family") else None,
            coverage_terms=payload.get("coverage_terms") if isinstance(payload.get("coverage_terms"), list) else None,
            coverage_summary=payload.get("coverage_summary") if isinstance(payload.get("coverage_summary"), str) and payload.get("coverage_summary") else None,
        )

        entries.append(
            DocumentInventoryEntry(
                doc_id=doc_id,
                label=label,
                title=title or label,
                discovery_terms=tuple(discovery_terms[:20]),
                summary_cues=tuple(summary_cues[:8]),
                inventory_summary=semantics.inventory_summary,
                coverage_summary=semantics.coverage_summary,
                coverage_terms=tuple(semantics.coverage_terms),
                document_family=semantics.document_family,
                document_type=semantics.document_type,
                document_purpose=semantics.document_purpose,
                audience=semantics.audience,
                evidence_style=semantics.evidence_style,
                structure_style=semantics.structure_style,
                facet_terms=tuple(semantics.facet_terms),
                topical_terms=topical_terms,
            )
        )
    return tuple(entries)


@lru_cache(maxsize=256)
def get_inventory_entry(doc_id: str) -> DocumentInventoryEntry | None:
    for entry in load_document_inventory():
        if entry.doc_id == doc_id:
            return entry
    return None


def shortlist_documents(query: str, limit: int = 6) -> list[DocumentInventoryEntry]:
    query_lower = query.lower()
    query_terms = _tokenize(query_lower, min_len=2)
    query_terms |= {
        token[:-1]
        for token in list(query_terms)
        if len(token) > 4 and token.endswith("s")
    }
    preferences = query_semantic_preferences(query)
    entries = list(load_document_inventory())
    term_doc_frequency: dict[str, int] = {}
    entry_semantic_term_map: dict[str, set[str]] = {}
    for entry in entries:
        semantic_terms = (
            _entry_semantic_terms(entry)
            | _tokenize(entry.title)
            | {term.lower() for term in entry.discovery_terms}
            | {term.lower() for term in entry.topical_terms}
        )
        semantic_terms -= GENERIC_DOC_TERMS
        entry_semantic_term_map[entry.doc_id] = semantic_terms
        for term in semantic_terms:
            term_doc_frequency[term] = term_doc_frequency.get(term, 0) + 1
    distinctive_query_terms = {
        term
        for term in query_terms
        if term_doc_frequency.get(term, 0) == 1 and term not in GENERIC_DOC_TERMS
    }

    ranked: list[tuple[float, DocumentInventoryEntry]] = []
    for entry in entries:
        title_terms = _tokenize(entry.title)
        label_terms = _tokenize(entry.label)
        discovery_terms = {term.lower() for term in entry.discovery_terms}
        summary_terms = {
            token
            for cue in entry.summary_cues
            for token in _tokenize(cue)
        }
        inventory_terms = _summary_terms(entry.inventory_summary)
        coverage_terms = _summary_terms(entry.coverage_summary) | {
            token for phrase in entry.coverage_terms for token in _tokenize(phrase, min_len=2)
        }
        facet_terms = _entry_semantic_terms(entry)
        topical_terms = {term.lower() for term in entry.topical_terms}

        title_overlap = len(title_terms & query_terms)
        label_overlap = len(label_terms & query_terms)
        discovery_overlap = len(discovery_terms & query_terms)
        facet_overlap = len(facet_terms & query_terms)
        summary_overlap = len(summary_terms & query_terms)
        topical_overlap = len(topical_terms & query_terms)
        inventory_overlap = len(inventory_terms & query_terms)
        coverage_overlap = len(coverage_terms & query_terms)
        unique_title_overlap = len((title_terms - GENERIC_DOC_TERMS) & query_terms)
        unique_discovery_overlap = len((discovery_terms - GENERIC_DOC_TERMS) & query_terms)
        unique_coverage_overlap = len((coverage_terms - GENERIC_DOC_TERMS) & query_terms)
        rare_overlap_bonus = sum(
            1.0 / max(term_doc_frequency.get(term, 1), 1)
            for term in (entry_semantic_term_map[entry.doc_id] & query_terms)
        )
        distinctive_overlap = len(entry_semantic_term_map[entry.doc_id] & distinctive_query_terms)

        exact_title_match = 1 if entry.title.lower() and entry.title.lower() in query_lower else 0
        score = (
            exact_title_match * 12
            + title_overlap * 5
            + label_overlap * 4
            + discovery_overlap * 4
            + facet_overlap * 3
            + summary_overlap * 2
            + inventory_overlap * 2.5
            + coverage_overlap * 3.0
            + topical_overlap * 1.5
            + unique_title_overlap * 4.0
            + unique_discovery_overlap * 3.0
            + unique_coverage_overlap * 3.5
            + rare_overlap_bonus * 6.0
            + distinctive_overlap * 8.0
        )
        if entry.document_family in preferences["families"]:
            score += 2.5
        if entry.document_purpose in preferences["purposes"]:
            score += 2.0
        anchor_candidates = [
            candidate
            for candidate in [entry.label.lower(), *discovery_terms, *entry.coverage_terms]
            if len(candidate) >= 5 and candidate not in INVENTORY_STOPWORDS
        ]
        if any(alias in query_lower for alias in anchor_candidates):
            score += 3.0
        if entry.doc_id in query_lower:
            score += 6.0
        if score > 0:
            ranked.append((score, entry))

    ranked.sort(key=lambda item: (-item[0], item[1].doc_id))
    return [entry for _, entry in ranked[:limit]]


__all__ = [
    "DocumentInventoryEntry",
    "build_inventory_summary",
    "get_inventory_entry",
    "load_document_inventory",
    "shortlist_documents",
]
