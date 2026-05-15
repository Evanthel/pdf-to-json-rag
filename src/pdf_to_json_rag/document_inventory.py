"""Document inventory built from extraction-time metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
from pathlib import Path
import re

from .document_facets import derive_document_facets, facet_token_terms


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


@dataclass(frozen=True)
class DocumentInventoryEntry:
    doc_id: str
    label: str
    title: str
    discovery_terms: tuple[str, ...]
    summary_cues: tuple[str, ...]
    inventory_summary: str
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


def build_inventory_summary(
    *,
    title: str,
    document_type: str,
    document_purpose: str,
    audience: str,
    evidence_style: str,
    structure_style: str,
    summary_cues: list[str] | tuple[str, ...],
) -> str:
    label = title.strip() if isinstance(title, str) else ""
    parts: list[str] = []
    if document_type:
        parts.append(document_type.replace("_", " "))
    if document_purpose:
        parts.append(document_purpose.replace("_", " "))
    if audience and audience != "general_professional":
        parts.append(f"for {audience.replace('_', ' ')}")
    if evidence_style:
        parts.append(evidence_style.replace("_", " "))
    if structure_style:
        parts.append(structure_style.replace("_", " "))
    if summary_cues:
        parts.append("topics: " + ", ".join(summary_cues[:3]))
    if not parts:
        return label
    if label:
        return f"{label} | " + "; ".join(parts)
    return "; ".join(parts)


def _summary_terms(summary: str) -> set[str]:
    return _tokenize(summary, min_len=2)


@lru_cache(maxsize=1)
def load_document_inventory() -> tuple[DocumentInventoryEntry, ...]:
    from .intent_config import get_document_profile

    documents_dir = Path(__file__).resolve().parents[2] / "data" / "documents"
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
        title = payload.get("title") if isinstance(payload.get("title"), str) else ""
        toc = [item for item in payload.get("toc", []) if isinstance(item, str)]
        summary_cues = [item for item in payload.get("summary_cues", []) if isinstance(item, str)]
        discovery_terms = [item for item in payload.get("discovery_terms", []) if isinstance(item, str)]

        derived_facets = derive_document_facets(
            source_pdf=payload.get("source_pdf", ""),
            title=title,
            toc=toc,
            summary_cues=summary_cues,
            leading_block_lines=[],
            metadata_values=[],
            page_count=payload.get("page_count", 0) if isinstance(payload.get("page_count"), int) else 0,
        )
        profile = get_document_profile(doc_id)
        label = profile.label if profile else title or _humanize_doc_id(doc_id)
        topical_terms = tuple(sorted(profile.topical_terms)) if profile else tuple()
        document_type = (
            payload.get("document_type")
            if isinstance(payload.get("document_type"), str)
            else str(derived_facets["document_type"])
        )
        document_purpose = (
            payload.get("document_purpose")
            if isinstance(payload.get("document_purpose"), str)
            else str(derived_facets["document_purpose"])
        )
        audience = (
            payload.get("audience")
            if isinstance(payload.get("audience"), str)
            else str(derived_facets["audience"])
        )
        evidence_style = (
            payload.get("evidence_style")
            if isinstance(payload.get("evidence_style"), str)
            else str(derived_facets["evidence_style"])
        )
        structure_style = (
            payload.get("structure_style")
            if isinstance(payload.get("structure_style"), str)
            else str(derived_facets["structure_style"])
        )
        facet_terms = tuple(
            item
            for item in (
                payload.get("facet_terms")
                if isinstance(payload.get("facet_terms"), list)
                else derived_facets["facet_terms"]
            )
            if isinstance(item, str)
        )
        inventory_summary = (
            payload.get("inventory_summary")
            if isinstance(payload.get("inventory_summary"), str)
            and payload.get("inventory_summary")
            else build_inventory_summary(
                title=title or label,
                document_type=document_type,
                document_purpose=document_purpose,
                audience=audience,
                evidence_style=evidence_style,
                structure_style=structure_style,
                summary_cues=summary_cues,
            )
        )

        entries.append(
            DocumentInventoryEntry(
                doc_id=doc_id,
                label=label,
                title=title or label,
                discovery_terms=tuple(discovery_terms[:20]),
                summary_cues=tuple(summary_cues[:8]),
                inventory_summary=inventory_summary,
                document_type=document_type,
                document_purpose=document_purpose,
                audience=audience,
                evidence_style=evidence_style,
                structure_style=structure_style,
                facet_terms=facet_terms,
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

    ranked: list[tuple[float, DocumentInventoryEntry]] = []
    for entry in load_document_inventory():
        title_terms = _tokenize(entry.title)
        label_terms = _tokenize(entry.label)
        discovery_terms = {term.lower() for term in entry.discovery_terms}
        summary_terms = {
            token
            for cue in entry.summary_cues
            for token in _tokenize(cue)
        }
        facet_terms = facet_token_terms(
            {
                "document_type": entry.document_type,
                "document_purpose": entry.document_purpose,
                "audience": entry.audience,
                "evidence_style": entry.evidence_style,
                "structure_style": entry.structure_style,
                "facet_terms": list(entry.facet_terms),
            }
        )
        topical_terms = {term.lower() for term in entry.topical_terms}
        inventory_terms = _summary_terms(entry.inventory_summary)

        title_overlap = len(title_terms & query_terms)
        label_overlap = len(label_terms & query_terms)
        discovery_overlap = len(discovery_terms & query_terms)
        facet_overlap = len(facet_terms & query_terms)
        summary_overlap = len(summary_terms & query_terms)
        topical_overlap = len(topical_terms & query_terms)
        inventory_overlap = len(inventory_terms & query_terms)

        exact_title_match = 1 if entry.title.lower() and entry.title.lower() in query_lower else 0
        score = (
            exact_title_match * 12
            + title_overlap * 5
            + label_overlap * 4
            + discovery_overlap * 4
            + facet_overlap * 3
            + summary_overlap * 2
            + inventory_overlap * 2.5
            + topical_overlap * 1.5
        )
        if score > 0:
            ranked.append((score, entry))

    ranked.sort(key=lambda item: (-item[0], item[1].doc_id))
    return [entry for _, entry in ranked[:limit]]
