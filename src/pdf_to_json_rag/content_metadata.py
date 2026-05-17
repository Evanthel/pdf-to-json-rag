"""Shared structural and semantic metadata helpers for extracted content."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9/\-]{2,}")
HEADING_PREFIX_RE = re.compile(r"^(chapter|part|appendix|section)\b", re.IGNORECASE)
QUESTION_PREFIX_RE = re.compile(r"^\d+[\).]?\s+")
TABLE_SIGNAL_RE = re.compile(r"\btable\b|\bcolumn\b|\brow\b|\bfigure\b", re.IGNORECASE)
LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+[\).])\s+")

SEMANTIC_STOPWORDS = {
    "about",
    "after",
    "among",
    "because",
    "before",
    "between",
    "chapter",
    "chapters",
    "common",
    "could",
    "document",
    "does",
    "during",
    "file",
    "files",
    "from",
    "have",
    "into",
    "into",
    "many",
    "more",
    "most",
    "other",
    "over",
    "page",
    "pages",
    "review",
    "section",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

GENERIC_SECTION_STOPWORDS = {
    "abstract",
    "appendix",
    "background",
    "bibliography",
    "chapter",
    "conclusion",
    "contents",
    "discussion",
    "figure",
    "foreword",
    "index",
    "introduction",
    "methods",
    "references",
    "results",
    "review",
    "section",
    "summary",
    "table",
}

CONTENT_HINT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("definition_like", ("defined as", "is a", "refers to")),
    ("overview_like", ("table of contents", "chapter", "this book", "this guide")),
    ("procedural_like", ("should", "must", "perform", "follow-up", "step", "checklist")),
    ("questionnaire_like", ("question", "not at all", "yes", "no", "response scale")),
    ("checklist_like", ("checklist", "screening", "contra-indications", "cautions")),
    ("risk_or_warning", ("risk", "warning", "adverse effect", "contra-indication", "danger")),
    ("quantitative_evidence", ("%", "odds", "ratio", "ci", "minutes", "days", "weeks")),
    ("comparative_evidence", ("compared with", "versus", "vs", "difference between")),
    ("conclusion_like", ("conclusion", "collective evidence", "suggests that", "supports")),
    ("table_like", ("table ", "column", "row", "scale:", "0 = none", "1 = ")),
)


def _normalize_text(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def classify_block_metadata(text: str) -> dict[str, object]:
    """Infer lightweight structural metadata for an extracted block."""
    normalized = _normalize_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokens = _tokenize(normalized)
    block_kind = "text"
    flags: list[str] = []
    lower = normalized.lower()

    if not normalized:
        block_kind = "unknown"
        flags.append("empty")
    else:
        if LIST_PREFIX_RE.match(text.lstrip()):
            block_kind = "list"
            flags.append("list_like")
        elif TABLE_SIGNAL_RE.search(normalized) and (":" in normalized or "|" in normalized):
            block_kind = "table_like"
            flags.append("table_like")
        elif len(lines) <= 2 and (
            normalized.isupper()
            or HEADING_PREFIX_RE.match(normalized)
            or (len(tokens) <= 10 and not normalized.endswith((".", "?", "!")))
        ):
            block_kind = "heading"
            flags.append("heading_like")

    if QUESTION_PREFIX_RE.match(normalized) or normalized.endswith("?"):
        flags.append("question_like")
    if normalized.count(":") >= 1:
        flags.append("colon_heavy")
    if sum(ch.isdigit() for ch in normalized) >= max(3, len(normalized) // 10):
        flags.append("digit_heavy")
    if len(lines) >= 3:
        flags.append("multi_line")
    if len(tokens) >= 60:
        flags.append("dense_paragraph")
    if any(term in lower for term in ("table ", "appendix", "checklist", "questionnaire")):
        flags.append("structured_signal")

    return {
        "block_kind": block_kind,
        "line_count": len(lines) or 1,
        "token_count": len(tokens),
        "structural_flags": sorted(set(flags)),
    }


def derive_chunk_semantics(
    *,
    text: str,
    section_title: str | None,
    source_block_kinds: list[str] | tuple[str, ...] = (),
    source_structural_flags: list[str] | tuple[str, ...] = (),
    limit: int = 14,
) -> tuple[list[str], list[str], list[str]]:
    """Infer semantic terms, content hints, and structural flags for a chunk."""
    normalized = _normalize_text(text)
    lower = normalized.lower()
    tokens = _tokenize(normalized)

    semantic_terms: list[str] = []
    seen_terms: set[str] = set()
    for token in tokens:
        if token in SEMANTIC_STOPWORDS:
            continue
        if token in seen_terms:
            continue
        seen_terms.add(token)
        semantic_terms.append(token)
        if len(semantic_terms) >= limit:
            break

    if section_title:
        for token in _tokenize(section_title):
            if token in GENERIC_SECTION_STOPWORDS or token in seen_terms:
                continue
            seen_terms.add(token)
            semantic_terms.append(token)
            if len(semantic_terms) >= limit:
                break

    content_hints = set()
    for hint, patterns in CONTENT_HINT_PATTERNS:
        if any(pattern in lower for pattern in patterns):
            content_hints.add(hint)

    block_kind_set = {item for item in source_block_kinds if item}
    flag_set = {item for item in source_structural_flags if item}

    if "heading" in block_kind_set:
        content_hints.add("heading_supported")
    if "table_like" in block_kind_set:
        content_hints.add("table_like")
    if "list" in block_kind_set:
        content_hints.add("list_like")
    if "question_like" in flag_set:
        content_hints.add("questionnaire_like")
    if "structured_signal" in flag_set:
        content_hints.add("structured_signal")

    structural_flags = set(flag_set)
    if section_title and section_title.upper().startswith(("CONCLUSION", "SUMMARY")):
        structural_flags.add("summary_section")
        content_hints.add("conclusion_like")
    if section_title and section_title.upper().startswith(("CONTENTS", "FOREWORD", "INTRODUCTION")):
        structural_flags.add("overview_section")
        content_hints.add("overview_like")
    if any(term in lower for term in ("evidence", "review", "meta-analysis")):
        content_hints.add("evidence_summary")

    return semantic_terms, sorted(content_hints), sorted(structural_flags)
