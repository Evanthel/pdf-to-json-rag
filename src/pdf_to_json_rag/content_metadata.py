"""Shared structural and semantic metadata helpers for extracted content."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9/\-]{2,}")
HEADING_PREFIX_RE = re.compile(r"^(chapter|part|appendix|section)\b", re.IGNORECASE)
QUESTION_PREFIX_RE = re.compile(r"^\d+[\).]?\s+")
TABLE_SIGNAL_RE = re.compile(r"\btable\b|\bcolumn\b|\brow\b|\bfigure\b", re.IGNORECASE)
LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+[\).])\s+")
FORM_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9/&,\-\s]{2,40}\s*[:\-]\s+\S")
KEY_VALUE_COMPACT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9/&,\-\s]{2,24}\s*[:\-]\s+\S")

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
    block_role = "paragraph"
    flags: list[str] = []
    block_labels: list[str] = []
    lower = normalized.lower()

    if not normalized:
        block_kind = "unknown"
        block_role = "unknown"
        flags.append("empty")
    else:
        is_list_like = bool(LIST_PREFIX_RE.match(text.lstrip()))
        has_table_signal = bool(TABLE_SIGNAL_RE.search(normalized))
        has_key_value_signal = bool(FORM_FIELD_RE.match(normalized) or normalized.count(":") >= 2)
        has_compact_key_value = bool(KEY_VALUE_COMPACT_RE.search(normalized))
        if is_list_like:
            block_kind = "list"
            flags.append("list_like")
            block_labels.append("list_item")
            if any(term in lower for term in ("checklist", "confirm", "verify", "screening")):
                block_role = "checklist_item"
                block_labels.append("checklist_item")
            else:
                block_role = "list_item"
        elif has_table_signal and (":" in normalized or "|" in normalized or "\t" in text):
            block_kind = "table_like"
            flags.append("table_like")
            block_role = "table_like"
            block_labels.append("table_like")
        elif has_key_value_signal and ("form" in lower or "registration" in lower or "address" in lower or "date of birth" in lower):
            block_kind = "text"
            block_role = "form_field"
            block_labels.append("form_field")
        elif has_compact_key_value:
            block_kind = "text"
            block_role = "key_value"
            block_labels.append("key_value")
        elif len(lines) <= 2 and (
            normalized.isupper()
            or HEADING_PREFIX_RE.match(normalized)
            or (len(tokens) <= 10 and not normalized.endswith((".", "?", "!")))
        ):
            block_kind = "heading"
            flags.append("heading_like")
            block_role = "heading"
            block_labels.append("heading")
        else:
            block_role = "paragraph"
            block_labels.append("paragraph")

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
    if any(term in lower for term in ("invoice", "account", "registration", "claimant-appellant", "court of appeals")):
        flags.append("domain_signal")

    quality_score = 0.62
    if block_role == "heading":
        quality_score += 0.12
    if block_role in {"table_like", "form_field", "key_value"}:
        quality_score += 0.08
    if "structured_signal" in flags:
        quality_score += 0.05
    if "multi_line" in flags:
        quality_score += 0.04
    if len(tokens) <= 1:
        quality_score -= 0.18
    if len(normalized) < 12:
        quality_score -= 0.1
    if "digit_heavy" in flags and block_role not in {"table_like", "form_field", "key_value"}:
        quality_score -= 0.08
    quality_score = max(0.15, min(0.98, round(quality_score, 3)))

    return {
        "block_kind": block_kind,
        "block_role": block_role,
        "line_count": len(lines) or 1,
        "token_count": len(tokens),
        "text_quality_score": quality_score,
        "block_labels": sorted(set(block_labels)),
        "structural_flags": sorted(set(flags)),
    }


def derive_chunk_semantics(
    *,
    text: str,
    section_title: str | None,
    source_block_kinds: list[str] | tuple[str, ...] = (),
    source_block_roles: list[str] | tuple[str, ...] = (),
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
    block_role_set = {item for item in source_block_roles if item}
    flag_set = {item for item in source_structural_flags if item}

    if "heading" in block_kind_set:
        content_hints.add("heading_supported")
    if "table_like" in block_kind_set:
        content_hints.add("table_like")
    if "list" in block_kind_set:
        content_hints.add("list_like")
    if "form_field" in block_role_set:
        content_hints.add("form_like")
    if "key_value" in block_role_set:
        content_hints.add("key_value_like")
    if "checklist_item" in block_role_set:
        content_hints.add("checklist_like")
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
