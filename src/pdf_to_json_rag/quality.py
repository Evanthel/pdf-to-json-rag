"""Chunk quality and noise-label helpers."""

from __future__ import annotations

import re


NOISY_SECTION_HINTS = {
    "DISCLAIMER",
    "METHODS",
    "QUESTION",
    "QUESTIONS",
    "GRADE",
    "REFERENCES",
}

STATISTICAL_NOISE_HINTS = (
    "95% ci",
    "rr ",
    "favours effect size",
    "results and statistical analysis",
    "not significant",
    "systematic review",
    "rcts",
    "proportion of people reporting",
    "no data from the following reference",
)

TOC_NOISE_HINTS = (
    "what are the effects of treatments for common cold",
    "to be covered in future updates",
    "covered elsewhere in clinical evidence",
)

DISCLAIMER_NOISE_HINTS = (
    "the information contained in this publication is intended for medical professionals",
    "readers should be aware",
    "to the fullest extent permitted by law",
)

BIBLIOGRAPHY_NOISE_HINTS = (
    "[pubmed]",
    "cochrane library",
    "search date",
    "http://www.fda.gov",
    "citation:",
    "competing interests:",
    "correspondence should be addressed",
)

NOISY_SECTION_RE = re.compile(r"^(P\s*=|RR\b|Population\b|Ref\b|Comment:|Very low\b|Low\b)")
TOC_LEADER_RE = re.compile(r"\.\s*\.\s*\.\s*\.")
PAGE_NUMBER_ONLY_RE = re.compile(r"^\d+$")


def classify_chunk_quality(
    text: str,
    section_title: str | None,
    extraction_method: str = "native",
) -> tuple[list[str], float]:
    """Return stable noise labels plus a coarse quality score."""
    labels: set[str] = set()
    section = (section_title or "").upper()
    text_lower = text.lower()
    normalized = " ".join(text_lower.split())

    if extraction_method in {"ocr", "mixed"}:
        labels.add("ocr_derived")

    if any(noisy in section for noisy in NOISY_SECTION_HINTS):
        labels.add("noisy_section")
    if NOISY_SECTION_RE.match(section):
        labels.add("statistical_section")
    if section.startswith("POPULATION") or section.startswith("REF"):
        labels.add("table_like_section")
    if section.startswith("COMMENT:"):
        labels.add("commentary_section")

    if "bmj publishing group" in text_lower or "all rights reserved" in text_lower:
        labels.add("boilerplate")
    if any(hint in text_lower for hint in BIBLIOGRAPHY_NOISE_HINTS):
        labels.add("bibliography")
    if any(hint in text_lower for hint in STATISTICAL_NOISE_HINTS):
        labels.add("statistical_noise")
    if any(hint in text_lower for hint in TOC_NOISE_HINTS):
        labels.add("toc_fragment")
    if any(hint in text_lower for hint in DISCLAIMER_NOISE_HINTS):
        labels.add("disclaimer")
    if TOC_LEADER_RE.search(text):
        labels.add("toc_leader")
    if PAGE_NUMBER_ONLY_RE.match(normalized):
        labels.add("page_number")
    if (
        len(normalized) < 120
        and "." not in text
        and ":" not in text
        and text.strip()
        and sum(1 for token in text.split() if token[:1].isupper()) / max(1, len(text.split())) >= 0.7
    ):
        labels.add("title_fragment")
    if len(normalized) < 25:
        labels.add("short_fragment")

    score = 1.0
    penalty_map = {
        "disclaimer": 0.45,
        "bibliography": 0.30,
        "toc_fragment": 0.35,
        "toc_leader": 0.25,
        "noisy_section": 0.20,
        "statistical_section": 0.20,
        "statistical_noise": 0.20,
        "table_like_section": 0.20,
        "boilerplate": 0.25,
        "commentary_section": 0.10,
        "page_number": 0.40,
        "title_fragment": 0.30,
        "short_fragment": 0.10,
    }
    for label in labels:
        score -= penalty_map.get(label, 0.0)

    score = max(0.0, min(1.0, score))
    return sorted(labels), score
