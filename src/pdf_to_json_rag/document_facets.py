"""Extraction-time and metadata-time document facet derivation."""

from __future__ import annotations

import re


DOCUMENT_TYPE_HINTS = {
    "questionnaire": ("questionnaire", "survey", "interview"),
    "checklist_appendix": ("checklist", "check list", "appendix"),
    "technical_manual": ("manual", "field manual", "technical"),
    "guidance_note": ("guidance note", "guidance"),
    "model_report": ("model report", "trigger", "anticipatory action"),
    "review_article": ("review", "meta-analysis", "systematic review", "literature"),
    "empirical_study": ("study", "trial", "evaluation", "cohort"),
    "book": ("chapter", "foreword", "part "),
}

DOCUMENT_PURPOSE_HINTS = {
    "teaching_reference": ("chapter", "learning", "introduction", "book"),
    "procedural_guidance": ("guidance", "protocol", "procedure", "recommended", "should"),
    "operational_checklist": ("checklist", "screening", "before", "follow-up"),
    "structured_data_capture": ("questionnaire", "survey", "interview"),
    "evidence_summary": ("review", "meta-analysis", "evidence", "conclusion"),
    "risk_or_trigger_assessment": ("trigger", "scenario", "forecast", "drought", "typhoon"),
    "empirical_reporting": ("study", "evaluation", "results", "methods"),
}

AUDIENCE_HINTS = {
    "learners": ("learning", "student", "chapter", "exercise"),
    "practitioners": ("guidance", "operational", "practice", "workflow"),
    "clinicians": ("patient", "therapy", "clinical", "opioid", "treatment"),
    "humanitarian_responders": ("humanitarian", "donors", "cyber threats", "incident", "model report"),
    "analysts": ("analysis", "forecast", "trigger", "model"),
}

EVIDENCE_STYLE_HINTS = {
    "educational_exposition": ("chapter", "foreword", "learning", "concept"),
    "procedural_guidance": ("guidance", "recommended", "procedure", "should"),
    "structured_form": ("questionnaire", "checklist", "yes/no", "appendix"),
    "technical_reference": ("manual", "table", "technical", "field"),
    "evidence_review": ("review", "meta-analysis", "literature"),
    "empirical_study": ("methods", "results", "study", "evaluation"),
    "model_summary": ("model report", "trigger", "forecast"),
}

STRUCTURE_STYLE_HINTS = {
    "chapter_book": ("chapter", "foreword", "part "),
    "report_sections": ("executive summary", "introduction", "conclusion", "recommendations"),
    "review_article": ("abstract", "methods", "results", "discussion"),
    "manual_reference": ("table of contents", "list of tables", "manual"),
    "questionnaire_grid": ("questionnaire", "interview", "table i"),
    "checklist_grid": ("checklist", "yes/no", "appendix"),
}


def _count_hint_matches(text: str, hints: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for hint in hints if hint in lowered)


def _best_facet_match(text: str, hint_map: dict[str, tuple[str, ...]]) -> str | None:
    best_value: str | None = None
    best_score = 0
    for value, hints in hint_map.items():
        score = _count_hint_matches(text, hints)
        if score > best_score:
            best_value = value
            best_score = score
    return best_value


def derive_document_facets(
    *,
    source_pdf: str,
    title: str | None,
    toc: list[str],
    summary_cues: list[str],
    leading_block_lines: list[str],
    metadata_values: list[str] | None = None,
    page_count: int = 0,
) -> dict[str, object]:
    signals = [
        source_pdf,
        title or "",
        *toc[:12],
        *summary_cues[:8],
        *leading_block_lines[:30],
        *(metadata_values or []),
    ]
    signal_text = "\n".join(part for part in signals if part).lower()

    document_type = _best_facet_match(signal_text, DOCUMENT_TYPE_HINTS)
    if document_type is None:
        if page_count >= 100 and ("chapter" in signal_text or "foreword" in signal_text):
            document_type = "book"
        elif "report" in signal_text:
            document_type = "report"
        else:
            document_type = "document"

    document_purpose = _best_facet_match(signal_text, DOCUMENT_PURPOSE_HINTS)
    if document_purpose is None:
        fallback_purpose = {
            "book": "teaching_reference",
            "guidance_note": "procedural_guidance",
            "questionnaire": "structured_data_capture",
            "checklist_appendix": "operational_checklist",
            "model_report": "risk_or_trigger_assessment",
            "review_article": "evidence_summary",
            "empirical_study": "empirical_reporting",
            "technical_manual": "procedural_guidance",
        }
        document_purpose = fallback_purpose.get(document_type, "reference_lookup")

    audience = _best_facet_match(signal_text, AUDIENCE_HINTS)
    if audience is None:
        fallback_audience = {
            "book": "learners",
            "guidance_note": "practitioners",
            "questionnaire": "practitioners",
            "checklist_appendix": "practitioners",
            "model_report": "analysts",
            "review_article": "clinicians",
            "empirical_study": "clinicians",
            "technical_manual": "practitioners",
        }
        audience = fallback_audience.get(document_type, "general_professional")

    evidence_style = _best_facet_match(signal_text, EVIDENCE_STYLE_HINTS)
    if evidence_style is None:
        fallback_evidence_style = {
            "book": "educational_exposition",
            "guidance_note": "procedural_guidance",
            "questionnaire": "structured_form",
            "checklist_appendix": "structured_form",
            "model_report": "model_summary",
            "review_article": "evidence_review",
            "empirical_study": "empirical_study",
            "technical_manual": "technical_reference",
        }
        evidence_style = fallback_evidence_style.get(document_type, "reference_summary")

    structure_style = _best_facet_match(signal_text, STRUCTURE_STYLE_HINTS)
    if structure_style is None:
        fallback_structure_style = {
            "book": "chapter_book",
            "guidance_note": "report_sections",
            "questionnaire": "questionnaire_grid",
            "checklist_appendix": "checklist_grid",
            "model_report": "report_sections",
            "review_article": "review_article",
            "empirical_study": "review_article",
            "technical_manual": "manual_reference",
        }
        structure_style = fallback_structure_style.get(document_type, "report_sections")

    facet_terms = [
        document_type,
        document_purpose,
        audience,
        evidence_style,
        structure_style,
    ]
    return {
        "document_type": document_type,
        "document_purpose": document_purpose,
        "audience": audience,
        "evidence_style": evidence_style,
        "structure_style": structure_style,
        "facet_terms": facet_terms,
    }


def facet_token_terms(facets: dict[str, object]) -> set[str]:
    tokens: set[str] = set()
    for value in facets.values():
        if isinstance(value, str):
            tokens.update(re.findall(r"[a-zA-Z]{3,}", value.lower()))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    tokens.update(re.findall(r"[a-zA-Z]{3,}", item.lower()))
    return tokens
