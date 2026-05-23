"""Extraction-time and metadata-time document facet derivation."""

from __future__ import annotations

import re


DOCUMENT_TYPE_HINTS = {
    "financial_statement": (
        "financial statement",
        "net worth",
        "total assets",
        "total liabilities",
        "cash in banks",
        "mortgage payable",
    ),
    "assessment_form": (
        "assessment form",
        "financial assessment",
        "care charge form",
        "service user",
        "representative details",
    ),
    "administrative_form": (
        "personal details",
        "date of birth",
        "national insurance",
        "telephone no",
        "telephone numbers",
    ),
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
    "financial_disclosure": (
        "financial statement",
        "net worth",
        "assets",
        "liabilities",
        "market value",
        "cash in banks",
    ),
    "financial_assessment": (
        "financial assessment",
        "care charge",
        "service user",
        "income",
        "capital",
        "local authority",
    ),
    "administrative_intake": (
        "personal details",
        "representative details",
        "date of birth",
        "address",
        "telephone",
    ),
    "teaching_reference": ("chapter", "learning", "introduction", "book"),
    "procedural_guidance": ("guidance", "protocol", "procedure", "recommended", "should"),
    "operational_checklist": ("checklist", "screening", "before", "follow-up"),
    "structured_data_capture": ("questionnaire", "survey", "interview"),
    "evidence_summary": ("review", "meta-analysis", "evidence", "conclusion"),
    "risk_or_trigger_assessment": ("trigger", "scenario", "forecast", "drought", "typhoon"),
    "empirical_reporting": ("study", "evaluation", "results", "methods"),
}

AUDIENCE_HINTS = {
    "applicants": ("applicant", "service user", "client", "borrower"),
    "case_workers": ("local authority", "assessor", "representative", "financial affairs"),
    "learners": ("learning", "student", "chapter", "exercise"),
    "practitioners": ("guidance", "operational", "practice", "workflow"),
    "clinicians": ("patient", "therapy", "clinical", "opioid", "treatment"),
    "humanitarian_responders": ("humanitarian", "donors", "cyber threats", "incident", "model report"),
    "analysts": ("analysis", "forecast", "trigger", "model"),
}

EVIDENCE_STYLE_HINTS = {
    "administrative_form": (
        "personal details",
        "date of birth",
        "address",
        "service user",
    ),
    "financial_form": (
        "financial statement",
        "net worth",
        "assets",
        "liabilities",
        "cash in banks",
    ),
    "educational_exposition": ("chapter", "foreword", "learning", "concept"),
    "procedural_guidance": ("guidance", "recommended", "procedure", "should"),
    "structured_form": ("questionnaire", "checklist", "yes/no", "appendix"),
    "technical_reference": ("manual", "table", "technical", "field"),
    "evidence_review": ("review", "meta-analysis", "literature"),
    "empirical_study": ("methods", "results", "study", "evaluation"),
    "model_summary": ("model report", "trigger", "forecast"),
}

STRUCTURE_STYLE_HINTS = {
    "financial_grid": ("financial statement", "net worth", "total assets", "total liabilities"),
    "administrative_form": (
        "personal details",
        "representative details",
        "date of birth",
        "telephone",
    ),
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


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


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
    if _contains_any(signal_text, DOCUMENT_TYPE_HINTS["financial_statement"]):
        document_type = "financial_statement"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["assessment_form"]):
        document_type = "assessment_form"
    elif (
        document_type in {None, "document", "report"}
        and _contains_any(signal_text, DOCUMENT_TYPE_HINTS["administrative_form"])
    ):
        document_type = "administrative_form"
    if document_type is None:
        if page_count >= 100 and ("chapter" in signal_text or "foreword" in signal_text):
            document_type = "book"
        elif "report" in signal_text:
            document_type = "report"
        else:
            document_type = "document"

    document_purpose = _best_facet_match(signal_text, DOCUMENT_PURPOSE_HINTS)
    if document_type == "financial_statement":
        document_purpose = "financial_disclosure"
    elif document_type == "assessment_form":
        document_purpose = "financial_assessment"
    elif document_type == "administrative_form" and document_purpose in {None, "reference_lookup"}:
        document_purpose = "administrative_intake"
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
            "financial_statement": "financial_disclosure",
            "assessment_form": "financial_assessment",
            "administrative_form": "administrative_intake",
        }
        document_purpose = fallback_purpose.get(document_type, "reference_lookup")

    audience = _best_facet_match(signal_text, AUDIENCE_HINTS)
    if document_type == "financial_statement" and audience in {None, "general_professional"}:
        audience = "applicants"
    elif document_type == "assessment_form":
        audience = "case_workers" if "local authority" in signal_text else "applicants"
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
            "financial_statement": "applicants",
            "assessment_form": "case_workers",
            "administrative_form": "applicants",
        }
        audience = fallback_audience.get(document_type, "general_professional")

    evidence_style = _best_facet_match(signal_text, EVIDENCE_STYLE_HINTS)
    if document_type in {"financial_statement", "assessment_form", "administrative_form"}:
        evidence_style = "financial_form" if document_type == "financial_statement" else "administrative_form"
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
            "financial_statement": "financial_form",
            "assessment_form": "administrative_form",
            "administrative_form": "administrative_form",
        }
        evidence_style = fallback_evidence_style.get(document_type, "reference_summary")

    structure_style = _best_facet_match(signal_text, STRUCTURE_STYLE_HINTS)
    if document_type == "financial_statement":
        structure_style = "financial_grid"
    elif document_type in {"assessment_form", "administrative_form"} and structure_style in {None, "report_sections"}:
        structure_style = "administrative_form"
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
            "financial_statement": "financial_grid",
            "assessment_form": "administrative_form",
            "administrative_form": "administrative_form",
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
