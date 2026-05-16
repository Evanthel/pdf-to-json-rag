"""Shared document-family classification built on top of facets."""

from __future__ import annotations


def classify_document_family(
    *,
    document_type: str,
    document_purpose: str,
    audience: str,
    evidence_style: str,
    structure_style: str,
) -> str:
    dt = (document_type or "").strip()
    dp = (document_purpose or "").strip()
    aud = (audience or "").strip()
    ev = (evidence_style or "").strip()
    ss = (structure_style or "").strip()

    if dt == "book" or ss == "chapter_book":
        return "educational_book"
    if dt == "guidance_note" or (dp == "procedural_guidance" and aud == "humanitarian_responders"):
        return "humanitarian_guidance"
    if dt == "model_report" or dp == "risk_or_trigger_assessment":
        return "humanitarian_model_report"
    if dt == "technical_manual" or ss == "manual_reference":
        return "technical_manual"
    if ev == "structured_form" or ss in {"questionnaire_grid", "checklist_grid"}:
        return "structured_form"
    if dt == "review_article" or ev == "evidence_review":
        return "clinical_review"
    if dt == "empirical_study" or ev == "empirical_study":
        return "clinical_study"
    if aud == "clinicians":
        return "clinical_reference"
    return "general_reference"
