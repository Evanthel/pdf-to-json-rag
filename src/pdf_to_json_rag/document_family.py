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
    if dt in {"financial_statement", "assessment_form", "administrative_form", "registration_form"} or dp in {
        "financial_disclosure",
        "financial_assessment",
        "administrative_intake",
        "registration_update",
    } or ss in {"financial_grid", "administrative_form"}:
        return "administrative_financial_form"
    if dt in {
        "court_opinion",
        "government_bulletin",
        "inspection_report",
        "agency_report",
        "environmental_site_record",
        "statistical_table",
        "web_job_listing",
        "institutional_correspondence",
    } or dp in {
        "legal_record",
        "public_notice",
        "institutional_reporting",
        "administrative_submission",
        "statistical_reference",
        "employment_listing",
        "institutional_communication",
    } or ev in {
        "legal_record",
        "government_notice",
        "statistical_table",
        "web_listing",
        "environmental_record",
        "institutional_correspondence",
    } or ss in {
        "legal_opinion",
        "government_notice",
        "data_table",
        "web_page_printout",
        "structured_site_record",
        "letterhead",
    }:
        return "government_public_record"
    if ev == "structured_form" or ss in {"questionnaire_grid", "checklist_grid"}:
        return "structured_form"
    if dt == "review_article" or ev == "evidence_review":
        return "clinical_review"
    if dt == "empirical_study" or ev == "empirical_study":
        return "clinical_study"
    if aud == "clinicians":
        return "clinical_reference"
    return "general_reference"
