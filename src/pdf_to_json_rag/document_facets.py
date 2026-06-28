"""Extraction-time and metadata-time document facet derivation."""

from __future__ import annotations

import re


DOCUMENT_TYPE_HINTS = {
    "registration_form": (
        "voter registration",
        "registration transfer form",
        "transfer form",
        "registration application",
        "change of address",
    ),
    "court_opinion": (
        "court of appeals",
        "claimant-appellant",
        "appellant",
        "appellee",
        "opinion",
        "order",
        "petitioner",
        "respondent",
        "federal circuit",
    ),
    "government_bulletin": (
        "congressman",
        "senator",
        "online office",
        "newsletter",
        "public notice",
        "office of",
    ),
    "inspection_report": (
        "inspection report",
        "inspection",
        "animal welfare",
        "inspection date",
        "facility inspection",
    ),
    "agency_report": (
        "department of",
        "agency report",
        "annual report",
        "office report",
        "bureau",
    ),
    "statistical_table": (
        "census of agriculture",
        "median farm size",
        "figures taken from",
        "by county",
        "county median",
    ),
    "web_job_listing": (
        "indeed.com",
        "online job postings",
        "job postings",
        "jobs on indeed.com",
        "powered by joomla",
    ),
    "environmental_site_record": (
        "waste site id",
        "waste site reclassification",
        "current waste site condition",
        "doe project manager",
        "ecology project manager",
    ),
    "institutional_correspondence": (
        "letterhead",
        "memorandum",
        "the rockefeller university",
        "university new york",
        "new york 10021",
    ),
    "financial_statement": (
        "financial statement",
        "net worth",
        "total assets",
        "total liabilities",
        "cash in banks",
        "mortgage payable",
    ),
    "legislative_amendment": (
        "amendment to h.r.",
        "offered by",
        "page ",
        "line ",
        "insert the following",
        "after \"",
        "violator is an individual",
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
        "ship information",
        "shipping company information",
        "application procedure",
        "adopt-a-ship",
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
    "registration_update": (
        "voter registration",
        "registration transfer",
        "change of address",
        "registration application",
    ),
    "legal_record": (
        "court of appeals",
        "claimant-appellant",
        "appellant",
        "appellee",
        "petitioner",
        "respondent",
        "opinion",
        "order",
    ),
    "public_notice": (
        "newsletter",
        "public notice",
        "office of",
        "congressman",
        "senator",
        "announcement",
    ),
    "institutional_reporting": (
        "inspection report",
        "annual report",
        "agency report",
        "department of",
        "bureau",
        "animal welfare",
        "waste site id",
        "waste site reclassification",
        "current waste site condition",
    ),
    "statistical_reference": (
        "census of agriculture",
        "median farm size",
        "figures taken from",
        "by county",
    ),
    "employment_listing": (
        "indeed.com",
        "online job postings",
        "job postings",
        "jobs on indeed.com",
    ),
    "institutional_communication": (
        "letterhead",
        "memorandum",
        "the rockefeller university",
        "university new york",
    ),
    "administrative_submission": (
        "application",
        "registration",
        "submission",
        "filed",
        "signed",
    ),
    "financial_disclosure": (
        "financial statement",
        "net worth",
        "assets",
        "liabilities",
        "market value",
        "cash in banks",
    ),
    "legislative_markup": (
        "amendment to h.r.",
        "offered by",
        "line ",
        "insert the following",
        "after \"",
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
        "ship information",
        "shipping company information",
        "application procedure",
        "adopt-a-ship",
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
    "filers": ("registrant", "voter", "applicant", "petitioner", "claimant"),
    "case_workers": ("local authority", "assessor", "representative", "financial affairs"),
    "learners": ("learning", "student", "chapter", "exercise"),
    "practitioners": ("guidance", "operational", "practice", "workflow"),
    "clinicians": ("patient", "therapy", "clinical", "opioid", "treatment"),
    "humanitarian_responders": ("humanitarian", "donors", "cyber threats", "incident", "model report"),
    "analysts": ("analysis", "forecast", "trigger", "model"),
    "public_readers": ("public notice", "newsletter", "constituent", "resident", "office of"),
    "legal_professionals": ("court of appeals", "appellant", "appellee", "petitioner", "respondent"),
    "lawmakers": ("amendment to h.r.", "offered by", "line ", "insert the following"),
    "officials": ("department of", "agency", "bureau", "inspection"),
    "job_seekers": ("indeed.com", "job postings", "jobs on indeed.com"),
    "institutional_staff": ("university", "memorandum", "letterhead"),
}

EVIDENCE_STYLE_HINTS = {
    "legal_record": (
        "court of appeals",
        "appellant",
        "appellee",
        "opinion",
        "order",
    ),
    "government_notice": (
        "newsletter",
        "public notice",
        "announcement",
        "office of",
    ),
    "statistical_table": ("census of agriculture", "median farm size", "by county", "figures taken from"),
    "web_listing": ("indeed.com", "online job postings", "powered by joomla"),
    "environmental_record": ("waste site id", "waste site reclassification", "project manager signature"),
    "institutional_correspondence": ("letterhead", "memorandum", "university"),
    "administrative_form": (
        "personal details",
        "date of birth",
        "address",
        "service user",
        "ship information",
        "shipping company information",
        "application procedure",
    ),
    "financial_form": (
        "financial statement",
        "net worth",
        "assets",
        "liabilities",
        "cash in banks",
    ),
    "legislative_markup": ("amendment to h.r.", "offered by", "line ", "insert the following"),
    "educational_exposition": ("chapter", "foreword", "learning", "concept"),
    "procedural_guidance": ("guidance", "recommended", "procedure", "should"),
    "structured_form": ("questionnaire", "checklist", "yes/no", "appendix"),
    "technical_reference": ("manual", "table", "technical", "field"),
    "evidence_review": ("review", "meta-analysis", "literature"),
    "empirical_study": ("methods", "results", "study", "evaluation"),
    "model_summary": ("model report", "trigger", "forecast"),
}

STRUCTURE_STYLE_HINTS = {
    "legal_opinion": (
        "court of appeals",
        "appellant",
        "appellee",
        "opinion",
        "order",
    ),
    "government_notice": (
        "newsletter",
        "public notice",
        "office of",
        "announcement",
    ),
    "data_table": ("census of agriculture", "median farm size", "by county"),
    "web_page_printout": ("powered by joomla", "generated:", "indeed.com"),
    "structured_site_record": ("waste site id", "waste site reclassification", "project manager signature"),
    "letterhead": ("letterhead", "university", "memorandum"),
    "financial_grid": ("financial statement", "net worth", "total assets", "total liabilities"),
    "legislative_markup": ("amendment to h.r.", "offered by", "line ", "insert the following"),
    "administrative_form": (
        "personal details",
        "representative details",
        "date of birth",
        "telephone",
        "ship information",
        "shipping company information",
        "application procedure",
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


def _looks_like_legislative_amendment(text: str) -> bool:
    if "amendment to h.r." in text or "amendment to h. r." in text:
        return True
    line_directive = bool(re.search(r"\bpage\s+\d+,\s+line\s+\d+\b", text))
    insertion_directive = "insert the following" in text or "after \"" in text
    sponsor_marker = "offered by" in text
    chamber_marker = " of illinois" in text or " of ill:rnois" in text
    return line_directive and insertion_directive and (sponsor_marker or chamber_marker)


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.62:
        return "moderate"
    return "low"


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
    if _looks_like_legislative_amendment(signal_text):
        document_type = "legislative_amendment"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["financial_statement"]):
        document_type = "financial_statement"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["assessment_form"]):
        document_type = "assessment_form"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["registration_form"]):
        document_type = "registration_form"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["court_opinion"]):
        document_type = "court_opinion"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["government_bulletin"]):
        document_type = "government_bulletin"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["inspection_report"]):
        document_type = "inspection_report"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["environmental_site_record"]):
        document_type = "environmental_site_record"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["statistical_table"]):
        document_type = "statistical_table"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["web_job_listing"]):
        document_type = "web_job_listing"
    elif _contains_any(signal_text, DOCUMENT_TYPE_HINTS["agency_report"]):
        document_type = "agency_report"
    elif (
        document_type in {None, "document", "report"}
        and _contains_any(signal_text, DOCUMENT_TYPE_HINTS["administrative_form"])
    ):
        document_type = "administrative_form"
    elif (
        document_type in {None, "document", "report"}
        and page_count <= 2
        and _contains_any(signal_text, DOCUMENT_TYPE_HINTS["institutional_correspondence"])
    ):
        document_type = "institutional_correspondence"
    if document_type is None:
        if page_count >= 100 and ("chapter" in signal_text or "foreword" in signal_text):
            document_type = "book"
        elif "report" in signal_text:
            document_type = "report"
        else:
            document_type = "document"

    document_purpose = _best_facet_match(signal_text, DOCUMENT_PURPOSE_HINTS)
    if document_type == "legislative_amendment":
        document_purpose = "legislative_markup"
    elif document_type == "financial_statement":
        document_purpose = "financial_disclosure"
    elif document_type == "assessment_form":
        document_purpose = "financial_assessment"
    elif document_type == "registration_form":
        document_purpose = "registration_update"
    elif document_type == "court_opinion":
        document_purpose = "legal_record"
    elif document_type == "government_bulletin" and document_purpose in {None, "reference_lookup"}:
        document_purpose = "public_notice"
    elif document_type in {"inspection_report", "agency_report"} and document_purpose in {None, "reference_lookup"}:
        document_purpose = "institutional_reporting"
    elif document_type == "environmental_site_record" and document_purpose in {None, "reference_lookup"}:
        document_purpose = "institutional_reporting"
    elif document_type == "statistical_table" and document_purpose in {None, "reference_lookup"}:
        document_purpose = "statistical_reference"
    elif document_type == "web_job_listing" and document_purpose in {None, "reference_lookup"}:
        document_purpose = "employment_listing"
    elif document_type == "institutional_correspondence" and document_purpose in {None, "reference_lookup"}:
        document_purpose = "institutional_communication"
    elif document_type == "administrative_form" and document_purpose in {None, "reference_lookup"}:
        document_purpose = "administrative_intake"
    if document_purpose is None:
        fallback_purpose = {
            "registration_form": "registration_update",
            "court_opinion": "legal_record",
            "government_bulletin": "public_notice",
            "inspection_report": "institutional_reporting",
            "agency_report": "institutional_reporting",
            "environmental_site_record": "institutional_reporting",
            "statistical_table": "statistical_reference",
            "web_job_listing": "employment_listing",
            "institutional_correspondence": "institutional_communication",
            "book": "teaching_reference",
            "guidance_note": "procedural_guidance",
            "questionnaire": "structured_data_capture",
            "checklist_appendix": "operational_checklist",
            "model_report": "risk_or_trigger_assessment",
            "review_article": "evidence_summary",
            "empirical_study": "empirical_reporting",
            "technical_manual": "procedural_guidance",
            "financial_statement": "financial_disclosure",
            "legislative_amendment": "legislative_markup",
            "assessment_form": "financial_assessment",
            "administrative_form": "administrative_intake",
        }
        document_purpose = fallback_purpose.get(document_type, "reference_lookup")

    audience = _best_facet_match(signal_text, AUDIENCE_HINTS)
    if document_type == "financial_statement" and audience in {None, "general_professional"}:
        audience = "applicants"
    elif document_type == "legislative_amendment" and audience in {None, "general_professional"}:
        audience = "lawmakers"
    elif document_type == "assessment_form":
        audience = "case_workers" if "local authority" in signal_text else "applicants"
    elif document_type == "registration_form" and audience in {None, "general_professional"}:
        audience = "filers"
    elif document_type == "court_opinion" and audience in {None, "general_professional"}:
        audience = "legal_professionals"
    elif document_type == "government_bulletin" and audience in {None, "general_professional"}:
        audience = "public_readers"
    elif document_type in {"inspection_report", "agency_report"} and audience in {None, "general_professional"}:
        audience = "officials"
    elif document_type == "environmental_site_record" and audience in {None, "general_professional"}:
        audience = "officials"
    elif document_type == "statistical_table" and audience in {None, "general_professional"}:
        audience = "analysts"
    elif document_type == "web_job_listing" and audience in {None, "general_professional"}:
        audience = "job_seekers"
    elif document_type == "institutional_correspondence" and audience in {None, "general_professional"}:
        audience = "institutional_staff"
    if audience is None:
        fallback_audience = {
            "registration_form": "filers",
            "court_opinion": "legal_professionals",
            "government_bulletin": "public_readers",
            "inspection_report": "officials",
            "agency_report": "officials",
            "environmental_site_record": "officials",
            "statistical_table": "analysts",
            "web_job_listing": "job_seekers",
            "institutional_correspondence": "institutional_staff",
            "book": "learners",
            "guidance_note": "practitioners",
            "questionnaire": "practitioners",
            "checklist_appendix": "practitioners",
            "model_report": "analysts",
            "review_article": "clinicians",
            "empirical_study": "clinicians",
            "technical_manual": "practitioners",
            "financial_statement": "applicants",
            "legislative_amendment": "lawmakers",
            "assessment_form": "case_workers",
            "administrative_form": "applicants",
        }
        audience = fallback_audience.get(document_type, "general_professional")

    evidence_style = _best_facet_match(signal_text, EVIDENCE_STYLE_HINTS)
    if document_type == "legislative_amendment":
        evidence_style = "legislative_markup"
    elif document_type in {"financial_statement", "assessment_form", "administrative_form", "registration_form"}:
        evidence_style = "financial_form" if document_type == "financial_statement" else "administrative_form"
    elif document_type == "court_opinion":
        evidence_style = "legal_record"
    elif document_type in {"government_bulletin", "inspection_report", "agency_report"}:
        evidence_style = "government_notice"
    elif document_type == "environmental_site_record":
        evidence_style = "environmental_record"
    elif document_type == "statistical_table":
        evidence_style = "statistical_table"
    elif document_type == "web_job_listing":
        evidence_style = "web_listing"
    elif document_type == "institutional_correspondence":
        evidence_style = "institutional_correspondence"
    if evidence_style is None:
        fallback_evidence_style = {
            "registration_form": "administrative_form",
            "court_opinion": "legal_record",
            "government_bulletin": "government_notice",
            "inspection_report": "government_notice",
            "agency_report": "government_notice",
            "environmental_site_record": "environmental_record",
            "statistical_table": "statistical_table",
            "web_job_listing": "web_listing",
            "institutional_correspondence": "institutional_correspondence",
            "book": "educational_exposition",
            "guidance_note": "procedural_guidance",
            "questionnaire": "structured_form",
            "checklist_appendix": "structured_form",
            "model_report": "model_summary",
            "review_article": "evidence_review",
            "empirical_study": "empirical_study",
            "technical_manual": "technical_reference",
            "financial_statement": "financial_form",
            "legislative_amendment": "legislative_markup",
            "assessment_form": "administrative_form",
            "administrative_form": "administrative_form",
        }
        evidence_style = fallback_evidence_style.get(document_type, "reference_summary")

    structure_style = _best_facet_match(signal_text, STRUCTURE_STYLE_HINTS)
    if document_type == "financial_statement":
        structure_style = "financial_grid"
    elif document_type == "legislative_amendment":
        structure_style = "legislative_markup"
    elif document_type in {"assessment_form", "administrative_form", "registration_form"} and structure_style in {None, "report_sections"}:
        structure_style = "administrative_form"
    elif document_type == "court_opinion" and structure_style in {None, "report_sections"}:
        structure_style = "legal_opinion"
    elif document_type in {"government_bulletin", "inspection_report", "agency_report"} and structure_style in {None, "report_sections"}:
        structure_style = "government_notice"
    elif document_type == "environmental_site_record" and structure_style in {None, "report_sections"}:
        structure_style = "structured_site_record"
    elif document_type == "statistical_table" and structure_style in {None, "report_sections"}:
        structure_style = "data_table"
    elif document_type == "web_job_listing" and structure_style in {None, "report_sections"}:
        structure_style = "web_page_printout"
    elif document_type == "institutional_correspondence" and structure_style in {None, "report_sections"}:
        structure_style = "letterhead"
    if structure_style is None:
        fallback_structure_style = {
            "registration_form": "administrative_form",
            "court_opinion": "legal_opinion",
            "government_bulletin": "government_notice",
            "inspection_report": "government_notice",
            "agency_report": "government_notice",
            "environmental_site_record": "structured_site_record",
            "statistical_table": "data_table",
            "web_job_listing": "web_page_printout",
            "institutional_correspondence": "letterhead",
            "book": "chapter_book",
            "guidance_note": "report_sections",
            "questionnaire": "questionnaire_grid",
            "checklist_appendix": "checklist_grid",
            "model_report": "report_sections",
            "review_article": "review_article",
            "empirical_study": "review_article",
            "technical_manual": "manual_reference",
            "financial_statement": "financial_grid",
            "legislative_amendment": "legislative_markup",
            "assessment_form": "administrative_form",
            "administrative_form": "administrative_form",
        }
        structure_style = fallback_structure_style.get(document_type, "report_sections")

    type_matches = _count_hint_matches(signal_text, DOCUMENT_TYPE_HINTS.get(document_type, ()))
    purpose_matches = _count_hint_matches(signal_text, DOCUMENT_PURPOSE_HINTS.get(document_purpose, ()))
    audience_matches = _count_hint_matches(signal_text, AUDIENCE_HINTS.get(audience, ()))
    evidence_matches = _count_hint_matches(signal_text, EVIDENCE_STYLE_HINTS.get(evidence_style, ()))
    structure_matches = _count_hint_matches(signal_text, STRUCTURE_STYLE_HINTS.get(structure_style, ()))

    semantic_confidence = 0.32
    semantic_confidence += min(type_matches, 3) * 0.12
    semantic_confidence += min(purpose_matches, 3) * 0.1
    semantic_confidence += min(audience_matches, 2) * 0.08
    semantic_confidence += min(evidence_matches, 2) * 0.06
    semantic_confidence += min(structure_matches, 2) * 0.05
    if title and title.strip():
        semantic_confidence += 0.06
    if summary_cues:
        semantic_confidence += 0.05
    if toc:
        semantic_confidence += 0.04
    if metadata_values:
        semantic_confidence += 0.04
    semantic_confidence = round(min(0.95, semantic_confidence), 3)

    semantic_rationale: list[str] = []
    if type_matches:
        semantic_rationale.append("explicit_document_type_cues")
    if purpose_matches:
        semantic_rationale.append("explicit_document_purpose_cues")
    if audience_matches:
        semantic_rationale.append("explicit_audience_cues")
    if evidence_matches:
        semantic_rationale.append("evidence_style_cues")
    if structure_matches:
        semantic_rationale.append("structure_style_cues")
    if summary_cues or toc:
        semantic_rationale.append("section_or_toc_support")
    if metadata_values:
        semantic_rationale.append("metadata_support")

    semantic_warnings: list[str] = []
    if document_type == "document":
        semantic_warnings.append("generic_document_type")
    if document_purpose == "reference_lookup":
        semantic_warnings.append("generic_document_purpose")
    if audience == "general_professional":
        semantic_warnings.append("generic_audience")
    if document_type in {
        "court_opinion",
        "legislative_amendment",
        "registration_form",
        "government_bulletin",
        "inspection_report",
        "agency_report",
        "environmental_site_record",
        "statistical_table",
        "web_job_listing",
        "institutional_correspondence",
    }:
        semantic_warnings = [warning for warning in semantic_warnings if warning != "generic_document_type"]
    if document_purpose in {
        "legal_record",
        "legislative_markup",
        "registration_update",
        "public_notice",
        "institutional_reporting",
        "administrative_submission",
        "statistical_reference",
        "employment_listing",
        "institutional_communication",
    }:
        semantic_warnings = [warning for warning in semantic_warnings if warning != "generic_document_purpose"]
    if type_matches == 0 and purpose_matches == 0:
        semantic_warnings.append("limited_explicit_semantic_cues")
    if not summary_cues and not toc:
        semantic_warnings.append("limited_structural_semantic_support")

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
        "semantic_confidence": semantic_confidence,
        "semantic_confidence_label": _confidence_label(semantic_confidence),
        "semantic_rationale": semantic_rationale,
        "semantic_warnings": semantic_warnings,
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
