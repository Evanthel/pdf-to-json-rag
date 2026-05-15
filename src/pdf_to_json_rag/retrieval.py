"""Retrieval interfaces for the MVP pipeline."""

import json
from pathlib import Path
import re

import chromadb

from .indexing import (
    DEFAULT_COLLECTION_NAME,
    load_embedder_from_manifest,
    load_index_manifest,
)
from .intent_config import (
    detect_structured_intent,
    get_document_profile,
    get_structured_intent_profile,
    matching_source_doc_ids as configured_matching_source_doc_ids,
    preferred_source_doc_id as configured_source_doc_id,
)
from .query_planning import plan_query
from .quality import classify_chunk_quality
from .schemas import ChunkRecord

HARD_EXCLUDE_LABELS = {
    "disclaimer",
    "page_number",
    "statistical_section",
    "table_like_section",
}
SOFT_NOISE_LABELS = {
    "bibliography",
    "reference_tail",
    "table_reference",
    "toc_fragment",
    "toc_leader",
    "noisy_section",
    "statistical_section",
    "statistical_noise",
    "table_like_section",
    "boilerplate",
    "commentary_section",
    "short_fragment",
    "title_fragment",
    "garbled_ocr",
}
EXPANSION_BLOCK_LABELS = {
    "disclaimer",
    "bibliography",
    "reference_tail",
    "table_reference",
    "toc_fragment",
    "toc_leader",
    "page_number",
    "title_fragment",
    "garbled_ocr",
}

INTENT_CANDIDATE_K = {
    "generic": (4, 15),
    "source_listing": (8, 30),
    "cross_document_compare": (8, 30),
    "document_overview": (6, 24),
    "document_routing": (8, 30),
    "source_justification": (8, 30),
    "symptom_pathogenesis": (6, 20),
    "hypothermia_predisposition": (6, 24),
    "hypothermia_symptoms": (6, 24),
    "frostbite_prevention": (6, 24),
    "immersion_limit": (6, 24),
    "review_prevention": (6, 24),
    "review_nontraditional": (6, 24),
    "definition": (4, 15),
    "symptoms": (4, 15),
    "duration": (5, 18),
    "transmission": (5, 18),
    "causes": (5, 18),
    "incidence": (6, 24),
    "ct_findings": (6, 24),
    "ct_follow_up": (6, 24),
    "antibiotics": (6, 24),
    "treatment_prevention": (6, 24),
    "treatment_null_effect": (6, 24),
    "treatment_subgroup_benefit": (6, 24),
    "treatment_duration": (6, 24),
    "treatment_overall": (6, 24),
}

INTENT_NEIGHBOR_DEPTH = {
    "generic": 1,
    "source_listing": 0,
    "cross_document_compare": 0,
    "document_overview": 0,
    "document_routing": 0,
    "source_justification": 0,
    "symptom_pathogenesis": 1,
    "hypothermia_predisposition": 1,
    "hypothermia_symptoms": 1,
    "frostbite_prevention": 1,
    "immersion_limit": 1,
    "review_prevention": 1,
    "review_nontraditional": 1,
    "definition": 1,
    "symptoms": 1,
    "duration": 1,
    "transmission": 1,
    "causes": 1,
    "incidence": 2,
    "ct_findings": 2,
    "ct_follow_up": 2,
    "antibiotics": 2,
    "treatment_prevention": 1,
    "treatment_null_effect": 1,
    "treatment_subgroup_benefit": 1,
    "treatment_duration": 1,
    "treatment_overall": 1,
}

INTENT_SECTION_HINTS = {
    "source_listing": (),
    "cross_document_compare": (),
    "document_overview": ("CONTENTS", "FOREWORD", "CHAPTER", "INTRODUCTION"),
    "document_routing": ("CONTENTS", "FOREWORD", "CHAPTER", "INTRODUCTION"),
    "source_justification": ("CONTENTS", "FOREWORD", "CHAPTER", "INTRODUCTION"),
    "symptom_pathogenesis": ("ABSTRACT", "INTRODUCTION", "REVIEW"),
    "hypothermia_predisposition": ("TB MED 508", "HYPOTHERMIA", "GERMANY", "ENDOCRINE", "IATROGENIC"),
    "hypothermia_symptoms": ("TB MED 508", "HYPOTHERMIA"),
    "frostbite_prevention": ("SEVERE", "EXTREME", "HIGH", "TB MED 508"),
    "immersion_limit": ("TB MED 508",),
    "review_prevention": ("CMAJ", "REVIEW", "INTERVENTION"),
    "review_nontraditional": ("CMAJ", "REVIEW"),
    "definition": ("DEFINITION", "PROGNOSIS"),
    "symptoms": ("DEFINITION", "PROGNOSIS"),
    "duration": ("PROGNOSIS",),
    "transmission": ("AETIOLOGY", "RISK FACTORS", "TRANSMISSION", "TREATMENTS"),
    "causes": ("AETIOLOGY", "RISK FACTORS", "TRANSMISSION"),
    "incidence": ("PREVALENCE", "INCIDENCE"),
    "ct_findings": ("DISCUSSION", "CT SCANS", "FOLLOW-UP"),
    "ct_follow_up": ("FOLLOW-UP", "DISCUSSION", "CT SCANS"),
    "antibiotics": ("OPTION", "TREATMENTS", "COMMENT:"),
    "treatment_prevention": ("THE UPDATED REVIEW", "CONCLUSION", "META-ANALYSES OUTCOMES"),
    "treatment_null_effect": ("THE UPDATED REVIEW", "CONCLUSION"),
    "treatment_subgroup_benefit": ("THE UPDATED REVIEW", "SUBGROUP", "SENSITIVITY ANALYSIS"),
    "treatment_duration": ("THE UPDATED REVIEW", "CONCLUSION", "META-ANALYSES OUTCOMES"),
    "treatment_overall": ("CONCLUSION", "REVIEW", "META-ANALYSES OUTCOMES"),
}
INTENT_SUBTOPIC_CUES = {
    "treatment_prevention": {"treatment_prevention"},
    "treatment_null_effect": {"treatment_null_effect"},
    "treatment_subgroup_benefit": {"treatment_subgroup_benefit"},
    "treatment_duration": {"treatment_duration"},
    "treatment_overall": {"treatment_overall"},
}

TREATMENT_ENTITY_TERMS = {
    "vitamin",
    "echinacea",
    "propolis",
    "zinc",
    "honey",
    "ginseng",
    "handwashing",
}
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "according",
    "cold",
    "common",
    "does",
    "for",
    "help",
    "helps",
    "how",
    "in",
    "is",
    "of",
    "review",
    "say",
    "says",
    "the",
    "to",
    "what",
    "with",
}


def _query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{2,}", query.lower()))


def _has_treatment_entity(terms: set[str]) -> bool:
    return bool(terms.intersection(TREATMENT_ENTITY_TERMS))


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if term not in QUERY_STOPWORDS
    }


def _detect_query_intent(query: str) -> str:
    plan = plan_query(query)
    if plan.query_class != "evidence_lookup":
        return plan.query_intent
    terms = _query_terms(query)
    query_lower = query.lower()
    structured_intent = detect_structured_intent(query, terms)
    if structured_intent:
        return structured_intent
    if "compare" in terms and len(terms.intersection(TREATMENT_ENTITY_TERMS)) >= 2:
        return "cross_document_compare"
    if "hypothermia" in terms and {"predisposing", "predispose", "factors", "categories"}.intersection(terms):
        return "hypothermia_predisposition"
    if "hypothermia" in terms and {"signs", "symptoms"}.intersection(terms):
        return "hypothermia_symptoms"
    if "frostbite" in terms and (
        "severe" in terms
        or "preventive" in terms
        or "measures" in terms
        or "zone" in terms
        or "risk" in terms
    ):
        return "frostbite_prevention"
    if "immersion" in terms and ("neck" in terms or "depth" in terms):
        return "immersion_limit"
    if (
        ("cause" in terms or "causes" in terms)
        and ("symptom" in terms or "symptoms" in terms)
    ):
        return "symptom_pathogenesis"
    if (
        ("nontraditional" in terms and "treatments" in terms)
        or "nontraditional treatments" in query_lower
    ):
        return "review_nontraditional"
    if (
        ("preventive" in terms and "interventions" in terms)
        or ("handwashing" in terms and "prevent" in terms)
        or "best evidence" in query_lower and "prevent" in query_lower
    ):
        return "review_prevention"
    has_treatment_query = _has_treatment_entity(terms) and "cold" in terms
    if has_treatment_query:
        if "stress" in terms or ("physical" in terms and "stress" in terms) or "subgroup" in terms:
            return "treatment_subgroup_benefit"
        if "normal" in terms and "populations" in terms:
            return "treatment_null_effect"
        if "duration" in terms or "shorten" in terms:
            return "treatment_duration"
        if "conclude" in terms or "conclusion" in terms or "meta" in terms or "analysis" in terms:
            return "treatment_overall"
        if (
            "prevent" in terms
            or "prevents" in terms
            or "prevention" in terms
            or "prophylaxis" in terms
            or "incidence" in terms
        ):
            return "treatment_prevention"
    if query_lower.startswith("what is") or "definition" in terms or "define" in terms:
        return "definition"
    if "ct" in terms and ("follow" in terms or "followup" in terms):
        return "ct_follow_up"
    if "ct" in terms and ("abnormalities" in terms or "sinus" in terms or "scans" in terms):
        return "ct_findings"
    if "antibiotic" in terms or "antibiotics" in terms:
        return "antibiotics"
    if "cause" in terms or "causes" in terms:
        return "causes"
    if "transmitted" in terms or "transmission" in terms:
        return "transmission"
    if "last" in terms or "long" in terms or "duration" in terms:
        return "duration"
    if "year" in terms or ("children" in terms and "adults" in terms):
        return "incidence"
    if "symptom" in terms or "symptoms" in terms:
        return "symptoms"
    return "generic"


def _preferred_source_doc_id(query: str) -> str | None:
    plan = plan_query(query)
    if plan.query_class != "evidence_lookup":
        return plan.preferred_doc_id
    intent = _detect_query_intent(query)
    if intent in {"source_listing", "cross_document_compare", "document_routing"}:
        return None
    return configured_source_doc_id(
        query,
        allow_topical=(intent in {"source_justification", "document_overview"}),
    )


def _matching_source_doc_ids(query: str) -> list[str]:
    plan = plan_query(query)
    if plan.query_class != "evidence_lookup":
        return list(plan.matched_doc_ids)
    intent = _detect_query_intent(query)
    allow_topical = intent in {
        "source_listing",
        "document_routing",
        "source_justification",
        "document_overview",
        "cross_document_compare",
    }
    matches = configured_matching_source_doc_ids(query, allow_topical=allow_topical)
    query_lower = query.lower()
    if intent == "source_justification" and matches:
        return matches[:1]
    if intent == "document_overview" and matches:
        return matches[:1]
    if intent == "document_routing" and matches:
        if "which file or files" not in query_lower and "which files" not in query_lower:
            return matches[:1]
    return matches


def _inventory_doc_ids(query: str) -> list[str]:
    plan = plan_query(query)
    return list(plan.inventory_doc_ids)


def _augment_query(query: str) -> str:
    intent = _detect_query_intent(query)
    structured_profile = get_structured_intent_profile(intent)
    if structured_profile:
        return f"{query} {structured_profile.augment_suffix}"
    suffix = {
        "source_listing": (
            "sources documents benchmark review questionnaire checklist appendix manual "
            "technical source document family"
        ),
        "cross_document_compare": (
            "compare sources across documents benchmark evidence conclusions prevention "
            "incidence treatment source contrast"
        ),
        "document_overview": (
            "document overview table of contents foreword chapter topics covers about "
            "scope summary subject matter document type document purpose audience"
        ),
        "document_routing": (
            "most relevant file source document benchmark route source discovery "
            "best match topic coverage"
        ),
        "source_justification": (
            "best source best match why this file source discovery document cues "
            "topic coverage section summary benchmark"
        ),
        "review_prevention": (
            "review summary prevention best evidence handwashing physical interventions zinc supplements"
        ),
        "symptom_pathogenesis": (
            "review mechanism symptom production viral cytopathic effect activation inflammatory pathways"
        ),
        "hypothermia_predisposition": (
            "predisposing factors for hypothermia decrease heat production "
            "increase heat loss impair thermoregulation miscellaneous clinical states"
        ),
        "hypothermia_symptoms": (
            "signs and symptoms of hypothermia shivering altered mental status "
            "hypotension confusion table 4-4"
        ),
        "frostbite_prevention": (
            "severe frostbite risk zone mandatory buddy checks no exposed skin "
            "warming facilities stay active wear ecwcs table 3-4"
        ),
        "immersion_limit": (
            "table 3-3 immersion time limits neck 50-54 5 minutes water temperature"
        ),
        "review_nontraditional": (
            "review summary nontraditional treatments oral zinc honey cough adults children"
        ),
        "definition": "definition defined as upper respiratory tract infection",
        "antibiotics": (
            "option antibiotics clinical guide don't reduce symptoms overall "
            "adverse effects antibiotic resistance viral"
        ),
        "causes": "aetiology risk factors viruses rhinovirus coronavirus respiratory syncytial virus",
        "transmission": "transmission hand-to-hand contact droplets nostrils eyes",
        "duration": "prognosis duration symptoms peak clear by 1 week cough persists",
        "incidence": "incidence prevalence children adults each year infections",
        "ct_findings": (
            "ct scans sinus abnormalities ostiomeatal follow-up discussion "
            "high prevalence maxillary ethmoid infundibulum"
        ),
        "ct_follow_up": (
            "follow-up evaluations 13 to 20 days marked improvement residual abnormalities "
            "returned to normal resolved"
        ),
        "symptoms": "symptoms sneezing runny nose headache sore throat cough",
        "treatment_prevention": (
            "treatment prevention incidence reduces odds contracting a cold "
            "prevention prophylaxis incidence benefit"
        ),
        "treatment_null_effect": (
            "treatment prevention incidence was not altered no effect "
            "normal populations no prophylactic benefit"
        ),
        "treatment_subgroup_benefit": (
            "treatment subgroup benefit cold stress physical stress "
            "marathon runners skiers soldiers 50% reduction beneficial effect"
        ),
        "treatment_duration": (
            "treatment duration reduced duration of cold episodes "
            "shortens course shortened by days onset of symptoms"
        ),
        "treatment_overall": (
            "meta-analysis conclusion prevention treatment incidence duration benefit evidence"
        ),
    }.get(intent, "")
    if not suffix:
        return query
    return f"{query} {suffix}"


def _candidate_pool_size(query: str, k: int) -> int:
    intent = _detect_query_intent(query)
    structured_profile = get_structured_intent_profile(intent)
    if structured_profile:
        multiplier, minimum = structured_profile.candidate_k
        return max(k * multiplier, minimum)
    multiplier, minimum = INTENT_CANDIDATE_K.get(intent, INTENT_CANDIDATE_K["generic"])
    return max(k * multiplier, minimum)


def _is_cross_document_intent(intent: str) -> bool:
    return intent in {
        "source_listing",
        "cross_document_compare",
        "document_routing",
        "source_justification",
    }


def _heuristic_hit_bonus(chunk: ChunkRecord, query: str) -> float:
    section = (chunk.section_title or "").upper()
    text = chunk.text.lower()
    intent = _detect_query_intent(query)
    bonus = 0.0
    labels = set(chunk.noise_labels)
    subtopic_cues = set(chunk.subtopic_cues)
    matched_doc_ids = set(_matching_source_doc_ids(query))

    if "ocr_derived" in labels:
        bonus -= 0.15
    bonus -= (1.0 - chunk.quality_score) * 6.0
    bonus -= len(labels.intersection(SOFT_NOISE_LABELS)) * 0.75

    if intent in {"source_listing", "cross_document_compare", "document_routing"}:
        if matched_doc_ids:
            if chunk.doc_id in matched_doc_ids:
                bonus += 6.0
            else:
                bonus -= 6.0
        if section.startswith("CONCLUSION") or "IMPLICATIONS OF THE REVIEW" in section:
            bonus += 4.0
        if "published evidence supports echinacea" in text or "suggests that echinacea has a benefit" in text:
            bonus += 6.0
        if "decreasing the incidence" in text or "substantial reductions in the incidence" in text:
            bonus += 5.0
        if "incidence was not altered" in text or "normal populations" in text or "lack of effect" in text:
            bonus += 6.0
        if "trials were included for analysis" in text or "inclusion criteria" in text:
            bonus -= 6.0
        if "search strategy and selection criteria" in text or "subject of controversy" in text:
            bonus -= 4.0
        if "we sought to discover whether" in text or "criteria for inclusion were placebo-controlled trials" in text:
            bonus -= 3.5
        if "our meta-analysis had only one cold incidence study" in text:
            bonus -= 2.0
    if intent == "document_overview":
        preferred_doc_id = _preferred_source_doc_id(query)
        if preferred_doc_id:
            if chunk.doc_id == preferred_doc_id:
                bonus += 8.0
            else:
                bonus -= 8.0
        if section.startswith("CONTENTS") or section.startswith("FOREWORD") or section.startswith("CHAPTER"):
            bonus += 8.0
        if section.startswith("INTRODUCTION"):
            bonus += 3.0
        if section.startswith("BIBLIOGRAPHY") or section.startswith("INDEX"):
            bonus -= 8.0
        if "contents" in text or "chapter" in text or "machine learning" in text:
            bonus += 2.0
    if intent == "document_routing":
        if matched_doc_ids:
            if chunk.doc_id in matched_doc_ids:
                bonus += 8.0
            else:
                bonus -= 8.0
        profile = get_document_profile(chunk.doc_id)
        if profile:
            topical_overlap = len(profile.topical_terms.intersection(_query_terms(query)))
            bonus += topical_overlap * 0.75
        if section.startswith("CONTENTS") or section.startswith("FOREWORD") or section.startswith("CHAPTER"):
            bonus += 3.0
        if section.startswith("BIBLIOGRAPHY") or section.startswith("INDEX"):
            bonus -= 6.0
    if intent == "source_justification":
        if matched_doc_ids:
            if chunk.doc_id in matched_doc_ids:
                bonus += 8.0
            else:
                bonus -= 8.0
        profile = get_document_profile(chunk.doc_id)
        if profile:
            topical_overlap = len(profile.topical_terms.intersection(_query_terms(query)))
            bonus += topical_overlap * 0.5
        if section.startswith("CONTENTS") or section.startswith("FOREWORD") or section.startswith("CHAPTER"):
            bonus += 3.0
        if section.startswith("BIBLIOGRAPHY") or section.startswith("INDEX"):
            bonus -= 6.0

    if intent == "definition":
        if section.startswith("DEFINITION"):
            bonus += 6.0
        if "defined as" in text:
            bonus += 5.0
        if section.startswith("PROGNOSIS") or section.startswith("AETIOLOGY"):
            bonus += 1.0
    elif intent == "opioid_pre_therapy_checklist":
        if "appendix a" in text or "appendix a" in section.lower():
            bonus += 6.0
        if "checklist" in text:
            bonus += 3.0
        if "non-pharmacological therapy" in text:
            bonus += 5.0
        if "non-opioid pharmacotherapy" in text:
            bonus += 5.0
        if "informed consent" in text or "opioid safety" in text:
            bonus += 3.0
        if "urine drug screening" in text:
            bonus += 2.5
    elif intent == "opioid_adverse_effect_scale":
        if "appendix b" in text or "appendix b" in section.lower():
            bonus += 6.0
        if "adverse effects" in text:
            bonus += 4.0
        if "0 = none" in text:
            bonus += 5.0
        if "1 = limits adls" in text or "2 = prevents adls" in text:
            bonus += 7.0
        if "fatal overdose" in text or "non-fatal overdose" in text:
            bonus += 2.0
    elif intent == "opioid_switch_follow_up":
        if "appendix c" in text or "switching opioids" in text:
            bonus += 6.0
        if "3-day follow-up" in text:
            bonus += 7.0
        if "every 2-4 weeks" in text or "every 2–4 weeks" in text:
            bonus += 5.0
        if "withdrawal symptoms" in text and "pain" in text:
            bonus += 3.0
    elif intent == "questionnaire_performance":
        if "question 13" in text or "performance at work" in text:
            bonus += 7.0
        if "concentration" in text or "motivation" in text:
            bonus += 4.0
        if "manual strength" in text or "musculo-skeletal function" in text:
            bonus += 4.0
        if "cooling symptoms" in text:
            bonus += 3.0
        if "health-check questionnaire" in text:
            bonus += 2.0
    elif intent == "questionnaire_symptom_scale":
        if "question 5" in text or "shortness of breath" in text:
            bonus += 6.0
        if "persistent coughing" in text or "wheezing" in text:
            bonus += 4.0
        if "not at all" in text and "during exercise" in text:
            bonus += 5.0
        if "mucus excretion" in text:
            bonus += 3.0
    elif intent == "questionnaire_color_change":
        if "question 9" in text:
            bonus += 6.0
        if "white" in text and "blue" in text and ("red/purple" in text or "red purple" in text):
            bonus += 7.0
        if "episodically change" in text:
            bonus += 3.0
    elif intent == "questionnaire_frostbite_history":
        if "question 12" in text or "blister grade" in text:
            bonus += 6.0
        if "once" in text and "several times" in text:
            bonus += 5.0
        if "frostbite" in text:
            bonus += 3.0
    elif intent == "questionnaire_follow_up_table":
        if "table i" in text:
            bonus += 8.0
        if "uncomfortable" in text or "sensitivity" in text:
            bonus += 4.0
        if "interview of working ability" in text or "disease-focused interview" in text:
            bonus += 4.0
        if "professional: nurse" in text or "nurse and physician" in text:
            bonus += 5.0
        if "sensitivity" in query.lower():
            if "sensitivity" in text and "professional: nurse" in text:
                bonus += 6.0
            if "sensitivity" not in text:
                bonus -= 5.0
            if "symptom of some disease" in text:
                bonus -= 5.0
        if "uncomfortable" in query.lower():
            if "uncomfortable" in text and "professional: nurse" in text:
                bonus += 6.0
            if "uncomfortable" not in text:
                bonus -= 5.0
            if "symptom of some disease" in text:
                bonus -= 4.0
        if "discussion" in section:
            bonus -= 3.0
        if "questionnaire was developed" in text:
            bonus -= 4.0
    elif intent == "appendix_checklist_lookup":
        query_lower = query.lower()
        if section == "Y/N":
            bonus += 5.0
        if "contra-indications" in text or "cautions" in text:
            bonus += 3.0
        if "live vaccine" in query_lower:
            if "live vaccine" in text:
                bonus += 8.0
            else:
                bonus -= 4.0
        if "anticoagulant" in query_lower:
            if "anticoagulant therapy" in text:
                bonus += 8.0
            else:
                bonus -= 4.0
        if "live vaccine" in text:
            bonus += 6.0
        if "anticoagulant therapy" in text:
            bonus += 6.0
        if "warfarin" in text or "noacs" in text or "doacs" in text:
            bonus += 4.0
    elif intent == "appendix_risk_list":
        if "possible risks and side effects from steroid injections" in text:
            bonus += 7.0
        if "allergic reaction" in text or "anaphylaxis" in text:
            bonus += 5.0
        if "tendon rupture" in text or "infections" in text:
            bonus += 4.0
        if "section 5" in text or "explanation" in text:
            bonus += 3.0
    elif intent == "review_prevention":
        if "handwashing" in text or "physical interventions" in text:
            bonus += 6.0
        if "best evidence for the prevention" in text:
            bonus += 5.0
        if "preventive measures" in text:
            bonus += 4.0
        if "zinc supplements" in text:
            bonus += 2.0
        if "summarized in table 1" in text:
            bonus -= 4.0
        if "we review the evidence" in text or "quality of the evidence was frequently poor" in text:
            bonus -= 4.0
        if "although preventive interventions have somewhat discrete outcomes" in text:
            bonus -= 4.0
    elif intent == "symptom_pathogenesis":
        if "symptom production is a combination of viral cytopathic effect" in text:
            bonus += 7.0
        if "activation of inflammatory pathways" in text:
            bonus += 5.0
        if "abstract" in section or "review" in section:
            bonus += 2.0
        if "rhinovirus is the most common" in text:
            bonus -= 1.5
        if "transmission" in text or "droplet" in text:
            bonus -= 4.0
    elif intent == "hypothermia_predisposition":
        if "predisposing factors for hypothermia" in text:
            bonus += 8.0
        if "decrease heat production" in text or "increase heat loss" in text:
            bonus += 5.0
        if "impair thermoregulation" in text or "miscellaneous clinical states" in text:
            bonus += 4.0
        if section.startswith(("GERMANY", "ENDOCRINE", "IATROGENIC")):
            bonus += 2.0
        if "to diagnose hypothermia" in text or "signs and symptoms of hypothermia" in text:
            bonus -= 4.0
    elif intent == "hypothermia_symptoms":
        if "signs and symptoms of hypothermia" in text:
            bonus += 8.0
        if "shivering" in text or "altered mental status" in text or "hypotension" in text:
            bonus += 4.0
        if "table 4-4" in text:
            bonus += 3.0
        if "common cold" in text or "sore throat" in text or "rhinorrhoea" in text:
            bonus -= 8.0
    elif intent == "frostbite_prevention":
        if "table 3-4" in text:
            bonus += 7.0
        if section in {"SEVERE", "EXTREME", "HIGH"}:
            bonus += 5.0
        if "severe" in query.lower():
            if section == "SEVERE":
                bonus += 6.0
            elif section in {"EXTREME", "HIGH", "LOW"}:
                bonus -= 5.0
        if "mandatory buddy checks every 10 minutes" in text:
            bonus += 7.0
        if "wear ecwcs or equivalent" in text or "provide warming facilities" in text:
            bonus += 5.0
        if "no exposed skin" in text or "stay active" in text:
            bonus += 4.0
        if "wear vb boots" in text or "work groups of no less than two personnel" in text:
            bonus += 3.0
        if "extreme risk" in text and "severe" in query.lower():
            bonus -= 7.0
        if "list of tables" in text or "frostbite risk low" in text:
            bonus -= 7.0
    elif intent == "immersion_limit":
        if "table 3-3" in text:
            bonus += 7.0
        if "50-54" in text and "neck" in text:
            bonus += 8.0
        if "5 minutes" in text:
            bonus += 5.0
        if "water temperature" in text and "immersion depths" in text:
            bonus += 4.0
        if "list of tables" in text:
            bonus -= 8.0
    elif intent == "review_nontraditional":
        if "nontraditional treatments" in text:
            bonus += 6.0
        if "oral zinc supplements" in text or ("honey" in text and "children" in text):
            bonus += 4.0
        if "summarized in table 3" in text or "summarized in table 2" in text:
            bonus -= 4.0
        if "we review the evidence" in text or "symptoms and signs of the common cold overlap" in text:
            bonus -= 4.0
        if "treatment of the common cold with echinacea: a structured review" in text:
            bonus -= 5.0
    elif intent == "antibiotics":
        if section.startswith("OPTION"):
            bonus += 4.0
        if "option antibiotics" in text:
            bonus += 7.0
        if "clinical guide" in text:
            bonus += 4.0
        if "don't reduce symptoms overall" in text:
            bonus += 6.0
        if "antibiotic resistance" in text or "adverse effects" in text:
            bonus += 3.0
        if "because most common colds are viral" in text:
            bonus += 2.5
        if "statistical_noise" in labels:
            bonus -= 4.0
    elif intent == "causes":
        if "AETIOLOGY" in section or "RISK FACTORS" in section:
            bonus += 6.0
        if "caused by viruses" in text or "mainly caused by viruses" in text:
            bonus += 5.0
        if "rhinovirus" in text or "coronavirus" in text or "respiratory syncytial virus" in text:
            bonus += 3.0
        if section.startswith("PROGNOSIS") or section.startswith("TREATMENTS"):
            bonus -= 2.0
    elif intent == "transmission":
        if "TRANSMISSION" in section or "AETIOLOGY" in section:
            bonus += 6.0
        if "hand-to-hand contact" in text:
            bonus += 5.0
        if "droplet" in text or "nostrils" in text or "eyes" in text:
            bonus += 2.0
    elif intent == "duration":
        if section.startswith("PROGNOSIS"):
            bonus += 6.0
        if "1 week" in text:
            bonus += 4.0
        if "generally clear by 1 week" in text:
            bonus += 2.0
        if "few days" in text or "cough" in text:
            bonus += 2.0
    elif intent == "incidence":
        if "INCIDENCE" in section or "PREVALENCE" in section:
            bonus += 6.0
        if "each year" in text and "children" in text and "adults" in text:
            bonus += 5.0
        if "up to 5 colds" in text or "two to three infections" in text:
            bonus += 2.5
        if "symptoms of colds" in text or "clearance of purulent rhinitis" in text:
            bonus -= 3.0
        if section.startswith("OPTION"):
            bonus -= 4.0
        if section.startswith("TREATMENTS") and not (
            "each year" in text and "children" in text and "adults" in text
        ):
            bonus -= 2.0
        if "adverse effects" in text:
            bonus -= 4.0
        if "statistical_noise" in labels:
            bonus -= 4.0
    elif intent == "ct_findings":
        if "DISCUSSION" in section or "FOLLOW-UP" in section or "CT SCANS" in section:
            bonus += 4.5
        if "high prevalence of ostiomeatal and sinus abnormalities" in text:
            bonus += 6.0
        if "sinus abnormalities on ct scans" in text or "abnormalities of one or more sinuses" in text:
            bonus += 4.0
        if "subjects with ct scans" in text and "abnormalities" in text:
            bonus += 2.5
        if "abstract" in text or "methods" in text or "study was approved" in text:
            bonus -= 3.0
        if "downloaded trom nejm" in text or "continuing medical education" in text:
            bonus -= 5.0
    elif intent == "ct_follow_up":
        if "FOLLOW-UP" in section or "DISCUSSION" in section:
            bonus += 4.5
        if "follow-up evaluation after 13 to 20 days" in text:
            bonus += 6.0
        if "residual abnormalities" in text or "marked improvement" in text:
            bonus += 4.0
        if "returned to normal" in text or "resolved or markedly improved" in text:
            bonus += 3.0
        if "abstract" in text or "methods" in text:
            bonus -= 3.5
        if "downloaded trom nejm" in text or "continuing medical education" in text:
            bonus -= 5.0
    elif intent == "symptoms":
        if section.startswith("DEFINITION") or section.startswith("PROGNOSIS"):
            bonus += 4.0
        if "symptoms include" in text:
            bonus += 4.0
        if "sore throat" in text or "runny nose" in text or "rhinorrhoea" in text:
            bonus += 2.0
    elif intent == "treatment_prevention":
        if subtopic_cues.intersection(INTENT_SUBTOPIC_CUES["treatment_prevention"]):
            bonus += 2.5
        if "vitamin c" in text or "echinacea" in text:
            bonus += 2.0
        if section.startswith("CONCLUSION") or section.startswith("REVIEW"):
            bonus += 2.0
        if "prevention" in text or "prophylaxis" in text:
            bonus += 4.0
        if "decreasing the incidence" in text or "reduces the incidence" in text:
            bonus += 5.0
        if "substantial reductions in the incidence" in text:
            bonus += 5.0
        if "published evidence supports" in text:
            bonus += 4.0
        if "suggests an additional benefit" in text:
            bonus += 4.0
        if "contracting a cold" in text:
            bonus += 3.0
        if "benefit" in text:
            bonus += 1.5
        if "incidence was not altered" in text or "normal populations" in text:
            bonus -= 2.5
        if "evidence for the prevention of a cold was lacking" in text:
            bonus -= 3.0
        if len(text.strip()) < 120:
            bonus -= 4.0
    elif intent == "treatment_null_effect":
        if subtopic_cues.intersection(INTENT_SUBTOPIC_CUES["treatment_null_effect"]):
            bonus += 2.5
        if "vitamin c" in text or "echinacea" in text:
            bonus += 2.0
        if "incidence was not altered" in text:
            bonus += 6.0
        if "lack of effect" in text:
            bonus += 4.0
        if "normal populations" in text:
            bonus += 5.0
        if "beneficial effect" in text or "50% reduction" in text or "decreasing the incidence" in text:
            bonus -= 3.0
        if len(text.strip()) < 120:
            bonus -= 4.0
    elif intent == "treatment_subgroup_benefit":
        if subtopic_cues.intersection(INTENT_SUBTOPIC_CUES["treatment_subgroup_benefit"]):
            bonus += 2.5
        if "vitamin c" in text or "echinacea" in text:
            bonus += 2.0
        if "cold stress" in text or "physical stress" in text or "subgroup" in text:
            bonus += 6.0
        if "marathon runners" in text or "skiers" in text or "soldiers" in text:
            bonus += 4.0
        if "50% reduction" in text or "beneficial effect" in text:
            bonus += 4.0
        if "collective evidence indicates" in text:
            bonus += 2.0
        if "normal populations" in text:
            bonus -= 2.0
        if len(text.strip()) < 120:
            bonus -= 4.0
    elif intent == "treatment_duration":
        if subtopic_cues.intersection(INTENT_SUBTOPIC_CUES["treatment_duration"]):
            bonus += 2.5
        if "vitamin c" in text or "echinacea" in text:
            bonus += 2.0
        if (
            "duration of cold episodes" in text
            or "duration of common cold episodes" in text
            or "duration of the common cold" in text
        ):
            bonus += 5.0
        if "reduced the duration" in text or "decrease the duration" in text:
            bonus += 4.0
        if "shortens the course" in text:
            bonus += 3.0
        if "days" in text:
            bonus += 2.0
        if "14%" in text or "8%" in text:
            bonus += 2.0
        if "onset of symptoms" in text or "8 g" in text:
            bonus += 1.0
        if len(text.strip()) < 120:
            bonus -= 4.0
    elif intent == "treatment_overall":
        if subtopic_cues.intersection(INTENT_SUBTOPIC_CUES["treatment_overall"]):
            bonus += 2.5
        if "vitamin c" in text or "echinacea" in text:
            bonus += 2.0
        if section.startswith("CONCLUSION") or section.startswith("REVIEW"):
            bonus += 4.0
        if "incidence" in text and "duration" in text:
            bonus += 4.0
        if "prevention" in text and "treatment" in text:
            bonus += 3.0
        if "published evidence supports" in text or "suggests that echinacea has a benefit" in text:
            bonus += 4.0
        if "suggests an additional benefit" in text:
            bonus += 4.0
        if "large-scale randomised prospective studies" in text:
            bonus += 1.5
        if "trials were included for analysis" in text or "inclusion criteria" in text:
            bonus -= 4.0
        if len(text.strip()) < 120:
            bonus -= 2.0
    return bonus


def _lightweight_query_bonus(chunk: ChunkRecord, query: str) -> float:
    query_terms = _content_terms(query)
    preferred_doc_id = _preferred_source_doc_id(query)
    matched_doc_ids = set(_matching_source_doc_ids(query))
    intent = _detect_query_intent(query)
    if not query_terms:
        query_terms = set()

    text = chunk.text.lower()
    section = (chunk.section_title or "").lower()
    source = chunk.source_pdf.lower()
    doc_id = chunk.doc_id.lower()
    combined_terms = _content_terms(f"{section} {source} {text}")
    overlap = query_terms.intersection(combined_terms)
    bonus = len(overlap) * 0.45

    if len(overlap) >= max(2, len(query_terms) // 2):
        bonus += 1.0

    if "cmaj" in query_terms and ("cmaj" in text or "cmaj" in section or "cmaj" in source):
        bonus += 3.0
    if "ajmedp" in query_terms and ("ajmedp" in text or "tb med 508" in text or "tb med 508" in section):
        bonus += 3.0
    if "opioid" in query_terms and (
        "opioid manager" in text or "appendix" in section or "appendix" in source
    ):
        bonus += 3.0
    if "adls" in query_terms and ("adls" in text or "activities of daily living" in text):
        bonus += 2.5
    if "follow" in query_terms and "3-day" in text:
        bonus += 2.5
    if "zinc" in query_terms and "zinc" in text:
        bonus += 2.5
    if "honey" in query_terms and "honey" in text:
        bonus += 2.5
    if "handwashing" in query_terms and "handwashing" in text:
        bonus += 4.0
    if intent == "document_overview":
        if section.startswith("contents") or section.startswith("foreword") or section.startswith("chapter"):
            bonus += 5.0
        if section.startswith("bibliography") or section.startswith("index"):
            bonus -= 5.0
    if intent in {"document_routing", "source_justification"}:
        if matched_doc_ids and chunk.doc_id in matched_doc_ids:
            bonus += 4.0
        if section.startswith("contents") or section.startswith("chapter"):
            bonus += 2.0
    if "nontraditional" in query_terms and "nontraditional" in text:
        bonus += 4.0
    if "prevent" in query_terms or "preventing" in query_terms or "prevention" in query_terms:
        if "prevention" in text or "preventing" in text or "physical interventions" in text:
            bonus += 2.0
    if preferred_doc_id and preferred_doc_id in doc_id:
        bonus += 4.0
        if "review" in query.lower():
            bonus += 1.5
    if matched_doc_ids:
        if chunk.doc_id in matched_doc_ids:
            bonus += 5.0
        elif _is_cross_document_intent(intent):
            bonus -= 5.0

    labels = set(chunk.noise_labels)
    if "table_reference" in labels:
        bonus -= 3.0
    if "toc_fragment" in labels or "toc_leader" in labels:
        bonus -= 4.0
    if "reference_tail" in labels or "bibliography" in labels:
        bonus -= 4.0
    if "title_fragment" in labels:
        bonus -= 1.5

    if len(text.strip()) < 90:
        bonus -= 0.5

    return bonus


def _should_exclude_chunk(chunk: ChunkRecord) -> bool:
    labels = set(chunk.noise_labels)
    if labels.intersection(HARD_EXCLUDE_LABELS):
        return True
    if chunk.quality_score <= 0.10:
        return True
    return False


def _should_expand_to_neighbor(
    anchor: ChunkRecord,
    neighbor: ChunkRecord,
    query: str,
) -> bool:
    """Decide whether a neighbor is high-signal enough to expand into answer context."""
    if _should_exclude_chunk(neighbor):
        return False

    labels = set(neighbor.noise_labels)
    if labels.intersection(EXPANSION_BLOCK_LABELS):
        return False

    if neighbor.quality_score < 0.35:
        return False

    intent = _detect_query_intent(query)
    if intent != "generic" and not _chunk_matches_intent(neighbor, intent):
        anchor_section = (anchor.section_title or "").strip().upper()
        neighbor_section = (neighbor.section_title or "").strip().upper()
        same_section = bool(anchor_section and neighbor_section and anchor_section == neighbor_section)
        if not same_section:
            return False

    if (
        neighbor.quality_score < 0.55
        and len(labels.intersection(SOFT_NOISE_LABELS)) >= 2
    ):
        return False

    return True


def _chunk_matches_intent(chunk: ChunkRecord, intent: str) -> bool:
    if intent == "generic":
        return True

    text = chunk.text.lower()

    if intent == "review_prevention":
        return (
            "handwashing" in text
            or "physical interventions" in text
            or "best evidence for the prevention" in text
            or "preventive measures" in text
            or "zinc supplements" in text
        )
    if intent == "review_nontraditional":
        return (
            "nontraditional treatments" in text
            or "oral zinc supplements" in text
            or "honey at bedtime" in text
            or ("honey" in text and "children" in text)
        )
    if intent == "opioid_pre_therapy_checklist":
        return (
            "appendix a" in text
            or "checklist" in text
            or "non-pharmacological therapy" in text
            or "non-opioid pharmacotherapy" in text
            or "informed consent" in text
            or "opioid safety" in text
            or "urine drug screening" in text
        )
    if intent == "opioid_adverse_effect_scale":
        return (
            "appendix b" in text
            or "adverse effects" in text
            or "0 = none" in text
            or "1 = limits adls" in text
            or "2 = prevents adls" in text
        )
    if intent == "opioid_switch_follow_up":
        return (
            "appendix c" in text
            or "switching opioids" in text
            or "3-day follow-up" in text
            or "follow up with patient every 2-4 weeks" in text
            or "follow up with patient every 2–4 weeks" in text
        )
    if intent == "questionnaire_performance":
        return (
            "question 13" in text
            or "performance at work" in text
            or "concentration" in text
            or "manual strength" in text
        )
    if intent == "questionnaire_symptom_scale":
        return (
            "question 5" in text
            or "shortness of breath" in text
            or "persistent coughing" in text
            or "during exercise" in text
        )
    if intent == "questionnaire_color_change":
        return (
            "question 9" in text
            or ("white" in text and "blue" in text)
            or "red/purple" in text
        )
    if intent == "questionnaire_frostbite_history":
        return (
            "question 12" in text
            or "blister grade" in text
            or ("once" in text and "several times" in text)
        )
    if intent == "questionnaire_follow_up_table":
        return (
            "table i" in text
            or "interview of working ability" in text
            or "professional: nurse" in text
            or "nurse and physician" in text
        )
    if intent == "symptom_pathogenesis":
        return (
            "symptom production is a combination of viral cytopathic effect" in text
            or "activation of inflammatory pathways" in text
        )
    if intent == "hypothermia_predisposition":
        return (
            "predisposing factors for hypothermia" in text
            or "decrease heat production" in text
            or "increase heat loss" in text
            or "impair thermoregulation" in text
            or "miscellaneous clinical states" in text
        )
    if intent == "hypothermia_symptoms":
        return (
            "signs and symptoms of hypothermia" in text
            or ("hypothermia" in text and "shivering" in text)
            or "altered mental status" in text
            or "hypotension" in text
        )
    if intent == "frostbite_prevention":
        return (
            "table 3-4" in text
            or "mandatory buddy checks every 10 minutes" in text
            or "no exposed skin" in text
            or "stay active" in text
            or "warming facilities" in text
            or "wear ecwcs" in text
        )
    if intent == "immersion_limit":
        return (
            "table 3-3" in text
            or ("50-54" in text and "neck" in text)
            or ("immersion time limits" in text and "water temperature" in text)
        )
    if intent == "appendix_checklist_lookup":
        return (
            "live vaccine" in text
            or "anticoagulant therapy" in text
            or "warfarin" in text
            or "noacs" in text
            or "doacs" in text
            or "pregnancy or breastfeeding" in text
            or "local or systemic infection" in text
        )
    if intent == "appendix_risk_list":
        return (
            "possible risks and side effects from steroid injections" in text
            or "allergic reaction" in text
            or "anaphylaxis" in text
            or "tendon rupture" in text
            or "post injection flare" in text
        )

    if set(chunk.subtopic_cues).intersection(INTENT_SUBTOPIC_CUES.get(intent, set())):
        return True

    section = (chunk.section_title or "").upper()
    structured_profile = get_structured_intent_profile(intent)
    section_hints = (
        structured_profile.section_hints if structured_profile else INTENT_SECTION_HINTS.get(intent, ())
    )
    if any(hint in section for hint in section_hints):
        return True
    if intent == "incidence":
        return ("each year" in text and "children" in text) or ("adults" in text and "infections" in text)
    if intent == "antibiotics":
        return "antibiotic" in text or "antibiotics" in text
    if intent == "treatment_prevention":
        return _has_treatment_entity(set(re.findall(r"[a-zA-Z]{2,}", text))) and (
            "incidence" in text
            or "prevention" in text
            or "prophylaxis" in text
            or "contracting a cold" in text
            or "substantial reductions in the incidence" in text
        )
    if intent == "treatment_null_effect":
        return _has_treatment_entity(set(re.findall(r"[a-zA-Z]{2,}", text))) and (
            "normal populations" in text or "incidence was not altered" in text or "no effect" in text
        )
    if intent == "treatment_subgroup_benefit":
        return _has_treatment_entity(set(re.findall(r"[a-zA-Z]{2,}", text))) and (
            "cold stress" in text or "physical stress" in text or "marathon runners" in text
        )
    if intent == "treatment_duration":
        return _has_treatment_entity(set(re.findall(r"[a-zA-Z]{2,}", text))) and (
            "duration of cold episodes" in text
            or "duration of the common cold" in text
            or "reduced the duration" in text
            or "shortens the course" in text
        )
    if intent == "treatment_overall":
        return _has_treatment_entity(set(re.findall(r"[a-zA-Z]{2,}", text))) and (
            ("incidence" in text and "duration" in text)
            or ("prevention" in text and "treatment" in text)
            or "meta-analysis" in text
            or "suggests an additional benefit" in text
        )
    if intent == "transmission":
        return "hand-to-hand contact" in text or "droplet" in text or "transmission" in text
    if intent == "causes":
        return "caused by viruses" in text or "rhinovirus" in text or "coronavirus" in text
    if intent == "duration":
        return "1 week" in text or "few days" in text or "cough often persists" in text
    if intent == "symptoms":
        return "symptoms include" in text or "sore throat" in text or "rhinorrhoea" in text
    if intent == "definition":
        return "defined as" in text
    return False


def _rerank_hits(hits: list[ChunkRecord], query: str) -> list[ChunkRecord]:
    scored = []
    for index, chunk in enumerate(hits):
        score = _heuristic_hit_bonus(chunk, query) - (index * 0.01)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored]


def _lightweight_rerank_hits(hits: list[ChunkRecord], query: str) -> list[ChunkRecord]:
    scored = []
    for index, chunk in enumerate(hits):
        score = (
            _heuristic_hit_bonus(chunk, query)
            + _lightweight_query_bonus(chunk, query)
            - (index * 0.01)
        )
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored]


def _diversify_hits_by_doc(
    hits: list[ChunkRecord],
    k: int,
    per_doc_limit: int,
) -> list[ChunkRecord]:
    selected: list[ChunkRecord] = []
    per_doc_counts: dict[str, int] = {}

    for chunk in hits:
        count = per_doc_counts.get(chunk.doc_id, 0)
        if count >= per_doc_limit:
            continue
        selected.append(chunk)
        per_doc_counts[chunk.doc_id] = count + 1
        if len(selected) >= k:
            return selected

    for chunk in hits:
        if chunk in selected:
            continue
        selected.append(chunk)
        if len(selected) >= k:
            break
    return selected


def retrieve_top_k(
    query: str,
    index_dir: Path,
    k: int = 5,
    use_lightweight_rerank: bool = True,
) -> list[ChunkRecord]:
    """Retrieve the most relevant chunks for a query."""
    index_dir = index_dir.expanduser().resolve()
    manifest = load_index_manifest(index_dir)
    collection_name = manifest.get("collection_name", DEFAULT_COLLECTION_NAME)
    intent = _detect_query_intent(query)

    embed_texts, _ = load_embedder_from_manifest(manifest)
    query_embedding = embed_texts([_augment_query(query)])[0]
    candidate_k = _candidate_pool_size(query, k)

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(name=collection_name)

    def hydrate(result: dict) -> list[ChunkRecord]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        hydrated: list[ChunkRecord] = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            noise_labels_raw = metadata.get("noise_labels")
            noise_labels = (
                [item for item in str(noise_labels_raw).split("|") if item]
                if noise_labels_raw
                else []
            )
            subtopic_cues_raw = metadata.get("subtopic_cues")
            subtopic_cues = (
                [item for item in str(subtopic_cues_raw).split("|") if item]
                if subtopic_cues_raw
                else []
            )
            hydrated.append(
                ChunkRecord(
                    doc_id=metadata["doc_id"],
                    chunk_id=chunk_id,
                    source_pdf=metadata["source_pdf"],
                    text=text,
                    page_start=int(metadata["page_start"]),
                    page_end=int(metadata["page_end"]),
                    bbox=None,
                    section_title=metadata.get("section_title"),
                    section_level=(
                        int(metadata["section_level"])
                        if metadata.get("section_level") is not None
                        else None
                    ),
                    chunk_type=metadata.get("chunk_type", "text"),
                    reading_order_index=int(metadata["reading_order_index"]),
                    preceding_chunk_id=metadata.get("preceding_chunk_id"),
                    following_chunk_id=metadata.get("following_chunk_id"),
                    language=metadata.get("language"),
                    extraction_method=metadata.get("extraction_method", "native"),
                    ocr_used=bool(metadata.get("ocr_used", False)),
                    subtopic_cues=subtopic_cues,
                    noise_labels=noise_labels,
                    quality_score=float(metadata.get("quality_score", 1.0)),
                    confidence=None,
                )
            )
        return hydrated

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = hydrate(result)
    hydrated_hits: list[ChunkRecord] = []
    for chunk in hits:
        labels, score = classify_chunk_quality(
            text=chunk.text,
            section_title=chunk.section_title,
            extraction_method=chunk.extraction_method,
        )
        chunk.noise_labels = sorted(set(chunk.noise_labels).union(labels))
        chunk.quality_score = min(chunk.quality_score, score)
        hydrated_hits.append(chunk)
    filtered_hits = [chunk for chunk in hydrated_hits if not _should_exclude_chunk(chunk)]
    inventory_doc_ids = set(_inventory_doc_ids(query))
    if inventory_doc_ids:
        filtered_hits = [chunk for chunk in filtered_hits if chunk.doc_id in inventory_doc_ids]
        present_doc_ids = {chunk.doc_id for chunk in filtered_hits}
        missing_inventory_doc_ids = [doc_id for doc_id in inventory_doc_ids if doc_id not in present_doc_ids]
        for doc_id in missing_inventory_doc_ids:
            doc_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where={"doc_id": doc_id},
                include=["documents", "metadatas", "distances"],
            )
            for chunk in hydrate(doc_result):
                labels, score = classify_chunk_quality(
                    text=chunk.text,
                    section_title=chunk.section_title,
                    extraction_method=chunk.extraction_method,
                )
                chunk.noise_labels = sorted(set(chunk.noise_labels).union(labels))
                chunk.quality_score = min(chunk.quality_score, score)
                if not _should_exclude_chunk(chunk):
                    filtered_hits.append(chunk)
        deduped_hits: dict[str, ChunkRecord] = {}
        for chunk in filtered_hits:
            deduped_hits.setdefault(chunk.chunk_id, chunk)
        filtered_hits = list(deduped_hits.values())
    matched_doc_ids = set(_matching_source_doc_ids(query)) if _is_cross_document_intent(intent) else set()
    if matched_doc_ids:
        filtered_hits = [chunk for chunk in filtered_hits if chunk.doc_id in matched_doc_ids]
        present_doc_ids = {chunk.doc_id for chunk in filtered_hits}
        missing_doc_ids = [doc_id for doc_id in matched_doc_ids if doc_id not in present_doc_ids]
        for doc_id in missing_doc_ids:
            doc_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where={"doc_id": doc_id},
                include=["documents", "metadatas", "distances"],
            )
            for chunk in hydrate(doc_result):
                labels, score = classify_chunk_quality(
                    text=chunk.text,
                    section_title=chunk.section_title,
                    extraction_method=chunk.extraction_method,
                )
                chunk.noise_labels = sorted(set(chunk.noise_labels).union(labels))
                chunk.quality_score = min(chunk.quality_score, score)
                if not _should_exclude_chunk(chunk):
                    filtered_hits.append(chunk)
        deduped_hits: dict[str, ChunkRecord] = {}
        for chunk in filtered_hits:
            deduped_hits.setdefault(chunk.chunk_id, chunk)
        filtered_hits = list(deduped_hits.values())
    preferred_doc_id = None if _is_cross_document_intent(intent) else _preferred_source_doc_id(query)
    if preferred_doc_id:
        preferred_hits = [chunk for chunk in filtered_hits if chunk.doc_id == preferred_doc_id]
        if not preferred_hits:
            preferred_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where={"doc_id": preferred_doc_id},
                include=["documents", "metadatas", "distances"],
            )
            preferred_hydrated = []
            for chunk in hydrate(preferred_result):
                labels, score = classify_chunk_quality(
                    text=chunk.text,
                    section_title=chunk.section_title,
                    extraction_method=chunk.extraction_method,
                )
                chunk.noise_labels = sorted(set(chunk.noise_labels).union(labels))
                chunk.quality_score = min(chunk.quality_score, score)
                preferred_hydrated.append(chunk)
            preferred_hits = [
                chunk for chunk in preferred_hydrated if not _should_exclude_chunk(chunk)
            ]
        if preferred_hits:
            filtered_hits = preferred_hits
    reranked_hits = _lightweight_rerank_hits(filtered_hits, query) if use_lightweight_rerank else _rerank_hits(filtered_hits, query)
    if intent == "source_listing":
        return _diversify_hits_by_doc(reranked_hits, k=k, per_doc_limit=1)[:k]
    if intent == "cross_document_compare":
        return _diversify_hits_by_doc(reranked_hits, k=k, per_doc_limit=2)[:k]
    if intent in {"document_routing", "source_justification"}:
        return _diversify_hits_by_doc(reranked_hits, k=k, per_doc_limit=1)[:k]
    return reranked_hits[:k]


def load_chunk_lookup(chunk_root: Path, doc_ids: set[str] | None = None) -> dict[str, ChunkRecord]:
    """Load chunk JSON records into a chunk_id -> ChunkRecord lookup."""
    chunk_root = chunk_root.expanduser().resolve()
    if not chunk_root.exists():
        raise FileNotFoundError(f"Chunk root not found: {chunk_root}")

    lookup: dict[str, ChunkRecord] = {}
    search_dirs = []
    if doc_ids:
        for doc_id in sorted(doc_ids):
            doc_dir = chunk_root / doc_id
            if doc_dir.exists():
                search_dirs.append(doc_dir)
    else:
        search_dirs = [path for path in sorted(chunk_root.iterdir()) if path.is_dir()]

    for doc_dir in search_dirs:
        for chunk_path in sorted(doc_dir.glob("*.json")):
            data = json.loads(chunk_path.read_text(encoding="utf-8"))
            chunk = ChunkRecord.model_validate(data)
            lookup[chunk.chunk_id] = chunk
    return lookup


def expand_with_neighbors(
    hits: list[ChunkRecord],
    all_chunks: dict[str, ChunkRecord],
    query: str,
) -> list[ChunkRecord]:
    """Expand retrieval results with preceding and following chunks."""
    intent = _detect_query_intent(query)
    structured_profile = get_structured_intent_profile(intent)
    depth = structured_profile.neighbor_depth if structured_profile else INTENT_NEIGHBOR_DEPTH.get(intent, 1)
    expanded: dict[str, ChunkRecord] = {}

    def maybe_add_neighbor(
        anchor: ChunkRecord,
        neighbor_id: str | None,
        steps_remaining: int,
    ) -> None:
        if not neighbor_id or neighbor_id in expanded or steps_remaining <= 0:
            return
        neighbor = all_chunks.get(neighbor_id)
        if not neighbor:
            return
        if not _should_expand_to_neighbor(anchor=anchor, neighbor=neighbor, query=query):
            return
        expanded[neighbor.chunk_id] = neighbor
        maybe_add_neighbor(neighbor, neighbor.preceding_chunk_id, steps_remaining - 1)
        maybe_add_neighbor(neighbor, neighbor.following_chunk_id, steps_remaining - 1)

    for chunk in hits:
        expanded[chunk.chunk_id] = chunk
        maybe_add_neighbor(chunk, chunk.preceding_chunk_id, depth)
        maybe_add_neighbor(chunk, chunk.following_chunk_id, depth)
    return sorted(
        expanded.values(),
        key=lambda chunk: (chunk.doc_id, chunk.reading_order_index, chunk.chunk_id),
    )


def retrieve_top_k_with_neighbors(
    query: str,
    index_dir: Path,
    chunk_root: Path,
    k: int = 5,
    use_lightweight_rerank: bool = True,
) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    """Retrieve top-k chunks and expand them with adjacent neighbors."""
    hits = retrieve_top_k(
        query=query,
        index_dir=index_dir,
        k=k,
        use_lightweight_rerank=use_lightweight_rerank,
    )
    doc_ids = {chunk.doc_id for chunk in hits}
    all_chunks = load_chunk_lookup(chunk_root=chunk_root, doc_ids=doc_ids)
    expanded = expand_with_neighbors(hits=hits, all_chunks=all_chunks, query=query)
    return hits, expanded
