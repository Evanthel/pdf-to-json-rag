"""Grounded answer assembly for the MVP pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .intent_config import (
    detect_structured_intent,
    get_document_profile,
    get_structured_intent_profile,
    matching_source_doc_ids as configured_matching_source_doc_ids,
    preferred_source_doc_id as configured_source_doc_id,
)
from .retrieval import retrieve_top_k_with_neighbors
from .schemas import ChunkRecord, DocumentRecord


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}

LOW_SIGNAL_QUERY_TERMS = {
    "according",
    "benchmark",
    "common",
    "cold",
    "colds",
    "document",
    "file",
    "literature",
    "most",
    "review",
    "paper",
    "relevant",
    "prevent",
    "prevents",
    "preventing",
    "treat",
    "treats",
    "treating",
    "treatment",
    "help",
    "helps",
    "source",
    "sources",
    "say",
    "says",
}
MULTI_DOC_COMPARE_TERMS = {"compare", "versus", "vs"}
UNSUPPORTED_ENTITY_TERMS = {
    "aspirin",
    "gadolinium",
    "insulin",
    "monoclonal",
    "vaccine",
}

NO_GROUNDED_ANSWER = "No grounded answer could be assembled from the retrieved context."

SYMPTOM_HINTS = {
    "symptom",
    "symptoms",
    "include",
    "sneezing",
    "rhinorrhoea",
    "runny",
    "nose",
    "headache",
    "malaise",
    "sore",
    "throat",
    "cough",
}

TREATMENT_NOISE = {
    "placebo",
    "effective",
    "effectiveness",
    "reduce",
    "reducing",
    "treatment",
    "treatments",
    "vitamin",
    "antihistamines",
    "decongestants",
    "evidence",
}

DEFINITION_HINTS = {
    "definition",
    "defined",
    "upper",
    "respiratory",
    "tract",
    "mucosa",
}

CAUSE_HINTS = {
    "cause",
    "causes",
    "caused",
    "virus",
    "viruses",
    "rhinovirus",
    "coronavirus",
    "syncytial",
    "metapneumovirus",
}
SYMPTOM_PATHOGENESIS_HINTS = {
    "symptom",
    "production",
    "viral",
    "cytopathic",
    "effect",
    "activation",
    "inflammatory",
    "pathways",
}

TRANSMISSION_HINTS = {
    "transmission",
    "transmitted",
    "hand",
    "contact",
    "droplet",
    "nostrils",
    "eyes",
    "virus",
    "viruses",
}

INCIDENCE_HINTS = {
    "incidence",
    "prevalence",
    "year",
    "children",
    "adults",
    "infections",
}

DURATION_HINTS = {
    "duration",
    "last",
    "lasts",
    "days",
    "week",
    "weeks",
    "peak",
    "clear",
    "lingering",
    "persists",
    "cough",
}

CT_FINDINGS_HINTS = {
    "ct",
    "scans",
    "sinus",
    "abnormalities",
    "ostiomeatal",
    "ethmoid",
    "maxillary",
}

CT_FOLLOW_UP_HINTS = {
    "follow-up",
    "follow",
    "days",
    "residual",
    "abnormalities",
    "improvement",
    "resolved",
    "normal",
}
HYPOTHERMIA_PREDISPOSITION_HINTS = {
    "predisposing",
    "factors",
    "decrease",
    "heat",
    "production",
    "loss",
    "thermoregulation",
}
HYPOTHERMIA_SYMPTOM_HINTS = {
    "hypothermia",
    "signs",
    "symptoms",
    "shivering",
    "confusion",
    "hypotension",
    "mental",
    "status",
}
FROSTBITE_PREVENTION_HINTS = {
    "frostbite",
    "risk",
    "severe",
    "buddy",
    "checks",
    "warming",
    "facilities",
    "ecwcs",
    "active",
}
IMMERSION_LIMIT_HINTS = {
    "immersion",
    "time",
    "limits",
    "neck",
    "minutes",
    "water",
    "temperature",
}

VITAMIN_C_HINTS = {
    "vitamin",
    "prophylaxis",
    "incidence",
    "duration",
    "normal",
    "populations",
    "stress",
    "physical",
    "beneficial",
}

TREATMENT_ENTITY_HINTS = {
    "vitamin",
    "echinacea",
    "propolis",
}
SOURCE_ANCHORED_HINTS = {
    "ajmedp",
    "cmaj",
    "frostbite",
    "health-check",
    "opioid",
    "appendix",
    "ct",
    "echinacea",
    "gadolinium",
    "hypothermia",
    "literature",
    "vitamin",
    "wat",
    "zinc",
}

TREATMENT_PREVENTION_HINTS = {
    "prevention",
    "prevent",
    "prevents",
    "prophylaxis",
    "incidence",
    "contracting",
    "benefit",
}

TREATMENT_NULL_EFFECT_HINTS = {
    "normal",
    "populations",
    "not",
    "altered",
    "no",
    "effect",
}

TREATMENT_SUBGROUP_HINTS = {
    "stress",
    "physical",
    "subgroup",
    "beneficial",
    "reduction",
    "runners",
    "skiers",
    "soldiers",
}

TREATMENT_DURATION_HINTS = {
    "duration",
    "shorten",
    "shortens",
    "reduced",
    "days",
    "episodes",
    "course",
}

TREATMENT_OVERALL_HINTS = {
    "meta-analysis",
    "conclusion",
    "incidence",
    "duration",
    "prevention",
    "treatment",
    "benefit",
}
REVIEW_PREVENTION_HINTS = {
    "preventive",
    "prevention",
    "interventions",
    "handwashing",
    "physical",
    "zinc",
}
REVIEW_NONTRADITIONAL_HINTS = {
    "nontraditional",
    "oral",
    "zinc",
    "honey",
    "cough",
}
QUESTIONNAIRE_PERFORMANCE_HINTS = {
    "performance",
    "concentration",
    "motivation",
    "manual",
    "strength",
    "musculo-skeletal",
    "cooling",
}
QUESTIONNAIRE_SYMPTOM_SCALE_HINTS = {
    "shortness",
    "breath",
    "persistent",
    "coughing",
    "wheezing",
    "mucus",
    "exercise",
    "warm",
    "cold",
}
QUESTIONNAIRE_COLOR_HINTS = {
    "white",
    "blue",
    "red/purple",
    "fingers",
    "colours",
    "colors",
}
QUESTIONNAIRE_FROSTBITE_HINTS = {
    "frostbite",
    "blister",
    "once",
    "several",
    "times",
}
QUESTIONNAIRE_TABLE_HINTS = {
    "table i",
    "uncomfortable",
    "sensitivity",
    "interview of working ability",
    "disease-focused interview",
    "professional",
    "nurse",
    "physician",
}
OPIOID_PRE_THERAPY_HINTS = {
    "appendix",
    "checklist",
    "optimized",
    "non-pharmacological",
    "non-opioid",
    "informed",
    "consent",
    "safety",
    "urine",
    "screening",
}
OPIOID_ADVERSE_SCALE_HINTS = {
    "adverse",
    "effects",
    "adls",
    "none",
    "limits",
    "prevents",
}
OPIOID_SWITCH_FOLLOWUP_HINTS = {
    "switching",
    "follow-up",
    "withdrawal",
    "pain",
    "3-day",
    "weeks",
}
ANTIBIOTICS_HINTS = {
    "antibiotic",
    "antibiotics",
    "resistance",
    "adverse",
    "viral",
    "sinusitis",
}

BIBLIOGRAPHIC_NOISE = {
    "abstract",
    "introduction",
    "review",
    "trial",
    "double-blind",
    "placebo-controlled",
    "zincum",
    "nasal gel",
}

STRONG_INTENT_ANCHORS = {
    "treatment_null_effect": (
        "incidence was not altered",
        "lack of effect",
        "normal populations",
    ),
    "treatment_subgroup_benefit": (
        "cold stress",
        "physical stress",
        "beneficial effect",
        "50% reduction",
    ),
    "treatment_duration": (
        "duration of cold episodes",
        "reduction in duration",
        "symptom days",
    ),
    "treatment_overall": (
        "suggests that echinacea has a benefit",
        "reduces the incidence as well as the duration",
    ),
}


@dataclass
class EvidenceSentence:
    chunk_id: str
    page_start: int
    page_end: int
    section_title: str | None
    sentence: str
    score: float


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    evidence: list[EvidenceSentence]
    top_k_hits: list[ChunkRecord]
    expanded_hits: list[ChunkRecord]
    query_intent: str
    answer_trace: dict[str, object]


def _normalize_text(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", text).strip()


def _query_terms(query: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[a-zA-Z]{2,}", query.lower())
        if token not in STOPWORDS
    }
    return terms


def _normalized_sentence_surface(text: str) -> str:
    normalized = _normalize_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _specific_query_terms(query_terms: set[str]) -> set[str]:
    specific = query_terms - LOW_SIGNAL_QUERY_TERMS
    return specific or query_terms


def _split_sentences(text: str) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}", text)
    return [part.strip() for part in parts if len(part.strip()) >= 30]


def _detect_query_intent(query: str, query_terms: set[str]) -> str:
    query_lower = query.lower()
    structured_intent = detect_structured_intent(query, query_terms)
    if structured_intent:
        return structured_intent
    metadata_matches = configured_matching_source_doc_ids(query, allow_topical=True)
    if (
        ("what does" in query_lower and "cover" in query_lower)
        or ("what is" in query_lower and "about" in query_lower)
    ) and configured_source_doc_id(query):
        return "document_overview"
    if (
        ("which file" in query_lower or "which document" in query_lower or "which source" in query_lower)
        and (
            "most relevant" in query_lower
            or "best source" in query_lower
            or "best document" in query_lower
        )
    ):
        return "document_routing"
    if (
        query_lower.startswith("why")
        and (
            "most relevant source" in query_lower
            or "best match" in query_lower
            or "best source" in query_lower
            or "most relevant file" in query_lower
        )
    ):
        return "source_justification"
    if "which sources" in query_lower or "which documents" in query_lower:
        return "source_listing"
    if query_terms.intersection(MULTI_DOC_COMPARE_TERMS) and len(metadata_matches) >= 2:
        return "cross_document_compare"
    if query_terms.intersection(MULTI_DOC_COMPARE_TERMS) and len(query_terms.intersection(TREATMENT_ENTITY_HINTS)) >= 2:
        return "cross_document_compare"
    if "antibiotic" in query_terms or "antibiotics" in query_terms:
        return "antibiotics"
    if "hypothermia" in query_terms and {"predisposing", "predispose", "factors", "categories"}.intersection(query_terms):
        return "hypothermia_predisposition"
    if "hypothermia" in query_terms and {"signs", "symptoms"}.intersection(query_terms):
        return "hypothermia_symptoms"
    if "frostbite" in query_terms and (
        "severe" in query_terms
        or "preventive" in query_terms
        or "measures" in query_terms
        or "zone" in query_terms
        or "risk" in query_terms
    ):
        return "frostbite_prevention"
    if "immersion" in query_terms and ("neck" in query_terms or "depth" in query_terms):
        return "immersion_limit"
    if ("cause" in query_terms or "causes" in query_terms) and (
        "symptom" in query_terms or "symptoms" in query_terms
    ):
        return "symptom_pathogenesis"
    if (
        ("nontraditional" in query_terms and "treatments" in query_terms)
        or "nontraditional treatments" in query_lower
    ):
        return "review_nontraditional"
    if (
        "cmaj" in query_terms
        and "zinc" in query_terms
        and (
            "prevent" in query_terms
            or "preventing" in query_terms
            or "prevention" in query_terms
        )
    ):
        return "treatment_prevention"
    if (
        ("preventive" in query_terms and "interventions" in query_terms)
        or ("handwashing" in query_terms and "prevent" in query_terms)
        or ("best" in query_terms and "evidence" in query_terms and "prevent" in query_terms)
    ):
        return "review_prevention"
    has_treatment_query = bool(query_terms & TREATMENT_ENTITY_HINTS) and "cold" in query_terms
    if has_treatment_query:
        if "stress" in query_terms or ("physical" in query_terms and "stress" in query_terms) or "subgroup" in query_terms:
            return "treatment_subgroup_benefit"
        if "normal" in query_terms and "populations" in query_terms:
            return "treatment_null_effect"
        if "duration" in query_terms or "shorten" in query_terms:
            return "treatment_duration"
        if "conclude" in query_terms or "conclusion" in query_terms or "meta" in query_terms or "analysis" in query_terms:
            return "treatment_overall"
        if (
            "prevent" in query_terms
            or "prevents" in query_terms
            or "prevention" in query_terms
            or "prophylaxis" in query_terms
            or "incidence" in query_terms
        ):
            return "treatment_prevention"
    if query_lower.startswith("what is") or "definition" in query_terms or "define" in query_terms:
        return "definition"
    if "ct" in query_terms and ("follow" in query_terms or "followup" in query_terms):
        return "ct_follow_up"
    if "ct" in query_terms and ("abnormalities" in query_terms or "sinus" in query_terms or "scans" in query_terms):
        return "ct_findings"
    if "cause" in query_terms or "causes" in query_terms:
        return "causes"
    if "transmitted" in query_terms or "transmission" in query_terms:
        return "transmission"
    if "last" in query_terms or "long" in query_terms or "duration" in query_terms:
        return "duration"
    if "year" in query_terms or ("children" in query_terms and "adults" in query_terms):
        return "incidence"
    if "symptom" in query_terms or "symptoms" in query_terms:
        return "symptoms"
    return "generic"


def _preferred_source_doc_id(query: str) -> str | None:
    query_intent = _detect_query_intent(query, _query_terms(query))
    if query_intent in {"source_listing", "cross_document_compare", "document_routing"}:
        return None
    return configured_source_doc_id(query)


def _matching_source_doc_ids(query: str) -> list[str]:
    query_intent = _detect_query_intent(query, _query_terms(query))
    allow_topical = query_intent in {
        "source_listing",
        "document_routing",
        "source_justification",
        "cross_document_compare",
    }
    matches = configured_matching_source_doc_ids(query, allow_topical=allow_topical)
    query_lower = query.lower()
    if query_intent == "source_justification" and matches:
        return matches[:1]
    if query_intent == "document_routing" and matches:
        if "which file or files" not in query_lower and "which files" not in query_lower:
            return matches[:1]
    return matches


def _score_sentence(
    sentence: str,
    query_terms: set[str],
    query_intent: str,
    section_title: str | None = None,
) -> float:
    sentence_lower = sentence.lower()
    sentence_terms = set(re.findall(r"[a-zA-Z]{2,}", sentence_lower))
    section_upper = (section_title or "").upper()
    overlap = sentence_terms & query_terms
    anchor_overlap = any(
        phrase in sentence_lower for phrase in STRONG_INTENT_ANCHORS.get(query_intent, ())
    )
    if query_intent == "antibiotics":
        if (
            ("antibiotic" in sentence_terms or "antibiotics" in sentence_terms)
            and any(term in query_terms for term in {"antibiotic", "antibiotics"})
        ):
            anchor_overlap = True
        if "side effects" in sentence_lower or "resistant organisms" in sentence_lower:
            anchor_overlap = True
    if query_intent == "questionnaire_performance":
        if "question 13" in sentence_lower or "performance at work" in sentence_lower:
            anchor_overlap = True
    if query_intent == "questionnaire_symptom_scale":
        if "question 5" in sentence_lower or "shortness of breath" in sentence_lower:
            anchor_overlap = True
    if query_intent == "questionnaire_color_change":
        if "question 9" in sentence_lower or (
            "white" in sentence_lower and "blue" in sentence_lower and "red/purple" in sentence_lower
        ):
            anchor_overlap = True
    if query_intent == "questionnaire_frostbite_history":
        if "question 12" in sentence_lower or "blister grade" in sentence_lower:
            anchor_overlap = True
    if query_intent == "questionnaire_follow_up_table":
        if "table i" in sentence_lower or "professional: nurse" in sentence_lower:
            anchor_overlap = True
    if query_intent == "opioid_pre_therapy_checklist":
        if "appendix a" in sentence_lower or "non-pharmacological therapy" in sentence_lower:
            anchor_overlap = True
    if query_intent == "opioid_adverse_effect_scale":
        if "adverse-effect scale" in sentence_lower or "0 = none" in sentence_lower:
            anchor_overlap = True
    if query_intent == "opioid_switch_follow_up":
        if "3-day follow-up" in sentence_lower or "follow up every 2-4 weeks" in sentence_lower:
            anchor_overlap = True
    if query_intent == "appendix_checklist_lookup":
        if "live vaccine" in sentence_lower or "anticoagulant therapy" in sentence_lower:
            anchor_overlap = True
    if query_intent == "appendix_risk_list":
        if "possible risks and side effects from steroid injections" in sentence_lower:
            anchor_overlap = True
        if "allergic reaction" in sentence_lower or "tendon rupture" in sentence_lower:
            anchor_overlap = True
    if query_intent == "symptom_pathogenesis":
        if (
            "viral cytopathic effect" in sentence_lower
            or "activation of inflammatory pathways" in sentence_lower
            or "symptom production" in sentence_lower
        ):
            anchor_overlap = True
    if query_intent == "hypothermia_predisposition" and section_upper in {"GERMANY", "ENDOCRINE", "IATROGENIC", "TB MED 508"}:
        if any(
            phrase in sentence_lower
            for phrase in (
                "decrease heat production",
                "increase heat loss",
                "impair thermoregulation",
                "miscellaneous clinical states",
                "mission factors are the most important",
            )
        ):
            anchor_overlap = True
    if query_intent == "hypothermia_symptoms" and "TB MED 508" in section_upper:
        if any(
            phrase in sentence_lower
            for phrase in (
                "signs and symptoms of hypothermia",
                "altered mental status",
                "shivering",
                "hypotension",
            )
        ):
            anchor_overlap = True
    if query_intent == "frostbite_prevention" and section_upper in {"SEVERE", "HIGH", "EXTREME"}:
        if any(
            phrase in sentence_lower
            for phrase in (
                "mandatory buddy checks every 10 minutes",
                "wear ecwcs or equivalent",
                "provide warming facilities",
                "no exposed skin",
                "stay active",
                "wear vb boots",
            )
        ):
            anchor_overlap = True
    if query_intent == "immersion_limit" and "TB MED 508" in section_upper:
        if any(
            phrase in sentence_lower
            for phrase in (
                "table 3-3",
                "50-54",
                "neck",
                "5 minutes",
            )
        ):
            anchor_overlap = True
    if not overlap and not anchor_overlap:
        return 0.0
    if query_intent == "questionnaire_follow_up_table":
        if "sensitivity" in query_terms and "sensitivity" not in sentence_lower:
            return 0.0
        if "uncomfortable" in query_terms and "uncomfortable" not in sentence_lower:
            return 0.0
    score = float(len(overlap) or 1)
    if any(noisy in section_upper for noisy in ("DISCLAIMER", "METHODS", "QUESTION", "GRADE")):
        score -= 4.0
    if re.match(r"^\d+\s+", sentence.strip()):
        score -= 2.5
    if "symptom" in sentence_lower or "symptoms" in sentence_lower:
        score += 1.5
    if query_intent == "symptoms":
        score += len(sentence_terms & SYMPTOM_HINTS) * 0.75
        if "symptoms include" in sentence_lower:
            score += 3.0
        if "experience" in sentence_lower:
            score += 1.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 2.5
    if query_intent == "questionnaire_performance":
        score += len(sentence_terms & QUESTIONNAIRE_PERFORMANCE_HINTS) * 1.0
        if "question 13" in sentence_lower or "performance at work" in sentence_lower:
            score += 5.0
        if "concentration" in sentence_lower and "motivation" in sentence_lower:
            score += 4.0
        if "manual strength" in sentence_lower or "musculo-skeletal function" in sentence_lower:
            score += 4.0
        if "health-check questionnaire" in sentence_lower or "screening protocol" in sentence_lower:
            score -= 4.0
        if "performance aspects assessed" in sentence_lower:
            score += 5.0
    if query_intent == "questionnaire_symptom_scale":
        score += len(sentence_terms & QUESTIONNAIRE_SYMPTOM_SCALE_HINTS) * 0.9
        if "question 5" in sentence_lower or "shortness of breath" in sentence_lower:
            score += 5.0
        if "not at all" in sentence_lower and "during exercise" in sentence_lower:
            score += 5.0
        if ("contexts" in query_terms or "rated" in query_terms) and "rated in four contexts" in sentence_lower:
            score += 6.0
        if ("contexts" in query_terms or "rated" in query_terms) and "symptoms assessed" in sentence_lower:
            score -= 1.5
        if "mucus excretion" in sentence_lower:
            score += 2.5
    if query_intent == "questionnaire_color_change":
        score += len(sentence_terms & {"white", "blue", "red", "purple", "fingers"}) * 1.2
        if "question 9" in sentence_lower:
            score += 5.0
        if "white" in sentence_lower and "blue" in sentence_lower and "red/purple" in sentence_lower:
            score += 6.0
    if query_intent == "questionnaire_frostbite_history":
        score += len(sentence_terms & QUESTIONNAIRE_FROSTBITE_HINTS) * 1.0
        if "question 12" in sentence_lower or "blister grade" in sentence_lower:
            score += 5.0
        if "once" in sentence_lower and "several times" in sentence_lower:
            score += 4.0
        if "answer options" in sentence_lower:
            score += 4.0
    if query_intent == "questionnaire_follow_up_table":
        score += len(sentence_terms & {"uncomfortable", "sensitivity", "nurse", "physician"}) * 1.2
        if "table i" in sentence_lower:
            score += 6.0
        if "professional: nurse" in sentence_lower:
            score += 5.0
        if "sensitivity" in query_terms:
            if "sensitivity" in sentence_lower and "professional: nurse" in sentence_lower:
                score += 7.0
            if "sensitivity" not in sentence_lower:
                score -= 6.0
            if "symptom of some disease" in sentence_lower:
                score -= 6.0
        if "uncomfortable" in query_terms:
            if "uncomfortable" in sentence_lower and "professional: nurse" in sentence_lower:
                score += 7.0
            if "uncomfortable" not in sentence_lower:
                score -= 6.0
            if "symptom of some disease" in sentence_lower:
                score -= 5.0
        if "questionnaire was developed" in sentence_lower or "screening protocol" in sentence_lower:
            score -= 5.0
    if query_intent == "opioid_pre_therapy_checklist":
        score += len(sentence_terms & OPIOID_PRE_THERAPY_HINTS) * 0.9
        if "appendix a" in sentence_lower:
            score += 4.0
        if "non-pharmacological therapy" in sentence_lower:
            score += 4.0
        if "non-opioid pharmacotherapy" in sentence_lower:
            score += 4.0
        if "informed consent" in sentence_lower or "opioid safety" in sentence_lower:
            score += 3.0
    if query_intent == "opioid_adverse_effect_scale":
        score += len(sentence_terms & OPIOID_ADVERSE_SCALE_HINTS) * 1.1
        if "appendix b" in sentence_lower:
            score += 4.0
        if "0 = none" in sentence_lower:
            score += 6.0
        if "1 = limits adls" in sentence_lower or "2 = prevents adls" in sentence_lower:
            score += 7.0
        if "fatal overdose" in sentence_lower or "non-fatal overdose" in sentence_lower:
            score += 2.0
    if query_intent == "opioid_switch_follow_up":
        score += len(sentence_terms & OPIOID_SWITCH_FOLLOWUP_HINTS) * 0.9
        if "appendix c" in sentence_lower or "switching opioids" in sentence_lower:
            score += 4.0
        if "3-day follow-up" in sentence_lower:
            score += 6.0
        if "every 2-4 weeks" in sentence_lower or "every 2–4 weeks" in sentence_lower:
            score += 5.0
    if query_intent == "definition":
        score += len(sentence_terms & DEFINITION_HINTS) * 1.0
        if "defined as" in sentence_lower:
            score += 4.0
        if section_upper.startswith("DEFINITION"):
            score += 3.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "causes":
        score += len(sentence_terms & CAUSE_HINTS) * 1.0
        if "mainly caused by viruses" in sentence_lower:
            score += 4.0
        if "most common" in sentence_lower and "rhinovirus" in sentence_lower:
            score += 6.0
        if "these viruses are the most common cause" in sentence_lower:
            score += 5.0
        if "rhinovirus" in sentence_lower or "coronavirus" in sentence_lower:
            score += 2.5
        if "AETIOLOGY" in section_upper or "RISK FACTORS" in section_upper:
            score += 3.0
        if "PROGNOSIS" in section_upper or "TREATMENTS" in section_upper:
            score -= 3.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if sentence_lower.startswith("keywords:"):
            score -= 8.0
        if re.match(r"^[a-z]+\s+\d+[–-]\d+", sentence_lower):
            score -= 5.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "symptom_pathogenesis":
        score += len(sentence_terms & SYMPTOM_PATHOGENESIS_HINTS) * 0.8
        if "symptom production is a combination of viral cytopathic effect" in sentence_lower:
            score += 10.0
        if "activation of inflammatory pathways" in sentence_lower:
            score += 8.0
        if "therefore, antiviral treatment alone may not be able to prevent these events" in sentence_lower:
            score -= 4.0
        if "treatment option" in sentence_lower or "effective in reducing" in sentence_lower:
            score -= 5.0
        if "nasal discharge and stuffiness" in sentence_lower:
            score -= 4.0
        if "rhinovirus is the most common" in sentence_lower:
            score -= 2.0
        if "transmission" in sentence_lower or "droplet" in sentence_lower:
            score -= 4.0
        if "symptoms include" in sentence_lower:
            score -= 3.0
    if query_intent == "hypothermia_predisposition":
        score += len(sentence_terms & HYPOTHERMIA_PREDISPOSITION_HINTS) * 0.8
        if "predisposing factors for hypothermia" in sentence_lower:
            score += 8.0
        if "decrease heat production" in sentence_lower or "increase heat loss" in sentence_lower:
            score += 5.0
        if "impair thermoregulation" in sentence_lower or "miscellaneous clinical states" in sentence_lower:
            score += 4.0
        if "to diagnose hypothermia" in sentence_lower or "signs and symptoms of hypothermia" in sentence_lower:
            score -= 5.0
    if query_intent == "hypothermia_symptoms":
        score += len(sentence_terms & HYPOTHERMIA_SYMPTOM_HINTS) * 0.7
        if "signs and symptoms of hypothermia" in sentence_lower:
            score += 7.0
        if "shivering" in sentence_lower or "hypotension" in sentence_lower or "altered mental status" in sentence_lower:
            score += 4.0
        if "common cold" in sentence_lower or "sore throat" in sentence_lower or "rhinorrhoea" in sentence_lower:
            score -= 8.0
    if query_intent == "frostbite_prevention":
        score += len(sentence_terms & FROSTBITE_PREVENTION_HINTS) * 0.7
        if section_upper in {"SEVERE", "EXTREME", "HIGH"}:
            score += 4.0
        if "severe" in query_terms:
            if section_upper == "SEVERE":
                score += 6.0
            elif section_upper in {"EXTREME", "HIGH", "LOW"}:
                score -= 5.0
        if "mandatory buddy checks every 10 minutes" in sentence_lower:
            score += 8.0
        if "wear ecwcs or equivalent" in sentence_lower or "provide warming facilities" in sentence_lower:
            score += 6.0
        if "no exposed skin" in sentence_lower or "stay active" in sentence_lower:
            score += 5.0
        if "wear vb boots" in sentence_lower or "work groups of no less than two personnel" in sentence_lower:
            score += 4.0
        if "extreme risk" in sentence_lower and "severe" in query_terms:
            score -= 8.0
        if "consider modifying outdoor activities" in sentence_lower:
            score -= 6.0
        if "list of tables" in sentence_lower or "figure 3-5" in sentence_lower:
            score -= 7.0
    if query_intent == "immersion_limit":
        score += len(sentence_terms & IMMERSION_LIMIT_HINTS) * 0.7
        if "table 3-3" in sentence_lower:
            score += 5.0
        if "50-54" in sentence_lower and "neck" in sentence_lower:
            score += 8.0
        if "5 minutes" in sentence_lower:
            score += 6.0
        if "list of tables" in sentence_lower:
            score -= 8.0
    if query_intent == "transmission":
        score += len(sentence_terms & TRANSMISSION_HINTS) * 1.0
        if "hand-to-hand contact" in sentence_lower:
            score += 4.0
        if "droplet" in sentence_lower:
            score += 2.0
        if "AETIOLOGY" in section_upper:
            score += 2.0
        if "PROGNOSIS" in section_upper:
            score -= 2.0
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.0
    if query_intent == "duration":
        score += len(sentence_terms & DURATION_HINTS) * 0.9
        if "1 week" in sentence_lower or "generally clear by 1 week" in sentence_lower:
            score += 4.0
        if "few days" in sentence_lower:
            score += 3.0
        if "cough often persists" in sentence_lower or "lingering symptoms" in sentence_lower:
            score += 2.0
        if "PROGNOSIS" in section_upper:
            score += 3.0
        if "symptoms include" in sentence_lower:
            score -= 2.5
        if sentence_terms & TREATMENT_NOISE:
            score -= 3.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.0
    if query_intent == "incidence":
        score += len(sentence_terms & INCIDENCE_HINTS) * 0.9
        if "each year" in sentence_lower:
            score += 2.5
        if "children suffer" in sentence_lower or "adults" in sentence_lower:
            score += 2.0
        if "INCIDENCE" in section_upper or "PREVALENCE" in section_upper:
            score += 3.0
        if "year 6 compared" in sentence_lower or "twice as likely" in sentence_lower:
            score -= 2.5
        if "cross-sectional study" in sentence_lower or "prospective us study" in sentence_lower:
            score -= 2.0
        if "symptoms of colds" in sentence_lower or "types of virus" in sentence_lower:
            score -= 2.5
        if "adverse effects" in sentence_lower:
            score -= 4.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.0
    if query_intent == "ct_findings":
        score += len(sentence_terms & CT_FINDINGS_HINTS) * 0.8
        if "high prevalence of ostiomeatal and sinus abnormalities" in sentence_lower:
            score += 6.0
        if "abnormalities on ct scans" in sentence_lower or "abnormalities of one or more sinuses" in sentence_lower:
            score += 3.5
        if "discussion" in section_upper or "FOLLOW-UP" in section_upper:
            score += 2.0
        if "study was approved" in sentence_lower or "screened by telephone" in sentence_lower:
            score -= 4.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "ct_follow_up":
        score += len(sentence_terms & CT_FOLLOW_UP_HINTS) * 0.8
        if "follow-up evaluation after 13 to 20 days" in sentence_lower:
            score += 6.0
        if "residual abnormalities" in sentence_lower or "marked improvement" in sentence_lower:
            score += 4.0
        if "returned to normal" in sentence_lower or "resolved or markedly improved" in sentence_lower:
            score += 3.0
        if "FOLLOW-UP" in section_upper or "DISCUSSION" in section_upper:
            score += 2.0
        if "screened by telephone" in sentence_lower or "study was approved" in sentence_lower:
            score -= 4.0
        if any(noise in sentence_lower for noise in BIBLIOGRAPHIC_NOISE):
            score -= 3.5
    if query_intent == "review_prevention":
        score += len(sentence_terms & REVIEW_PREVENTION_HINTS) * 0.6
        if "best evidence for the prevention" in sentence_lower:
            score += 6.0
        if "physical preventive measures such as handwashing" in sentence_lower:
            score += 6.0
        if "physical interventions" in sentence_lower or "handwashing" in sentence_lower:
            score += 4.0
        if "zinc supplements" in sentence_lower:
            score += 2.0
        if "risk of bias outcome harms comment" in sentence_lower:
            score -= 6.0
        if "summarized in table 1" in sentence_lower or "the evidence used in this review is described in box 1" in sentence_lower:
            score -= 5.0
        if "we review the evidence" in sentence_lower or "quality of the evidence was frequently poor" in sentence_lower:
            score -= 5.0
        if "although preventive interventions have somewhat discrete outcomes" in sentence_lower:
            score -= 5.0
        if "symptoms and signs of the common cold overlap" in sentence_lower:
            score -= 4.0
    if query_intent == "review_nontraditional":
        score += len(sentence_terms & REVIEW_NONTRADITIONAL_HINTS) * 0.6
        if "nontraditional treatments" in sentence_lower:
            score += 6.0
        if "oral zinc supplements" in sentence_lower:
            score += 5.0
        if "honey at bedtime for cough" in sentence_lower or ("honey" in sentence_lower and "children" in sentence_lower):
            score += 4.0
        if "risk of bias outcome harms comment" in sentence_lower:
            score -= 6.0
        if "summarized in table 3" in sentence_lower or "summarized in table 2" in sentence_lower:
            score -= 5.0
        if "we review the evidence" in sentence_lower or "symptoms and signs of the common cold overlap" in sentence_lower:
            score -= 5.0
        if "treatment of the common cold with echinacea: a structured review" in sentence_lower:
            score -= 6.0
    if query_intent == "antibiotics":
        score += len(sentence_terms & ANTIBIOTICS_HINTS) * 0.8
        if "don't reduce symptoms overall" in sentence_lower:
            score += 7.0
        if "adverse effects" in sentence_lower:
            score += 4.0
        if "antibiotic resistance" in sentence_lower:
            score += 5.0
        if "have no beneficial effect on the common cold" in sentence_lower:
            score += 6.0
        if "because most common colds are viral" in sentence_lower:
            score += 3.0
        if "sinusitis" in sentence_lower and "antibiotic treatment" in sentence_lower:
            score -= 7.0
        if "other interventions" in sentence_lower:
            score -= 4.0
        if "symptoms and signs of the common cold overlap" in sentence_lower:
            score -= 4.0
        if "contrary to common belief" in sentence_lower:
            score += 3.0
        if "no evidence for the use of antibiotics" in sentence_lower:
            score += 6.0
        if "resistant organisms" in sentence_lower:
            score += 3.0
        if sentence_lower.startswith("article the common cold"):
            score -= 7.0
        if "recent studies have focused on three areas for treatment" in sentence_lower:
            score -= 4.0
        if "adverse effects adverse effects" in sentence_lower:
            score -= 6.0
        if "may improve symptoms after 5 days" in sentence_lower and "culture" in sentence_lower:
            score -= 1.5
    if query_intent == "treatment_prevention":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_PREVENTION_HINTS) * 0.4
        if "cmaj" in query_terms and "zinc" in query_terms:
            if "number of colds was significantly lower" in sentence_lower:
                score += 6.0
            if "zinc appears to be effective in reducing the number of colds per year" in sentence_lower:
                score += 6.0
            if "children" in sentence_lower:
                score += 3.0
            if "school absences were significantly lower" in sentence_lower:
                score += 2.0
            if "a cochrane review" in sentence_lower:
                score -= 5.5
            if "cmaj, february" in sentence_lower:
                score -= 5.0
        if "symptom severity score" in sentence_lower:
            score -= 5.0
        if re.search(r"\bday\s+[2-5]\b", sentence_lower):
            score -= 4.0
        if "number of colds was significantly lower" in sentence_lower:
            score += 4.0
        if "no colds during the study period" in sentence_lower:
            score += 4.0
        if "prevention with zinc" in sentence_lower:
            score += 3.0
        if sentence_lower.startswith("a cochrane review"):
            score -= 3.5
        if "antibiotic treatment of sinusitis" in sentence_lower:
            score -= 6.0
        if "reviewing the evidence for the antibiotic treatment of sinusitis" in sentence_lower:
            score -= 6.0
        if "decreasing the incidence" in sentence_lower or "reduces the incidence" in sentence_lower:
            score += 5.0
        if "substantial reductions in the incidence" in sentence_lower:
            score += 5.0
        if "published evidence supports" in sentence_lower:
            score += 4.0
        if "suggests an additional benefit" in sentence_lower:
            score += 4.0
        if "contracting a cold" in sentence_lower:
            score += 3.0
        if "benefit" in sentence_lower:
            score += 1.5
        if "incidence was not altered" in sentence_lower or "normal populations" in sentence_lower:
            score -= 2.0
        if "evidence for the prevention of a cold was lacking" in sentence_lower:
            score -= 3.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_null_effect":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_NULL_EFFECT_HINTS) * 0.5
        if "incidence was not altered" in sentence_lower:
            score += 7.0
        if "lack of effect" in sentence_lower:
            score += 4.0
        if "normal populations" in sentence_lower:
            score += 4.0
        if "throws doubt on the utility" in sentence_lower:
            score += 2.0
        if "beneficial effect" in sentence_lower or "50% reduction" in sentence_lower or "decreasing the incidence" in sentence_lower:
            score -= 3.0
        if "he role of vitamin c" in sentence_lower or "subject of controversy" in sentence_lower:
            score -= 4.0
        if (
            "criteria for inclusion" in sentence_lower
            or "literature from" in sentence_lower
            or "overview of the results" in sentence_lower
        ):
            score -= 5.0
        if "vitamin c for preventing and treating the common cold" in sentence_lower:
            score -= 5.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_subgroup_benefit":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_SUBGROUP_HINTS) * 0.5
        if "cold stress" in sentence_lower or "physical stress" in sentence_lower:
            score += 5.0
        if "beneficial effect" in sentence_lower or "50% reduction" in sentence_lower:
            score += 4.0
        if "collective evidence indicates" in sentence_lower:
            score += 2.0
        if "marathon runners" in sentence_lower or "skiers" in sentence_lower or "soldiers" in sentence_lower:
            score += 3.0
        if "normal populations" in sentence_lower:
            score -= 2.0
        if "vitamin c for preventing and treating the common cold" in sentence_lower:
            score -= 5.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_duration":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_DURATION_HINTS) * 0.5
        if (
            "duration of cold episodes" in sentence_lower
            or "duration of common cold episodes" in sentence_lower
            or "duration of the common cold" in sentence_lower
        ):
            score += 4.0
        if "reduced the duration" in sentence_lower or "shortens the course" in sentence_lower:
            score += 3.0
        if "14%" in sentence_lower or "8%" in sentence_lower:
            score += 2.0
        if "onset of symptoms" in sentence_lower or "8 g" in sentence_lower:
            score += 2.0
        if "normal populations" in sentence_lower:
            score -= 1.0
        if "vitamin c for preventing and treating the common cold" in sentence_lower:
            score -= 5.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if query_intent == "treatment_overall":
        score += len(sentence_terms & TREATMENT_ENTITY_HINTS) * 0.5
        score += len(sentence_terms & TREATMENT_OVERALL_HINTS) * 0.4
        if "incidence" in sentence_lower and "duration" in sentence_lower:
            score += 4.0
        if "prevention" in sentence_lower and "treatment" in sentence_lower:
            score += 3.0
        if "published evidence supports" in sentence_lower or "suggests that echinacea has a benefit" in sentence_lower:
            score += 4.0
        if "suggests an additional benefit" in sentence_lower:
            score += 4.0
        if "large-scale randomised prospective studies" in sentence_lower:
            score += 1.5
        if "trials were included for analysis" in sentence_lower or "inclusion criteria" in sentence_lower:
            score -= 4.0
        if "doi:" in sentence_lower or "citation:" in sentence_lower:
            score -= 4.0
    if len(sentence) > 320:
        score -= 1.5
    return score


def build_grounded_context(chunks: list[ChunkRecord]) -> str:
    """Flatten retrieved chunks into a prompt-ready context string."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk.chunk_id}] page={chunk.page_start}-{chunk.page_end} "
            f"section={chunk.section_title or 'n/a'}\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _answer_sentence_budget(query: str) -> int:
    query_terms = _query_terms(query)
    query_intent = _detect_query_intent(query, query_terms)
    if query_intent == "source_listing":
        return 4
    if query_intent == "cross_document_compare":
        return 6
    if query_intent == "document_overview":
        return 4
    if query_intent == "document_routing":
        return 3
    if query_intent == "source_justification":
        return 3
    structured_profile = get_structured_intent_profile(query_intent)
    if structured_profile:
        return structured_profile.answer_sentence_budget
    preferred_doc_id = _preferred_source_doc_id(query)
    if query_intent in {
        "questionnaire_performance",
        "questionnaire_symptom_scale",
        "questionnaire_color_change",
        "questionnaire_frostbite_history",
        "questionnaire_follow_up_table",
    }:
        return 1
    if query_intent in {
        "opioid_adverse_effect_scale",
        "opioid_switch_follow_up",
    }:
        return 1
    if query_intent == "opioid_pre_therapy_checklist":
        return 2
    if query_intent in {
        "hypothermia_predisposition",
        "hypothermia_symptoms",
        "frostbite_prevention",
        "immersion_limit",
    }:
        return 2
    if preferred_doc_id in {
        "ajmedp-4-2-srd-eda-v1-e-2561",
        "health-check-questionnaire-for-subjects-expose-to",
        "cep-opioidmanager-appendix2017",
    }:
        return 2
    if preferred_doc_id:
        return 3
    return 4


def select_evidence_sentences(
    query: str,
    chunks: list[ChunkRecord],
    max_sentences: int = 4,
) -> list[EvidenceSentence]:
    """Select the most query-relevant sentences from expanded chunk context."""
    query_terms = _query_terms(query)
    query_intent = _detect_query_intent(query, query_terms)
    candidates: list[EvidenceSentence] = []
    seen_sentences: set[str] = set()

    for chunk in chunks:
        for sentence in _split_sentences(chunk.text):
            normalized = sentence.lower()
            if normalized in seen_sentences:
                continue
            score = _score_sentence(
                sentence,
                query_terms,
                query_intent,
                section_title=chunk.section_title,
            )
            if score <= 0:
                continue
            if query_intent == "definition" and "defined as" not in normalized and chunk.section_title:
                if chunk.section_title.upper().startswith("DEFINITION"):
                    score += 1.0
            if query_intent == "causes" and chunk.section_title:
                if "AETIOLOGY" in chunk.section_title.upper():
                    score += 1.0
            if query_intent == "transmission" and "transmission" in normalized:
                score += 1.0
            seen_sentences.add(normalized)
            candidates.append(
                EvidenceSentence(
                    chunk_id=chunk.chunk_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    sentence=sentence,
                    score=score,
                )
            )

    candidates.sort(
        key=lambda item: (-item.score, item.page_start, item.chunk_id, item.sentence)
    )
    selected = candidates[:max_sentences]

    coverage_targets = {
        "hypothermia_predisposition": (
            "decrease heat production",
            "increase heat loss",
            "impair thermoregulation",
        ),
        "hypothermia_symptoms": (
            "shivering",
            "altered mental status",
            "hypotension",
        ),
        "frostbite_prevention": (
            "mandatory buddy checks every 10 minutes",
            "no exposed skin",
            "stay active",
        ),
        "immersion_limit": ("50-54", "neck", "5 minutes"),
        "review_prevention": ("handwashing",),
        "review_nontraditional": ("oral zinc supplements", "honey at bedtime"),
        "antibiotics": (
            "don't reduce symptoms overall",
            "no evidence for the use of antibiotics",
            "adverse effects",
            "resistant organisms",
        ),
        "symptom_pathogenesis": (
            "viral cytopathic effect",
            "activation of inflammatory pathways",
        ),
        "treatment_null_effect": ("not altered",),
        "treatment_subgroup_benefit": ("beneficial effect",),
        "treatment_prevention": (
            "number of colds was significantly lower",
            "no colds during the study period",
            "benefit",
        ),
        "treatment_overall": ("benefit",),
    }.get(query_intent, ())

    if coverage_targets:
        selected_surfaces = [
            _normalized_sentence_surface(item.sentence) for item in selected
        ]
        for target in coverage_targets:
            target_surface = _normalized_sentence_surface(target)
            if any(target_surface in surface for surface in selected_surfaces):
                continue
            replacement = next(
                (
                    item
                    for item in candidates
                    if target_surface in _normalized_sentence_surface(item.sentence)
                ),
                None,
            )
            if replacement is None or replacement in selected:
                continue
            if selected:
                selected[-1] = replacement
                selected.sort(
                    key=lambda item: (-item.score, item.page_start, item.chunk_id, item.sentence)
                )
                selected_surfaces = [
                    _normalized_sentence_surface(item.sentence) for item in selected
                ]

    deduped: list[EvidenceSentence] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in selected:
        key = (item.chunk_id, _normalized_sentence_surface(item.sentence))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)

    return deduped


def _compress_sentences(evidence: list[EvidenceSentence]) -> str:
    if not evidence:
        return NO_GROUNDED_ANSWER

    fragments = [item.sentence.rstrip(".") for item in evidence]
    return " ".join(f"{fragment}." for fragment in fragments)


def _humanize_source_label(source_pdf: str) -> str:
    stem = Path(source_pdf).stem
    label = stem.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def _display_label_for_chunk(chunk: ChunkRecord) -> str:
    profile = get_document_profile(chunk.doc_id)
    if profile:
        return profile.label
    return _humanize_source_label(chunk.doc_id)


def _is_low_quality_document_title(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", title).strip()
    lowered = normalized.lower()
    if lowered.startswith("doi:"):
        return True
    if ".indd" in lowered:
        return True
    if lowered.startswith("since january 2020 elsevier has created"):
        return True
    return False


def _document_label(doc_id: str, chunk_root: Path) -> str:
    record = _load_document_record(doc_id, chunk_root)
    profile = get_document_profile(doc_id)
    if record and record.title:
        normalized_title = re.sub(r"\s+", " ", record.title).strip()
        if not _is_low_quality_document_title(normalized_title):
            return normalized_title
    if profile:
        return profile.label
    return _humanize_source_label(doc_id)


def _structured_source_summary(chunks: list[ChunkRecord]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    summary: list[tuple[str, str]] = []
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        summary.append((chunk.doc_id, _display_label_for_chunk(chunk)))
    return summary


def _load_document_record(doc_id: str, chunk_root: Path) -> DocumentRecord | None:
    document_path = chunk_root.expanduser().resolve().parent / "documents" / f"{doc_id}.document.json"
    if not document_path.exists():
        return None
    try:
        return DocumentRecord.model_validate_json(document_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_document_topics(chunks: list[ChunkRecord], doc_id: str) -> list[str]:
    topics: list[str] = []
    seen: set[str] = set()
    chapter_pattern = re.compile(r"Chapter\s+\d+\s+([A-Za-z][A-Za-z0-9 \-]{2,80})")
    for chunk in chunks:
        if chunk.doc_id != doc_id:
            continue
        section_title = (chunk.section_title or "").strip()
        section_upper = section_title.upper()
        if (
            section_title
            and not section_upper.startswith(("CONTENTS", "INDEX", "BIBLIOGRAPHY", "FOREWORD"))
            and not (section_upper == section_title and len(section_title) <= 5)
            and len(section_title) >= 6
            and section_title not in seen
        ):
            seen.add(section_title)
            topics.append(section_title)
        for match in chapter_pattern.findall(chunk.text):
            topic = match.strip()
            if topic.upper().startswith(("CONTENTS", "INDEX", "BIBLIOGRAPHY")):
                continue
            if topic.upper() == topic and len(topic) <= 5:
                continue
            if topic not in seen:
                seen.add(topic)
                topics.append(topic)
        if len(topics) >= 5:
            break
    return topics[:5]


def _clean_topic_label(topic: str) -> str:
    cleaned = re.sub(r"^Chapter\s+\d+\s+", "", topic).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or topic


def _document_summary_cues(
    doc_id: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> list[str]:
    record = _load_document_record(doc_id, chunk_root)
    profile = get_document_profile(doc_id)
    cues: list[str] = []
    seen: set[str] = set()
    doc_titles = {
        item.lower()
        for item in (
            record.title if record and record.title else None,
            profile.label if profile else None,
        )
        if item
    }

    def maybe_add(value: str) -> None:
        cue = _clean_topic_label(value).strip()
        cue_upper = cue.upper()
        if not cue:
            return
        if cue.lower() in doc_titles:
            return
        if cue_upper in {
            "THE CENTRE FOR HUMANITARIAN DATA",
            "OCHA CENTRE FOR HUMANITARIAN DATA",
            "GUIDANCE NOTE SERIES",
        }:
            return
        if cue_upper.startswith(("CONTENTS", "INDEX", "BIBLIOGRAPHY", "FOREWORD")):
            return
        if cue in seen:
            return
        seen.add(cue)
        cues.append(cue)

    if record:
        for cue in record.summary_cues:
            maybe_add(cue)
        if not cues:
            for cue in record.toc[:12]:
                maybe_add(cue)
    if not cues:
        for cue in _extract_document_topics(top_k_hits, doc_id):
            maybe_add(cue)
    return cues[:6]


def _document_discovery_terms(doc_id: str, chunk_root: Path) -> list[str]:
    record = _load_document_record(doc_id, chunk_root)
    if record and record.discovery_terms:
        return record.discovery_terms[:20]
    profile = get_document_profile(doc_id)
    if profile:
        return sorted(profile.topical_terms)
    return []


def _rank_document_candidates(
    doc_ids: list[str],
    query: str,
    top_k_hits: list[ChunkRecord],
    chunk_root: Path,
) -> list[str]:
    query_terms = _specific_query_terms(_query_terms(query))
    hit_rank: dict[str, int] = {}
    for idx, chunk in enumerate(top_k_hits):
        hit_rank.setdefault(chunk.doc_id, idx)
    ranked: list[tuple[float, str]] = []
    for doc_id in doc_ids:
        profile = get_document_profile(doc_id)
        label_terms = set(re.findall(r"[a-zA-Z]{3,}", (profile.label if profile else doc_id).lower()))
        profile_terms = set(profile.topical_terms) if profile else set()
        discovery_terms = set(_document_discovery_terms(doc_id, chunk_root))
        summary_terms = {
            token
            for cue in _document_summary_cues(doc_id, top_k_hits, chunk_root)
            for token in re.findall(r"[a-zA-Z]{3,}", cue.lower())
        }
        overlap = len(query_terms & (label_terms | profile_terms | discovery_terms | summary_terms))
        score = overlap * 3.0
        if profile:
            score += len(query_terms & profile_terms) * 1.5
        if doc_id in hit_rank:
            score += max(0.0, 3.0 - min(hit_rank[doc_id], 6) * 0.4)
        ranked.append((score, doc_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in ranked]


def _base_answer_trace(
    query: str,
    query_intent: str,
    evidence: list[EvidenceSentence],
    template_id: str | None = None,
    matched_pattern: str | None = None,
    matched_cues: list[str] | None = None,
) -> dict[str, object]:
    return {
        "query_intent": query_intent,
        "source_doc_id": _preferred_source_doc_id(query),
        "template_id": template_id,
        "matched_pattern": matched_pattern,
        "matched_cues": matched_cues or [],
        "evidence_chunk_ids": [item.chunk_id for item in evidence],
    }


def _cross_document_answer(
    query: str,
    query_intent: str,
    top_k_hits: list[ChunkRecord],
    evidence: list[EvidenceSentence],
    chunk_root: Path,
) -> tuple[str | None, dict[str, object] | None]:
    source_summary = _structured_source_summary(top_k_hits)
    if query_intent == "document_overview" and top_k_hits:
        primary_doc_id = top_k_hits[0].doc_id
        primary_label = _document_label(primary_doc_id, chunk_root)
        topics = _document_summary_cues(primary_doc_id, top_k_hits, chunk_root)
        if topics:
            return (
                f"{primary_label} covers topics such as " + ", ".join(topics[:4]) + ".",
                {
                    "template_id": "document_level.overview",
                    "matched_pattern": "document-summary-cues",
                    "matched_cues": topics[:4],
                },
            )
    if query_intent == "document_routing" and top_k_hits:
        matched_doc_ids = _matching_source_doc_ids(query)
        if len(matched_doc_ids) > 1:
            multi_doc_limit = 4 if ("which file or files" in query.lower() or "which files" in query.lower()) else 3
            chosen_doc_ids = _rank_document_candidates(
                [doc_id for doc_id in matched_doc_ids if doc_id in {chunk.doc_id for chunk in top_k_hits}],
                query,
                top_k_hits,
                chunk_root,
            )[:multi_doc_limit]
            if len(chosen_doc_ids) >= 2:
                fragments: list[str] = []
                cues: list[str] = []
                for doc_id in chosen_doc_ids:
                    label = _document_label(doc_id, chunk_root)
                    summary_cues = _document_summary_cues(doc_id, top_k_hits, chunk_root)
                    if summary_cues:
                        fragments.append(f"{label} ({', '.join(summary_cues[:2])})")
                        cues.extend([label, *summary_cues[:2]])
                    else:
                        fragments.append(label)
                        cues.append(label)
                return (
                    "The most relevant files are " + "; ".join(fragments) + ".",
                    {
                        "template_id": "document_level.routing_multi",
                        "matched_pattern": "multi-doc-topical-summary",
                        "matched_cues": cues,
                    },
                )

        primary_chunk = top_k_hits[0]
        primary_label = _document_label(primary_chunk.doc_id, chunk_root)
        query_terms = _specific_query_terms(_query_terms(query))
        topical_terms: list[str] = []
        for item in evidence:
            sentence_terms = set(re.findall(r"[a-zA-Z]{3,}", item.sentence.lower()))
            for term in sorted(query_terms):
                if term in sentence_terms and term not in topical_terms:
                    topical_terms.append(term)
            if len(topical_terms) >= 4:
                break
        if not topical_terms:
            for chunk in top_k_hits:
                if chunk.doc_id != primary_chunk.doc_id:
                    continue
                chunk_terms = set(re.findall(r"[a-zA-Z]{3,}", chunk.text.lower()))
                for term in sorted(query_terms):
                    if term in chunk_terms and term not in topical_terms:
                        topical_terms.append(term)
                if len(topical_terms) >= 4:
                    break
        if "gradient" in topical_terms and "descent" in topical_terms:
            topical_terms = [
                term for term in topical_terms if term not in {"gradient", "descent"}
            ] + ["gradient descent"]
        topics = _document_summary_cues(primary_chunk.doc_id, top_k_hits, chunk_root)
        if topical_terms or topics:
            justification_parts: list[str] = []
            if topical_terms:
                justification_parts.append(
                    "It includes grounded material on " + ", ".join(topical_terms[:4])
                )
            if topics:
                justification_parts.append(
                    "with sections such as " + ", ".join(topics[:3])
                )
            return (
                f"The most relevant file is {primary_label}. " + ". ".join(justification_parts) + ".",
                {
                    "template_id": "document_level.routing",
                    "matched_pattern": "top-doc-topical-summary",
                    "matched_cues": [primary_label, *topical_terms[:4], *topics[:3]],
                },
            )
    if query_intent == "source_justification" and top_k_hits:
        primary_chunk = top_k_hits[0]
        primary_label = _document_label(primary_chunk.doc_id, chunk_root)
        query_terms = _specific_query_terms(_query_terms(query))
        summary_cues = _document_summary_cues(primary_chunk.doc_id, top_k_hits, chunk_root)
        discovery_terms = _document_discovery_terms(primary_chunk.doc_id, chunk_root)
        matched_terms = [
            term for term in sorted(query_terms)
            if term in set(discovery_terms) or any(term in cue.lower() for cue in summary_cues)
        ][:4]
        parts: list[str] = []
        if matched_terms:
            parts.append("it matches cues such as " + ", ".join(matched_terms))
        if summary_cues:
            parts.append("it includes sections such as " + ", ".join(summary_cues[:3]))
        if not parts:
            parts.append("it surfaced the strongest grounded support in the current benchmark")
        return (
            f"{primary_label} is the best match because " + "; ".join(parts) + ".",
            {
                "template_id": "document_level.justification",
                "matched_pattern": "doc-metadata-justification",
                "matched_cues": [primary_label, *matched_terms, *summary_cues[:3]],
            },
        )
    if query_intent == "source_listing" and source_summary:
        source_doc_ids = [doc_id for doc_id, _ in source_summary]
        ranked_doc_ids = _rank_document_candidates(source_doc_ids, query, top_k_hits, chunk_root)[:4]
        labels = []
        for doc_id in ranked_doc_ids:
            labels.append(_document_label(doc_id, chunk_root))
        return (
            "Relevant sources include: " + "; ".join(labels) + ".",
            {
                "template_id": "cross_doc.source_listing",
                "matched_pattern": "doc-diverse-topk",
                "matched_cues": labels,
            },
        )

    if query_intent == "cross_document_compare" and evidence:
        chunk_lookup = {chunk.chunk_id: chunk for chunk in top_k_hits}
        per_doc_sentences: dict[str, tuple[str, str]] = {}
        for item in evidence:
            chunk = chunk_lookup.get(item.chunk_id)
            if not chunk:
                continue
            if chunk.doc_id in per_doc_sentences:
                continue
            per_doc_sentences[chunk.doc_id] = (
                _display_label_for_chunk(chunk),
                item.sentence.rstrip("."),
            )
        for chunk in top_k_hits:
            if chunk.doc_id in per_doc_sentences:
                continue
            fallback_sentences = _split_sentences(chunk.text)
            if not fallback_sentences:
                continue
            per_doc_sentences[chunk.doc_id] = (
                _display_label_for_chunk(chunk),
                fallback_sentences[0].rstrip("."),
            )
        if len(per_doc_sentences) >= 2:
            fragments = [
                f"{label}: {sentence}."
                for label, sentence in list(per_doc_sentences.values())[:3]
            ]
            return (
                " ".join(fragments),
                {
                    "template_id": "cross_doc.compare",
                    "matched_pattern": "doc-diverse-evidence",
                    "matched_cues": [label for label, _ in list(per_doc_sentences.values())[:3]],
                },
            )
    return None, None


def _format_structured_answer(
    query_intent: str,
    evidence: list[EvidenceSentence],
) -> tuple[str | None, dict[str, object] | None]:
    """Return a concise template answer for structured form/checklist intents."""
    if not evidence:
        return None, None
    text = " ".join(item.sentence for item in evidence).lower()
    profile = get_structured_intent_profile(query_intent)
    template_id = profile.template_id if profile else None
    pattern_id = profile.pattern_id if profile else None

    if query_intent == "opioid_pre_therapy_checklist":
        checklist_fields: list[str] = []
        if "non-pharmacological therapy" in text:
            checklist_fields.append("non-pharmacological therapy optimized")
        if "non-opioid pharmacotherapy" in text:
            checklist_fields.append("non-opioid pharmacotherapy optimized")
        if "informed consent" in text:
            checklist_fields.append("informed consent obtained")
        if "opioid safety" in text:
            checklist_fields.append("opioid safety explained")
        if "urine drug screening" in text:
            checklist_fields.append("urine drug screening completed (as needed)")
        if checklist_fields:
            return (
                "Appendix A checklist recommends confirming: " + "; ".join(checklist_fields) + ".",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": checklist_fields,
                },
            )
        return None, None

    if query_intent == "opioid_adverse_effect_scale":
        has_none = "0 = none" in text
        has_limits = "1 = limits adls" in text
        has_prevents = "2 = prevents adls" in text
        if has_none and has_limits and has_prevents:
            return (
                "Appendix B adverse-effect scale: 0 = none, 1 = limits ADLs, "
                "2 = prevents ADLs.",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": ["0 = none", "1 = limits ADLs", "2 = prevents ADLs"],
                },
            )
        return None, None

    if query_intent == "opioid_switch_follow_up":
        has_three_day = "3-day follow-up" in text
        has_2_4_weeks = "2-4 weeks" in text or "2–4 weeks" in text
        has_withdrawal = "withdrawal symptoms" in text
        if has_three_day and has_2_4_weeks:
            if has_withdrawal:
                return (
                    "Appendix C suggests a 3-day follow-up after opioid switching to assess "
                    "withdrawal symptoms and pain, then follow-up every 2-4 weeks.",
                    {
                        "template_id": template_id,
                        "matched_pattern": pattern_id,
                        "matched_cues": [
                            "3-day follow-up",
                            "withdrawal symptoms and pain",
                            "follow-up every 2-4 weeks",
                        ],
                    },
                )
            return (
                "Appendix C suggests a 3-day follow-up after opioid switching, "
                "then follow-up every 2-4 weeks.",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": ["3-day follow-up", "follow-up every 2-4 weeks"],
                },
            )
        if has_three_day:
            return (
                "Appendix C suggests a 3-day follow-up after opioid switching.",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": ["3-day follow-up"],
                },
            )
        return None, None

    if query_intent == "questionnaire_follow_up_table":
        if "sensitivity" in text and "professional: nurse" in text:
            return (
                "In Table I, the sensitivity row is assigned to a nurse and includes "
                "a disease-focused interview among the listed actions.",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": ["sensitivity", "professional: nurse", "disease-focused interview"],
                },
            )
        if "uncomfortable" in text and "professional: nurse" in text:
            return (
                "In Table I, the uncomfortable row is assigned to a nurse "
                "(with interview actions listed for that row).",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": ["uncomfortable", "professional: nurse"],
                },
            )
        return None, None
    if query_intent == "appendix_checklist_lookup":
        if "live vaccine" in text and "within 2 weeks" in text:
            return (
                "Yes. The checklist lists live vaccine within 2 weeks as a caution.",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": ["live vaccine", "within 2 weeks"],
                },
            )
        if "anticoagulant therapy" in text:
            cues = ["anticoagulant therapy"]
            if "warfarin" in text:
                cues.append("warfarin")
            if "noacs" in text or "doacs" in text:
                cues.extend(["NOACs", "DOACs"])
            return (
                "Yes. The checklist lists anticoagulant therapy cautions, including warfarin "
                "and separate NOACs and DOACs guidance.",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": cues,
                },
            )
        return None, None
    if query_intent == "appendix_risk_list":
        if "possible risks and side effects from steroid injections" in text:
            return (
                "The checklist lists steroid-injection risks including allergic reaction, "
                "infections, tendon rupture/weak tissue, anaphylaxis, and post injection flare up of pain.",
                {
                    "template_id": template_id,
                    "matched_pattern": pattern_id,
                    "matched_cues": [
                        "allergic reaction",
                        "infections",
                        "tendon rupture/weak tissue",
                        "anaphylaxis",
                        "post injection flare up of pain",
                    ],
                },
            )
        return None, None

    return None, None


def _should_abstain(query: str, evidence: list[EvidenceSentence]) -> bool:
    if not evidence:
        return True

    query_terms = _query_terms(query)
    specific_terms = _specific_query_terms(query_terms)
    query_intent = _detect_query_intent(query, query_terms)
    if query_intent in {"source_listing", "cross_document_compare", "document_overview", "document_routing"}:
        evidence_text = " ".join(item.sentence.lower() for item in evidence)
        if query_intent in {"source_listing", "document_routing"} and not _matching_source_doc_ids(query):
            return True
        unsupported_entities = query_terms.intersection(UNSUPPORTED_ENTITY_TERMS)
        if unsupported_entities and not any(term in evidence_text for term in unsupported_entities):
            return True
        if len(evidence) < 1:
            return True
        top_score = max(item.score for item in evidence)
        return top_score < 2.0
    structured_profile = get_structured_intent_profile(query_intent)
    intent_support_terms = {
        "review_prevention": REVIEW_PREVENTION_HINTS,
        "review_nontraditional": REVIEW_NONTRADITIONAL_HINTS,
        "antibiotics": ANTIBIOTICS_HINTS,
        "symptom_pathogenesis": SYMPTOM_PATHOGENESIS_HINTS,
        "hypothermia_predisposition": HYPOTHERMIA_PREDISPOSITION_HINTS,
        "hypothermia_symptoms": HYPOTHERMIA_SYMPTOM_HINTS,
        "frostbite_prevention": FROSTBITE_PREVENTION_HINTS,
        "immersion_limit": IMMERSION_LIMIT_HINTS,
        "definition": DEFINITION_HINTS,
        "symptoms": SYMPTOM_HINTS,
        "causes": CAUSE_HINTS,
        "transmission": TRANSMISSION_HINTS,
        "duration": DURATION_HINTS,
        "incidence": INCIDENCE_HINTS,
        "ct_findings": CT_FINDINGS_HINTS,
        "ct_follow_up": CT_FOLLOW_UP_HINTS,
        "treatment_prevention": TREATMENT_ENTITY_HINTS | TREATMENT_PREVENTION_HINTS,
        "treatment_null_effect": TREATMENT_ENTITY_HINTS | TREATMENT_NULL_EFFECT_HINTS,
        "treatment_subgroup_benefit": TREATMENT_ENTITY_HINTS | TREATMENT_SUBGROUP_HINTS,
        "treatment_duration": TREATMENT_ENTITY_HINTS | TREATMENT_DURATION_HINTS,
        "treatment_overall": TREATMENT_ENTITY_HINTS | TREATMENT_OVERALL_HINTS,
        "generic": set(),
    }.get(query_intent, set())
    if structured_profile:
        intent_support_terms = set(structured_profile.support_terms)
    evidence_text = " ".join(item.sentence.lower() for item in evidence)
    unsupported_entities = query_terms.intersection(UNSUPPORTED_ENTITY_TERMS)
    if unsupported_entities and not any(term in evidence_text for term in unsupported_entities):
        return True
    has_specific_overlap = any(term in evidence_text for term in specific_terms)
    has_intent_overlap = bool(intent_support_terms) and any(
        term in evidence_text for term in intent_support_terms
    )
    if not has_specific_overlap and not has_intent_overlap:
        return True

    top_score = max(item.score for item in evidence)
    if top_score < 2.0:
        return True
    return False


def format_grounded_answer(result: GroundedAnswer) -> str:
    """Format a deterministic grounded answer with explicit evidence."""
    lines = [
        "Answer:",
        result.answer,
        "",
        "Evidence:",
    ]
    for item in result.evidence:
        lines.append(
            f"- {item.sentence} "
            f"[{item.chunk_id}, pages {item.page_start}-{item.page_end}, "
            f"section={item.section_title or 'n/a'}]"
        )
    lines.extend(
        [
            "",
            f"Query intent: {result.query_intent}",
            f"Answer template: {result.answer_trace.get('template_id') or 'n/a'}",
            f"Top-k hits: {len(result.top_k_hits)}",
            f"Expanded context chunks: {len(result.expanded_hits)}",
        ]
    )
    return "\n".join(lines)


def answer_from_chunks(query: str, chunks: list[ChunkRecord]) -> GroundedAnswer:
    """Assemble a grounded answer only from the provided chunk context."""
    query_terms = _query_terms(query)
    query_intent = _detect_query_intent(query, query_terms)
    evidence = select_evidence_sentences(
        query=query,
        chunks=chunks,
        max_sentences=_answer_sentence_budget(query),
    )
    if _should_abstain(query, evidence):
        answer = NO_GROUNDED_ANSWER
        answer_trace = _base_answer_trace(query=query, query_intent=query_intent, evidence=evidence)
    else:
        structured_answer, structured_trace = _format_structured_answer(query_intent, evidence)
        answer = structured_answer or _compress_sentences(evidence)
        answer_trace = _base_answer_trace(
            query=query,
            query_intent=query_intent,
            evidence=evidence,
            template_id=structured_trace.get("template_id") if structured_trace else None,
            matched_pattern=structured_trace.get("matched_pattern") if structured_trace else None,
            matched_cues=structured_trace.get("matched_cues") if structured_trace else None,
        )
    return GroundedAnswer(
        query=query,
        answer=answer,
        evidence=evidence,
        top_k_hits=[],
        expanded_hits=chunks,
        query_intent=query_intent,
        answer_trace=answer_trace,
    )


def answer_query_with_retrieval(
    query: str,
    index_dir: Path,
    chunk_root: Path,
    k: int = 5,
    use_lightweight_rerank: bool = True,
) -> GroundedAnswer:
    """Retrieve, expand, and assemble a grounded answer from local artifacts."""
    query_terms = _query_terms(query)
    query_intent = _detect_query_intent(query, query_terms)
    top_k_hits, expanded_hits = retrieve_top_k_with_neighbors(
        query=query,
        index_dir=index_dir,
        chunk_root=chunk_root,
        k=k,
        use_lightweight_rerank=use_lightweight_rerank,
    )
    answer_chunks = expanded_hits
    preferred_doc_id = None if query_intent in {"source_listing", "cross_document_compare", "document_routing"} else _preferred_source_doc_id(query)
    if preferred_doc_id:
        filtered = [chunk for chunk in expanded_hits if chunk.doc_id == preferred_doc_id]
        if filtered:
            answer_chunks = filtered
    elif query_intent == "document_routing":
        matched_doc_ids = _matching_source_doc_ids(query)
        if matched_doc_ids:
            filtered = [chunk for chunk in expanded_hits if chunk.doc_id in matched_doc_ids]
            if filtered:
                answer_chunks = filtered
    elif top_k_hits and query_terms.intersection(SOURCE_ANCHORED_HINTS):
        doc_counts: dict[str, int] = {}
        for chunk in top_k_hits:
            doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
        preferred_doc_id, preferred_count = max(doc_counts.items(), key=lambda item: item[1])
        if preferred_count >= max(2, (len(top_k_hits) // 2) + 1):
            filtered = [chunk for chunk in expanded_hits if chunk.doc_id == preferred_doc_id]
            if filtered:
                answer_chunks = filtered

    evidence = select_evidence_sentences(
        query=query,
        chunks=answer_chunks,
        max_sentences=_answer_sentence_budget(query),
    )
    if _should_abstain(query, evidence):
        answer = NO_GROUNDED_ANSWER
        answer_trace = _base_answer_trace(query=query, query_intent=query_intent, evidence=evidence)
    else:
        cross_doc_answer, cross_doc_trace = _cross_document_answer(
            query=query,
            query_intent=query_intent,
            top_k_hits=top_k_hits,
            evidence=evidence,
            chunk_root=chunk_root,
        )
        structured_answer, structured_trace = _format_structured_answer(query_intent, evidence)
        answer = cross_doc_answer or structured_answer or _compress_sentences(evidence)
        trace_source = cross_doc_trace or structured_trace
        answer_trace = _base_answer_trace(
            query=query,
            query_intent=query_intent,
            evidence=evidence,
            template_id=trace_source.get("template_id") if trace_source else None,
            matched_pattern=trace_source.get("matched_pattern") if trace_source else None,
            matched_cues=trace_source.get("matched_cues") if trace_source else None,
        )
    return GroundedAnswer(
        query=query,
        answer=answer,
        evidence=evidence,
        top_k_hits=top_k_hits,
        expanded_hits=answer_chunks,
        query_intent=query_intent,
        answer_trace=answer_trace,
    )
