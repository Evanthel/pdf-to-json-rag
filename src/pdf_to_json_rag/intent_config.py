"""Shared intent metadata for structured form and appendix queries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredIntentProfile:
    intent: str
    source_doc_id: str | None
    source_anchors: tuple[str, ...]
    template_id: str
    pattern_id: str
    candidate_k: tuple[int, int]
    neighbor_depth: int
    section_hints: tuple[str, ...]
    support_terms: frozenset[str]
    augment_suffix: str
    answer_sentence_budget: int = 1


SOURCE_DOC_ANCHORS = {
    "ajmedp": "ajmedp-4-2-srd-eda-v1-e-2561",
    "tb med 508": "ajmedp-4-2-srd-eda-v1-e-2561",
    "health-check questionnaire": "health-check-questionnaire-for-subjects-expose-to",
    "questionnaire for subjects exposed to cold": "health-check-questionnaire-for-subjects-expose-to",
    "subjects exposed to cold": "health-check-questionnaire-for-subjects-expose-to",
    "opioid manager appendix": "cep-opioidmanager-appendix2017",
    "opioid manager appendices": "cep-opioidmanager-appendix2017",
    "appendix a checklist": "cep-opioidmanager-appendix2017",
    "appendix b": "cep-opioidmanager-appendix2017",
    "appendix c": "cep-opioidmanager-appendix2017",
    "switching opioids": "cep-opioidmanager-appendix2017",
    "pre injection checklist": "appendix-2-examples-of-pre-injection-check-lists-final",
    "pre injection check list": "appendix-2-examples-of-pre-injection-check-lists-final",
    "appendix 2 examples of pre injection check lists": "appendix-2-examples-of-pre-injection-check-lists-final",
    "cmaj": "prevention-and-treatment-of-the-common-cold",
    "literature review": "the-common-cold-a-review-of-the-literature",
    "wat review": "the-common-cold-a-review-of-the-literature",
    "dennis wat": "the-common-cold-a-review-of-the-literature",
    "echinacea": "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold",
    "vitamin c": "vitamin-c-for-preventing-and-treating-the-common-cold",
    "ct study": "ct-study-of-the-common-cold-scanned",
}


STRUCTURED_INTENT_PROFILES = {
    "questionnaire_performance": StructuredIntentProfile(
        intent="questionnaire_performance",
        source_doc_id="health-check-questionnaire-for-subjects-expose-to",
        source_anchors=(
            "health-check questionnaire",
            "questionnaire for subjects exposed to cold",
        ),
        template_id="structured.questionnaire.performance",
        pattern_id="field-row",
        candidate_k=(6, 20),
        neighbor_depth=1,
        section_hints=(
            "HEALTH-CHECK QUESTIONNAIRE FOR SUBJECTS EXPOSED TO COLD",
            "INTERVIEW",
            "DEVELOPMENT RESULTS",
        ),
        support_terms=frozenset(
            {
                "performance",
                "concentration",
                "motivation",
                "manual",
                "strength",
                "musculo-skeletal",
                "cooling",
            }
        ),
        augment_suffix=(
            "question 13 performance at work concentration motivation manual strength "
            "musculo-skeletal function cooling symptoms"
        ),
    ),
    "questionnaire_symptom_scale": StructuredIntentProfile(
        intent="questionnaire_symptom_scale",
        source_doc_id="health-check-questionnaire-for-subjects-expose-to",
        source_anchors=(
            "health-check questionnaire",
            "questionnaire for subjects exposed to cold",
        ),
        template_id="structured.questionnaire.symptom_scale",
        pattern_id="field-row",
        candidate_k=(6, 20),
        neighbor_depth=1,
        section_hints=(
            "HEALTH-CHECK QUESTIONNAIRE FOR SUBJECTS EXPOSED TO COLD",
            "INTERVIEW",
            "DEVELOPMENT RESULTS",
        ),
        support_terms=frozenset(
            {
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
        ),
        augment_suffix=(
            "question 5 not at all in the warm in the cold during exercise shortness of "
            "breath coughing wheezing mucus excretion"
        ),
    ),
    "questionnaire_color_change": StructuredIntentProfile(
        intent="questionnaire_color_change",
        source_doc_id="health-check-questionnaire-for-subjects-expose-to",
        source_anchors=(
            "health-check questionnaire",
            "questionnaire for subjects exposed to cold",
        ),
        template_id="structured.questionnaire.color_change",
        pattern_id="field-row",
        candidate_k=(6, 20),
        neighbor_depth=1,
        section_hints=(
            "HEALTH-CHECK QUESTIONNAIRE FOR SUBJECTS EXPOSED TO COLD",
            "INTERVIEW",
        ),
        support_terms=frozenset({"white", "blue", "red/purple", "fingers", "colours", "colors"}),
        augment_suffix="question 9 fingers colors white blue red purple episodically change",
    ),
    "questionnaire_frostbite_history": StructuredIntentProfile(
        intent="questionnaire_frostbite_history",
        source_doc_id="health-check-questionnaire-for-subjects-expose-to",
        source_anchors=(
            "health-check questionnaire",
            "questionnaire for subjects exposed to cold",
        ),
        template_id="structured.questionnaire.frostbite_history",
        pattern_id="field-row",
        candidate_k=(6, 20),
        neighbor_depth=1,
        section_hints=(
            "HEALTH-CHECK QUESTIONNAIRE FOR SUBJECTS EXPOSED TO COLD",
            "INTERVIEW",
        ),
        support_terms=frozenset({"frostbite", "blister", "once", "several", "times"}),
        augment_suffix="question 12 frostbite blister grade or worse no once several times",
    ),
    "questionnaire_follow_up_table": StructuredIntentProfile(
        intent="questionnaire_follow_up_table",
        source_doc_id="health-check-questionnaire-for-subjects-expose-to",
        source_anchors=(
            "health-check questionnaire",
            "questionnaire for subjects exposed to cold",
        ),
        template_id="structured.questionnaire.follow_up_table",
        pattern_id="table-row",
        candidate_k=(6, 20),
        neighbor_depth=1,
        section_hints=("DEVELOPMENT RESULTS", "DISCUSSION"),
        support_terms=frozenset(
            {
                "table i",
                "uncomfortable",
                "sensitivity",
                "interview of working ability",
                "disease-focused interview",
                "professional",
                "nurse",
                "physician",
            }
        ),
        augment_suffix=(
            "table i uncomfortable sensitivity symptom of some disease in cold nurse "
            "physician interview of working ability disease-focused interview"
        ),
    ),
    "opioid_pre_therapy_checklist": StructuredIntentProfile(
        intent="opioid_pre_therapy_checklist",
        source_doc_id="cep-opioidmanager-appendix2017",
        source_anchors=(
            "opioid manager appendix",
            "opioid manager appendices",
            "appendix a checklist",
        ),
        template_id="structured.opioid.checklist",
        pattern_id="field-row",
        candidate_k=(6, 22),
        neighbor_depth=1,
        section_hints=("APPENDIX A", "CHECKLIST"),
        support_terms=frozenset(
            {
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
        ),
        augment_suffix=(
            "appendix a checklist optimized before opioid therapy non-pharmacological "
            "therapy non-opioid pharmacotherapy informed consent opioid safety urine drug screening"
        ),
        answer_sentence_budget=2,
    ),
    "opioid_adverse_effect_scale": StructuredIntentProfile(
        intent="opioid_adverse_effect_scale",
        source_doc_id="cep-opioidmanager-appendix2017",
        source_anchors=("opioid manager appendix", "opioid manager appendices", "appendix b"),
        template_id="structured.opioid.adverse_scale",
        pattern_id="legend-scale",
        candidate_k=(6, 22),
        neighbor_depth=1,
        section_hints=("APPENDIX B", "MONITORING"),
        support_terms=frozenset({"adverse", "effects", "adls", "none", "limits", "prevents"}),
        augment_suffix="appendix b adverse effect scale 0 none 1 limits adls 2 prevents adls",
    ),
    "opioid_switch_follow_up": StructuredIntentProfile(
        intent="opioid_switch_follow_up",
        source_doc_id="cep-opioidmanager-appendix2017",
        source_anchors=("opioid manager appendix", "opioid manager appendices", "appendix c", "switching opioids"),
        template_id="structured.opioid.switch_follow_up",
        pattern_id="follow-up-schedule",
        candidate_k=(6, 22),
        neighbor_depth=1,
        section_hints=("APPENDIX C", "SWITCHING OPIOIDS"),
        support_terms=frozenset({"switching", "follow-up", "withdrawal", "pain", "3-day", "weeks"}),
        augment_suffix=(
            "appendix c switching opioids 3-day follow-up withdrawal symptoms pain "
            "follow up every 2-4 weeks"
        ),
    ),
    "appendix_checklist_lookup": StructuredIntentProfile(
        intent="appendix_checklist_lookup",
        source_doc_id=None,
        source_anchors=(),
        template_id="structured.checklist.lookup",
        pattern_id="field-row",
        candidate_k=(6, 20),
        neighbor_depth=1,
        section_hints=("Y/N", "CHECKLIST", "CAUTIONS"),
        support_terms=frozenset(
            {
                "checklist",
                "caution",
                "cautions",
                "contra-indications",
                "contraindications",
                "live vaccine",
                "anticoagulant",
                "warfarin",
                "noacs",
                "doacs",
                "pregnancy",
                "infection",
            }
        ),
        augment_suffix=(
            "checklist caution cautions contraindications yes no live vaccine anticoagulant "
            "warfarin noacs doacs pregnancy infection"
        ),
    ),
    "appendix_risk_list": StructuredIntentProfile(
        intent="appendix_risk_list",
        source_doc_id=None,
        source_anchors=(),
        template_id="structured.checklist.risk_list",
        pattern_id="field-row",
        candidate_k=(6, 20),
        neighbor_depth=1,
        section_hints=("SECTION 5", "EXPLANATION", "RISKS", "SIDE EFFECTS"),
        support_terms=frozenset(
            {
                "checklist",
                "possible risks",
                "side effects",
                "allergic reaction",
                "infections",
                "tendon rupture",
                "anaphylaxis",
                "post injection flare",
            }
        ),
        augment_suffix=(
            "checklist possible risks side effects steroid injections allergic reaction "
            "infections tendon rupture anaphylaxis"
        ),
        answer_sentence_budget=2,
    ),
}


def get_structured_intent_profile(intent: str) -> StructuredIntentProfile | None:
    return STRUCTURED_INTENT_PROFILES.get(intent)


def preferred_source_doc_id(query: str) -> str | None:
    query_lower = query.lower()
    for anchor, doc_id in SOURCE_DOC_ANCHORS.items():
        if anchor in query_lower:
            return doc_id
    return None


def matching_source_doc_ids(query: str) -> list[str]:
    query_lower = query.lower()
    seen: list[str] = []
    for anchor, doc_id in SOURCE_DOC_ANCHORS.items():
        if anchor in query_lower and doc_id not in seen:
            seen.append(doc_id)
    return seen


def detect_structured_intent(query: str, query_terms: set[str]) -> str | None:
    query_lower = query.lower()
    if (
        "health-check questionnaire" in query_lower
        or "questionnaire for subjects exposed to cold" in query_lower
    ):
        if "question 13" in query_lower or ("performance" in query_terms and "work" in query_terms):
            return "questionnaire_performance"
        if "question 5" in query_lower or (
            {"shortness", "breath", "wheezing", "coughing", "mucus"}.intersection(query_terms)
            and ("rated" in query_terms or "contexts" in query_terms)
        ):
            return "questionnaire_symptom_scale"
        if "question 9" in query_lower or ("fingers" in query_terms and "colors" in query_terms):
            return "questionnaire_color_change"
        if "question 12" in query_lower or ("frostbite" in query_terms and "options" in query_terms):
            return "questionnaire_frostbite_history"
        if "table i" in query_lower:
            return "questionnaire_follow_up_table"
    if "opioid manager" in query_lower and "appendix a" in query_lower and (
        "optimized" in query_terms or "checklist" in query_terms
    ):
        return "opioid_pre_therapy_checklist"
    if "appendix b" in query_lower and (
        "adverse" in query_terms or "adl" in query_terms or "scores" in query_terms or "mean" in query_terms
    ):
        return "opioid_adverse_effect_scale"
    if (
        "appendix c" in query_lower
        or "switching opioids" in query_lower
    ) and ("follow-up" in query_lower or "follow" in query_terms):
        return "opioid_switch_follow_up"
    if "checklist" in query_terms and (
        ("side" in query_terms and "effects" in query_terms)
        or "risks" in query_terms
    ):
        return "appendix_risk_list"
    if "checklist" in query_terms and (
        "caution" in query_terms
        or "cautions" in query_terms
        or "contraindications" in query_terms
        or "contra" in query_terms
    ):
        return "appendix_checklist_lookup"
    return None
