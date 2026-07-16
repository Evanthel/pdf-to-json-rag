"""Shared intent metadata for structured form and appendix queries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
import re

from .config import PATHS
from .document_facets import derive_document_facets, facet_token_terms

MATCH_STOPWORDS = {
    "about",
    "action",
    "actions",
    "and",
    "are",
    "best",
    "benchmark",
    "book",
    "books",
    "compare",
    "covers",
    "data",
    "document",
    "documents",
    "file",
    "files",
    "for",
    "from",
    "guidance",
    "humanitarian",
    "impact",
    "impacts",
    "in",
    "is",
    "management",
    "model",
    "most",
    "note",
    "notes",
    "or",
    "predictive",
    "relevant",
    "report",
    "response",
    "review",
    "risk",
    "risks",
    "source",
    "sources",
    "technical",
    "that",
    "the",
    "this",
    "trigger",
    "triggers",
    "what",
    "which",
    "why",
    "with",
}


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


@dataclass(frozen=True)
class DocumentProfile:
    doc_id: str
    label: str
    aliases: tuple[str, ...]
    topical_terms: frozenset[str] = frozenset()


DOCUMENT_PROFILES = {
    "ajmedp-4-2-srd-eda-v1-e-2561": DocumentProfile(
        doc_id="ajmedp-4-2-srd-eda-v1-e-2561",
        label="AJMedP-4-2 SRD EDA V1 E 2561",
        aliases=("ajmedp", "tb med 508"),
        topical_terms=frozenset({"hypothermia", "frostbite", "immersion", "manual"}),
    ),
    "health-check-questionnaire-for-subjects-expose-to": DocumentProfile(
        doc_id="health-check-questionnaire-for-subjects-expose-to",
        label="Health-check questionnaire for subjects exposed to cold",
        aliases=(
            "health-check questionnaire",
            "questionnaire for subjects exposed to cold",
            "subjects exposed to cold",
        ),
        topical_terms=frozenset({"questionnaire", "cold", "frostbite", "performance", "symptoms"}),
    ),
    "cep-opioidmanager-appendix2017": DocumentProfile(
        doc_id="cep-opioidmanager-appendix2017",
        label="CEP OpioidManager Appendix 2017",
        aliases=(
            "opioid manager appendix",
            "opioid manager appendices",
            "appendix a checklist",
            "appendix b",
            "appendix c",
            "switching opioids",
        ),
        topical_terms=frozenset({"opioid", "checklist", "monitoring", "follow-up", "appendix"}),
    ),
    "appendix-2-examples-of-pre-injection-check-lists-final": DocumentProfile(
        doc_id="appendix-2-examples-of-pre-injection-check-lists-final",
        label="Appendix 2 examples of pre injection check lists",
        aliases=(
            "pre injection checklist",
            "pre injection check list",
            "appendix 2 examples of pre injection check lists",
        ),
        topical_terms=frozenset({"checklist", "steroid", "injection", "anticoagulant", "vaccine"}),
    ),
    "prevention-and-treatment-of-the-common-cold": DocumentProfile(
        doc_id="prevention-and-treatment-of-the-common-cold",
        label="Prevention and treatment of the common cold",
        aliases=("cmaj",),
        topical_terms=frozenset({"zinc", "honey", "handwashing", "interventions", "prevention"}),
    ),
    "the-common-cold-a-review-of-the-literature": DocumentProfile(
        doc_id="the-common-cold-a-review-of-the-literature",
        label="The common cold: a review of the literature",
        aliases=("literature review", "wat review", "dennis wat"),
        topical_terms=frozenset({"review", "rhinovirus", "antibiotics", "symptoms", "pathogenesis"}),
    ),
    "common-cold-clinincal-evidence": DocumentProfile(
        doc_id="common-cold-clinincal-evidence",
        label="Common cold clinical evidence",
        aliases=("common cold clinical evidence", "clinical evidence review"),
        topical_terms=frozenset(
            {"common", "cold", "symptoms", "sneezing", "rhinorrhoea", "runny", "nose", "sore", "throat", "cough"}
        ),
    ),
    "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold": DocumentProfile(
        doc_id="evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold",
        label="Evaluation of echinacea for the prevention and treatment of the common cold",
        aliases=("echinacea",),
        topical_terms=frozenset({"echinacea", "incidence", "duration", "prevention"}),
    ),
    "vitamin-c-for-preventing-and-treating-the-common-cold": DocumentProfile(
        doc_id="vitamin-c-for-preventing-and-treating-the-common-cold",
        label="Vitamin C for Preventing and Treating the Common Cold",
        aliases=("vitamin c",),
        topical_terms=frozenset({"vitamin", "prophylaxis", "duration", "normal", "stress"}),
    ),
    "ct-study-of-the-common-cold-scanned": DocumentProfile(
        doc_id="ct-study-of-the-common-cold-scanned",
        label="CT study of the common cold",
        aliases=("ct study",),
        topical_terms=frozenset({"ct", "sinus", "abnormalities", "follow-up"}),
    ),
    "lbdl": DocumentProfile(
        doc_id="lbdl",
        label="The Little Book of Deep Learning",
        aliases=("lbdl", "the little book of deep learning", "françois fleuret", "francois fleuret"),
        topical_terms=frozenset(
            {
                "deep",
                "learning",
                "backpropagation",
                "gradient",
                "descent",
                "transformers",
                "attention",
                "language",
                "classification",
                "denoising",
                "architectures",
            }
        ),
    ),
    "guidance-note-data-incident-management": DocumentProfile(
        doc_id="guidance-note-data-incident-management",
        label="Guidance Note: Data Incident Management",
        aliases=(
            "data incident management",
            "incident management guidance",
            "humanitarian data incident management",
        ),
        topical_terms=frozenset(
            {
                "data",
                "incident",
                "management",
                "breach",
                "response",
                "humanitarian",
                "sensitive",
                "disclosure",
            }
        ),
    ),
    "guidance-note-responsible-data-sharing-with-donors": DocumentProfile(
        doc_id="guidance-note-responsible-data-sharing-with-donors",
        label="Guidance Note: Responsible Data Sharing with Donors",
        aliases=(
            "responsible data sharing with donors",
            "data sharing with donors",
            "donor data sharing",
        ),
        topical_terms=frozenset(
            {
                "donors",
                "donor",
                "sharing",
                "responsible",
                "data",
                "financial",
                "sensitive",
                "disclosure",
            }
        ),
    ),
    "guidance-note-on-the-implications-of-cyber-threats-for-humanitarians": DocumentProfile(
        doc_id="guidance-note-on-the-implications-of-cyber-threats-for-humanitarians",
        label="Guidance Note on the Implications of Cyber Threats for Humanitarians",
        aliases=(
            "cyber threats for humanitarians",
            "cyber threats guidance",
            "humanitarian cyber threats",
        ),
        topical_terms=frozenset(
            {
                "cyber",
                "threats",
                "humanitarians",
                "humanitarian",
                "security",
                "digital",
                "risks",
                "attacks",
            }
        ),
    ),
}

SOURCE_DOC_ANCHORS = {
    alias: profile.doc_id
    for profile in DOCUMENT_PROFILES.values()
    for alias in profile.aliases
}


@lru_cache(maxsize=1)
def _document_metadata_index() -> dict[str, dict[str, object]]:
    documents_dir = PATHS.data_documents
    index: dict[str, dict[str, object]] = {}
    if not documents_dir.exists():
        return index
    for path in documents_dir.glob("*.document.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_id = payload.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        title = payload.get("title") if isinstance(payload.get("title"), str) else ""
        summary_cues = [item for item in payload.get("summary_cues", []) if isinstance(item, str)]
        discovery_terms = [item for item in payload.get("discovery_terms", []) if isinstance(item, str)]
        derived_facets = derive_document_facets(
            source_pdf=payload.get("source_pdf", ""),
            title=title,
            toc=[item for item in payload.get("toc", []) if isinstance(item, str)],
            summary_cues=summary_cues,
            leading_block_lines=[],
            metadata_values=[],
            page_count=payload.get("page_count", 0) if isinstance(payload.get("page_count"), int) else 0,
        )
        facet_terms = [item for item in payload.get("facet_terms", []) if isinstance(item, str)] or list(
            derived_facets["facet_terms"]
        )
        index[doc_id] = {
            "title": title,
            "summary_cues": summary_cues,
            "discovery_terms": discovery_terms,
            "document_type": payload.get("document_type") if isinstance(payload.get("document_type"), str) else derived_facets["document_type"],
            "document_purpose": payload.get("document_purpose") if isinstance(payload.get("document_purpose"), str) else derived_facets["document_purpose"],
            "audience": payload.get("audience") if isinstance(payload.get("audience"), str) else derived_facets["audience"],
            "evidence_style": payload.get("evidence_style") if isinstance(payload.get("evidence_style"), str) else derived_facets["evidence_style"],
            "structure_style": payload.get("structure_style") if isinstance(payload.get("structure_style"), str) else derived_facets["structure_style"],
            "facet_terms": facet_terms,
        }
    return index


def _metadata_anchor_matches(query_lower: str) -> list[str]:
    matches: list[str] = []
    for doc_id, meta in _document_metadata_index().items():
        phrases = []
        title = meta.get("title") or ""
        if title:
            phrases.append(title.lower())
        phrases.extend(cue.lower() for cue in meta.get("summary_cues", [])[:3])
        for phrase in phrases:
            compact = re.sub(r"\s+", " ", phrase).strip()
            if len(compact) >= 12 and compact in query_lower and doc_id not in matches:
                matches.append(doc_id)
                break
    return matches


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
        answer_sentence_budget=3,
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
    "opioid_med_legend": StructuredIntentProfile(
        intent="opioid_med_legend",
        source_doc_id="cep-opioidmanager-appendix2017",
        source_anchors=("opioid manager appendix", "opioid manager appendices", "appendix b"),
        template_id="structured.opioid.med_legend",
        pattern_id="legend-scale",
        candidate_k=(6, 22),
        neighbor_depth=1,
        section_hints=("APPENDIX B", "MED"),
        support_terms=frozenset({"med", "morphine", "equivalent", "dose"}),
        augment_suffix="appendix b med morphine equivalent dose legend chart",
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
        answer_sentence_budget=2,
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


def get_document_profile(doc_id: str) -> DocumentProfile | None:
    return DOCUMENT_PROFILES.get(doc_id)


def preferred_source_doc_id(query: str, allow_topical: bool = False) -> str | None:
    matches = matching_source_doc_ids(query, allow_topical=allow_topical)
    return matches[0] if matches else None


def _filtered_terms(text: str, min_len: int = 2) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z]{%d,}" % min_len, text.lower())
        if term not in MATCH_STOPWORDS
    }


def matching_source_doc_ids(query: str, allow_topical: bool = False) -> list[str]:
    query_lower = query.lower()
    query_terms = _filtered_terms(query_lower, min_len=2)
    query_terms |= {
        term[:-1]
        for term in list(query_terms)
        if len(term) > 4 and term.endswith("s")
    }
    seen: list[str] = []
    for anchor, doc_id in SOURCE_DOC_ANCHORS.items():
        if anchor in query_lower and doc_id not in seen:
            seen.append(doc_id)
    for doc_id in _metadata_anchor_matches(query_lower):
        if doc_id not in seen:
            seen.append(doc_id)
    if seen or not allow_topical:
        return seen

    topical_matches: list[tuple[int, str]] = []
    candidate_doc_ids = set(DOCUMENT_PROFILES.keys()) | set(_document_metadata_index().keys())
    for doc_id in candidate_doc_ids:
        profile = DOCUMENT_PROFILES.get(doc_id)
        meta = _document_metadata_index().get(doc_id, {})
        profile_terms = (
            {term for term in profile.topical_terms if term not in MATCH_STOPWORDS}
            if profile
            else set()
        )
        title = str(meta.get("title", "")) or (profile.label if profile else "")
        title_terms = _filtered_terms(title, min_len=3)
        cue_terms = {
            token
            for cue in meta.get("summary_cues", [])[:6]
            for token in _filtered_terms(cue, min_len=3)
        }
        discovery_terms = {
            str(term).lower()
            for term in meta.get("discovery_terms", [])
            if str(term).lower() not in MATCH_STOPWORDS
        }
        facet_terms = {
            term
            for term in facet_token_terms(
                {
                    "document_type": meta.get("document_type", ""),
                    "document_purpose": meta.get("document_purpose", ""),
                    "audience": meta.get("audience", ""),
                    "evidence_style": meta.get("evidence_style", ""),
                    "structure_style": meta.get("structure_style", ""),
                    "facet_terms": meta.get("facet_terms", []),
                }
            )
            if term not in MATCH_STOPWORDS
        }
        profile_overlap = len(profile_terms.intersection(query_terms))
        title_overlap = len(title_terms.intersection(query_terms))
        cue_overlap = len(cue_terms.intersection(query_terms))
        discovery_overlap = len(discovery_terms.intersection(query_terms))
        facet_overlap = len(facet_terms.intersection(query_terms))
        score = (
            title_overlap * 4
            + discovery_overlap * 3
            + facet_overlap * 3
            + cue_overlap * 2
            + profile_overlap
        )
        if score >= 3 and (
            title_overlap
            or discovery_overlap
            or facet_overlap
            or cue_overlap
            or profile_overlap >= 2
        ):
            topical_matches.append((score, doc_id))
    topical_matches.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in topical_matches]


def resolve_preferred_source_doc_id(
    query: str,
    *,
    query_class: str,
    query_intent: str,
    planned_preferred_doc_id: str | None = None,
) -> str | None:
    if query_class != "evidence_lookup":
        return planned_preferred_doc_id
    if query_intent in {"source_listing", "cross_document_compare", "document_routing"}:
        return None

    query_lower = query.lower()
    if query_intent == "antibiotics":
        if "review" in query_lower or "literature" in query_lower:
            return "the-common-cold-a-review-of-the-literature"
        return "common-cold-clinincal-evidence"
    if query_intent in {"treatment_null_effect", "treatment_subgroup_benefit"} and "vitamin" in query_lower:
        return "vitamin-c-for-preventing-and-treating-the-common-cold"
    if query_intent in {"ct_findings", "ct_follow_up"} and (
        "ct" in query_lower or "scan" in query_lower or "sinus" in query_lower
    ):
        return "ct-study-of-the-common-cold-scanned"
    return preferred_source_doc_id(query, allow_topical=True)


def resolve_matching_source_doc_ids(
    query: str,
    *,
    query_class: str,
    query_intent: str,
    planned_matched_doc_ids: tuple[str, ...] | list[str],
    query_terms: set[str],
    unsupported_entities: set[str],
) -> list[str]:
    if query_class != "evidence_lookup":
        return list(planned_matched_doc_ids)

    explicit_matches = matching_source_doc_ids(query, allow_topical=False)
    if query_intent in {"source_listing", "document_routing"} and unsupported_entities and not explicit_matches:
        return []

    allow_topical = query_intent in {
        "source_listing",
        "document_routing",
        "source_justification",
        "document_overview",
        "cross_document_compare",
    }
    matches = matching_source_doc_ids(query, allow_topical=allow_topical)
    query_lower = query.lower()
    if query_intent in {"source_justification", "document_overview"} and matches:
        return matches[:1]
    if query_intent == "document_routing" and matches:
        if "which file or files" not in query_lower and "which files" not in query_lower:
            return matches[:1]
    return matches


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
    if "appendix b" in query_lower and "med" in query_terms and (
        "stand" in query_terms or "mean" in query_terms
    ):
        return "opioid_med_legend"
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
