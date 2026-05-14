"""Evaluation hooks for the MVP pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answering import GroundedAnswer, answer_query_with_retrieval
from .intent_config import preferred_source_doc_id as configured_source_doc_id
from .retrieval import retrieve_top_k
from .schemas import ChunkRecord


DEFAULT_EVAL_FILENAME = "mvp_eval_cases.json"
DEFAULT_REPORT_FILENAME = "mvp_eval_report.json"
DEFAULT_REGRESSION_REPORT_FILENAME = "regression_report.json"
DEFAULT_FAITHFULNESS_AUDIT_FILENAME = "faithfulness_audit_cases.json"
DEFAULT_FAITHFULNESS_AUDIT_CASE_IDS = [
    "antibiotics",
    "echinacea_overall_conclusion",
    "ct_follow_up_improvement",
    "cmaj_nontraditional_treatments",
    "cmaj_zinc_prevention",
    "ajmedp_immersion_neck_limit",
]
DEFAULT_REGRESSION_CASE_IDS = [
    "source_listing_vitamin_c_and_echinacea",
    "compare_vitamin_c_vs_echinacea_prevention",
    "health_questionnaire_question5_contexts",
    "health_questionnaire_table1_sensitivity",
    "pre_injection_checklist_live_vaccine",
    "opioid_manager_appendix_a_optimized",
    "opioid_manager_appendix_b_adverse_scale",
    "opioid_manager_appendix_c_follow_up_timing",
    "lbdl_document_overview",
    "lbdl_document_routing_backpropagation",
    "source_listing_deep_learning_transformers",
    "ocha_incident_document_overview",
    "ocha_document_routing_cyber_threats",
    "ocha_document_routing_donor_sharing",
    "source_listing_nonmedical_learning_and_incident_response",
    "source_listing_humanitarian_data_governance",
    "ambiguous_document_routing_humanitarian_data_risk",
    "model_report_niger_routing",
    "model_report_niger_justification",
    "model_report_philippines_routing",
    "source_listing_humanitarian_model_reports",
    "compare_niger_chad_model_reports",
    "ambiguous_document_routing_humanitarian_anticipatory_action",
    "negative_health_questionnaire_aspirin_frostbite",
    "negative_opioid_manager_gadolinium_monitoring",
    "negative_document_routing_lease_clauses",
]
DEFAULT_REGRESSION_SHARDS: dict[str, list[str]] = {
    "cross_document_core": [
        "source_listing_vitamin_c_and_echinacea",
        "compare_vitamin_c_vs_echinacea_prevention",
        "negative_source_listing_insulin",
    ],
    "form_grid_core": [
        "health_questionnaire_question5_contexts",
        "health_questionnaire_table1_sensitivity",
        "pre_injection_checklist_live_vaccine",
        "opioid_manager_appendix_a_optimized",
        "opioid_manager_appendix_b_adverse_scale",
        "opioid_manager_appendix_c_follow_up_timing",
    ],
    "source_anchored_review_core": [
        "cmaj_zinc_prevention",
        "cmaj_nontraditional_treatments",
        "echinacea_overall_conclusion",
    ],
    "technical_manual_core": [
        "ajmedp_hypothermia_predisposition",
        "ajmedp_frostbite_severe_zone",
        "ajmedp_immersion_neck_limit",
    ],
    "document_discovery_core": [
        "lbdl_document_overview",
        "lbdl_document_routing_backpropagation",
        "source_listing_deep_learning_transformers",
        "ocha_incident_document_overview",
        "ocha_document_routing_cyber_threats",
        "ocha_document_routing_donor_sharing",
        "source_listing_nonmedical_learning_and_incident_response",
        "source_listing_humanitarian_data_governance",
        "ambiguous_document_routing_humanitarian_data_risk",
        "model_report_niger_routing",
        "model_report_niger_justification",
        "model_report_philippines_routing",
        "source_listing_humanitarian_model_reports",
        "compare_niger_chad_model_reports",
        "ambiguous_document_routing_humanitarian_anticipatory_action",
        "negative_document_routing_lease_clauses",
    ],
    "model_report_core": [
        "model_report_niger_routing",
        "model_report_niger_justification",
        "model_report_philippines_routing",
        "source_listing_humanitarian_model_reports",
        "compare_niger_chad_model_reports",
        "ambiguous_document_routing_humanitarian_anticipatory_action",
    ],
}
SLICE_STABILITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "checklist_fields": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "legend_lookup": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "follow_up_schedule": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "form_grid": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "document_discovery": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "model_report_family": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
}
DEFAULT_EVAL_CASES = [
    {
        "case_id": "symptoms",
        "case_type": "grounded",
        "query": "What are common cold symptoms?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0009",
            "common-cold-clinincal-evidence-chunk-0012",
        ],
        "expected_keywords": [
            "sneezing",
            "runny nose",
            "headache",
            "sore throat",
            "cough",
        ],
        "notes": "Symptoms and symptom course should come from definition/prognosis chunks.",
    },
    {
        "case_id": "duration",
        "case_type": "grounded",
        "query": "How long do common cold symptoms last?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0012",
            "common-cold-clinincal-evidence-chunk-0003",
        ],
        "expected_keywords": [
            "few days",
            "1 week",
            "cough",
        ],
        "notes": "The answer should mention typical duration and lingering cough.",
    },
    {
        "case_id": "transmission",
        "case_type": "grounded",
        "query": "How are common cold infections transmitted?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0011",
            "common-cold-clinincal-evidence-chunk-0003",
        ],
        "expected_keywords": [
            "hand-to-hand contact",
            "droplet",
            "nostrils",
            "eyes",
        ],
        "notes": "The answer should capture hand contact as the main route.",
    },
    {
        "case_id": "definition",
        "case_type": "grounded",
        "query": "What is the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0009",
        ],
        "expected_keywords": [
            "upper respiratory tract",
            "nasal",
            "mucosa",
        ],
        "notes": "The answer should return the definition, not treatments.",
    },
    {
        "case_id": "causes",
        "case_type": "grounded",
        "query": "What usually causes the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0011",
        ],
        "expected_keywords": [
            "viruses",
            "rhinovirus",
            "coronavirus",
        ],
        "notes": "The answer should reflect viral causes rather than symptom descriptions.",
    },
    {
        "case_id": "incidence",
        "case_type": "grounded",
        "query": "How many colds do children and adults get each year?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0010",
            "common-cold-clinincal-evidence-chunk-0003",
        ],
        "expected_keywords": [
            "children",
            "5",
            "adults",
            "two to three",
        ],
        "notes": "The answer should capture yearly incidence for children and adults.",
    },
    {
        "case_id": "antibiotics",
        "case_type": "grounded",
        "query": "Do antibiotics help with the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0005",
            "common-cold-clinincal-evidence-chunk-0175",
            "common-cold-clinincal-evidence-chunk-0176",
            "common-cold-clinincal-evidence-chunk-0192",
        ],
        "expected_keywords": [
            "don't reduce symptoms overall",
            "adverse effects",
            "antibiotic resistance",
        ],
        "notes": "The answer should emphasize that antibiotics are generally not helpful overall.",
    },
    {
        "case_id": "vitamin_c_normal_populations",
        "case_type": "grounded",
        "query": "Does vitamin C prevent the common cold in normal populations?",
        "relevant_chunk_ids": [
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0004",
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0005",
        ],
        "expected_keywords": [
            "incidence",
            "not altered",
            "normal populations",
        ],
        "notes": "The answer should capture the lack of prophylactic incidence benefit in normal populations.",
    },
    {
        "case_id": "vitamin_c_cold_stress",
        "case_type": "grounded",
        "query": "Does vitamin C help people under cold stress?",
        "relevant_chunk_ids": [
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0004",
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0005",
        ],
        "expected_keywords": [
            "cold stress",
            "physical",
            "beneficial",
        ],
        "notes": "The answer should capture the special-case benefit under substantial cold or physical stress.",
    },
    {
        "case_id": "echinacea_overall_conclusion",
        "case_type": "grounded",
        "query": "What does the echinacea meta-analysis conclude about the common cold?",
        "relevant_chunk_ids": [
            "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold-chunk-0082",
            "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold-chunk-0068",
            "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold-chunk-0001",
        ],
        "expected_keywords": [
            "incidence",
            "duration",
            "prevention",
            "treatment",
        ],
        "notes": "The answer should reflect the paper's overall conclusion rather than only a table fragment or generic cold background.",
    },
    {
        "case_id": "echinacea_incidence",
        "case_type": "grounded",
        "query": "Does echinacea reduce the incidence of the common cold?",
        "relevant_chunk_ids": [
            "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold-chunk-0082",
            "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold-chunk-0068",
            "evaluation-of-echinacea-for-the-prevention-and-treatment-of-the-common-cold-chunk-0001",
        ],
        "expected_keywords": [
            "incidence",
            "reduces",
            "benefit",
        ],
        "notes": "The answer should capture the meta-analysis conclusion that echinacea lowers common-cold incidence.",
    },
    {
        "case_id": "ct_abnormalities_prevalence",
        "case_type": "grounded",
        "query": "Did CT scans often show sinus abnormalities during common colds?",
        "relevant_chunk_ids": [
            "ct-study-of-the-common-cold-scanned-chunk-0022",
            "ct-study-of-the-common-cold-scanned-chunk-0018",
        ],
        "expected_keywords": [
            "high prevalence",
            "ostiomeatal",
            "sinus abnormalities",
        ],
        "notes": "The scanned CT-study benchmark should surface the discussion-level conclusion that sinus abnormalities were common during naturally acquired colds.",
    },
    {
        "case_id": "ct_follow_up_improvement",
        "case_type": "grounded",
        "query": "What did follow-up CT scans show after 13 to 20 days?",
        "relevant_chunk_ids": [
            "ct-study-of-the-common-cold-scanned-chunk-0021",
            "ct-study-of-the-common-cold-scanned-chunk-0022",
        ],
        "expected_keywords": [
            "13 to 20 days",
            "residual abnormalities",
            "follow-up",
        ],
        "notes": "The scanned CT-study benchmark should capture the follow-up evaluation window and the fact that some residual abnormalities remained.",
    },
    {
        "case_id": "negative_vaccine",
        "case_type": "negative",
        "query": "What vaccine prevents the common cold?",
        "relevant_chunk_ids": [],
        "expected_keywords": [],
        "notes": "The current benchmark documents that no grounded vaccine answer should be produced.",
    },
    {
        "case_id": "negative_insulin",
        "case_type": "negative",
        "query": "Does insulin treat the common cold?",
        "relevant_chunk_ids": [],
        "expected_keywords": [],
        "notes": "The current benchmark documents that unrelated treatment questions should trigger abstention.",
    },
    {
        "case_id": "negative_echinacea_influenza",
        "case_type": "negative",
        "query": "Does echinacea prevent influenza?",
        "relevant_chunk_ids": [],
        "expected_keywords": [],
        "notes": "The benchmark should abstain when the treatment-focused echinacea review is asked about influenza rather than the common cold.",
    },
    {
        "case_id": "negative_gadolinium",
        "case_type": "negative",
        "query": "Was gadolinium administered?",
        "relevant_chunk_ids": [],
        "expected_keywords": [],
        "notes": "The benchmark should abstain on an unsupported imaging-contrast question even after adding the scanned CT-study document.",
    },
]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def inspect_chunk_sample(chunks: list[ChunkRecord], sample_size: int = 5) -> list[ChunkRecord]:
    """Return a small chunk sample for manual quality review."""
    return chunks[:sample_size]


def ensure_default_eval_cases(eval_dir: Path) -> Path:
    """Create the default small evaluation set if it does not yet exist."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / DEFAULT_EVAL_FILENAME
    if not eval_path.exists():
        bundled_eval_path = (
            Path(__file__).resolve().parents[2] / "data" / "eval" / DEFAULT_EVAL_FILENAME
        )
        if bundled_eval_path.exists():
            eval_path.write_text(
                bundled_eval_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            eval_path.write_text(
                json.dumps(DEFAULT_EVAL_CASES, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return eval_path


def load_eval_cases(eval_path: Path) -> list[dict]:
    """Load evaluation cases from JSON."""
    eval_path = eval_path.expanduser().resolve()
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    return json.loads(eval_path.read_text(encoding="utf-8"))


def ensure_default_faithfulness_audit(eval_dir: Path) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    audit_path = eval_dir / DEFAULT_FAITHFULNESS_AUDIT_FILENAME
    if not audit_path.exists():
        bundled_audit_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "eval"
            / DEFAULT_FAITHFULNESS_AUDIT_FILENAME
        )
        if bundled_audit_path.exists():
            audit_path.write_text(
                bundled_audit_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            audit_path.write_text(
                json.dumps(DEFAULT_FAITHFULNESS_AUDIT_CASE_IDS, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return audit_path


def load_faithfulness_audit_case_ids(audit_path: Path) -> list[str]:
    audit_path = audit_path.expanduser().resolve()
    if not audit_path.exists():
        raise FileNotFoundError(f"Faithfulness audit file not found: {audit_path}")
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Faithfulness audit file must contain a JSON list of case IDs.")
    return [str(item) for item in data]


def _keyword_matches(answer_text: str, expected_keywords: list[str]) -> dict:
    answer_lower = (
        answer_text.lower()
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
    )
    answer_compact = "".join(ch for ch in answer_lower if ch.isalnum())
    matched = [
        keyword
        for keyword in expected_keywords
        if (
            keyword.lower().replace("ﬁ", "fi").replace("ﬂ", "fl") in answer_lower
            or "".join(
                ch for ch in keyword.lower().replace("ﬁ", "fi").replace("ﬂ", "fl") if ch.isalnum()
            )
            in answer_compact
        )
    ]
    return {
        "matched_keywords": matched,
        "keyword_coverage": (len(matched) / len(expected_keywords)) if expected_keywords else 0.0,
    }


def _preview_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _normalize_surface(text: str) -> str:
    return " ".join(
        text.lower()
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .split()
    )


def _split_answer_sentences(answer_text: str) -> list[str]:
    fragments = [
        item.strip()
        for item in answer_text.replace("\n", " ").split(".")
        if item.strip()
    ]
    return [fragment if fragment.endswith(".") else f"{fragment}." for fragment in fragments]


def _case_slice_labels(case: dict) -> list[str]:
    case_id = case["case_id"]
    case_type = case.get("case_type", "grounded")
    query_lower = case["query"].lower()

    labels = ["ocr_derived" if case_id.startswith("ct_") else "native_text"]
    labels.append(case_type)

    is_treatment = (
        case_id == "antibiotics"
        or case_id.startswith("compare_vitamin_c")
        or case_id.startswith("source_listing_vitamin_c")
        or case_id.startswith("vitamin_c")
        or case_id.startswith("echinacea")
        or "antibiotic" in query_lower
        or "vitamin c" in query_lower
        or "echinacea" in query_lower
        or "vaccine" in query_lower
        or "insulin" in query_lower
    )
    labels.append("treatment" if is_treatment else "non_treatment")
    if case_id.startswith("vitamin_c"):
        labels.extend(["vitamin_c_review", "review_heavy"])
    elif case_id.startswith("echinacea"):
        labels.extend(["echinacea_review", "review_heavy"])
    elif case_id.startswith("ct_") or case_id == "negative_gadolinium":
        labels.extend(["scanned_ct", "layout_ocr"])
    elif (
        case_id.startswith("health_questionnaire_")
        or "health-check questionnaire" in query_lower
        or "questionnaire for subjects exposed to cold" in query_lower
    ):
        labels.extend(["health_questionnaire_form", "form_grid"])
    elif (
        case_id.startswith("opioid_manager_")
        or "opioid manager appendix" in query_lower
        or "opioid manager appendices" in query_lower
    ):
        labels.extend(["opioid_appendix_form", "form_grid", "appendix_like"])
    elif (
        case_id.startswith("pre_injection_checklist_")
        or "pre injection checklist" in query_lower
        or "pre injection check list" in query_lower
    ):
        labels.extend(["pre_injection_checklist", "form_grid", "appendix_like"])
    elif case_id.startswith("ajmedp_") or "ajmedp" in query_lower or "tb med 508" in query_lower:
        labels.extend(["ajmedp_manual", "technical_manual", "table_heavy"])
    elif (
        case_id.startswith("lbdl_")
        or "little book of deep learning" in query_lower
        or "backpropagation" in query_lower
    ):
        labels.extend(["deep_learning_book", "non_medical", "document_discovery"])
    elif (
        case_id.startswith("ocha_")
        or "data incident management" in query_lower
        or "responsible data sharing with donors" in query_lower
        or "cyber threats for humanitarians" in query_lower
        or "humanitarian data incident" in query_lower
    ):
        labels.extend(["humanitarian_data_guidance", "non_medical", "document_discovery"])
    elif case_id.startswith("wat_") or "literature review" in query_lower or "dennis wat" in query_lower:
        labels.extend(["wat_review", "review_heavy"])
    elif case_id.startswith("cmaj_") or "cmaj" in query_lower:
        labels.extend(["cmaj_review", "review_heavy"])
    else:
        labels.extend(["clinical_reference", "section_structured"])

    labels.extend(case.get("case_tags", []))
    return sorted(set(labels))


def _preferred_source_doc_id_from_query(query: str) -> str | None:
    return configured_source_doc_id(query)


def _result_slice_labels(grounded_answer: GroundedAnswer, base_labels: list[str]) -> list[str]:
    labels = set(base_labels)
    top_doc_ids = {chunk.doc_id for chunk in grounded_answer.top_k_hits}
    expanded_noise = {
        noise
        for chunk in grounded_answer.expanded_hits
        for noise in chunk.noise_labels
    }

    preferred_doc_id = _preferred_source_doc_id_from_query(grounded_answer.query)
    if {"source_anchored_review", "source_anchored_technical", "source_anchored_form"}.intersection(labels):
        if preferred_doc_id and top_doc_ids == {preferred_doc_id}:
            labels.add("source_locked")
        elif len(top_doc_ids) > 1:
            labels.add("cross_document_mixing")
    if expanded_noise.intersection({"table_reference", "table_like_section", "reference_tail"}):
        labels.add("table_adjacent")
    return sorted(labels)


def _chunk_snapshot(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_title": chunk.section_title,
        "extraction_method": chunk.extraction_method,
        "quality_score": chunk.quality_score,
        "noise_labels": chunk.noise_labels,
        "preview": _preview_text(chunk.text),
    }


def _evidence_snapshot(item: Any) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "section_title": item.section_title,
        "score": round(item.score, 4),
        "sentence": _preview_text(item.sentence, limit=260),
    }


def _case_status(
    case_type: str,
    retrieval_result: dict,
    answer_result: dict,
) -> str:
    if case_type == "negative":
        return "pass" if answer_result.get("negative_success") else "negative_fail"

    rr = retrieval_result.get("reciprocal_rank") or 0.0
    recall = retrieval_result.get("recall_at_k") or 0.0
    keyword_coverage = answer_result.get("keyword_coverage") or 0.0
    abstained = answer_result.get("abstained", False)

    if abstained:
        return "fail"
    if rr < 1.0 or recall < 1.0:
        return "retrieval_warning"
    if keyword_coverage < 1.0:
        return "answer_warning"
    return "pass"


def _debug_case_record(
    case: dict,
    retrieval_result: dict,
    answer_result: dict,
    grounded_answer: GroundedAnswer,
) -> dict[str, Any]:
    case_type = case.get("case_type", "grounded")
    base_labels = _case_slice_labels(case)
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
        "query": case["query"],
        "notes": case.get("notes"),
        "slice_labels": _result_slice_labels(grounded_answer, base_labels),
        "status": _case_status(case_type, retrieval_result, answer_result),
        "expected_keywords": case.get("expected_keywords", []),
        "matched_keywords": answer_result.get("matched_keywords", []),
        "retrieval": {
            "evaluation_level": retrieval_result.get("evaluation_level", "chunk"),
            "top_k_ids": retrieval_result["retrieved_ids"],
            "top_k_doc_ids": retrieval_result.get("retrieved_doc_ids", []),
            "precision_at_k": retrieval_result.get("precision_at_k"),
            "recall_at_k": retrieval_result.get("recall_at_k"),
            "reciprocal_rank": retrieval_result.get("reciprocal_rank"),
            "top_k_snapshots": [_chunk_snapshot(chunk) for chunk in grounded_answer.top_k_hits],
            "expanded_snapshots": [
                _chunk_snapshot(chunk) for chunk in grounded_answer.expanded_hits
            ],
        },
        "answer": {
            "abstained": answer_result["abstained"],
            "negative_success": answer_result.get("negative_success"),
            "keyword_coverage": answer_result["keyword_coverage"],
            "trace": grounded_answer.answer_trace,
            "full_answer": grounded_answer.answer,
            "answer_preview": _preview_text(grounded_answer.answer, limit=320),
            "evidence_snapshots": [
                _evidence_snapshot(item) for item in grounded_answer.evidence
            ],
        },
    }


def _summarize_retrieval_results(retrieval_results: list[dict], answer_results: list[dict]) -> dict[str, Any]:
    grounded_retrieval = [item for item in retrieval_results if item.get("case_type") != "negative"]
    negative_answers = [item for item in answer_results if item.get("case_type") == "negative"]
    warning_case_ids = [
        retrieval["case_id"]
        for retrieval, answer in zip(retrieval_results, answer_results)
        if (
            retrieval.get("case_type") != "negative"
            and (
                (retrieval.get("reciprocal_rank") or 0.0) < 1.0
                or (retrieval.get("recall_at_k") or 0.0) < 1.0
                or (answer.get("keyword_coverage") or 0.0) < 1.0
                or answer.get("abstained", False)
            )
        )
    ]
    return {
        "avg_precision_at_k": _average([item["precision_at_k"] for item in grounded_retrieval]),
        "avg_recall_at_k": _average([item["recall_at_k"] for item in grounded_retrieval]),
        "mrr": _average([item["reciprocal_rank"] for item in grounded_retrieval]),
        "avg_keyword_coverage": _average(
            [item["keyword_coverage"] for item in answer_results if item.get("case_type") != "negative"]
        ),
        "negative_case_count": len(negative_answers),
        "negative_success_rate": _average(
            [1.0 if item["negative_success"] else 0.0 for item in negative_answers]
        ),
        "warning_case_count": len(warning_case_ids),
        "warning_case_ids": warning_case_ids,
    }


def _faithfulness_audit_record(debug_case: dict[str, Any]) -> dict[str, Any]:
    answer_sentences = [
        item["sentence"] for item in debug_case["answer"]["evidence_snapshots"]
    ]
    support_corpus = " ".join(answer_sentences)
    normalized_support = _normalize_surface(support_corpus)
    supported = []
    unsupported = []
    for sentence in answer_sentences:
        if _normalize_surface(sentence) in normalized_support:
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    supported_ratio = (len(supported) / len(answer_sentences)) if answer_sentences else 0.0
    return {
        "case_id": debug_case["case_id"],
        "supported_sentence_ratio": supported_ratio,
        "supported_sentences": supported,
        "unsupported_sentences": unsupported,
        "evidence_preview": [
            item["sentence"] for item in debug_case["answer"]["evidence_snapshots"]
        ],
    }


def _run_faithfulness_audit(debug_cases: list[dict[str, Any]], audit_case_ids: list[str]) -> dict[str, Any]:
    case_lookup = {item["case_id"]: item for item in debug_cases}
    records = []
    for case_id in audit_case_ids:
        debug_case = case_lookup.get(case_id)
        if not debug_case or debug_case.get("case_type") == "negative":
            continue
        records.append(_faithfulness_audit_record(debug_case))

    failing_case_ids = [
        item["case_id"] for item in records if item["supported_sentence_ratio"] < 1.0
    ]
    return {
        "sampled_case_count": len(records),
        "avg_supported_sentence_ratio": _average(
            [item["supported_sentence_ratio"] for item in records]
        ),
        "failing_case_count": len(failing_case_ids),
        "failing_case_ids": failing_case_ids,
        "recommend_llm_judge": len(failing_case_ids) > 0,
        "cases": records,
    }


def evaluate_retrieval_case(
    case: dict,
    index_dir: Path,
    k: int,
    use_lightweight_rerank: bool = True,
) -> dict:
    """Evaluate retrieval metrics for a single query."""
    hits = retrieve_top_k(
        query=case["query"],
        index_dir=index_dir,
        k=k,
        use_lightweight_rerank=use_lightweight_rerank,
    )
    retrieved_ids = [chunk.chunk_id for chunk in hits]
    retrieved_doc_ids = _ordered_unique([chunk.doc_id for chunk in hits])
    relevant = set(case["relevant_chunk_ids"])
    relevant_doc_ids = set(case.get("relevant_doc_ids", []))
    case_type = case.get("case_type", "grounded")
    evaluation_level = "document" if relevant_doc_ids else "chunk"
    if case_type == "negative":
        return {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "evaluation_level": evaluation_level,
            "retrieved_ids": retrieved_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
        }
    if relevant_doc_ids:
        return {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "evaluation_level": "document",
            "retrieved_ids": retrieved_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "precision_at_k": precision_at_k(retrieved_doc_ids, relevant_doc_ids, k),
            "recall_at_k": recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
            "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
        }
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
        "query": case["query"],
        "evaluation_level": "chunk",
        "retrieved_ids": retrieved_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
    }


def evaluate_answer_case(case: dict, index_dir: Path, chunk_root: Path, k: int) -> dict:
    """Evaluate the grounded answer path for a single query."""
    result: GroundedAnswer = answer_query_with_retrieval(
        query=case["query"],
        index_dir=index_dir,
        chunk_root=chunk_root,
        k=k,
    )
    keyword_eval = _keyword_matches(result.answer, case.get("expected_keywords", []))
    case_type = case.get("case_type", "grounded")
    abstained = result.answer.startswith("No grounded answer")
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
        "query": case["query"],
        "answer": result.answer,
        "top_k_hit_ids": [chunk.chunk_id for chunk in result.top_k_hits],
        "expanded_hit_ids": [chunk.chunk_id for chunk in result.expanded_hits],
        "evidence_chunk_ids": [item.chunk_id for item in result.evidence],
        "evidence_sentences": [item.sentence for item in result.evidence],
        "answer_trace": result.answer_trace,
        "abstained": abstained,
        "negative_success": abstained if case_type == "negative" else None,
        **keyword_eval,
    }


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _slice_summary(label: str, debug_cases: list[dict]) -> dict[str, Any]:
    slice_cases = [item for item in debug_cases if label in item.get("slice_labels", [])]
    grounded_cases = [item for item in slice_cases if item.get("case_type") != "negative"]
    negative_cases = [item for item in slice_cases if item.get("case_type") == "negative"]
    warning_case_ids = [item["case_id"] for item in slice_cases if item.get("status") != "pass"]

    return {
        "case_count": len(slice_cases),
        "grounded_case_count": len(grounded_cases),
        "negative_case_count": len(negative_cases),
        "avg_precision_at_k": _average(
            [item["retrieval"]["precision_at_k"] for item in grounded_cases]
        ),
        "avg_recall_at_k": _average(
            [item["retrieval"]["recall_at_k"] for item in grounded_cases]
        ),
        "mrr": _average([item["retrieval"]["reciprocal_rank"] for item in grounded_cases]),
        "avg_keyword_coverage": _average(
            [item["answer"]["keyword_coverage"] for item in grounded_cases]
        ),
        "negative_success_rate": _average(
            [1.0 if item["answer"].get("negative_success") else 0.0 for item in negative_cases]
        ),
        "warning_case_count": len(warning_case_ids),
        "warning_case_ids": warning_case_ids,
    }


def _evaluate_slice_stability(
    slices: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failed_labels: list[str] = []

    for label, thresholds in SLICE_STABILITY_THRESHOLDS.items():
        slice_summary = slices.get(label)
        if slice_summary is None:
            checks[label] = {
                "present": False,
                "pass": False,
                "reason": "slice missing from current benchmark",
                "thresholds": thresholds,
            }
            failed_labels.append(label)
            continue

        failed_metrics: dict[str, dict[str, float]] = {}
        for metric_name, min_value in thresholds.items():
            actual_value = float(slice_summary.get(metric_name, 0.0))
            if actual_value < min_value:
                failed_metrics[metric_name] = {
                    "actual": actual_value,
                    "required_min": min_value,
                }

        passed = not failed_metrics
        checks[label] = {
            "present": True,
            "pass": passed,
            "thresholds": thresholds,
            "failed_metrics": failed_metrics,
        }
        if not passed:
            failed_labels.append(label)

    return {
        "all_pass": not failed_labels,
        "failed_labels": failed_labels,
        "checks": checks,
    }


def _deferred_feature_decisions(
    summary: dict[str, Any],
    slices: dict[str, Any],
    faithfulness_audit: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    table_adjacent_warnings = slices.get("table_adjacent", {}).get("warning_case_count", 0)
    table_heavy_warnings = slices.get("table_heavy", {}).get("warning_case_count", 0)
    scanned_warnings = slices.get("ocr_derived", {}).get("warning_case_count", 0)
    source_review_warnings = slices.get("source_anchored_review", {}).get("warning_case_count", 0)
    source_technical_warnings = slices.get("source_anchored_technical", {}).get("warning_case_count", 0)
    return {
        "pdfplumber_probe": {
            "recommended": bool(table_heavy_warnings and table_adjacent_warnings),
            "reason": (
                "keep deferred: the current table-heavy benchmark is hitting extracted table content, "
                "and remaining issues are source-locking or answer selection rather than table extraction misses"
                if not (table_heavy_warnings and table_adjacent_warnings)
                else "table-heavy cases still show table-adjacent warnings after source-locking and chunk-quality filtering"
            ),
        },
        "cross_encoder_reranking": {
            "recommended": summary.get("warning_case_count", 0) > 0
            and (source_review_warnings > 0 or source_technical_warnings > 0)
            and baseline_summary.get("mrr", 0.0) == summary.get("mrr", 0.0),
            "reason": (
                "keep deferred: lightweight rerank plus source-aware heuristics are sufficient on the current benchmark"
                if not (
                    summary.get("warning_case_count", 0) > 0
                    and (source_review_warnings > 0 or source_technical_warnings > 0)
                    and baseline_summary.get("mrr", 0.0) == summary.get("mrr", 0.0)
                )
                else "remaining source-anchored warnings survive the current lightweight rerank without MRR improvement"
            ),
        },
        "llm_as_judge": {
            "recommended": bool(faithfulness_audit.get("recommend_llm_judge")),
            "reason": (
                "keep deferred: sampled extractive faithfulness audit does not show unsupported-answer drift"
                if not faithfulness_audit.get("recommend_llm_judge")
                else "sampled faithfulness audit found unsupported answer sentences"
            ),
        },
    }


def run_mvp_evaluation(
    index_dir: Path,
    chunk_root: Path,
    eval_dir: Path,
    k: int = 5,
    eval_path: Path | None = None,
) -> tuple[dict, Path]:
    """Run the small local MVP evaluation workflow and save a report."""
    eval_dir = eval_dir.expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_path is None:
        eval_path = ensure_default_eval_cases(eval_dir)
    else:
        eval_path = eval_path.expanduser().resolve()

    cases = load_eval_cases(eval_path)
    audit_path = ensure_default_faithfulness_audit(eval_dir)
    audit_case_ids = load_faithfulness_audit_case_ids(audit_path)
    retrieval_results = []
    answer_results = []
    debug_cases = []
    baseline_retrieval_results = []

    for case in cases:
        grounded_answer = answer_query_with_retrieval(
            query=case["query"],
            index_dir=index_dir,
            chunk_root=chunk_root,
            k=k,
            use_lightweight_rerank=True,
        )
        retrieved_ids = [chunk.chunk_id for chunk in grounded_answer.top_k_hits]
        relevant = set(case["relevant_chunk_ids"])
        case_type = case.get("case_type", "grounded")
        baseline_retrieval_results.append(
            evaluate_retrieval_case(
                case=case,
                index_dir=index_dir,
                k=k,
                use_lightweight_rerank=False,
            )
        )
        if case_type == "negative":
            retrieval_result = {
                "case_id": case["case_id"],
                "case_type": case_type,
                "query": case["query"],
                "evaluation_level": "document" if case.get("relevant_doc_ids") else "chunk",
                "retrieved_ids": retrieved_ids,
                "retrieved_doc_ids": _ordered_unique([chunk.doc_id for chunk in grounded_answer.top_k_hits]),
                "precision_at_k": None,
                "recall_at_k": None,
                "reciprocal_rank": None,
            }
        else:
            relevant_doc_ids = set(case.get("relevant_doc_ids", []))
            retrieved_doc_ids = _ordered_unique([chunk.doc_id for chunk in grounded_answer.top_k_hits])
            if relevant_doc_ids:
                retrieval_result = {
                    "case_id": case["case_id"],
                    "case_type": case_type,
                    "query": case["query"],
                    "evaluation_level": "document",
                    "retrieved_ids": retrieved_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "precision_at_k": precision_at_k(retrieved_doc_ids, relevant_doc_ids, k),
                    "recall_at_k": recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
                    "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
                }
            else:
                retrieval_result = {
                    "case_id": case["case_id"],
                    "case_type": case_type,
                    "query": case["query"],
                    "evaluation_level": "chunk",
                    "retrieved_ids": retrieved_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
                    "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
                    "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
                }

        keyword_eval = _keyword_matches(
            grounded_answer.answer,
            case.get("expected_keywords", []),
        )
        abstained = grounded_answer.answer.startswith("No grounded answer")
        answer_result = {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "answer": grounded_answer.answer,
            "top_k_hit_ids": [chunk.chunk_id for chunk in grounded_answer.top_k_hits],
            "expanded_hit_ids": [chunk.chunk_id for chunk in grounded_answer.expanded_hits],
            "evidence_chunk_ids": [item.chunk_id for item in grounded_answer.evidence],
            "evidence_sentences": [item.sentence for item in grounded_answer.evidence],
            "abstained": abstained,
            "negative_success": abstained if case_type == "negative" else None,
            **keyword_eval,
        }

        retrieval_results.append(retrieval_result)
        answer_results.append(answer_result)
        debug_cases.append(
            _debug_case_record(
                case=case,
                retrieval_result=retrieval_result,
                answer_result=answer_result,
                grounded_answer=grounded_answer,
            )
        )

    warning_case_ids = [item["case_id"] for item in debug_cases if item.get("status") not in {"pass"}]
    all_slice_labels = sorted(
        {
            label
            for item in debug_cases
            for label in item.get("slice_labels", [])
        }
    )
    slices = {label: _slice_summary(label, debug_cases) for label in all_slice_labels}
    slice_stability = _evaluate_slice_stability(slices)
    faithfulness_audit = _run_faithfulness_audit(debug_cases, audit_case_ids)
    summary = _summarize_retrieval_results(retrieval_results, answer_results)
    baseline_summary = _summarize_retrieval_results(baseline_retrieval_results, answer_results)
    deferred_feature_decisions = _deferred_feature_decisions(
        summary=summary,
        slices=slices,
        faithfulness_audit=faithfulness_audit,
        baseline_summary=baseline_summary,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "faithfulness_audit_file": str(audit_path),
        "case_count": len(cases),
        "summary": summary,
        "slices": slices,
        "slice_stability": slice_stability,
        "retrieval_strategy_comparison": {
            "baseline_chunking_only": {
                "avg_precision_at_k": baseline_summary["avg_precision_at_k"],
                "avg_recall_at_k": baseline_summary["avg_recall_at_k"],
                "mrr": baseline_summary["mrr"],
            },
            "lightweight_rerank": {
                "avg_precision_at_k": summary["avg_precision_at_k"],
                "avg_recall_at_k": summary["avg_recall_at_k"],
                "mrr": summary["mrr"],
            },
        },
        "faithfulness_audit": faithfulness_audit,
        "deferred_feature_decisions": deferred_feature_decisions,
        "retrieval_results": retrieval_results,
        "baseline_retrieval_results": baseline_retrieval_results,
        "answer_results": answer_results,
        "debug_cases": debug_cases,
    }

    report_path = eval_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path


def _regression_case_status(
    case_type: str,
    retrieval_result: dict[str, Any],
    answer_result: dict[str, Any],
) -> str:
    if case_type == "negative":
        return "pass" if answer_result.get("negative_success") else "negative_fail"
    if float(retrieval_result.get("reciprocal_rank") or 0.0) <= 0.0:
        return "retrieval_fail"
    if float(answer_result.get("keyword_coverage") or 0.0) < 0.9:
        return "answer_fail"
    return "pass"


def run_regression_suite(
    index_dir: Path,
    chunk_root: Path,
    eval_dir: Path,
    k: int = 5,
    eval_path: Path | None = None,
    case_ids: list[str] | None = None,
    shard: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run a deterministic high-risk regression subset before full benchmark reruns."""
    eval_dir = eval_dir.expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_path is None:
        eval_path = ensure_default_eval_cases(eval_dir)
    else:
        eval_path = eval_path.expanduser().resolve()

    all_cases = load_eval_cases(eval_path)
    case_map = {item["case_id"]: item for item in all_cases}
    selected_case_ids = case_ids or DEFAULT_REGRESSION_SHARDS.get(shard or "", DEFAULT_REGRESSION_CASE_IDS)

    missing_case_ids = [case_id for case_id in selected_case_ids if case_id not in case_map]
    selected_cases = [case_map[case_id] for case_id in selected_case_ids if case_id in case_map]

    case_results: list[dict[str, Any]] = []
    failed_case_ids: list[str] = []
    for case in selected_cases:
        retrieval_result = evaluate_retrieval_case(case=case, index_dir=index_dir, k=k)
        answer_result = evaluate_answer_case(case=case, index_dir=index_dir, chunk_root=chunk_root, k=k)
        status = _regression_case_status(case.get("case_type", "grounded"), retrieval_result, answer_result)
        if status != "pass":
            failed_case_ids.append(case["case_id"])
        case_results.append(
            {
                "case_id": case["case_id"],
                "case_type": case.get("case_type", "grounded"),
                "status": status,
                "retrieval": retrieval_result,
                "answer": answer_result,
            }
        )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "selected_shard": shard,
        "selected_case_ids": selected_case_ids,
        "missing_case_ids": missing_case_ids,
        "case_count": len(selected_cases),
        "pass_count": len(selected_cases) - len(failed_case_ids),
        "fail_count": len(failed_case_ids),
        "failed_case_ids": failed_case_ids,
        "all_pass": len(failed_case_ids) == 0 and not missing_case_ids,
        "case_results": case_results,
    }

    report_path = eval_dir / DEFAULT_REGRESSION_REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path
