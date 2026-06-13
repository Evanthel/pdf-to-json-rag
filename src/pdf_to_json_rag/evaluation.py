"""Evaluation hooks for the MVP pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .answering import GroundedAnswer, answer_query_with_retrieval
from .indexing import build_local_index
from .intent_config import resolve_preferred_source_doc_id
from .llm_output import parsed_json_payload, parse_strict_json_output
from .llm_runtime import prompt_command_payload, run_prompt_command
from .query_planning import plan_query
from .retrieval import retrieve_top_k, retrieve_top_k_with_neighbors
from .schemas import ChunkRecord


DEFAULT_EVAL_FILENAME = "mvp_eval_cases.json"
DEFAULT_REPORT_FILENAME = "mvp_eval_report.json"
DEFAULT_REGRESSION_REPORT_FILENAME = "regression_report.json"
DEFAULT_RUNTIME_COMPARISON_REPORT_FILENAME = "runtime_mode_comparison.json"
DEFAULT_RUNTIME_PROMOTION_SNAPSHOT_FILENAME = "runtime_promotion_snapshot.json"
DEFAULT_FAITHFULNESS_AUDIT_FILENAME = "faithfulness_audit_cases.json"
LLM_JUDGE_PROMPT_TEMPLATE_ID = "faithfulness_context_judge.v1"
LLM_JUDGE_COMMAND_ENV = "PDF_TO_JSON_RAG_JUDGE_COMMAND"
LLM_JUDGE_RULES = (
    "Judge only whether the answer is supported by the provided source context.",
    "Do not use outside knowledge to fill gaps.",
    "Mark a sentence unsupported if the context does not directly support it.",
    "Return strict JSON only.",
)
LLM_JUDGE_OUTPUT_SCHEMA = {
    "faithful": "boolean",
    "supported_sentence_ratio": "number between 0 and 1",
    "unsupported_sentences": "array of strings",
    "rationale": "short string grounded in the provided context",
}
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
    "lbdl_document_type",
    "lbdl_document_routing_backpropagation",
    "source_listing_deep_learning_transformers",
    "ocha_incident_document_overview",
    "ocha_incident_document_purpose",
    "ocha_document_routing_cyber_threats",
    "ocha_document_routing_donor_sharing",
    "source_listing_nonmedical_learning_and_incident_response",
    "source_listing_humanitarian_data_governance",
    "ambiguous_document_routing_humanitarian_data_risk",
    "model_report_niger_routing",
    "model_report_niger_document_type",
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
    "source_anchor_contract_core": [
        "ajmedp_frostbite_severe_zone",
        "ajmedp_hypothermia_symptoms",
        "negative_ajmedp_aspirin_frostbite",
    ],
    "document_discovery_core": [
        "lbdl_document_overview",
        "lbdl_document_type",
        "lbdl_document_routing_backpropagation",
        "source_listing_deep_learning_transformers",
        "ocha_incident_document_overview",
        "ocha_incident_document_purpose",
        "ocha_document_routing_cyber_threats",
        "ocha_document_routing_donor_sharing",
        "source_listing_nonmedical_learning_and_incident_response",
        "source_listing_humanitarian_data_governance",
        "ambiguous_document_routing_humanitarian_data_risk",
        "model_report_niger_routing",
        "model_report_niger_document_type",
        "model_report_niger_justification",
        "model_report_philippines_routing",
        "source_listing_humanitarian_model_reports",
        "compare_niger_chad_model_reports",
        "ambiguous_document_routing_humanitarian_anticipatory_action",
        "negative_document_routing_lease_clauses",
    ],
    "model_report_core": [
        "model_report_niger_routing",
        "model_report_niger_document_type",
        "model_report_niger_justification",
        "model_report_philippines_routing",
        "source_listing_humanitarian_model_reports",
        "compare_niger_chad_model_reports",
        "ambiguous_document_routing_humanitarian_anticipatory_action",
    ],
    "document_facets_core": [
        "lbdl_document_type",
        "ocha_incident_document_purpose",
        "model_report_niger_document_type",
    ],
    "retrieval_contract_core": [
        "symptoms",
        "lbdl_document_overview",
        "model_report_niger_justification",
        "source_listing_nonmedical_learning_and_incident_response",
        "compare_vitamin_c_vs_echinacea_prevention",
    ],
    "retrieval_synthesis_core": [
        "lbdl_document_overview",
        "lbdl_document_routing_backpropagation",
        "source_listing_humanitarian_model_reports",
        "model_report_niger_justification",
        "compare_niger_chad_model_reports",
    ],
    "query_planning_core": [
        "lbdl_document_overview",
        "lbdl_document_type",
        "lbdl_document_routing_backpropagation",
        "ocha_incident_document_purpose",
        "source_listing_humanitarian_model_reports",
        "compare_niger_chad_model_reports",
        "model_report_niger_justification",
    ],
    "answer_modes_core": [
        "symptoms",
        "lbdl_document_overview",
        "lbdl_document_routing_backpropagation",
        "source_listing_humanitarian_model_reports",
        "compare_niger_chad_model_reports",
        "model_report_niger_justification",
    ],
    "document_family_core": [
        "lbdl_document_type",
        "ocha_incident_document_purpose",
        "model_report_niger_document_type",
        "health_questionnaire_question5_contexts",
        "ajmedp_hypothermia_predisposition",
    ],
    "inventory_coverage_core": [
        "lbdl_document_overview",
        "ocha_incident_document_overview",
        "model_report_niger_routing",
        "ocha_document_routing_cyber_threats",
        "source_listing_humanitarian_data_governance",
    ],
    "relationship_core": [
        "compare_vitamin_c_vs_echinacea_prevention",
        "compare_niger_chad_model_reports",
        "source_listing_nonmedical_learning_and_incident_response",
    ],
    "document_pipeline_core": [
        "lbdl_document_overview",
        "lbdl_document_routing_backpropagation",
        "source_listing_humanitarian_model_reports",
        "compare_niger_chad_model_reports",
        "model_report_niger_justification",
    ],
    "structure_chunking_core": [
        "ajmedp_immersion_neck_limit",
        "health_questionnaire_question5_contexts",
        "health_questionnaire_table1_sensitivity",
        "opioid_manager_appendix_b_adverse_scale",
        "opioid_manager_appendix_c_follow_up_timing",
        "pre_injection_checklist_live_vaccine",
        "pre_injection_checklist_side_effects",
    ],
    "evidence_anchor_core": [
        "antibiotics",
        "vitamin_c_normal_populations",
        "vitamin_c_cold_stress",
        "echinacea_overall_conclusion",
        "ct_follow_up_improvement",
        "cmaj_zinc_prevention",
        "wat_antibiotics_review",
    ],
    "section_reconstruction_core": [
        "lbdl_document_overview",
        "ocha_incident_document_overview",
        "health_questionnaire_question5_contexts",
        "pre_injection_checklist_side_effects",
        "ajmedp_immersion_neck_limit",
    ],
    "document_selection_core": [
        "lbdl_document_routing_backpropagation",
        "source_listing_humanitarian_model_reports",
        "ambiguous_document_routing_humanitarian_data_risk",
        "model_report_niger_justification",
        "compare_niger_chad_model_reports",
    ],
    "semantic_document_understanding_core": [
        "lbdl_document_type",
        "lbdl_document_audience",
        "ocha_incident_document_purpose",
        "ocha_incident_document_audience",
        "model_report_niger_document_type",
        "model_report_niger_document_audience",
    ],
    "confidence_aware_document_core": [
        "lbdl_document_confidence",
        "ocha_incident_document_confidence",
        "model_report_niger_document_confidence",
    ],
    "trust_policy_document_core": [
        "lbdl_document_classification_rationale",
        "ocha_incident_document_classification_rationale",
        "lbdl_document_classification_limits",
        "model_report_niger_document_classification_limits",
    ],
    "document_maintenance_core": [
        "lbdl_document_overview",
        "lbdl_document_routing_backpropagation",
        "source_listing_humanitarian_model_reports",
        "model_report_niger_justification",
        "compare_niger_chad_model_reports",
    ],
    "structured_form_maintenance_core": [
        "health_questionnaire_table1_sensitivity",
        "pre_injection_checklist_live_vaccine",
        "opioid_manager_appendix_a_optimized",
        "opioid_manager_appendix_b_adverse_scale",
        "opioid_manager_appendix_c_follow_up_timing",
    ],
    "layout_robustness_core": [
        "health_questionnaire_question5_contexts",
        "health_questionnaire_table1_sensitivity",
        "pre_injection_checklist_side_effects",
        "ajmedp_immersion_neck_limit",
        "lbdl_document_overview",
    ],
    "single_doc_random_pdf_core": [
        "lbdl_document_overview",
        "lbdl_document_type",
        "ocha_incident_document_overview",
        "model_report_niger_document_type",
        "pre_injection_checklist_live_vaccine",
    ],
    "table_layout_robustness_core": [
        "health_questionnaire_table1_sensitivity",
        "opioid_manager_appendix_b_adverse_scale",
        "ajmedp_immersion_neck_limit",
        "ct_follow_up_improvement",
        "pre_injection_checklist_side_effects",
    ],
    "form_layout_robustness_core": [
        "health_questionnaire_question5_contexts",
        "pre_injection_checklist_live_vaccine",
        "opioid_manager_appendix_a_optimized",
        "opioid_manager_appendix_c_follow_up_timing",
        "lbdl_document_overview",
    ],
    "processing_layer_core": [
        "health_questionnaire_table1_sensitivity",
        "pre_injection_checklist_live_vaccine",
        "ajmedp_immersion_neck_limit",
        "lbdl_document_overview",
        "model_report_niger_document_type",
    ],
    "processing_strategy_core": [
        "health_questionnaire_table1_sensitivity",
        "pre_injection_checklist_live_vaccine",
        "source_listing_nonmedical_learning_and_incident_response",
        "lbdl_document_overview",
        "model_report_niger_document_type",
    ],
}
DEFAULT_RUNTIME_COMPARISON_CASE_IDS = [
    "symptoms",
    "vitamin_c_normal_populations",
    "vitamin_c_cold_stress",
    "lbdl_document_overview",
    "source_listing_humanitarian_model_reports",
    "compare_niger_chad_model_reports",
    "model_report_niger_justification",
]
RUNTIME_COMPARISON_MODES = (
    "baseline",
    "sentence-transformers",
    "cross-encoder",
    "llm-synthesis",
)
SLICE_STABILITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "checklist_fields": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "legend_lookup": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "follow_up_schedule": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "form_grid": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "document_discovery": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "document_facets": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "query_planning": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "document_inventory": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "inventory_summary": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "answer_mode_document_level": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "answer_mode_cross_document": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "answer_mode_grounded_evidence": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "answer_contract": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "document_family_reasoning": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "inventory_coverage": {"mrr": 1.0, "avg_keyword_coverage": 0.95, "negative_success_rate": 1.0},
    "relationship_reasoning": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "model_report_family": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
}
LAYER_STABILITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "processing": {
        "avg_metadata_completeness": 0.7,
        "avg_strategy_signal_rate": 0.75,
    },
    "retrieval": {
        "avg_recall_at_k": 1.0,
        "mrr": 1.0,
    },
    "answer_faithfulness": {
        "avg_supported_sentence_ratio": 1.0,
        "avg_keyword_coverage": 0.95,
    },
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


def _support_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{3,}", _normalize_surface(text))
        if token not in {"document", "file", "files", "source", "sources", "relevant", "include", "includes"}
    }


def _sentence_supported_by_fragments(sentence: str, support_fragments: list[str]) -> bool:
    normalized_sentence = _normalize_surface(sentence)
    sentence_tokens = _support_tokens(sentence)
    if not sentence_tokens:
        return False
    for fragment in support_fragments:
        normalized_fragment = _normalize_surface(fragment)
        if normalized_sentence in normalized_fragment:
            return True
        fragment_tokens = _support_tokens(fragment)
        if not fragment_tokens:
            continue
        overlap = sentence_tokens.intersection(fragment_tokens)
        if len(overlap) >= max(2, min(len(sentence_tokens), len(fragment_tokens)) // 2):
            return True
    return False


def _case_slice_labels(case: dict) -> list[str]:
    case_id = case["case_id"]
    case_type = case.get("case_type", "grounded")
    query_lower = case["query"].lower()

    labels = ["ocr_derived" if case_id.startswith("ct_") else "native_text"]
    labels.append(case_type)

    discovery_like = any(
        tag in case.get("case_tags", [])
        for tag in (
            "document_overview",
            "document_routing",
            "source_listing",
            "source_justification",
            "cross_document",
            "document_facets",
        )
    )
    if discovery_like:
        labels.append("query_planning")
    if any(
        tag in case.get("case_tags", [])
        for tag in (
            "document_discovery",
            "document_overview",
            "document_routing",
            "source_listing",
            "source_justification",
            "cross_document",
            "document_facets",
            "ambiguous_routing",
        )
    ):
        labels.append("document_inventory")
    if any(
        tag in case.get("case_tags", [])
        for tag in (
            "document_overview",
            "document_routing",
            "source_listing",
            "source_justification",
        )
    ):
        labels.append("inventory_summary")
        labels.append("inventory_coverage")
    if any(
        tag in case.get("case_tags", [])
        for tag in (
            "document_overview",
            "document_routing",
            "source_listing",
            "source_justification",
            "cross_document",
            "document_facets",
        )
    ):
        labels.append("document_family_reasoning")

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
    plan = plan_query(query)
    return resolve_preferred_source_doc_id(
        query,
        query_class=plan.query_class,
        query_intent=plan.query_intent,
        planned_preferred_doc_id=plan.preferred_doc_id,
    )


def _result_slice_labels(grounded_answer: GroundedAnswer, base_labels: list[str]) -> list[str]:
    labels = set(base_labels)
    answer_mode = grounded_answer.answer_trace.get("answer_mode")
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
    if answer_mode in {"document_overview", "document_routing", "source_justification"}:
        labels.add("answer_mode_document_level")
    elif answer_mode == "cross_document_compare" or answer_mode == "source_listing":
        labels.add("answer_mode_cross_document")
    elif answer_mode == "grounded_evidence":
        labels.add("answer_mode_grounded_evidence")
    answer_contract = grounded_answer.answer_trace.get("answer_contract") or {}
    if answer_contract:
        labels.add("answer_contract")
    if answer_contract.get("coverage_terms") or answer_contract.get("summary_type"):
        labels.add("inventory_coverage")
    if answer_contract.get("relationship"):
        labels.add("relationship_reasoning")
    return sorted(labels)


def _chunk_snapshot(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_title": chunk.section_title,
        "section_path": chunk.section_path,
        "section_kind": chunk.section_kind,
        "section_role": chunk.section_role,
        "chunk_type": chunk.chunk_type,
        "chunk_strategy": chunk.chunk_strategy,
        "section_content_hints": chunk.section_content_hints,
        "layout_signals": chunk.layout_signals,
        "extraction_method": chunk.extraction_method,
        "text_source": chunk.text_source,
        "text_quality_score": chunk.text_quality_score,
        "source_block_roles": chunk.source_block_roles,
        "source_block_kinds": chunk.source_block_kinds,
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
            "support_trace": grounded_answer.answer_trace.get("support_trace", []),
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


def build_llm_judge_prompt(
    *,
    question: str,
    answer: str,
    source_context: list[str],
) -> str:
    """Build a strict faithfulness judge prompt without invoking an LLM."""
    rules = "\n".join(f"- {rule}" for rule in LLM_JUDGE_RULES)
    context = "\n".join(f"[source {index}] {fragment}" for index, fragment in enumerate(source_context, start=1))
    output_schema = json.dumps(LLM_JUDGE_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return (
        f"Prompt template: {LLM_JUDGE_PROMPT_TEMPLATE_ID}\n\n"
        "Task:\n"
        "Evaluate whether the answer is faithful to the provided source context.\n\n"
        "Rules:\n"
        f"{rules}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Source context:\n"
        f"{context}\n\n"
        "Return JSON matching this schema:\n"
        f"{output_schema}"
    )


def _llm_judge_prompt_contract(
    *,
    debug_case: dict[str, Any],
    source_context: list[str],
    runtime_payload: dict[str, object] | None = None,
) -> dict[str, Any]:
    prompt = build_llm_judge_prompt(
        question=str(debug_case.get("query", "")),
        answer=str(debug_case["answer"].get("full_answer", "")),
        source_context=source_context,
    )
    runtime_invoked = bool((runtime_payload or {}).get("invoked"))
    return {
        "template_id": LLM_JUDGE_PROMPT_TEMPLATE_ID,
        "runtime": "local_command" if runtime_invoked else "not_invoked",
        "judge_model": None,
        "grounding_rules": list(LLM_JUDGE_RULES),
        "output_schema": dict(LLM_JUDGE_OUTPUT_SCHEMA),
        "source_context_count": len(source_context),
        "prompt_char_count": len(prompt),
        "outside_knowledge_allowed": False,
        "strict_json_required": True,
        "prompt_preview": _preview_text(prompt, limit=500),
    }


def _llm_judge_runtime_payload(prompt: str) -> dict[str, object]:
    result = run_prompt_command(prompt, LLM_JUDGE_COMMAND_ENV)
    payload = prompt_command_payload(result)
    payload["env_var"] = LLM_JUDGE_COMMAND_ENV
    payload["provider"] = "local_command" if result.configured else None
    payload["json_valid"] = False
    payload["parsed_json"] = None
    payload["strict_json_parser"] = {}

    if result.status == "ok" and result.stdout:
        parsed_output = parse_strict_json_output(result.stdout, require_object=True)
        payload["strict_json_parser"] = parsed_json_payload(parsed_output)
        if parsed_output.ok:
            payload["json_valid"] = True
            payload["parsed_json"] = parsed_output.value
        else:
            payload["status"] = parsed_output.status
    return payload


def _faithfulness_audit_record(debug_case: dict[str, Any]) -> dict[str, Any]:
    answer_mode = str(debug_case["answer"]["trace"].get("answer_mode", "grounded_evidence"))
    support_trace = debug_case["answer"].get("support_trace", [])
    if answer_mode in {"document_overview", "document_routing", "source_justification", "source_listing", "cross_document_compare"}:
        answer_sentences = _split_answer_sentences(debug_case["answer"]["full_answer"])
        support_fragments: list[str] = []
        for item in support_trace:
            inventory_summary = item.get("inventory_summary")
            if isinstance(inventory_summary, str) and inventory_summary:
                support_fragments.append(inventory_summary)
            support_fragments.extend(str(fragment) for fragment in item.get("support_fragments", []))
            support_fragments.extend(str(fragment) for fragment in item.get("support_sentences", []))
            support_fragments.extend(str(fragment) for fragment in item.get("summary_cues", []))
            support_fragments.extend(str(fragment) for fragment in item.get("section_titles", []))
            support_fragments.extend(str(fragment) for fragment in item.get("coverage_terms", []))
            support_fragments.extend(str(fragment) for fragment in item.get("matched_terms", []))
    else:
        answer_sentences = [
            item["sentence"] for item in debug_case["answer"]["evidence_snapshots"]
        ]
        support_fragments = list(answer_sentences)

    support_corpus = " ".join(support_fragments)
    normalized_support = _normalize_surface(support_corpus)
    supported = []
    unsupported = []
    for sentence in answer_sentences:
        if _normalize_surface(sentence) in normalized_support or _sentence_supported_by_fragments(sentence, support_fragments):
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    supported_ratio = (len(supported) / len(answer_sentences)) if answer_sentences else 0.0
    llm_judge_prompt = build_llm_judge_prompt(
        question=str(debug_case.get("query", "")),
        answer=str(debug_case["answer"].get("full_answer", "")),
        source_context=support_fragments[:12],
    )
    llm_judge_runtime = _llm_judge_runtime_payload(llm_judge_prompt)
    return {
        "case_id": debug_case["case_id"],
        "supported_sentence_ratio": supported_ratio,
        "supported_sentences": supported,
        "unsupported_sentences": unsupported,
        "claim_alignment": debug_case["answer"].get("trace", {}).get("claim_alignment", {}),
        "evidence_preview": support_fragments[:6],
        "llm_judge_prompt_contract": _llm_judge_prompt_contract(
            debug_case=debug_case,
            source_context=support_fragments[:12],
            runtime_payload=llm_judge_runtime,
        ),
        "llm_judge_runtime": llm_judge_runtime,
    }


def _chunk_like_has(chunk: Any, key: str) -> bool:
    if isinstance(chunk, dict):
        value = chunk.get(key)
    else:
        value = getattr(chunk, key, None)
    if isinstance(value, list):
        return bool(value)
    return value is not None and value != ""


def _processing_layer_record(chunks: list[Any]) -> dict[str, Any]:
    if not chunks:
        return {
            "pass": False,
            "chunk_count": 0,
            "metadata_completeness": 0.0,
            "structure_signal_rate": 0.0,
            "strategy_signal_rate": 0.0,
            "quality_signal_rate": 0.0,
        }

    structure_hits = 0
    strategy_hits = 0
    quality_hits = 0
    source_hits = 0
    for chunk in chunks:
        if (
            _chunk_like_has(chunk, "section_role")
            or _chunk_like_has(chunk, "section_kind")
            or _chunk_like_has(chunk, "section_path")
            or _chunk_like_has(chunk, "section_content_hints")
            or _chunk_like_has(chunk, "layout_signals")
        ):
            structure_hits += 1
        if _chunk_like_has(chunk, "chunk_strategy") or _chunk_like_has(chunk, "chunk_type"):
            strategy_hits += 1
        if _chunk_like_has(chunk, "text_quality_score") or _chunk_like_has(chunk, "quality_score"):
            quality_hits += 1
        if (
            _chunk_like_has(chunk, "source_block_roles")
            or _chunk_like_has(chunk, "source_block_kinds")
            or _chunk_like_has(chunk, "extraction_method")
            or _chunk_like_has(chunk, "text_source")
        ):
            source_hits += 1

    chunk_count = len(chunks)
    metadata_completeness = (
        structure_hits + strategy_hits + quality_hits + source_hits
    ) / float(chunk_count * 4)
    structure_signal_rate = structure_hits / float(chunk_count)
    strategy_signal_rate = strategy_hits / float(chunk_count)
    quality_signal_rate = quality_hits / float(chunk_count)
    return {
        "pass": (
            metadata_completeness >= 0.75
            and structure_signal_rate >= 0.75
            and strategy_signal_rate >= 0.75
        ),
        "chunk_count": chunk_count,
        "metadata_completeness": metadata_completeness,
        "structure_signal_rate": structure_signal_rate,
        "strategy_signal_rate": strategy_signal_rate,
        "quality_signal_rate": quality_signal_rate,
    }


def _retrieval_layer_record(retrieval_result: dict[str, Any], answer_result: dict[str, Any]) -> dict[str, Any]:
    case_type = retrieval_result.get("case_type", "grounded")
    if case_type == "negative":
        return {
            "pass": bool(answer_result.get("negative_success")),
            "evaluation_level": retrieval_result.get("evaluation_level", "chunk"),
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
        }
    recall = float(retrieval_result.get("recall_at_k") or 0.0)
    rr = float(retrieval_result.get("reciprocal_rank") or 0.0)
    return {
        "pass": recall >= 1.0 and rr >= 1.0,
        "evaluation_level": retrieval_result.get("evaluation_level", "chunk"),
        "precision_at_k": float(retrieval_result.get("precision_at_k") or 0.0),
        "recall_at_k": recall,
        "reciprocal_rank": rr,
    }


def _answer_faithfulness_layer_record(
    debug_case: dict[str, Any],
    faithfulness_record: dict[str, Any] | None,
) -> dict[str, Any]:
    case_type = debug_case.get("case_type", "grounded")
    answer = debug_case["answer"]
    if case_type == "negative":
        return {
            "pass": bool(answer.get("negative_success")),
            "supported_sentence_ratio": None,
            "keyword_coverage": None,
            "abstained": bool(answer.get("abstained")),
        }
    supported_sentence_ratio = (
        float(faithfulness_record.get("supported_sentence_ratio"))
        if faithfulness_record is not None
        else 0.0
    )
    keyword_coverage = float(answer.get("keyword_coverage") or 0.0)
    abstained = bool(answer.get("abstained"))
    return {
        "pass": (not abstained) and keyword_coverage >= 1.0 and supported_sentence_ratio >= 1.0,
        "supported_sentence_ratio": supported_sentence_ratio,
        "keyword_coverage": keyword_coverage,
        "abstained": abstained,
    }


def _layer_summary(debug_cases: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for layer_name in ("processing", "retrieval", "answer_faithfulness"):
        layer_records = [item["layers"][layer_name] for item in debug_cases if layer_name in item.get("layers", {})]
        pass_case_ids = [
            item["case_id"]
            for item in debug_cases
            if item.get("layers", {}).get(layer_name, {}).get("pass")
        ]
        failing_case_ids = [
            item["case_id"]
            for item in debug_cases
            if not item.get("layers", {}).get(layer_name, {}).get("pass")
        ]
        summary: dict[str, Any] = {
            "case_count": len(layer_records),
            "pass_count": len(pass_case_ids),
            "pass_rate": (len(pass_case_ids) / len(layer_records)) if layer_records else 0.0,
            "failing_case_count": len(failing_case_ids),
            "failing_case_ids": failing_case_ids,
        }
        if layer_name == "processing":
            summary.update(
                {
                    "avg_metadata_completeness": _average(
                        [float(item.get("metadata_completeness", 0.0)) for item in layer_records]
                    ),
                    "avg_structure_signal_rate": _average(
                        [float(item.get("structure_signal_rate", 0.0)) for item in layer_records]
                    ),
                    "avg_strategy_signal_rate": _average(
                        [float(item.get("strategy_signal_rate", 0.0)) for item in layer_records]
                    ),
                }
            )
        elif layer_name == "retrieval":
            grounded_records = [
                item for item, debug_case in zip(layer_records, debug_cases)
                if debug_case.get("case_type") != "negative"
            ]
            summary.update(
                {
                    "avg_recall_at_k": _average(
                        [float(item.get("recall_at_k", 0.0) or 0.0) for item in grounded_records]
                    ),
                    "mrr": _average(
                        [float(item.get("reciprocal_rank", 0.0) or 0.0) for item in grounded_records]
                    ),
                }
            )
        else:
            grounded_records = [
                item for item, debug_case in zip(layer_records, debug_cases)
                if debug_case.get("case_type") != "negative"
            ]
            summary.update(
                {
                    "avg_supported_sentence_ratio": _average(
                        [float(item.get("supported_sentence_ratio", 0.0) or 0.0) for item in grounded_records]
                    ),
                    "avg_keyword_coverage": _average(
                        [float(item.get("keyword_coverage", 0.0) or 0.0) for item in grounded_records]
                    ),
                }
            )
        summaries[layer_name] = summary
    summaries["all_pass"] = all(
        summary.get("failing_case_count", 0) == 0
        for name, summary in summaries.items()
        if name != "all_pass"
    )
    return summaries


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
    judge_invoked_case_count = sum(
        1 for item in records if item.get("llm_judge_runtime", {}).get("invoked")
    )
    judge_valid_json_count = sum(
        1 for item in records if item.get("llm_judge_runtime", {}).get("json_valid")
    )
    contract_validation = _faithfulness_contract_validation(records)
    return {
        "sampled_case_count": len(records),
        "avg_supported_sentence_ratio": _average(
            [item["supported_sentence_ratio"] for item in records]
        ),
        "failing_case_count": len(failing_case_ids),
        "failing_case_ids": failing_case_ids,
        "recommend_llm_judge": len(failing_case_ids) > 0,
        "llm_judge_prompt_contract": {
            "template_id": LLM_JUDGE_PROMPT_TEMPLATE_ID,
            "runtime": "local_command" if judge_invoked_case_count else "not_invoked",
            "sampled_prompt_count": len(records),
            "judge_invoked_case_count": judge_invoked_case_count,
            "judge_valid_json_count": judge_valid_json_count,
            "outside_knowledge_allowed": False,
            "strict_json_required": True,
        },
        "contract_validation": contract_validation,
        "cases": records,
    }


def _faithfulness_contract_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, object]] = []
    for record in records:
        case_id = str(record.get("case_id", ""))
        contract = record.get("llm_judge_prompt_contract", {})
        runtime = record.get("llm_judge_runtime", {})
        checks.extend(
            [
                {
                    "case_id": case_id,
                    "name": "judge_template_id",
                    "passed": contract.get("template_id") == LLM_JUDGE_PROMPT_TEMPLATE_ID,
                },
                {
                    "case_id": case_id,
                    "name": "judge_forbids_outside_knowledge",
                    "passed": contract.get("outside_knowledge_allowed") is False,
                },
                {
                    "case_id": case_id,
                    "name": "judge_requires_strict_json",
                    "passed": contract.get("strict_json_required") is True,
                },
                {
                    "case_id": case_id,
                    "name": "judge_has_source_context",
                    "passed": int(contract.get("source_context_count") or 0) > 0,
                },
                {
                    "case_id": case_id,
                    "name": "runtime_reports_parser_contract",
                    "passed": (
                        not runtime.get("invoked")
                        or isinstance(runtime.get("strict_json_parser"), dict)
                    ),
                },
            ]
        )
    failed = [item for item in checks if not item["passed"]]
    return {
        "all_pass": not failed,
        "check_count": len(checks),
        "failed_checks": failed,
    }


def evaluate_retrieval_case(
    case: dict,
    index_dir: Path,
    chunk_root: Path | None,
    k: int,
    use_lightweight_rerank: bool = True,
) -> dict:
    """Evaluate retrieval metrics for a single query."""
    if chunk_root is not None:
        hits, _ = retrieve_top_k_with_neighbors(
            query=case["query"],
            index_dir=index_dir,
            chunk_root=chunk_root,
            k=k,
            use_lightweight_rerank=use_lightweight_rerank,
        )
    else:
        hits = retrieve_top_k(
            query=case["query"],
            index_dir=index_dir,
            k=k,
            use_lightweight_rerank=use_lightweight_rerank,
        )
    retrieved_ids = [chunk.chunk_id for chunk in hits]
    retrieved_doc_ids = _ordered_unique([chunk.doc_id for chunk in hits])
    relevant = set(case.get("relevant_chunk_ids", []))
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
            if (
                metric_name == "negative_success_rate"
                and int(slice_summary.get("negative_case_count", 0)) == 0
            ):
                continue
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


def _evaluate_layer_stability(layer_summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failed_layers: list[str] = []

    for layer_name, thresholds in LAYER_STABILITY_THRESHOLDS.items():
        summary = layer_summary.get(layer_name)
        if summary is None:
            checks[layer_name] = {
                "present": False,
                "pass": False,
                "reason": "layer missing from current evaluation report",
                "thresholds": thresholds,
            }
            failed_layers.append(layer_name)
            continue

        failed_metrics: dict[str, dict[str, float]] = {}
        for metric_name, min_value in thresholds.items():
            actual_value = float(summary.get(metric_name, 0.0) or 0.0)
            if actual_value < min_value:
                failed_metrics[metric_name] = {
                    "actual": actual_value,
                    "required_min": min_value,
                }

        passed = not failed_metrics
        checks[layer_name] = {
            "present": True,
            "pass": passed,
            "thresholds": thresholds,
            "failed_metrics": failed_metrics,
        }
        if not passed:
            failed_layers.append(layer_name)

    return {
        "all_pass": not failed_layers,
        "failed_layers": failed_layers,
        "checks": checks,
    }


def _architecture_gates(
    *,
    summary: dict[str, Any],
    layer_stability: dict[str, Any],
    slice_stability: dict[str, Any],
    faithfulness_audit: dict[str, Any],
    is_default_eval_suite: bool,
) -> dict[str, Any]:
    faithfulness_pass = not bool(faithfulness_audit.get("recommend_llm_judge"))
    warning_free = int(summary.get("warning_case_count", 0)) == 0
    layer_pass = bool(layer_stability.get("all_pass"))
    slice_pass = bool(slice_stability.get("all_pass")) if is_default_eval_suite else None

    reasons: list[str] = []
    if not warning_free:
        reasons.append("benchmark warnings present")
    if not layer_pass:
        reasons.append("layer stability thresholds not met")
    if is_default_eval_suite and not bool(slice_stability.get("all_pass")):
        reasons.append("slice stability thresholds not met")
    if not faithfulness_pass:
        reasons.append("faithfulness audit recommends deeper review")

    all_pass = warning_free and layer_pass and faithfulness_pass
    if is_default_eval_suite:
        all_pass = all_pass and bool(slice_stability.get("all_pass"))

    return {
        "all_pass": all_pass,
        "warning_free": warning_free,
        "layer_stability_pass": layer_pass,
        "slice_stability_pass": slice_pass,
        "faithfulness_pass": faithfulness_pass,
        "is_default_eval_suite": is_default_eval_suite,
        "reasons": reasons,
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
    default_eval_path = ensure_default_eval_cases(eval_dir)
    if eval_path is None:
        eval_path = default_eval_path
    else:
        eval_path = eval_path.expanduser().resolve()
    is_default_eval_suite = eval_path == default_eval_path

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
        relevant = set(case.get("relevant_chunk_ids", []))
        case_type = case.get("case_type", "grounded")
        baseline_retrieval_results.append(
            evaluate_retrieval_case(
                case=case,
                index_dir=index_dir,
                chunk_root=chunk_root,
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
    full_faithfulness = {
        item["case_id"]: _faithfulness_audit_record(item)
        for item in debug_cases
        if item.get("case_type") != "negative"
    }
    for debug_case in debug_cases:
        case_id = debug_case["case_id"]
        processing_chunks = debug_case["retrieval"]["top_k_snapshots"] + debug_case["retrieval"]["expanded_snapshots"]
        debug_case["layers"] = {
            "processing": _processing_layer_record(processing_chunks),
            "retrieval": _retrieval_layer_record(
                next(item for item in retrieval_results if item["case_id"] == case_id),
                next(item for item in answer_results if item["case_id"] == case_id),
            ),
            "answer_faithfulness": _answer_faithfulness_layer_record(
                debug_case,
                full_faithfulness.get(case_id),
            ),
        }
    layer_summary = _layer_summary(debug_cases)
    layer_stability = _evaluate_layer_stability(layer_summary)
    summary = _summarize_retrieval_results(retrieval_results, answer_results)
    baseline_summary = _summarize_retrieval_results(baseline_retrieval_results, answer_results)
    architecture_gates = _architecture_gates(
        summary=summary,
        layer_stability=layer_stability,
        slice_stability=slice_stability,
        faithfulness_audit=faithfulness_audit,
        is_default_eval_suite=is_default_eval_suite,
    )
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
        "layer_summary": layer_summary,
        "layer_stability": layer_stability,
        "slices": slices,
        "slice_stability": slice_stability,
        "architecture_gates": architecture_gates,
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
        retrieval_result = evaluate_retrieval_case(case=case, index_dir=index_dir, chunk_root=chunk_root, k=k)
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


def _load_all_chunk_records(chunk_root: Path) -> list[ChunkRecord]:
    chunk_root = chunk_root.expanduser().resolve()
    chunks: list[ChunkRecord] = []
    for chunk_path in sorted(chunk_root.glob("*/*.json")):
        data = json.loads(chunk_path.read_text(encoding="utf-8"))
        chunks.append(ChunkRecord.model_validate(data))
    return chunks


def _with_runtime_env(updates: dict[str, str | None]):
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_runtime_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _retrieval_result_from_grounded_answer(
    *,
    case: dict[str, Any],
    grounded_answer: GroundedAnswer,
    k: int,
) -> dict[str, Any]:
    retrieved_ids = [chunk.chunk_id for chunk in grounded_answer.top_k_hits]
    retrieved_doc_ids = _ordered_unique([chunk.doc_id for chunk in grounded_answer.top_k_hits])
    case_type = case.get("case_type", "grounded")
    if case_type == "negative":
        return {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "evaluation_level": "document" if case.get("relevant_doc_ids") else "chunk",
            "retrieved_ids": retrieved_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
        }

    relevant_doc_ids = set(case.get("relevant_doc_ids", []))
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

    relevant = set(case.get("relevant_chunk_ids", []))
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


def _answer_result_from_grounded_answer(
    *,
    case: dict[str, Any],
    grounded_answer: GroundedAnswer,
) -> dict[str, Any]:
    case_type = case.get("case_type", "grounded")
    keyword_eval = _keyword_matches(
        grounded_answer.answer,
        case.get("expected_keywords", []),
    )
    abstained = grounded_answer.answer.startswith("No grounded answer")
    return {
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


def _runtime_signal_summary(answers: list[GroundedAnswer]) -> dict[str, Any]:
    backend_counts: dict[str, int] = {}
    cross_encoder_fallback_count = 0
    llm_configured_count = 0
    llm_invoked_count = 0
    llm_used_count = 0
    for answer in answers:
        for chunk in answer.top_k_hits + answer.expanded_hits:
            backend_code = chunk.retrieval_signals.get("rerank_backend_code")
            backend = {
                0.0: "heuristic",
                1.0: "lightweight",
                2.0: "cross_encoder",
            }.get(backend_code, "unknown")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if chunk.retrieval_signals.get("cross_encoder_fallback"):
                cross_encoder_fallback_count += 1
        synthesis_runtime = answer.answer_trace.get("synthesis_runtime", {})
        if synthesis_runtime.get("configured"):
            llm_configured_count += 1
        if synthesis_runtime.get("invoked"):
            llm_invoked_count += 1
        if synthesis_runtime.get("used_for_final_answer"):
            llm_used_count += 1
    return {
        "rerank_backend_counts": backend_counts,
        "cross_encoder_fallback_chunk_count": cross_encoder_fallback_count,
        "llm_configured_case_count": llm_configured_count,
        "llm_invoked_case_count": llm_invoked_count,
        "llm_used_case_count": llm_used_count,
    }


def _evaluate_runtime_mode(
    *,
    mode: str,
    cases: list[dict[str, Any]],
    index_dir: Path,
    chunk_root: Path,
    k: int,
    index_manifest: dict[str, Any],
    env_updates: dict[str, str | None],
) -> dict[str, Any]:
    previous_env = _with_runtime_env(env_updates)
    try:
        retrieval_results: list[dict[str, Any]] = []
        answer_results: list[dict[str, Any]] = []
        case_results: list[dict[str, Any]] = []
        grounded_answers: list[GroundedAnswer] = []
        failed_case_ids: list[str] = []
        for case in cases:
            grounded_answer = answer_query_with_retrieval(
                query=case["query"],
                index_dir=index_dir,
                chunk_root=chunk_root,
                k=k,
                use_lightweight_rerank=True,
            )
            grounded_answers.append(grounded_answer)
            retrieval_result = _retrieval_result_from_grounded_answer(
                case=case,
                grounded_answer=grounded_answer,
                k=k,
            )
            answer_result = _answer_result_from_grounded_answer(
                case=case,
                grounded_answer=grounded_answer,
            )
            status = _regression_case_status(
                case.get("case_type", "grounded"),
                retrieval_result,
                answer_result,
            )
            if status != "pass":
                failed_case_ids.append(case["case_id"])
            retrieval_results.append(retrieval_result)
            answer_results.append(answer_result)
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case.get("case_type", "grounded"),
                    "status": status,
                    "retrieval": retrieval_result,
                    "answer": {
                        "keyword_coverage": answer_result.get("keyword_coverage"),
                        "abstained": answer_result.get("abstained"),
                        "negative_success": answer_result.get("negative_success"),
                        "answer_preview": _preview_text(str(answer_result.get("answer", "")), limit=240),
                    },
                    "runtime": {
                        "synthesis_runtime": grounded_answer.answer_trace.get("synthesis_runtime", {}),
                        "claim_alignment": grounded_answer.answer_trace.get("claim_alignment", {}),
                    },
                }
            )

        summary = _summarize_retrieval_results(retrieval_results, answer_results)
        runtime_signals = _runtime_signal_summary(grounded_answers)
        return {
            "mode": mode,
            "case_count": len(cases),
            "pass_count": len(cases) - len(failed_case_ids),
            "fail_count": len(failed_case_ids),
            "failed_case_ids": failed_case_ids,
            "all_pass": len(failed_case_ids) == 0,
            "summary": summary,
            "index_manifest": {
                "embedding_backend": index_manifest.get("embedding_backend"),
                "embedding_model": index_manifest.get("embedding_model"),
                "embedding_fallback_reason": index_manifest.get("embedding_fallback_reason"),
                "chunk_count": index_manifest.get("chunk_count"),
            },
            "runtime_signals": runtime_signals,
            "case_results": case_results,
        }
    finally:
        _restore_runtime_env(previous_env)


def _runtime_mode_deltas(
    baseline: dict[str, Any],
    mode_result: dict[str, Any],
) -> dict[str, float | int]:
    base_summary = baseline.get("summary", {})
    summary = mode_result.get("summary", {})
    return {
        "pass_count_delta": int(mode_result.get("pass_count", 0)) - int(baseline.get("pass_count", 0)),
        "avg_precision_at_k_delta": round(
            float(summary.get("avg_precision_at_k") or 0.0)
            - float(base_summary.get("avg_precision_at_k") or 0.0),
            4,
        ),
        "avg_recall_at_k_delta": round(
            float(summary.get("avg_recall_at_k") or 0.0)
            - float(base_summary.get("avg_recall_at_k") or 0.0),
            4,
        ),
        "mrr_delta": round(
            float(summary.get("mrr") or 0.0) - float(base_summary.get("mrr") or 0.0),
            4,
        ),
        "avg_keyword_coverage_delta": round(
            float(summary.get("avg_keyword_coverage") or 0.0)
            - float(base_summary.get("avg_keyword_coverage") or 0.0),
            4,
        ),
    }


def _runtime_promotion_gate(
    *,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "candidate_mode": candidate.get("mode") if candidate else None,
            "promotable": False,
            "checks": [],
            "reasons": ["baseline or candidate result is missing"],
        }

    baseline_summary = baseline.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    candidate_manifest = candidate.get("index_manifest", {})
    checks = [
        {
            "name": "candidate_is_active",
            "passed": candidate_manifest.get("embedding_backend") == "sentence-transformers",
            "details": {
                "embedding_backend": candidate_manifest.get("embedding_backend"),
                "embedding_model": candidate_manifest.get("embedding_model"),
                "embedding_fallback_reason": candidate_manifest.get("embedding_fallback_reason"),
            },
        },
        {
            "name": "no_pass_count_regression",
            "passed": int(candidate.get("pass_count", 0)) >= int(baseline.get("pass_count", 0)),
            "details": {
                "baseline_pass_count": baseline.get("pass_count", 0),
                "candidate_pass_count": candidate.get("pass_count", 0),
            },
        },
        {
            "name": "recall_not_lower",
            "passed": float(candidate_summary.get("avg_recall_at_k") or 0.0)
            >= float(baseline_summary.get("avg_recall_at_k") or 0.0),
            "details": {
                "baseline_avg_recall_at_k": baseline_summary.get("avg_recall_at_k"),
                "candidate_avg_recall_at_k": candidate_summary.get("avg_recall_at_k"),
            },
        },
        {
            "name": "mrr_not_lower",
            "passed": float(candidate_summary.get("mrr") or 0.0)
            >= float(baseline_summary.get("mrr") or 0.0),
            "details": {
                "baseline_mrr": baseline_summary.get("mrr"),
                "candidate_mrr": candidate_summary.get("mrr"),
            },
        },
        {
            "name": "warnings_not_higher",
            "passed": int(candidate_summary.get("warning_case_count") or 0)
            <= int(baseline_summary.get("warning_case_count") or 0),
            "details": {
                "baseline_warning_case_count": baseline_summary.get("warning_case_count"),
                "candidate_warning_case_count": candidate_summary.get("warning_case_count"),
            },
        },
    ]
    failed = [item for item in checks if not item["passed"]]
    return {
        "candidate_mode": candidate.get("mode"),
        "promotable": not failed,
        "checks": checks,
        "reasons": [item["name"] for item in failed],
    }


def run_runtime_mode_comparison(
    *,
    index_dir: Path,
    chunk_root: Path,
    eval_dir: Path,
    k: int = 5,
    eval_path: Path | None = None,
    case_ids: list[str] | None = None,
    shard: str | None = None,
    modes: list[str] | None = None,
    all_cases: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Compare default retrieval against optional local model/runtime modes."""
    eval_dir = eval_dir.expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_path is None:
        eval_path = ensure_default_eval_cases(eval_dir)
    else:
        eval_path = eval_path.expanduser().resolve()

    eval_cases = load_eval_cases(eval_path)
    case_map = {item["case_id"]: item for item in eval_cases}
    if all_cases:
        selected_case_ids = [item["case_id"] for item in eval_cases]
    else:
        selected_case_ids = (
            case_ids
            or DEFAULT_REGRESSION_SHARDS.get(shard or "", DEFAULT_RUNTIME_COMPARISON_CASE_IDS)
        )
    selected_modes = modes or list(RUNTIME_COMPARISON_MODES)
    unknown_modes = [mode for mode in selected_modes if mode not in RUNTIME_COMPARISON_MODES]
    missing_case_ids = [case_id for case_id in selected_case_ids if case_id not in case_map]
    selected_cases = [case_map[case_id] for case_id in selected_case_ids if case_id in case_map]

    index_dir = index_dir.expanduser().resolve()
    chunk_root = chunk_root.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="pdf-to-json-rag-runtime-compare-") as workspace:
        workspace_path = Path(workspace)
        mode_index_dirs: dict[str, Path] = {"baseline": index_dir, "cross-encoder": index_dir, "llm-synthesis": index_dir}
        mode_manifests: dict[str, dict[str, Any]] = {}
        baseline_manifest_path = index_dir / "index_manifest.json"
        mode_manifests["baseline"] = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
        mode_manifests["cross-encoder"] = mode_manifests["baseline"]
        mode_manifests["llm-synthesis"] = mode_manifests["baseline"]

        if "sentence-transformers" in selected_modes:
            chunks = _load_all_chunk_records(chunk_root)
            sentence_index_dir = workspace_path / "sentence_transformers_index"
            previous_env = _with_runtime_env({"PDF_TO_JSON_RAG_EMBEDDING_BACKEND": "sentence-transformers"})
            try:
                mode_manifests["sentence-transformers"] = build_local_index(
                    chunks=chunks,
                    index_dir=sentence_index_dir,
                )
            finally:
                _restore_runtime_env(previous_env)
            mode_index_dirs["sentence-transformers"] = sentence_index_dir

        mode_envs = {
            "baseline": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
                "PDF_TO_JSON_RAG_LLM_COMMAND": None,
            },
            "sentence-transformers": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
                "PDF_TO_JSON_RAG_LLM_COMMAND": None,
            },
            "cross-encoder": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": "1",
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
                "PDF_TO_JSON_RAG_LLM_COMMAND": None,
            },
            "llm-synthesis": {
                "PDF_TO_JSON_RAG_USE_CROSS_ENCODER": None,
            },
        }

        mode_results: list[dict[str, Any]] = []
        for mode in selected_modes:
            if mode in unknown_modes:
                continue
            if mode == "sentence-transformers" and mode not in mode_index_dirs:
                continue
            mode_results.append(
                _evaluate_runtime_mode(
                    mode=mode,
                    cases=selected_cases,
                    index_dir=mode_index_dirs[mode],
                    chunk_root=chunk_root,
                    k=k,
                    index_manifest=mode_manifests[mode],
                    env_updates=mode_envs[mode],
                )
            )

    baseline_result = next((item for item in mode_results if item["mode"] == "baseline"), None)
    sentence_transformers_result = next(
        (item for item in mode_results if item["mode"] == "sentence-transformers"),
        None,
    )
    deltas = {
        item["mode"]: _runtime_mode_deltas(baseline_result, item)
        for item in mode_results
        if baseline_result is not None and item["mode"] != "baseline"
    }
    promotion_gates = {}
    if sentence_transformers_result is not None:
        promotion_gates["sentence-transformers"] = _runtime_promotion_gate(
            baseline=baseline_result,
            candidate=sentence_transformers_result,
        )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "selected_shard": shard,
        "all_cases": all_cases,
        "selected_case_ids": selected_case_ids,
        "missing_case_ids": missing_case_ids,
        "unknown_modes": unknown_modes,
        "available_modes": list(RUNTIME_COMPARISON_MODES),
        "case_count": len(selected_cases),
        "mode_results": mode_results,
        "baseline_deltas": deltas,
        "promotion_gates": promotion_gates,
        "all_pass": (
            not missing_case_ids
            and not unknown_modes
            and all(item.get("all_pass") for item in mode_results)
        ),
    }

    report_path = eval_dir / DEFAULT_RUNTIME_COMPARISON_REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path
