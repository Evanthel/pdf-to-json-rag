"""Built-in evaluation cases, shards, and benchmark constants."""

from __future__ import annotations


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
    "unknown_document_semantics_core": [
        "lbdl_document_type",
        "lbdl_document_audience",
        "ocha_incident_document_purpose",
        "ocha_incident_document_audience",
        "model_report_niger_document_type",
        "model_report_niger_document_audience",
        "lbdl_document_confidence",
        "ocha_incident_document_confidence",
        "model_report_niger_document_confidence",
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
    "form_grid": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "document_discovery": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "document_facets": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "query_planning": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "document_inventory": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "inventory_summary": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "answer_mode_document_level": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "answer_mode_cross_document": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "answer_mode_grounded_evidence": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "answer_contract": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
    "document_family_reasoning": {"mrr": 1.0, "avg_keyword_coverage": 0.95},
    "inventory_coverage": {
        "mrr": 1.0,
        "avg_keyword_coverage": 0.95,
        "negative_success_rate": 1.0,
    },
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
