"""Retrieval, faithfulness, layer, and audit metrics."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .answering import GroundedAnswer
from .intent_config import resolve_preferred_source_doc_id
from .llm_output import parsed_json_payload, parse_strict_json_output
from .llm_runtime import prompt_command_payload, run_prompt_command
from .query_planning import plan_query
from .schemas import ChunkRecord


from .evaluation_cases import (
    DEFAULT_EVAL_CASES,
    DEFAULT_EVAL_FILENAME,
    DEFAULT_FAITHFULNESS_AUDIT_CASE_IDS,
    DEFAULT_FAITHFULNESS_AUDIT_FILENAME,
    LLM_JUDGE_COMMAND_ENV,
    LLM_JUDGE_OUTPUT_SCHEMA,
    LLM_JUDGE_PROMPT_TEMPLATE_ID,
    LLM_JUDGE_RULES,
)

def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def inspect_chunk_sample(
    chunks: list[ChunkRecord], sample_size: int = 5
) -> list[ChunkRecord]:
    """Return a small chunk sample for manual quality review."""
    return chunks[:sample_size]


def ensure_default_eval_cases(eval_dir: Path) -> Path:
    """Create the default small evaluation set if it does not yet exist."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / DEFAULT_EVAL_FILENAME
    if not eval_path.exists():
        bundled_eval_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "eval"
            / DEFAULT_EVAL_FILENAME
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
                json.dumps(
                    DEFAULT_FAITHFULNESS_AUDIT_CASE_IDS, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
    return audit_path


def load_faithfulness_audit_case_ids(audit_path: Path) -> list[str]:
    audit_path = audit_path.expanduser().resolve()
    if not audit_path.exists():
        raise FileNotFoundError(f"Faithfulness audit file not found: {audit_path}")
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            "Faithfulness audit file must contain a JSON list of case IDs."
        )
    return [str(item) for item in data]


def _keyword_matches(answer_text: str, expected_keywords: list[str]) -> dict:
    answer_lower = answer_text.lower().replace("ﬁ", "fi").replace("ﬂ", "fl")
    answer_compact = "".join(ch for ch in answer_lower if ch.isalnum())
    matched = [
        keyword
        for keyword in expected_keywords
        if (
            keyword.lower().replace("ﬁ", "fi").replace("ﬂ", "fl") in answer_lower
            or "".join(
                ch
                for ch in keyword.lower().replace("ﬁ", "fi").replace("ﬂ", "fl")
                if ch.isalnum()
            )
            in answer_compact
        )
    ]
    return {
        "matched_keywords": matched,
        "keyword_coverage": (len(matched) / len(expected_keywords))
        if expected_keywords
        else 0.0,
    }


def _preview_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _normalize_surface(text: str) -> str:
    return " ".join(text.lower().replace("ﬁ", "fi").replace("ﬂ", "fl").split())


def _split_answer_sentences(answer_text: str) -> list[str]:
    fragments = [
        item.strip()
        for item in answer_text.replace("\n", " ").split(".")
        if item.strip()
    ]
    return [
        fragment if fragment.endswith(".") else f"{fragment}." for fragment in fragments
    ]


def _support_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{3,}", _normalize_surface(text))
        if token
        not in {
            "document",
            "file",
            "files",
            "source",
            "sources",
            "relevant",
            "include",
            "includes",
        }
    }


def _sentence_supported_by_fragments(
    sentence: str, support_fragments: list[str]
) -> bool:
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
    elif (
        case_id.startswith("ajmedp_")
        or "ajmedp" in query_lower
        or "tb med 508" in query_lower
    ):
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
        labels.extend(
            ["humanitarian_data_guidance", "non_medical", "document_discovery"]
        )
    elif (
        case_id.startswith("wat_")
        or "literature review" in query_lower
        or "dennis wat" in query_lower
    ):
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


def _result_slice_labels(
    grounded_answer: GroundedAnswer, base_labels: list[str]
) -> list[str]:
    labels = set(base_labels)
    answer_mode = grounded_answer.answer_trace.get("answer_mode")
    top_doc_ids = {chunk.doc_id for chunk in grounded_answer.top_k_hits}
    expanded_noise = {
        noise for chunk in grounded_answer.expanded_hits for noise in chunk.noise_labels
    }

    preferred_doc_id = _preferred_source_doc_id_from_query(grounded_answer.query)
    if {
        "source_anchored_review",
        "source_anchored_technical",
        "source_anchored_form",
    }.intersection(labels):
        if preferred_doc_id and top_doc_ids == {preferred_doc_id}:
            labels.add("source_locked")
        elif len(top_doc_ids) > 1:
            labels.add("cross_document_mixing")
    if expanded_noise.intersection(
        {"table_reference", "table_like_section", "reference_tail"}
    ):
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
            "top_k_snapshots": [
                _chunk_snapshot(chunk) for chunk in grounded_answer.top_k_hits
            ],
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


def _summarize_retrieval_results(
    retrieval_results: list[dict], answer_results: list[dict]
) -> dict[str, Any]:
    grounded_retrieval = [
        item for item in retrieval_results if item.get("case_type") != "negative"
    ]
    negative_answers = [
        item for item in answer_results if item.get("case_type") == "negative"
    ]
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
        "avg_precision_at_k": _average(
            [item["precision_at_k"] for item in grounded_retrieval]
        ),
        "avg_recall_at_k": _average(
            [item["recall_at_k"] for item in grounded_retrieval]
        ),
        "mrr": _average([item["reciprocal_rank"] for item in grounded_retrieval]),
        "avg_keyword_coverage": _average(
            [
                item["keyword_coverage"]
                for item in answer_results
                if item.get("case_type") != "negative"
            ]
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
    context = "\n".join(
        f"[source {index}] {fragment}"
        for index, fragment in enumerate(source_context, start=1)
    )
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
    answer_mode = str(
        debug_case["answer"]["trace"].get("answer_mode", "grounded_evidence")
    )
    support_trace = debug_case["answer"].get("support_trace", [])
    if answer_mode in {
        "document_overview",
        "document_routing",
        "source_justification",
        "source_listing",
        "cross_document_compare",
    }:
        answer_sentences = _split_answer_sentences(debug_case["answer"]["full_answer"])
        support_fragments: list[str] = []
        for item in support_trace:
            inventory_summary = item.get("inventory_summary")
            if isinstance(inventory_summary, str) and inventory_summary:
                support_fragments.append(inventory_summary)
            support_fragments.extend(
                str(fragment) for fragment in item.get("support_fragments", [])
            )
            support_fragments.extend(
                str(fragment) for fragment in item.get("support_sentences", [])
            )
            support_fragments.extend(
                str(fragment) for fragment in item.get("summary_cues", [])
            )
            support_fragments.extend(
                str(fragment) for fragment in item.get("section_titles", [])
            )
            support_fragments.extend(
                str(fragment) for fragment in item.get("coverage_terms", [])
            )
            support_fragments.extend(
                str(fragment) for fragment in item.get("matched_terms", [])
            )
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
        if _normalize_surface(
            sentence
        ) in normalized_support or _sentence_supported_by_fragments(
            sentence, support_fragments
        ):
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    supported_ratio = (
        (len(supported) / len(answer_sentences)) if answer_sentences else 0.0
    )
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
        "claim_alignment": debug_case["answer"]
        .get("trace", {})
        .get("claim_alignment", {}),
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
        if _chunk_like_has(chunk, "chunk_strategy") or _chunk_like_has(
            chunk, "chunk_type"
        ):
            strategy_hits += 1
        if _chunk_like_has(chunk, "text_quality_score") or _chunk_like_has(
            chunk, "quality_score"
        ):
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


def _retrieval_layer_record(
    retrieval_result: dict[str, Any], answer_result: dict[str, Any]
) -> dict[str, Any]:
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
        "pass": (not abstained)
        and keyword_coverage >= 1.0
        and supported_sentence_ratio >= 1.0,
        "supported_sentence_ratio": supported_sentence_ratio,
        "keyword_coverage": keyword_coverage,
        "abstained": abstained,
    }


def _layer_summary(debug_cases: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for layer_name in ("processing", "retrieval", "answer_faithfulness"):
        layer_records = [
            item["layers"][layer_name]
            for item in debug_cases
            if layer_name in item.get("layers", {})
        ]
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
            "pass_rate": (len(pass_case_ids) / len(layer_records))
            if layer_records
            else 0.0,
            "failing_case_count": len(failing_case_ids),
            "failing_case_ids": failing_case_ids,
        }
        if layer_name == "processing":
            summary.update(
                {
                    "avg_metadata_completeness": _average(
                        [
                            float(item.get("metadata_completeness", 0.0))
                            for item in layer_records
                        ]
                    ),
                    "avg_structure_signal_rate": _average(
                        [
                            float(item.get("structure_signal_rate", 0.0))
                            for item in layer_records
                        ]
                    ),
                    "avg_strategy_signal_rate": _average(
                        [
                            float(item.get("strategy_signal_rate", 0.0))
                            for item in layer_records
                        ]
                    ),
                }
            )
        elif layer_name == "retrieval":
            grounded_records = [
                item
                for item, debug_case in zip(layer_records, debug_cases)
                if debug_case.get("case_type") != "negative"
            ]
            summary.update(
                {
                    "avg_recall_at_k": _average(
                        [
                            float(item.get("recall_at_k", 0.0) or 0.0)
                            for item in grounded_records
                        ]
                    ),
                    "mrr": _average(
                        [
                            float(item.get("reciprocal_rank", 0.0) or 0.0)
                            for item in grounded_records
                        ]
                    ),
                }
            )
        else:
            grounded_records = [
                item
                for item, debug_case in zip(layer_records, debug_cases)
                if debug_case.get("case_type") != "negative"
            ]
            summary.update(
                {
                    "avg_supported_sentence_ratio": _average(
                        [
                            float(item.get("supported_sentence_ratio", 0.0) or 0.0)
                            for item in grounded_records
                        ]
                    ),
                    "avg_keyword_coverage": _average(
                        [
                            float(item.get("keyword_coverage", 0.0) or 0.0)
                            for item in grounded_records
                        ]
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


def _run_faithfulness_audit(
    debug_cases: list[dict[str, Any]], audit_case_ids: list[str]
) -> dict[str, Any]:
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
                    "passed": contract.get("template_id")
                    == LLM_JUDGE_PROMPT_TEMPLATE_ID,
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
