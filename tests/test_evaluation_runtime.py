"""Evaluation layers, runtime comparisons, and LLM contract tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from unittest import mock


from tests.cli_test_support import CliPublicSurfaceTestBase

from pdf_to_json_rag import cli as cli_module
from pdf_to_json_rag.answering import answer_from_chunks
from pdf_to_json_rag.evaluation import (
    _answer_faithfulness_layer_record,
    _evaluate_layer_stability,
    _faithfulness_contract_validation,
    _faithfulness_audit_record,
    _processing_layer_record,
    _retrieval_layer_record,
    build_llm_judge_prompt,
    run_runtime_mode_comparison,
)
from pdf_to_json_rag.llm_output import parse_strict_json_output
from pdf_to_json_rag.llm_runtime import prompt_command_payload, provider_for_env_command
from pdf_to_json_rag.indexing import build_local_index
from pdf_to_json_rag.schemas import ChunkRecord


class EvaluationRuntimeTests(CliPublicSurfaceTestBase):
    def test_opt_in_llm_synthesis_command_can_replace_final_answer(self) -> None:
        fake_llm = self.workspace / "fake_llm.py"
        fake_llm.write_text(
            "import sys\n"
            "sys.stdin.read()\n"
            "print('LLM grounded answer [demo-chunk-1]')\n",
            encoding="utf-8",
        )
        chunk = ChunkRecord(
            doc_id="demo-doc",
            chunk_id="demo-chunk-1",
            source_pdf="demo.pdf",
            text="Safety checks are required before field work.",
            page_start=1,
            page_end=1,
            reading_order_index=0,
            section_title="Safety Checks",
        )

        with mock.patch.dict(
            os.environ,
            {"PDF_TO_JSON_RAG_LLM_COMMAND": f"{sys.executable} {fake_llm}"},
        ):
            result = answer_from_chunks("What safety checks are required?", [chunk])

        self.assertEqual(result.answer, "LLM grounded answer [demo-chunk-1]")
        synthesis_runtime = result.answer_trace["synthesis_runtime"]
        self.assertTrue(synthesis_runtime["configured"])
        self.assertTrue(synthesis_runtime["invoked"])
        self.assertTrue(synthesis_runtime["used_for_final_answer"])
        self.assertEqual(
            result.answer_trace["synthesis_prompt_contract"]["runtime"], "local_command"
        )

    def test_evaluation_layers_distinguish_processing_retrieval_and_faithfulness(
        self,
    ) -> None:
        processing = _processing_layer_record(
            [
                {
                    "section_role": "form",
                    "section_kind": "checklist_section",
                    "section_path": ["Demo", "Checklist"],
                    "chunk_strategy": "form_rows",
                    "text_quality_score": 0.9,
                    "source_block_roles": ["form_field"],
                    "source_block_kinds": ["text"],
                }
            ]
        )
        retrieval = _retrieval_layer_record(
            {
                "case_type": "grounded",
                "evaluation_level": "document",
                "precision_at_k": 1.0,
                "recall_at_k": 1.0,
                "reciprocal_rank": 1.0,
            },
            {"abstained": False},
        )
        answer_faithfulness = _answer_faithfulness_layer_record(
            {
                "case_type": "grounded",
                "answer": {
                    "keyword_coverage": 1.0,
                    "abstained": False,
                },
            },
            {"supported_sentence_ratio": 1.0},
        )
        self.assertTrue(processing["pass"])
        self.assertTrue(retrieval["pass"])
        self.assertTrue(answer_faithfulness["pass"])

    def test_llm_judge_prompt_contract_is_context_only_and_not_invoked(self) -> None:
        prompt = build_llm_judge_prompt(
            question="What is supported?",
            answer="The answer is supported.",
            source_context=["The answer is supported."],
        )
        self.assertIn("Do not use outside knowledge", prompt)
        self.assertIn("Return strict JSON only", prompt)
        self.assertIn("unsupported_sentences", prompt)

        record = _faithfulness_audit_record(
            {
                "case_id": "demo",
                "case_type": "grounded",
                "query": "What is supported?",
                "answer": {
                    "trace": {"answer_mode": "grounded_evidence"},
                    "support_trace": [],
                    "full_answer": "The answer is supported.",
                    "evidence_snapshots": [
                        {
                            "sentence": "The answer is supported.",
                        }
                    ],
                },
            }
        )
        contract = record["llm_judge_prompt_contract"]
        self.assertEqual(contract["template_id"], "faithfulness_context_judge.v1")
        self.assertEqual(contract["runtime"], "not_invoked")
        self.assertFalse(contract["outside_knowledge_allowed"])
        self.assertTrue(contract["strict_json_required"])
        self.assertGreater(contract["prompt_char_count"], 0)
        runtime = record["llm_judge_runtime"]
        self.assertFalse(runtime["configured"])
        self.assertFalse(runtime["invoked"])

    def test_opt_in_llm_judge_command_records_strict_json_result(self) -> None:
        fake_judge = self.workspace / "fake_judge.py"
        fake_judge.write_text(
            "import sys\n"
            "sys.stdin.read()\n"
            "print('```json')\n"
            'print(\'{"faithful": true, "supported_sentence_ratio": 1.0, '
            '"unsupported_sentences": [], "rationale": "Supported by supplied context."}\')\n'
            "print('```')\n",
            encoding="utf-8",
        )

        with mock.patch.dict(
            os.environ,
            {"PDF_TO_JSON_RAG_JUDGE_COMMAND": f"{sys.executable} {fake_judge}"},
        ):
            record = _faithfulness_audit_record(
                {
                    "case_id": "demo",
                    "case_type": "grounded",
                    "query": "What is supported?",
                    "answer": {
                        "trace": {"answer_mode": "grounded_evidence"},
                        "support_trace": [],
                        "full_answer": "The answer is supported.",
                        "evidence_snapshots": [
                            {
                                "sentence": "The answer is supported.",
                            }
                        ],
                    },
                }
            )

        contract = record["llm_judge_prompt_contract"]
        runtime = record["llm_judge_runtime"]
        self.assertEqual(contract["runtime"], "local_command")
        self.assertTrue(runtime["configured"])
        self.assertTrue(runtime["invoked"])
        self.assertTrue(runtime["json_valid"])
        self.assertTrue(runtime["strict_json_parser"]["ok"])
        self.assertEqual(runtime["provider_id"], "local_command")
        self.assertEqual(runtime["provider_kind"], "subprocess")
        self.assertTrue(runtime["parsed_json"]["faithful"])

    def test_strict_json_output_parser_accepts_single_json_fence_only(self) -> None:
        raw = parse_strict_json_output('{"ok": true}')
        fenced = parse_strict_json_output('```json\n{"ok": true}\n```')
        noisy = parse_strict_json_output('answer:\n```json\n{"ok": true}\n```')
        duplicate = parse_strict_json_output(
            '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```'
        )

        self.assertTrue(raw.ok)
        self.assertEqual(raw.output_format, "raw_json")
        self.assertTrue(fenced.ok)
        self.assertEqual(fenced.output_format, "fenced_json")
        self.assertFalse(noisy.ok)
        self.assertEqual(noisy.status, "text_outside_fence")
        self.assertFalse(duplicate.ok)
        self.assertEqual(duplicate.status, "multiple_fenced_blocks")

    def test_prompt_provider_payload_reports_provider_contract(self) -> None:
        provider = provider_for_env_command("PDF_TO_JSON_RAG_TEST_LLM_COMMAND")
        with mock.patch.dict(os.environ, {}, clear=True):
            result = provider.run("hello")

        payload = prompt_command_payload(result)
        self.assertFalse(payload["configured"])
        self.assertFalse(payload["invoked"])
        self.assertEqual(payload["provider_id"], "local_command")
        self.assertEqual(payload["provider_kind"], "subprocess")

    def test_prompt_provider_normalizes_timeout_byte_streams(self) -> None:
        provider = provider_for_env_command("PDF_TO_JSON_RAG_TEST_LLM_COMMAND")
        timeout = subprocess.TimeoutExpired(
            cmd=["fake-llm"],
            timeout=1.0,
            output=b"partial response",
            stderr=b"deadline exceeded",
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    "PDF_TO_JSON_RAG_TEST_LLM_COMMAND": "fake-llm",
                    "PDF_TO_JSON_RAG_LLM_TIMEOUT_SECONDS": "1",
                },
            ),
            mock.patch(
                "pdf_to_json_rag.llm_runtime.subprocess.run", side_effect=timeout
            ),
        ):
            result = provider.run("hello")

        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.stdout, "partial response")
        self.assertEqual(result.stderr_preview, "deadline exceeded")

    def test_faithfulness_contract_validation_gate_passes_on_valid_records(
        self,
    ) -> None:
        record = _faithfulness_audit_record(
            {
                "case_id": "demo",
                "case_type": "grounded",
                "query": "What is supported?",
                "answer": {
                    "trace": {"answer_mode": "grounded_evidence"},
                    "support_trace": [],
                    "full_answer": "The answer is supported.",
                    "evidence_snapshots": [
                        {
                            "sentence": "The answer is supported.",
                        }
                    ],
                },
            }
        )

        validation = _faithfulness_contract_validation([record])
        self.assertTrue(validation["all_pass"])
        self.assertGreaterEqual(validation["check_count"], 5)
        self.assertEqual(validation["failed_checks"], [])

    def test_runtime_mode_comparison_reports_opt_in_llm_usage(self) -> None:
        chunk_root = self.workspace / "chunks"
        doc_chunk_dir = chunk_root / "doc"
        doc_chunk_dir.mkdir(parents=True)
        chunk = ChunkRecord(
            doc_id="doc",
            chunk_id="c1",
            source_pdf="demo.pdf",
            text="Common cold symptoms include cough and fever.",
            page_start=1,
            page_end=1,
            reading_order_index=1,
        )
        (doc_chunk_dir / "c1.json").write_text(
            json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
        index_dir = self.workspace / "index"
        build_local_index([chunk], index_dir=index_dir)
        eval_dir = self.workspace / "eval"
        eval_dir.mkdir()
        eval_path = eval_dir / "cases.json"
        eval_path.write_text(
            json.dumps(
                [
                    {
                        "case_id": "demo_case",
                        "query": "What are common cold symptoms?",
                        "relevant_chunk_ids": ["c1"],
                        "expected_keywords": ["cough", "fever"],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        fake_llm = self.workspace / "fake_llm.py"
        fake_llm.write_text(
            "import sys\n"
            "sys.stdin.read()\n"
            "print('Common cold symptoms include cough and fever [c1].')\n",
            encoding="utf-8",
        )

        with mock.patch.dict(
            os.environ,
            {"PDF_TO_JSON_RAG_LLM_COMMAND": f"{sys.executable} {fake_llm}"},
        ):
            report, report_path = run_runtime_mode_comparison(
                index_dir=index_dir,
                chunk_root=chunk_root,
                eval_dir=eval_dir,
                eval_path=eval_path,
                case_ids=["demo_case"],
                modes=["baseline", "llm-synthesis"],
            )

        self.assertTrue(report_path.exists())
        mode_results = {item["mode"]: item for item in report["mode_results"]}
        self.assertEqual(
            mode_results["baseline"]["runtime_signals"]["llm_used_case_count"], 0
        )
        self.assertEqual(
            mode_results["llm-synthesis"]["runtime_signals"]["llm_used_case_count"], 1
        )
        self.assertGreater(
            mode_results["baseline"]["runtime_signals"]["avg_query_latency_ms"], 0
        )
        self.assertGreater(
            mode_results["llm-synthesis"]["runtime_signals"]["mode_wall_seconds"], 0
        )
        self.assertGreater(
            mode_results["baseline"]["case_results"][0]["runtime"]["query_latency_ms"],
            0,
        )
        self.assertTrue(report["all_pass"])

    def test_runtime_mode_comparison_all_cases_and_promotion_gate(self) -> None:
        chunk_root = self.workspace / "chunks-all-cases"
        doc_chunk_dir = chunk_root / "doc"
        doc_chunk_dir.mkdir(parents=True)
        chunks = [
            ChunkRecord(
                doc_id="doc",
                chunk_id="c1",
                source_pdf="demo.pdf",
                text="Common cold symptoms include cough and fever.",
                page_start=1,
                page_end=1,
                reading_order_index=1,
            ),
            ChunkRecord(
                doc_id="doc",
                chunk_id="c2",
                source_pdf="demo.pdf",
                text="Rest and hydration are common supportive care steps.",
                page_start=1,
                page_end=1,
                reading_order_index=2,
            ),
        ]
        for chunk in chunks:
            (doc_chunk_dir / f"{chunk.chunk_id}.json").write_text(
                json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False),
                encoding="utf-8",
            )
        index_dir = self.workspace / "index-all-cases"
        build_local_index(chunks, index_dir=index_dir)
        eval_dir = self.workspace / "eval-all-cases"
        eval_dir.mkdir()
        eval_path = eval_dir / "cases.json"
        eval_path.write_text(
            json.dumps(
                [
                    {
                        "case_id": "symptoms_case",
                        "query": "What symptoms are mentioned?",
                        "relevant_chunk_ids": ["c1"],
                        "expected_keywords": ["cough", "fever"],
                    },
                    {
                        "case_id": "care_case",
                        "query": "What supportive care is mentioned?",
                        "relevant_chunk_ids": ["c2"],
                        "expected_keywords": ["hydration"],
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report, _ = run_runtime_mode_comparison(
            index_dir=index_dir,
            chunk_root=chunk_root,
            eval_dir=eval_dir,
            eval_path=eval_path,
            modes=["baseline", "sentence-transformers"],
            all_cases=True,
        )

        self.assertTrue(report["all_cases"])
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["selected_case_ids"], ["symptoms_case", "care_case"])
        self.assertIn("sentence-transformers", report["promotion_gates"])
        decision_gate = cli_module._model_decision_gate_from_runtime_report(report)
        self.assertEqual(decision_gate["default_backend"], "auto")
        self.assertFalse(decision_gate["default_change_allowed"])
        self.assertTrue(
            any(
                item["backend"] == "sentence-transformers"
                for item in decision_gate["decisions"]
            )
        )

    def test_layer_stability_passes_for_green_layer_summary(self) -> None:
        stability = _evaluate_layer_stability(
            {
                "processing": {
                    "avg_metadata_completeness": 0.8,
                    "avg_strategy_signal_rate": 1.0,
                },
                "retrieval": {
                    "avg_recall_at_k": 1.0,
                    "mrr": 1.0,
                },
                "answer_faithfulness": {
                    "avg_supported_sentence_ratio": 1.0,
                    "avg_keyword_coverage": 1.0,
                },
            }
        )
        self.assertTrue(stability["all_pass"])
        self.assertEqual(stability["failed_layers"], [])


if __name__ == "__main__":
    unittest.main()
