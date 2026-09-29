"""Core packaged CLI command and JSON-contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


from tests.cli_test_support import CliPublicSurfaceTestBase, REPO_ROOT

from pdf_to_json_rag import cli as cli_module


class CliPublicSurfaceTests(CliPublicSurfaceTestBase):
    def test_demo_profile_json(self) -> None:
        process = self._run("demo-profile", "--json")
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        profile = payload["result"]["profile"]
        self.assertEqual(profile["name"], "public-safe-local-demo")
        self.assertGreaterEqual(len(profile["workflow"]), 5)

    def test_doctor_after_init(self) -> None:
        self._run("init", "--json")
        process = self._run("doctor", "--json")
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertTrue(result["ready_for_public_cli"])
        self.assertIn("next_steps", result)
        self.assertGreaterEqual(len(result["next_steps"]), 1)
        check_names = {item["name"] for item in result["checks"]}
        self.assertIn("package_metadata_present", check_names)
        self.assertIn("example_assets_present", check_names)
        self.assertIn("demo_pdf_generation_available", check_names)
        self.assertIn("pdf_inspector_available", check_names)
        self.assertIn("pdfplumber_available", check_names)
        self.assertIn("embedding_backend_configured", check_names)
        inspector_check = next(
            item
            for item in result["checks"]
            if item["name"] == "pdf_inspector_available"
        )
        self.assertTrue(inspector_check["passed"])
        self.assertEqual(inspector_check["details"]["version"], "0.2.6")
        self.assertEqual(inspector_check["details"]["requested_mode"], "assist")
        self.assertEqual(inspector_check["details"]["effective_mode"], "assist")
        self.assertEqual(result["runtime"]["embedding"]["requested_backend"], "auto")
        self.assertEqual(
            result["runtime"]["embedding"]["effective_backend"], "hash-fallback"
        )

    def test_runtime_check_reports_default_auto_backend(self) -> None:
        process = self._run("runtime-check", "--json")
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        embedding = payload["result"]["embedding"]
        decision = payload["result"]["runtime_decision"]
        self.assertEqual(payload["result"]["install_context"]["version"], "0.2.0")
        self.assertTrue(
            payload["result"]["install_context"]["module_path"].endswith("cli.py")
        )
        self.assertEqual(embedding["requested_backend"], "auto")
        self.assertEqual(embedding["effective_backend"], "hash-fallback")
        self.assertEqual(decision["default_backend"], "auto")
        self.assertIn("fall back", decision["not_default_reason"])
        self.assertEqual(decision["backend_policy"]["default_backend"]["name"], "auto")
        self.assertEqual(
            decision["backend_policy"]["default_backend"]["fallback_backend"], "hash"
        )
        self.assertEqual(
            decision["backend_policy"]["experimental_backends"][0]["status"],
            "experimental_opt_in",
        )
        self.assertFalse(decision["backend_policy"]["llm_synthesis"]["default_enabled"])
        self.assertEqual(
            payload["result"]["default_policy"]["embedding_backend"], "auto"
        )
        self.assertEqual(
            payload["result"]["default_policy"]["sentence_transformers"],
            "default_when_cached",
        )
        self.assertEqual(payload["result"]["default_policy"]["llm_synthesis"], "opt_in")

    def test_runtime_check_reports_sentence_transformer_env_request(self) -> None:
        env = dict(self.base_env)
        env["PDF_TO_JSON_RAG_EMBEDDING_BACKEND"] = "sentence-transformers"
        env["PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL"] = (
            "definitely-not-cached-local-model"
        )
        process = subprocess.run(
            [sys.executable, "-m", "pdf_to_json_rag", "runtime-check", "--json"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(process.stdout)
        embedding = payload["result"]["embedding"]
        self.assertEqual(embedding["requested_backend"], "sentence-transformers")
        self.assertEqual(embedding["effective_backend"], "hash-fallback")
        self.assertIn("not cached locally", embedding["fallback_reason"])

    def test_runtime_promotion_report_summarizes_saved_gate(self) -> None:
        eval_dir = Path(self.base_env["PDF_TO_JSON_RAG_DATA_DIR"]) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        report_path = eval_dir / "runtime_mode_comparison.json"
        report_path.write_text(
            json.dumps(
                {
                    "all_cases": True,
                    "case_count": 2,
                    "mode_results": [
                        {
                            "mode": "baseline",
                            "pass_count": 2,
                            "fail_count": 0,
                            "summary": {"mrr": 0.9, "avg_recall_at_k": 0.8},
                            "index_manifest": {"embedding_backend": "hash-fallback"},
                        },
                        {
                            "mode": "sentence-transformers",
                            "pass_count": 2,
                            "fail_count": 0,
                            "summary": {"mrr": 1.0, "avg_recall_at_k": 1.0},
                            "index_manifest": {
                                "embedding_backend": "sentence-transformers"
                            },
                        },
                    ],
                    "baseline_deltas": {"sentence-transformers": {"mrr_delta": 0.1}},
                    "promotion_gates": {
                        "sentence-transformers": {
                            "promotable": True,
                            "checks": [],
                            "reasons": [],
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        process = self._run("runtime-promotion-report", "--json")
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["result"]["available"])
        self.assertTrue(payload["result"]["promotion_ready"])
        self.assertEqual(payload["result"]["candidate"]["pass_count"], 2)
        snapshot_path = Path(payload["result"]["promotion_snapshot_path"])
        self.assertTrue(snapshot_path.exists())
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["candidate_mode"], "sentence-transformers")
        self.assertFalse(snapshot["recommended_default_change"])
        self.assertEqual(
            payload["result"]["default_decision"]["default_backend"], "auto"
        )
        self.assertEqual(
            payload["result"]["default_decision"]["preferred_backend_when_cached"],
            "sentence-transformers",
        )
        self.assertEqual(
            payload["result"]["default_decision"]["fallback_backend"], "hash"
        )
        self.assertEqual(
            payload["result"]["default_decision"]["cross_encoder"],
            "experimental_opt_in_only",
        )
        self.assertFalse(
            payload["result"]["model_decision_gate"]["default_change_allowed"]
        )
        decisions = {
            item["backend"]: item
            for item in payload["result"]["model_decision_gate"]["decisions"]
        }
        self.assertEqual(
            decisions["sentence-transformers"]["status"], "recommended_opt_in"
        )
        self.assertTrue(decisions["sentence-transformers"]["model_helped"])

    def test_create_demo_pdf_json(self) -> None:
        self._run("init", "--json")
        demo_path = self.workspace / "generated-demo.pdf"
        process = self._run("create-demo-pdf", "--path", str(demo_path), "--json")
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertEqual(result["pdf"], str(demo_path.resolve()))
        self.assertTrue(demo_path.exists())
        self.assertGreaterEqual(len(result["suggested_queries"]), 1)

    def test_smoke_check_end_to_end_json(self) -> None:
        self._run("init", "--json")
        output_path = self.workspace / "smoke.json"
        process = self._run(
            "smoke-check",
            "--pdf",
            str(self.pdf_path),
            "--query",
            "What does this file cover?",
            "--json",
            "--output",
            str(output_path),
        )
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertTrue(result["all_pass"])
        self.assertTrue(result["document"]["inventory_summary"])
        self.assertIn("processing_diagnostics", result)
        self.assertTrue(result["processing_diagnostics"]["technical_processed"])
        self.assertEqual(result["index"]["embedding"]["requested_backend"], "auto")
        self.assertEqual(
            result["index"]["embedding"]["effective_backend"], "hash-fallback"
        )
        self.assertTrue(result["answer"]["answer"])
        self.assertIn("quality_profile_summary", result)
        self.assertNotIn("quality_profile", result)
        self._assert_public_workflow_contract(result, smoke=True)
        written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written["command"], "smoke-check")
        self.assertTrue(written["result"]["all_pass"])

    def test_run_workflow_public_json_contract_is_compact(self) -> None:
        self._run("init", "--json")
        process = self._run(
            "run-workflow",
            "--pdf",
            str(self.pdf_path),
            "--query",
            "What does this file cover?",
            "--json",
        )
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self._assert_public_workflow_contract(result)
        self.assertTrue(result["answer"]["answer"])
        self.assertIn(
            result["quality_profile_summary"]["overall_status"],
            {"pass", "warn", "review", "fail", "skip", "unknown"},
        )

        verbose_payload = json.loads(
            self._run(
                "run-workflow",
                "--pdf",
                str(self.pdf_path),
                "--query",
                "What does this file cover?",
                "--json",
                "--verbose",
            ).stdout
        )
        verbose_result = verbose_payload["result"]
        self.assertIn("artifacts", verbose_result)
        self.assertIn("quality_profile", verbose_result)
        self.assertIn("top_k_hits", verbose_result["answer"])

    def test_assess_pdf_end_to_end_json_is_compact(self) -> None:
        self._run("init", "--json")
        demo_path = self.workspace / "assess-demo.pdf"
        self._run("create-demo-pdf", "--path", str(demo_path), "--json")
        process = self._run(
            "assess-pdf",
            "--pdf",
            str(demo_path),
            "--json",
        )
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["processing_status"], "pass")
        self.assertEqual(result["semantic_status"], "pass")
        self.assertEqual(result["retrieval_status"], "pass")
        self.assertEqual(result["answer_trust"], "pass")
        self.assertEqual(result["recommended_next_action"], "none")
        self.assertEqual(result["acceptance_profile"], "short_document")
        self.assertIn("answer_supported_by_document_semantics_only", result["messages"])
        self.assertNotIn("workflow", result)
        self._assert_assess_pdf_contract(result)

        verbose_payload = json.loads(
            self._run(
                "assess-pdf",
                "--pdf",
                str(demo_path),
                "--json",
                "--verbose",
            ).stdout
        )
        self.assertIn("workflow", verbose_payload["result"])
        self.assertIn("quality_profile", verbose_payload["result"]["workflow"])

    def test_create_demo_then_smoke_then_answer_query_chain(self) -> None:
        self._run("init", "--json")
        generated_pdf = self.workspace / "generated-demo.pdf"
        self._run("create-demo-pdf", "--path", str(generated_pdf), "--format", "json")
        extract = self._run(
            "extract-native",
            "--pdf",
            str(generated_pdf),
            "--format",
            "json",
        )
        extract_payload = json.loads(extract.stdout)
        doc_id = extract_payload["result"]["doc_id"]
        inspect_payload = json.loads(
            self._run(
                "inspect-document",
                "--doc-id",
                doc_id,
                "--format",
                "json",
            ).stdout
        )
        self.assertTrue(inspect_payload["ok"])
        self.assertEqual(inspect_payload["result"]["doc_id"], doc_id)
        self.assertEqual(inspect_payload["result"]["document_type"], "guidance_note")
        self.assertGreaterEqual(inspect_payload["result"]["section_count"], 1)
        self.assertIsNotNone(inspect_payload["result"]["structure_confidence"])
        self.assertIsNotNone(inspect_payload["result"]["layout_confidence"])
        self.assertIsNotNone(inspect_payload["result"]["semantic_confidence"])
        self.assertTrue(inspect_payload["result"]["semantic_confidence_label"])
        self.assertIn(
            "block_role_counts", inspect_payload["result"]["extraction_summary"]
        )
        self.assertIn(
            "text_source_counts", inspect_payload["result"]["extraction_summary"]
        )
        self.assertIn(
            "layout_signal_counts", inspect_payload["result"]["extraction_summary"]
        )
        self.assertIn("table_probe", inspect_payload["result"]["extraction_summary"])
        self.assertIn("pdf_inspector", inspect_payload["result"]["extraction_summary"])
        self.assertEqual(
            inspect_payload["result"]["extraction_summary"]["pdf_inspector"]["version"],
            "0.2.6",
        )
        self.assertIn("processing_diagnostics", inspect_payload["result"])
        self.assertFalse(
            inspect_payload["result"]["processing_diagnostics"]["technical_processed"]
        )
        self.assertIn(
            "low_text_coverage",
            inspect_payload["result"]["processing_diagnostics"]["taxonomy"],
        )
        self.assertIn("section_role", inspect_payload["result"]["sections"][0])
        self.assertIn("layout_signals", inspect_payload["result"]["sections"][0])
        self.assertIn("text_source_profile", inspect_payload["result"]["sections"][0])
        smoke = self._run(
            "smoke-check",
            "--pdf",
            str(generated_pdf),
            "--query",
            "What does this file cover?",
            "--format",
            "json",
        )
        smoke_payload = json.loads(smoke.stdout)
        self.assertTrue(smoke_payload["ok"])
        self.assertTrue(smoke_payload["result"]["all_pass"])
        self.assertIn("embedding", smoke_payload["result"]["index"])
        self.assertEqual(
            smoke_payload["result"]["index"]["embedding"]["requested_backend"], "auto"
        )
        self.assertIsNotNone(
            smoke_payload["result"]["document"]["structure_confidence"]
        )
        self.assertIsNotNone(smoke_payload["result"]["document"]["layout_confidence"])
        self.assertIsNotNone(smoke_payload["result"]["document"]["semantic_confidence"])
        self.assertEqual(
            smoke_payload["result"]["quality_profile_summary"]["statuses"][
                "answer_trust"
            ],
            "pass",
        )

        index_dir = self.data_dir / "index" / "workflow_smoke"
        answer = self._run(
            "answer-query",
            "--query",
            "What does this file cover?",
            "--index-dir",
            str(index_dir),
            "--format",
            "json",
            "--verbose",
        )
        answer_payload = json.loads(answer.stdout)
        self.assertTrue(answer_payload["ok"])
        self.assertTrue(answer_payload["result"]["answer"])
        self.assertNotIn(
            "No grounded answer could be assembled",
            answer_payload["result"]["answer"],
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["answer_mode"],
            "document_overview",
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["document_selection"]["strategy"],
            "single_doc_overview",
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["document_synthesis"][
                "support_scope"
            ],
            "selected_docs",
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["document_synthesis"][
                "selected_chunk_count"
            ],
            len(answer_payload["result"]["expanded_hits"]),
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["retrieval_contract"][
                "retrieval_path"
            ],
            "document_understanding",
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["retrieval_contract"]["doc_scope"],
            "all_docs",
        )
        synthesis_prompt_contract = answer_payload["result"]["answer_trace"][
            "synthesis_prompt_contract"
        ]
        self.assertEqual(
            synthesis_prompt_contract["template_id"], "grounded_context_only.v1"
        )
        self.assertEqual(synthesis_prompt_contract["runtime"], "not_invoked")
        self.assertFalse(synthesis_prompt_contract["outside_knowledge_allowed"])
        self.assertTrue(synthesis_prompt_contract["requires_chunk_citations"])
        self.assertEqual(
            synthesis_prompt_contract["context_chunk_count"],
            len(answer_payload["result"]["expanded_hits"]),
        )
        self.assertGreater(synthesis_prompt_contract["prompt_char_count"], 0)
        synthesis_runtime = answer_payload["result"]["answer_trace"][
            "synthesis_runtime"
        ]
        self.assertFalse(synthesis_runtime["configured"])
        self.assertFalse(synthesis_runtime["invoked"])
        self.assertFalse(synthesis_runtime["used_for_final_answer"])
        self.assertIn(
            "shortlist_breakdown",
            answer_payload["result"]["answer_trace"]["document_selection"],
        )
        support_trace = answer_payload["result"]["answer_trace"]["support_trace"]
        alignment = answer_payload["result"]["answer_trace"]["claim_alignment"]
        self.assertEqual(alignment["unsupported_claim_count"], 0)
        self.assertGreaterEqual(len(support_trace), 1)
        self.assertTrue(support_trace[0]["section_summaries"])
        self.assertTrue(support_trace[0]["section_paths"])
        self.assertIsNotNone(support_trace[0]["structure_confidence"])
        self.assertIsNotNone(support_trace[0]["layout_confidence"])
        self.assertIsNotNone(support_trace[0]["semantic_confidence"])
        self.assertIsNotNone(support_trace[0]["classification_confidence"])
        self.assertTrue(support_trace[0]["trust_policy"])
        self.assertTrue(support_trace[0]["semantic_rationale"])
        self.assertTrue(answer_payload["result"]["top_k_hits"][0]["section_path"])
        self.assertIsNotNone(
            answer_payload["result"]["top_k_hits"][0]["structure_confidence"]
        )
        self.assertIsNotNone(
            answer_payload["result"]["top_k_hits"][0]["layout_confidence"]
        )
        self.assertTrue(answer_payload["result"]["top_k_hits"][0]["chunk_strategy"])
        self.assertIn("layout_signals", answer_payload["result"]["top_k_hits"][0])

    def test_error_json_for_missing_index(self) -> None:
        self._run("init", "--json")
        process = self._run(
            "answer-query",
            "--query",
            "What does this file cover?",
            "--index-dir",
            str(self.workspace / "missing-index"),
            "--json",
            expect_ok=False,
        )
        self.assertNotEqual(process.returncode, 0)
        payload = json.loads(process.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "missing_index")

    def test_packaged_example_assets_fallback_when_local_examples_are_incomplete(
        self,
    ) -> None:
        fake_root = self.workspace / "fake-install-root"
        fake_examples = fake_root / "examples"
        fake_examples.mkdir(parents=True, exist_ok=True)
        (fake_examples / "public_demo_profile.json").write_text("{}", encoding="utf-8")

        with mock.patch.object(cli_module, "_project_examples_dir", return_value=None):
            examples_dir = cli_module._available_examples_dir()
            self.assertEqual(examples_dir, cli_module._packaged_examples_dir())
            payload = cli_module._load_example_json("public_demo_queries.json")
            self.assertIsInstance(payload, list)
            self.assertGreaterEqual(len(payload), 1)


if __name__ == "__main__":
    unittest.main()
