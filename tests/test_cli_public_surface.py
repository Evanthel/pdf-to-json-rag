"""Public-surface smoke tests for the packaged CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pdf_to_json_rag import cli as cli_module
from pdf_to_json_rag import retrieval as retrieval_module
from pdf_to_json_rag.answering import answer_from_chunks
from pdf_to_json_rag.chunking import chunk_document, normalize_reading_order
from pdf_to_json_rag.content_metadata import classify_block_metadata, infer_layout_signals
from pdf_to_json_rag.document_facets import derive_document_facets
from pdf_to_json_rag.document_semantics import interpret_document_semantics
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
from pdf_to_json_rag.extraction import (
    ExtractedBlock,
    _build_extracted_block,
    _sort_page_blocks_reading_order,
    extract_pdfplumber_table_blocks,
    probe_pdfplumber_tables,
)
from pdf_to_json_rag.llm_output import parse_strict_json_output
from pdf_to_json_rag.llm_runtime import prompt_command_payload, provider_for_env_command
from pdf_to_json_rag.indexing import build_local_index
from pdf_to_json_rag.query_planning import plan_query
from pdf_to_json_rag.retrieval import build_retrieval_contract
from pdf_to_json_rag.schemas import ChunkRecord, DocumentRecord


class CliPublicSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.data_dir = self.workspace / "data"
        self.pdf_path = self.workspace / "demo.pdf"
        self._create_demo_pdf(self.pdf_path)
        self.base_env = os.environ.copy()
        self.base_env["PYTHONPATH"] = str(REPO_ROOT / "src")
        self.base_env["PDF_TO_JSON_RAG_DATA_DIR"] = str(self.data_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_demo_pdf(self, path: Path) -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Demo Safety Guide\n\n"
            "This guide covers safety checks, incident response, and reporting steps.\n"
            "It is intended for operations staff and gives procedural guidance.\n"
            "Section 1: Preparation\n"
            "Section 2: Response\n"
            "Section 3: Follow-up\n",
        )
        doc.save(path)
        doc.close()

    def _assert_public_workflow_contract(self, result: dict[str, object], *, smoke: bool = False) -> None:
        expected_keys = (
            cli_module.PUBLIC_COMPACT_SMOKE_KEYS
            if smoke
            else cli_module.PUBLIC_COMPACT_WORKFLOW_KEYS
        )
        self.assertEqual(set(result), set(expected_keys))
        self.assertEqual(set(result["document"]), set(cli_module.PUBLIC_COMPACT_DOCUMENT_KEYS))
        self.assertEqual(set(result["index"]), set(cli_module.PUBLIC_COMPACT_INDEX_KEYS))
        self.assertEqual(set(result["answer"]), set(cli_module.PUBLIC_COMPACT_ANSWER_KEYS))
        self.assertEqual(
            set(result["quality_profile_summary"]),
            {"available", "overall_status", "statuses", "reasons", "recommended_next_action"},
        )
        self.assertEqual(
            set(result["processing_diagnostics"]),
            {"status", "taxonomy", "technical_processed", "structurally_reliable", "recommended_next_action", "summary"},
        )
        self.assertNotIn("artifacts", result)
        self.assertNotIn("quality_profile", result)
        self.assertNotIn("top_k_hits", result["answer"])
        self.assertNotIn("expanded_hits", result["answer"])
        self.assertNotIn("evidence", result["answer"])

    def _assert_assess_pdf_contract(self, result: dict[str, object]) -> None:
        self.assertEqual(set(result), set(cli_module.PUBLIC_ASSESS_PDF_KEYS))
        self.assertNotIn("workflow", result)
        self.assertIsInstance(result["messages"], list)
        self.assertIn(
            result["acceptance_profile"],
            {
                "scanned_pdf",
                "form_heavy_pdf",
                "table_heavy_pdf",
                "short_document",
                "medium_document",
                "long_document",
            },
        )

    def _run(self, *args: str, expect_ok: bool = True) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            [sys.executable, "-m", "pdf_to_json_rag", *args],
            cwd=REPO_ROOT,
            env=self.base_env,
            capture_output=True,
            text=True,
        )
        if expect_ok and process.returncode != 0:
            self.fail(f"Command failed: {' '.join(args)}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")
        return process

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
        self.assertIn("pdfplumber_available", check_names)
        self.assertIn("embedding_backend_configured", check_names)
        self.assertEqual(result["runtime"]["embedding"]["requested_backend"], "hash")
        self.assertEqual(result["runtime"]["embedding"]["effective_backend"], "hash-fallback")

    def test_runtime_check_reports_default_hash_backend(self) -> None:
        process = self._run("runtime-check", "--json")
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        embedding = payload["result"]["embedding"]
        decision = payload["result"]["runtime_decision"]
        self.assertEqual(payload["result"]["install_context"]["version"], "0.1.0")
        self.assertTrue(payload["result"]["install_context"]["module_path"].endswith("cli.py"))
        self.assertEqual(embedding["requested_backend"], "hash")
        self.assertEqual(embedding["effective_backend"], "hash-fallback")
        self.assertEqual(decision["default_backend"], "hash")
        self.assertIn("sentence-transformer", decision["not_default_reason"])
        self.assertEqual(decision["backend_policy"]["default_backend"]["name"], "hash")
        self.assertEqual(
            decision["backend_policy"]["experimental_backends"][0]["status"],
            "experimental_opt_in",
        )
        self.assertFalse(decision["backend_policy"]["llm_synthesis"]["default_enabled"])
        self.assertEqual(payload["result"]["default_policy"]["llm_synthesis"], "opt_in")

    def test_runtime_check_reports_sentence_transformer_env_request(self) -> None:
        env = dict(self.base_env)
        env["PDF_TO_JSON_RAG_EMBEDDING_BACKEND"] = "sentence-transformers"
        env["PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL"] = "definitely-not-cached-local-model"
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
                            "index_manifest": {"embedding_backend": "sentence-transformers"},
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
        self.assertEqual(payload["result"]["default_decision"]["default_backend"], "hash")
        self.assertEqual(
            payload["result"]["default_decision"]["recommended_opt_in_backend"],
            "sentence-transformers",
        )
        self.assertEqual(payload["result"]["default_decision"]["cross_encoder"], "experimental_opt_in_only")

    def test_corpus_sampling_manifest_is_deterministic(self) -> None:
        entries = [
            cli_module.LocalPdfCorpusEntry(
                digest="B",
                pdf_path=self.workspace / "B.pdf",
                urlkey="",
                original="",
                pages=1,
                file_size=200,
                creator_tool="",
                producer="",
                bucket="short_doc",
            ),
            cli_module.LocalPdfCorpusEntry(
                digest="A",
                pdf_path=self.workspace / "A.pdf",
                urlkey="",
                original="",
                pages=1,
                file_size=100,
                creator_tool="",
                producer="",
                bucket="form_like",
            ),
            cli_module.LocalPdfCorpusEntry(
                digest="C",
                pdf_path=self.workspace / "C.pdf",
                urlkey="",
                original="",
                pages=3,
                file_size=300,
                creator_tool="",
                producer="",
                bucket="medium_doc",
            ),
        ]
        sampled = cli_module._sample_local_pdf_corpus(entries, 3)
        manifest = cli_module._corpus_sampling_manifest(
            entries,
            sampled,
            sample_profile="quick",
            requested_sample_size=3,
        )

        self.assertEqual(manifest["sampling_algorithm"], "bucket_round_robin_v1")
        self.assertEqual(manifest["selected_digests"], ["A", "B", "C"])
        self.assertEqual(manifest["selected_bucket_counts"], {"form_like": 1, "medium_doc": 1, "short_doc": 1})
        self.assertEqual(len(manifest["selected_digest_checksum"]), 64)

    def test_release_check_compact_payload_lists_gate_statuses(self) -> None:
        compact = cli_module._release_check_compact_payload(
            {
                "doctor": {
                    "ready_for_public_cli": True,
                    "runtime": {"runtime_decision": {"default_backend": "hash"}},
                },
                "public_surface": {
                    "all_pass": True,
                    "smoke": {"smoke_all_pass": True},
                },
                "maintainer_checks": {
                    "available": True,
                    "all_pass": True,
                    "package_check": {"all_pass": True, "skipped": False},
                    "unittests": {"passed": True, "skipped": False},
                },
                "internal_regressions": {
                    "benchmark_assets_available": True,
                    "selected_shards": ["query_planning_core"],
                    "skipped": False,
                    "all_pass": True,
                    "results": [
                        {
                            "shard": "query_planning_core",
                            "all_pass": True,
                            "pass_count": 7,
                            "fail_count": 0,
                            "failed_case_ids": [],
                        }
                    ],
                },
                "local_corpus_sanity": {
                    "available": True,
                    "result": {
                        "architecture_gates": {"all_pass": True},
                        "sample_manifest": {"selected_digest_checksum": "abc"},
                        "follow_up_actions": [],
                    },
                },
                "overall_pass": True,
                "recommendation": {"release_ready": True},
            }
        )

        self.assertEqual(compact["overall"]["status"], "pass")
        self.assertEqual(compact["runtime_decision"]["default_backend"], "hash")
        self.assertTrue(compact["product_gate"]["all_pass"])
        self.assertEqual(compact["product_gate"]["public_path"]["status"], "pass")
        self.assertEqual(compact["product_gate"]["benchmark"]["status"], "pass")
        self.assertEqual(compact["product_gate"]["corpus"]["status"], "pass")
        self.assertEqual(compact["internal_regressions"]["selected_shard_count"], 1)
        self.assertEqual(compact["internal_regressions"]["shards"][0]["status"], "pass")
        self.assertEqual(compact["local_corpus_sanity"]["gate"]["status"], "pass")

    def test_release_check_compact_payload_marks_corpus_review_with_examples(self) -> None:
        compact = cli_module._release_check_compact_payload(
            {
                "doctor": {"ready_for_public_cli": True, "runtime": {"runtime_decision": {}}},
                "public_surface": {"all_pass": True, "smoke": {"smoke_all_pass": True}},
                "maintainer_checks": {
                    "available": True,
                    "all_pass": True,
                    "package_check": {"all_pass": True, "skipped": False},
                    "unittests": {"passed": True, "skipped": False},
                },
                "internal_regressions": {
                    "benchmark_assets_available": True,
                    "selected_shards": ["query_planning_core"],
                    "skipped": False,
                    "all_pass": True,
                    "results": [],
                },
                "local_corpus_sanity": {
                    "available": True,
                    "result": {
                        "architecture_gates": {"all_pass": False},
                        "sample_manifest": {"selected_digest_checksum": "abc"},
                        "follow_up_actions": [
                            {
                                "bucket": "scan_like",
                                "focus": "semantics",
                                "priority": "high",
                                "failure_examples": [
                                    {
                                        "pdf": "/tmp/example.pdf",
                                        "doc_id": "example",
                                        "reasons": ["low_semantic_confidence"],
                                        "document_type": "document",
                                        "document_purpose": "reference_lookup",
                                        "semantic_confidence": 0.47,
                                    }
                                ],
                            }
                        ],
                    },
                },
                "overall_pass": False,
                "recommendation": {"release_ready": False},
            }
        )

        corpus_gate = compact["product_gate"]["corpus"]
        self.assertFalse(compact["product_gate"]["all_pass"])
        self.assertEqual(corpus_gate["status"], "review")
        self.assertEqual(corpus_gate["follow_up_count"], 1)
        self.assertEqual(corpus_gate["failure_examples"][0]["bucket"], "scan_like")
        self.assertEqual(corpus_gate["failure_examples"][0]["doc_id"], "example")

    def test_public_beta_check_compact_payload_aggregates_release_gates(self) -> None:
        payload = cli_module._public_beta_check_compact_payload(
            {
                "doctor": {
                    "ready_for_public_cli": True,
                    "runtime": {
                        "runtime_decision": {
                            "default_backend": "hash",
                            "recommended_opt_in_backend": "sentence-transformers",
                            "not_default_reason": "sentence-transformers is opt-in.",
                        }
                    },
                },
                "public_surface": {
                    "all_pass": True,
                    "smoke": {
                        "smoke_all_pass": True,
                        "quality_profile_summary": {
                            "available": True,
                            "overall_status": "pass",
                            "statuses": {"answer_trust": "pass"},
                            "reasons": [],
                        },
                    },
                },
                "maintainer_checks": {
                    "available": True,
                    "all_pass": True,
                    "package_check": {
                        "all_pass": True,
                        "skipped": False,
                        "readme_flow": {
                            "all_pass": True,
                            "steps": [
                                {"name": "init", "ok": True, "returncode": 0},
                                {"name": "runtime-check", "ok": True, "returncode": 0},
                            ],
                        },
                    },
                    "unittests": {"passed": True, "skipped": False},
                },
                "internal_regressions": {
                    "benchmark_assets_available": True,
                    "selected_shards": ["query_planning_core"],
                    "skipped": False,
                    "all_pass": True,
                    "results": [],
                },
                "local_corpus_sanity": {
                    "available": True,
                    "result": {
                        "architecture_gates": {"all_pass": True},
                        "sample_manifest": {"selected_digest_checksum": "abc"},
                        "follow_up_actions": [],
                    },
                },
                "overall_pass": True,
                "recommendation": {"release_ready": True},
            }
        )

        gate_statuses = {item["name"]: item["status"] for item in payload["gates"]}
        self.assertTrue(payload["all_pass"])
        self.assertEqual(gate_statuses["installed_readme_flow"], "pass")
        self.assertEqual(gate_statuses["runtime_default_policy"], "pass")
        self.assertEqual(payload["runtime_decision"]["default_backend"], "hash")
        self.assertEqual(payload["scope"]["sentence_transformers"], "recommended_opt_in_only")
        self.assertEqual(payload["public_smoke_quality"]["overall_status"], "pass")

    def test_answer_contract_health_and_quality_profile_are_readable(self) -> None:
        trace = {
            "answer_mode": "document_overview",
            "retrieval_contract": {"retrieval_path": "document_understanding"},
            "document_synthesis": {"support_scope": "selected_docs", "support_doc_ids": ["demo"]},
            "answer_contract": {"primary_doc_ids": ["demo"]},
            "claim_alignment": {"status": "supported"},
            "support_trace": [{"doc_id": "demo"}],
        }
        health = cli_module._answer_contract_health(trace)
        profile = cli_module._workflow_quality_profile(
            {
                "document": {
                    "document_type": "guidance_note",
                    "document_purpose": "procedural_guidance",
                    "inventory_summary": "Demo guide",
                    "structure_confidence": 0.8,
                    "layout_confidence": 0.75,
                    "semantic_confidence": 0.9,
                    "semantic_confidence_label": "high",
                    "section_count": 1,
                    "extraction_summary": {
                        "native_blocks": 2,
                        "pages_requiring_ocr": 0,
                        "pages_processed_with_ocr": 0,
                        "ocr_used": False,
                    },
                },
                "index": {"chunk_count": 1},
                "answer": {
                    "answer": "Demo guide covers safety.",
                    "answer_trace": trace,
                    "contract_health": health,
                },
            }
        )

        self.assertTrue(health["all_pass"])
        self.assertEqual(health["retrieval_path"], "document_understanding")
        self.assertEqual(health["retrieval_contract_status"]["status"], "pass")
        self.assertEqual(health["support_coverage"]["claim_count"], 0)
        self.assertTrue(health["answer_source_mix"]["document_semantics"]["present"])
        self.assertEqual(profile["processing_quality"]["status"], "pass")
        self.assertEqual(profile["overall_status"], "pass")
        self.assertEqual(profile["recommended_next_action"], "none")
        self.assertEqual(profile["semantic_confidence"]["classification_status"], "well_supported")
        self.assertEqual(profile["retrieval_readiness"]["support_doc_ids"], ["demo"])
        self.assertEqual(profile["retrieval_readiness"]["retrieval_contract_status"]["status"], "pass")
        self.assertTrue(profile["retrieval_readiness"]["answer_source_mix"]["support_trace"]["present"])
        self.assertEqual(profile["answer_trust"]["status"], "pass")
        self.assertEqual(profile["processing_quality"]["drilldown"]["text_extraction_coverage"], None)
        summary = cli_module._quality_profile_summary(profile)
        self.assertEqual(summary["overall_status"], "pass")
        self.assertEqual(summary["recommended_next_action"], "none")

    def test_retrieval_contract_status_warns_on_support_mismatch(self) -> None:
        trace = {
            "answer_mode": "document_overview",
            "candidate_doc_ids": ["alpha"],
            "retrieval_contract": {"retrieval_path": "document_understanding"},
            "document_selection": {"selected_doc_ids": ["alpha"], "candidate_doc_ids": ["alpha"]},
            "document_synthesis": {
                "support_scope": "selected_docs",
                "support_doc_ids": ["beta"],
                "answer_chunk_doc_ids": ["gamma"],
            },
            "answer_contract": {"primary_doc_ids": ["alpha"]},
            "claim_alignment": {
                "claim_count": 1,
                "supported_claim_count": 0,
                "weak_claim_count": 1,
                "unsupported_claim_count": 0,
                "supported_claim_ratio": 0.0,
                "alignment_status": "pass",
                "claims": [
                    {
                        "claim": "Alpha covers safety.",
                        "status": "weak",
                        "score": 0.5,
                        "chunk_id": None,
                        "support_preview": "document type: guidance note",
                    }
                ],
            },
            "support_trace": [{"doc_id": "beta", "support_fragments": ["document type: guidance note"]}],
        }

        status = cli_module._retrieval_contract_status(trace)
        coverage = cli_module._support_coverage(trace)
        mix = cli_module._answer_source_mix(trace)

        self.assertEqual(status["status"], "warn")
        self.assertIn("support_docs_match_selection", status["reasons"])
        self.assertIn("answer_chunks_match_support_docs", status["reasons"])
        self.assertEqual(coverage["weak_claim_count"], 1)
        self.assertEqual(coverage["document_semantics_claim_count"], 1)
        self.assertTrue(mix["document_semantics"]["present"])

    def test_answer_trust_reviews_weak_or_unsupported_claims(self) -> None:
        trace = {
            "answer_mode": "document_overview",
            "retrieval_contract": {"retrieval_path": "document_understanding"},
            "document_synthesis": {"support_scope": "selected_docs", "support_doc_ids": ["demo"]},
            "answer_contract": {"primary_doc_ids": ["demo"]},
            "claim_alignment": {
                "claim_count": 2,
                "supported_claim_count": 1,
                "weak_claim_count": 1,
                "unsupported_claim_count": 0,
                "supported_claim_ratio": 0.5,
                "alignment_status": "needs_review",
            },
            "support_trace": [{"doc_id": "demo"}],
        }
        health = cli_module._answer_contract_health(trace)
        profile = cli_module._workflow_quality_profile(
            {
                "document": {
                    "document_type": "guidance_note",
                    "document_purpose": "procedural_guidance",
                    "inventory_summary": "Demo guide",
                    "structure_confidence": 0.8,
                    "layout_confidence": 0.75,
                    "semantic_confidence": 0.9,
                    "semantic_confidence_label": "high",
                    "section_count": 1,
                    "extraction_summary": {"native_blocks": 2},
                },
                "index": {"chunk_count": 1},
                "answer": {
                    "answer": "Demo guide covers safety.",
                    "answer_trace": trace,
                    "contract_health": health,
                },
            }
        )

        self.assertEqual(profile["answer_trust"]["status"], "review")
        self.assertEqual(profile["overall_status"], "review")
        self.assertEqual(profile["recommended_next_action"], "review_claim_alignment")
        self.assertIn("weak_claims_present", profile["answer_trust"]["reasons"])

    def test_quality_profile_recommends_processing_follow_up_on_low_signal_payload(self) -> None:
        profile = cli_module._workflow_quality_profile(
            {
                "document": {
                    "document_type": "",
                    "document_purpose": "",
                    "inventory_summary": "",
                    "structure_confidence": None,
                    "layout_confidence": None,
                    "semantic_confidence": None,
                    "extraction_summary": {},
                },
                "index": {"chunk_count": 0},
                "answer": {
                    "answer": "",
                    "answer_trace": {},
                    "contract_health": {},
                },
            }
        )

        self.assertEqual(profile["overall_status"], "fail")
        self.assertEqual(profile["processing_quality"]["status"], "fail")
        self.assertEqual(profile["recommended_next_action"], "inspect_document_or_try_ocr")
        self.assertIn("chunks_created", profile["processing_quality"]["reasons"])

    def test_processing_diagnostics_classifies_scan_form_and_table_payloads(self) -> None:
        scan = cli_module._processing_diagnostics(
            {
                "page_count": 2,
                "section_count": 1,
                "structure_confidence": 0.72,
                "layout_confidence": 0.7,
                "extraction_summary": {
                    "native_blocks": 0,
                    "ocr_used": True,
                    "pages_requiring_ocr": 2,
                    "pages_processed_with_ocr": 2,
                },
            },
            {"chunk_count": 1},
        )
        form = cli_module._processing_diagnostics(
            {
                "page_count": 1,
                "section_count": 2,
                "structure_confidence": 0.8,
                "layout_confidence": 0.76,
                "extraction_summary": {
                    "native_blocks": 8,
                    "block_role_counts": {"form_field": 3, "key_value": 2},
                },
            },
            {"chunk_count": 2},
        )
        table = cli_module._processing_diagnostics(
            {
                "page_count": 1,
                "section_count": 1,
                "structure_confidence": 0.78,
                "layout_confidence": 0.62,
                "extraction_summary": {
                    "native_blocks": 6,
                    "block_role_counts": {"table_like": 3},
                    "layout_signal_counts": {"table_like": 2},
                },
            },
            {"chunk_count": 2},
        )

        self.assertIn("native_text_low", scan["taxonomy"])
        self.assertIn("ocr_required", scan["taxonomy"])
        self.assertIn("low_text_coverage", scan["taxonomy"])
        self.assertTrue(scan["technical_processed"])
        self.assertFalse(scan["structurally_reliable"])
        self.assertEqual(scan["status"], "review")
        self.assertEqual(form["taxonomy"], ["table_or_form_heavy"])
        self.assertEqual(form["status"], "warn")
        self.assertTrue(form["structurally_reliable"])
        self.assertIn("table_or_form_heavy", table["taxonomy"])
        self.assertEqual(table["status"], "warn")

    def test_assessment_profiles_cover_unknown_pdf_shapes(self) -> None:
        base_doc = {
            "page_count": 1,
            "semantic_confidence_label": "high",
            "extraction_summary": {
                "block_role_counts": {},
                "layout_signal_counts": {},
            },
        }
        self.assertEqual(cli_module._assessment_profile(base_doc, {"taxonomy": []}), "short_document")
        self.assertEqual(
            cli_module._assessment_profile(
                {**base_doc, "page_count": 20},
                {"taxonomy": []},
            ),
            "long_document",
        )
        self.assertEqual(
            cli_module._assessment_profile(
                base_doc,
                {"taxonomy": ["ocr_required", "native_text_low"]},
            ),
            "scanned_pdf",
        )
        self.assertEqual(
            cli_module._assessment_profile(
                {
                    **base_doc,
                    "extraction_summary": {
                        "block_role_counts": {"form_field": 2, "key_value": 2},
                        "layout_signal_counts": {"form_like": 1},
                    },
                },
                {"taxonomy": ["table_or_form_heavy"]},
            ),
            "form_heavy_pdf",
        )
        self.assertEqual(
            cli_module._assessment_profile(
                {
                    **base_doc,
                    "extraction_summary": {
                        "block_role_counts": {"table_like": 3},
                        "layout_signal_counts": {"table_like": 1},
                    },
                },
                {"taxonomy": ["table_or_form_heavy"]},
            ),
            "table_heavy_pdf",
        )

    def test_corpus_profile_compare_reports_snapshot_deltas(self) -> None:
        baseline_path = self.workspace / "quick.json"
        candidate_path = self.workspace / "balanced.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "sample_profile": "quick",
                    "sample_size": 4,
                    "sample_manifest": {"selected_digest_checksum": "aaa"},
                    "summary": {
                        "technical_pass_rate": 1.0,
                        "semantic_pass_rate": 0.75,
                        "avg_structure_confidence": 0.7,
                        "avg_layout_confidence": 0.7,
                        "avg_semantic_confidence": 0.8,
                        "specific_document_rate": 0.75,
                        "specific_purpose_rate": 0.75,
                        "low_confidence_rate": 0.25,
                        "trust_limited_rate": 0.0,
                    },
                    "architecture_gates": {"all_pass": True},
                    "follow_up_actions": [],
                }
            ),
            encoding="utf-8",
        )
        candidate_path.write_text(
            json.dumps(
                {
                    "sample_profile": "balanced",
                    "sample_size": 12,
                    "sample_manifest": {"selected_digest_checksum": "bbb"},
                    "summary": {
                        "technical_pass_rate": 1.0,
                        "semantic_pass_rate": 1.0,
                        "avg_structure_confidence": 0.72,
                        "avg_layout_confidence": 0.71,
                        "avg_semantic_confidence": 0.85,
                        "specific_document_rate": 1.0,
                        "specific_purpose_rate": 1.0,
                        "low_confidence_rate": 0.0,
                        "trust_limited_rate": 0.0,
                    },
                    "architecture_gates": {"all_pass": True},
                    "follow_up_actions": [],
                }
            ),
            encoding="utf-8",
        )

        payload = cli_module._corpus_profile_compare_payload(
            baseline_path=baseline_path,
            candidate_path=candidate_path,
        )

        self.assertTrue(payload["available"])
        self.assertTrue(payload["all_pass"])
        self.assertTrue(payload["sample_changed"])
        self.assertEqual(payload["deltas"]["semantic_pass_rate"], 0.25)
        self.assertEqual(payload["regressions"], [])
        self.assertTrue(payload["corpus_diff_summary"]["all_pass"])
        checksum_check = {
            item["name"]: item["status"]
            for item in payload["corpus_diff_summary"]["checks"]
        }
        self.assertEqual(checksum_check["sample_checksum"], "skip")

    def test_corpus_snapshot_aliases_and_compact_write_do_not_require_reprocessing(self) -> None:
        payload = {
            "sample_profile": "quick",
            "sample_size": 2,
            "sample_manifest": {"selected_digest_checksum": "abc"},
            "summary": {
                "technical_pass_rate": 1.0,
                "semantic_pass_rate": 1.0,
                "avg_structure_confidence": 0.8,
                "avg_layout_confidence": 0.8,
                "avg_semantic_confidence": 0.9,
                "specific_document_rate": 1.0,
                "specific_purpose_rate": 1.0,
                "low_confidence_rate": 0.0,
                "trust_limited_rate": 0.0,
            },
            "bucket_diagnostics": {"short_doc": {"sample_count": 2}},
            "architecture_gates": {"all_pass": True},
            "corpus_contract": {"all_pass": True},
            "follow_up_actions": [],
            "results": [{"large": "ignored in compact snapshot"}],
        }
        original_data_eval = cli_module.PATHS.data_eval
        object.__setattr__(cli_module.PATHS, "data_eval", self.workspace)
        try:
            snapshot_path = cli_module._write_corpus_sanity_snapshot(payload)
            latest_path = cli_module._corpus_snapshot_path_for_profile("latest")
            quick_latest_path = cli_module._corpus_snapshot_path_for_profile("quick-latest")
            compact_path = self.workspace / "corpus_sanity_quick_compact_snapshot.json"
        finally:
            object.__setattr__(cli_module.PATHS, "data_eval", original_data_eval)

        self.assertEqual(snapshot_path, self.workspace / "corpus_sanity_snapshot.json")
        self.assertEqual(latest_path, self.workspace / "corpus_sanity_snapshot.json")
        self.assertEqual(quick_latest_path, self.workspace / "corpus_sanity_quick_snapshot.json")
        compact = json.loads(compact_path.read_text(encoding="utf-8"))
        self.assertNotIn("results", compact)
        self.assertEqual(compact["sample_manifest"]["selected_digest_checksum"], "abc")

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

    def test_pdfplumber_table_probe_is_optional(self) -> None:
        with mock.patch("importlib.util.find_spec", return_value=None):
            probe = probe_pdfplumber_tables(self.pdf_path)
        self.assertFalse(probe["available"])
        self.assertEqual(probe["engine"], "pdfplumber")
        self.assertEqual(probe["reason"], "not_installed")

    def test_pdfplumber_tables_can_be_supplemental_blocks(self) -> None:
        class FakeTable:
            bbox = (10.0, 20.0, 190.0, 120.0)

            def extract(self) -> list[list[str]]:
                return [["Name", "Score"], ["Alpha", "10"]]

        class FakePage:
            width = 200.0
            height = 400.0

            def find_tables(self) -> list[FakeTable]:
                return [FakeTable()]

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self) -> "FakePdf":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        fake_pdfplumber = types.SimpleNamespace(open=lambda _path: FakePdf())
        with (
            mock.patch("importlib.util.find_spec", return_value=object()),
            mock.patch.dict(sys.modules, {"pdfplumber": fake_pdfplumber}),
        ):
            blocks, probe = extract_pdfplumber_table_blocks(self.pdf_path, doc_id="demo")

        self.assertTrue(probe["available"])
        self.assertEqual(probe["table_count"], 1)
        self.assertEqual(probe["supplemental_block_count"], 1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_kind, "table_like")
        self.assertEqual(blocks[0].block_role, "table_like")
        self.assertIn("pdfplumber_table", blocks[0].structural_flags)
        self.assertIn("Name | Score", blocks[0].text)
        self.assertEqual(blocks[0].bbox, [0.05, 0.05, 0.95, 0.3])

    def test_corpus_sample_profile_resolution(self) -> None:
        self.assertEqual(cli_module._resolve_corpus_sample_size("quick", None), 4)
        self.assertEqual(cli_module._resolve_corpus_sample_size("balanced", None), 12)
        self.assertEqual(cli_module._resolve_corpus_sample_size("stress", None), 24)
        self.assertEqual(cli_module._resolve_corpus_sample_size("stress", 3), 3)

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
        self.assertEqual(result["index"]["embedding"]["requested_backend"], "hash")
        self.assertEqual(result["index"]["embedding"]["effective_backend"], "hash-fallback")
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
        self.assertIn("block_role_counts", inspect_payload["result"]["extraction_summary"])
        self.assertIn("text_source_counts", inspect_payload["result"]["extraction_summary"])
        self.assertIn("layout_signal_counts", inspect_payload["result"]["extraction_summary"])
        self.assertIn("table_probe", inspect_payload["result"]["extraction_summary"])
        self.assertIn("processing_diagnostics", inspect_payload["result"])
        self.assertFalse(inspect_payload["result"]["processing_diagnostics"]["technical_processed"])
        self.assertIn("low_text_coverage", inspect_payload["result"]["processing_diagnostics"]["taxonomy"])
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
        self.assertEqual(smoke_payload["result"]["index"]["embedding"]["requested_backend"], "hash")
        self.assertIsNotNone(smoke_payload["result"]["document"]["structure_confidence"])
        self.assertIsNotNone(smoke_payload["result"]["document"]["layout_confidence"])
        self.assertIsNotNone(smoke_payload["result"]["document"]["semantic_confidence"])
        self.assertEqual(
            smoke_payload["result"]["quality_profile_summary"]["statuses"]["answer_trust"],
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
            answer_payload["result"]["answer_trace"]["document_synthesis"]["support_scope"],
            "selected_docs",
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["document_synthesis"]["selected_chunk_count"],
            len(answer_payload["result"]["expanded_hits"]),
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["retrieval_contract"]["retrieval_path"],
            "document_understanding",
        )
        self.assertEqual(
            answer_payload["result"]["answer_trace"]["retrieval_contract"]["doc_scope"],
            "all_docs",
        )
        synthesis_prompt_contract = answer_payload["result"]["answer_trace"]["synthesis_prompt_contract"]
        self.assertEqual(synthesis_prompt_contract["template_id"], "grounded_context_only.v1")
        self.assertEqual(synthesis_prompt_contract["runtime"], "not_invoked")
        self.assertFalse(synthesis_prompt_contract["outside_knowledge_allowed"])
        self.assertTrue(synthesis_prompt_contract["requires_chunk_citations"])
        self.assertEqual(
            synthesis_prompt_contract["context_chunk_count"],
            len(answer_payload["result"]["expanded_hits"]),
        )
        self.assertGreater(synthesis_prompt_contract["prompt_char_count"], 0)
        synthesis_runtime = answer_payload["result"]["answer_trace"]["synthesis_runtime"]
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
        self.assertIsNotNone(answer_payload["result"]["top_k_hits"][0]["structure_confidence"])
        self.assertIsNotNone(answer_payload["result"]["top_k_hits"][0]["layout_confidence"])
        self.assertTrue(answer_payload["result"]["top_k_hits"][0]["chunk_strategy"])
        self.assertIn("layout_signals", answer_payload["result"]["top_k_hits"][0])

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
        self.assertEqual(result.answer_trace["synthesis_prompt_contract"]["runtime"], "local_command")

    def test_plan_query_distinguishes_type_purpose_audience_confidence_rationale_and_limits(self) -> None:
        type_payload = json.loads(
            self._run(
                "plan-query",
                "--query",
                "What kind of document is this?",
                "--json",
            ).stdout
        )
        purpose_payload = json.loads(
            self._run(
                "plan-query",
                "--query",
                "What is the purpose of this document?",
                "--json",
            ).stdout
        )
        audience_payload = json.loads(
            self._run(
                "plan-query",
                "--query",
                "Who is this document for?",
                "--json",
            ).stdout
        )
        confidence_payload = json.loads(
            self._run(
                "plan-query",
                "--query",
                "How confident is this document classification?",
                "--json",
            ).stdout
        )
        rationale_payload = json.loads(
            self._run(
                "plan-query",
                "--query",
                "Why is this document classified this way?",
                "--json",
            ).stdout
        )
        limits_payload = json.loads(
            self._run(
                "plan-query",
                "--query",
                "What are the main limits of this document classification?",
                "--json",
            ).stdout
        )
        self.assertEqual(type_payload["result"]["query_intent"], "document_type")
        self.assertEqual(purpose_payload["result"]["query_intent"], "document_purpose")
        self.assertEqual(audience_payload["result"]["query_intent"], "document_audience")
        self.assertEqual(confidence_payload["result"]["query_intent"], "document_confidence")
        self.assertEqual(rationale_payload["result"]["query_intent"], "document_classification_rationale")
        self.assertEqual(limits_payload["result"]["query_intent"], "document_classification_limits")

    def test_retrieval_contract_splits_single_doc_document_understanding_and_cross_document(self) -> None:
        evidence_contract = build_retrieval_contract(
            "What are common cold symptoms?",
            plan=plan_query("What are common cold symptoms?"),
        )
        overview_contract = build_retrieval_contract(
            "What does this file cover?",
            plan=plan_query("What does this file cover?"),
        )
        listing_contract = build_retrieval_contract(
            "Which sources discuss prevention or procedural guidance?",
            plan=plan_query("Which sources discuss prevention or procedural guidance?"),
        )
        nonmedical_listing_plan = plan_query(
            "Which sources in the benchmark discuss deep learning or data incident response?"
        )
        vitamin_null_plan = plan_query("Does vitamin C prevent the common cold in normal populations?")
        vitamin_stress_plan = plan_query("Does vitamin C help people under cold stress?")
        cmaj_prevention_plan = plan_query(
            "What preventive interventions have the best evidence in the CMAJ common cold review?"
        )
        vitamin_null_contract = build_retrieval_contract(
            vitamin_null_plan.query,
            plan=vitamin_null_plan,
        )
        vitamin_stress_contract = build_retrieval_contract(
            vitamin_stress_plan.query,
            plan=vitamin_stress_plan,
        )
        cmaj_prevention_contract = build_retrieval_contract(
            cmaj_prevention_plan.query,
            plan=cmaj_prevention_plan,
        )

        self.assertEqual(evidence_contract.retrieval_path, "single_document_qa")
        self.assertEqual(overview_contract.retrieval_path, "document_understanding")
        self.assertEqual(listing_contract.retrieval_path, "cross_document_discovery")
        self.assertEqual(listing_contract.doc_scope, "candidate_docs")
        self.assertEqual(listing_contract.diversify_per_doc_limit, 1)
        self.assertEqual(nonmedical_listing_plan.answer_mode, "source_listing")
        self.assertIn("lbdl", nonmedical_listing_plan.candidate_doc_ids)
        self.assertIn(
            "guidance-note-data-incident-management",
            nonmedical_listing_plan.candidate_doc_ids,
        )
        self.assertEqual(vitamin_null_plan.query_intent, "treatment_null_effect")
        self.assertEqual(vitamin_stress_plan.query_intent, "treatment_subgroup_benefit")
        self.assertEqual(cmaj_prevention_plan.query_intent, "review_prevention")
        self.assertEqual(vitamin_null_contract.doc_scope, "preferred_doc")
        self.assertEqual(vitamin_stress_contract.doc_scope, "preferred_doc")
        self.assertEqual(cmaj_prevention_contract.doc_scope, "preferred_doc")
        self.assertEqual(
            vitamin_null_contract.preferred_doc_id,
            "vitamin-c-for-preventing-and-treating-the-common-cold",
        )
        self.assertEqual(
            vitamin_stress_contract.preferred_doc_id,
            "vitamin-c-for-preventing-and-treating-the-common-cold",
        )
        self.assertEqual(
            cmaj_prevention_contract.preferred_doc_id,
            "prevention-and-treatment-of-the-common-cold",
        )

    def test_cross_encoder_rerank_is_optional_and_records_backend_signal(self) -> None:
        class FakeCrossEncoder:
            def predict(self, pairs):
                return [0.9 if "high value" in text else 0.1 for _, text in pairs]

        chunks = [
            ChunkRecord(
                doc_id="doc",
                chunk_id="low",
                source_pdf="demo.pdf",
                text="low value support text",
                page_start=1,
                page_end=1,
                reading_order_index=1,
            ),
            ChunkRecord(
                doc_id="doc",
                chunk_id="high",
                source_pdf="demo.pdf",
                text="high value support text",
                page_start=1,
                page_end=1,
                reading_order_index=2,
            ),
        ]

        with mock.patch.dict(os.environ, {"PDF_TO_JSON_RAG_USE_CROSS_ENCODER": "1"}):
            with mock.patch.object(
                retrieval_module,
                "_load_cross_encoder",
                return_value=(FakeCrossEncoder(), None),
            ):
                reranked, fallback_reason = retrieval_module._cross_encoder_rerank_hits(chunks, "value")

        self.assertIsNone(fallback_reason)
        self.assertIsNotNone(reranked)
        self.assertEqual(reranked[0].chunk_id, "high")
        self.assertEqual(
            reranked[0].retrieval_signals["rerank_backend_code"],
            retrieval_module.RERANK_BACKEND_CROSS_ENCODER,
        )
        self.assertEqual(reranked[0].retrieval_signals["cross_encoder_signal"], 0.9)

    def test_cross_encoder_unavailable_falls_back_to_lightweight_rerank(self) -> None:
        chunks = [
            ChunkRecord(
                doc_id="doc",
                chunk_id="a",
                source_pdf="demo.pdf",
                text="plain support text",
                page_start=1,
                page_end=1,
                reading_order_index=1,
            )
        ]

        with mock.patch.dict(os.environ, {"PDF_TO_JSON_RAG_USE_CROSS_ENCODER": "1"}):
            with mock.patch.object(
                retrieval_module,
                "_load_cross_encoder",
                return_value=(None, "not installed"),
            ):
                reranked = retrieval_module._select_reranked_hits(
                    hits=chunks,
                    query="support",
                    use_lightweight_rerank=True,
                )

        self.assertEqual(len(reranked), 1)
        self.assertEqual(
            reranked[0].retrieval_signals["rerank_backend_code"],
            retrieval_module.RERANK_BACKEND_LIGHTWEIGHT,
        )
        self.assertEqual(reranked[0].retrieval_signals["cross_encoder_fallback"], 1.0)

    def test_expanded_context_is_reranked_after_neighbor_expansion(self) -> None:
        class FakeCrossEncoder:
            def predict(self, pairs):
                return [0.95 if "neighbor answer" in text else 0.2 for _, text in pairs]

        expanded = [
            ChunkRecord(
                doc_id="doc",
                chunk_id="anchor",
                source_pdf="demo.pdf",
                text="anchor context",
                page_start=1,
                page_end=1,
                reading_order_index=1,
            ),
            ChunkRecord(
                doc_id="doc",
                chunk_id="neighbor",
                source_pdf="demo.pdf",
                text="neighbor answer context",
                page_start=1,
                page_end=1,
                reading_order_index=2,
            ),
        ]

        with mock.patch.dict(os.environ, {"PDF_TO_JSON_RAG_USE_CROSS_ENCODER": "1"}):
            with mock.patch.object(
                retrieval_module,
                "_load_cross_encoder",
                return_value=(FakeCrossEncoder(), None),
            ):
                reranked = retrieval_module.rerank_expanded_context(
                    expanded,
                    "answer",
                    use_lightweight_rerank=True,
                )

        self.assertEqual(reranked[0].chunk_id, "neighbor")
        self.assertEqual(reranked[0].retrieval_signals["expanded_context_rank"], 1.0)
        self.assertEqual(
            reranked[0].retrieval_signals["rerank_backend_code"],
            retrieval_module.RERANK_BACKEND_CROSS_ENCODER,
        )

    def test_evaluation_layers_distinguish_processing_retrieval_and_faithfulness(self) -> None:
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
            "print('{\"faithful\": true, \"supported_sentence_ratio\": 1.0, "
            "\"unsupported_sentences\": [], \"rationale\": \"Supported by supplied context.\"}')\n"
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
        duplicate = parse_strict_json_output('```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```')

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

    def test_answer_trace_includes_claim_alignment_status(self) -> None:
        chunks = [
            ChunkRecord(
                doc_id="doc",
                chunk_id="c1",
                source_pdf="demo.pdf",
                text="Common cold symptoms include cough, fever, and sore throat.",
                page_start=1,
                page_end=1,
                reading_order_index=1,
            )
        ]

        answer = answer_from_chunks("What are common cold symptoms?", chunks)
        alignment = answer.answer_trace["claim_alignment"]
        self.assertGreaterEqual(alignment["claim_count"], 1)
        self.assertIn(alignment["alignment_status"], {"pass", "needs_review"})
        self.assertIn("claims", alignment)

    def test_source_anchored_grounded_answer_filters_to_preferred_document(self) -> None:
        chunks = [
            ChunkRecord(
                doc_id="ajmedp-4-2-srd-eda-v1-e-2561",
                chunk_id="ajmedp-4-2-srd-eda-v1-e-2561-chunk-0124",
                source_pdf="ajmedp.pdf",
                text=(
                    "Severe. Mandatory buddy checks every 10 minutes. "
                    "Wear ECWCS or equivalent and wind protection including head, hands, feet, face."
                ),
                page_start=124,
                page_end=124,
                reading_order_index=1,
                section_title="Severe",
            ),
            ChunkRecord(
                doc_id="actionable-gamification-full-book",
                chunk_id="actionable-gamification-full-book-chunk-0010",
                source_pdf="gamification.pdf",
                text="Gamification systems use quests, points, and progress loops to influence behavior.",
                page_start=10,
                page_end=10,
                reading_order_index=2,
                section_title="Motivation",
            ),
            ChunkRecord(
                doc_id="actionable-gamification-full-book",
                chunk_id="actionable-gamification-full-book-chunk-0011",
                source_pdf="gamification.pdf",
                text="Player journeys can be optimized through badges and social feedback.",
                page_start=11,
                page_end=11,
                reading_order_index=3,
                section_title="Motivation",
            ),
            ChunkRecord(
                doc_id="actionable-gamification-full-book",
                chunk_id="actionable-gamification-full-book-chunk-0012",
                source_pdf="gamification.pdf",
                text="A system designer can use scarcity, ownership, and status loops.",
                page_start=12,
                page_end=12,
                reading_order_index=4,
                section_title="Motivation",
            ),
        ]

        answer = answer_from_chunks(
            "According to AJMedP Table 3-4, what is recommended for the severe frostbite risk zone?",
            chunks,
        )

        self.assertIn("Mandatory buddy checks every 10 minutes", answer.answer)
        self.assertIn("ECWCS", answer.answer)
        self.assertEqual(
            answer.answer_trace["document_synthesis"]["support_scope"],
            "source_anchor_preferred_doc",
        )

    def test_faithfulness_contract_validation_gate_passes_on_valid_records(self) -> None:
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

    def test_optional_low_confidence_semantics_multipass_is_env_gated(self) -> None:
        base = interpret_document_semantics(
            source_pdf="mystery.pdf",
            title="Mystery",
            toc=[],
            summary_cues=[],
            discovery_terms=["financial statement", "total assets", "total liabilities"],
            leading_block_lines=[],
            metadata_values=[],
            page_count=1,
        )
        with mock.patch.dict(os.environ, {"PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS": "1"}):
            reviewed = interpret_document_semantics(
                source_pdf="mystery.pdf",
                title="Mystery",
                toc=[],
                summary_cues=[],
                discovery_terms=["financial statement", "total assets", "total liabilities"],
                leading_block_lines=[],
                metadata_values=[],
                page_count=1,
            )

        self.assertNotIn("optional_low_confidence_multipass_accepted", base.semantic_rationale)
        self.assertIn("optional_low_confidence_multipass_accepted", reviewed.semantic_rationale)
        self.assertGreaterEqual(reviewed.semantic_confidence, base.semantic_confidence)

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
        self.assertEqual(mode_results["baseline"]["runtime_signals"]["llm_used_case_count"], 0)
        self.assertEqual(mode_results["llm-synthesis"]["runtime_signals"]["llm_used_case_count"], 1)
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

    def test_packaged_example_assets_fallback_when_local_examples_are_incomplete(self) -> None:
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

    def test_inline_section_chunking_preserves_document_root_context(self) -> None:
        document = DocumentRecord(
            doc_id="demo-inline",
            source_pdf="demo-inline.pdf",
            page_count=1,
            title="Demo Safety Guide",
            detected_language="en",
        )
        blocks = [
            ExtractedBlock(
                block_id="demo-inline-block-001",
                page_num=0,
                text="Background This guide explains field safety procedures.",
                bbox=None,
                reading_order_index=0,
                block_kind="text",
            ),
            ExtractedBlock(
                block_id="demo-inline-block-002",
                page_num=0,
                text="CHECKLIST Confirm PPE and radio contact before deployment.",
                bbox=None,
                reading_order_index=1,
                block_kind="text",
                structural_flags=["structured_signal"],
            ),
        ]
        chunks = chunk_document(document, blocks, target_chars=80, min_chunk_chars=40)
        checklist_chunk = next(chunk for chunk in chunks if chunk.section_title == "CHECKLIST")
        self.assertEqual(checklist_chunk.section_path, ["Demo Safety Guide", "CHECKLIST"])
        self.assertEqual(checklist_chunk.section_kind, "checklist_section")
        self.assertIn("checklist_like", checklist_chunk.section_content_hints)
        self.assertIsNotNone(checklist_chunk.structure_confidence)
        self.assertIsNotNone(checklist_chunk.layout_confidence)

    def test_structured_form_segments_split_into_row_like_chunks(self) -> None:
        document = DocumentRecord(
            doc_id="demo-form",
            source_pdf="demo-form.pdf",
            page_count=1,
            title="Appendix Example",
            detected_language="en",
            structure_confidence=0.7,
            layout_confidence=0.7,
        )
        blocks = [
            ExtractedBlock(
                block_id="demo-form-block-001",
                page_num=0,
                text=(
                    "Appendix A – Checklist pre-opioid checklist fields: has non-pharmacological therapy been optimized; "
                    "has non-opioid pharmacotherapy been optimized; informed consent obtained; opioid safety explained; "
                    "urine drug screening completed."
                ),
                bbox=None,
                reading_order_index=0,
                block_kind="table_like",
                structural_flags=["structured_signal"],
            ),
        ]
        chunks = chunk_document(document, blocks, target_chars=200, min_chunk_chars=80)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(chunk.chunk_type in {"table", "checklist"} for chunk in chunks))
        self.assertTrue(any("informed consent obtained" in chunk.text for chunk in chunks))

    def test_block_metadata_distinguishes_form_field_and_heading(self) -> None:
        form_field = classify_block_metadata("Date of Birth: 12/12/1980")
        heading = classify_block_metadata("UNITED STATES COURT OF APPEALS")
        self.assertEqual(form_field["block_role"], "form_field")
        self.assertIn("form_field", form_field["block_labels"])
        self.assertEqual(heading["block_role"], "heading")
        self.assertIn("heading", heading["block_labels"])

    def test_relative_font_signal_promotes_heading_role(self) -> None:
        block = _build_extracted_block(
            block_id="font-heading",
            page_num=0,
            text="Executive Summary",
            bbox=[0.1, 0.1, 0.8, 0.14],
            reading_order_index=0,
            extraction_method="native",
            font_size=18.0,
            relative_font_size=1.5,
            font_is_bold=False,
            toc_entries=set(),
        )
        self.assertEqual(block.block_role, "heading")
        self.assertIn("relative_font_heading", block.structural_flags)
        self.assertEqual(block.font_size, 18.0)
        self.assertEqual(block.relative_font_size, 1.5)

    def test_toc_signal_promotes_heading_role(self) -> None:
        block = _build_extracted_block(
            block_id="toc-heading",
            page_num=0,
            text="Risk Assessment",
            bbox=[0.1, 0.2, 0.8, 0.24],
            reading_order_index=1,
            extraction_method="native",
            toc_entries={"risk assessment"},
        )
        self.assertEqual(block.block_role, "heading")
        self.assertIn("toc_heading", block.structural_flags)

    def test_infer_layout_signals_detects_multi_column_and_form_density(self) -> None:
        signals = infer_layout_signals(
            block_roles=["form_field", "form_field", "paragraph", "paragraph"],
            structural_flags=["structured_signal", "multi_line"],
            bboxes=[
                [0.05, 0.1, 0.35, 0.2],
                [0.62, 0.1, 0.9, 0.2],
                [0.06, 0.25, 0.34, 0.4],
                [0.64, 0.25, 0.92, 0.4],
            ],
            page_span=2,
        )
        self.assertIn("form_dense", signals)
        self.assertIn("multi_column_like", signals)
        self.assertIn("multi_page_span", signals)

    def test_multi_column_native_blocks_sort_by_column_before_row(self) -> None:
        blocks = [
            (70.0, 100.0, 250.0, 130.0, "left column first"),
            (340.0, 100.0, 520.0, 130.0, "right column first"),
            (72.0, 150.0, 248.0, 180.0, "left column second"),
            (342.0, 150.0, 522.0, 180.0, "right column second"),
        ]
        ordered = _sort_page_blocks_reading_order(blocks, page_width=600.0)
        self.assertEqual(
            [item[4] for item in ordered],
            [
                "left column first",
                "left column second",
                "right column first",
                "right column second",
            ],
        )

    def test_normalize_reading_order_uses_bbox_for_multi_column_blocks(self) -> None:
        blocks = [
            ExtractedBlock(
                block_id="left-1",
                page_num=0,
                text="left column first",
                bbox=[0.10, 0.10, 0.40, 0.14],
                reading_order_index=0,
            ),
            ExtractedBlock(
                block_id="right-1",
                page_num=0,
                text="right column first",
                bbox=[0.58, 0.10, 0.88, 0.14],
                reading_order_index=1,
            ),
            ExtractedBlock(
                block_id="left-2",
                page_num=0,
                text="left column second",
                bbox=[0.11, 0.18, 0.39, 0.22],
                reading_order_index=2,
            ),
            ExtractedBlock(
                block_id="right-2",
                page_num=0,
                text="right column second",
                bbox=[0.59, 0.18, 0.89, 0.22],
                reading_order_index=3,
            ),
        ]
        ordered = normalize_reading_order(blocks)
        self.assertEqual([block.block_id for block in ordered], ["left-1", "left-2", "right-1", "right-2"])

    def test_chunk_records_keep_block_provenance_and_section_role(self) -> None:
        document = DocumentRecord(
            doc_id="demo-provenance",
            source_pdf="demo-provenance.pdf",
            page_count=1,
            title="Registration Packet",
            detected_language="en",
            structure_confidence=0.76,
            layout_confidence=0.74,
        )
        blocks = [
            ExtractedBlock(
                block_id="demo-provenance-block-001",
                page_num=0,
                text="VOTER REGISTRATION TRANSFER FORM",
                bbox=None,
                reading_order_index=0,
                block_kind="heading",
                block_role="heading",
            ),
            ExtractedBlock(
                block_id="demo-provenance-block-002",
                page_num=0,
                text="Date of Birth: 01/01/1990",
                bbox=None,
                reading_order_index=1,
                block_kind="text",
                block_role="form_field",
            ),
        ]
        chunks = chunk_document(document, blocks, target_chars=200, min_chunk_chars=20)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.section_role, "form")
        self.assertEqual(chunk.chunk_strategy, "form_rows")
        self.assertIn("demo-provenance-block-002", chunk.source_block_ids)
        self.assertIn("form_field", chunk.source_block_roles)
        self.assertIn("form_dense", chunk.layout_signals)
        self.assertEqual(chunk.text_source, "native")

    def test_financial_form_semantics_are_not_generic(self) -> None:
        facets = derive_document_facets(
            source_pdf="Financial-Statement.pdf",
            title="Personal Financial Statement",
            toc=[],
            summary_cues=["Net Worth", "Total Assets", "Total Liabilities"],
            leading_block_lines=[
                "PERSONAL FINANCIAL STATEMENT",
                "Net Worth (Total Assets - Total Liabilities)",
                "Cash in Banks and Notes Due to Banks",
            ],
            metadata_values=[],
            page_count=2,
        )
        self.assertEqual(facets["document_type"], "financial_statement")
        self.assertEqual(facets["document_purpose"], "financial_disclosure")
        self.assertEqual(facets["audience"], "applicants")
        self.assertGreaterEqual(facets["semantic_confidence"], 0.75)
        self.assertEqual(facets["semantic_confidence_label"], "high")

    def test_registration_form_semantics_are_not_generic(self) -> None:
        facets = derive_document_facets(
            source_pdf="Voter-Registration-Transfer-Form.pdf",
            title="Voter Registration Transfer Form",
            toc=[],
            summary_cues=["Voter Registration", "Transfer Form", "Change of Address"],
            leading_block_lines=[
                "VOTER REGISTRATION TRANSFER FORM",
                "Use this form to update your voter registration address.",
            ],
            metadata_values=[],
            page_count=1,
        )
        self.assertEqual(facets["document_type"], "registration_form")
        self.assertEqual(facets["document_purpose"], "registration_update")
        self.assertEqual(facets["audience"], "filers")
        self.assertGreaterEqual(facets["semantic_confidence"], 0.75)

    def test_court_opinion_semantics_are_not_generic(self) -> None:
        facets = derive_document_facets(
            source_pdf="07-7236.pdf",
            title="United States Court of Appeals for the Federal Circuit",
            toc=[],
            summary_cues=["Court of Appeals", "Claimant-Appellant", "Opinion and Order"],
            leading_block_lines=[
                "United States Court of Appeals for the Federal Circuit",
                "FORTUNATA CAPELLAN, Claimant-Appellant,",
                "Opinion and Order",
            ],
            metadata_values=[],
            page_count=20,
        )
        self.assertEqual(facets["document_type"], "court_opinion")
        self.assertEqual(facets["document_purpose"], "legal_record")
        self.assertEqual(facets["audience"], "legal_professionals")
        self.assertGreaterEqual(facets["semantic_confidence"], 0.75)

    def test_unknown_document_semantics_cover_public_record_buckets(self) -> None:
        statistical = derive_document_facets(
            source_pdf="30 median farm size 2004.pdf",
            title="30% Median Farm Size by County",
            toc=[],
            summary_cues=["County", "Figures taken from the 2002 Census of Agriculture"],
            leading_block_lines=[
                "Colorado Agricultural Development Authority",
                "30% of Median Farm Size by County",
                "Figures taken from the 2002 Census of Agriculture",
            ],
            metadata_values=[],
            page_count=1,
        )
        self.assertEqual(statistical["document_type"], "statistical_table")
        self.assertEqual(statistical["document_purpose"], "statistical_reference")
        self.assertEqual(statistical["audience"], "analysts")
        self.assertGreaterEqual(statistical["semantic_confidence"], 0.75)

        job_listing = derive_document_facets(
            source_pdf="index2.pdf",
            title="Utah GIS Portal",
            toc=[],
            summary_cues=["Indeed.com index of Utah GIS jobs", "Utah GIS Jobs on indeed.com"],
            leading_block_lines=[
                "Indeed.com is an index of several online job postings.",
                "Location = Utah AND Description CONTAINS GIS",
                "Powered by Joomla! Generated: 25 April, 2010",
            ],
            metadata_values=[],
            page_count=1,
        )
        self.assertEqual(job_listing["document_type"], "web_job_listing")
        self.assertEqual(job_listing["document_purpose"], "employment_listing")
        self.assertEqual(job_listing["audience"], "job_seekers")
        self.assertGreaterEqual(job_listing["semantic_confidence"], 0.75)

        environmental = derive_document_facets(
            source_pdf="waste-site-reclassification.pdf",
            title="Waste Site Reclassification Form",
            toc=[],
            summary_cues=["Waste Site Reclassification Form", "Description of current waste site condition"],
            leading_block_lines=[
                "Originator Charlie Shipler Waste Site ID: 200-W48",
                "Description of current waste site condition:",
                "DOE Project Manager Signature Date",
                "Ecology Project Manager Signature Date",
            ],
            metadata_values=[],
            page_count=1,
        )
        self.assertEqual(environmental["document_type"], "environmental_site_record")
        self.assertEqual(environmental["document_purpose"], "institutional_reporting")
        self.assertEqual(environmental["audience"], "officials")
        self.assertGreaterEqual(environmental["semantic_confidence"], 0.75)

        correspondence = derive_document_facets(
            source_pdf="scanned-letter.pdf",
            title="The Rockefeller University",
            toc=[],
            summary_cues=["THE ROCKEFELLER UNIVERSITY NEW YORK 10021-6399"],
            leading_block_lines=[
                "THE ROCKEFELLER UNIVERSITY NEW YORK 10021-6399",
                "University letterhead",
            ],
            metadata_values=[],
            page_count=1,
        )
        self.assertEqual(correspondence["document_type"], "institutional_correspondence")
        self.assertEqual(correspondence["document_purpose"], "institutional_communication")
        self.assertEqual(correspondence["audience"], "institutional_staff")
        self.assertGreaterEqual(correspondence["semantic_confidence"], 0.75)

    def test_layout_sanity_check_json_for_multiple_pdfs(self) -> None:
        second_pdf = self.workspace / "financial-form.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Personal Financial Statement\n\n"
            "Cash in bank; notes due to banks; accounts payable; real estate mortgage payable.\n"
            "Section A: Assets\n"
            "Section B: Liabilities\n",
        )
        doc.save(second_pdf)
        doc.close()

        self._run("init", "--json")
        process = self._run(
            "layout-sanity-check",
            "--pdfs",
            f"{self.pdf_path},{second_pdf}",
            "--json",
        )
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertEqual(result["pdf_count"], 2)
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(result["all_pass"])
        for item in result["results"]:
            self.assertTrue(item["all_pass"])
            self.assertTrue(item["overview_answer"])
            self.assertTrue(item["type_answer"])
            self.assertTrue(item["purpose_answer"])
            self.assertTrue(item["audience_answer"])
            self.assertTrue(item["confidence_answer"])
            self.assertTrue(item["rationale_answer"])
            self.assertTrue(item["limits_answer"])
            self.assertIsNotNone(item["structure_confidence"])
            self.assertIsNotNone(item["layout_confidence"])
            self.assertIsNotNone(item["semantic_confidence"])
            self.assertTrue(item["semantic_confidence_label"])
            self.assertTrue(item["classification_status"])
            self.assertTrue(item["trust_policy"])

    def test_corpus_sanity_check_with_local_override(self) -> None:
        corpus_dir = self.workspace / "pdf-corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        entries = [
            (
                "FORMENTRY",
                "http://example.test/forms/financial_statement.pdf",
                "Personal Financial Statement\n\nTotal Assets\nTotal Liabilities\nNet Worth\n",
                2,
                "Acrobat PDFMaker",
                "Adobe PDF Library",
            ),
            (
                "SCANENTRY",
                "http://example.test/scans/checklist.pdf",
                "Checklist Appendix\n\nConfirm identity\nConfirm medication\nConfirm follow-up\n",
                1,
                "Acrobat Capture 3.0",
                "Scan Producer",
            ),
            (
                "GUIDEENTRY",
                "http://example.test/guidance/incident_guide.pdf",
                "Guidance Note\n\nThis guidance explains incident management and reporting.\n",
                4,
                "Word",
                "Microsoft Print to PDF",
            ),
        ]

        for digest, _, text, pages, _, _ in entries:
            doc = fitz.open()
            for _ in range(pages):
                page = doc.new_page()
                page.insert_text((72, 72), text)
            doc.save(corpus_dir / f"{digest}.pdf")
            doc.close()

        metadata_lines = [
            "urlkey,timestamp,original,mimetype,statuscode,digest,pdf_version,creator_tool,producer,date_created,pages,page_width,page_height,surface_area,file_size,sha256,sha512"
        ]
        for digest, original, _, pages, creator_tool, producer in entries:
            metadata_lines.append(
                ",".join(
                    [
                        original.removeprefix("http://"),
                        "20260526000000",
                        original,
                        "application/pdf",
                        "200",
                        digest,
                        "1.4",
                        creator_tool,
                        producer,
                        "2026-05-26T00:00:00Z",
                        str(pages),
                        "612",
                        "792",
                        "94",
                        "12000",
                        "sha256",
                        "sha512",
                    ]
                )
            )
        (corpus_dir / "lcwa_gov_pdf_metadata.csv").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

        self._run("init", "--json")
        process = self._run(
            "corpus-sanity-check",
            "--corpus-dir",
            str(corpus_dir),
            "--sample-size",
            "3",
            "--json",
        )
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["corpus_pdf_count"], 3)
        self.assertEqual(result["sample_profile"], "custom")
        self.assertEqual(result["requested_sample_size"], 3)
        self.assertEqual(result["sample_size"], 3)
        self.assertEqual(len(result["results"]), 3)
        self.assertTrue(Path(result["snapshot_path"]).exists())
        self.assertIn("classification_status_counts", result["summary"])
        self.assertIn("trust_policy_counts", result["summary"])
        self.assertIn("bucket_counts", result["summary"])
        self.assertIn("semantic_pass_rate", result["summary"])
        self.assertIn("specific_document_rate", result["summary"])
        self.assertIn("specific_purpose_rate", result["summary"])
        self.assertIn("layer_summary", result)
        self.assertIn("layer_stability", result)
        self.assertIn("architecture_gates", result)
        self.assertIn("bucket_diagnostics", result)
        self.assertIn("follow_up_actions", result)
        self.assertIn("contract_gate", result)
        self.assertTrue(result["contract_gate"]["all_pass"])
        self.assertIn("processing", result["layer_summary"])
        self.assertIn("semantics", result["layer_summary"])
        self.assertIn("trust", result["layer_summary"])
        self.assertIn("technical_pass_rate", result["layer_summary"]["processing"])
        self.assertIn("semantic_pass_rate", result["layer_summary"]["semantics"])
        self.assertIn("semantic_gate_pass", result["architecture_gates"])
        self.assertIn("bucket_gate_pass", result["architecture_gates"])
        self.assertIn("bucket_follow_up_count", result["architecture_gates"])
        self.assertGreaterEqual(result["summary"]["bucket_counts"].get("form_like", 0), 1)
        self.assertGreaterEqual(result["bucket_diagnostics"]["form_like"]["sample_count"], 1)
        self.assertIn("dominant_failure_reasons", result["bucket_diagnostics"]["form_like"])
        self.assertIn("failure_examples", result["bucket_diagnostics"]["form_like"])
        self.assertGreater(result["summary"]["specific_document_rate"], 0.0)
        self.assertGreater(result["summary"]["specific_purpose_rate"], 0.0)
        self.assertTrue(all(item["overview_answer"] for item in result["results"]))
        self.assertTrue(all(item["confidence_answer"] for item in result["results"]))


if __name__ == "__main__":
    unittest.main()
