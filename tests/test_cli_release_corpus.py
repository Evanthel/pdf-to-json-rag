"""Release gates, corpus snapshots, and real-PDF workflow tests."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import fitz

from tests.cli_test_support import CliPublicSurfaceTestBase

from pdf_to_json_rag import cli as cli_module


class CliReleaseCorpusTests(CliPublicSurfaceTestBase):
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
        self.assertEqual(
            manifest["selected_bucket_counts"],
            {"form_like": 1, "medium_doc": 1, "short_doc": 1},
        )
        self.assertEqual(len(manifest["selected_digest_checksum"]), 64)

    def test_release_check_compact_payload_lists_gate_statuses(self) -> None:
        compact = cli_module._release_check_compact_payload(
            {
                "doctor": {
                    "ready_for_public_cli": True,
                    "runtime": {"runtime_decision": {"default_backend": "auto"}},
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
        self.assertEqual(compact["runtime_decision"]["default_backend"], "auto")
        self.assertTrue(compact["product_gate"]["all_pass"])
        self.assertEqual(compact["product_gate"]["public_path"]["status"], "pass")
        self.assertEqual(compact["product_gate"]["benchmark"]["status"], "pass")
        self.assertEqual(compact["product_gate"]["corpus"]["status"], "pass")
        self.assertEqual(compact["internal_regressions"]["selected_shard_count"], 1)
        self.assertEqual(compact["internal_regressions"]["shards"][0]["status"], "pass")
        self.assertEqual(compact["local_corpus_sanity"]["gate"]["status"], "pass")

    def test_release_check_compact_payload_marks_corpus_review_with_examples(
        self,
    ) -> None:
        compact = cli_module._release_check_compact_payload(
            {
                "doctor": {
                    "ready_for_public_cli": True,
                    "runtime": {"runtime_decision": {}},
                },
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
                            "default_backend": "auto",
                            "recommended_opt_in_backend": "sentence-transformers",
                            "not_default_reason": "auto uses cached sentence-transformers with hash fallback.",
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
        self.assertEqual(payload["runtime_decision"]["default_backend"], "auto")
        self.assertEqual(
            payload["scope"]["sentence_transformers"], "default_when_cached"
        )
        self.assertEqual(payload["public_smoke_quality"]["overall_status"], "pass")

    def test_real_pdf_modes_default_fast_and_all_explicit(self) -> None:
        self.assertEqual(cli_module._real_pdf_modes_from_arg(None), ["default-auto"])
        self.assertEqual(
            cli_module._real_pdf_modes_from_arg("all"),
            ["default-auto", "hash-baseline", "cross-encoder", "llm-synthesis"],
        )

    def test_real_pdf_answer_quality_summary_reports_weak_cases(self) -> None:
        summary = cli_module._real_pdf_answer_quality_summary(
            [
                {
                    "case_id": "strong",
                    "bucket": "form_like",
                    "answer_keyword_coverage": {"coverage": 1.0},
                    "answer_preview": "Ship Name and Home Port are listed.",
                },
                {
                    "case_id": "weak",
                    "bucket": "form_like",
                    "answer_keyword_coverage": {"coverage": 0.0},
                    "answer_preview": "No grounded answer could be assembled from the retrieved context.",
                },
            ]
        )

        self.assertEqual(summary["weak_case_count"], 1)
        self.assertEqual(summary["abstained_case_count"], 1)
        self.assertEqual(summary["weak_case_ids"], ["weak"])
        self.assertEqual(summary["weak_buckets"][0]["bucket"], "form_like")

    def test_real_pdf_weak_case_workbench_reports_missing_keywords_and_chunks(
        self,
    ) -> None:
        workbench = cli_module._real_pdf_weak_case_workbench(
            [
                {
                    "case_id": "weak-form",
                    "bucket": "form_like",
                    "query": "What fields are requested?",
                    "rank": 1,
                    "retrieved_doc_ids": ["form-doc"],
                    "selected_chunk_ids": ["form-doc-chunk-0001"],
                    "answer_keyword_coverage": {
                        "coverage": 0.25,
                        "matched": ["Old Address"],
                        "missing": ["New Address", "Social Security"],
                    },
                    "evidence_keyword_coverage": {
                        "coverage": 1.0,
                        "matched": ["Old Address", "New Address", "Social Security"],
                        "missing": [],
                    },
                    "answer_preview": "Old Address is visible.",
                }
            ]
        )

        self.assertEqual(workbench["weak_case_count"], 1)
        item = workbench["cases"][0]
        self.assertEqual(item["case_id"], "weak-form")
        self.assertEqual(item["missing_keywords"], ["New Address", "Social Security"])
        self.assertEqual(item["selected_chunks"], ["form-doc-chunk-0001"])
        self.assertEqual(
            item["suggested_action"],
            "improve answer extraction around missing keywords",
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
        self.assertEqual(payload["review_status"], "pass")
        self.assertEqual(payload["corpus_review"]["status"], "pass")
        self.assertFalse(
            payload["corpus_review"]["model_experiment_scope"]["worth_running"]
        )
        self.assertTrue(payload["corpus_diff_summary"]["all_pass"])
        checksum_check = {
            item["name"]: item["status"]
            for item in payload["corpus_diff_summary"]["checks"]
        }
        self.assertEqual(checksum_check["sample_checksum"], "skip")

    def test_corpus_profile_compare_review_scopes_model_experiments(self) -> None:
        baseline_path = self.workspace / "quick-review.json"
        candidate_path = self.workspace / "balanced-review.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "sample_profile": "quick",
                    "sample_size": 4,
                    "sample_manifest": {"selected_digest_checksum": "aaa"},
                    "summary": {
                        "technical_pass_rate": 1.0,
                        "semantic_pass_rate": 1.0,
                        "avg_structure_confidence": 0.75,
                        "avg_layout_confidence": 0.8,
                        "avg_semantic_confidence": 0.9,
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
        candidate_path.write_text(
            json.dumps(
                {
                    "sample_profile": "balanced",
                    "sample_size": 12,
                    "sample_manifest": {"selected_digest_checksum": "bbb"},
                    "summary": {
                        "technical_pass_rate": 1.0,
                        "semantic_pass_rate": 1.0,
                        "avg_structure_confidence": 0.65,
                        "avg_layout_confidence": 0.8,
                        "avg_semantic_confidence": 0.9,
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

        self.assertFalse(payload["all_pass"])
        self.assertEqual(payload["review_status"], "review")
        review = payload["corpus_review"]
        self.assertEqual(review["top_metrics"][0]["metric"], "avg_structure_confidence")
        self.assertEqual(
            review["model_experiment_scope"]["candidate_backends"], ["cross-encoder"]
        )
        self.assertFalse(review["model_experiment_scope"]["default_change_allowed"])

    def test_corpus_snapshot_aliases_and_compact_write_do_not_require_reprocessing(
        self,
    ) -> None:
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
            quick_latest_path = cli_module._corpus_snapshot_path_for_profile(
                "quick-latest"
            )
            compact_path = self.workspace / "corpus_sanity_quick_compact_snapshot.json"
        finally:
            object.__setattr__(cli_module.PATHS, "data_eval", original_data_eval)

        self.assertEqual(snapshot_path, self.workspace / "corpus_sanity_snapshot.json")
        self.assertEqual(latest_path, self.workspace / "corpus_sanity_snapshot.json")
        self.assertEqual(
            quick_latest_path, self.workspace / "corpus_sanity_quick_snapshot.json"
        )
        compact = json.loads(compact_path.read_text(encoding="utf-8"))
        self.assertNotIn("results", compact)
        self.assertEqual(compact["sample_manifest"]["selected_digest_checksum"], "abc")

    def test_corpus_sample_profile_resolution(self) -> None:
        self.assertEqual(cli_module._resolve_corpus_sample_size("quick", None), 4)
        self.assertEqual(cli_module._resolve_corpus_sample_size("balanced", None), 12)
        self.assertEqual(cli_module._resolve_corpus_sample_size("stress", None), 24)
        self.assertEqual(cli_module._resolve_corpus_sample_size("stress", 3), 3)

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

    def test_real_ground_truth_check_accepts_explicit_public_corpus_directory(
        self,
    ) -> None:
        corpus_dir = self.workspace / "public-corpus"
        corpus_dir.mkdir()
        public_pdf = corpus_dir / "public-sample.pdf"
        self._create_demo_pdf(public_pdf)
        eval_path = self.workspace / "public-ground-truth.json"
        eval_path.write_text(
            json.dumps(
                [
                    {
                        "case_id": "public_safety_guide",
                        "pdf_digest": "public-sample",
                        "bucket": "public_ci",
                        "query": "What does the guide cover?",
                        "expected_keywords": ["safety checks", "incident response"],
                        "evidence_keywords": ["safety checks", "incident response"],
                    }
                ]
            ),
            encoding="utf-8",
        )

        process = self._run(
            "real-ground-truth-check",
            "--corpus-dir",
            str(corpus_dir),
            "--eval-file",
            str(eval_path),
            "--modes",
            "default-auto",
            "--json",
        )

        result = json.loads(process.stdout)["result"]
        self.assertTrue(result["all_pass"])
        self.assertTrue(result["quality_gate"]["passed"])
        self.assertTrue(result["processing_quality"]["all_pass"])
        self.assertEqual(result["pdf_count"], 1)
        self.assertEqual(result["case_count"], 1)
        self.assertGreater(result["default_index_manifest"]["chunk_count"], 0)

    def test_real_ground_truth_keyword_coverage_tolerates_pdf_spacing_loss(
        self,
    ) -> None:
        result = cli_module._keyword_coverage(
            "The checklist is composedof13sectionsand25items.",
            ["13 sections", "25 items"],
        )

        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["missing"], [])

    def test_real_ground_truth_check_fails_when_explicit_corpus_is_missing(
        self,
    ) -> None:
        eval_path = self.workspace / "missing-corpus-ground-truth.json"
        eval_path.write_text(
            json.dumps(
                [
                    {
                        "case_id": "missing_public_pdf",
                        "pdf_digest": "missing-public-pdf",
                        "bucket": "public_ci",
                        "query": "What does this file cover?",
                        "expected_keywords": ["missing"],
                        "evidence_keywords": ["missing"],
                    }
                ]
            ),
            encoding="utf-8",
        )

        process = self._run(
            "real-ground-truth-check",
            "--corpus-dir",
            str(self.workspace / "missing-corpus"),
            "--eval-file",
            str(eval_path),
            "--modes",
            "default-auto",
            "--json",
            expect_ok=False,
        )

        self.assertNotEqual(process.returncode, 0)
        payload = json.loads(process.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "missing_local_pdf_corpus")

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
        (corpus_dir / "lcwa_gov_pdf_metadata.csv").write_text(
            "\n".join(metadata_lines) + "\n", encoding="utf-8"
        )

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
        self.assertGreaterEqual(
            result["summary"]["bucket_counts"].get("form_like", 0), 1
        )
        self.assertGreaterEqual(
            result["bucket_diagnostics"]["form_like"]["sample_count"], 1
        )
        self.assertIn(
            "dominant_failure_reasons", result["bucket_diagnostics"]["form_like"]
        )
        self.assertIn("failure_examples", result["bucket_diagnostics"]["form_like"])
        self.assertGreater(result["summary"]["specific_document_rate"], 0.0)
        self.assertGreater(result["summary"]["specific_purpose_rate"], 0.0)
        self.assertTrue(all(item["overview_answer"] for item in result["results"]))
        self.assertTrue(all(item["confidence_answer"] for item in result["results"]))


if __name__ == "__main__":
    unittest.main()
