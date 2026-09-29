"""CLI quality profiles, diagnostics, and answer-trust tests."""

from __future__ import annotations

import unittest


from tests.cli_test_support import CliPublicSurfaceTestBase

from pdf_to_json_rag import cli as cli_module


class CliQualityProfileTests(CliPublicSurfaceTestBase):
    def test_answer_contract_health_and_quality_profile_are_readable(self) -> None:
        trace = {
            "answer_mode": "document_overview",
            "retrieval_contract": {"retrieval_path": "document_understanding"},
            "document_synthesis": {
                "support_scope": "selected_docs",
                "support_doc_ids": ["demo"],
            },
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
        self.assertEqual(
            profile["semantic_confidence"]["classification_status"], "well_supported"
        )
        self.assertEqual(profile["retrieval_readiness"]["support_doc_ids"], ["demo"])
        self.assertEqual(
            profile["retrieval_readiness"]["retrieval_contract_status"]["status"],
            "pass",
        )
        self.assertTrue(
            profile["retrieval_readiness"]["answer_source_mix"]["support_trace"][
                "present"
            ]
        )
        self.assertEqual(profile["answer_trust"]["status"], "pass")
        self.assertEqual(
            profile["processing_quality"]["drilldown"]["text_extraction_coverage"], None
        )
        summary = cli_module._quality_profile_summary(profile)
        self.assertEqual(summary["overall_status"], "pass")
        self.assertEqual(summary["recommended_next_action"], "none")

    def test_retrieval_contract_status_warns_on_support_mismatch(self) -> None:
        trace = {
            "answer_mode": "document_overview",
            "candidate_doc_ids": ["alpha"],
            "retrieval_contract": {"retrieval_path": "document_understanding"},
            "document_selection": {
                "selected_doc_ids": ["alpha"],
                "candidate_doc_ids": ["alpha"],
            },
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
            "support_trace": [
                {
                    "doc_id": "beta",
                    "support_fragments": ["document type: guidance note"],
                }
            ],
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
            "document_synthesis": {
                "support_scope": "selected_docs",
                "support_doc_ids": ["demo"],
            },
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

    def test_quality_profile_recommends_processing_follow_up_on_low_signal_payload(
        self,
    ) -> None:
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
        self.assertEqual(
            profile["recommended_next_action"], "inspect_document_or_try_ocr"
        )
        self.assertIn("chunks_created", profile["processing_quality"]["reasons"])

    def test_processing_diagnostics_classifies_scan_form_and_table_payloads(
        self,
    ) -> None:
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
        self.assertEqual(
            cli_module._assessment_profile(base_doc, {"taxonomy": []}), "short_document"
        )
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
                    "document_type": "financial_statement",
                    "extraction_summary": {
                        "block_role_counts": {"key_value": 3},
                        "layout_signal_counts": {},
                    },
                },
                {"taxonomy": ["table_or_form_heavy"]},
            ),
            "table_heavy_pdf",
        )
        support = cli_module._structure_support_summary(
            {
                **base_doc,
                "document_type": "financial_statement",
                "section_count": 4,
                "extraction_summary": {"block_role_counts": {"key_value": 3}},
            },
            {"taxonomy": ["table_or_form_heavy"], "drilldown": {"chunk_count": 6}},
        )
        self.assertEqual(support["status"], "structured")
        self.assertGreaterEqual(support["table_signal_count"], 3)


if __name__ == "__main__":
    unittest.main()
