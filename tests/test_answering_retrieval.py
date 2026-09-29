"""Answer assembly, query planning, and retrieval contract tests."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock


from tests.cli_test_support import CliPublicSurfaceTestBase

from pdf_to_json_rag import document_inventory as document_inventory_module
from pdf_to_json_rag import intent_config as intent_config_module
from pdf_to_json_rag import retrieval as retrieval_module
from pdf_to_json_rag.answering import (
    EvidenceSentence,
    _should_abstain,
    answer_from_chunks,
)
from pdf_to_json_rag.query_planning import plan_query
from pdf_to_json_rag.retrieval import build_retrieval_contract
from pdf_to_json_rag.schemas import ChunkRecord


class AnsweringRetrievalTests(CliPublicSurfaceTestBase):
    def test_monoclonal_cell_culture_evidence_does_not_answer_clinical_prevention(
        self,
    ) -> None:
        cell_culture_evidence = EvidenceSentence(
            chunk_id="review-chunk-1",
            page_start=1,
            page_end=1,
            section_title="Experimental studies",
            sentence=(
                "When human cell cultures are pretreated with monoclonal antibodies, "
                "viral infection appears blocked."
            ),
            score=5.0,
        )
        direct_clinical_evidence = EvidenceSentence(
            chunk_id="review-chunk-2",
            page_start=2,
            page_end=2,
            section_title="Clinical evidence",
            sentence="A clinical trial found monoclonal antibodies did not prevent common colds.",
            score=5.0,
        )

        query = "Do monoclonal antibodies prevent the common cold?"
        self.assertTrue(_should_abstain(query, [cell_culture_evidence]))
        self.assertFalse(_should_abstain(query, [direct_clinical_evidence]))

    def test_generic_form_question_uses_context_snippet_fallback(self) -> None:
        chunk = ChunkRecord(
            doc_id="ship-application",
            chunk_id="ship-fields",
            source_pdf="ship.pdf",
            text="SHIP INFORMATION Ship Name: Example Star Ship Type: Cargo Type of Cargo: Grain Home Port: Baltimore",
            page_start=1,
            page_end=1,
            reading_order_index=1,
            section_title="SHIP INFORMATION",
            section_path=["Application for Ship Participation", "SHIP INFORMATION"],
            quality_score=1.0,
        )

        result = answer_from_chunks(
            "What ship information fields are requested?", [chunk]
        )

        self.assertNotIn("No grounded answer could be assembled", result.answer)
        self.assertIn("Ship Name", result.answer)
        self.assertIn("Home Port", result.answer)
        self.assertEqual(
            result.answer_trace["template_id"],
            "grounded_evidence.context_snippet_fallback",
        )

    def test_context_snippet_fallback_preserves_compact_form_rows(self) -> None:
        chunk = ChunkRecord(
            doc_id="ship-application",
            chunk_id="ship-fields",
            source_pdf="ship.pdf",
            text=(
                "Name of Company: Adopt-A-Ship Company Point of Contact: Telephone Number: "
                "Email: Fax: Ship Name: Ship Type/Length: Type of Cargo: Home Port/Sailing Routes:"
            ),
            page_start=1,
            page_end=1,
            reading_order_index=1,
            section_title="SHIP INFORMATION",
            section_path=["Application for Ship Participation", "SHIP INFORMATION"],
            quality_score=1.0,
        )

        result = answer_from_chunks(
            "What ship information fields are requested?", [chunk]
        )

        self.assertIn("Ship Name", result.answer)
        self.assertIn("Type of Cargo", result.answer)
        self.assertIn("Home Port", result.answer)

    def test_context_snippet_fallback_extracts_parenthetical_form_fields(self) -> None:
        chunk = ChunkRecord(
            doc_id="voter-transfer",
            chunk_id="voter-fields",
            source_pdf="voter.pdf",
            text=(
                "Voter Registration Transfer Form Voter's Name (Please Print)________________ "
                "Old Address________________ New Address________________ "
                "Social Security No (last 4 digits).____"
            ),
            page_start=1,
            page_end=1,
            reading_order_index=1,
            section_title="Voter Registration Transfer Form",
            section_path=["Voter Registration Transfer Form"],
            quality_score=1.0,
        )

        result = answer_from_chunks(
            "What voter registration transfer fields are requested?", [chunk]
        )

        self.assertIn("Voter's Name", result.answer)
        self.assertIn("Old Address", result.answer)
        self.assertIn("New Address", result.answer)
        self.assertIn("Social Security", result.answer)

    def test_context_snippet_fallback_extracts_hyphenated_program_name(self) -> None:
        chunk = ChunkRecord(
            doc_id="ship-application",
            chunk_id="ship-program",
            source_pdf="ship.pdf",
            text="Application for Ship Participation in the Adopt-A-Ship Program.",
            page_start=1,
            page_end=1,
            reading_order_index=1,
            section_title="Application for Ship Participation",
            section_path=["Application for Ship Participation"],
            quality_score=1.0,
        )

        result = answer_from_chunks(
            "What program is mentioned in the application?", [chunk]
        )

        self.assertIn("Adopt-A-Ship Program", result.answer)

    def test_context_snippet_fallback_prefers_claimant_caption_name(self) -> None:
        respondent_chunk = ChunkRecord(
            doc_id="court-opinion",
            chunk_id="respondent-caption",
            source_pdf="court.pdf",
            text="Claimant-Appellant, v. JAMES B. PEAKE, M.D., Respondent-Appellee.",
            page_start=1,
            page_end=1,
            reading_order_index=1,
            section_title="Caption",
            section_path=["Caption"],
            quality_score=1.0,
        )
        claimant_chunk = ChunkRecord(
            doc_id="court-opinion",
            chunk_id="claimant-caption",
            source_pdf="court.pdf",
            text="FORTUNATA CAPELLAN, Claimant-Appellant.",
            page_start=1,
            page_end=1,
            reading_order_index=2,
            section_title="Caption",
            section_path=["Caption"],
            quality_score=1.0,
        )

        result = answer_from_chunks(
            "Who is the claimant-appellant in the court opinion?",
            [respondent_chunk, claimant_chunk],
        )

        self.assertIn("FORTUNATA CAPELLAN", result.answer)
        self.assertNotIn("JAMES B. PEAKE", result.answer)

    def test_plan_query_distinguishes_type_purpose_audience_confidence_rationale_and_limits(
        self,
    ) -> None:
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
        self.assertEqual(
            audience_payload["result"]["query_intent"], "document_audience"
        )
        self.assertEqual(
            confidence_payload["result"]["query_intent"], "document_confidence"
        )
        self.assertEqual(
            rationale_payload["result"]["query_intent"],
            "document_classification_rationale",
        )
        self.assertEqual(
            limits_payload["result"]["query_intent"], "document_classification_limits"
        )

    def test_retrieval_contract_splits_single_doc_document_understanding_and_cross_document(
        self,
    ) -> None:
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
        with (
            mock.patch.object(
                intent_config_module, "_document_metadata_index", return_value={}
            ),
            mock.patch.object(
                document_inventory_module, "load_document_inventory", return_value=()
            ),
        ):
            nonmedical_listing_plan = plan_query(
                "Which sources in the benchmark discuss deep learning or data incident response?"
            )
        vitamin_null_plan = plan_query(
            "Does vitamin C prevent the common cold in normal populations?"
        )
        vitamin_stress_plan = plan_query(
            "Does vitamin C help people under cold stress?"
        )
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
                reranked, fallback_reason = retrieval_module._cross_encoder_rerank_hits(
                    chunks, "value"
                )

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

    def test_qip_recipient_query_prefers_title_recipient_slide(self) -> None:
        recipient_slide = ChunkRecord(
            doc_id="microsoft-powerpoint-copy-of-v-quality-improvement-program",
            chunk_id="qip-title",
            source_pdf="Microsoft PowerPoint - Copy of V - Quality Improvement Program.pdf",
            text="Terry Smith Division Director Division of Long-Term Care Department of Medical Assistance Services",
            page_start=1,
            page_end=1,
            reading_order_index=3,
            section_title="The Joint Commission on Health Care",
            section_path=[
                "Microsoft PowerPoint - Copy of V - Quality Improvement Program",
                "The Joint Commission on Health Care",
            ],
            quality_score=0.7,
            noise_labels=["title_fragment"],
        )
        outline_slide = ChunkRecord(
            doc_id="microsoft-powerpoint-copy-of-v-quality-improvement-program",
            chunk_id="qip-outline",
            source_pdf="Microsoft PowerPoint - Copy of V - Quality Improvement Program.pdf",
            text="Background Civil Money Penalty Funds QIP Advisory Committee Committee Discussions",
            page_start=2,
            page_end=2,
            reading_order_index=6,
            section_title="Presentation Outline",
            section_path=[
                "Microsoft PowerPoint - Copy of V - Quality Improvement Program",
                "Presentation Outline",
            ],
            quality_score=1.0,
        )
        off_topic = ChunkRecord(
            doc_id="court-opinion",
            chunk_id="court",
            source_pdf="court.pdf",
            text="The affidavit was presented to the regional office and reviewed by the court.",
            page_start=8,
            page_end=8,
            reading_order_index=50,
            section_title="Court Proceedings",
            quality_score=1.0,
        )

        reranked = retrieval_module._rerank_hits(
            [off_topic, outline_slide, recipient_slide],
            "Who was the QIP presentation presented to?",
            rank_prior_weight=0.0,
        )

        self.assertEqual(reranked[0].chunk_id, "qip-title")
        self.assertGreater(
            reranked[0].retrieval_signals["total"],
            reranked[1].retrieval_signals["total"],
        )

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

    def test_source_anchored_grounded_answer_filters_to_preferred_document(
        self,
    ) -> None:
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


if __name__ == "__main__":
    unittest.main()
