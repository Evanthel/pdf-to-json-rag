"""Public-surface smoke tests for the packaged CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pdf_to_json_rag import cli as cli_module
from pdf_to_json_rag.chunking import chunk_document
from pdf_to_json_rag.document_facets import derive_document_facets
from pdf_to_json_rag.extraction import ExtractedBlock
from pdf_to_json_rag.schemas import DocumentRecord


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
        self.assertTrue(result["answer"]["answer"])
        written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written["command"], "smoke-check")
        self.assertTrue(written["result"]["all_pass"])

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
        self.assertIsNotNone(smoke_payload["result"]["document"]["structure_confidence"])
        self.assertIsNotNone(smoke_payload["result"]["document"]["layout_confidence"])
        self.assertIsNotNone(smoke_payload["result"]["document"]["semantic_confidence"])

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
        self.assertIn(
            "shortlist_breakdown",
            answer_payload["result"]["answer_trace"]["document_selection"],
        )
        support_trace = answer_payload["result"]["answer_trace"]["support_trace"]
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
                page_num=0,
                text="Background This guide explains field safety procedures.",
                bbox=None,
                reading_order_index=0,
                block_kind="text",
            ),
            ExtractedBlock(
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
        self.assertEqual(result["sample_size"], 3)
        self.assertEqual(len(result["results"]), 3)
        self.assertIn("classification_status_counts", result["summary"])
        self.assertIn("trust_policy_counts", result["summary"])
        self.assertIn("bucket_counts", result["summary"])
        self.assertGreaterEqual(result["summary"]["bucket_counts"].get("form_like", 0), 1)
        self.assertTrue(all(item["overview_answer"] for item in result["results"]))
        self.assertTrue(all(item["confidence_answer"] for item in result["results"]))


if __name__ == "__main__":
    unittest.main()
