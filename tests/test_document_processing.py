"""Extraction, layout, chunking, and document semantics tests."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


from tests.cli_test_support import CliPublicSurfaceTestBase

from pdf_to_json_rag import indexing as indexing_module
from pdf_to_json_rag import retrieval as retrieval_module
from pdf_to_json_rag.chunking import (
    chunk_document,
    normalize_reading_order,
    process_saved_document_to_chunks,
)
from pdf_to_json_rag.content_metadata import (
    classify_block_metadata,
    infer_layout_signals,
)
from pdf_to_json_rag.document_facets import derive_document_facets
from pdf_to_json_rag.document_semantics import interpret_document_semantics
from pdf_to_json_rag.extraction import (
    ExtractedBlock,
    _build_extracted_block,
    _sort_page_blocks_reading_order,
    extract_pdfplumber_table_blocks,
    process_native_pdf_to_json,
    probe_pdfplumber_tables,
)
from pdf_to_json_rag.indexing import build_local_index
from pdf_to_json_rag.schemas import ChunkRecord, DocumentRecord


class DocumentProcessingTests(CliPublicSurfaceTestBase):
    def test_build_local_index_closes_chroma_client_on_failure(self) -> None:
        chunk = ChunkRecord(
            doc_id="doc",
            chunk_id="chunk-1",
            source_pdf="demo.pdf",
            text="Grounded content.",
            page_start=1,
            page_end=1,
            reading_order_index=1,
        )
        client = mock.Mock()
        client.get_or_create_collection.return_value.add.side_effect = RuntimeError(
            "add failed"
        )

        def embedder(texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

        with (
            mock.patch.object(
                indexing_module, "local_chroma_client", return_value=client
            ),
            mock.patch.object(
                indexing_module,
                "_load_embedder",
                return_value=(
                    embedder,
                    {
                        "embedding_backend": "hash-fallback",
                        "embedding_model": "hash-384",
                    },
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "add failed"),
        ):
            build_local_index([chunk], index_dir=self.workspace / "failing-index")

        client.close.assert_called_once_with()

    def test_retrieval_query_closes_chroma_client_on_failure(self) -> None:
        client = mock.Mock()
        client.get_collection.return_value.query.side_effect = RuntimeError(
            "query failed"
        )

        with (
            mock.patch.object(
                retrieval_module, "local_chroma_client", return_value=client
            ),
            self.assertRaisesRegex(RuntimeError, "query failed"),
        ):
            retrieval_module._query_local_collection(
                index_dir=self.workspace / "index",
                collection_name="collection",
                query_embedding=[0.0] * 384,
                candidate_k=5,
            )

        client.close.assert_called_once_with()

    def test_close_chroma_client_stops_pre_1_0_system(self) -> None:
        class LegacyClient:
            def __init__(self) -> None:
                self._identifier = "legacy-index"
                self._system = mock.Mock()
                self._identifier_to_system = {self._identifier: self._system}

        client = LegacyClient()
        indexing_module.close_chroma_client(client)

        client._system.stop.assert_called_once_with()
        self.assertEqual(client._identifier_to_system, {})

    def test_cleanup_unused_segments_closes_sqlite_connection(self) -> None:
        index_dir = self.workspace / "cleanup-index"
        index_dir.mkdir()
        (index_dir / "chroma.sqlite3").touch()
        connection = mock.Mock()
        connection.execute.return_value.fetchall.return_value = []

        with mock.patch.object(
            indexing_module.sqlite3, "connect", return_value=connection
        ):
            removed = indexing_module.cleanup_unused_segment_dirs(index_dir)

        self.assertEqual(removed, [])
        connection.close.assert_called_once_with()

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
            blocks, probe = extract_pdfplumber_table_blocks(
                self.pdf_path, doc_id="demo"
            )

        self.assertTrue(probe["available"])
        self.assertEqual(probe["table_count"], 1)
        self.assertEqual(probe["supplemental_block_count"], 1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_kind, "table_like")
        self.assertEqual(blocks[0].block_role, "table_like")
        self.assertIn("pdfplumber_table", blocks[0].structural_flags)
        self.assertIn("Name | Score", blocks[0].text)
        self.assertEqual(blocks[0].bbox, [0.05, 0.05, 0.95, 0.3])

    def test_presentation_slide_chunks_carry_role_metadata(self) -> None:
        document = DocumentRecord(
            doc_id="qip-demo",
            source_pdf="Microsoft PowerPoint - Copy of V - Quality Improvement Program.pdf",
            page_count=1,
            title="Microsoft PowerPoint - Copy of V - Quality Improvement Program",
            detected_language="en",
        )
        blocks = [
            ExtractedBlock(
                block_id="qip-title",
                page_num=0,
                text="The Joint Commission on Health Care",
                bbox=None,
                reading_order_index=1,
                block_kind="heading",
                block_role="heading",
            ),
            ExtractedBlock(
                block_id="qip-presenter",
                page_num=0,
                text="Terry Smith Division Director Division of Long-Term Care Department of Medical Assistance Services",
                bbox=None,
                reading_order_index=2,
                block_kind="text",
                block_role="paragraph",
            ),
        ]

        chunks = chunk_document(document, blocks, target_chars=400, min_chunk_chars=40)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_role, "presentation_title")
        self.assertEqual(chunks[0].chunk_strategy, "presentation_slide")
        self.assertIn("presentation_title", chunks[0].content_hints)

    def test_optional_low_confidence_semantics_multipass_is_env_gated(self) -> None:
        base = interpret_document_semantics(
            source_pdf="mystery.pdf",
            title="Mystery",
            toc=[],
            summary_cues=[],
            discovery_terms=[
                "financial statement",
                "total assets",
                "total liabilities",
            ],
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
                discovery_terms=[
                    "financial statement",
                    "total assets",
                    "total liabilities",
                ],
                leading_block_lines=[],
                metadata_values=[],
                page_count=1,
            )

        self.assertNotIn(
            "optional_low_confidence_multipass_accepted", base.semantic_rationale
        )
        self.assertIn(
            "optional_low_confidence_multipass_accepted", reviewed.semantic_rationale
        )
        self.assertGreaterEqual(reviewed.semantic_confidence, base.semantic_confidence)

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
        checklist_chunk = next(
            chunk for chunk in chunks if chunk.section_title == "CHECKLIST"
        )
        self.assertEqual(
            checklist_chunk.section_path, ["Demo Safety Guide", "CHECKLIST"]
        )
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
        self.assertTrue(
            all(chunk.chunk_type in {"table", "checklist"} for chunk in chunks)
        )
        self.assertTrue(
            any("informed consent obtained" in chunk.text for chunk in chunks)
        )

    def test_inspection_report_chunks_keep_heading_field_text(self) -> None:
        document = DocumentRecord(
            doc_id="demo-inspection",
            source_pdf="demo-inspection.pdf",
            page_count=1,
            title="Annual Inspection Report",
            detected_language="en",
            document_type="inspection_report",
            structure_confidence=0.68,
            layout_confidence=0.8,
        )
        blocks = [
            ExtractedBlock(
                block_id="inspection-heading-owner",
                page_num=0,
                text="SHARON GRAVES",
                bbox=None,
                reading_order_index=0,
                block_kind="heading",
                block_role="heading",
            ),
            ExtractedBlock(
                block_id="inspection-site",
                page_num=0,
                text="ROCK CREEK KENNEL on-site inspection record with attending veterinarian notes.",
                bbox=None,
                reading_order_index=1,
                block_kind="text",
                block_role="form_field",
            ),
        ]

        chunks = chunk_document(document, blocks, target_chars=300, min_chunk_chars=20)
        chunk_text = " ".join(chunk.text for chunk in chunks)

        self.assertIn("SHARON GRAVES", chunk_text)
        self.assertIn("ROCK CREEK KENNEL", chunk_text)

    def test_short_heading_payload_document_keeps_heading_text_in_chunk(self) -> None:
        document = DocumentRecord(
            doc_id="demo-bulletin",
            source_pdf="demo-bulletin.pdf",
            page_count=1,
            title="The Online Office of Congressman Danny K. Davis",
            detected_language="en",
            document_type="government_bulletin",
            document_purpose="public_notice",
        )
        blocks = [
            ExtractedBlock(
                block_id="bulletin-title",
                page_num=0,
                text="The Online Office of Congressman Danny K. Davis",
                bbox=None,
                reading_order_index=0,
                block_kind="heading",
                block_role="heading",
            ),
            ExtractedBlock(
                block_id="bulletin-newsletter",
                page_num=0,
                text="ENewsletter",
                bbox=None,
                reading_order_index=1,
                block_kind="heading",
                block_role="heading",
            ),
            ExtractedBlock(
                block_id="bulletin-coming-soon",
                page_num=0,
                text="Coming Soon!",
                bbox=None,
                reading_order_index=2,
                block_kind="heading",
                block_role="heading",
            ),
            ExtractedBlock(
                block_id="bulletin-footer",
                page_num=0,
                text="http://www.davis.house.gov Powered by Joomla! Generated: 29 January, 2009, 21:18",
                bbox=None,
                reading_order_index=3,
                block_kind="text",
                block_role="key_value",
            ),
        ]

        chunks = chunk_document(document, blocks, target_chars=300, min_chunk_chars=20)
        chunk_text = " ".join(chunk.text for chunk in chunks)

        self.assertEqual(len(chunks), 1)
        self.assertIn("The Online Office of Congressman Danny K. Davis", chunk_text)
        self.assertIn("ENewsletter", chunk_text)
        self.assertIn("Coming Soon", chunk_text)

    def test_generated_pdf_hard_cases_have_processing_level_support_contracts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            source_dir = output_dir / "source-pdfs"
            source_dir.mkdir()
            cases: dict[str, tuple[str, list[list[tuple[str, float]]]]] = {
                "ship": (
                    "Adopt-A-Ship Application Form",
                    [
                        [
                            ("Adopt-A-Ship Application Form", 16.0),
                            (
                                "Ship Name: Liberty Star\n"
                                "Home Port: Baltimore\n"
                                "Sponsor: International Propeller Club\n"
                                "Email: adopt@propellerclubhq.com",
                                11.0,
                            ),
                        ]
                    ],
                ),
                "voter": (
                    "Voter Registration Address Change Form",
                    [
                        [
                            ("Voter Registration Address Change Form", 16.0),
                            (
                                "Voter's Name: Jane Citizen\n"
                                "Old Address: 1 First Street\n"
                                "New Address: 2 Second Street\n"
                                "Social Security: optional last four",
                                11.0,
                            ),
                        ]
                    ],
                ),
                "farm": (
                    "Farm Table Row Summary",
                    [
                        [
                            ("Table 1 County Row Summary", 16.0),
                            ("Table row State Acres Farms", 11.0),
                            ("Colorado State 200 60", 11.0),
                            ("Baca 1120 336", 11.0),
                        ]
                    ],
                ),
                "qip": (
                    "QIP Presentation",
                    [
                        [
                            ("Joint Commission on Health Care", 18.0),
                            (
                                "Terry Smith, Division Director. Presented to the Joint Commission on Health Care.",
                                11.0,
                            ),
                        ],
                        [
                            ("Presentation Outline", 18.0),
                            (
                                "Program Overview\n"
                                "Quality indicators\n"
                                "Reporting schedule\n"
                                "Implementation milestones for health care quality reporting.",
                                11.0,
                            ),
                        ],
                        [
                            ("Mission", 18.0),
                            (
                                "The Mission of the Virginia Quality Improvement Program is to improve care "
                                "and sustain reliable nursing support across facilities.",
                                11.0,
                            ),
                        ],
                    ],
                ),
            }
            processed: dict[str, list[ChunkRecord]] = {}
            for name, (title, pages) in cases.items():
                pdf_path = source_dir / f"{name}.pdf"
                self._create_text_pdf(pdf_path, title=title, pages=pages)
                extraction, document, native_path, document_path = (
                    process_native_pdf_to_json(
                        pdf_path,
                        output_dir / name,
                    )
                )
                self.assertGreater(len(extraction.blocks), 0)
                _document, chunks, _saved_paths = process_saved_document_to_chunks(
                    native_path=native_path,
                    document_path=document_path,
                    output_dir=output_dir / name,
                )
                self.assertGreater(len(chunks), 0)
                processed[name] = chunks

        ship_support = " ".join(
            chunk.text
            for chunk in processed["ship"]
            if chunk.section_role == "form" and chunk.chunk_strategy == "form_rows"
        )
        self.assertIn("Ship Name", ship_support)
        self.assertIn("Home Port", ship_support)
        self.assertIn("International Propeller Club", ship_support)
        self.assertIn("adopt@propellerclubhq.com", ship_support)

        voter_support = " ".join(
            chunk.text
            for chunk in processed["voter"]
            if chunk.section_role == "form" and chunk.chunk_strategy == "form_rows"
        )
        self.assertIn("Old Address", voter_support)
        self.assertIn("New Address", voter_support)
        self.assertIn("Voter's Name", voter_support)
        self.assertIn("Social Security", voter_support)

        farm_table_chunks = [
            chunk
            for chunk in processed["farm"]
            if chunk.chunk_type == "table" and chunk.chunk_strategy == "table_rows"
        ]
        farm_support = " ".join(chunk.text for chunk in farm_table_chunks)
        self.assertIn("Colorado State 200 60", farm_support)
        self.assertIn("Baca 1120 336", farm_support)

        qip_chunks = processed["qip"]
        self.assertTrue(
            any(
                chunk.section_role == "presentation_title"
                and "presentation_title" in chunk.content_hints
                and "Joint Commission on Health Care" in chunk.section_title
                and "Terry Smith" in chunk.text
                for chunk in qip_chunks
            )
        )
        self.assertTrue(
            any(
                chunk.section_role == "presentation_outline"
                and "presentation_outline" in chunk.content_hints
                for chunk in qip_chunks
            )
        )
        self.assertTrue(
            any(
                chunk.section_role == "presentation_mission"
                and "presentation_mission" in chunk.content_hints
                and "Virginia Quality Improvement Program" in chunk.text
                for chunk in qip_chunks
            )
        )

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
        self.assertEqual(
            [block.block_id for block in ordered],
            ["left-1", "left-2", "right-1", "right-2"],
        )

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
            summary_cues=[
                "Court of Appeals",
                "Claimant-Appellant",
                "Opinion and Order",
            ],
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
            summary_cues=[
                "County",
                "Figures taken from the 2002 Census of Agriculture",
            ],
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
            summary_cues=[
                "Indeed.com index of Utah GIS jobs",
                "Utah GIS Jobs on indeed.com",
            ],
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
            summary_cues=[
                "Waste Site Reclassification Form",
                "Description of current waste site condition",
            ],
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
        self.assertEqual(
            correspondence["document_type"], "institutional_correspondence"
        )
        self.assertEqual(
            correspondence["document_purpose"], "institutional_communication"
        )
        self.assertEqual(correspondence["audience"], "institutional_staff")
        self.assertGreaterEqual(correspondence["semantic_confidence"], 0.75)


if __name__ == "__main__":
    unittest.main()
