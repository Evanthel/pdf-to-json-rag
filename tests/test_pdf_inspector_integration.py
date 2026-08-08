"""Focused tests for the fail-open pdf-inspector integration."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pdf_to_json_rag import cli as cli_module
from pdf_to_json_rag import extraction as extraction_module
from pdf_to_json_rag import pdf_inspector_adapter as adapter_module
from pdf_to_json_rag.extraction import ExtractedBlock
from pdf_to_json_rag.pdf_inspector_adapter import PdfInspectorResult


class PdfInspectorAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pdf_path = Path("/tmp/adapter-test.pdf")

    def _raw_result(self, **overrides: object) -> object:
        values: dict[str, object] = {
            "pdf_type": "mixed",
            "confidence": 0.91,
            "processing_time_ms": 17,
            "page_count": 3,
            "pages_needing_ocr": [1, 3],
            "pages_with_tables": [2],
            "pages_with_columns": [2, 3],
            "ocr_reasons_by_page": [
                types.SimpleNamespace(page=1, reasons=["empty_text"]),
                types.SimpleNamespace(page=3, reasons=["encoding_issue"]),
            ],
            "has_encoding_issues": True,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_process_result_normalizes_one_based_pages(self) -> None:
        fake_module = types.SimpleNamespace(process_pdf=lambda _path: self._raw_result())
        with (
            mock.patch.object(adapter_module, "_load_module", return_value=fake_module),
            mock.patch.object(adapter_module, "_installed_version", return_value="0.2.6"),
        ):
            result = adapter_module.inspect_pdf_with_pdf_inspector(
                self.pdf_path,
                expected_page_count=3,
                mode="assist",
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.pages_needing_ocr, [0, 2])
        self.assertEqual(result.pages_with_tables, [1])
        self.assertEqual(result.pages_with_columns, [1, 2])
        self.assertEqual(result.ocr_reasons_by_page[2], ["encoding_issue"])
        self.assertEqual(result.to_summary()["pages_needing_ocr"], [1, 3])

    def test_missing_attributes_are_safe_defaults(self) -> None:
        fake_module = types.SimpleNamespace(
            process_pdf=lambda _path: types.SimpleNamespace(page_count=1)
        )
        with mock.patch.object(adapter_module, "_load_module", return_value=fake_module):
            result = adapter_module.inspect_pdf_with_pdf_inspector(
                self.pdf_path,
                expected_page_count=1,
                mode="shadow",
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.effective_mode, "shadow")
        self.assertEqual(result.pages_needing_ocr, [])
        self.assertIsNone(result.confidence)

    def test_import_call_and_page_count_failures_fall_back(self) -> None:
        with mock.patch.object(adapter_module, "_load_module", side_effect=ImportError("missing")):
            import_failure = adapter_module.inspect_pdf_with_pdf_inspector(
                self.pdf_path,
                expected_page_count=1,
                mode="assist",
            )
        self.assertEqual(import_failure.effective_mode, "off")
        self.assertEqual(import_failure.fallback_reason, "import_failed:ImportError")

        failing_module = types.SimpleNamespace(
            process_pdf=mock.Mock(side_effect=ValueError("bad pdf"))
        )
        with mock.patch.object(adapter_module, "_load_module", return_value=failing_module):
            call_failure = adapter_module.inspect_pdf_with_pdf_inspector(
                self.pdf_path,
                expected_page_count=1,
                mode="assist",
            )
        self.assertEqual(call_failure.fallback_reason, "process_failed:ValueError")

        mismatch_module = types.SimpleNamespace(
            process_pdf=lambda _path: self._raw_result(page_count=2)
        )
        with mock.patch.object(adapter_module, "_load_module", return_value=mismatch_module):
            mismatch = adapter_module.inspect_pdf_with_pdf_inspector(
                self.pdf_path,
                expected_page_count=1,
                mode="assist",
            )
        self.assertEqual(mismatch.fallback_reason, "page_count_mismatch:2:1")

    def test_modes_support_default_shadow_off_and_invalid_fallback(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(adapter_module.resolve_pdf_inspector_mode()[1], "assist")
        self.assertEqual(adapter_module.resolve_pdf_inspector_mode("shadow")[1], "shadow")

        with mock.patch.object(adapter_module, "_load_module") as loader:
            off = adapter_module.inspect_pdf_with_pdf_inspector(
                self.pdf_path,
                expected_page_count=1,
                mode="off",
            )
        loader.assert_not_called()
        self.assertEqual(off.status, "skipped")

        invalid = adapter_module.inspect_pdf_with_pdf_inspector(
            self.pdf_path,
            expected_page_count=1,
            mode="unsafe",
        )
        self.assertEqual(invalid.status, "fallback")
        self.assertEqual(invalid.effective_mode, "off")
        self.assertEqual(invalid.fallback_reason, "invalid_mode:unsafe")

    def test_selected_markdown_preserves_zero_based_pages(self) -> None:
        fake_pages = [
            types.SimpleNamespace(page=2, markdown="page three"),
            types.SimpleNamespace(page=0, markdown="page one"),
        ]
        extractor = mock.Mock(return_value=types.SimpleNamespace(pages=fake_pages))
        with mock.patch.object(
            adapter_module,
            "_load_module",
            return_value=types.SimpleNamespace(extract_pages_markdown=extractor),
        ):
            markdown, error = adapter_module.extract_candidate_page_markdown(
                self.pdf_path,
                [2, 0],
            )
        self.assertIsNone(error)
        self.assertEqual(markdown, {0: "page one", 2: "page three"})
        extractor.assert_called_once_with(str(self.pdf_path), pages=[0, 2])


class PdfInspectorExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_pdf(self, name: str, blocks: list[tuple[float, str]]) -> Path:
        path = self.workspace / name
        document = fitz.open()
        page = document.new_page()
        for y, text in blocks:
            page.insert_text((72, y), text)
        document.save(path)
        document.close()
        return path

    def _inspector(self, **overrides: object) -> PdfInspectorResult:
        values: dict[str, object] = {
            "requested_mode": "assist",
            "effective_mode": "assist",
            "status": "ok",
            "version": "0.2.6",
            "page_count": 1,
            "confidence": 0.9,
        }
        values.update(overrides)
        return PdfInspectorResult(**values)  # type: ignore[arg-type]

    def _extract_with_inspector(
        self,
        pdf_path: Path,
        inspector: PdfInspectorResult,
        *,
        markdown: dict[int, str] | None = None,
    ) -> extraction_module.NativePdfExtraction:
        with (
            mock.patch.object(
                extraction_module,
                "inspect_pdf_with_pdf_inspector",
                return_value=inspector,
            ),
            mock.patch.object(
                extraction_module,
                "extract_pdfplumber_table_blocks",
                return_value=([], {"engine": "pdfplumber", "available": False}),
            ),
            mock.patch.object(
                extraction_module,
                "extract_candidate_page_markdown",
                return_value=(markdown or {}, None),
            ),
        ):
            return extraction_module.extract_native_pdf(pdf_path)

    def test_false_ocr_alarm_preserves_good_pymupdf_text(self) -> None:
        text = (
            "This native paragraph is intentionally long and readable, with enough ordinary "
            "letters and numbers to remain the canonical source for citations and retrieval."
        )
        pdf_path = self._create_pdf("good.pdf", [(72, text)])
        inspector = self._inspector(pages_needing_ocr=[0])
        with mock.patch.object(extraction_module, "extract_page_with_ocr") as ocr:
            extraction = self._extract_with_inspector(pdf_path, inspector)

        ocr.assert_not_called()
        self.assertFalse(extraction.pages[0].needs_ocr)
        self.assertFalse(extraction.pages[0].ocr_used)
        self.assertTrue(extraction.pages[0].extraction_engine_disagreement)
        self.assertIn(text[:60], extraction.pages[0].text)
        self.assertEqual(
            extraction.pdf_inspector["disagreements"][0]["action"],
            "native_text_preserved",
        )

    def test_inspector_ocr_signal_requires_native_suspicion(self) -> None:
        text = "Readable native text with just over forty characters total."
        pdf_path = self._create_pdf("short.pdf", [(72, text)])
        inspector = self._inspector(pages_needing_ocr=[0])
        with mock.patch.object(
            extraction_module,
            "extract_page_with_ocr",
            return_value=[],
        ) as ocr:
            extraction = self._extract_with_inspector(pdf_path, inspector)

        ocr.assert_called_once()
        self.assertTrue(extraction.pages[0].needs_ocr)
        self.assertFalse(extraction.pages[0].ocr_used)

    def test_existing_weak_text_trigger_is_preserved_without_inspector_agreement(self) -> None:
        pdf_path = self._create_pdf("weak.pdf", [(72, "Very short text")])
        inspector = self._inspector(pages_needing_ocr=[])
        with mock.patch.object(
            extraction_module,
            "extract_page_with_ocr",
            return_value=[],
        ) as ocr:
            extraction = self._extract_with_inspector(pdf_path, inspector)

        ocr.assert_called_once()
        self.assertTrue(extraction.pages[0].needs_ocr)
        self.assertTrue(extraction.pages[0].extraction_engine_disagreement)

    def test_damaged_encoding_heuristics_and_missing_tesseract_are_safe(self) -> None:
        reasons = extraction_module._native_text_suspicion_reasons(
            "Readable prefix \ufffd \x00 " + ("normal " * 20)
        )
        self.assertIn("invalid_unicode_marker", reasons)
        with mock.patch.object(extraction_module.shutil, "which", return_value=None):
            self.assertEqual(
                extraction_module.extract_page_with_ocr(Path("missing.pdf"), 0),
                [],
            )

    def test_valid_markdown_table_becomes_deterministic_supplemental_block(self) -> None:
        pdf_path = self._create_pdf(
            "table.pdf",
            [
                (
                    72,
                    "Quarterly revenue and operating margin results are summarized below for "
                    "the finance review and management discussion.",
                ),
                (180, "The conclusion describes the outlook for the next reporting period."),
            ],
        )
        markdown = (
            "| Quarter | Revenue | Operating margin |\n"
            "| --- | --- | --- |\n"
            "| Q1 2026 | 1250000 | 18 percent |"
        )
        inspector = self._inspector(pages_with_tables=[0])
        extraction = self._extract_with_inspector(
            pdf_path,
            inspector,
            markdown={0: markdown},
        )

        tables = [
            block
            for block in extraction.blocks
            if "pdf_inspector_table" in block.structural_flags
        ]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].block_id, "table-p001-pdf-inspector-table-001")
        self.assertEqual(tables[0].page_num, 0)
        self.assertEqual(tables[0].extraction_method, "native")
        self.assertTrue(extraction.pages[0].pdf_inspector_content_used)
        self.assertEqual(extraction.pdf_inspector["tables_added"], 1)
        artifact = extraction_module.native_extraction_to_dict(extraction)
        self.assertIn("pdf_inspector", artifact)
        self.assertTrue(artifact["pages"][0]["pdf_inspector_content_used"])
        self.assertIn("pdf_inspector_signals", artifact["pages"][0])

    def test_invalid_markdown_and_ocr_pages_do_not_add_tables(self) -> None:
        self.assertEqual(
            extraction_module._extract_valid_markdown_tables(
                "| Header | Value |\n| no separator | here |\n| row | value |"
            ),
            [],
        )
        self.assertEqual(
            extraction_module._extract_valid_markdown_tables(
                "| Header | Value |\n| --- | --- |"
            ),
            [],
        )

        pdf_path = self._create_pdf("ocr-table.pdf", [(72, "Short scan label")])
        inspector = self._inspector(
            pages_needing_ocr=[0],
            pages_with_tables=[0],
        )
        with (
            mock.patch.object(extraction_module, "extract_page_with_ocr", return_value=[]),
            mock.patch.object(
                extraction_module,
                "extract_candidate_page_markdown",
                return_value=({}, None),
            ) as markdown_extractor,
            mock.patch.object(
                extraction_module,
                "inspect_pdf_with_pdf_inspector",
                return_value=inspector,
            ),
            mock.patch.object(
                extraction_module,
                "extract_pdfplumber_table_blocks",
                return_value=([], {}),
            ),
        ):
            extraction = extraction_module.extract_native_pdf(pdf_path)
        markdown_extractor.assert_called_once_with(pdf_path.resolve(), [])
        self.assertEqual(extraction.pdf_inspector["tables_added"], 0)

    def test_table_deduplication_and_insertion_preserve_existing_order(self) -> None:
        first = ExtractedBlock("first", 0, "Quarterly revenue", None, 0)
        second = ExtractedBlock("second", 0, "Final outlook", None, 1)
        table = ExtractedBlock(
            "table",
            0,
            "| Quarter | Revenue |\n| --- | --- |\n| Q1 2026 | 1250000 |",
            None,
            0,
            block_kind="table_like",
            block_role="table_like",
        )
        inserted = extraction_module._insert_supplemental_table([first, second], table)
        self.assertEqual([block.block_id for block in inserted], ["first", "table", "second"])
        self.assertTrue(extraction_module._is_duplicate_table(table.text, [table]))

    def test_shadow_mode_records_signals_without_using_markdown(self) -> None:
        pdf_path = self._create_pdf(
            "shadow.pdf",
            [(72, "A sufficiently long native paragraph for shadow mode diagnostics only. " * 2)],
        )
        inspector = self._inspector(
            effective_mode="shadow",
            pages_with_tables=[0],
            pages_with_columns=[0],
        )
        extraction = self._extract_with_inspector(pdf_path, inspector)
        self.assertFalse(extraction.pages[0].pdf_inspector_content_used)
        self.assertIn("pdf_inspector_columns", extraction.pages[0].pdf_inspector_signals)
        self.assertEqual(extraction.pdf_inspector["table_markdown_status"], "shadow")

    def test_processing_diagnostics_exposes_engine_taxonomy_and_summary(self) -> None:
        diagnostics = cli_module._processing_diagnostics(
            {
                "page_count": 1,
                "section_count": 1,
                "structure_confidence": 0.8,
                "layout_confidence": 0.8,
                "extraction_summary": {
                    "native_blocks": 2,
                    "pages_requiring_ocr": 0,
                    "pdf_inspector": {
                        "status": "ok",
                        "has_encoding_issues": True,
                        "disagreements": [{"page_num": 1}],
                    },
                },
            },
            {"chunk_count": 2},
        )
        self.assertIn("encoding_uncertain", diagnostics["taxonomy"])
        self.assertIn("extraction_engine_disagreement", diagnostics["taxonomy"])
        self.assertEqual(diagnostics["status"], "warn")
        self.assertEqual(diagnostics["summary"]["pdf_inspector"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
