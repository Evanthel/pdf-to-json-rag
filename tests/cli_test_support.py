"""Shared workspace fixture for packaged CLI tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pdf_to_json_rag import cli as cli_module  # noqa: E402


class CliPublicSurfaceTestBase(unittest.TestCase):
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

    def _create_text_pdf(
        self,
        path: Path,
        *,
        title: str,
        pages: list[list[tuple[str, float]]],
    ) -> None:
        doc = fitz.open()
        doc.set_metadata({"title": title})
        try:
            for page_lines in pages:
                page = doc.new_page()
                y = 72.0
                for text, font_size in page_lines:
                    page.insert_text((72, y), text, fontsize=font_size)
                    y += font_size + 24.0
            doc.save(path)
        finally:
            doc.close()

    def _assert_public_workflow_contract(
        self, result: dict[str, object], *, smoke: bool = False
    ) -> None:
        expected_keys = (
            cli_module.PUBLIC_COMPACT_SMOKE_KEYS
            if smoke
            else cli_module.PUBLIC_COMPACT_WORKFLOW_KEYS
        )
        self.assertEqual(set(result), set(expected_keys))
        self.assertEqual(
            set(result["document"]), set(cli_module.PUBLIC_COMPACT_DOCUMENT_KEYS)
        )
        self.assertEqual(
            set(result["index"]), set(cli_module.PUBLIC_COMPACT_INDEX_KEYS)
        )
        self.assertEqual(
            set(result["answer"]), set(cli_module.PUBLIC_COMPACT_ANSWER_KEYS)
        )
        self.assertEqual(
            set(result["quality_profile_summary"]),
            {
                "available",
                "overall_status",
                "statuses",
                "reasons",
                "recommended_next_action",
            },
        )
        self.assertEqual(
            set(result["processing_diagnostics"]),
            {
                "status",
                "taxonomy",
                "technical_processed",
                "structurally_reliable",
                "recommended_next_action",
                "summary",
            },
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
        self.assertIsInstance(result["structure_support"], dict)
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

    def _run(
        self, *args: str, expect_ok: bool = True
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            [sys.executable, "-m", "pdf_to_json_rag", *args],
            cwd=REPO_ROOT,
            env=self.base_env,
            capture_output=True,
            text=True,
        )
        if expect_ok and process.returncode != 0:
            self.fail(
                f"Command failed: {' '.join(args)}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )
        return process
