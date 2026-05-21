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

        index_dir = self.data_dir / "index" / "workflow_smoke"
        answer = self._run(
            "answer-query",
            "--query",
            "What does this file cover?",
            "--index-dir",
            str(index_dir),
            "--format",
            "json",
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
        support_trace = answer_payload["result"]["answer_trace"]["support_trace"]
        self.assertGreaterEqual(len(support_trace), 1)
        self.assertTrue(support_trace[0]["section_summaries"])

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


if __name__ == "__main__":
    unittest.main()
