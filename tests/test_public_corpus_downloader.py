from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_public_corpus.py"
SPEC = importlib.util.spec_from_file_location("download_public_corpus", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup guard
    raise RuntimeError(f"Could not load downloader script: {SCRIPT_PATH}")
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


def manifest_document(*, filename: str = "sample.pdf", url: str = "https://example.org/sample.pdf") -> dict[str, object]:
    content = b"%PDF-1.4\npublic corpus fixture\n"
    return {
        "id": "sample-document",
        "filename": filename,
        "download_url": url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


class PublicCorpusDownloaderTests(unittest.TestCase):
    def test_rejects_filenames_outside_output_directory(self) -> None:
        invalid_names = (
            "../escape.pdf",
            "/tmp/escape.pdf",
            "nested/escape.pdf",
            r"C:\\temp\\escape.pdf",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for filename in invalid_names:
                with self.subTest(filename=filename):
                    with self.assertRaisesRegex(ValueError, "Invalid manifest filename"):
                        DOWNLOADER.download_document(
                            manifest_document(filename=filename),
                            output_dir,
                            overwrite=False,
                        )

    def test_rejects_non_https_and_credentialed_urls(self) -> None:
        invalid_urls = (
            "http://example.org/sample.pdf",
            "file:///tmp/sample.pdf",
            "https://user:secret@example.org/sample.pdf",
            "--config=/tmp/curlrc",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for url in invalid_urls:
                with self.subTest(url=url):
                    with self.assertRaisesRegex(ValueError, "Invalid HTTPS download URL"):
                        DOWNLOADER.download_document(
                            manifest_document(url=url),
                            output_dir,
                            overwrite=False,
                        )

    def test_curl_is_restricted_to_https_and_receives_url_as_a_value(self) -> None:
        content = b"%PDF-1.4\npublic corpus fixture\n"

        def fake_run(args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
            self.assertTrue(check)
            output_path = Path(args[args.index("--output") + 1])
            output_path.write_bytes(content)
            return subprocess.CompletedProcess(args=args, returncode=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(DOWNLOADER.shutil, "which", return_value="/usr/bin/curl"),
                patch.object(DOWNLOADER.subprocess, "run", side_effect=fake_run) as run,
            ):
                result = DOWNLOADER.download_document(
                    manifest_document(),
                    Path(temp_dir),
                    overwrite=False,
                )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--proto") + 1], "=https")
        self.assertEqual(command[command.index("--proto-redir") + 1], "=https")
        self.assertEqual(command[command.index("--url") + 1], "https://example.org/sample.pdf")
        self.assertEqual(result["status"], "downloaded")


if __name__ == "__main__":
    unittest.main()
