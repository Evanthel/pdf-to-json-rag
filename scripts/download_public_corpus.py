#!/usr/bin/env python3
"""Download and integrity-check the public PDF corpus candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "public_pdf_corpus_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pdf" / "public-corpus"
USER_AGENT = "pdf-to-json-rag-public-corpus/0.1 (+https://github.com/Evanthel/pdf-to-json-rag)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"Manifest has no documents: {path}")
    return payload


def _validated_download_target(
    document: dict[str, object],
    output_dir: Path,
) -> tuple[str, str, Path]:
    document_id = str(document["id"]).strip()
    filename = str(document["filename"])
    if not document_id:
        raise ValueError("Manifest document ID must not be empty")
    if (
        not filename
        or filename != filename.strip()
        or len(filename) > 255
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or Path(filename).suffix.lower() != ".pdf"
    ):
        raise ValueError(f"Invalid manifest filename for {document_id}")

    download_url = str(document["download_url"]).strip()
    parsed_url = urlparse(download_url)
    if (
        parsed_url.scheme.lower() != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError(f"Invalid HTTPS download URL for {document_id}")

    output_root = output_dir.expanduser().resolve()
    target = (output_root / filename).resolve()
    if target.parent != output_root or not target.is_relative_to(output_root):
        raise ValueError(f"Download target escapes output directory: {filename}")
    return document_id, download_url, target


def download_document(document: dict[str, object], output_dir: Path, overwrite: bool) -> dict[str, object]:
    document_id, download_url, target = _validated_download_target(document, output_dir)
    expected_sha256 = str(document["sha256"])
    expected_bytes = int(document["bytes"])

    if target.exists() and not overwrite:
        actual_sha256 = sha256_file(target)
        if actual_sha256 == expected_sha256 and target.stat().st_size == expected_bytes:
            return {
                "id": document_id,
                "status": "verified_existing",
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": actual_sha256,
            }
        raise ValueError(
            f"Existing file failed integrity validation: {target}. "
            "Use --overwrite to replace it."
        )

    temporary = target.with_suffix(target.suffix + ".part")
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required to download the public corpus")
    try:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--retry-all-errors",
                "--max-time",
                "120",
                "--user-agent",
                USER_AGENT,
                "--output",
                str(temporary),
                "--url",
                download_url,
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {document_id}: {exc}") from exc

    byte_count = temporary.stat().st_size
    actual_sha256 = sha256_file(temporary)
    if byte_count != expected_bytes or actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Integrity mismatch for {document_id}: "
            f"expected {expected_bytes} bytes/{expected_sha256}, "
            f"received {byte_count} bytes/{actual_sha256}"
        )

    temporary.replace(target)
    return {
        "id": document_id,
        "status": "downloaded",
        "path": str(target),
        "bytes": byte_count,
        "sha256": actual_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ids", nargs="*", help="Download only these document IDs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest.expanduser().resolve())
    documents = manifest["documents"]
    selected_ids = set(args.ids or [])
    selected = [item for item in documents if not selected_ids or item.get("id") in selected_ids]
    available_ids = {str(item.get("id")) for item in documents}
    missing_ids = sorted(selected_ids - available_ids)
    if missing_ids:
        raise ValueError(f"Unknown document IDs: {', '.join(missing_ids)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [download_document(item, output_dir, args.overwrite) for item in selected]
    payload = {
        "ok": True,
        "manifest": str(args.manifest.expanduser().resolve()),
        "output_dir": str(output_dir),
        "document_count": len(results),
        "results": results,
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(f"{item['status']}: {item['id']} -> {item['path']}")
        print(f"Verified {len(results)} public corpus PDFs in {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
