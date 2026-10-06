#!/usr/bin/env python3
"""Build deterministic Kaggle dataset and notebook publishing assets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DATASET_DIR = HERE / "dataset"
NOTEBOOK_DIR = HERE / "notebook"

RUNTIME_SOURCE = REPOSITORY_ROOT / "data/eval/runtime_backend_comparison_snapshot.json"
PUBLIC_PDF_SOURCE = REPOSITORY_ROOT / "data/eval/public_pdf_benchmark_snapshot.json"
LICENSE_SOURCE = REPOSITORY_ROOT / "LICENSE"
COVER_SOURCE = REPOSITORY_ROOT / "docs/images/pdf-to-json-rag-flow.png"

DATASET_ID = "piotrobiegly/pdf-to-json-rag-benchmark-results"
NOTEBOOK_ID = "piotrobiegly/pdf-to-json-rag-retrieval-benchmark"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def backend_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in snapshot["modes"]:
        quality = mode["quality"]
        latency = mode["latency"]
        runtime = mode["runtime"]
        artifact = mode["model_artifact"]
        rows.append(
            {
                "mode": mode["mode"],
                "pipeline": mode["pipeline"],
                "cpu": mode["cpu"],
                "gpu": mode["gpu"],
                "local_model": mode["local_model"],
                "fit": mode["fit"],
                "case_count": quality["case_count"],
                "pass_count": quality["pass_count"],
                "pass_rate": quality["pass_count"] / quality["case_count"],
                "precision_at_5": quality["avg_precision_at_k"],
                "recall_at_5": quality["avg_recall_at_k"],
                "mrr": quality["mrr"],
                "keyword_coverage": quality["avg_keyword_coverage"],
                "warning_case_count": quality["warning_case_count"],
                "avg_query_latency_ms": latency["avg_query_latency_ms"],
                "median_query_latency_ms": latency["median_query_latency_ms"],
                "p95_query_latency_ms": latency["p95_query_latency_ms"],
                "mode_wall_seconds": latency["mode_wall_seconds"],
                "index_build_seconds": runtime["index_build_seconds"],
                "model_weight_mib": artifact["primary_weight_mib"],
                "embedding_backend": runtime["embedding_backend"],
                "embedding_model": runtime["embedding_model"],
            }
        )
    return rows


def public_pdf_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    corpus = snapshot["corpus"]
    results = snapshot["results"]
    runtime = snapshot["runtime"]
    inputs = snapshot["inputs"]
    return [
        {
            "generated_at_utc": snapshot["generated_at_utc"],
            "pdf_count": corpus["pdf_count"],
            "case_count": corpus["case_count"],
            "chunk_count": corpus["chunk_count"],
            "k": inputs["k"],
            "pass_count": results["pass_count"],
            "fail_count": results["fail_count"],
            "pass_rate": results["pass_count"] / corpus["case_count"],
            "recall_at_5": results["avg_recall_at_k"],
            "mrr": results["mrr"],
            "answer_keyword_coverage": results["avg_answer_keyword_coverage"],
            "evidence_keyword_coverage": results["avg_evidence_keyword_coverage"],
            "quality_gate_passed": results["quality_gate_passed"],
            "embedding_backend": runtime["embedding_backend"],
            "embedding_model": runtime["embedding_model"],
            "llm_used_case_count": runtime["llm_used_case_count"],
        }
    ]


BACKEND_FIELDS = [
    "mode",
    "pipeline",
    "cpu",
    "gpu",
    "local_model",
    "fit",
    "case_count",
    "pass_count",
    "pass_rate",
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "keyword_coverage",
    "warning_case_count",
    "avg_query_latency_ms",
    "median_query_latency_ms",
    "p95_query_latency_ms",
    "mode_wall_seconds",
    "index_build_seconds",
    "model_weight_mib",
    "embedding_backend",
    "embedding_model",
]

PUBLIC_PDF_FIELDS = [
    "generated_at_utc",
    "pdf_count",
    "case_count",
    "chunk_count",
    "k",
    "pass_count",
    "fail_count",
    "pass_rate",
    "recall_at_5",
    "mrr",
    "answer_keyword_coverage",
    "evidence_keyword_coverage",
    "quality_gate_passed",
    "embedding_backend",
    "embedding_model",
    "llm_used_case_count",
]


def markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def build_notebook() -> dict[str, Any]:
    cells = [
        markdown_cell(
            """
# PDF-to-JSON RAG: retrieval quality vs runtime cost

This executable case study examines two release snapshots from a local-first Document AI project: a controlled comparison of three retrieval/reranking backends and a required quality gate over public PDFs.

The goal is not to claim that a larger model is automatically better. It is to make the engineering trade-off visible: **retrieval quality, query latency, index-build time, model size, and offline operation**.

**Project links:** [GitHub repository](https://github.com/Evanthel/pdf-to-json-rag) · [Interactive Hugging Face demo](https://huggingface.co/spaces/Evanthel/pdf-to-json-rag-demo)
"""
        ),
        markdown_cell(
            """
## System under evaluation

```text
PDF → structured document JSON → section-aware chunks
    → local vector index → retrieval/reranking
    → grounded answer + page/chunk citations
```

The full application handles native text, OCR fallback, document routing, extraction diagnostics, persistent ChromaDB indexes, and answer abstention. This notebook focuses narrowly on the evidence needed to choose a default retrieval backend.
"""
        ),
        code_cell(
            """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import Markdown, display
except ImportError:  # Keeps the repository-side smoke test dependency-light.
    def Markdown(value):
        return value

    def display(value):
        print(value)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#3559E0", "#34A77B", "#F08A4B"]

def locate_dataset() -> Path:
    local = Path.cwd() / "deploy" / "kaggle" / "dataset"
    if (local / "runtime_backend_comparison.json").exists():
        return local

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        matches = list(kaggle_root.glob("**/runtime_backend_comparison.json"))
        if matches:
            return matches[0].parent

    raise FileNotFoundError(
        "Dataset files not found. Attach 'PDF-to-JSON RAG Benchmark Results' "
        "on Kaggle or run this notebook from the project repository root."
    )

DATA_DIR = locate_dataset()
print(f"Reading benchmark data from: {DATA_DIR}")
"""
        ),
        code_cell(
            """
runtime = json.loads((DATA_DIR / "runtime_backend_comparison.json").read_text())
public_pdf = json.loads((DATA_DIR / "public_pdf_benchmark.json").read_text())
backends = pd.read_csv(DATA_DIR / "backend_comparison.csv")
public_summary = pd.read_csv(DATA_DIR / "public_pdf_summary.csv")

assert runtime["source"]["all_pass"] is True
assert public_pdf["results"]["quality_gate_passed"] is True
assert len(backends) == 3

display(Markdown(
    f"**Loaded:** {runtime['source']['case_count']} controlled evaluation cases, "
    f"{runtime['source']['comparison_chunk_count']:,} indexed chunks, and "
    f"{public_pdf['corpus']['pdf_count']} public PDFs."
))
"""
        ),
        markdown_cell(
            """
## 1. Backend comparison

All three modes were evaluated over the same 77 cases and 2,071 chunks at `k=5`.

- **Baseline:** deterministic hash-384 embeddings plus lightweight reranking; no model weights.
- **Sentence transformer:** `all-MiniLM-L6-v2` embeddings plus lightweight reranking.
- **Cross-encoder:** hash retrieval followed by `ms-marco-MiniLM-L-6-v2` reranking.

Latency is a single-process measurement on Apple Silicon and includes first-use model loading. Treat the absolute milliseconds as host-specific and the relative difference as directional.
"""
        ),
        code_cell(
            """
columns = [
    "mode", "case_count", "pass_rate", "precision_at_5", "recall_at_5", "mrr",
    "avg_query_latency_ms", "p95_query_latency_ms", "index_build_seconds",
    "model_weight_mib", "gpu",
]
comparison = backends[columns].copy()
comparison["pass_rate"] = comparison["pass_rate"].map(lambda value: f"{value:.1%}")
display(comparison.round(3))
"""
        ),
        code_cell(
            """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

quality = backends.set_index("mode")[["recall_at_5", "mrr"]]
quality.plot(kind="bar", ax=axes[0], color=COLORS[:2], width=0.72)
axes[0].set_title("Retrieval quality")
axes[0].set_ylabel("score")
axes[0].set_xlabel("")
axes[0].set_ylim(0.90, 1.01)
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(["Recall@5", "MRR"], frameon=False, loc="lower right")

latency = backends.set_index("mode")[["avg_query_latency_ms", "p95_query_latency_ms"]]
latency.plot(kind="bar", ax=axes[1], color=[COLORS[0], COLORS[2]], width=0.72)
axes[1].set_title("Query latency")
axes[1].set_ylabel("milliseconds")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(["Average", "p95"], frameon=False)

fig.suptitle("Same test set, different runtime cost", fontsize=15, fontweight="bold")
fig.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
baseline_latency = backends.loc[
    backends["mode"] == "baseline", "avg_query_latency_ms"
].iloc[0]
tradeoffs = backends[[
    "mode", "avg_query_latency_ms", "index_build_seconds", "model_weight_mib", "fit"
]].copy()
tradeoffs["query_latency_vs_baseline"] = (
    tradeoffs["avg_query_latency_ms"] / baseline_latency
).round(2)
display(tradeoffs.round(2))

same_quality = (
    backends[["recall_at_5", "mrr", "keyword_coverage"]].nunique() == 1
).all()
cross_multiplier = tradeoffs.loc[
    tradeoffs["mode"] == "cross-encoder", "query_latency_vs_baseline"
].iloc[0]

display(Markdown(
    f"**Observed decision:** quality was {'identical' if same_quality else 'different'} "
    f"on this test set, while the cross-encoder averaged **{cross_multiplier:.2f}×** "
    "the baseline query latency. The snapshot therefore keeps `auto` as the default, "
    "prefers sentence-transformer retrieval when its model is cached, and retains the "
    "hash backend as the dependency-light offline fallback."
))
"""
        ),
        markdown_cell(
            """
## 2. Required public-PDF quality gate

The second snapshot comes from a required GitHub Actions shard. It processes three reviewed public PDFs, builds a fresh index, and evaluates six grounded retrieval cases. The original PDFs are deliberately **not redistributed** in this Kaggle dataset; the source repository records their URLs, licenses, and SHA-256 checksums.

No LLM was used in this run, so the result isolates deterministic extraction, retrieval, and evidence-grounding behavior.
"""
        ),
        code_cell(
            """
summary_columns = [
    "pdf_count", "case_count", "chunk_count", "pass_rate", "recall_at_5", "mrr",
    "answer_keyword_coverage", "evidence_keyword_coverage", "embedding_backend",
    "llm_used_case_count",
]
display(public_summary[summary_columns].round(3))

observed = {
    "Pass rate": float(public_summary.loc[0, "pass_rate"]),
    "MRR": float(public_summary.loc[0, "mrr"]),
    "Evidence coverage": float(public_summary.loc[0, "evidence_keyword_coverage"]),
}
required = {
    "Pass rate": public_pdf["quality_gate"]["min_pass_rate"],
    "MRR": public_pdf["quality_gate"]["min_mrr"],
    "Evidence coverage": public_pdf["quality_gate"]["min_evidence_keyword_coverage"],
}

gate = pd.DataFrame({"Observed": observed, "Required": required})
ax = gate.plot(kind="bar", figsize=(9, 4.5), color=[COLORS[1], "#AAB2C8"], width=0.72)
ax.set_title("Public-PDF benchmark: observed result vs release gate")
ax.set_ylabel("score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.tick_params(axis="x", rotation=0)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.show()
"""
        ),
        markdown_cell(
            """
## What the results do — and do not — show

**Supported by these snapshots**

- The deterministic fallback is a credible default for this portfolio-scale corpus.
- Adding a sentence-transformer improved semantics available to the system without changing measured quality on this particular set, but it added model download and index-build cost.
- Cross-encoder reranking added the highest latency and did not improve these aggregate metrics, so it remains opt-in.
- The release gate is reproducible in CI and fails when processing or retrieval quality drops below explicit thresholds.

**Limitations**

- 77 controlled cases and six public-PDF cases are useful regression evidence, not a universal ranking benchmark.
- Exact latency depends on hardware, process warm-up, model cache state, and corpus size.
- Aggregate retrieval metrics can hide per-document errors; the main project also stores case-level diagnostics.
- The snapshot does not evaluate generated prose from a hosted LLM.
"""
        ),
        markdown_cell(
            """
## Reproduce the full benchmark

The compact files attached to this notebook are immutable release snapshots. To rerun extraction and retrieval from source:

```bash
git clone https://github.com/Evanthel/pdf-to-json-rag.git
cd pdf-to-json-rag
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[semantic]'
pdf-to-json-rag compare-runtime-modes \
  --modes baseline,sentence-transformers,cross-encoder \
  --all-cases --json
```

The repository documents the public corpus, integrity checks, CI gates, and security limits. For the product flow itself, open the [Hugging Face Space](https://huggingface.co/spaces/Evanthel/pdf-to-json-rag-demo) and use a public, non-confidential PDF.
"""
        ),
        markdown_cell(
            """
## Conclusion

The result is intentionally pragmatic: **a more complex retrieval backend must earn its runtime and operational cost on measured quality**. For this release, it did not. The project therefore keeps a deterministic offline path, makes semantic retrieval available when cached, and treats the cross-encoder as an explicit experiment rather than an automatic upgrade.
"""
        ),
    ]

    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"cell-{index:02d}"

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def dataset_readme(runtime: dict[str, Any], public_pdf: dict[str, Any]) -> str:
    return f"""# PDF-to-JSON RAG Benchmark Results

Compact, machine-readable release snapshots from the [PDF-to-JSON RAG](https://github.com/Evanthel/pdf-to-json-rag) local-first Document AI project.

## Contents

- `runtime_backend_comparison.json`: full snapshot comparing baseline, sentence-transformer, and cross-encoder modes.
- `backend_comparison.csv`: flat table derived from the runtime snapshot.
- `public_pdf_benchmark.json`: compact result from the required public-PDF CI gate.
- `public_pdf_summary.csv`: flat one-row summary of that quality gate.

The runtime comparison covers **{runtime['source']['case_count']} cases** and **{runtime['source']['comparison_chunk_count']:,} chunks**. The public-PDF snapshot covers **{public_pdf['corpus']['pdf_count']} documents**, **{public_pdf['corpus']['case_count']} cases**, and **{public_pdf['corpus']['chunk_count']} chunks**.

## Provenance and reproducibility

The JSON files are copied without semantic changes from tracked release snapshots in the GitHub repository. The CSV files are deterministic projections generated by `deploy/kaggle/build_assets.py`.

Source commands, benchmark environment, CI run URL, input hashes, runtime selection, and quality gates remain embedded in the JSON files. Use the source repository to rerun the full pipeline.

## Licensing and source PDFs

The project and these generated benchmark artifacts are distributed under the MIT License; see `LICENSE.txt`.

No source PDFs, extracted document text, model weights, credentials, or user uploads are included. The public benchmark downloads reviewed documents at runtime from their original publishers and verifies SHA-256 checksums. Original document licenses and attribution remain with their respective publishers and are documented in the source repository.

## Limitations

These results are regression and engineering evidence for one project version, not a universal leaderboard. Latency is host-specific and includes first-use model loading. The public-PDF snapshot used deterministic retrieval without an LLM.
"""


def expected_text_files() -> dict[Path, str]:
    runtime = load_json(RUNTIME_SOURCE)
    public_pdf = load_json(PUBLIC_PDF_SOURCE)

    dataset_metadata = {
        "title": "PDF-to-JSON RAG Benchmark Results",
        "subtitle": "Retrieval quality and runtime snapshots for a local-first Document AI pipeline",
        "description": (
            "Compact release snapshots comparing three retrieval/reranking modes and a "
            "required public-PDF quality gate. Includes JSON provenance and analysis-ready "
            "CSV tables; excludes source PDFs, extracted text, model weights, and user data."
        ),
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "keywords": [
            "artificial intelligence",
        ],
        "resources": [
            {
                "path": "backend_comparison.csv",
                "description": "Flat comparison of quality, latency, model size, and hardware requirements.",
            },
            {
                "path": "runtime_backend_comparison.json",
                "description": "Full backend comparison snapshot with environment and decision metadata.",
            },
            {
                "path": "public_pdf_summary.csv",
                "description": "Analysis-ready summary of the required public-PDF CI quality gate.",
            },
            {
                "path": "public_pdf_benchmark.json",
                "description": "Full public-PDF benchmark snapshot with input hashes and gate thresholds.",
            },
        ],
    }

    kernel_metadata = {
        "id": NOTEBOOK_ID,
        "title": "PDF-to-JSON RAG Retrieval Benchmark",
        "code_file": "pdf_to_json_rag_retrieval_benchmark.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    return {
        DATASET_DIR / "runtime_backend_comparison.json": json_text(runtime),
        DATASET_DIR / "public_pdf_benchmark.json": json_text(public_pdf),
        DATASET_DIR / "backend_comparison.csv": csv_text(
            BACKEND_FIELDS, backend_rows(runtime)
        ),
        DATASET_DIR / "public_pdf_summary.csv": csv_text(
            PUBLIC_PDF_FIELDS, public_pdf_rows(public_pdf)
        ),
        DATASET_DIR / "dataset-metadata.json": json_text(dataset_metadata),
        DATASET_DIR / "README.md": dataset_readme(runtime, public_pdf),
        DATASET_DIR / "LICENSE.txt": LICENSE_SOURCE.read_text(encoding="utf-8"),
        NOTEBOOK_DIR / "kernel-metadata.json": json_text(kernel_metadata),
        NOTEBOOK_DIR / "pdf_to_json_rag_retrieval_benchmark.ipynb": json_text(
            build_notebook()
        ),
    }


def write_assets(files: dict[Path, str]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for path, contents in files.items():
        path.write_text(contents, encoding="utf-8")
    shutil.copyfile(COVER_SOURCE, DATASET_DIR / "dataset-cover-image.png")


def check_assets(files: dict[Path, str]) -> list[str]:
    problems: list[str] = []
    for path, expected in files.items():
        if not path.exists():
            problems.append(f"missing: {path.relative_to(REPOSITORY_ROOT)}")
        elif path.read_text(encoding="utf-8") != expected:
            problems.append(f"stale: {path.relative_to(REPOSITORY_ROOT)}")

    cover = DATASET_DIR / "dataset-cover-image.png"
    if not cover.exists():
        problems.append(f"missing: {cover.relative_to(REPOSITORY_ROOT)}")
    elif cover.read_bytes() != COVER_SOURCE.read_bytes():
        problems.append(f"stale: {cover.relative_to(REPOSITORY_ROOT)}")
    return problems


def execute_notebook_code() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    warnings.filterwarnings(
        "ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown"
    )
    namespace: dict[str, Any] = {"__name__": "__kaggle_notebook_check__"}
    notebook = build_notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if generated files are stale")
    mode.add_argument(
        "--verify",
        action="store_true",
        help="check generated files and execute all notebook code cells locally",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = expected_text_files()

    if not args.check and not args.verify:
        write_assets(files)
        print("Kaggle assets generated.")
        return 0

    problems = check_assets(files)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print("Run deploy/kaggle/build_assets.py to refresh the package.", file=sys.stderr)
        return 1

    if args.verify:
        execute_notebook_code()
        print("Kaggle assets are current and all notebook code cells executed successfully.")
    else:
        print("Kaggle assets are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
