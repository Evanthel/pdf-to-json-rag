# Public PDF corpus candidates

This document describes the public, reproducible corpus selected for extraction and RAG evaluation. A required three-document shard is active in CI; the remaining documents are candidates for broader and stress evaluation. The source PDFs are not committed to Git. The tracked manifest records stable source URLs, license evidence, expected byte counts, SHA-256 hashes, page counts, and intended benchmark roles.

The selection is recorded in [`data/eval/public_pdf_corpus_manifest.json`](../data/eval/public_pdf_corpus_manifest.json).

## Download and verify

Download every candidate into the gitignored `pdf/public-corpus/` directory:

```bash
python scripts/download_public_corpus.py --json
```

Download a smaller subset:

```bash
python scripts/download_public_corpus.py \
  --ids psmi-checklist-2025 open-data-canvas-questionnaire-2023 irs-form-w9-2024 \
  --json
```

The downloader rejects any file whose byte count or SHA-256 differs from the reviewed artifact. A changed upstream file must be reviewed visually and legally before its manifest entry is updated.

## Selected coverage

| Candidate | Pages | License | Primary layout value |
| --- | ---: | --- | --- |
| PSMI recommendations and checklist | 16 | CC BY 4.0 | Mixed columns, tables, figures, checklist |
| Pediatric osteomyelitis systematic review | 21 | CC BY 4.0 | Large and multi-page evidence tables |
| Asthma self-knowledge questionnaire study | 17 | CC BY 4.0 | Statistical tables and correlation matrix |
| APOSTEL-AS study protocol | 8 | CC BY 4.0 | Structured protocol and long author metadata |
| TOURISME compendium | 238 | CC BY 4.0 | Long branded report, repeated templates, internal links |
| Open Data Canvas questionnaire | 2 | CC BY 4.0 | Form-like questions, radio grids, dense tables |
| Brazilian dark matter report | 10 | CC BY 4.0 | Technical sections, equations, citations, very long author list |
| IRS Form W-9 (March 2024) | 6 | U.S. federal public domain | Interactive fields, checkboxes, dense instructions |

The candidates were reviewed as rendered pages, not only as extracted text. All eight contain usable native text. Together they cover short, medium, and long documents; mixed reading order; tables; questionnaire structure; technical references; and interactive form fields.

## Initial pipeline baseline

All eight PDFs completed the current `layout-sanity-check` without a technical extraction failure. A manual review of its semantic output found five clearly incorrect document-type classifications:

| Candidate | Expected type | Observed type | Review |
| --- | --- | --- | --- |
| PSMI checklist | Guideline or position paper | Empirical study | Incorrect, high confidence |
| Pediatric osteomyelitis review | Systematic review | Agency report | Incorrect, high confidence |
| Asthma questionnaire study | Research article | Questionnaire | Partially acceptable |
| APOSTEL-AS protocol | Study protocol | Court opinion | Incorrect, high confidence |
| TOURISME compendium | Best-practices compendium | Report | Acceptable, low confidence |
| Open Data Canvas questionnaire | Questionnaire | Court opinion | Incorrect, high confidence |
| Brazilian dark matter report | Community report | Report | Type acceptable; purpose debatable |
| IRS Form W-9 | Government tax form | Legislative amendment | Incorrect, high confidence |

This is the main value of activating a public corpus: the current top-level `all_pass` result demonstrates contract completeness and technical processing, but it does not yet demonstrate semantic correctness. The reviewed expectations are stored in the manifest as the seed for executable ground truth; they should become assertions before the corpus is presented as a passing benchmark.

## Required CI shard

The `Retrieval Benchmarks` workflow runs automatically for every pull request and every push to `main`. It downloads and verifies three documents from this corpus:

- PSMI recommendations and checklist,
- Open Data Canvas questionnaire,
- IRS Form W-9.

The six tracked cases in [`data/eval/public_pdf_ground_truth_cases.json`](../data/eval/public_pdf_ground_truth_cases.json) exercise PDF extraction, JSON document creation, chunking, index construction, retrieval, and evidence-keyword support. The workflow has no missing-assets skip path: a missing download, integrity mismatch, processing failure, absent index, or failed benchmark gate makes the job fail.

The full JSON report and Markdown summary are uploaded by every run as the `public-pdf-retrieval-benchmark` artifact. A small, reviewed result from the latest accepted `main` run is tracked in [`data/eval/public_pdf_benchmark_snapshot.json`](../data/eval/public_pdf_benchmark_snapshot.json) so the core metrics and their exact input hashes remain visible after CI artifacts expire.

Run the same required gate locally:

```bash
python scripts/download_public_corpus.py \
  --ids psmi-checklist-2025 open-data-canvas-questionnaire-2023 irs-form-w9-2024 \
  --output-dir pdf/public-corpus \
  --json

pdf-to-json-rag real-ground-truth-check \
  --corpus-dir pdf/public-corpus \
  --eval-file data/eval/public_pdf_ground_truth_cases.json \
  --modes default-auto \
  --json
```

## License policy

- Europe PMC was used to discover the four medical publications and confirm their CC BY metadata. Their publisher PDFs also embed a CC BY 4.0 license link.
- The three Zenodo records explicitly declare CC BY 4.0.
- The W-9 is an unmodified U.S. federal government work covered by 17 U.S.C. 105. Do not imply IRS endorsement.
- Preserve creator, title, DOI, record URL, and license metadata in benchmark reports and derived fixtures.
- Do not replace a reviewed PDF with another version merely because the title matches.

This is a technical selection record, not legal advice. If a source changes its license or file contents, stop using the affected entry until it has been reviewed again.

## Remaining benchmark work

1. Add a project-owned synthetic scan fixture for OCR testing. The external corpus intentionally does not include an ambiguously licensed historical scan.
2. Convert the reviewed semantic classification expectations in the manifest into executable assertions.
3. Expand ground truth beyond the required six cases to cover multi-page tables, form structure, and stricter answer faithfulness.
4. Keep the 238-page compendium in an optional stress shard so the default CI path stays fast.
