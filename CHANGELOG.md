# Changelog

## Unreleased

- Added controlled hash vs sentence-transformer vs cross-encoder comparisons with fresh indexes, per-query latency statistics, model activation checks, and a compact tracked snapshot.
- Documented measured quality, latency, model-weight, CPU, and optional-GPU trade-offs in the README.
- Tightened abstention so cell-culture monoclonal-antibody evidence is not presented as clinical prevention or treatment evidence.

## v0.2.0 — 2026-09-28

`v0.2.0` is the first non-beta portfolio release of the project: a local-first Document AI workflow that turns PDFs into structured JSON and grounded, cited answers.

### Highlights

- Added a browser-based local workspace for uploading PDFs, asking questions, inspecting cited chunks, and reviewing extraction-quality signals.
- Integrated `pdf-inspector` as a fail-open assistant for diagnostics, cautious OCR routing, and validated missing-table recovery while keeping PyMuPDF canonical.
- Strengthened document-aware query planning, retrieval contracts, grounded answering, abstention behavior, and processing diagnostics.
- Added a required CI benchmark over three downloaded, SHA-256-verified public PDFs with six ground-truth retrieval cases.
- Added a compact tracked benchmark snapshot and full CI artifact for reproducible evaluation.
- Reworked the README around the real web workspace, architecture, measured results, and automatically generated demo output.

### Validation snapshot

- Maintained evaluation suite: 77/77 retrieval cases and 77/77 answer-faithfulness cases passing, with Recall@5 and MRR of 1.000.
- Required public-PDF CI shard: 6/6 cases passing, with Recall@5, MRR, and evidence-keyword coverage of 1.000.
- CI covers Python 3.10 and 3.13, Ruff, dependency auditing, and the public-PDF benchmark.

### Notes

- The default remains local-first and works without a hosted model; deterministic hash embeddings are used when a cached sentence-transformer model is unavailable.
- OCR, layout interpretation, and retrieval are intentionally heuristic-first. The reported benchmark is a regression baseline, not a claim of universal PDF performance.

## v0.1.0-beta — 2026-09-12

- Established the packaged CLI, public demo path, structured PDF extraction, local indexing, document-aware retrieval, grounded answers, and release checks.
