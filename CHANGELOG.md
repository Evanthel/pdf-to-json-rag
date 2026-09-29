# Changelog

## Unreleased

## v0.3.0 — 2026-09-29

`v0.3.0` focuses on measurable quality, reproducible builds, and security hardening while preserving the local-first Document AI workflow introduced in `v0.2.0`.

### Highlights

- Streamlined the public documentation around the README, focused technical references, and a concise roadmap; removed the internal sprint log and long-form project plan.
- Added a strict incremental mypy gate for eight small, stable core modules, with a declared `typecheck` dependency group for reproducible local and CI runs.
- Added subprocess-aware branch coverage to Python 3.13 CI, with a measured 56.4% Ubuntu baseline, a 56% non-regression gate, and an uploaded XML report.
- Added controlled hash vs sentence-transformer vs cross-encoder comparisons with fresh indexes, per-query latency statistics, model activation checks, and a compact tracked snapshot.
- Documented measured quality, latency, model-weight, CPU, and optional-GPU trade-offs in the README.
- Tightened abstention so cell-culture monoclonal-antibody evidence is not presented as clinical prevention or treatment evidence.
- Split the largest CLI, answering, evaluation, and CLI-test modules into focused components without changing the public command surface.

### Security and reproducibility

- Added web-upload limits of 100 MiB and 500 pages, plus a 50-million-pixel preflight limit per rendered page, to bound PDF processing resources.
- Removed local command `stderr` content from public JSON payloads so credentials or wrapper diagnostics cannot be copied into reports.
- Added a committed `uv.lock`; CI now installs frozen dependency sets and audits them with narrowly documented ChromaDB exceptions.
- Hardened public-corpus downloads with HTTPS validation, redirects disabled, size limits, retries, and SHA-256 verification before benchmark processing.
- Pinned GitHub Actions by commit SHA, fixed runner images to Ubuntu 24.04, minimized workflow permissions, and kept the public-PDF benchmark mandatory on `main`.
- Made sentence-transformer dependencies optional, upgraded `pdf-inspector` to 0.2.7, and pinned ONNX Runtime below the hosted-runner `SIGILL` regression.

### Migration notes

- Install learned semantic retrieval explicitly with `pip install '.[semantic]'`; the base install continues to provide deterministic hash retrieval without model weights.
- Runtime JSON payloads now expose `stderr_char_count` instead of `stderr_preview`. Consumers should use the count for diagnostics and must not expect command stderr content.
- The web workspace rejects PDFs that exceed the upload, page-count, or per-page render limits above. The CLI processing path remains available for trusted local files that require different operating limits.

### Validation snapshot

- The full local suite passes 117/117 tests at 58.3% subprocess-aware branch coverage, above the 56% non-regression gate.
- Local release validation passes the installed-wheel README flow, package checks, and all 25 maintained regression shards.
- CI covers Python 3.10 and 3.13, Ruff, strict incremental mypy, dependency auditing, and the required public-PDF retrieval benchmark.
- The reviewed ChromaDB exceptions apply only to server-side authorization and collection-update paths that the embedded `PersistentClient` architecture does not expose; see [SECURITY.md](./SECURITY.md#accepted-chromadb-advisories).

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
