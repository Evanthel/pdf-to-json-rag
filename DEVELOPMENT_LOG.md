# DEVELOPMENT_LOG

## Purpose

This file records the project at a practical level without mirroring every code edit. It is intended as a concise engineering log rather than a public changelog.

Internal development iterations in this file use `v1.x` labels. Public releases follow semantic versioning starting at `0.1.0-beta`.

## Current State

Current implementation level: `v0.2.6`
Current public version: `0.1.0-beta`

The project now behaves as a local-first, domain-agnostic `PDF -> JSON -> retrieval -> grounded answer` pipeline with explicit document-intelligence behavior on top of chunk retrieval.

Public release-path validation passing:

- `python -m unittest tests.test_cli_public_surface`
- `package-check`
- `release-check` for the packaged/public surface

Repo-local regression shards currently passing:

- `query_planning_core`
- `answer_modes_core`
- `document_family_core`
- `document_facets_core`
- `inventory_coverage_core`
- `relationship_core`

Broad benchmark note after `v0.2.6`:

- the targeted maintainer shard set is green again
- the broad 67-case benchmark is now recovered on the `v0.2.x` section-aware architecture
- current broad benchmark scope:
  - `Cases`: `67`
  - `Indexed sample documents`: `19`
- latest full rerun on the current `v0.2.6` code path:
  - `precision@5`: `0.533`
  - `recall@5`: `1.0`
  - `MRR`: `1.0`
  - `avg_keyword_coverage`: `1.0`
  - `negative_success_rate`: `1.0`
  - `warning_case_count`: `0`
- the next internal hardening target is no longer wide retrieval parity; it is cleaner document-level support tracing and faithfulness semantics

## Delivered Architecture

### Document Processing

- Native extraction with `PyMuPDF`
- OCR fallback with `pytesseract`
- Document-level JSON artifacts in `data/documents/`
- Chunk-level JSON artifacts in `data/chunks/`
- Reading-order normalization, section heuristics, noise filtering, OCR provenance
- Extraction-time metadata including:
  - `summary_cues`
  - `discovery_terms`
  - document facets
  - document families
  - inventory summaries
  - coverage summaries

### Retrieval and Answering

- Local indexing with `ChromaDB`
- Intent-aware retrieval with lightweight reranking
- Adjacent-chunk expansion
- Source-aware locking
- Cross-document source matching
- Query planning that separates:
  - evidence lookup
  - document discovery
  - document-facet questions
  - cross-document comparison
- Answer modes for:
  - grounded evidence answers
  - document overview
  - document routing
  - source listing
  - source justification
  - cross-document comparison
- Lightweight answer contracts for document-level and cross-document answer paths
- Relationship reasoning for overlap, complement, and divergence
- Tool-facing CLI inspection paths for listing documents, inspecting a document, planning a query, and exporting JSON answers

### Evaluation

- Hand-built multi-document benchmark in `data/eval/mvp_eval_cases.json`
- Sampled faithfulness audit in `data/eval/faithfulness_audit_cases.json`
- Per-case debug output with retrieval snapshots, answer traces, and evidence snippets
- Slice-level reporting for structure family, source family, OCR, discovery, and architecture-oriented paths
- Small deterministic regression shards for high-risk paths

## Milestone Summary

### MVP to v1.5

- Built the local-first core: extraction, chunking, indexing, retrieval, grounded answering, and the first evaluation loop
- Added OCR fallback, stronger chunking, and better retrieval behavior
- Introduced the first multi-document and OCR-derived benchmark coverage

### v1.6 to v1.13

- Broadened the benchmark across review papers, scanned material, manuals, questionnaires, and checklist-style appendices
- Hardened the structured-form path
- Added richer evaluation slices, deterministic regressions, and sampled faithfulness checks
- Added cross-document source listing and comparison

### v1.14 to v1.17

- Shifted from single-document evidence answers toward domain-agnostic document discovery
- Added non-medical, public-safe source families
- Introduced document overview, routing, and source justification
- Pushed more routing behavior into extraction-time metadata instead of source-specific runtime heuristics

### v1.18 to v1.20

- Added document facets as a reusable metadata layer
- Added document inventory and query planning
- Added explicit answer modes for discovery vs evidence behavior
- Made comparison/routing answers more document-aware instead of chunk-accidental
- Added architecture-focused evaluation slices and regressions

### v1.21

- Added a shared document-family layer for books, guidance notes, model reports, manuals, forms, and clinical references
- Normalized answer contracts for overview, routing, source justification, comparison, and evidence-style paths
- Added document-level relationship signals for overlap, complement, and divergence
- Added `document_family_core` plus answer-contract-oriented slice checks

### v1.22 to v1.26

- Consolidated document semantics into a shared metadata interpretation layer for facets, coverage, inventory summaries, and document families
- Strengthened inventory-first routing with coverage-aware and rarity-aware shortlist scoring
- Tightened document-level and cross-document answer contracts around reusable answer modes instead of looser answer-time heuristics
- Added dedicated regression coverage for inventory summaries and relationship reasoning
- Added CLI inspection and planning paths that move the repo closer to a publishable local tool instead of a benchmark-only codebase

### v1.27

- Added package-first project metadata and module entry points for a cleaner install/run path
- Unified JSON output contracts across listing, inspection, planning, retrieval, answering, and evaluation commands
- Added a single end-to-end `run-workflow` path for local smoke usage
- Added public-safe `examples/` assets so the user-facing flow is separated from ignored local benchmark PDFs
- Re-validated the planning and answer-mode regressions after the product-surface pass

### v1.28

- Added a packaged `smoke-check` path for first-run validation of `extract -> chunk -> index -> plan -> answer`
- Tightened CLI error contracts so common failures return stable human-readable and JSON diagnostics
- Added public-safe trimmed example JSON outputs for inspect / plan / answer command shapes
- Shifted the docs further toward a first public tool flow instead of the internal benchmark harness

### v1.29-v1.33

- Added an isolated `PDF_TO_JSON_RAG_DATA_DIR` path so the CLI can run cleanly outside the repo-local benchmark workspace
- Added public-surface smoke tests that generate a tiny PDF and validate the packaged CLI path end to end
- Added user-facing release helpers:
  - `help`
  - `doctor`
  - `demo-profile`
  - command aliases
  - `--output` JSON export
- Split public CLI onboarding docs from internal evaluation notes
- Re-ran the public CLI tests, core regression shards, and the full benchmark after the release-facing pass

## Validation Summary

Representative validation that has already been completed:

- native extraction smoke test on `medical/Common_cold_clinincal_evidence.pdf`
- extraction-to-JSON smoke test on the same sample
- chunk generation smoke test on the same sample
- local vector index smoke test
- retrieval smoke test for common-cold queries
- OCR fallback smoke test on synthetic image-only PDFs
- repeated full-benchmark reruns across mixed source families
- repeated regression-shard checks on structured-form, cross-document, document-discovery, document-facet, query-planning, and answer-mode paths
- isolated public-surface CLI smoke tests via `python -m unittest tests.test_cli_public_surface`
- public `doctor`, `demo-profile`, and `help` command checks
- full-benchmark rerun after the `v1.29-v1.33` release-facing pass

### v1.34-v1.41

- Hardened document processing with extraction-time block metadata and chunk-level semantic metadata.
- Reduced brittle retrieval behavior with semantic overlap, structural-reference alignment, and more coverage-aware evidence selection.
- Made the public quickstart self-contained with `create-demo-pdf`, `doctor`, `package-check`, and `release-check`.
- Added packaged-install verification through a temporary wheel-build and clean install path.
- Added public-safe pre-release artifacts and aligned the repo around a first `v0.1.0-beta` release candidate.

### v0.1.1-v0.1.2

- Fixed post-beta public-path issues in `document_overview`, `inspect-document`, and `doctor`.
- Added clearer next-step guidance in the CLI after initialization, extraction, chunking, and indexing.
- Reworked the public quickstart so the shortest path goes through `smoke-check` and `run-workflow` before the manual multi-step path.
- Aligned install/onboarding docs around `python -m pip install .` and a clear local-development fallback.

### v0.2.0

- Added extraction-time document sections as a reusable structure layer in the saved document JSON.
- Reworked chunk construction so section boundaries come from extraction-time structure and chunk metadata inherits section summaries and coverage terms.
- Made retrieval scoring more inspectable by exposing quality, semantic, structural, metadata, and rank signals in the runtime payload.
- Tightened document-overview answers so they render a cleaner section-aware summary instead of mostly replaying inventory strings.
- Changed the default embedding path to deterministic local fallback, with `sentence-transformers` now opt-in for stronger local embeddings.

### v0.2.1-v0.2.4

- Fixed install-time path resolution so the CLI defaults to a user data directory outside the repo instead of writing under the Python installation.
- Added packaged example assets and install-safe loading for `demo-profile`, `doctor`, `create-demo-pdf`, and related public-surface commands.
- Hardened `package-check` so it validates a real installed wheel from a clean temporary workspace instead of accidentally relying on the source checkout.
- Split `release-check` into clearer layers:
  - public-surface smoke
  - maintainer package/test gates
  - optional benchmark-only regressions
- Re-validated the packaged/public release path after these changes and confirmed that the remaining red status is limited to a small set of repo-local regression cases after the `v0.2.0` refactor.

### v0.2.5

- Recovered the core repo-local failures in:
  - `query_planning_core`
  - `answer_modes_core`
  - `document_family_core`
  - `relationship_core`
- Added a narrow anchor-recovery retrieval path for high-risk intents where embedding-first hits were hiding the correct chunk inside the right document.
- Tightened section-aware and source-aware scoring for:
  - common-cold symptom lookup
  - questionnaire context lookup
  - AJMedP hypothermia predisposition lookup
  - LBDL document routing
  - vitamin-C vs echinacea prevention comparison
- Revalidated the public release path and maintainer gates after the recovery pass.
- Re-ran the broad benchmark and confirmed that full-benchmark parity is still the next internal target, even though the public release path and maintainer shard set are now green.

### v0.2.6

- Tightened `release-check` so internal benchmark regressions only run when the active data root really contains full benchmark assets, instead of any partial inventory/index pair.
- Fixed unsupported-query behavior for:
  - `document_routing` queries like lease/rent clauses
  - treatment-style queries that ask about unsupported targets like influenza
- Fixed evaluator assumptions so document-level benchmark cases can use `relevant_doc_ids` without crashing the full rerun.
- Recovered the final broad-benchmark warnings on the section-aware architecture, including:
  - `symptoms`
  - `antibiotics`
  - `cmaj_zinc_prevention`
  - `wat_rhinovirus_most_common`
  - `source_listing_vitamin_c_and_echinacea`
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `release-check`
  - the full 67-case benchmark
- End state:
  - public release gates: green
  - maintainer shard set: green
  - full 67-case benchmark: green
  - sampled faithfulness audit: still flags two document-level cases, which is now the next internal target

## Deferred Features

Still intentionally deferred:

- `pdfplumber`
- cross-encoder reranking
- automated LLM-as-a-judge evaluation
- cloud deployment
- multi-document schema extraction
- visual grounding UI

These stay out of scope unless future architecture work exposes failures the current local stack cannot absorb.
