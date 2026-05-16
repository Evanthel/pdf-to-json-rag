# WORK_LOG

## Purpose

This file records the project at a practical level without mirroring every code edit. It is intended as a concise engineering log rather than a full changelog.

## Current State

Current implementation level: `v1.28`

The project now behaves as a local-first, domain-agnostic `PDF -> JSON -> retrieval -> grounded answer` pipeline with explicit document-intelligence behavior on top of chunk retrieval.

Current benchmark state:

- `Cases`: `67`
- `Indexed sample documents`: `19`
- `precision@5`: `0.521`
- `recall@5`: `1.0`
- `MRR`: `1.0`
- `avg_keyword_coverage`: `1.0`
- `negative_success_rate`: `1.0`
- `warning_case_count`: `0`
- `slice_stability_all_pass`: `True`

Key regression shards currently passing:

- `document_facets_core`
- `document_family_core`
- `inventory_coverage_core`
- `relationship_core`
- `query_planning_core`
- `answer_modes_core`

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

## Deferred Features

Still intentionally deferred:

- `pdfplumber`
- cross-encoder reranking
- automated LLM-as-a-judge evaluation
- cloud deployment
- multi-document schema extraction
- visual grounding UI

These stay out of scope unless future architecture work exposes failures the current local stack cannot absorb.
