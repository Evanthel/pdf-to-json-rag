# PDF-to-JSON RAG

Local-first, domain-agnostic PDF-to-JSON RAG tool for turning PDFs into structured JSON, routing queries across documents, and returning grounded CLI answers.

## Lineage

This repo is a personal implementation inspired by:

- the upstream course repo [https-deeplearning-ai/sc-landingai](https://github.com/https-deeplearning-ai/sc-landingai)
- the course [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information)

The working course fork and setup/debug history are preserved separately in:

- [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai)

This codebase is intentionally separate from that fork. The fork captures the baseline course reproduction and AWS-side learning path; this repo moves toward a local-first, JSON-first implementation with tighter control over chunking, retrieval, and evaluation.

## Current Status

Current version: `v1.41`

The tool currently runs end to end across a mixed benchmark of review papers, OCR-derived scans, technical manuals, questionnaires, checklist-style appendices, non-medical books, open guidance notes, and short model-report style documents.

It supports both:

- single-document grounded evidence questions
- cross-document queries such as source listing and source comparison
- document-discovery queries such as “what does this file cover?” and “which file is most relevant for X?”
- document-facet queries such as “what kind of document is this?” and “what is its purpose?”
- document-family queries such as “is this a manual, guidance note, model report, or book?”
- source-justification queries such as “why is this the best source?”
- ambiguity-aware routing queries that should surface more than one relevant source
- query-planned paths that separate evidence lookup, document discovery, cross-document comparison, and document-facet questions before retrieval
- normalized answer contracts for overview, routing, comparison, and evidence-style answers
- inventory-level coverage summaries and relationship reasoning for document-level and cross-document answers
- tool-oriented inspection paths such as document listing, document inspection, query planning, and JSON answer output
- a packaged CLI entry path for `python -m pdf_to_json_rag` and `pdf-to-json-rag`
- a packaged smoke-check path for first-run validation
- public-safe example JSON outputs for inspect / plan / answer command shapes
- an isolated `PDF_TO_JSON_RAG_DATA_DIR` path for testing and local tool use without polluting the repo workspace
- public release helpers such as `doctor`, `demo-profile`, `create-demo-pdf`, `package-check`, `release-check`, command aliases, and `--output` JSON export
- a recommended first public release tag of `v0.1.0-beta` once `release-check` passes, including the packaged-install gate

Current benchmark snapshot:

- `Cases`: `67`
- `Indexed sample documents`: `19`
- `precision@5`: `0.521`
- `recall@5`: `1.0`
- `MRR`: `1.0`
- `avg_keyword_coverage`: `1.0`
- `negative_success_rate`: `1.0`
- `warning_case_count`: `0`

## What Works

- Native PDF extraction with `PyMuPDF`
- OCR fallback for weak or missing native text using `pytesseract`
- Document-level JSON output in `data/documents/`
- Chunk generation with reading-order preservation, section detection, overflow splitting, noise filtering, and OCR provenance
- Local vector indexing with `ChromaDB`
- Retrieval with intent-aware reranking, chunk quality labels, source-aware locking, and cross-document source matching
- Adjacent-chunk expansion for context reconstruction
- Grounded answer assembly with explicit evidence citations and structured answer traces
- Cross-document source-listing and comparison answers
- Lightweight document-overview, document-routing, and source-justification answers
- Extraction-time summary cues for document-level overview answers
- Extraction-time `discovery_terms` for source selection and mixed-domain routing
- Extraction-time document facets for type, purpose, audience, evidence style, and structure style
- Shared document-semantics interpretation for facets, coverage summaries, and inventory summaries
- Reusable inventory summaries derived from extraction-time metadata
- Reusable coverage summaries and coverage terms for routing and overview behavior
- Extraction-time document-family classification for books, guidance notes, model reports, manuals, forms, and clinical references
- Extraction-time block metadata for block kind, token density, line count, and structural flags across native and OCR paths
- Chunk-level semantic metadata such as semantic terms, content hints, and source block kinds carried into the index
- Query planning that separates evidence lookup, document discovery, cross-document comparison, and document-facet questions
- Explicit answer modes for overview, routing, source-listing, source-justification, comparison, and evidence lookup
- Inventory-first document shortlisting before chunk-level retrieval
- Retrieval bonuses that align structural references such as `Appendix B`, `Table 3-4`, and `Question 9` with the right chunks
- Lightweight answer contracts that make document-level and cross-document answer paths more inspectable
- Coverage-aware evidence selection that prefers complementary support over repeated high-score fragments
- Relationship reasoning for overlap, complement, and divergence between sources
- Ambiguity-aware multi-source routing for mixed-domain discovery queries
- Tool-facing CLI paths for listing documents, inspecting document metadata, planning queries, and returning JSON answers
- A packaged project entry path through `pyproject.toml`, `python -m pdf_to_json_rag`, and the `pdf-to-json-rag` console script
- A `smoke-check` command that validates the packaged end-to-end workflow path
- A `create-demo-pdf` command that generates a public-safe sample PDF for quickstart and smoke checks
- A `release-check` command that combines public-surface smoke validation with key maintainer regression shards
- A `package-check` command that builds a wheel and verifies the packaged CLI from a clean temporary install root
- A `--format text|json` option for the public CLI surface, in addition to `--json`
- Stable CLI error envelopes for missing inputs, missing index state, and argument errors
- A `doctor` command that separates required public-tool readiness from optional OCR support and internal benchmark readiness
- A `demo-profile` command that exposes the public-safe onboarding flow
- Optional JSON export with `--output /path/to/file.json`
- Multi-document evaluation with regression shards, per-case debug snapshots, slice summaries, rerank comparison, and deferred-feature decision checkpoints
- Automated public-surface CLI smoke tests that run against an isolated temporary data root

## Workflow

1. Extract a PDF into `*.native.json` and `*.document.json`
2. Convert extracted blocks into chunk JSON files
3. Build a persistent local vector index from chunk text and metadata
4. Plan the query as evidence lookup, document discovery, cross-document, or document-facet behavior
5. Shortlist candidate documents from inventory metadata when the query needs document-level routing
6. Retrieve top-k chunks for the shortlisted sources
7. Expand with adjacent chunks when needed
8. Assemble a grounded answer from the expanded context, or a source-level / document-level answer for source discovery, comparison, overview, routing, and facet questions
9. Evaluate the result on the full benchmark or on a smaller regression shard

## How to Run

Minimal local flow:

```bash
pip install -e .
export PDF_TO_JSON_RAG_DATA_DIR=/tmp/pdf-to-json-rag-data
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
pdf-to-json-rag package-check --json
pdf-to-json-rag demo-profile --json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag extract-native --pdf /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag chunk-document --doc-id your-doc-id --json
pdf-to-json-rag build-index --doc-id your-doc-id --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
```

Retrieve without answer assembly:

```bash
pdf-to-json-rag retrieve --query "How are common cold infections transmitted?" --k 5 --json
```

Build one local index across multiple extracted documents:

```bash
pdf-to-json-rag build-index --doc-id doc-a,doc-b --json
```

Run the full benchmark:

```bash
pdf-to-json-rag evaluate-mvp --k 5 --json
```

Run a smaller regression shard:

```bash
pdf-to-json-rag evaluate-regression --k 5 --shard cross_document_core --json
```

Inspect document inventory and planning paths:

```bash
pdf-to-json-rag list-documents --json
pdf-to-json-rag inspect-document --doc-id common-cold-clinincal-evidence --json
pdf-to-json-rag plan-query --query "Which file is most relevant for drought triggers?" --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
pdf-to-json-rag run-workflow --pdf /path/to/file.pdf --query "What does this file cover?" --json
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
pdf-to-json-rag release-check --json
```

Write JSON output to a file:

```bash
pdf-to-json-rag answer-query --query "What does this file cover?" --json --output answer.json
```

## Key Files

- `project-plan.md`
  Scope, milestones, and deferred items.
- `WORK_LOG.md`
  High-level implementation log and smoke-test history.
- `src/pdf_to_json_rag/`
  Extraction, chunking, indexing, retrieval, answering, and evaluation code.
- `data/eval/mvp_eval_cases.json`
  Hand-built benchmark cases.
- `data/eval/faithfulness_audit_cases.json`
  Sampled faithfulness-audit set.
- `examples/`
  Public-safe workflow assets, example queries, and trimmed example JSON outputs.
- `docs/CLI_QUICKSTART.md`
  Public onboarding path for the packaged CLI.
- `docs/CLI_REFERENCE.md`
  User-facing command reference and aliases.
- `docs/INTERNAL_EVALUATION.md`
  Internal benchmark and regression notes.
- `data/eval/mvp_eval_report.json`
  Generated locally by the evaluation workflow and ignored by default.

## Evaluation Snapshot

The saved evaluation report currently includes:

- per-case debug records with top-k retrieval snapshots, expanded-context snapshots, answer previews, answer traces, and evidence snippets
- document-family and structure slices such as `review_summary`, `table_heavy`, `form_grid`, `appendix_like`, `scanned_ct`, `source_anchored_review`, `source_anchored_technical`, `source_anchored_form`, `cross_document`, and `document_facets`
- document-discovery slices spanning books, guidance notes, model reports, manuals, and mixed-domain routing cases
- a compact `document_facets_core` regression shard for document type and document purpose questions
- a compact `inventory_coverage_core` regression shard for inventory-summary and coverage-aware routing behavior
- a compact `query_planning_core` regression shard for query-class separation, document routing, source listing, and cross-document comparison
- a compact `answer_modes_core` regression shard for explicit answer-mode separation
- a compact `document_family_core` regression shard for shared document-family reasoning
- a compact `relationship_core` regression shard for overlap / complement / divergence reasoning
- answer-contract slice checks for document-level and comparison-style answer paths
- a retrieval-strategy comparison between the chunking-first baseline and the current lightweight reranking pass
- a sampled faithfulness audit over selected grounded cases
- explicit deferred-feature decisions for `pdfplumber`, cross-encoder reranking, and `LLM-as-a-judge`

On the current benchmark:

- OCR-derived, technical/manual, form/grid, appendix/checklist, cross-document, and document-discovery slices are stable
- query-planning and document-inventory slices are stable
- answer-mode separation is stable
- document-family reasoning and answer-contract slices are stable
- the lightweight reranking pass is still sufficient
- `pdfplumber`, cross-encoder reranking, and `LLM-as-a-judge` are still not justified by the observed failure modes

## Limitations

- OCR fallback is still heuristic and not fully layout-aware
- OCR grouping rebuilds paragraph-like blocks rather than true layout regions
- Chunking is still heuristic rather than fully semantic
- Section detection is improved, but still rule-based and fragile on unfamiliar layouts
- Retrieval is stronger than earlier versions, but still depends on heuristic scoring, chunk-quality labels, structural cues, and a lightweight reranking pass rather than a more model-driven reranker
- Document facets are useful, but still heuristic and not yet learned from a richer metadata or classifier layer
- Document-family classification is compact and useful, but still heuristic rather than trained
- Query planning and document inventory are explicit now, but still built from heuristic metadata rather than a learned planner or classifier
- Document-level summaries and answer contracts are reusable now, but still generated from heuristic metadata rather than a stronger summarization/classification layer
- Block metadata and chunk semantics improve robustness, but they are still derived from handcrafted rules rather than a richer learned representation
- Structured-form and appendix handling are broader than before, but still validated on a narrow set of questionnaire/checklist examples
- Cross-document and document-discovery behavior are implemented, but still benchmarked on a modest hand-built set of source-discovery, overview, routing, and comparison queries
- Grounded answers are extractive, not LLM-synthesized
- The benchmark is broader than before, but still hand-built and not yet domain-diverse enough to prove true generalization
- The scanned benchmark still uses a narrow OCR-heavy set rather than a broader scanned-document collection
- The sampled faithfulness audit is useful as a checkpoint, but not yet a substitute for broader human review
- Multilingual robustness is not validated yet

## Notes on Reference Material

This repo was brainstormed with ideas from:

- [DeepLearning.AI Skill Builder](https://skillbuilder.deeplearning.ai/)
- ChatGPT 5.4

Earlier in development, a small set of course notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied into a temporary `references/` folder and used only as design input for OCR fallback planning, reading-order/layout handling, schema design, and grounding-aware RAG flow.

Those reference notebooks were removed from the final repo structure. The current codebase is a separate local implementation rather than a notebook-derived copy.

## Version Log

### v1.0-v1.13

- established the local-first extraction, chunking, indexing, grounded-answering, and early evaluation core
- expanded from single-document evidence lookup to structured-form handling and initial cross-document behavior

### v1.14-v1.21

- shifted the repo toward document discovery, overview, routing, source justification, and comparison
- added document facets, document families, query planning, answer modes, and document-level answer contracts

### v1.22-v1.28

- consolidated document semantics and inventory-first routing
- turned the repo into a packageable CLI tool with `run-workflow`, `smoke-check`, and public-safe example assets

### v1.29-v1.38

- hardened the public CLI surface with isolated data roots, smoke tests, and release helpers
- strengthened processing and retrieval with block metadata, chunk semantics, structural alignment, and coverage-aware evidence selection

### v1.39-v1.41

- made the public quickstart self-contained with `create-demo-pdf`
- added `release-check` and `package-check` for public-surface and packaged-install validation
- tightened `doctor`, output formatting, and public release artifacts
- confirmed the first public release should ship as `v0.1.0-beta`
