# PDF-to-JSON RAG

Local-first, domain-agnostic PDF-to-JSON RAG tool for turning PDFs into structured JSON, routing queries across documents, and returning grounded CLI answers.

![Public CLI readiness check](./pdf_json_gh_repo.png)

## Lineage

This repo is a personal implementation inspired by:

- the upstream course repo [https-deeplearning-ai/sc-landingai](https://github.com/https-deeplearning-ai/sc-landingai)
- the course [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information)

The working course fork and setup/debug history are preserved separately in:

- [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai)

This codebase is intentionally separate from that fork. The fork captures the baseline course reproduction and AWS-side learning path; this repo moves toward a local-first, JSON-first implementation with tighter control over chunking, retrieval, and evaluation.

## Current Status

Current public version: `0.1.0-beta`

Internal development iterations in this repo use `v1.x` labels. Public releases follow semantic versioning starting at `0.1.0-beta`.

Current post-beta focus:

- `v0.3.9`: completed the candidate-document handoff simplification with an explicit document-selection contract
- current emphasis:
  - stronger document-processing structure signals, not more benchmark volume
  - broad-benchmark stability on the simplified stack
  - release gates that include the simplified document-level path and structure-sensitive chunking cases
  - document-level support traces that stay faithful to the rendered answer sentences
  - delaying any learned reranker until the simplified baseline proves insufficient

Current working-tree architecture upgrades:

- extraction-time document sections and section-aware chunk metadata
- feature-based query planning with explicit mode scores and rationale
- simplified shortlist scoring with explicit breakdowns
- two-stage document-level retrieval:
  - candidate document selection
  - chunk retrieval inside selected documents
- explicit `document_selection` traces that carry:
  - candidate docs
  - ranked docs
  - selected docs
  - primary doc
  - selection strategy
- unified support-trace building for overview, routing, listing, justification, and comparison answers
- support traces that now carry section summaries, section-level hints, and answer-shaped document facts
- richer structure-aware chunk metadata:
  - `section_content_hints`
  - `chunk_type`
  - stronger section-level heading hierarchy
- structure-aware chunk boundaries for table-heavy and questionnaire-like sections
- compact default JSON answer payloads with `--verbose` for full retrieval/debug state
- deterministic local embedding fallback by default
- packaged example assets and install-safe example loading
- maintainer release gates that distinguish public-surface checks from benchmark-only regressions

The current tool runs end to end across a mixed local benchmark that includes native-text papers, OCR-derived scans, technical manuals, questionnaires, checklist-style appendices, books, guidance notes, and short model-report documents.

Current validation state:

- the public install/release path passes `package-check` and `release-check`
- core maintainer regression shards pass:
  - `query_planning_core`
  - `answer_modes_core`
  - `document_pipeline_core`
  - `structure_chunking_core`
  - `evidence_anchor_core`
  - `document_family_core`
  - `inventory_coverage_core`
  - `relationship_core`
- the full 67-case benchmark is green on the simplified `v0.3.x` baseline:
  - `precision@5 = 0.5309`
  - `recall@5 = 1.0`
  - `MRR = 1.0`
  - `avg_keyword_coverage = 1.0`
  - `negative_success_rate = 1.0`
  - `warning_case_count = 0`
- the sampled faithfulness audit remains green:
  - `sampled_case_count = 20`
  - `avg_supported_sentence_ratio = 1.0`
  - `failing_case_count = 0`
- the current decision after `v0.3.9` is to keep learned reranking deferred:
  - the simplified heuristic baseline is green on release gates, maintainer shards, the full benchmark, and the sampled faithfulness audit

## Capabilities

- Extract native-text PDFs into document-level and chunk-level JSON artifacts.
- Fall back to OCR with `pytesseract` when native extraction is weak or missing.
- Build a local vector index with `ChromaDB`.
- Carry extraction-time document sections into chunking, inspection, and document-level synthesis.
- Answer single-document evidence questions with grounded citations.
- Route document-level and cross-document queries such as:
  - `What does this file cover?`
  - `What kind of document is this?`
  - `Which file is most relevant for X?`
  - `Why is this the best source?`
  - `What do these sources have in common or how do they differ?`
- Expose inspection and planning paths through a packaged CLI:
  - `list-documents`
  - `inspect-document`
  - `plan-query`
  - `answer-query`
  - `run-workflow`
  - `smoke-check`
- Expose maintainer-facing release gates through:
  - `package-check`
  - `release-check`

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

## Quickstart

```bash
python -m pip install .
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

Use a fresh `PDF_TO_JSON_RAG_DATA_DIR` for quickstart and release-check runs so old local artifacts do not make `doctor` or `release-check` look greener than the current session really is.

Optional stronger local embeddings:

```bash
export PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1
```

Fastest full workflow:

```bash
pdf-to-json-rag run-workflow --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

Maintainer release validation:

```bash
pdf-to-json-rag package-check --json
pdf-to-json-rag release-check --json
```

When you are validating new local code in a source checkout before reinstalling the package, prefer:

```bash
PYTHONPATH=src python -m pdf_to_json_rag release-check --json
PYTHONPATH=src python -m pdf_to_json_rag evaluate-mvp --top-k 5 --json
```

For end users, the main public validation path is still:

```bash
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

More detailed usage lives in:

- [docs/CLI_QUICKSTART.md](./docs/CLI_QUICKSTART.md)
- [docs/CLI_REFERENCE.md](./docs/CLI_REFERENCE.md)

## Key Files

- `project-plan.md`
  Scope, milestones, and deferred items.
- `DEVELOPMENT_LOG.md`
  Internal engineering log and implementation history.
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

Current gate status:

- public release gates are green
- the maintainer shard set used by `release-check` is green
- `release-check` now covers both `document_pipeline_core` and `evidence_anchor_core` as part of the maintainer regression gate
- `release-check` now also covers `structure_chunking_core`, so structure-sensitive form/manual/checklist cases stay in the default maintainer gate
- `release-check` now distinguishes cleanly between:
  - public-surface gates
  - maintainer package/test gates
  - internal benchmark regressions that should only run when full benchmark assets are actually present
- `release-check` now recommends the current public beta tag instead of the stale pre-`v0.3.x` suggestion
- the current internal focus is no longer broad benchmark churn; it is preserving the simplified document-level baseline, the new `document_selection` contract, and only revisiting a learned reranker if future failures justify it

## Limitations

- OCR fallback is still heuristic and not fully layout-aware
- OCR grouping rebuilds paragraph-like blocks rather than true layout regions
- Chunking is still heuristic rather than fully semantic
- Section detection is improved, but still rule-based and fragile on unfamiliar layouts
- Retrieval is stronger than earlier versions, but still depends on heuristic scoring, chunk-quality labels, structural cues, and a lightweight reranking pass rather than a more model-driven reranker
- The default public path now prefers deterministic local embeddings over automatic model downloads; stronger local sentence-transformer embeddings are opt-in
- Document facets are useful, but still heuristic and not yet learned from a richer metadata or classifier layer
- Document-family classification is compact and useful, but still heuristic rather than trained
- Query planning and document inventory are explicit now, but still built from heuristic metadata rather than a learned planner or classifier
- Document-level summaries, shortlist decisions, and answer contracts are cleaner and more inspectable now, but still generated from heuristic metadata rather than a stronger summarization/classification layer
- Block metadata and chunk semantics improve robustness, but they are still derived from handcrafted rules rather than a richer learned representation
- Structured-form and appendix handling are broader than before, but still validated on a narrow set of questionnaire/checklist examples
- Cross-document and document-discovery behavior are implemented, but still benchmarked on a modest hand-built set of source-discovery, overview, routing, and comparison queries
- The `v0.3.0` simplification pass reduced document-level glue code, but the pipeline is still heuristic-first rather than model-first
- Grounded answers are extractive, not LLM-synthesized
- The benchmark is broader than before, but still hand-built and not yet domain-diverse enough to prove true generalization
- The scanned benchmark still uses a narrow OCR-heavy set rather than a broader scanned-document collection
- The sampled faithfulness audit is useful as a checkpoint, but it still surfaces weaker document-level support traces and is not yet a substitute for broader human review
- Multilingual robustness is not validated yet

## Notes on Reference Material

This repo was brainstormed with ideas from:

- [DeepLearning.AI Skill Builder](https://skillbuilder.deeplearning.ai/)
- ChatGPT 5.4

Earlier in development, a small set of course notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied into a temporary `references/` folder and used only as design input for OCR fallback planning, reading-order/layout handling, schema design, and grounding-aware RAG flow.

Those reference notebooks were removed from the final repo structure. The current codebase is a separate local implementation rather than a notebook-derived copy.
