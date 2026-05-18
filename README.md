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

Current public release target: `0.1.0-beta`

Internal development iterations in this repo use `v1.x` labels. Public releases follow semantic versioning starting at `0.1.0-beta`.

The current tool runs end to end across a mixed local benchmark that includes native-text papers, OCR-derived scans, technical manuals, questionnaires, checklist-style appendices, books, guidance notes, and short model-report documents.

Current benchmark snapshot:

- `Cases`: `67`
- `Indexed sample documents`: `19`
- `precision@5`: `0.521`
- `recall@5`: `1.0`
- `MRR`: `1.0`
- `avg_keyword_coverage`: `1.0`
- `negative_success_rate`: `1.0`
- `warning_case_count`: `0`

## Capabilities

- Extract native-text PDFs into document-level and chunk-level JSON artifacts.
- Fall back to OCR with `pytesseract` when native extraction is weak or missing.
- Build a local vector index with `ChromaDB`.
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
pip install -e .
export PDF_TO_JSON_RAG_DATA_DIR=/tmp/pdf-to-json-rag-data
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag extract-native --pdf /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag chunk-document --doc-id your-doc-id --json
pdf-to-json-rag build-index --doc-id your-doc-id --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
```

First-release validation:

```bash
pdf-to-json-rag package-check --json
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
pdf-to-json-rag release-check --json
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
