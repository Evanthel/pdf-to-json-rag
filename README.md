# PDF-to-JSON RAG

Local-first, domain-agnostic PDF-to-JSON RAG pipeline with structured extraction, source-aware retrieval, and an inspectable multi-document evaluation loop.

## Lineage

This repo is a personal implementation inspired by:

- the upstream course repo [https-deeplearning-ai/sc-landingai](https://github.com/https-deeplearning-ai/sc-landingai)
- the course [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information)

The working course fork and setup/debug history are preserved separately in:

- [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai)

This codebase is intentionally separate from that fork. The fork captures the baseline course reproduction and AWS-side learning path; this repo moves toward a local-first, JSON-first implementation with tighter control over chunking, retrieval, and evaluation.

## Current Status

Current version: `v1.13`

The pipeline currently runs end-to-end across a mixed benchmark of review papers, OCR-derived scans, technical manuals, questionnaires, and checklist-style appendices.

It supports both:

- single-document grounded evidence questions
- cross-document queries such as source listing and source comparison

Current benchmark snapshot:

- `Cases`: `48`
- `Indexed sample documents`: `11`
- `precision@5`: `0.312`
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
- Multi-document evaluation with regression shards, per-case debug snapshots, slice summaries, rerank comparison, and deferred-feature decision checkpoints

## Workflow

1. Extract a PDF into `*.native.json` and `*.document.json`
2. Convert extracted blocks into chunk JSON files
3. Build a persistent local vector index from chunk text and metadata
4. Retrieve top-k chunks for a query
5. Expand with adjacent chunks when needed
6. Assemble a grounded answer from the expanded context, or a source-level answer for cross-document discovery/comparison queries
7. Evaluate the result on the full benchmark or on a smaller regression shard

## How to Run

Minimal local flow:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pdf_to_json_rag.cli extract-native --pdf /path/to/file.pdf
PYTHONPATH=src python -m pdf_to_json_rag.cli chunk-document --doc-id your-doc-id
PYTHONPATH=src python -m pdf_to_json_rag.cli build-index --doc-id your-doc-id
PYTHONPATH=src python -m pdf_to_json_rag.cli answer-query --query "What are common cold symptoms?"
```

Retrieve without answer assembly:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli retrieve --query "How are common cold infections transmitted?" --k 5
```

Build one local index across multiple extracted documents:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli build-index --doc-id doc-a,doc-b
```

Run the full benchmark:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli evaluate-mvp --k 5
```

Run a smaller regression shard:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli evaluate-regression --k 5 --shard cross_document_core
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
- `data/eval/mvp_eval_report.json`
  Generated locally by the evaluation workflow and ignored by default.

## Evaluation Snapshot

The saved evaluation report currently includes:

- per-case debug records with top-k retrieval snapshots, expanded-context snapshots, answer previews, answer traces, and evidence snippets
- document-family and structure slices such as `review_summary`, `table_heavy`, `form_grid`, `appendix_like`, `scanned_ct`, `source_anchored_review`, `source_anchored_technical`, `source_anchored_form`, and `cross_document`
- a retrieval-strategy comparison between the chunking-first baseline and the current lightweight reranking pass
- a sampled faithfulness audit over selected grounded cases
- explicit deferred-feature decisions for `pdfplumber`, cross-encoder reranking, and `LLM-as-a-judge`

On the current benchmark:

- OCR-derived, technical/manual, form/grid, appendix/checklist, and cross-document slices are stable
- the lightweight reranking pass is still sufficient
- `pdfplumber`, cross-encoder reranking, and `LLM-as-a-judge` are still not justified by the observed failure modes

## Limitations

- OCR fallback is still heuristic and not fully layout-aware
- OCR grouping rebuilds paragraph-like blocks rather than true layout regions
- Chunking is still heuristic rather than fully semantic
- Section detection is improved, but still rule-based and fragile on unfamiliar layouts
- Retrieval still depends on lightweight heuristics, chunk quality labels, and a small lexical reranking pass
- Structured-form and appendix handling are broader than before, but still validated on a narrow set of questionnaire/checklist examples
- Cross-document behavior is now implemented, but still benchmarked on a small set of source-discovery and comparison queries
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

### v1.0-v1.2

These versions established the local-first core:

- native extraction with `PyMuPDF`
- OCR fallback through `pytesseract`
- document and chunk JSON outputs
- local indexing with `ChromaDB`
- grounded extractive answering
- the first multi-document benchmark and retrieval/answer cleanup passes

### v1.3-v1.5

These versions focused on structure and retrieval quality:

- added the echinacea document and the first OCR-derived scanned benchmark
- introduced chunk quality labels, neighbor-expansion gating, and richer debug reports
- improved OCR-to-chunk handoff
- pushed chunking-first improvements on mixed review/treatment summaries instead of adding a heavier reranker too early

### v1.6-v1.9

These versions broadened the benchmark and turned deferred-feature questions into measured decisions:

- added review-heavy, table-heavy, manual, questionnaire, and checklist-style sources
- introduced richer evaluation slices and retrieval-strategy comparison
- added a sampled faithfulness audit
- kept `pdfplumber`, cross-encoder reranking, and `LLM-as-a-judge` deferred because the benchmark still did not justify them

### v1.10-v1.11

These versions made the structured-form path more maintainable:

- added the opioid appendix family as another structured-form benchmark source
- generalized form logic into reusable pattern families
- added deterministic regression shards for high-risk source-anchored form cases

### v1.12-v1.13

These versions pushed the project beyond single-document answering:

- moved structured intent metadata into a shared declarative config
- added structured answer traces to debug output
- added another appendix-heavy checklist source without another one-off code path
- added cross-document intents for source listing and source comparison
- wired multi-source document matching into retrieval and stabilized it with a dedicated regression shard

Current `v1.13` benchmark state:

- `precision@5 = 0.312`
- `recall@5 = 1.0`
- `MRR = 1.0`
- `avg_keyword_coverage = 1.0`
- `negative_success_rate = 1.0`
- `warning_case_count = 0`
