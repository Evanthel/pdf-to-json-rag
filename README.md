# PDF-to-JSON RAG

Local-first PDF-to-JSON RAG pipeline with structured extraction, grounded retrieval, and a small inspectable evaluation loop.

## Lineage

This repo is a personal implementation inspired by:

- the upstream course repo [https-deeplearning-ai/sc-landingai](https://github.com/https-deeplearning-ai/sc-landingai)
- the course [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information)

The working course fork and setup/debug history are preserved separately in:

- [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai)

This codebase is intentionally separate from that fork and focuses on a local-first JSON pipeline rather than the original AWS/LandingAI lab structure.

The split is deliberate:

- the fork captures the baseline course reproduction and AWS-side learning path
- this repo moves toward a local-first implementation
- it also makes the project JSON-first rather than markdown-first, with more direct control over chunking, retrieval, and evaluation

## Current Status

Current version: `v1.4`.

The repo is in a working local-first state: PDF extraction, chunking, local indexing, grounded answering, and benchmark evaluation all run end-to-end across multiple sample documents.

The current implementation is still deliberately heuristic-heavy, but `v1.4` is now complete: the benchmark spans four indexed documents, includes a locally derived scanned CT-study PDF plus negative abstention checks, and now has OCR-focused cleanup, generalized treatment-evidence handling, per-case debug records, and simple slice summaries. The next iteration should be chunking-first rather than reranking-first.

## What Works

- Native PDF extraction with `PyMuPDF`
- OCR fallback for pages with weak or missing native text, with paragraph-like OCR block recovery and cleaner OCR-to-chunk handoff instead of a single full-page text blob
- OCR provenance carried through extraction and chunk metadata
- Document-level JSON output in `data/documents/`
- Chunk generation with reading-order preservation, section detection, paragraph-aware cleanup, sentence-aware overflow splitting, cleaner `Key points`-style summary splitting, and chunk-level noise labels / quality scores
- Local vector indexing with `ChromaDB`
- Retrieval from the local index with intent-aware reranking, noise-label-aware filtering, heuristic noise suppression, quality-gated neighbor expansion, and generalized treatment-evidence retrieval behavior
- Adjacent-chunk expansion for context reconstruction
- Grounded answer assembly with explicit evidence citations and targeted answer-selection heuristics for treatment prevention, null-effect, subgroup-benefit, duration, and overall-conclusion queries
- multi-document evaluation workflow across four indexed sample documents, with grounded cases, negative abstention checks, per-case debug snapshots, and simple benchmark slices

## Workflow

1. Extract a PDF into `*.native.json` and `*.document.json`, using OCR fallback when native text is too weak
2. Convert extracted blocks into chunk JSON files with chunk-level provenance
3. Build a persistent local vector index from chunk text and metadata
4. Retrieve top-k chunks for a query
5. Expand with adjacent chunks
6. Assemble a grounded answer from the expanded context
7. Evaluate retrieval and answer quality on a hand-built multi-document benchmark with negative queries and inspect the saved per-case debug snapshots

## How to Run

Minimal local flow:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pdf_to_json_rag.cli extract-native --pdf /path/to/file.pdf
PYTHONPATH=src python -m pdf_to_json_rag.cli chunk-document --doc-id your-doc-id
PYTHONPATH=src python -m pdf_to_json_rag.cli build-index --doc-id your-doc-id
PYTHONPATH=src python -m pdf_to_json_rag.cli answer-query --query "What are common cold symptoms?"
```

To inspect retrieval directly:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli retrieve --query "How are common cold infections transmitted?" --k 5
```

To build one local index across multiple extracted documents:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli build-index --doc-id doc-a,doc-b
```

To run the local benchmark:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli evaluate-mvp --k 5
```

## Key Files

- `project-plan.md`
  The current scope, MVP definition, and deferred items.
- `WORK_LOG.md`
  High-level implementation log and smoke-test history.
- `src/pdf_to_json_rag/`
  Core extraction, chunking, indexing, retrieval, answering, and evaluation code.
- `data/eval/mvp_eval_cases.json`
  Small local evaluation set.
- `data/eval/mvp_eval_report.json`
  Generated locally by the evaluation workflow and ignored by default.

## Evaluation Snapshot

Current local benchmark snapshot:

- `precision@5`: `0.431`
- `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.974`
- negative case success rate: `1.0`

This reflects the current 17-case benchmark after adding a locally derived scanned CT-study document, two new OCR-heavy grounded cases, and a fourth negative abstention check. The benchmark now measures whether the pipeline can stay coherent not only across treatment-heavy native PDFs, but also when OCR-derived evidence is mixed into the same local index.

The saved report now also includes per-case debug records with top-k retrieval snapshots, expanded-context snapshots, answer previews, and evidence snippets so regressions are faster to inspect after each quality pass.

It now also stores simple slice summaries for:

- `native_text` vs `ocr_derived`
- `treatment` vs `non_treatment`

At the moment the slice summaries show that:

- the `ocr_derived` slice has recovered to full recall and MRR after the scanned cleanup pass
- the only remaining warning case sits in the treatment-heavy native-text slice

## Limitations

- OCR fallback is more structured than before, but still heuristic and not fully layout-aware
- OCR grouping currently rebuilds coarse paragraph-like blocks rather than true document layout regions
- Chunking is still heuristic rather than fully semantic
- Section detection is improved, but still rule-based and fragile on other document layouts
- Retrieval quality still depends on lightweight heuristics, including rule-based chunk quality labeling
- Multi-evidence answer assembly is improved, but some queries can still pull in secondary epidemiology, prognosis, or treatment-summary details that are relevant-but-not-ideal
- The generalized treatment-evidence heuristics now work across vitamin-C and echinacea cases, but they are still heuristic and not yet validated on a broader treatment corpus
- The remaining keyword-coverage miss is still treatment-heavy, especially where null-effect and subgroup-benefit findings are mixed in the same source block
- Grounded answers are extractive, not LLM-synthesized
- The evaluation set is still small and currently covers only four indexed sample documents
- The scanned benchmark currently uses one locally derived OCR-heavy document rather than a broader scanned-document set
- The richer debug report helps inspection, but the benchmark is still hand-built and not yet broad enough to validate true generalization
- Table extraction and harder layout handling are still deferred
- Multi-document and multilingual robustness are not validated yet

## Notes on Reference Material

This repo was brainstormed with ideas from:

- [DeepLearning.AI Skill Builder](https://skillbuilder.deeplearning.ai/)
- ChatGPT 5.4

Earlier in development, a small set of course notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied into a temporary `references/` folder and used only as design input for:

- OCR fallback planning
- reading-order and layout handling
- schema design
- grounding-aware RAG flow

Those reference notebooks were not kept as part of the final repo structure. The current codebase is a separate local implementation rather than a notebook-derived copy.

## Near-Term Next Steps

- split mixed treatment-summary chunks more cleanly so prevention, null-effect, subgroup-benefit, and duration evidence stop sharing the same answer context
- make chunk boundaries more section-aware inside long review-summary blocks before trying any reranking upgrade
- only prototype lightweight reranking if the chunking-first pass does not remove the last treatment-heavy warning

## Version Log

### 2026-05-06

First working local-first MVP completed.

At this point the project supports:

- native PDF extraction with `PyMuPDF`
- document-level JSON output
- chunk generation with reading-order preservation and lightweight section detection
- local vector indexing with `ChromaDB`
- top-k retrieval from the local index
- adjacent-chunk expansion
- grounded extractive answer assembly with explicit citations
- small local evaluation workflow for retrieval and answer inspection

This version also includes one post-MVP quality pass over chunking, retrieval reranking, and answer heuristics.

### 2026-05-07

`v1.1` robustness and quality work completed across OCR fallback, evaluation, retrieval, and chunking.

This day included:

- real OCR fallback through `pytesseract` for pages flagged by the native-text heuristic
- preservation of OCR provenance in extraction and chunk metadata
- an expanded 7-case local benchmark
- a small heuristic pass to cover new `causes` and `incidence` query types exposed by the broader evaluation set
- widened the candidate pool before heuristic reranking
- improved reranking for `incidence` and `antibiotics` queries
- added stronger penalties for reference-heavy and table-like statistical noise
- paragraph-aware cleanup before chunk assembly
- sentence-aware splitting for oversized segments
- better filtering of boilerplate, TOC-like fragments, and reference-heavy noise
- a broadened gold set for acceptable `antibiotics` summary evidence

By the end of the day, the benchmark had moved to:

- `precision@5 = 0.371`
- `recall@5 = 0.929`
- `MRR = 1.0`
- `avg_keyword_coverage = 1.0`

At this point the main remaining weakness is retrieval recall on some multi-evidence cases, especially `incidence`, rather than missing core pipeline pieces.

### 2026-05-08

`v1.2` was completed and then followed by a first treatment-specific quality pass on the expanded multi-document benchmark.

This day included:

- cleaner splitting of `Key points`-style summary blocks so transmission, incidence, antibiotics, and symptom bullets stop collapsing into the same chunk
- upgrading OCR fallback from a single page-level text blob to coarse paragraph-like OCR blocks
- adding chunk-level noise labels and quality scores
- wiring those labels into retrieval filtering and reranking
- adding intent-aware candidate-pool sizing and neighbor expansion for multi-evidence queries
- expanding evaluation to a second document and adding negative abstention checks
- tuning retrieval and answer selection for vitamin-C prevention, cold-stress, and duration queries

By the end of the pass, the benchmark had moved to:

- `precision@5 = 0.400`
- `recall@5 = 1.0`
- `MRR = 1.0`
- `avg_keyword_coverage = 0.926`
- `negative_success_rate = 1.0`

At this point the main remaining work is not the basic local pipeline anymore, but broader robustness testing on additional treatment-heavy or noisier documents.

### 2026-05-09

`v1.3` and `v1.4` together pushed the project from broader cross-document generalization into a harder scanned-document benchmark, OCR stabilization, better treatment-answer behavior, and clearer evaluation diagnostics.

This day included:

- processing the echinacea meta-analysis as a third local document
- rebuilding the shared index across three documents
- adding new echinacea grounded cases and a new influenza abstention case
- rerunning the benchmark to test how current heuristics generalize beyond the vitamin-C paper
- extending chunk quality labels into neighbor-expansion gating before answer assembly
- generalizing the earlier vitamin-C-specific heuristics into broader treatment-evidence categories
- cleaning OCR-derived text more aggressively before chunk assembly so top/bottom line noise is less likely to survive into chunks
- adding a richer evaluation/debug report with top-k snapshots, expanded-context snapshots, answer previews, and evidence snippets per case
- deriving a noisier image-based PDF from the CT-study source
- extracting and chunking that scanned document through the existing OCR path
- rebuilding the shared local index across four documents
- adding new scanned-document grounded cases and a new negative contrast-agent abstention case
- rerunning the benchmark to surface OCR-derived retrieval failures explicitly
- adding simple slice summaries so native-text and OCR-derived behavior can be compared directly in the saved report
- recovering the scanned CT cases back to full recall and MRR through OCR cleanup and scanned-path retrieval tuning
- tightening treatment-heavy answer selection until the warning set dropped to a single remaining case
- using the expanded benchmark to decide that the next gain should come from stronger chunking before any reranking upgrade

By the end of the pass, the benchmark had moved to:

- `precision@5 = 0.431`
- `recall@5 = 1.0`
- `MRR = 1.0`
- `avg_keyword_coverage = 0.974`
- `negative_success_rate = 1.0`

At this point the scanned OCR slice is stable, the report is easier to inspect, and the remaining gap points more toward chunk-boundary quality than toward retrieval ranking.
