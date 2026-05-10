# PDF-to-JSON RAG

Local-first PDF-to-JSON RAG pipeline with structured extraction, source-aware retrieval, and an inspectable multi-document evaluation loop.

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

Current version: `v1.8`.

The repo is in a working local-first state: PDF extraction, chunking, local indexing, grounded answering, and benchmark evaluation all run end-to-end across seven indexed sample documents.

The current implementation is still deliberately heuristic-heavy, but `v1.8` is now complete: the benchmark has been broadened with a genuinely table-heavy technical manual, source-anchored technical/table queries, richer slice summaries for source-locking and table-heavy behavior, and an explicit re-check of whether `pdfplumber`, cross-encoder reranking, or LLM-as-a-judge should come back into scope. The current benchmark state is warning-free, with full recall, full keyword coverage, and full abstention on the hand-built benchmark, and the next useful work is broader structured-form/generalization pressure rather than another narrow local fix on the same document set.

## What Works

- Native PDF extraction with `PyMuPDF`
- OCR fallback for pages with weak or missing native text, with paragraph-like OCR block recovery and cleaner OCR-to-chunk handoff instead of a single full-page text blob
- OCR provenance carried through extraction and chunk metadata
- Document-level JSON output in `data/documents/`
- Chunk generation with reading-order preservation, section detection, paragraph-aware cleanup, sentence-aware overflow splitting, cleaner `Key points`-style summary splitting, section-aware review chunk boundaries, chunk-level treatment subtopic cues, and chunk-level noise labels / quality scores
- Local vector indexing with `ChromaDB`
- Retrieval from the local index with intent-aware reranking, noise-label-aware filtering, heuristic noise suppression, quality-gated neighbor expansion, chunk-subtopic-aware treatment retrieval, generalized treatment-evidence retrieval behavior, and source-aware document locking for anchored review and technical/manual queries
- Adjacent-chunk expansion for context reconstruction
- Grounded answer assembly with explicit evidence citations and targeted answer-selection heuristics for treatment prevention, null-effect, subgroup-benefit, duration, and overall-conclusion queries
- multi-document evaluation workflow across seven indexed sample documents, with grounded cases, negative abstention checks, per-case debug snapshots, document-family/layout slices, source-locking and table-heavy slices, lightweight-rerank comparison, a sampled faithfulness audit, and explicit deferred-feature decision checkpoints

## Workflow

1. Extract a PDF into `*.native.json` and `*.document.json`, using OCR fallback when native text is too weak
2. Convert extracted blocks into chunk JSON files with chunk-level provenance
3. Build a persistent local vector index from chunk text and metadata
4. Retrieve top-k chunks for a query
5. Expand with adjacent chunks
6. Assemble a grounded answer from the expanded context
7. Evaluate retrieval and answer quality on a hand-built multi-document benchmark with negative queries, slice summaries, rerank comparison, deferred-feature decision checkpoints, and inspect the saved per-case debug snapshots

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

- `precision@5`: `0.328`
- `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`
- negative case success rate: `1.0`
- warning case count: `0`

This reflects the current 30-case benchmark after `v1.8`. The benchmark now spans seven indexed documents and is broad enough to compare the current lightweight reranking pass against the chunking-first baseline on a mix of section-structured native PDFs, review-heavy treatment documents, OCR-derived evidence, source-anchored review queries, and table-heavy technical-manual queries.

The saved report now also includes:

- per-case debug records with top-k retrieval snapshots, expanded-context snapshots, answer previews, and evidence snippets
- document-family and structure slices such as `cmaj_review`, `wat_review`, `ajmedp_manual`, `source_anchored_review`, `source_anchored_technical`, `source_locked`, `review_summary`, `table_heavy`, `scanned_ct`, and `section_structured`
- a retrieval-strategy comparison between the chunking-first baseline and the current lightweight reranking pass
- a sampled faithfulness audit over selected grounded cases
- explicit deferred-feature decisions for `pdfplumber`, cross-encoder reranking, and `LLM-as-a-judge`

At the moment the report shows that:

- the `ocr_derived` / scanned CT slice remains stable
- the `source_anchored_review` slice is warning-free on the current seven-document benchmark
- the `table_heavy` and `source_anchored_technical` slices are also warning-free on the current benchmark
- the lightweight reranking pass is still sufficient for the current benchmark, so a cross-encoder is not yet justified
- the sampled faithfulness audit does not currently justify bringing back LLM-as-a-judge
- the current seventh table-heavy document still does not justify bringing `pdfplumber` back into scope

## Limitations

- OCR fallback is more structured than before, but still heuristic and not fully layout-aware
- OCR grouping currently rebuilds coarse paragraph-like blocks rather than true document layout regions
- Chunking is still heuristic rather than fully semantic
- Section detection is improved, but still rule-based and fragile on other document layouts
- Retrieval quality still depends on lightweight heuristics, including rule-based chunk quality labeling and a small lexical reranking pass
- Multi-evidence answer assembly is improved, but some grounded answers can still include secondary sentences that are relevant-but-not-ideal once the core evidence has already been captured
- The generalized treatment-evidence heuristics and subtopic cues now work across vitamin-C and echinacea cases, but they are still heuristic and not yet validated on a broader treatment corpus
- Source-aware review handling is now benchmark-clean, but it still needs validation on a broader set of source-anchored review documents
- Source-aware technical/manual handling is now benchmark-clean on one table-heavy manual, but it still needs validation on broader form, appendix, and grid-heavy documents
- Grounded answers are extractive, not LLM-synthesized
- The evaluation set is still small and currently covers only seven indexed sample documents
- The scanned benchmark still uses one locally derived OCR-heavy document rather than a broader scanned-document set
- The richer debug report helps inspection, but the benchmark is still hand-built and not yet broad enough to validate true generalization
- The benchmark now includes one genuinely table-heavy technical manual, but that is still too narrow to justify reviving `pdfplumber` on its own
- The sampled faithfulness audit currently matches the extractive design of the answer layer; it is useful as a checkpoint, but not yet a substitute for broader human review
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

- add an eighth benchmark document that is more form/grid-heavy than the current technical manual, so structured-form pressure is separated from table-heavy pressure
- tighten answer compression on dense source-anchored technical/manual answers so benchmark-clean answers are also shorter to read
- expand the benchmark and audit slices again only if broader evidence reopens concrete cases for `pdfplumber`, cross-encoder reranking, or LLM-as-a-judge

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

`v1.3`, `v1.4`, and `v1.5` together pushed the project from broader cross-document generalization into a harder scanned-document benchmark, OCR stabilization, treatment-answer cleanup, and a chunking-first review pass.

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
- replacing the earlier paragraph splitter with a line-aware paragraph grouping pass that is less likely to fragment wrapped review text
- splitting mixed treatment-summary paragraphs more cleanly, especially where null-effect, subgroup-benefit, duration, and therapeutic findings previously shared the same chunk
- adding more section-aware review chunk boundaries inside long summary-heavy blocks
- persisting lightweight treatment subtopic cues into chunk JSON and index metadata
- using those cues for light treatment-aware retrieval bonuses instead of adding another dedicated reranking stage

By the end of the pass, the benchmark had moved to:

- `precision@5 = 0.354`
- `recall@5 = 1.0`
- `MRR = 1.0`
- `avg_keyword_coverage = 1.0`
- `negative_success_rate = 1.0`
- `warning_case_count = 0`

At this point the benchmark was warning-free again, and the next useful work was no longer local cleanup on the same four documents but broader generalization pressure.

### 2026-05-10

`v1.6`, `v1.7`, and `v1.8` broadened the benchmark enough to turn the deferred-feature discussion into measured decisions instead of guesses.

This day included:

- adding a fifth benchmark document built from the CMAJ prevention/treatment review
- adding a sixth layout-hostile review document based on *The common cold: a review of the literature*
- adding a seventh genuinely table-heavy technical manual based on `AJMedP-4-2_SRD_EDA_V1_E_2561.pdf`
- expanding the benchmark to 30 cases, including new review-summary, source-anchored, technical/table-heavy, and negative cases
- adding richer evaluation slices for document family and structure, including review-heavy and scanned-path views
- prototyping a lightweight reranking pass only after the broader benchmark exposed real ranking failures on the new review-summary queries
- comparing that reranking pass against the chunking-first baseline inside the saved report
- adding a sampled extractive faithfulness audit to decide whether LLM-as-a-judge should come back into scope
- refining review-summary chunking, chunk-quality labels, and source-aware answer assembly enough to stabilize the broader benchmark without bringing `pdfplumber` or cross-encoder reranking back in
- adding source-aware technical/manual locking plus table-heavy query intents for hypothermia, frostbite-risk, and immersion-limit cases
- re-checking whether the table-heavy manual actually justifies `pdfplumber`, and keeping it deferred because the core table content is already being extracted well enough by the current `PyMuPDF`-first path
- adding explicit deferred-feature decisions to the saved report for `pdfplumber`, cross-encoder reranking, and `LLM-as-a-judge`

By the end of the pass, the benchmark stood at:

- `precision@5 = 0.328`
- `recall@5 = 1.0`
- `MRR = 1.0`
- `avg_keyword_coverage = 1.0`
- `negative_success_rate = 1.0`
- `warning_case_count = 0`

At this point the benchmark is broad enough to make three current decisions explicit:

- the lightweight reranking pass remains sufficient on the current seven-document benchmark, so a cross-encoder is still not justified
- `pdfplumber` still stays deferred because even the current seven-document benchmark, including the new table-heavy manual, does not expose a true table/text extraction failure that requires it
- LLM-as-a-judge still stays deferred because the sampled extractive faithfulness audit does not show unsupported-answer drift on the current pipeline
