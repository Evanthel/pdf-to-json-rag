# WORK_LOG

## Purpose

This file tracks implementation progress for the `pdf-to-json-rag` working area at a practical level.

It is intentionally high-level and does not try to mirror every code edit line by line.

## Current Direction

The project direction was narrowed to a local-first MVP for a `PDF-to-JSON` RAG pipeline.

The chosen MVP focus is:

- native-text PDFs first
- OCR only as fallback
- JSON-first document representation
- local vector retrieval
- grounded answering from retrieved chunks

Several more advanced ideas were deliberately deferred for later stages, including heavier table handling, reranking, and automated evaluation with a judge model.

## Reference Material Used

During the early planning phase, a small set of notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied locally as temporary reference material.

They were used only to inform:

- OCR fallback planning
- reading-order and layout handling
- schema design
- grounding-aware RAG flow

The implementation in this repo is not a notebook copy. The temporary reference folder was used as design input and then removed from the active repo structure.

## Planning Work Completed

The original project idea was converted into a more concrete implementation plan.

The current plan already includes:

- a narrowed MVP scope
- a first-pass JSON schema for documents and chunks
- milestone ordering
- a list of explicitly deferred features

## Scaffold Created

A minimal project scaffold was created under:

- `pdf-to-json-rag/src/pdf_to_json_rag/`

It includes the core pipeline layers:

- extraction
- chunking
- indexing
- retrieval
- grounded answering
- evaluation
- local configuration and data paths

Basic data directories were also created for:

- input PDFs
- document-level JSON outputs
- chunk-level JSON outputs
- local index artifacts
- evaluation assets

## Implementation Progress

### MVP Step 1

Native PDF extraction with `PyMuPDF` has been implemented as the first real working step.

The current extraction stage already supports:

- opening a PDF locally
- extracting native text blocks
- preserving page-level and block-level structure
- generating normalized bounding boxes
- building document-level metadata needed for later JSON output
- marking pages that may require OCR fallback later

### MVP Step 2

Document artifacts are now written to JSON in:

- `pdf-to-json-rag/data/documents/`

The current flow saves two files per processed PDF:

- `*.native.json` for the raw native extraction result
- `*.document.json` for the normalized document-level record used by later stages

This gives the project a stable checkpoint between extraction and chunking.

### MVP Step 3

Chunk generation from extracted blocks is now implemented.

The current MVP chunking flow:

- loads saved native extraction blocks
- normalizes reading order
- uses a lightweight header heuristic to track section titles
- groups nearby text blocks into chunk-sized units
- writes one JSON file per chunk to `data/chunks/<doc_id>/`
- links adjacent chunks with previous / next IDs

The document-level JSON is also rewritten so it contains embedded chunk metadata.

### MVP Step 4

Local vector indexing for chunk JSON outputs is now implemented.

The current indexing flow:

- loads chunk JSON files from `data/chunks/<doc_id>/`
- builds embeddings locally
- stores vectors, texts, and metadata in a persistent `ChromaDB` index
- writes an index manifest to `data/index/index_manifest.json`

The implementation also includes a deterministic local hash-based fallback embedder.

That fallback keeps the pipeline runnable offline when the main embedding model is not available locally.

### MVP Step 5

Retrieval from the local index is now implemented.

The current retrieval flow:

- loads the saved index manifest
- loads the matching embedder for the current index
- embeds the query locally
- queries the persistent `ChromaDB` collection
- reconstructs typed `ChunkRecord` hits from returned texts and metadata

The indexing step now also removes stale UUID segment folders after rebuilds, so `data/index/` stays cleaner across repeated local runs.

### MVP Step 6

Adjacent-chunk expansion on top of retrieval is now implemented.

The current expansion flow:

- takes the top-k retrieved chunk hits
- loads saved chunk JSON records for the relevant documents
- adds preceding and following chunks when available
- returns the expanded context in reading order

This gives the next answering stage a wider but still grounded local context window.

### MVP Step 7

Grounded answer assembly on top of expanded retrieval is now implemented.

The current answering flow:

- retrieves top-k chunks from the local index
- expands them with adjacent chunks
- scores candidate sentences against the query
- assembles a deterministic answer only from selected evidence sentences
- formats the answer with explicit chunk and page citations

This is still a lightweight extractive MVP, not a generative answer model, but it already produces grounded, inspectable outputs.

### MVP Step 8

A small local evaluation workflow is now implemented.

The current evaluation flow:

- uses a small hand-built evaluation set stored in `data/eval/mvp_eval_cases.json`
- runs retrieval metrics on each case
- runs the grounded answer path on each case
- checks simple keyword coverage in the resulting answers
- writes a structured report to `data/eval/mvp_eval_report.json`

This is intentionally lightweight, but it already makes it easier to spot retrieval and answer-quality regressions while iterating on chunking and heuristics.

## Verification Completed

### Import / Scaffold Checks

The initial scaffold was smoke-tested to confirm that:

- imports work
- core schemas load correctly
- the project paths resolve correctly

### Native Extraction Smoke Test

Native extraction was smoke-tested successfully on:

- `medical/Common_cold_clinincal_evidence.pdf`

Verified at a high level:

- document ID generation works
- page counting works
- title detection returned a usable value
- native text blocks were extracted
- no OCR fallback was needed for that sample file

### JSON Output Smoke Test

The extraction-to-JSON path was also smoke-tested successfully on:

- `medical/Common_cold_clinincal_evidence.pdf`

Verified at a high level:

- both output JSON files were created
- the native artifact includes pages, blocks, metadata, and reading-order fields
- the document artifact includes document metadata and extraction summary
- outputs were written to `data/documents/` as expected

### Chunking Smoke Test

Chunk generation was smoke-tested successfully on:

- `medical/Common_cold_clinincal_evidence.pdf`

Verified at a high level:

- chunk generation completed from saved extraction artifacts
- chunk JSON files were written to `data/chunks/common-cold-clinincal-evidence/`
- the document-level JSON was updated to include chunk metadata
- adjacent chunk linking fields were populated

### Local Index Smoke Test

Local vector indexing was smoke-tested successfully on:

- chunks generated from `medical/Common_cold_clinincal_evidence.pdf`

Verified at a high level:

- a persistent ChromaDB index was created under `data/index/`
- an `index_manifest.json` file was written
- the final index build succeeded with `sentence-transformers`
- the active embedding model is `all-MiniLM-L6-v2`

### Retrieval Smoke Test

Retrieval was smoke-tested successfully with the query:

- `common cold symptoms`

Verified at a high level:

- top-k retrieval returned grounded chunk hits from the indexed document
- returned hits were reconstructed as typed chunk records
- result previews aligned with expected common-cold content
- retrieval worked against the cleaned persistent local index

### Adjacent-Chunk Expansion Smoke Test

Expanded retrieval was smoke-tested successfully with the query:

- `common cold symptoms`

Verified at a high level:

- 5 top-k hits expanded to 9 context chunks
- neighbor expansion pulled in surrounding chunks on both sides where available
- the expanded output stayed in document reading order
- the expanded context looked suitable for grounded answer assembly

### Grounded Answer Smoke Test

Grounded answer assembly was smoke-tested successfully with the query:

- `What are common cold symptoms?`

Verified at a high level:

- the answer was assembled only from retrieved evidence sentences
- the final answer included explicit chunk and page citations
- the selected evidence reflected symptom content rather than treatment-effect snippets
- the output looked suitable as an MVP grounded answer path

### MVP Evaluation Smoke Test

The full evaluation workflow was smoke-tested successfully on:

- `data/eval/mvp_eval_cases.json`

Verified at a high level:

- the evaluation report was written to `data/eval/mvp_eval_report.json`
- retrieval metrics were computed for all four benchmark cases
- answer keyword coverage was computed for all four benchmark cases
- the report already surfaces a weaker definition case, which is useful for the next iteration cycle

Current summary metrics from the smoke test:

- average `precision@5`: `0.35`
- average `recall@5`: `1.0`
- `MRR`: `0.6875`
- average answer keyword coverage: `0.7083`

## Post-MVP Quality Iteration

After the first full MVP pass, the next iteration focused on quality rather than adding new modules.

The main changes were:

- improved section detection for inline labels such as `DEFINITION`, `AETIOLOGY/ RISK FACTORS`, and `PROGNOSIS`
- refreshed chunk outputs and local index after the chunking change
- tightened sentence-selection heuristics for definition and transmission queries
- added lightweight retrieval-side query augmentation and section-aware reranking
- updated the local evaluation set to match the new chunk IDs after re-chunking

This materially improved the weakest early cases, especially the definition query.

### Updated Quality Check

After the quality iteration, the local evaluation workflow was rerun.

Updated summary metrics:

- average `precision@5`: `0.35`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`

The most visible improvement was that:

- the definition query now surfaces the `DEFINITION` chunk first
- the transmission query is anchored in the `AETIOLOGY/ RISK FACTORS` chunk
- grounded answers are more section-aligned and less polluted by obvious noise sections

## Current State

What is already in place:

- project folder
- MVP plan
- scaffold
- native extraction step
- document-level JSON writing
- chunk generation and chunk JSON writing
- local vector indexing
- local retrieval from the persistent index
- adjacent-chunk expansion on retrieved hits
- grounded answer assembly from expanded retrieval
- a small local evaluation workflow with saved reports
- one post-MVP quality tuning pass over chunking, retrieval, and answer heuristics

What is not yet done:

- broader evaluation coverage beyond the first small benchmark set

## Next Suggested Step

The next sensible implementation step is:

- expand the evaluation set beyond the initial benchmark queries

After that:

- decide whether to keep the extractive answer path or swap in an LLM synthesis stage
