# PDF-to-JSON RAG Pipeline

_This plan was brainstormed with [DeepLearning.AI / Skill Builder](https://skillbuilder.deeplearning.ai/) and ChatGPT 5.4, then narrowed into a local-first MVP._

## 1. Document Processing: PDF to Structured JSON

- Hybrid Extraction:
    - Primary Tool: `PyMuPDF` for speed, coordinates, and metadata.
    - Specialized Tool: `pdfplumber` for complex tables.
    - Fallback: `pytesseract` (OCR) for scanned/image-based PDFs.
- Robust Layout Handling:
    - Multi-Column: Detect columns by clustering text block X-coordinates. Implement a reading order sort (sort by Y, then X) to prevent jumbled text.
    - Header Detection: Avoid brittle rules. Use *relative* font size/weight compared to surrounding body text. Prioritize extracting the document's internal Table of Contents (TOC) when available, as it's the most reliable source.
- Chunking Strategy:
    - Employ semantic chunking over fixed-size chunks to preserve context.
    - Use a recursive approach: split by paragraph, then by sentence.
- Output Format:
    - JSON objects containing the `text` and rich `metadata` (e.g., page number, source document, section header).

## 2. Retrieval & Synthesis: The RAG Core

- Embedding: Use a high-performance open-source model (e.g., `all-MiniLM-L6-v2`).
- Vector Storage: Start with a local, file-based vector store like `FAISS` or `ChromaDB` for portability.
- Handling Multi-Chunk Queries:
    - Initial Retrieval: Use a higher `k` value to retrieve a larger set of candidate chunks.
    - Contextual Expansion: Programmatically retrieve chunks immediately preceding and succeeding the top-k results to build a fuller context.
    - Re-ranking: Use a more powerful cross-encoder model to re-score the expanded chunk set for relevance before passing it to the LLM.
- Synthesis: Use precise prompt engineering to instruct the LLM to synthesize an answer based *only* on the provided context chunks.

## 3. Multi-Stage Evaluation Plan

- Pre-LLM Evaluation (Pipeline Health):
    - Processing Quality: Manually review a sample of JSON outputs for completeness, accuracy, and structural integrity.
    - Retrieval Effectiveness: Create a ground-truth dataset (query-to-relevant-chunk mappings) and measure retrieval using standard metrics: `Precision@k`, `Recall@k`, and `Mean Reciprocal Rank (MRR)`.
- Post-LLM Evaluation (Faithfulness & Groundedness):
    - Primary Method (Human-in-the-loop): Manually compare the final LLM answer against the specific context it was given to verify that every statement is supported and no information is hallucinated.
    - Automated Method (LLM-as-a-Judge): Use a separate LLM with a specific prompt to programmatically score whether the generated answer is faithful to the provided source context.

## 4. MVP Scope

The first working version should stay local-first and deliberately narrow.

- Inputs:
    - Native text PDFs first
    - OCR fallback only for pages with missing or unusable text
- Core extraction stack:
    - `PyMuPDF` for page text, blocks, coordinates, and document metadata
    - `pytesseract` only when text extraction quality is poor
- Output:
    - JSON-first chunk files
    - One normalized document-level JSON file plus chunk-level JSON records
- Retrieval:
    - Open-source embeddings
    - Local vector store (`ChromaDB` or `FAISS`)
- Answering:
    - Query -> retrieve top-k chunks -> expand with adjacent chunks -> grounded response
- Evaluation:
    - Manual JSON quality review
    - Small retrieval benchmark with a hand-built query set

## 5. MVP JSON Schema

Each chunk should have a stable and explicit schema from the beginning.

- `doc_id`
- `chunk_id`
- `source_pdf`
- `text`
- `page_start`
- `page_end`
- `bbox`
- `section_title`
- `section_level`
- `chunk_type`
- `reading_order_index`
- `preceding_chunk_id`
- `following_chunk_id`
- `language`
- `extraction_method`
- `ocr_used`
- `confidence`

The document-level JSON should include:

- `doc_id`
- `source_pdf`
- `page_count`
- `title`
- `toc`
- `detected_language`
- `extraction_summary`
- `chunks`

## 6. MVP Milestones

1. Build a parser that extracts page text, block coordinates, and document metadata with `PyMuPDF`.
2. Add a quality heuristic to detect pages that need OCR fallback.
3. Implement reading-order normalization for multi-column layouts.
4. Define section-aware chunking and write chunk JSON outputs.
5. Store embeddings locally and build a simple retrieval interface.
6. Add adjacent-chunk expansion for better context reconstruction.
7. Write a minimal grounded-answer function that answers only from retrieved chunks.
8. Create a tiny evaluation set and measure retrieval quality on it.

## 6.1. v1.1 Milestones

The first post-MVP iteration should improve robustness without changing the project direction.

1. Implement real OCR fallback for pages flagged by the native-text quality heuristic.
2. Expand the local evaluation set beyond the initial four benchmark queries.
3. Use the expanded benchmark to tighten retrieval and answer heuristics against observed failure modes.
4. Improve section-aware chunking where noisy headings or boilerplate still leak into retrieval, including paragraph-aware cleanup and sentence-aware overflow splitting.
5. Keep the pipeline local-first and inspectable before adding any heavier LLM synthesis layer.

## 6.2. v1.2 Plan

The next iteration should improve recall and structure quality without changing the overall local-first architecture.

1. Split mixed summary blocks more cleanly, especially `Key points`-style chunks that currently mix transmission, incidence, and treatment signals.
2. Make OCR fallback more structure-aware by recovering multiple OCR paragraphs or line groups instead of a single full-page text block.
3. Add chunk-level noise labels or quality flags and use them directly in retrieval filtering and reranking.
4. Improve recall for multi-evidence queries, especially `incidence`, by tuning neighbor expansion depth and intent-aware retrieval behavior.
5. Expand evaluation beyond the current single-document benchmark with at least one second document and a few harder negative or low-evidence queries.


## 7. Explicitly Deferred

The following are useful, but should not block MVP or any early version:

- `pdfplumber` for difficult table extraction
- cross-encoder reranking
- automated LLM-as-a-judge evaluation
- cloud deployment
- multi-document schema extraction
- visual grounding UI
