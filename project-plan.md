# PDF-to-JSON RAG Pipeline

_This plan was brainstormed with [DeepLearning.AI / Skill Builder](https://skillbuilder.deeplearning.ai/) and ChatGPT 5.4, then narrowed into a local-first MVP._

Internal development iterations in this plan use `v1.x` labels. Public releases follow semantic versioning starting at `0.1.0-beta`.

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

The first working version stayed local-first and deliberately narrow:

- native-text PDFs first, OCR fallback only where needed
- `PyMuPDF` plus `pytesseract`
- document-level and chunk-level JSON outputs
- local embeddings and local vector store
- grounded answers from retrieved local context
- a small hand-built retrieval benchmark

## 5. Current Baseline

Current public version: `0.1.0-beta`
Current internal implementation level: `v0.7.0`

What the repo now has:

- extraction-time sections with `section_path` and `section_kind`
- structure-aware chunking and chunk metadata
- feature-based query planning
- explicit `document_selection` traces
- distinct document-level type / purpose / audience / overview answers
- confidence-aware document classification answers and semantic confidence signals
- explicit classification-rationale and classification-limits answers
- compact default JSON answers with richer debug output behind `--verbose`
- public install/release checks through `doctor`, `smoke-check`, `package-check`, and `release-check`
- local semantic sanity checks for unfamiliar PDFs through `layout-sanity-check`
- a local corpus sampler over repo-local `pdf/` artifacts through `corpus-sanity-check`

Current validation snapshot:

- public release gates: green
- maintainer regression gates: green
- full 77-case benchmark: green
  - `precision@5 = 0.6031`
  - `recall@5 = 1.0`
  - `MRR = 1.0`
  - `avg_keyword_coverage = 1.0`
  - `negative_success_rate = 1.0`
  - `warning_case_count = 0`
- sampled faithfulness audit: green
  - `avg_supported_sentence_ratio = 1.0`
  - `failing_case_count = 0`

Detailed implementation history lives in [DEVELOPMENT_LOG.md](./DEVELOPMENT_LOG.md).

## 6. Current Roadmap

### 6.1. Recently Completed

- `v0.4.0-v0.4.3`
  - stronger section reconstruction
  - richer structure-aware chunk metadata
  - new `section_reconstruction_core` and `document_selection_core` shards
- `v0.4.4+`
  - lower maintenance cost in the document-level path
  - more shared selection and answer-assembly logic
  - no learned reranker added
- `v0.4.6`
  - split document-level rendering into smaller shared mode-specific helpers
  - kept the public answer contract stable while reducing internal branching
- `v0.4.7-v0.4.9`
  - preserved document-root structure context for inline and synthetic section splits
  - simplified structured-form answer rendering into shared helper families
  - added `structured_form_maintenance_core` to the maintainer regression gate
- `v0.5.0-v0.5.4`
  - added document and chunk `structure_confidence` / `layout_confidence` metadata
  - hardened single-document overview fallback on top of the current document-selection contract
  - added `layout_robustness_core` and `single_doc_random_pdf_core` to the maintainer regression gate
- `v0.5.5-v0.5.9`
  - improved table-like and form-heavy chunk splitting on more unfamiliar layouts
  - added conservative single-document wording when recovered structure is limited
  - added `table_layout_robustness_core` and `form_layout_robustness_core` to the maintainer regression gate
  - added a local-only `layout-sanity-check` path for unfamiliar external PDFs without hardcoding private files into the benchmark
- `v0.6.0-v0.6.4`
  - improved document typing and purpose inference for unfamiliar financial/admin forms
  - split document-level answers into clearer type, purpose, audience, and overview render paths
  - added `semantic_document_understanding_core` to the maintainer regression gate
  - extended local `layout-sanity-check` to report overview, type, purpose, and audience answers for unfamiliar PDFs
- `v0.6.5-v0.6.8`
  - added semantic confidence signals and confidence-aware document classification answers
  - added `confidence_aware_document_core` to the maintainer regression gate
  - extended local `layout-sanity-check` to report confidence answers and semantic-confidence metadata for unfamiliar PDFs
- `v0.6.9-v0.7.0`
  - added classification-rationale and classification-limits answers to the document-level trust layer
  - added `trust_policy_document_core` to the maintainer regression gate
  - added a local-only `corpus-sanity-check` path that samples repo-local `pdf/` artifacts through `lcwa_gov_pdf_metadata.csv`

### 6.2. Next Steps

1. Keep improving unknown-document semantic understanding and trust signalling before considering heavier retrieval changes.
2. Use the local `pdf/` corpus more systematically to find failure slices beyond the curated benchmark.
3. Keep evaluation focused on architecture gates and unknown-document sanity checks rather than benchmark growth for its own sake.
4. Revisit a learned reranker only if the current heuristic baseline stops passing release gates and unknown-document sanity gates.

## 7. Explicitly Deferred

- `pdfplumber` for difficult table extraction
- cross-encoder reranking
- automated LLM-as-a-judge evaluation
- cloud deployment
- multi-document schema extraction
- visual grounding UI
