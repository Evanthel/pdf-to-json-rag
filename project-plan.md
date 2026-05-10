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

## 6.2. v1.2 Milestones

The second post-MVP iteration improved recall, structure quality, and evaluation breadth without changing the overall local-first architecture.

1. Split mixed summary blocks more cleanly, especially `Key points`-style chunks that currently mix transmission, incidence, and treatment signals.
2. Make OCR fallback more structure-aware by recovering multiple OCR paragraphs or line groups instead of a single full-page text block.
3. Add chunk-level noise labels or quality flags and use them directly in retrieval filtering and reranking.
4. Improve recall for multi-evidence queries, especially `incidence`, by tuning neighbor expansion depth and intent-aware retrieval behavior.
5. Expand evaluation beyond the current single-document benchmark with at least one second document and a few harder negative or low-evidence queries.

All five milestones are now complete in the current local implementation. A short follow-up pass also improved treatment-specific retrieval and answer selection for the new vitamin-C document cases without changing the core local-first design.

## 6.3. v1.3 Milestones

The third post-MVP iteration focused on generalization rather than adding heavier model complexity.

Completed scope:

1. Added a third benchmark document built around the echinacea meta-analysis.
2. Extended chunk quality labels into neighbor-expansion gating before answer assembly.
3. Replaced vitamin-C-specific treatment heuristics with more general treatment-evidence categories.
4. Improved the OCR-to-chunk handoff on low-text or scanned pages.
5. Added a richer evaluation/debug report with per-case retrieval and answer snapshots.

All five milestones are now complete in the current local implementation. The next useful work should focus less on core local pipeline pieces and more on harder scanned documents, broader benchmark coverage, and clearer generalization limits.

## 6.4. v1.4 Milestones

The fourth post-MVP iteration focused on noisier inputs, clearer benchmark slices, and deciding what kind of quality work should come next.

Completed scope:

1. Added a locally derived scanned CT-study benchmark document plus new grounded and negative cases.
2. Tightened OCR cleanup, OCR paragraph grouping, and scanned-case retrieval behavior on that benchmark.
3. Added simple benchmark slicing for `native_text` vs `ocr_derived` and `treatment` vs `non_treatment`.
4. Reduced the remaining treatment-heavy warning cases with a targeted answer-selection pass.
5. Used the expanded benchmark to decide that the next gain should come from stronger chunking before any lightweight reranking upgrade.

All five milestones are now complete in the current local implementation. The benchmark spans 17 cases across four indexed documents, and the OCR-derived slice is stable enough that the next iteration can focus on chunk-boundary quality rather than scanned-path recovery.

## 6.5. v1.5 Milestones

The fifth post-MVP iteration stayed chunking-first and used the current benchmark to resolve the last treatment-heavy warning without adding another retrieval-specific complexity layer.

Completed scope:

1. Split mixed treatment-summary chunks more cleanly where prevention, null-effect, subgroup-benefit, and duration evidence previously coexisted in the same chunk.
2. Made chunk boundaries more section-aware inside long review-summary blocks instead of relying mostly on page/block edges plus overflow splitting.
3. Added lightweight chunk-level treatment subtopic cues and persisted them through chunk JSON and index metadata.
4. Re-ran the benchmark after the chunking pass and confirmed that the warning-free state stayed stable without another answer-selection exception.
5. Confirmed that the chunking-first pass did not stall, so no lightweight reranking prototype was needed on the current benchmark.

All five milestones are now complete in the current local implementation. The benchmark remains warning-free across 17 cases, so the next useful step is broader generalization pressure rather than another local optimization loop on the same documents.

## 6.6. v1.6 Milestones

The sixth post-MVP iteration broadened the benchmark enough to make explicit decisions about which deferred items are worth reviving.

Completed scope:

1. Added a fifth benchmark document based on the CMAJ prevention/treatment review and folded it into the shared local index.
2. Expanded the evaluation report with document-family and structure-oriented slices, including review-heavy and summary-bullet cases.
3. Prototyped a lightweight reranking pass after the broader benchmark exposed real ranking failures on new review-summary queries, and compared it against the chunking-first baseline.
4. Added a tiny sampled faithfulness audit and used it to decide that LLM-as-a-judge still does not need to return yet for the current extractive answer path.
5. Revisited the deferred list with the broadened benchmark and kept `pdfplumber`, cross-encoder reranking, cloud deployment, schema extraction, and visual grounding UI out of scope for now.

All five milestones are now complete in the current local implementation. The benchmark now spans 21 cases across five indexed documents, includes richer slices, and records both retrieval-strategy comparison and a small faithfulness-audit decision.

## 6.7. v1.7 Milestones

The seventh post-MVP iteration focused on broadening the benchmark again, cleaning up source-anchored review behavior, and turning deferred-feature decisions into explicit benchmark outputs.

Completed scope:

1. Removed the remaining warning cases, including the legacy `antibiotics` path and the source-anchored `cmaj_zinc_prevention` answer assembly.
2. Added a sixth benchmark document that is layout-hostile enough to stress review-summary mixing and disclaimer/reference-tail noise without pretending it is a true table benchmark.
3. Expanded the evaluation slices so source-anchored review queries and newer review-heavy cases can be separated more clearly in the report.
4. Used the sixth benchmark document to test whether `pdfplumber` should return, and kept it deferred because the new failures were not true table/text extraction misses.
5. Kept cross-encoder reranking, automated LLM-as-a-judge, cloud deployment, multi-document schema extraction, and visual grounding UI deferred, while making those decisions explicit in the evaluation report.

All five milestones are now complete in the current local implementation. The benchmark now spans 25 cases across six indexed documents and is warning-free again, with explicit decision checkpoints for the next deferred features.

## 6.8. v1.8 Milestones

The eighth post-MVP iteration used a genuinely table-heavy technical manual to decide whether table-specific extraction support actually needs to return yet.

Completed scope:

1. Added a seventh benchmark document built from a large table-heavy technical manual instead of another review-style paper.
2. Added source-anchored technical/manual benchmark cases for hypothermia, frostbite-risk guidance, immersion-limit lookup, and a new unsupported technical-query negative case.
3. Expanded benchmark slices to separate source-anchored technical cases, table-heavy behavior, and source-locking outcomes from the existing review-heavy slices.
4. Re-checked whether the new table-heavy document actually justifies bringing `pdfplumber` back, and kept it deferred because the current failures were answer-selection and source-locking issues rather than table extraction misses.
5. Re-checked whether the broader benchmark now justifies cross-encoder reranking or LLM-as-a-judge, and kept both deferred after the final warning-free rerun.

All five milestones are now complete in the current local implementation. The benchmark now spans 30 cases across seven indexed documents, remains warning-free, and includes an explicit table-heavy/manual slice that still does not justify reviving the heavier deferred features.

## 6.9. v1.9 Plan

The next iteration should broaden the benchmark again, but this time with form/grid-heavy pressure rather than another review or prose-heavy document.

1. Add an eighth benchmark document that is more questionnaire-, form-, or grid-heavy than the current technical manual, so table-like pressure is separated from structured-form pressure.
2. Add benchmark cases and slices for form/grid extraction, appendix-heavy sections, numeric lookups, and source-anchored technical answers on that new document.
3. Tighten answer compression for dense technical/table answers so the benchmark-clean path is also shorter and easier to inspect.
4. If the new form/grid benchmark exposes real cell-boundary or field-label extraction misses, prototype a narrow structured-table/form assist path and compare it only on those flagged slices.
5. Revisit cross-encoder reranking or LLM-as-a-judge only if the expanded eight-document benchmark reopens concrete retrieval or faithfulness warnings.


## 7. Explicitly Deferred

The following are useful, but should not block MVP or any early version:

- `pdfplumber` for difficult table extraction
- cross-encoder reranking
- automated LLM-as-a-judge evaluation
- cloud deployment
- multi-document schema extraction
- visual grounding UI
