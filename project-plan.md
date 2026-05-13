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

## 6.9. v1.9 Milestones

The ninth post-MVP iteration separated structured-form pressure from table-heavy/manual pressure and used that broader benchmark to re-check whether any heavier deferred feature really needs to return.

Completed scope:

1. Added an eighth benchmark document built around a health-check questionnaire that is more form/grid-heavy than the existing technical manual.
2. Added benchmark cases and slices for source-anchored form questions, grid-style response options, numeric option lookups, and a late-table / appendix-like follow-up case.
3. Tightened answer compression for dense source-anchored questionnaire/table answers so the benchmark-clean path is shorter and easier to inspect.
4. Prototyped a narrow structured-form assist path for flagged questionnaire/table blocks instead of reviving a broader table-extraction dependency.
5. Re-checked cross-encoder reranking, `pdfplumber`, and LLM-as-a-judge on the expanded eight-document benchmark and kept them deferred after the final warning-free rerun.

All five milestones are now complete in the current local implementation. The benchmark now spans 36 cases across eight indexed documents, remains warning-free, and includes a dedicated `form_grid` / `source_anchored_form` slice that still does not justify reviving the heavier deferred features.

## 6.10. v1.10 Milestones

The tenth post-MVP iteration focused on stress-testing and generalizing structured-form behavior across another appendix/checklist-heavy document family.

Completed scope:

1. Added a ninth benchmark document (`CEP_OpioidManager_Appendix2017.pdf`) to extend the form/grid slice beyond a single questionnaire-style source.
2. Expanded the benchmark with new source-anchored opioid-appendix cases for checklist-field retrieval, adverse-effect scale lookup, follow-up timing extraction, and a new unsupported-entity negative case.
3. Reworked the structured-form assist path into a cleaner per-document-family dispatch, then added a narrow opioid-appendix normalization path instead of another one-off retrieval-only patch.
4. Added intent and source-locking support for the opioid appendix family in retrieval/answering/evaluation so row/field-style queries remain anchored to the correct source and section family.
5. Re-checked deferred items on the broader nine-document benchmark and kept `pdfplumber`, cross-encoder reranking, and LLM-as-a-judge deferred after a warning-free rerun.

All five milestones are now complete in the current local implementation. The benchmark now spans 41 cases across nine indexed documents, remains warning-free, and keeps the heavier deferred features out of scope.

## 6.11. v1.11 Milestones

The eleventh post-MVP iteration focused on making the form/appendix path more maintainable and adding stronger quality gates without introducing heavyweight model components.

Completed scope:

1. Replaced part of the document-specific structured-form rewrites with a reusable pattern layer (`field-row`, `legend-scale`, `follow-up-schedule`) and kept dispatch per document family.
2. Added concise structured answer templates for checklist/legend/follow-up intents so source-anchored form answers are shorter while remaining grounded.
3. Added form-family slice checks (`checklist_fields`, `legend_lookup`, `follow_up_schedule`) with explicit per-slice stability thresholds.
4. Added a deterministic regression suite (`evaluate-regression`) for high-risk source-anchored form cases as a fast pre-check before full benchmark reruns.
5. Re-checked deferred features after the expanded nine-document benchmark and kept `pdfplumber`, cross-encoder reranking, and LLM-as-a-judge deferred.

All five milestones are now complete in the current local implementation. The benchmark remains warning-free at 41 cases across nine indexed documents, and slice-stability checks plus regression gating now provide earlier failure signals than global warning counts alone.

## 6.12. v1.12 Milestones

The twelfth post-MVP iteration focused on reducing rule sprawl, improving inspectability, and checking whether the structured-form path still generalizes to another appendix family.

Completed scope:

1. Moved structured form and appendix intent metadata into a shared declarative config so retrieval and answering read from one source of truth.
2. Added `answer_trace` metadata for structured answers so matched templates, cues, and evidence can be inspected directly in debug output.
3. Added regression shards to the CLI for faster local pre-checks on cross-document, form-grid, source-anchored review, and technical/manual cases.
4. Added an additional appendix-heavy checklist source and generalized the existing structured-form path to cover it without another per-document special case.
5. Re-checked the deferred list and kept `pdfplumber`, cross-encoder reranking, and LLM-as-a-judge out of scope because the current local architecture still resolves the new cases cleanly.

All five milestones are now complete in the current local implementation. The benchmark remained warning-free while the structured-form path became more declarative and easier to inspect.

## 6.13. v1.13 Milestones

The thirteenth post-MVP iteration shifted from single-document question answering toward explicit cross-document behavior.

Completed scope:

1. Added cross-document query intents for source listing and source comparison instead of treating all queries like single-document retrieval.
2. Added multi-source document matching from the query and wired it into retrieval so cross-document intents focus on the intended source families.
3. Tightened cross-document reranking to prefer conclusion, prevention, and null-effect evidence over introduction, methods, or inclusion-criteria chunks.
4. Added a dedicated regression shard for cross-document behavior and used it to stabilize the new multi-file path before rerunning the full benchmark.
5. Expanded the evaluation set with source-listing, comparison, and unsupported-entity cross-document cases, then synchronized the gold targets to the current chunk layout.

All five milestones are now complete in the current local implementation. The benchmark now spans 48 cases across 11 indexed documents and remains warning-free while covering both single-document and cross-document grounded behavior.

## 6.14. v1.14 Plan

The next iteration should push the project toward a more domain-agnostic document pipeline instead of continuing to grow only the medical benchmark.

1. Add at least one clearly non-medical benchmark document family and verify that current chunking, retrieval, and answering rules do not silently depend on medical vocabulary.
2. Separate document-family matching from domain vocabulary more cleanly so source-aware behavior can scale beyond the current review/manual/checklist benchmark set.
3. Add a lightweight document-level answer mode for questions like “what does this file cover?” or “which file is most relevant for X?” that sits between source listing and detailed evidence assembly.
4. Expand evaluation so cross-document cases can score at the document level when the user intent is source discovery rather than exact chunk retrieval.
5. Re-check deferred items only if the broader, more domain-diverse benchmark exposes failures that the current chunking-first plus lightweight-rerank architecture cannot absorb.


## 7. Explicitly Deferred

The following are useful, but should not block MVP or any early version:

- `pdfplumber` for difficult table extraction
- cross-encoder reranking
- automated LLM-as-a-judge evaluation
- cloud deployment
- multi-document schema extraction
- visual grounding UI
