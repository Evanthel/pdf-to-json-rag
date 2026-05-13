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

## v1.1 Robustness Pass

The next iteration focused on two things:

- expanding the benchmark beyond the first tiny eval set
- replacing OCR detection-only logic with a real fallback path

### OCR Fallback Update

The extraction stage now performs real OCR fallback with `pytesseract` for pages whose native text is too weak.

This iteration added:

- page rendering from PDF to image for OCR processing
- fallback OCR text extraction for image-only or low-text pages
- explicit provenance so extracted blocks and downstream chunks can be marked as `native`, `ocr`, or `mixed`
- separate tracking for pages that merely triggered the heuristic versus pages that were actually processed with OCR

The current OCR path is intentionally simple: it restores page text, but does not yet rebuild a detailed OCR block layout.

### Expanded Evaluation Set

The local benchmark was expanded from 4 to 7 cases.

The added cases cover:

- causes / viral aetiology
- yearly incidence in children and adults
- antibiotics as a treatment-oriented question

This made the benchmark meaningfully harder and more useful as a regression check.

### Heuristic Follow-Up

Once the benchmark was expanded, a new failure mode appeared on `causes` queries.

To address that, a small follow-up pass added:

- intent detection for `causes`
- intent detection for `incidence`
- retrieval-side query augmentation for those intents
- answer-side sentence scoring tuned to prefer `AETIOLOGY/ RISK FACTORS` and `INCIDENCE/ PREVALENCE` evidence

### OCR Smoke Test

Real OCR fallback was smoke-tested successfully on a synthetic image-only PDF page created locally for validation.

Verified at a high level:

- the page triggered the OCR heuristic
- OCR was actually applied
- the first extracted block was marked with extraction method `ocr`
- recovered text included the expected symptom wording from the synthetic page

### Expanded Evaluation Smoke Test

The full evaluation workflow was rerun on the expanded 7-case benchmark.

Updated summary metrics:

- average `precision@5`: `0.314`
- average `recall@5`: `0.929`
- `MRR`: `0.929`
- average answer keyword coverage: `1.0`

Interpretation:

- answer coverage is currently strong across the expanded benchmark
- retrieval still finds a relevant chunk first for most cases, but not all
- the main remaining weakness is ranking precision on harder retrieval cases such as `incidence` and `antibiotics`

## Retrieval Precision Follow-Up

After the first v1.1 evaluation pass, the next iteration focused specifically on improving retrieval precision for the harder `incidence` and `antibiotics` cases.

### Retrieval Update

This iteration changed the retrieval path in three practical ways:

- widened the candidate pool fetched from the vector store before reranking
- added stronger intent-aware reranking for `incidence` and `antibiotics`
- added stronger penalties for references, table-like statistical fragments, and other citation-heavy noise

This was deliberately smaller than a reranker-model change. The goal was to improve ranking quality without leaving the current local-first and inspectable setup.

### Retrieval Quality Check

The updated retrieval behavior was checked directly on:

- `How many colds do children and adults get each year?`
- `Do antibiotics help with the common cold?`

Observed improvements:

- the `incidence` query now surfaces the key incidence chunks first, including the yearly-frequency summary chunk
- the `antibiotics` query now surfaces the main antibiotics option chunk first, with much less bibliography-style noise in top results

### Updated Evaluation Check

The expanded 7-case benchmark was rerun after the retrieval precision pass.

Updated summary metrics:

- average `precision@5`: `0.343`
- average `recall@5`: `1.0`
- `MRR`: `0.929`
- average answer keyword coverage: `1.0`

Interpretation:

- retrieval precision improved from the earlier v1.1 pass
- all current benchmark cases now retrieve at least one relevant chunk within top-5
- the main remaining ambiguity is in the `antibiotics` case, where a clinically relevant summary chunk can surface before the currently hand-marked gold chunk

## Chunking And Noise-Handling Follow-Up

The next iteration focused on improving chunk boundaries and reducing obvious retrieval noise before adding any heavier model components.

### Chunking Update

This pass added:

- paragraph-aware cleanup before chunk assembly
- sentence-aware splitting for oversized text segments
- stronger filtering of boilerplate and repeated publication fragments
- more aggressive suppression of TOC-like and bibliography-like content before it becomes a chunk

The practical goal was not to make chunking fully semantic yet, but to stop carrying obvious junk into the index.

### Evaluation Set Update

The benchmark was also updated to reflect the current chunk layout after the chunking refactor.

In addition:

- the `antibiotics` gold set was broadened to include acceptable summary-style evidence chunks, not only the most canonical intervention chunk

This makes the benchmark better aligned with the current grounded answer path.

### Retrieval Noise Handling Update

On top of the chunking changes, retrieval penalties were tightened further for:

- disclaimer-like chunks
- question / TOC-like chunks
- bibliography / citation-heavy chunks
- table-like statistical fragments with section titles such as `Population`, `RR`, or `P = ...`

### Updated Evaluation Check

After the chunking and noise-handling pass, the full 7-case benchmark was rerun.

Updated summary metrics:

- average `precision@5`: `0.371`
- average `recall@5`: `0.929`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`

Interpretation:

- relevant chunk ranking is now stronger at position 1 across the full current benchmark
- precision improved again compared with the earlier v1.1 retrieval-only pass
- the remaining gap is recall on some multi-evidence cases, especially `incidence`, not answer coverage

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
- a local evaluation workflow with an expanded 7-case benchmark
- one post-MVP quality tuning pass over chunking, retrieval, and answer heuristics
- real OCR fallback for low-text pages
- extraction/chunk provenance for `native`, `ocr`, and `mixed` content
- paragraph-aware and sentence-aware chunk cleanup before indexing

## Early v1.2 Chunk-Boundary Pass

The next iteration moved from general robustness into cleaner structure handling for mixed summary evidence.

This pass focused on the first `v1.2` task:

- splitting `Key points`-style summary material into cleaner single-topic chunks
- reducing the amount of TOC-like summary debris that could still leak into early treatment chunks
- re-aligning the local benchmark with the new chunk boundaries after the split

### Chunking Update

The summary cleanup pass added:

- a dedicated `Key points` summary mode in chunking
- bullet-aware boundaries so adjacent bullets no longer collapse into one mixed summary chunk
- better suppression of dotted TOC-leader fragments
- filtering of short TOC-like headings and standalone page-number fragments before chunk assembly

The practical effect is that early-page summary evidence is now separated more cleanly across:

- transmission / incidence
- decongestants
- antibiotics
- vitamin C / duration-related treatment notes
- antihistamines

instead of mixing several of those signals into the same chunk.

### Benchmark Realignment

Because the summary split changed the early chunk boundaries, the 7-case local benchmark was also updated to match the current chunk IDs.

This was a structural remap, not a benchmark expansion. The goal was to keep regression checks meaningful after the re-chunking pass.

### Updated Evaluation Check

After rebuilding the chunks and local index, the full 7-case benchmark was rerun.

Updated summary metrics:

- average `precision@5`: `0.400`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`

Interpretation:

- the current benchmark now reflects the post-split chunk layout correctly
- top-1 retrieval remains strong across the current eval set
- the `Key points` cleanup improved structural separation without hurting answer coverage
- the next meaningful gains will likely come from structure-aware OCR, chunk-level noise labels, and better multi-evidence recall rather than another broad heuristic pass

## Early v1.2 OCR Structure Pass

The next iteration focused on the second `v1.2` task:

- making OCR fallback more structure-aware than a single full-page text block

### OCR Update

The OCR fallback path in extraction now:

- uses `pytesseract.image_to_data(...)` rather than only `image_to_string(...)`
- rebuilds OCR output into multiple coarse blocks with per-block bounding boxes
- groups OCR lines globally on the page using vertical-gap and column-alignment heuristics
- falls back to a single OCR text blob only if structured OCR extraction fails

The practical effect is that low-text or image-only pages no longer lose all internal structure at the extraction stage.

### OCR Smoke Tests

The updated OCR path was smoke-tested on a synthetic image-only PDF page created locally.

Verified at a high level:

- the page triggered OCR fallback
- OCR was actually applied
- the fallback returned multiple OCR blocks rather than a single page blob
- the resulting OCR blocks preserved paragraph-like separation
- downstream chunking still marked the content with extraction method `ocr`

Observed synthetic smoke-test result:

- `pages_requiring_ocr = 1`
- `pages_processed_with_ocr = 1`
- `ocr_blocks = 2`
- downstream `chunk_method = ocr`

### Interpretation

This does not make OCR fully layout-aware yet, but it moves the pipeline to a better intermediate state:

- OCR pages now preserve coarse local structure
- chunking can operate on multiple OCR-derived segments instead of one giant fallback block
- the next structural gains are more likely to come from noise labels and retrieval filtering than from another immediate OCR rewrite

## Early v1.2 Chunk-Quality Label Pass

The next iteration focused on the third `v1.2` task:

- adding chunk-level noise labels / quality flags and using them directly in retrieval filtering and reranking

### Quality Label Update

This pass added explicit chunk quality metadata:

- `noise_labels`
- `quality_score`

The labels are now assigned during chunk creation and persisted in:

- chunk JSON files
- document-level embedded chunk metadata
- vector index metadata

The current labeling is still heuristic, but it gives the retrieval layer a stable quality signal instead of forcing every path to rediscover the same chunk problems from raw text each time.

### Retrieval Update

Retrieval now uses these labels in two ways:

- hard filtering for the most obviously bad chunk classes, such as disclaimer-like and table/statistical section artifacts
- softer reranking penalties for bibliography, TOC-like, boilerplate, and other lower-signal chunk classes

This does not remove all retrieval noise, but it makes the retrieval path more inspectable and more modular than the earlier regex-only heuristics.

### Verification

The pass was checked at a high level by:

- rebuilding chunk JSON outputs for `common-cold-clinincal-evidence`
- rebuilding the local vector index
- inspecting saved noise labels on representative chunks
- rerunning the 7-case benchmark

Representative examples after relabeling:

- early treatment-summary chunk `0003` stayed clean with no noise labels
- abstract-like chunk `0001` was marked with bibliography/statistical-style noise
- GRADE/statistical tail chunks were penalized or filtered more explicitly through labels

### Updated Evaluation Check

After the chunk-quality label pass, the 7-case benchmark was rerun.

Summary metrics stayed at:

- average `precision@5`: `0.400`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`

Interpretation:

- the main value of this pass is not a large metric jump
- instead, the retrieval pipeline now has a cleaner and more explicit internal quality signal
- that should make the next recall-focused iteration easier to control, especially for multi-evidence cases and harder future documents

## Early v1.2 Multi-Evidence Retrieval Pass

The next iteration focused on the fourth `v1.2` task:

- improving recall behavior for multi-evidence queries through intent-aware retrieval and neighbor expansion

### Retrieval Update

This pass changed retrieval in two practical ways:

- candidate-pool size is now chosen per query intent instead of using one fixed multiplier for every query
- neighbor expansion is now intent-aware, with different expansion depth and neighbor acceptance rules for different query types

The main target was `incidence`, where multi-evidence queries benefit from recovering both prevalence-style and summary-style chunks without opening the same amount of surrounding context for every other query type.

### Answering Follow-Up

Because broader retrieval context can also surface more adjacent noise, the answer layer got a small matching cleanup for `incidence`:

- additional penalties for cohort-history and cross-sectional-statistic sentences that are related but not central to the annual-frequency question

This was deliberately small and local. The goal was not to redesign answer synthesis, only to make the wider context more usable.

### Verification

The pass was checked with:

- direct `answer-query` inspection for `How many colds do children and adults get each year?`
- direct `retrieve-expanded` inspection for the same query
- a rerun of the full 7-case benchmark

Observed effects:

- expanded context for `incidence` became smaller and more targeted than the earlier one-size-fits-all expansion path
- the retrieval/answer pipeline now uses a more explicit query-type policy for multi-evidence questions
- benchmark metrics remained stable while the internal retrieval behavior became easier to control

### Updated Evaluation Check

After the multi-evidence retrieval pass, the 7-case benchmark was rerun.

Summary metrics stayed at:

- average `precision@5`: `0.400`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`

Interpretation:

- this pass did not aim for a headline metric jump on the current small benchmark
- its main value is architectural: multi-evidence retrieval is no longer driven by a single generic expansion policy
- that should matter more when evaluation broadens to new documents and harder low-signal queries

## Early v1.2 Multi-Document Evaluation Pass

The next iteration focused on the fifth `v1.2` task:

- expanding evaluation beyond the first single-document benchmark

### Evaluation Expansion Update

This pass added a second local sample PDF to the working evaluation setup:

- `Vitamin_C_for_Preventing_and_Treating_the_Common_Cold.pdf`

The local workflow now supports building one combined index across multiple chunk directories, rather than assuming a single-document benchmark.

The benchmark itself was expanded in two directions:

- grounded cross-document cases that should retrieve the narrower vitamin-C paper rather than only the broad common-cold review
- negative / unsupported queries that should trigger abstention rather than a fabricated grounded answer

### Evaluation Structure Update

The evaluation set now distinguishes between:

- `grounded` cases
- `negative` cases

Negative cases are excluded from classical `precision@k` / `recall@k` / `MRR` averaging and instead tracked through a separate abstention success rate.

### Answering Follow-Up

To make the negative cases meaningful, the answer layer got a lightweight abstention path.

It now returns:

- `No grounded answer could be assembled from the retrieved context.`

when the retrieved evidence does not cover any sufficiently specific query terms beyond generic common-cold boilerplate.

This also exposed one false-abstention regression on the `causes` query, which was then corrected with intent-aware support terms.

### Verification

The evaluation expansion pass was checked by:

- processing a second PDF into document JSON and chunks
- rebuilding one combined local index across both sample documents
- testing vitamin-C-specific retrieval queries directly
- testing negative questions such as vaccine / insulin queries
- rerunning the full evaluation benchmark

Observed behavior:

- negative queries now abstain instead of returning fabricated grounded answers
- the benchmark is materially harder than the earlier single-document setup
- the second document exposes real retrieval and answer-selection weaknesses that the first benchmark did not show

### Updated Evaluation Check

After the multi-document evaluation pass, the benchmark grew from 7 to 11 cases.

Updated summary metrics:

- average `precision@5`: `0.400`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.926`
- negative-case success rate: `1.0`

Interpretation:

- the broader benchmark is doing what it should: it keeps the system from looking artificially complete while still exposing real cross-document weaknesses
- retrieval still recovers all currently marked gold evidence within top-5 on grounded cases
- negative abstention behavior is stable on the current two unsupported test questions
- treatment-specific retrieval / answer quality on the vitamin-C document improved enough to restore a clean top-1 benchmark picture
- stronger suppression of boilerplate, TOC-like, and bibliography-like noise is now part of the retrieval path rather than only an ad hoc benchmark fix

### Treatment-Specific Follow-Up

After the second document exposed weaker treatment evidence handling, a targeted follow-up pass was added for vitamin-C questions.

That pass included:

- treatment-specific retrieval intent detection for vitamin-C prevention, cold-stress, and duration queries
- query augmentation tuned to the actual language used in the vitamin-C review
- tighter answer-sentence selection so generic paper-title or citation-like lines stop winning over treatment findings

Verified at a high level:

- the `normal populations` case now surfaces the correct prevention evidence first
- the `cold stress` case now surfaces the subgroup reduction evidence much more cleanly
- the duration/prophylaxis case now favors the duration finding instead of broader surrounding treatment context
- negative abstention behavior stayed intact after the treatment-specific tuning

What is not yet done:

- validation that current treatment-specific heuristics generalize beyond the vitamin-C review
- more semantic chunk boundary logic beyond the current heuristic cleanup

## v1.3 Task 1

The first `v1.3` step was to broaden the benchmark again with a third document and additional evaluation cases.

For this pass, the chosen third document was:

- `Evaluation_of_echinacea_for_the_prevention_and_treatment_of_the_common_cold.pdf`

The goal was not just to add more volume, but to test whether the current treatment-heavy retrieval and answer heuristics still behave sensibly once another review paper shares the same local index.

### Benchmark Expansion Work

This pass included:

- extracting and chunking the echinacea meta-analysis into the local JSON pipeline
- rebuilding the shared local index across all three current documents
- adding new grounded echinacea cases
- adding a new negative abstention case for influenza

### Verification

This pass was checked by:

- direct retrieval tests for echinacea incidence and overall-conclusion queries
- direct answer-path checks for the same queries
- a negative abstention check on `Does echinacea prevent influenza?`
- a full benchmark rerun after updating the eval set

Observed behavior:

- the third document integrated cleanly into the local index
- the new influenza case abstained correctly
- top-k retrieval stayed strong on the new echinacea queries
- answer keyword coverage dropped even though retrieval metrics stayed high, which is a useful signal rather than a regression to hide

### Updated Evaluation Check

After the third-document expansion, the benchmark grew from 11 to 14 cases.

Updated summary metrics:

- average `precision@5`: `0.400`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.879`
- negative-case success rate: `1.0`

Interpretation:

- retrieval still looks very strong on the current benchmark
- the broadened benchmark is now better at exposing cross-document answer-selection weakness
- the main signal from this pass is not top-k failure, but that treatment-heavy multi-document answering remains more fragile than retrieval alone suggests

## v1.3 Task 2

The second `v1.3` step was to make chunk-quality labeling affect neighbor expansion directly, not only top-k reranking.

This pass focused on the point where low-signal adjacent chunks can still leak into answer assembly even when the primary retrieval set already looks clean.

### Expansion-Gating Work

This pass included:

- defining a separate set of expansion-block labels for obvious low-signal neighbor content
- gating neighbor expansion on chunk quality score as well as noise labels
- requiring weakly matched neighbors to either match the current query intent or stay in the same local section as the anchor chunk

### Verification

This pass was checked by:

- rerunning the full 14-case benchmark
- spot-checking expanded answer context on the new echinacea benchmark queries

Observed behavior:

- the answer context is now filtered through a more explicit quality-control step before assembly
- benchmark headline metrics stayed stable, which is acceptable for this pass
- the main value of the change is architectural: the answer layer no longer depends only on top-k cleanup to avoid obvious low-signal neighbors

### Updated Evaluation Check

After the expansion-gating pass, the benchmark remained at:

- average `precision@5`: `0.400`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.879`
- negative-case success rate: `1.0`

Interpretation:

- this was a control-quality pass more than a headline-metrics pass
- the repo now has a cleaner separation between top-k retrieval filtering and downstream context-expansion filtering
- this gives a better base for the next step, which is to generalize treatment-specific heuristics instead of only adding more query-time exceptions

## v1.3 Task 3

The third `v1.3` step was to replace the earlier vitamin-C-specific treatment logic with broader treatment-evidence categories that also work on the echinacea document.

This pass focused on moving from document-specific query handling toward reusable evidence types such as prevention, null effect, subgroup benefit, duration, and overall treatment conclusions.

### Treatment-Generalization Work

This pass included:

- replacing the earlier vitamin-C-only retrieval intents with broader treatment-evidence intents
- doing the same on the answer-selection side
- tightening a few scoring rules so prevention, overall-treatment, and duration cases do not drift toward methods-style or generic background sentences
- refining the echinacea eval gold set so it better reflects acceptable summary-evidence chunks rather than only one narrow formulation

### Verification

This pass was checked by:

- rerunning the full 14-case benchmark
- spot-checking vitamin-C prevention answers after the generalization
- spot-checking echinacea overall-conclusion and incidence answers after the generalization
- checking that the negative influenza abstention case still holds

Observed behavior:

- the generalized treatment-evidence logic now works across both vitamin-C and echinacea queries
- the benchmark recovered from the first too-broad version of the pass after the scoring was narrowed again
- the duration answer path improved once the answer layer got an explicit `duration` intent instead of falling back to generic scoring

### Updated Evaluation Check

After the generalized treatment-evidence pass, the benchmark moved to:

- average `precision@5`: `0.436`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.909`
- negative-case success rate: `1.0`

Interpretation:

- retrieval remains strong across the three-document benchmark
- the generalized treatment-evidence pass improved over the earlier task-1/task-2 state instead of just preserving it
- the remaining answer-quality misses are now concentrated in a few treatment-heavy cases rather than spread across the benchmark

## v1.3 Task 4

The fourth `v1.3` step was to improve the OCR-to-chunk handoff on low-text or scanned pages before adding a noisier scanned benchmark document.

This pass focused on reducing header/footer carryover and making OCR-derived paragraph text cleaner before it reaches chunk assembly.

### OCR Handoff Work

This pass included:

- filtering more OCR line noise near page top/bottom bands, especially DOI / URL / copyright / page-number style lines
- normalizing OCR line text before paragraph grouping
- joining OCR lines into paragraph text more carefully, including simple hyphenation repair
- adding OCR-specific block cleanup in chunking so linebreak noise and obvious OCR header/footer fragments are less likely to survive into chunk segments

### Verification

This pass was checked by:

- rerunning the full 14-case benchmark to confirm no regression on the current three-document set
- a synthetic image-only OCR smoke test with header/footer-style noise added to the page

Observed behavior:

- the OCR fallback still produced a chunk from the synthetic page
- the obvious top DOI/header line and bottom page marker no longer dominated the OCR-derived chunk content
- the current benchmark stayed stable, which is what this pass needed to preserve

### Updated Evaluation Check

After the OCR handoff cleanup pass, the benchmark remained at:

- average `precision@5`: `0.436`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.909`
- negative-case success rate: `1.0`

Interpretation:

- this was another infrastructure-quality pass rather than a direct benchmark-boost pass
- the repo is now in a better position to add a genuinely noisier scanned benchmark document
- OCR-derived content is still noisy in a real-world sense, but the handoff into chunking is cleaner and more controllable than before

## v1.3 Task 5

The fifth `v1.3` step was to make the evaluation output more useful for regression analysis, not by changing the benchmark itself, but by making each case easier to inspect after a quality pass.

### Evaluation Debug Report Work

This pass included:

- adding per-case debug records to the saved evaluation report
- storing top-k retrieval snapshots with chunk metadata and short previews
- storing expanded-context snapshots so neighbor-expansion behavior is visible after each run
- storing answer previews and evidence snippets for each case
- adding simple case-status labels and warning-case IDs so weaker cases are easier to spot without reading the whole report

### Verification

This pass was checked by:

- rerunning the full 14-case benchmark
- confirming that the generated report includes `debug_cases`
- spot-checking the new warning-case summary fields

Observed behavior:

- benchmark headline metrics stayed unchanged, which is expected because this pass improves observability rather than retrieval quality
- the report is now much more useful for regression review because it shows what the retriever and answer layer actually surfaced for each case
- current warning cases are visible immediately without scanning the full raw report

### Updated Evaluation Check

After the richer debug-report pass, the benchmark remained at:

- average `precision@5`: `0.436`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.909`
- negative-case success rate: `1.0`

Interpretation:

- the repo now has stronger evaluation visibility without changing the current retrieval/answer baseline
- the remaining work is no longer “make the report readable”, but “stress the pipeline on noisier scanned evidence and broader treatment-heavy documents”

## v1.4 Task 1

The first `v1.4` step was to stop relying only on native-text benchmark documents and add a genuinely noisier scanned case to the local evaluation loop.

### Scanned Benchmark Setup

This pass included:

- creating a local image-based scanned version of the CT-study PDF from the course medical sample set
- sending that derived PDF through the existing OCR extraction path
- chunking the scanned document and rebuilding the shared local index across all current benchmark documents
- extending the evaluation set with two OCR-heavy grounded CT-study cases and one additional negative abstention case

### Verification

This pass was checked by:

- confirming that all six pages of the derived CT-study document triggered OCR fallback
- confirming that chunk JSON files were produced for the scanned document
- rerunning the full benchmark after rebuilding the shared index

Observed behavior:

- the OCR path remained operational on the derived scanned document
- negative abstention still held on the new contrast-agent query
- the new CT-study cases lowered recall and MRR, which is useful because it exposes a real OCR-derived retrieval weakness instead of only adding another easy native-text case

### Updated Evaluation Check

After the scanned-benchmark expansion pass, the benchmark moved to:

- average `precision@5`: `0.415`
- average `recall@5`: `0.962`
- `MRR`: `0.891`
- average answer keyword coverage: `0.923`
- negative-case success rate: `1.0`

Interpretation:

- this is the first benchmark pass that clearly separates native-text performance from OCR-derived performance
- the current OCR path is good enough to keep the pipeline running end-to-end, but not yet strong enough to preserve earlier retrieval metrics on the scanned cases
- the next useful work is to audit OCR-derived chunk quality and retrieval behavior on the CT-study cases rather than immediately widening the benchmark again

## v1.4 Task 2

The second `v1.4` step was to tighten the scanned OCR path where the new CT-study cases exposed real weaknesses.

### OCR Cleanup And Grouping Work

This pass included:

- splitting OCR paragraph groups more aggressively around structural headings such as `Abstract`, `Methods`, `Discussion`, and `Follow-up Evaluations`
- dropping more OCR-specific title, author, footer, and reprint-credit fragments before they become blocks
- adding stronger OCR fragment cleanup during chunking
- labeling obviously garbled OCR chunks more explicitly
- adding lightweight CT-specific retrieval and answer intents for the new scanned benchmark questions

### Verification

This pass was checked by:

- regenerating the scanned CT-study extraction and chunk outputs
- rebuilding the shared index across all four documents
- rerunning the full 17-case benchmark
- checking that the scanned CT-study cases recovered while negative abstention still held

Observed behavior:

- the scanned CT-study document now produces fewer chunks, with less obvious author/title/footer carryover
- the scanned-case retrieval drop seen right after `v1.4 / task 1` was recovered
- the remaining benchmark warnings are again concentrated in treatment-heavy native-text cases rather than the scanned CT-study cases

### Updated Evaluation Check

After the OCR cleanup and scanned-retrieval pass, the benchmark moved to:

- average `precision@5`: `0.431`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.923`
- negative-case success rate: `1.0`

Interpretation:

- the scanned benchmark is now useful without dragging down the whole evaluation loop
- OCR-derived cases are still noisier than native-text cases, but the current local path is good enough to preserve full recall and MRR on the expanded benchmark
- the next useful step is no longer OCR cleanup itself, but clearer benchmark slicing so native and OCR-derived behavior can be compared directly in the report

## v1.4 Task 3

The third `v1.4` step was to make the evaluation report easier to read at a higher level by adding a few simple benchmark slices.

### Benchmark Slicing Work

This pass included:

- adding per-case slice labels to the debug records
- adding aggregate summaries for `native_text` vs `ocr_derived`
- adding aggregate summaries for `treatment` vs `non_treatment`

### Verification

This pass was checked by:

- rerunning the full 17-case benchmark
- confirming that the saved report now contains a `slices` section
- checking that the OCR-derived slice and treatment slice reflect the current warning distribution

Observed behavior:

- benchmark headline metrics did not change, which is expected
- the report now shows directly that the remaining warnings are concentrated in treatment-heavy native-text cases
- the current OCR-derived slice is back to full recall and MRR after the scanned cleanup pass

### Updated Evaluation Check

After the slicing pass, the benchmark remained at:

- average `precision@5`: `0.431`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.923`
- negative-case success rate: `1.0`

Interpretation:

- the report is now easier to use for deciding where the next quality pass should land
- the next useful work is no longer better reporting, but improving the remaining treatment-heavy answer warnings without regressing the scanned path

## v1.4 Task 4

The fourth `v1.4` step was to tighten answer selection on the remaining treatment-heavy warning cases without undoing the scanned-path stabilization work.

### Treatment-Focused Answer Pass

This pass included:

- normalizing ligature-heavy answer text more robustly during keyword-coverage evaluation
- adding a small coverage guard in answer selection for treatment intents so key phrases such as `benefit`, `beneficial effect`, and `not altered` are less likely to be dropped when evidence is already present in the retrieved context
- tightening `treatment_null_effect` scoring so methods-style or duration-style vitamin-C sentences are less likely to outrank the actual null-effect evidence

### Verification

This pass was checked by:

- rerunning the full 17-case benchmark
- checking the three remaining treatment-heavy warning cases directly
- confirming that the scanned OCR slice did not regress

Observed behavior:

- benchmark headline retrieval metrics stayed fixed, which is what this pass was meant to preserve
- average keyword coverage improved noticeably
- warning cases dropped from three to one

### Updated Evaluation Check

After the treatment-focused answer-selection pass, the benchmark moved to:

- average `precision@5`: `0.431`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.974`
- negative-case success rate: `1.0`

Interpretation:

- the remaining weakness is now a narrow treatment-specific edge case rather than a broader cluster
- the next useful step should be a design choice about chunking vs lightweight reranking, not another round of narrow intent exceptions

## v1.4 Task 5

The fifth `v1.4` step was not another heuristic tweak. It was a benchmark-driven design decision about where the next useful gain should come from.

### Decision

The current benchmark now points to a chunking-first next iteration rather than a reranking-first one.

### Why This Direction

The decision is based on the current state of the benchmark:

- retrieval headline metrics are already saturated on the current 17-case setup
- the OCR-derived slice has recovered to full recall and MRR
- the remaining warning is narrow and treatment-heavy rather than a broad ranking failure
- the remaining gap looks more like a mixed-summary / chunk-boundary problem than a top-k ordering problem

### Practical Outcome

No new metric jump was expected from this step.

The value of the pass is architectural:

- `v1.4` is now complete
- the next useful work should focus on stronger chunk boundaries inside long treatment-summary regions
- lightweight reranking should only be revisited if that chunking-first pass stalls

## v1.5 Task 1

The first `v1.5` step focused on cleaner chunk boundaries inside treatment-heavy summary material rather than another retrieval-side heuristic pass.

### Chunking Work

This pass included:

- replacing the earlier regex-heavy paragraph splitter with a line-aware paragraph grouping pass so wrapped review text is less likely to fragment into artificial mini-segments
- splitting mixed treatment-summary paragraphs more cleanly, especially where null-effect, subgroup-benefit, duration, and therapeutic findings previously coexisted inside one chunk
- dropping single-letter paragraph debris more aggressively during chunk normalization
- re-chunking all four indexed documents and rebuilding the shared local index
- realigning the benchmark gold chunk IDs to the new finer-grained chunk layout

### Answering Follow-Up

The chunk split exposed one practical edge case:

- strong null-effect evidence such as `incidence was not altered` could become too specific to survive candidate filtering when the query itself used broader surface wording

To keep the benchmark aligned with the new chunk boundaries, the answer layer got one small follow-up:

- strong treatment-intent anchor phrases can now survive scoring even when raw lexical overlap with the query is thin

This was deliberately small and tied to the new chunk layout, not a return to broader retrieval-specific tuning.

### Verification

This pass was checked by:

- inspecting the re-split vitamin-C treatment chunks directly
- rebuilding the shared index across all four documents
- rerunning the full 17-case benchmark after updating the gold chunk IDs

Observed behavior:

- the vitamin-C treatment summary now separates null-effect, subgroup-benefit, duration, and therapeutic findings more cleanly
- the benchmark no longer carries the last mixed-summary warning case
- OCR-derived CT-study cases remain stable after the chunking-first pass

### Updated Evaluation Check

After the first `v1.5` chunking-first pass, the benchmark moved to:

- average `precision@5`: `0.354`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`
- negative-case success rate: `1.0`
- warning-case count: `0`

Interpretation:

- the lower `precision@5` now mainly reflects a finer-grained benchmark and narrower gold chunk targets after the chunk split
- recall, MRR, and answer coverage are all back to their ceiling values on the current 17-case setup
- the next useful step is to make long review-summary chunk boundaries more section-aware before deciding whether any reranking upgrade is still worth prototyping

## v1.5 Tasks 2-5

The remaining `v1.5` work completed the chunking-first iteration instead of switching to a reranking-first path.

### Section-Aware Chunking Work

This pass included:

- adding more section-aware review chunk boundaries inside long summary-heavy review blocks
- splitting inline review headings such as `Introduction`, `Methods`, `Search strategy and selection criteria`, and `Conclusion` into cleaner chunk scopes
- keeping those boundaries local and inspectable instead of introducing a heavier segmentation model

### Subtopic-Cue Work

This pass also added lightweight chunk-level treatment subtopic cues for:

- prevention
- null effect
- subgroup benefit
- duration
- overall conclusion

The cues are now persisted in:

- chunk JSON
- document-level chunk metadata
- index metadata

They also contribute small retrieval bonuses for matching treatment-intent queries, but do not constitute a new reranking stage.

### Benchmark Realignment And Validation

Because the review-heavy chunk layout changed again, the benchmark gold chunk IDs were realigned to the new structure for the affected documents.

After that:

- the full 17-case benchmark was rerun
- the warning-free state held
- no additional answer-selection exception was needed beyond the small null-effect anchor fix already introduced during task 1

### Updated Evaluation Check

After the full `v1.5` pass, the benchmark remained at:

- average `precision@5`: `0.354`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`
- negative-case success rate: `1.0`
- warning-case count: `0`

Interpretation:

- the current benchmark does not justify adding a lightweight reranking prototype yet
- on the present document set, chunking-first work was enough to restore a clean warning-free state
- the next useful iteration is to broaden the benchmark and use that broader pressure to decide which of the currently deferred features should come back into scope first

## Next Suggested Step

The next sensible implementation step is:

- broaden the benchmark again with a more form- or grid-heavy eighth document so questionnaire/layout pressure is separated from the current table-heavy manual pressure

After that:

- tighten answer compression on dense source-anchored technical/manual answers so the benchmark-clean path is also cleaner to read
- only revive narrower deferred paths if the broader benchmark keeps exposing concrete failure modes

## v1.6 Tasks 1-5

`v1.6` was the first pass where the benchmark itself became the tool for deciding which deferred features should return, rather than treating those decisions as design assumptions.

### Broader Benchmark Work

This pass included:

- adding a fifth benchmark document based on the CMAJ prevention/treatment review
- expanding the benchmark to 21 cases, including new review-summary and negative cases
- rebuilding the shared local index across five documents
- widening the saved evaluation slices so review-heavy, source-anchored, and scanned-path cases can be inspected separately

### Retrieval And Answering Work

This pass also included:

- prototyping a lightweight reranking pass only after the broader benchmark exposed real ranking pressure on the new review-summary queries
- comparing that reranking pass against the chunking-first baseline inside the evaluation report
- refining review-summary chunk splitting and source-aware answer assembly to reduce cross-document leakage
- adding stronger chunk-quality labels for table-reference and reference-tail noise

### Faithfulness Audit Work

To avoid reintroducing heavier evaluation machinery too early, the pass added:

- a small sampled faithfulness audit over selected grounded cases
- a saved audit summary in the evaluation report
- an explicit decision checkpoint for whether LLM-as-a-judge should return

### Updated Evaluation Check

After the full `v1.6` pass, the benchmark stood at:

- average `precision@5`: `0.350`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `0.917`
- negative-case success rate: `1.0`
- warning-case count: `2`

Interpretation:

- the five-document benchmark is now broad enough to justify keeping the lightweight reranking pass over the chunking-only baseline
- the remaining answer-quality debt is concentrated in two treatment-heavy review/native cases: `antibiotics` and `cmaj_zinc_prevention`
- the current benchmark still does not justify reviving `pdfplumber` or LLM-as-a-judge, because the new failures are review-summary and table-reference noise rather than true table-extraction or unsupported-answer failures

## v1.7 Tasks 1-5

`v1.7` broadened the benchmark again and then used that broader setup to turn deferred-feature decisions into explicit artifacts instead of informal conclusions.

### Broader Benchmark Work

This pass included:

- adding a sixth benchmark document based on *The common cold: a review of the literature*
- expanding the benchmark from 21 to 25 cases
- adding new source-anchored review cases plus one new negative case
- rebuilding the shared local index across six documents

### Retrieval, Answering, And Slice Work

This pass also included:

- making source-anchored review queries lock more aggressively onto their intended document
- adding a dedicated symptom-pathogenesis intent for the new literature-review benchmark
- labeling Elsevier/COVID disclaimer noise more explicitly
- extending evaluation slices so `source_anchored_review` and the new review-heavy cases are tracked directly
- adding explicit deferred-feature decisions into the saved evaluation report

### Deferred-Feature Decision Check

The sixth benchmark document was useful because it created new pressure without forcing premature complexity:

- it exposed source-mixing, disclaimer noise, and review-summary issues
- it did **not** expose a true table/text extraction miss that would justify reviving `pdfplumber`
- it also did not justify a cross-encoder reranker or LLM-as-a-judge on the current benchmark

### Updated Evaluation Check

After the full `v1.7` pass, the benchmark stood at:

- average `precision@5`: `0.344`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`
- negative-case success rate: `1.0`
- warning-case count: `0`

Interpretation:

- the benchmark is warning-free again across six indexed documents and 25 cases
- the current lightweight reranking pass is still enough on this benchmark
- the next useful work is not another local heuristic fix, but a stronger seventh benchmark document that can reopen `pdfplumber` or cross-encoder questions only if it really creates those failure modes

## v1.8 Tasks 1-5

`v1.8` added a genuinely table-heavy technical manual and used it to re-test whether any of the heavier deferred features needed to come back.

### Broader Benchmark Work

This pass included:

- adding `AJMedP-4-2_SRD_EDA_V1_E_2561.pdf` as the seventh benchmark document
- extracting it through the current `PyMuPDF`-first path, then chunking and indexing it alongside the existing six documents
- expanding the benchmark from 25 to 30 cases with source-anchored technical/manual queries plus one new negative case

### Retrieval, Answering, And Slice Work

This pass also included:

- source-aware document locking for `AJMedP` / `TB MED 508` technical queries
- new technical/manual intents for hypothermia predisposition, hypothermia symptoms, frostbite-risk guidance, and immersion-limit lookup
- stronger noise labeling for `List of Tables` / `List of Figures`-style chunks
- expanded slices for `table_heavy`, `source_anchored_technical`, and `source_locked`
- a focused cleanup of the remaining Wat review answer-selection warning so the broadened benchmark stayed warning-free

### Deferred-Feature Decision Check

The table-heavy benchmark was useful because it created real manual/table pressure without forcing more tooling:

- the new manual exposed answer-selection and source-locking issues first
- once those were fixed, the table-heavy slice became warning-free
- because the core table content was already extracted well enough, `pdfplumber` still did not need to return
- the benchmark also remained stable enough that neither cross-encoder reranking nor LLM-as-a-judge had to come back

### Updated Evaluation Check

After the full `v1.8` pass, the benchmark stood at:

- average `precision@5`: `0.328`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`
- negative-case success rate: `1.0`
- warning-case count: `0`

Interpretation:

- the benchmark is warning-free again across seven indexed documents and 30 cases
- the new `table_heavy` and `source_anchored_technical` slices are clean on the current benchmark
- `pdfplumber` still stays deferred because the current table-heavy document does not expose a true table/text extraction miss
- the current lightweight reranking pass and sampled faithfulness audit remain sufficient on this benchmark

## v1.9 Tasks 1-5

`v1.9` broadened the benchmark again, but this time with structured-form pressure rather than another review-heavy or purely table-heavy document.

### Broader Benchmark Work

This pass included:

- adding `Health-check_questionnaire_for_subjects_expose_to_.pdf` as the eighth benchmark document
- extracting it through the current native-text path, chunking it, and rebuilding the shared index across eight documents
- expanding the benchmark from 30 to 36 cases
- adding new source-anchored questionnaire, form/grid, numeric-option, appendix-like, and negative abstention cases

### Structured-Form Work

This pass also included:

- adding a narrow structured-form assist path for flagged questionnaire/table blocks instead of reviving a broader table dependency
- normalizing questionnaire items so grid-like questions are rewritten into cleaner source-anchored answerable chunks
- normalizing the late follow-up `Table I` rows into more readable row summaries
- tightening source-aware retrieval with a constrained fallback pass when anchored queries drift away from the intended document
- tightening answer compression and row-specific answer selection for source-anchored questionnaire/table cases

### Deferred-Feature Decision Check

The new questionnaire benchmark was useful because it created a different kind of pressure from the table-heavy technical manual:

- it exposed structured-form / source-locking issues rather than a true table-extraction miss
- the narrow structured-form assist path was enough to recover those issues on the current benchmark
- `pdfplumber` still did not need to return
- cross-encoder reranking still did not need to return
- `LLM-as-a-judge` still did not need to return

### Updated Evaluation Check

After the full `v1.9` pass, the benchmark stood at:

- average `precision@5`: `0.305`
- average `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`
- negative-case success rate: `1.0`
- warning-case count: `0`

Interpretation:

- the benchmark is warning-free again across eight indexed documents and 36 cases
- the new `form_grid` and `source_anchored_form` slices are clean on the current benchmark
- the current manual/table + questionnaire benchmark still does not justify reviving `pdfplumber`
- the current lightweight reranking pass and sampled faithfulness audit remain sufficient on this broader benchmark

## v1.10-v1.11

These iterations broadened and then stabilized the structured-form path instead of only adding one more benchmark source.

What changed:

- added the opioid appendix family as a ninth benchmark source
- added new source-anchored checklist, legend, and follow-up schedule cases
- refactored the form path into reusable pattern families instead of document-specific rewrites
- added deterministic regression checks for high-risk source-anchored form cases

Outcome:

- the benchmark stayed warning-free at 41 cases
- the form/grid path became more maintainable
- the project gained earlier regression signals than global benchmark reruns alone

## v1.12-v1.13

These iterations shifted focus from rule sprawl and source-anchored inspection toward explicit cross-document behavior.

What changed:

- moved structured intent metadata into a shared declarative config
- added `answer_trace` to structured outputs and debug reports
- added an appendix-heavy checklist source and generalized the checklist path without a new per-document branch
- added cross-document intents for source listing and source comparison
- wired multi-source document matching into retrieval for cross-document queries
- added a dedicated `cross_document_core` regression shard and expanded the full eval set

Outcome:

- the benchmark now spans 48 cases across 11 indexed documents
- it remains warning-free
- the pipeline can now answer both single-document evidence questions and multi-file source-discovery / comparison questions
- the project direction is now closer to the original goal of a general PDF-to-JSON RAG pipeline rather than a benchmark tied only to one medical question family
