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
Current package metadata version: `0.1.0`
Current internal implementation level: `v1.48.0`

`0.1.0-beta` is the public release label. The package version remains PEP440-compatible `0.1.0` until the beta checkpoint is cut as a formal package release.

What the repo now has:

- extraction-time block roles with per-block text provenance and quality signals
- native/OCR page fusion instead of one global OCR fallback decision
- extraction-time layout signals and per-page processing summaries
- explicit multi-column reading-order normalization from extraction through chunking
- relative font-size, bold-font, and TOC-backed heading signals during extraction
- optional `pdfplumber` table probe metadata and supplemental `table_like` blocks when the `tables` extra is installed
- extraction-time sections with `section_path` and `section_kind`
- section roles, layout signals, text-source profiles, and source-block traces carried into chunking and inspection
- structure-aware chunking and chunk metadata
- chunk-level block provenance, block-role profiles, layout signals, and explicit chunk strategies
- feature-based query planning
- evidence-intent planning for treatment null-effect and subgroup-benefit questions
- explicit `document_selection` traces
- distinct document-level type / purpose / audience / overview answers
- confidence-aware document classification answers and semantic confidence signals
- explicit classification-rationale and classification-limits answers
- compact default JSON answers with richer debug output behind `--verbose`
- public install/release checks through `doctor`, `smoke-check`, `package-check`, and `release-check`
- local semantic sanity checks for unfamiliar PDFs through `layout-sanity-check`
- a local corpus sampler over repo-local `pdf/` artifacts through `corpus-sanity-check`
- bucket-level corpus diagnostics and follow-up actions for unknown-document sanity checks
- corpus sample profiles, saved corpus snapshots, failure examples, and a contract gate for corpus diagnostics
- stronger unknown-document typing for registration forms, court opinions, government bulletins, and inspection-style records
- corpus-level semantic pass metrics that separate technical success from semantic understanding
- a dedicated `processing_layer_core` shard for block typing, section-role recovery, and chunk provenance
- a dedicated `processing_strategy_core` shard for strategy-aware chunking on structure-heavy inputs
- an explicit retrieval contract for `single_document_qa`, `document_understanding`, and `cross_document_discovery`
- optional cross-encoder reranking behind an env flag, with lightweight reranking as the stable fallback
- runtime-mode comparison for baseline, sentence-transformers, cross-encoder, and opt-in LLM synthesis paths
- full-suite runtime comparison and green promotion gate for optional sentence-transformer embeddings
- explicit runtime diagnostics and promotion summary commands
- explicit runtime decision output that keeps `hash` as default and recommends sentence-transformers only as opt-in
- explicit embedding backend selection with `hash`, `sentence-transformers`, and `auto`
- saved promotion snapshots and backend payloads in build/workflow/smoke outputs
- verified installed-entrypoint public path and a repeatable `readme-smoke-check`
- aggregated public beta validation through `public-beta-check`
- reranking of the neighbor-expanded context before answer synthesis
- an explicit grounded-only synthesis prompt contract over selected context chunks, with opt-in local-command execution
- an LLM-as-judge prompt contract for faithfulness scoring in evaluation reports, with opt-in local-command JSON judging
- strict JSON/fence parsing for opt-in model and judge outputs
- answer-claim/evidence alignment diagnostics in answer traces and faithfulness records
- provider metadata around the current env-command prompt runtime
- prompt/eval contract validation for sampled faithfulness gates
- optional low-confidence semantic multipass behind `PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS=1`
- a shared `document_synthesis` handoff for selection strategy, support scope, and answer chunks
- answer `contract_health` and workflow `quality_profile` blocks for unknown-PDF processing quality, semantic confidence, retrieval readiness, and answer trust
- a layer-aware evaluation summary for `processing`, `retrieval`, and `answer_faithfulness`
- layer-stability and architecture-gate summaries that turn those layers into explicit evaluation gates
- corpus-level `processing / semantics / trust` layers and an unknown-document architecture gate
- deterministic corpus sampling manifests for quick/balanced/stress unknown-PDF sanity checks
- compact release-check summaries with full maintainer payloads behind `--verbose`

Current validation snapshot:

- public CLI tests rerun in the current milestone: green
- processing-layer maintainer shards rerun in the current milestone: green
- retrieval-contract maintainer shard rerun in the current milestone: green
- retrieval-synthesis maintainer shard rerun in the current milestone: green
- evaluation-layer public/unit validation rerun in the current milestone: green
- layer-gate validation rerun in the current milestone: green
- corpus-gate validation rerun in the current milestone: green
- full 77-case benchmark: green
  - `precision@5 = 0.6031`
  - `recall@5 = 1.0`
  - `MRR = 1.0`
  - `avg_keyword_coverage = 1.0`
  - `negative_success_rate = 1.0`
  - `warning_case_count = 0`
  - `answer_faithfulness_failing_case_count = 0`
  - `architecture_gates.all_pass = true`
- sampled faithfulness audit: green
  - `avg_supported_sentence_ratio = 1.0`
  - `failing_case_count = 0`
  - `llm_judge_prompt_contract = faithfulness_context_judge.v1`
  - `contract_validation.all_pass = true`

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
- `v0.7.1-v0.7.4`
  - improved unknown-document typing on the repo-local corpus for registration forms, court opinions, government bulletins, and inspection-style records
  - reduced fallback to `document/reference_lookup` on unfamiliar local-corpus PDFs
  - added corpus-level semantic pass metrics and explicit technical-vs-semantic corpus gate reporting
- `v0.8.0`
  - added an extraction-time block model with roles, text provenance, and text-quality signals
  - added native/OCR page fusion so source choice is more local and less brittle
  - rewrote section and chunk metadata to carry section roles, source-block ids, source-block roles, and block-role profiles
  - added `processing_layer_core` to keep block typing, section-role recovery, and chunk provenance in the maintainer regression gate
- `v0.9.0`
  - split retrieval into explicit contracts for single-document QA, document understanding, and cross-document discovery
  - exposed compact `retrieval_contract` traces in answer payloads
  - added `retrieval_contract_core` to keep those answer-path separations covered in the maintainer regression gate
- `v1.0.0`
  - added extraction-time layout signals and per-page processing summaries
  - enriched section metadata with layout signals and text-source profiles
  - upgraded chunking to carry explicit chunk strategies, layout signals, and text-quality summaries
  - added `processing_strategy_core` to keep strategy-aware chunking covered in the maintainer regression gate
- `v1.1.0`
  - added a shared `document_synthesis` handoff between retrieval and document-level answering
  - aligned support scope, selected docs, and answer chunks across overview, routing, listing, justification, and compare modes
  - added `retrieval_synthesis_core` to keep that handoff covered in the maintainer regression gate
- `v1.2.0`
  - split evaluation reporting into explicit `processing`, `retrieval`, and `answer_faithfulness` layers
  - added per-case layer status records and a layer summary to the saved evaluation report
  - exposed the same layer summary in `evaluate-mvp --json` and CLI text output
- `v1.3.0`
  - added `layer_stability` thresholds for `processing`, `retrieval`, and `answer_faithfulness`
  - added `architecture_gates` so full-suite and partial-suite evaluations can return an explicit gate decision
  - exposed those gates in `evaluate-mvp --json` and CLI text output
- `v1.4.0`
  - added corpus-layer summaries for `processing`, `semantics`, and `trust` on top of `corpus-sanity-check`
  - added a corpus architecture gate for repo-local unknown-document sampling
  - surfaced that local corpus gate as a local-only advisory signal inside `release-check`
- `v1.5.0`
  - added bucket-level diagnostics to `corpus-sanity-check`
  - added deterministic follow-up actions that identify whether the next work belongs in processing, semantics, layout, or trust policy
  - surfaced the local corpus follow-up count in text `release-check` output
- `v1.6.0-v1.6.4`
  - added `quick`, `balanced`, and `stress` corpus sample profiles so unknown-document checks can control cost explicitly
  - saved corpus sanity snapshots under `data/eval/` for comparison across versions
  - added failure examples to follow-up actions so bucket failures point at concrete PDFs and reasons
  - added a corpus contract gate for bucket diagnostics, follow-up actions, and architecture-gate consistency
  - kept the work aligned to processing, retrieval-contract readiness, and architecture-focused evaluation rather than adding a learned reranker
- `v1.7.0`
  - added explicit multi-column reading-order normalization using block x-coordinate clusters
  - applied the same bbox-aware ordering when chunking previously saved extracted blocks
  - added regression coverage so two-column pages read column-by-column instead of row-interleaving columns
- `v1.8.0`
  - added extraction-time font metadata for native PyMuPDF blocks
  - promoted short blocks to headings when supported by relative font size, bold font, or TOC membership
  - preserved font signals in native JSON and loaded block artifacts
- `v1.9.0`
  - added an optional `pdfplumber` table probe without making it a hard public dependency
  - exposed probe status in `doctor` and extraction summaries
  - added a `tables` package extra for installs that want the specialized table probe path
- `v1.10.0`
  - converted the optional `pdfplumber` path from probe-only into supplemental `table_like` block generation
  - normalized detected table rows into pipe-separated table text that can flow through existing chunking
  - kept `pdfplumber` optional so public installs without the `tables` extra remain stable
- `v1.11.0`
  - added optional cross-encoder reranking for local environments that have a cross-encoder model available
  - exposed the active rerank backend in chunk payloads
  - kept lightweight reranking as the default and fallback path
- `v1.12.0`
  - added a rerank pass over the neighbor-expanded chunk set before answer synthesis
  - added rank signals for both initial retrieval and expanded context ranking
  - kept the same lightweight/cross-encoder fallback behavior for the expanded context
- `v1.13.0`
  - taught the planner to classify treatment evidence subquestions instead of leaving them as `generic`
  - routed vitamin C null-effect and cold-stress subgroup queries to the vitamin C source through the retrieval contract
  - restored the `evidence_anchor_core` shard to green
- `v1.14.0`
  - added an LLM-ready grounded synthesis prompt contract that forbids outside knowledge and requires chunk citations
  - exposed prompt-contract metadata in answer traces for human and future LLM-as-judge review
  - kept LLM invocation disabled by default, so current answers remain deterministic/extractive
- `v1.15.0`
  - added an LLM-as-judge prompt contract for faithfulness scoring against source context
  - embedded judge-contract metadata in sampled faithfulness audit records and summaries
  - kept automated judge model execution disabled by default
- `v1.16.0`
  - added provider-agnostic local command execution for grounded synthesis via `PDF_TO_JSON_RAG_LLM_COMMAND`
  - added provider-agnostic local command execution for strict-JSON judge diagnostics via `PDF_TO_JSON_RAG_JUDGE_COMMAND`
  - kept both execution paths disabled by default and exposed runtime status in answer/evaluation payloads
- `v1.17.0-v1.21.0`
  - added strict JSON/fence parsing for prompt-runtime outputs
  - added answer-claim/evidence alignment status as diagnostic trace data
  - formalized provider metadata around the current env-command runtime
  - added prompt/eval contract validation for sampled faithfulness records
  - added optional low-confidence semantic multipass without changing default semantics
- `v1.22.0`
  - added `compare-runtime-modes` to measure baseline, sentence-transformers, cross-encoder, and opt-in LLM synthesis on the same case set
  - made optional model comparisons offline-safe by reporting fallback/runtime availability instead of silently trying network downloads
  - saved comparison reports under `data/eval/runtime_mode_comparison.json`
- `v1.23.0`
  - added `--all-cases` and a promotion gate for sentence-transformer embeddings
  - measured local `all-MiniLM-L6-v2` across the full 77-case eval suite
  - found one full-suite answer regression despite improved recall/MRR, which was fixed in `v1.24.0`
- `v1.24.0`
  - fixed source-anchored evidence synthesis when optional sentence-transformer retrieval returns a correct named-source top hit plus off-source neighbors
  - reran the full 77-case baseline vs local `all-MiniLM-L6-v2` comparison: both modes pass `77/77`, and the sentence-transformer promotion gate is green
- `v1.25.0-v1.29.0`
  - added runtime promotion reporting and explicit runtime diagnostics
  - added `PDF_TO_JSON_RAG_EMBEDDING_BACKEND=hash|sentence-transformers|auto` while keeping hash as the default path
  - added `source_anchor_contract_core` and wired it into release-check
  - updated CLI docs for backend selection, model availability, and promotion readiness
- `v1.30.0-v1.34.0`
  - added install/runtime context to backend diagnostics
  - added requested/effective/fallback embedding payloads to index, smoke, and workflow outputs
  - added saved promotion snapshots after green full-suite runtime comparisons
  - documented local `all-MiniLM-L6-v2` as recommended opt-in, not a silent default change
- `v1.35.0-v1.38.0`
  - clarified tracked vs ignored eval artifacts before commit prep
  - verified installed `pdf-to-json-rag` after `python -m pip install .`
  - confirmed public demo smoke path through the installed entrypoint
  - refreshed docs toward a stable checkpoint
- `v1.39.0-v1.43.0`
  - clarified public beta label vs package metadata version policy
  - added an installed README smoke gate that replays install, init, doctor, create-demo-pdf, smoke-check, and runtime-check
  - added deterministic corpus sample manifests with bucket counts and digest checksums
  - exposed runtime decision fields for default backend, recommended opt-in backend, and not-default rationale
  - made `release-check --json` compact by default while preserving the full payload behind `--verbose`
- `v1.44.0-v1.48.0`
  - added `public-beta-check` as the single pre-tag aggregator over installed README smoke, runtime decision, corpus quick gate, and compact release summary
  - added answer `contract_health` for retrieval path, support scope, selected docs, support docs, and claim-alignment presence
  - added workflow `quality_profile` so random-PDF runs expose processing quality, semantic confidence, retrieval readiness, and answer-trust status
  - extended smoke checks to assert the presence of contract health and quality profile blocks


### 6.2. Next Steps

1. Keep the public beta path stable: install, init, doctor, demo PDF, smoke-check, runtime-check.
2. Use `public-beta-check --json` before tagging; use `release-check --json --verbose` only when full maintainer detail is needed.
3. Keep `hash` as the default backend while treating sentence-transformers as the recommended opt-in backend.
4. Expand corpus sanity only through deterministic sample profiles and compact manifests, not by tracking local PDFs or generated indexes.
5. Keep learned components opt-in until they beat the simpler baseline on clear failure patterns.

## 7. Explicitly Deferred

- deeper `pdfplumber` table schema normalization beyond supplemental table blocks
- mandatory/default cross-encoder reranking
- automated LLM-as-a-judge execution against a configured judge model
- cloud deployment
- multi-document schema extraction
- visual grounding UI
