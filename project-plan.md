# PDF-to-JSON RAG Pipeline

_This plan was brainstormed with [DeepLearning.AI / Skill Builder](https://skillbuilder.deeplearning.ai/) and ChatGPT 5.4, then narrowed into a local-first MVP._

Internal development iterations in this plan use `vN.x` labels. Public releases follow semantic versioning starting at `0.1.0-beta`.

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
Current internal implementation level: `v4.5.0`

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
- stronger public-record semantics for statistical tables, web job listings, environmental site records, and institutional correspondence from local unknown-PDF corpus buckets
- corpus-level semantic pass metrics that separate technical success from semantic understanding
- `unknown_document_semantics_core` as a maintainer shard for unfamiliar-document type, purpose, audience, and confidence behavior
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
- answer `contract_health` and workflow `quality_profile` blocks for unknown-PDF processing drilldown, semantic confidence, retrieval readiness reasons, and answer trust
- document-level claim alignment using support-trace fragments for metadata claims
- explicit quality-profile thresholds and public-smoke quality summary in `public-beta-check`
- `quality_profile.overall_status` and `recommended_next_action` as the user-facing random-PDF contract
- compact processing failure taxonomy in `processing_diagnostics`
- retrieval/synthesis contract status, support coverage, and answer source mix
- compact `assess-pdf` acceptance summaries for unfamiliar PDFs
- a layer-aware evaluation summary for `processing`, `retrieval`, and `answer_faithfulness`
- layer-stability and architecture-gate summaries that turn those layers into explicit evaluation gates
- corpus-level `processing / semantics / trust` layers and an unknown-document architecture gate
- deterministic corpus sampling manifests for quick/balanced/stress unknown-PDF sanity checks
- compact corpus snapshots and saved profile comparison without reprocessing PDFs
- corpus review workbench output with `pass/review/fail`, top review metrics, and opt-in model experiment scope
- compact release `product_gate` summary over public path, benchmark, and corpus pass/review state
- compact public workflow payloads by default, with full debug payloads behind `--verbose`
- frozen public compact JSON contracts for `run-workflow`, `smoke-check`, and `assess-pdf`
- explicit backend policy that keeps `hash` as default and keeps cross-encoder/LLM paths opt-in
- model decision gates for runtime comparison and promotion reports, always with `default_change_allowed=false`
- compact release-check summaries with full maintainer payloads behind `--verbose`

Current validation snapshot:

- public CLI tests rerun in the current milestone: green
- public compact workflow contract tests rerun in the current milestone: green
- unknown-document semantics shard rerun in the current milestone: green, `9/9`
- balanced local corpus sanity rerun in the current milestone: green, `12/12` technical and semantic pass, no follow-up actions
- quick local corpus sanity rerun in the current milestone: green, `4/4` technical and semantic pass, no follow-up actions
- quick-latest vs balanced-latest corpus profile compare rerun in the current milestone: review due lower average structure confidence on the larger sample
- corpus review and model-decision focused tests rerun in the current milestone: green
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
- `v1.49.0-v1.53.0`
  - expanded workflow `quality_profile` with processing drilldown over extraction summary, OCR/native path, sections, chunks, and table/form signals
  - added retrieval readiness reasons so warn/fail states point to missing support scope, docs, or support payloads
  - tightened answer trust so weak or unsupported claims produce `review` instead of `pass`
  - added `corpus-profile-compare` for saved quick/balanced/stress snapshot comparison without reprocessing PDFs
- `v1.54.0-v1.58.0`
  - aligned document-level claim scoring with `support_trace` so metadata claims can be supported by document semantics
  - added explicit quality-profile thresholds for processing, semantics, retrieval readiness, and answer trust
  - surfaced public-smoke quality summary inside `public-beta-check`
  - verified the public demo path now reaches `answer_trust=pass` when support-trace metadata backs the answer
- `v1.59.0-v1.63.0`
  - promoted `quality_profile` into the central UX contract with `overall_status`, `statuses`, `reasons`, and `recommended_next_action`
  - normalized follow-up actions for pass, processing failure, semantic failure, retrieval review, and claim-alignment review
  - added tests for pass and low-signal quality profiles so random-PDF results do not require reading the full payload
- `v1.64.0-v1.68.0`
  - added compact `processing_diagnostics` to `inspect-document`, `run-workflow`, and `smoke-check`
  - added processing taxonomy: `native_text_low`, `ocr_required`, `weak_sections`, `table_or_form_heavy`, `layout_uncertain`, `low_text_coverage`
  - separated `technical_processed` from `structurally_reliable`
  - added scan-like, form-like, and table-like taxonomy tests
- `v1.69.0-v1.73.0`
  - added `retrieval_contract_status`, `support_coverage`, and `answer_source_mix`
  - connected retrieval contract consistency into `quality_profile.retrieval_readiness`
  - added mismatch tests for candidate/selected/support/chunk document contracts
- `v1.74.0-v1.78.0`
  - added compact corpus sanity snapshots next to ignored full snapshots
  - added `quick-latest` / `balanced-latest` style snapshot aliases
  - added `corpus_diff_summary` for pass/fail/skip corpus comparison checks
- `v1.79.0-v1.83.0`
  - made `smoke-check --json` and `run-workflow --json` compact by default
  - kept full workflow diagnostics behind `--verbose`
  - added explicit `--compact` payload flag
- `v1.84.0-v1.88.0`
  - added a unified runtime `backend_policy`
  - clarified `runtime-promotion-report.default_decision`
  - kept cross-encoder experimental and LLM synthesis opt-in only
- `v1.89.0-v1.93.0`
  - consolidated docs around the public beta path and moved sprint detail into this plan/log
- `v1.94.0-v1.99.0`
  - beta-freeze validation scope: tests, package/runtime/release checks, compact workflow smoke, and corpus snapshot comparison
- `v2.0.0-v2.4.0`
  - added public `assess-pdf --pdf ... --json` as a compact real-world PDF acceptance layer
  - added acceptance profiles for scanned, form-heavy, table-heavy, short, medium, and long PDFs
  - exposed one-line status fields for processing, semantics, retrieval, answer trust, and next action
  - kept the full workflow payload behind `--verbose`
- `v2.5.0-v2.9.0`
  - reviewed balanced local-corpus failures by bucket instead of by one-off PDF path
  - added reusable unknown-document semantics for statistical tables, web job listings, environmental site records, and institutional correspondence
  - added `unknown_document_semantics_core` and wired it into `release-check`
  - improved balanced corpus sanity from semantic follow-up actions to a green `12/12` technical and semantic pass
- `v3.0.0-v3.4.0`
  - froze compact JSON result keys for `run-workflow`, `smoke-check`, and `assess-pdf`
  - added contract tests that fail if public fields are removed from compact workflow wrappers
  - documented schema-like compact contracts in the CLI reference
  - verified full workflow debug payloads remain behind `--verbose`
- `v3.5.0-v3.9.0`
  - refreshed local quick and balanced corpus snapshots without tracking generated reports
  - verified `corpus-profile-compare --baseline-profile quick-latest --candidate-profile balanced-latest`
  - added compact release `product_gate` output for `public_path`, `benchmark`, and `corpus`
  - added compact corpus failure examples to release summaries when corpus status is review
- `v4.0.0-v4.5.0`
  - added `corpus_review` to `corpus-profile-compare` with pass/review/fail status and top review metrics
  - added `model_experiment_scope` so opt-in model runs are tied to measured corpus review metrics
  - added `model_decision_gate` to runtime comparison and promotion reporting
  - kept hash as the default backend and made every model decision explicitly default-change-disabled


### 6.2. Next Steps

1. Run optional model experiments only when `corpus_review.model_experiment_scope.worth_running` is true.
2. Use `model_decision_gate` to decide whether a backend stays experimental or recommended opt-in.
3. Keep public compact contracts stable while optional model reports evolve.
4. Do not change defaults unless an explicit future default-change gate is added and passes.
5. Continue improving structure-aware baseline first when corpus review points to processing/layout metrics.

## 7. Explicitly Deferred

- deeper `pdfplumber` table schema normalization beyond supplemental table blocks
- mandatory/default cross-encoder reranking
- automated LLM-as-a-judge execution against a configured judge model
- cloud deployment
- multi-document schema extraction
- visual grounding UI
