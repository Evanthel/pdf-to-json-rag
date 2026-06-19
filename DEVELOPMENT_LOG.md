# DEVELOPMENT_LOG

## Purpose

This file records the project at a practical level without mirroring every code edit. It is intended as a concise engineering log rather than a public changelog.

Internal development iterations in this file use `vN.x` labels. Public releases follow semantic versioning starting at `0.1.0-beta`.

## Current State

Current implementation level: `v4.5.0`
Current public version: `0.1.0-beta`
Current package metadata version: `0.1.0`

The project now behaves as a local-first, domain-agnostic `PDF -> JSON -> retrieval -> grounded answer` pipeline with explicit document-intelligence behavior on top of chunk retrieval.

Latest public release-path validation before the current unreleased checkpoint:

- `python -m unittest tests.test_cli_public_surface`
- `package-check`
- `release-check` for the packaged/public surface

Repo-local regression shards currently passing in saved runs or current milestone reruns:

- `query_planning_core`
- `answer_modes_core`
- `document_pipeline_core`
- `structure_chunking_core`
- `section_reconstruction_core`
- `document_selection_core`
- `document_maintenance_core`
- `structured_form_maintenance_core`
- `layout_robustness_core`
- `single_doc_random_pdf_core`
- `table_layout_robustness_core`
- `form_layout_robustness_core`
- `semantic_document_understanding_core`
- `unknown_document_semantics_core`
- `confidence_aware_document_core`
- `trust_policy_document_core`
- `processing_layer_core`
- `processing_strategy_core`
- `retrieval_contract_core`
- `retrieval_synthesis_core`
- `evidence_anchor_core`
- `source_anchor_contract_core`
- `document_family_core`
- `document_facets_core`
- `inventory_coverage_core`
- `relationship_core`

Broad benchmark note after `v4.5.0`:

- the targeted maintainer shard set run in this milestone is green
- the opt-in local-command LLM synthesis and LLM-as-judge paths are covered in unit tests while remaining disabled by default
- strict JSON/fence parsing, provider metadata, answer-claim alignment, prompt/eval contract validation, and opt-in semantic multipass behavior have targeted unit coverage
- `compare-runtime-modes` now compares baseline, sentence-transformers, cross-encoder, and opt-in LLM synthesis paths on the same cases while reporting fallback/runtime availability
- `runtime-check` reports the requested/effective embedding backend, optional model availability, and opt-in runtime state without building an index
- `runtime-check` now also reports the runtime decision: `hash` default, recommended opt-in backend, promotion-snapshot source, and not-default rationale
- `runtime-promotion-report` summarizes the latest saved runtime comparison and promotion gate without rerunning the benchmark, and writes a compact promotion snapshot after green full-suite comparisons
- full-suite `compare-runtime-modes --all-cases --modes baseline,sentence-transformers` with local `all-MiniLM-L6-v2` is green for both modes; sentence-transformers improves recall/MRR and passes the promotion gate while remaining opt-in
- installed-entrypoint verification passes after `python -m pip install .`; `runtime-check`, `doctor`, `create-demo-pdf`, and `smoke-check` work through `pdf-to-json-rag`
- `readme-smoke-check` now replays the installed public README flow in one maintainer command without running benchmark regressions
- `corpus-sanity-check` now returns a deterministic sample manifest with bucket counts, selected digests, and a sample checksum
- `release-check --json` now returns a compact public/maintainer/shard/corpus gate summary by default; use `--verbose` for the full legacy payload
- `public-beta-check --json` aggregates installed README smoke, runtime default decision, corpus quick gate, and compact release summary into one pre-tag check
- workflow payloads now expose `quality_profile` for processing quality, semantic confidence, retrieval readiness, and answer trust
- answer payloads now expose `contract_health` so retrieval path, support scope, selected docs, support docs, and claim-alignment presence are easy to gate
- `quality_profile` now includes processing drilldown, retrieval readiness reasons, and stricter answer trust review status for weak or unsupported claims
- `corpus-profile-compare` compares saved corpus profile snapshots without reprocessing PDFs
- `corpus-profile-compare` now returns `corpus_review` with top review metrics and opt-in model experiment scope
- document-level claim alignment now includes support-trace fragments, so metadata claims can be supported by document semantics instead of only chunk text
- `quality_profile` now carries explicit thresholds, and `public-beta-check` includes the public smoke quality summary
- `quality_profile` now exposes `overall_status`, normalized `statuses`, aggregate `reasons`, and `recommended_next_action`
- `processing_diagnostics` now exposes processing taxonomy, `technical_processed`, and `structurally_reliable`
- answer payloads now expose `retrieval_contract_status`, `support_coverage`, and `answer_source_mix`
- `assess-pdf` exposes the compact real-world PDF acceptance decision for public use
- `unknown_document_semantics_core` is now part of the maintainer gate and passes `9/9`
- balanced local-corpus sanity now passes `12/12` technical and semantic checks with no follow-up actions
- corpus sanity writes compact snapshots and compares them without reprocessing PDFs
- release summaries now expose a compact `product_gate` across public path, benchmark, and corpus pass/review status
- workflow JSON is compact by default; full workflow diagnostics require `--verbose`
- compact JSON contracts for `run-workflow`, `smoke-check`, and `assess-pdf` are now covered by public-surface contract tests
- runtime policy is explicit: `hash` default, sentence-transformers recommended opt-in, cross-encoder experimental, and LLM synthesis opt-in only
- runtime comparison and promotion reports now include `model_decision_gate` with `default_change_allowed=false`
- current broad benchmark scope:
  - `Cases`: `77`
  - `Indexed sample documents`: `25`
- latest saved full rerun after the `v1.15.0` judge-contract checkpoint:
  - `precision@5`: `0.6031`
  - `recall@5`: `1.0`
  - `MRR`: `1.0`
  - `avg_keyword_coverage`: `1.0`
  - `negative_success_rate`: `1.0`
  - `warning_case_count`: `0`
  - `answer_faithfulness_failing_case_count`: `0`
  - `architecture_gates.all_pass`: `true`
- sampled faithfulness audit after the support-trace pass:
  - `sampled_case_count`: `20`
  - `avg_supported_sentence_ratio`: `1.0`
  - `failing_case_count`: `0`
  - `llm_judge_prompt_contract`: `faithfulness_context_judge.v1`
  - `contract_validation.all_pass`: `true`
- `release-check` now includes `document_pipeline_core`, `structure_chunking_core`, `section_reconstruction_core`, `document_selection_core`, `document_maintenance_core`, and `evidence_anchor_core` in the maintainer regression gate.
- `release-check` now also includes `structured_form_maintenance_core`, `layout_robustness_core`, and `single_doc_random_pdf_core`.
- `release-check` now also includes `table_layout_robustness_core` and `form_layout_robustness_core`.
- `release-check` now recommends the current public beta tag: `v0.1.0-beta`.
- the current decision is to keep learned reranking optional; the stronger structure-aware baseline remains the default, while cross-encoder reranking can now be tested locally behind an env flag.
- the new maintenance direction is preserving document-root section context and shrinking the structured-form / document-level branching surface rather than adding heavier retrieval machinery.
- the new robustness direction is exposing simple structure/layout confidence signals and testing single-document behavior on a more diverse sanity slice before considering a heavier learned retrieval layer.
- the new semantics direction is improving document type, purpose, and audience understanding on unfamiliar PDFs before considering a learned reranker.
- the new UX direction is exposing semantic confidence and confidence-aware classification answers instead of only returning a bare heuristic label.
- the new maintainer direction is using the repo-local `pdf/` corpus as a local-only unknown-document sanity source instead of relying only on the curated benchmark.
- the new semantics direction is lowering the share of repo-local unknown PDFs that fall back to `document/reference_lookup`.
- the new local-corpus direction is distinguishing technical success from semantic success with explicit corpus-level pass metrics.
- the current processing direction is moving more structure recovery into extraction-time block typing, text provenance, and section-role traces before touching retrieval.
- the current processing direction also includes layout-signal-aware sections and strategy-aware chunking, so structure is less dependent on answer-time rescue heuristics.
- the current processing direction now includes explicit multi-column reading-order normalization rather than row-interleaving columns.
- the current processing direction now uses relative font-size, bold-font, and TOC-backed signals for heading detection.
- the current processing direction now includes optional `pdfplumber` supplemental table blocks, while deeper table schema normalization remains a later step.
- the current corpus direction now reports bucket-specific failure reasons and follow-up actions, so unknown-document work can target processing, semantics, layout, or trust-policy issues directly.
- the current corpus direction also saves comparable snapshots and validates the bucket/action contract directly.
- the current retrieval direction is separating single-document QA, document-understanding, and cross-document discovery while keeping cross-encoder reranking opt-in until it proves value.
- the current synthesis direction is keeping document selection, support scope, and answer chunks on one shared handoff instead of re-deriving them inside each document-level renderer.
- the current evaluation direction is separating processing, retrieval, and answer-faithfulness signals instead of treating the benchmark as one flat pass/fail surface.
- the current gating direction is turning those layer summaries into explicit architecture gates instead of leaving them as report-only diagnostics.
- the current corpus direction is turning repo-local unknown-document sampling into a real `processing / semantics / trust` gate instead of a descriptive sanity report.
- the current prompt-runtime direction is enforcing strict output parsing and provider boundaries before adding more model behavior.
- the current faithfulness direction is exposing claim/evidence alignment status as diagnostic payload, not as a replacement for human review.

## Delivered Architecture

### Document Processing

- Native extraction with `PyMuPDF`
- OCR fallback with `pytesseract`
- Extraction-time block roles, text provenance, and text-quality signals
- Extraction-time layout signals and per-page processing summaries
- Native/OCR page fusion instead of one global fallback decision
- Document-level JSON artifacts in `data/documents/`
- Chunk-level JSON artifacts in `data/chunks/`
- Reading-order normalization, section heuristics, noise filtering, OCR provenance
- Section roles, text-source profiles, layout signals, and source-block traces carried into chunking and inspection
- Extraction-time metadata including:
  - `summary_cues`
  - `discovery_terms`
  - document facets
  - document families
  - inventory summaries
  - coverage summaries

### Retrieval and Answering

- Local indexing with `ChromaDB`
- Intent-aware retrieval with lightweight reranking
- Adjacent-chunk expansion
- Source-aware locking
- Cross-document source matching
- Query planning that separates:
  - evidence lookup
  - document discovery
  - document-facet questions
  - cross-document comparison
- Answer modes for:
  - grounded evidence answers
  - document overview
  - document routing
  - source listing
  - source justification
  - cross-document comparison
- Lightweight answer contracts for document-level and cross-document answer paths
- Relationship reasoning for overlap, complement, and divergence
- Tool-facing CLI inspection paths for listing documents, inspecting a document, planning a query, and exporting JSON answers

### Evaluation

- Hand-built multi-document benchmark in `data/eval/mvp_eval_cases.json`
- Sampled faithfulness audit in `data/eval/faithfulness_audit_cases.json`
- Per-case debug output with retrieval snapshots, answer traces, and evidence snippets
- Slice-level reporting for structure family, source family, OCR, discovery, and architecture-oriented paths
- Small deterministic regression shards for high-risk paths

## Milestone Summary

### MVP to v1.5

- Built the local-first core: extraction, chunking, indexing, retrieval, grounded answering, and the first evaluation loop
- Added OCR fallback, stronger chunking, and better retrieval behavior
- Introduced the first multi-document and OCR-derived benchmark coverage

### v1.6 to v1.13

- Broadened the benchmark across review papers, scanned material, manuals, questionnaires, and checklist-style appendices
- Hardened the structured-form path
- Added richer evaluation slices, deterministic regressions, and sampled faithfulness checks
- Added cross-document source listing and comparison

### v1.14 to v1.17

- Shifted from single-document evidence answers toward domain-agnostic document discovery
- Added non-medical, public-safe source families
- Introduced document overview, routing, and source justification
- Pushed more routing behavior into extraction-time metadata instead of source-specific runtime heuristics

### v1.18 to v1.20

- Added document facets as a reusable metadata layer
- Added document inventory and query planning
- Added explicit answer modes for discovery vs evidence behavior
- Made comparison/routing answers more document-aware instead of chunk-accidental
- Added architecture-focused evaluation slices and regressions

### v1.21

- Added a shared document-family layer for books, guidance notes, model reports, manuals, forms, and clinical references
- Normalized answer contracts for overview, routing, source justification, comparison, and evidence-style paths
- Added document-level relationship signals for overlap, complement, and divergence
- Added `document_family_core` plus answer-contract-oriented slice checks

### v1.22 to v1.26

- Consolidated document semantics into a shared metadata interpretation layer for facets, coverage, inventory summaries, and document families
- Strengthened inventory-first routing with coverage-aware and rarity-aware shortlist scoring
- Tightened document-level and cross-document answer contracts around reusable answer modes instead of looser answer-time heuristics
- Added dedicated regression coverage for inventory summaries and relationship reasoning
- Added CLI inspection and planning paths that move the repo closer to a publishable local tool instead of a benchmark-only codebase

### v1.27

- Added package-first project metadata and module entry points for a cleaner install/run path
- Unified JSON output contracts across listing, inspection, planning, retrieval, answering, and evaluation commands
- Added a single end-to-end `run-workflow` path for local smoke usage
- Added public-safe `examples/` assets so the user-facing flow is separated from ignored local benchmark PDFs
- Re-validated the planning and answer-mode regressions after the product-surface pass

### v1.28

- Added a packaged `smoke-check` path for first-run validation of `extract -> chunk -> index -> plan -> answer`
- Tightened CLI error contracts so common failures return stable human-readable and JSON diagnostics
- Added public-safe trimmed example JSON outputs for inspect / plan / answer command shapes
- Shifted the docs further toward a first public tool flow instead of the internal benchmark harness

### v1.29-v1.33

- Added an isolated `PDF_TO_JSON_RAG_DATA_DIR` path so the CLI can run cleanly outside the repo-local benchmark workspace
- Added public-surface smoke tests that generate a tiny PDF and validate the packaged CLI path end to end
- Added user-facing release helpers:
  - `help`
  - `doctor`
  - `demo-profile`
  - command aliases
  - `--output` JSON export
- Split public CLI onboarding docs from internal evaluation notes
- Re-ran the public CLI tests, core regression shards, and the full benchmark after the release-facing pass

## Validation Summary

Representative validation that has already been completed:

- native extraction smoke test on `medical/Common_cold_clinincal_evidence.pdf`
- extraction-to-JSON smoke test on the same sample
- chunk generation smoke test on the same sample
- local vector index smoke test
- retrieval smoke test for common-cold queries
- OCR fallback smoke test on synthetic image-only PDFs
- repeated full-benchmark reruns across mixed source families
- repeated regression-shard checks on structured-form, cross-document, document-discovery, document-facet, query-planning, and answer-mode paths
- isolated public-surface CLI smoke tests via `python -m unittest tests.test_cli_public_surface`
- public `doctor`, `demo-profile`, and `help` command checks
- full-benchmark rerun after the `v1.29-v1.33` release-facing pass

### v1.34-v1.41

- Hardened document processing with extraction-time block metadata and chunk-level semantic metadata.
- Reduced brittle retrieval behavior with semantic overlap, structural-reference alignment, and more coverage-aware evidence selection.
- Made the public quickstart self-contained with `create-demo-pdf`, `doctor`, `package-check`, and `release-check`.
- Added packaged-install verification through a temporary wheel-build and clean install path.
- Added public-safe pre-release artifacts and aligned the repo around a first `v0.1.0-beta` release candidate.

### v0.4.7-v0.4.9

- Preserved document-root `section_path` context for inline headings, review-section splits, and other synthetic section boundaries created during chunking.
- Added richer synthetic-section hints so checklist/questionnaire-like inline sections keep more useful structure metadata instead of dropping back to flat report sections.
- Simplified structured-form answer rendering into shared helper families for checklist, legend, follow-up, and lookup patterns.
- Added `structured_form_maintenance_core` to the maintainer regression gate and re-validated the full structure-aware benchmark baseline after the cleanup.

### v0.5.0-v0.5.4

- Added simple `structure_confidence` and `layout_confidence` signals to document, section, and chunk metadata so the pipeline can expose how trustworthy its recovered structure is.
- Hardened single-document overview behavior so document-level answers can still fall back to metadata and selection traces when chunk evidence is sparse but document semantics remain usable.
- Kept the heuristic-first baseline explicit by softening overview phrasing when structure/layout confidence is lower instead of pretending the system is equally certain on every PDF.
- Added `layout_robustness_core` and `single_doc_random_pdf_core` as sanity gates for unfamiliar layouts and single-document behavior on a more diverse slice than the original curated benchmark path.

### v0.5.5-v0.5.9

- Improved table-like and form-heavy chunk reconstruction so semicolon-heavy rows, checklist-style fields, and questionnaire-like segments split more cleanly instead of collapsing into one paragraph chunk.
- Preserved the current heuristic-first baseline by making single-document overview wording more conservative when the recovered structure is weak.
- Added `table_layout_robustness_core` and `form_layout_robustness_core` to the maintainer release gate so unfamiliar layout behavior is tracked explicitly instead of only through the broad benchmark.
- Kept the public release path green while pushing more of the random-PDF risk into targeted layout sanity gates rather than a learned reranker.
- Added a local-only `layout-sanity-check` maintainer path so unfamiliar external PDFs can be exercised in isolated temp workspaces without embedding private file paths into the benchmark assets.

### v0.6.0-v0.6.4

- Improved document typing and purpose inference so unfamiliar financial/admin PDFs classify more specifically as `financial_statement`, `assessment_form`, or `administrative_form` instead of collapsing back to a generic document bucket.
- Split document-level answers into clearer type, purpose, audience, and overview render paths while keeping the default public JSON compact.
- Added `semantic_document_understanding_core` so source-specific type/purpose/audience questions are tracked in the maintainer regression gate.
- Extended the local `layout-sanity-check` path to return overview, type, purpose, and audience answers on unfamiliar PDFs without embedding private files into the benchmark.

### v0.6.5-v0.6.8

- Added semantic confidence signals and compact rationale/warning fields so unfamiliar document classification has a more explicit trust contract.
- Added confidence-aware document answers for source-specific classification questions instead of forcing everything through type/purpose/audience phrasing.
- Added `confidence_aware_document_core` to the maintainer regression gate and revalidated the full benchmark on the new semantic baseline.
- Extended `layout-sanity-check` to return confidence answers and semantic-confidence metadata for unfamiliar PDFs.

### v0.6.9-v0.7.0

- Added explicit document-classification rationale and classification-limits answers on top of the existing semantic confidence layer.
- Added `trust_policy_document_core` so the maintainer regression gate checks not only confidence answers, but also why a classification is being made and what its current limits are.
- Added a local-only `corpus-sanity-check` maintainer path that samples the repo-local `pdf/` corpus through `lcwa_gov_pdf_metadata.csv`.
- Hardened the corpus loader against non-UTF-8 metadata and filtered out obviously broken `pages=0` artifacts before sampling.
- Added a fallback chunk emission path for very short documents so short form-like PDFs do not fail with `No chunks provided for indexing.`

### v0.7.1-v0.7.4

- Improved unknown-document typing and purpose inference on the repo-local `pdf/` corpus for:
  - registration forms
  - court opinions
  - government bulletins
  - inspection-style records
- Reduced the share of sampled local-corpus PDFs that collapse to `document/reference_lookup` by using stronger content cues and better purpose fallbacks.
- Added corpus-level semantic pass metrics so `corpus-sanity-check` now reports:
  - `technical_all_pass`
  - `semantic_all_pass`
  - specific-type / specific-purpose rates
  - low-confidence and trust-limited rates
- Kept the public heuristic-first baseline stable while making unknown-document semantics more useful on the local corpus.

### v0.8.0

- Added an extraction-time block model with roles such as `heading`, `table_like`, `key_value`, `checklist_item`, and `form_field`.
- Added per-block `text_source` and `text_quality_score` metadata plus native/OCR page fusion so source choice is less brittle.
- Reworked saved document and chunk artifacts to carry:
  - `section_role`
  - `source_block_ids`
  - `source_block_roles`
  - `block_role_profile`
- Extended `inspect-document` so the processing layer is inspectable through:
  - `extraction_summary.block_role_counts`
  - `extraction_summary.text_source_counts`
  - per-section role/source-block traces
- Added `processing_layer_core` so block typing, section-role recovery, and chunk provenance are covered by the maintainer regression gate.

### v0.9.0

- Split retrieval into explicit contracts for:
  - `single_document_qa`
  - `document_understanding`
  - `cross_document_discovery`
- Aligned retrieval filtering, candidate backfill, diversification, and neighbor expansion to those contracts instead of one shared fallback path.
- Exposed a compact `retrieval_contract` block in answer traces so the current retrieval path is inspectable in CLI JSON output.
- Added `retrieval_contract_core` to the maintainer regression gate so answer-path separation is tested directly instead of only through the broad benchmark.

### v1.0.0

- Added extraction-time layout signals and per-page processing summaries so the document layer has a clearer view of form-like, table-like, list-dense, and multi-column-like pages.
- Enriched sections with:
  - `block_count`
  - `text_source_profile`
  - `layout_signals`
- Upgraded chunks to carry:
  - explicit `chunk_strategy`
  - `layout_signals`
  - averaged `text_quality_score`
- Made strategy-aware chunking part of the processing baseline instead of a loose collection of structure-heavy flush heuristics.
- Added `processing_strategy_core` to the maintainer regression gate so structure-aware chunk strategies are covered directly.

### v1.1.0

- Added a shared `document_synthesis` handoff so retrieval and document-level answering agree on selected docs, support scope, and answer chunks.
- Simplified overview, routing, listing, justification, and compare renderers to consume one synthesis context instead of rebuilding support scope per mode.
- Added `retrieval_synthesis_core` to the maintainer regression gate so the retrieval-to-answer handoff is covered directly.

### v1.2.0

- Split the saved evaluation report into explicit `processing`, `retrieval`, and `answer_faithfulness` layers.
- Added per-case layer status records so failures can be attributed to the right stage instead of only surfacing as broad benchmark warnings.
- Exposed the same layer summary through `evaluate-mvp --json` and the CLI text output.

### v1.3.0

- Added `layer_stability` thresholds for the `processing`, `retrieval`, and `answer_faithfulness` layers.
- Added `architecture_gates` so full-suite and partial-suite evaluations can return an explicit gate decision instead of only a descriptive report.
- Exposed those gates through `evaluate-mvp --json` and the CLI text output.

### v1.4.0

- Added corpus-layer summaries for `processing`, `semantics`, and `trust` inside `corpus-sanity-check`.
- Added a corpus architecture gate so repo-local unknown-document sampling returns an explicit decision instead of only rates and counts.
- Surfaced that local corpus gate inside `release-check` as a local-only advisory signal.

### v1.5.0

- Added bucket-level diagnostics to `corpus-sanity-check` for technical, semantic, specificity, confidence, and trust-limited rates.
- Added deterministic follow-up actions so local unknown-document sampling points to the next maintenance focus instead of only reporting aggregate failure.
- Surfaced local corpus follow-up counts in text `release-check` output.

### v1.6.0-v1.6.4

- Added corpus sample profiles: `quick`, `balanced`, and `stress`.
- Added saved corpus sanity snapshots under `data/eval/corpus_sanity_snapshot.json`.
- Added concrete failure examples to bucket-level follow-up actions.
- Added a corpus contract gate for bucket diagnostics, follow-up actions, and bucket architecture gate consistency.
- Updated docs to make profile-based corpus checks the default maintainer path.

### v1.7.0

- Added explicit multi-column reading-order normalization based on x-coordinate clusters.
- Applied bbox-aware ordering in `chunking.normalize_reading_order` for saved extracted blocks.
- Added regression coverage for column-by-column ordering on two-column page layouts.

### v1.8.0

- Added native PyMuPDF font metadata to extracted blocks.
- Added relative font-size, bold-font, and TOC-backed heading signals during extraction.
- Preserved font metadata through native JSON serialization and block loading for chunking.

### v1.9.0

- Added an optional `pdfplumber` table probe that reports availability and per-page table counts when installed.
- Exposed `pdfplumber_available` as an optional `doctor` capability.
- Added the `tables` package extra for installs that want the optional table probe path.

### v1.10.0

- Converted the optional `pdfplumber` path from probe-only into supplemental `table_like` block generation.
- Normalized extracted table rows into pipe-separated table text so existing chunking can classify them as table chunks.
- Kept the public install path stable when `pdfplumber` is not installed.

### v1.11.0

- Added optional cross-encoder reranking behind `PDF_TO_JSON_RAG_USE_CROSS_ENCODER=1`.
- Kept lightweight reranking as the default and fallback path when the model or dependency is unavailable.
- Exposed the active rerank backend in chunk payloads and added unit coverage for opt-in and fallback behavior.

### v1.12.0

- Added a rerank pass over the neighbor-expanded chunk set before answer synthesis.
- Preserved separate `initial_retrieval_rank` and `expanded_context_rank` signals so retrieval diagnostics can distinguish candidate retrieval from context ordering.
- Reused the same optional cross-encoder path and lightweight fallback for expanded context ranking.

### v1.13.0

- Added evidence-intent planning for treatment subquestions that were previously left as `generic`.
- Routed vitamin C normal-population null-effect and cold-stress subgroup-benefit queries to the vitamin C source through the retrieval contract.
- Restored `evidence_anchor_core` to green after the expanded-context rerank work exposed those retrieval-anchor gaps.

### v1.14.0

- Added an LLM-ready synthesis prompt contract over selected answer chunks.
- The prompt contract requires context-only answers, chunk-ID citations, and abstention when support is insufficient.
- Exposed prompt-contract metadata in answer traces so human review and future LLM-as-judge scoring can inspect the exact grounding boundary without invoking an LLM by default.

### v1.15.0

- Added an LLM-as-judge prompt contract for faithfulness scoring.
- The judge contract compares the final answer against source context only, forbids outside knowledge, and requires strict JSON output.
- Embedded judge-contract metadata in sampled faithfulness audit records and summaries while keeping model execution disabled by default.

### v1.16.0

- Added a provider-agnostic local command runtime for prompt-based LLM hooks.
- `PDF_TO_JSON_RAG_LLM_COMMAND` can now run grounded synthesis over selected answer chunks and replace the final answer only when explicitly configured and successful.
- `PDF_TO_JSON_RAG_JUDGE_COMMAND` can now run a strict-JSON faithfulness judge and store parsed judge diagnostics in evaluation records.
- Kept both runtime hooks disabled by default, with public metadata showing configured/invoked/status/usage state.

### v1.17.0

- Added a local strict JSON/fence parser for opt-in LLM and judge output.
- Accepted raw JSON objects or one clean `json`/`jsonc` fenced block.
- Rejected empty output, multiple fenced blocks, non-JSON fences, text outside the fence, invalid JSON, and non-object payloads when an object is required.

### v1.18.0

- Added answer-claim/evidence alignment diagnostics.
- Answer traces now report claim count, supported/weak/unsupported counts, supported ratio, and per-claim support previews.
- Faithfulness audit records carry the same alignment status alongside sampled sentence-support checks.

### v1.19.0

- Split the prompt runtime behind a small provider protocol.
- Kept the current env-command subprocess provider as the only implementation and preserved the existing public env vars.
- Runtime payloads now expose `provider_id` and `provider_kind`.

### v1.20.0

- Added a prompt/eval contract validation gate for sampled faithfulness records.
- The gate checks judge template identity, no-outside-knowledge policy, strict-JSON requirement, source-context availability, and parser-contract reporting for invoked runtimes.

### v1.21.0

- Added optional low-confidence semantic multipass behind `PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS=1`.
- The second pass enriches low-confidence document-facet review with existing title/TOC/discovery metadata and only accepts the reviewed facets when confidence improves.
- Default document semantics remain single-pass and unchanged unless the env flag is set.

### v1.22.0

- Added `compare-runtime-modes` for side-by-side measurement of the current baseline, optional sentence-transformer embeddings, optional cross-encoder reranking, and opt-in local-command LLM synthesis.
- The comparison writes `data/eval/runtime_mode_comparison.json` and reports pass/fail counts, retrieval metrics, keyword coverage, embedding backend, rerank backend counts, cross-encoder fallback counts, and LLM usage counts.
- Tightened optional model loading so local comparisons do not try network downloads unless `PDF_TO_JSON_RAG_ALLOW_MODEL_DOWNLOAD=1` is explicitly set.

### v1.23.0

- Added `--all-cases` to `compare-runtime-modes` so runtime comparisons can run against the full evaluation suite instead of only the small comparison subset or a shard.
- Added a sentence-transformer promotion gate that requires an active sentence-transformer backend, no pass-count regression, recall not lower than baseline, MRR not lower than baseline, and warning count not higher than baseline.
- Ran the full 77-case comparison with local `all-MiniLM-L6-v2`: retrieval recall improved from `0.9923` to `1.0` and MRR improved from `0.9897` to `1.0`, but `ajmedp_frostbite_severe_zone` regressed from pass to answer failure. This was fixed in `v1.24.0`.

### v1.24.0

- Fixed source-anchored grounded evidence synthesis so named-source queries can use preferred-document hits from both `top_k_hits` and neighbor-expanded context.
- Prevented dominant-document rescue from overriding explicit source anchors when top-k is polluted by another document family.
- Added a regression test for the AJMedP severe-frostbite case with majority off-source context.
- Reran the full 77-case `baseline,sentence-transformers` comparison with local `all-MiniLM-L6-v2`: both modes pass `77/77`; sentence-transformers reaches `recall@5=1.0`, `MRR=1.0`, `warning_case_count=0`, and `promotion_gates.sentence-transformers.promotable=true`.

### v1.25.0-v1.29.0

- Added `runtime-check` for explicit backend/runtime diagnostics without building an index.
- Added `runtime-promotion-report` to summarize the latest saved runtime comparison and sentence-transformer promotion gate without rerunning the benchmark.
- Added `PDF_TO_JSON_RAG_EMBEDDING_BACKEND=hash|sentence-transformers|auto`; default remains deterministic hash, while legacy `PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1` still works.
- Added `source_anchor_contract_core` to make named-source evidence behavior easy to run as a compact shard.
- Included the source-anchor contract shard in `release-check` and updated docs around backend selection, runtime diagnostics, and promotion readiness.

### v1.30.0-v1.34.0

- Added install context to `runtime-check` so source vs installed entrypoint diagnostics are explicit.
- Added normalized `index.embedding` payloads to `build-index`, `run-workflow`, and `smoke-check` outputs.
- Added saved runtime promotion snapshots at `data/eval/runtime_promotion_snapshot.json` after green full-suite sentence-transformer gates.
- Documented local `all-MiniLM-L6-v2` as the recommended opt-in embedding backend while keeping deterministic hash as the default.
- Refreshed quickstart/reference docs around backend fallback visibility and promotion snapshots.

### v1.35.0-v1.38.0

- Clarified `.gitignore` policy: generated eval reports stay ignored, while `runtime_promotion_snapshot.json` remains a tracked promotion checkpoint.
- Verified the installed console script after `python -m pip install .`; `runtime-check` reports the installed `site-packages` module path and the public demo smoke path passes.
- Performed a docs final pass to keep one canonical backend policy: hash default, local `all-MiniLM-L6-v2` recommended opt-in, cross-encoder and LLM hooks experimental.

### v1.39.0-v1.43.0

- Clarified the release-state policy: public docs use `0.1.0-beta`, while package metadata remains PEP440-compatible `0.1.0`.
- Added `readme-smoke-check` as a repeatable installed-entrypoint public smoke gate covering install, init, doctor, demo PDF generation, smoke-check, and runtime-check.
- Extended `package-check` so the packaged wheel validation reuses the same installed README flow and includes runtime-check status.
- Added deterministic corpus sample manifests to `corpus-sanity-check` so quick/balanced/stress local-corpus runs expose bucket counts, selected digests, and a checksum.
- Added `runtime_decision` output to `runtime-check`, including `default_backend`, `recommended_opt_in_backend`, and the reason sentence-transformers is not the default.
- Made `release-check --json` compact by default with explicit pass/fail/skip records for public, maintainer, shard, runtime, and corpus gates; the full payload remains available with `--verbose`.
- Re-ran final package and source release gates after installed-entrypoint verification.

### v1.44.0-v1.48.0

- Added `public-beta-check` as a single pre-tag gate over installed README smoke, runtime decision, corpus quick, and compact release summary.
- Added answer `contract_health` to make retrieval/synthesis support contracts directly gateable in public JSON.
- Added workflow `quality_profile` so unknown PDFs report processing, semantic, retrieval, and answer-trust status.
- Extended smoke checks and public-surface tests around these contract fields.

### v1.49.0-v1.53.0

- Expanded workflow `quality_profile` with processing drilldown from extraction summary, OCR/native path, section/chunk counts, and table/form signals.
- Added retrieval readiness reasons so diagnostics say why a path is `warn` or `fail`.
- Tightened answer trust: weak or unsupported claims now produce `review`, not `pass`.
- Added `corpus-profile-compare` to compare saved quick/balanced/stress corpus snapshots without reprocessing PDFs.

### v1.54.0-v1.58.0

- Aligned document-level claim scoring with `support_trace`, so metadata claims such as audience/type/purpose can be supported by document semantics.
- Added explicit `quality_profile` thresholds for processing, semantic confidence, retrieval readiness, and answer trust.
- Surfaced public smoke quality summary inside `public-beta-check`.
- Verified the public demo smoke path now reports `answer_trust=pass` when support-trace metadata backs the answer.

### v1.59.0-v1.63.0

- Promoted `quality_profile` into the main random-PDF UX contract with `overall_status`, `statuses`, `reasons`, and `recommended_next_action`.
- Added normalized follow-up actions for processing failures, semantic failures, retrieval-contract review, and claim-alignment review.
- Added public-surface tests for both pass and low-signal quality profiles.

### v1.64.0-v1.68.0

- Added compact `processing_diagnostics` to `inspect-document`, `run-workflow`, and `smoke-check`.
- Added processing taxonomy for `native_text_low`, `ocr_required`, `weak_sections`, `table_or_form_heavy`, `layout_uncertain`, and `low_text_coverage`.
- Split processing state into `technical_processed` and `structurally_reliable`.
- Added scan-like, form-like, and table-like taxonomy tests.

### v1.69.0-v1.73.0

- Added `retrieval_contract_status`, `support_coverage`, and `answer_source_mix` to answer payloads.
- Connected retrieval contract consistency to `quality_profile.retrieval_readiness`.
- Added tests for support-document and answer-chunk mismatch diagnostics.

### v1.74.0-v1.78.0

- Added compact corpus sanity snapshots next to ignored full snapshots.
- Added `quick-latest`, `balanced-latest`, `stress-latest`, and `latest` snapshot aliases.
- Added `corpus_diff_summary` with pass/fail/skip checks for snapshot comparison.

### v1.79.0-v1.83.0

- Made workflow JSON compact by default for `smoke-check` and `run-workflow`.
- Kept full workflow debug payloads behind `--verbose`.
- Added an explicit `--compact` flag for public workflow payloads.

### v1.84.0-v1.88.0

- Added unified `backend_policy` to `runtime-check`.
- Added `default_decision` to `runtime-promotion-report`.
- Kept sentence-transformers opt-in, cross-encoder experimental, and LLM synthesis off by default.

### v1.89.0-v1.93.0

- Consolidated public docs around compact workflow output, backend policy, corpus snapshots, and beta validation.
- Kept detailed sprint history in `DEVELOPMENT_LOG.md` and `project-plan.md`.

### v1.94.0-v1.99.0

- Beta freeze scope: no learned reranker/default LLM changes.
- Final validation focuses on tests, package/runtime/release checks, compact workflow smoke, and corpus snapshot comparison.

### v2.0.0-v2.4.0

- Added public `assess-pdf --pdf ... --json` over the existing workflow path.
- Added compact acceptance fields: `overall_status`, `processing_status`, `semantic_status`, `retrieval_status`, `answer_trust`, and `recommended_next_action`.
- Added acceptance profiles for scanned, form-heavy, table-heavy, short, medium, and long PDFs.
- Added diagnostic messages for structurally weak processing, semantic guesses, OCR/scan paths, table/form layouts, and document-semantics-only support.
- Kept full workflow diagnostics behind `--verbose`.

### v2.5.0-v2.9.0

- Reviewed balanced corpus failures by bucket rather than adding one-off PDF exceptions.
- Added reusable public-record semantics for statistical tables, web job listings, environmental site records, and institutional correspondence.
- Added `unknown_document_semantics_core` to cover unfamiliar-document type, purpose, audience, and confidence behavior in the maintainer regression gate.
- Reran focused local-corpus failures and confirmed all four prior semantic failures now classify as specific high-confidence document types.
- Reran balanced corpus sanity and confirmed `12/12` technical pass, `12/12` semantic pass, architecture gate pass, and no follow-up actions.

### v3.0.0-v3.4.0

- Added public compact contract constants for `run-workflow`, `smoke-check`, `assess-pdf`, compact document, compact index, and compact answer payloads.
- Added public-surface tests that fail if compact workflow wrapper keys are removed or debug-only payloads leak into default JSON.
- Documented the schema-like compact contract in `docs/CLI_REFERENCE.md`.
- Verified debug-only fields such as `artifacts`, full `quality_profile`, `top_k_hits`, `expanded_hits`, and `evidence` remain behind `--verbose`.

### v3.5.0-v3.9.0

- Refreshed local quick and balanced corpus snapshots without tracking generated reports.
- Verified `corpus-profile-compare --baseline-profile quick-latest --candidate-profile balanced-latest`; the larger balanced sample currently returns review because average structure confidence is lower than quick.
- Added compact `product_gate` output to `release-check --json` for `public_path`, `benchmark`, and `corpus`.
- Added compact corpus failure examples to release summaries when corpus status is review.

### v4.0.0-v4.5.0

- Added `corpus_review` to `corpus-profile-compare` so saved corpus snapshots produce `pass`, `review`, or `fail` rather than only metric deltas.
- Added top review metrics and `model_experiment_scope` so optional model work is tied to measured corpus review/failure signals.
- Added `model_decision_gate` to runtime comparison and runtime promotion reporting.
- Kept `hash` as the default backend; sentence-transformers can be recommended opt-in, cross-encoder remains experimental opt-in, and LLM synthesis remains opt-in only.

### v0.1.1-v0.1.2

- Fixed post-beta public-path issues in `document_overview`, `inspect-document`, and `doctor`.
- Added clearer next-step guidance in the CLI after initialization, extraction, chunking, and indexing.
- Reworked the public quickstart so the shortest path goes through `smoke-check` and `run-workflow` before the manual multi-step path.
- Aligned install/onboarding docs around `python -m pip install .` and a clear local-development fallback.

### v0.2.0

- Added extraction-time document sections as a reusable structure layer in the saved document JSON.
- Reworked chunk construction so section boundaries come from extraction-time structure and chunk metadata inherits section summaries and coverage terms.
- Made retrieval scoring more inspectable by exposing quality, semantic, structural, metadata, and rank signals in the runtime payload.
- Tightened document-overview answers so they render a cleaner section-aware summary instead of mostly replaying inventory strings.
- Changed the default embedding path to deterministic local fallback, with `sentence-transformers` now opt-in for stronger local embeddings.

### v0.2.1-v0.2.4

- Fixed install-time path resolution so the CLI defaults to a user data directory outside the repo instead of writing under the Python installation.
- Added packaged example assets and install-safe loading for `demo-profile`, `doctor`, `create-demo-pdf`, and related public-surface commands.
- Hardened `package-check` so it validates a real installed wheel from a clean temporary workspace instead of accidentally relying on the source checkout.
- Split `release-check` into clearer layers:
  - public-surface smoke
  - maintainer package/test gates
  - optional benchmark-only regressions
- Re-validated the packaged/public release path after these changes and confirmed that the remaining red status is limited to a small set of repo-local regression cases after the `v0.2.0` refactor.

### v0.2.5

- Recovered the core repo-local failures in:
  - `query_planning_core`
  - `answer_modes_core`
  - `document_family_core`
  - `relationship_core`
- Added a narrow anchor-recovery retrieval path for high-risk intents where embedding-first hits were hiding the correct chunk inside the right document.
- Tightened section-aware and source-aware scoring for:
  - common-cold symptom lookup
  - questionnaire context lookup
  - AJMedP hypothermia predisposition lookup
  - LBDL document routing
  - vitamin-C vs echinacea prevention comparison
- Revalidated the public release path and maintainer gates after the recovery pass.
- Re-ran the broad benchmark and confirmed that full-benchmark parity is still the next internal target, even though the public release path and maintainer shard set are now green.

### v0.2.6

- Tightened `release-check` so internal benchmark regressions only run when the active data root really contains full benchmark assets, instead of any partial inventory/index pair.
- Fixed unsupported-query behavior for:
  - `document_routing` queries like lease/rent clauses
  - treatment-style queries that ask about unsupported targets like influenza
- Fixed evaluator assumptions so document-level benchmark cases can use `relevant_doc_ids` without crashing the full rerun.
- Recovered the final broad-benchmark warnings on the section-aware architecture, including:
  - `symptoms`
  - `antibiotics`
  - `cmaj_zinc_prevention`
  - `wat_rhinovirus_most_common`
  - `source_listing_vitamin_c_and_echinacea`
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `release-check`
  - the full 67-case benchmark
- End state:
  - public release gates: green
  - maintainer shard set: green
  - full 67-case benchmark: green
  - sampled faithfulness audit: still flags two document-level cases, which is now the next internal target

### v0.2.7

- Added explicit `support_trace` payloads for document-level answer modes instead of relying on mostly empty chunk-evidence fields.
- Extended document-level answer assembly so overview, routing, source listing, source justification, and cross-document comparison expose structured support fragments derived from inventory, facets, sections, and matched cues.
- Reworked the sampled faithfulness audit so document-level answers are judged against their support contract rather than only chunk-evidence sentences.
- Improved human-readable CLI rendering so document-level answers show `Support:` instead of a misleading empty `Evidence:` block.
- Re-ran:
  - `python -m unittest tests.test_cli_public_surface`
  - `release-check`
  - the full 67-case benchmark
- End state:
  - public release gates: green
  - maintainer shard set: green
  - full 67-case benchmark: green
  - sampled faithfulness audit: green

### v0.3.0

- Replaced the most brittle literal query routing with a feature-based planner that returns:
  - per-mode scores
  - explicit chosen rationale
  - shortlist-aware document metadata
- Simplified inventory shortlist scoring into four inspectable buckets:
  - title/label overlap
  - semantic/discovery overlap
  - facet/purpose/family fit
  - rarity/distinctive bonus
- Split document-level retrieval into:
  - candidate-document selection
  - chunk retrieval inside those candidates
- Unified document-level answer building around shared support entries instead of separate hand-built branches for overview, routing, listing, justification, and comparison.
- Shortened the default CLI JSON surface and pushed full retrieval/debug payloads behind `--verbose`.
- Added `document_pipeline_core` to keep the simplified document-level path under regression coverage.
- Recovered the regressions reopened by the simplification pass, including:
  - `lbdl_document_routing_backpropagation`
  - `compare_vitamin_c_vs_echinacea_prevention`
  - `wat_antibiotics_review`
  - `antibiotics`
  - `vitamin_c_normal_populations`
  - `vitamin_c_cold_stress`
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `release-check`
  - `query_planning_core`
  - `answer_modes_core`
  - `document_pipeline_core`
  - the targeted rerun covering all reopened warning cases

### v0.3.1

- Froze the simplified `v0.3.0` baseline by rerunning the full broad benchmark and restoring a green 67-case report.
- Tightened the remaining source-anchor-sensitive evidence lookup seams for:
  - `antibiotics`
  - `wat_antibiotics_review`
  - `vitamin_c_normal_populations`
  - `vitamin_c_cold_stress`
- Added `document_pipeline_core` to the default `release-check` maintainer regression gate.
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `release-check`
  - `document_pipeline_core`
  - the full broad benchmark
- Made the architecture decision explicit: a learned reranker is still not justified on top of the current simplified baseline.

### v0.3.2-v0.3.4

- Centralized source-anchor resolution so retrieval, answering, and evaluation all reuse the same preferred-source and matched-source helpers.
- Added `evidence_anchor_core` as a compact regression shard for the highest-risk source-sensitive evidence cases:
  - `antibiotics`
  - `wat_antibiotics_review`
  - `vitamin_c_normal_populations`
  - `vitamin_c_cold_stress`
  - `echinacea_overall_conclusion`
  - `ct_follow_up_improvement`
  - `cmaj_zinc_prevention`
- Extended the default `release-check` maintainer regression gate to include `evidence_anchor_core`.
- Fixed `release-check` metadata so the recommendation now points at the real public beta tag instead of the stale pre-`v0.3.x` suggestion.
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `evaluate-regression --shard evidence_anchor_core`
  - `release-check`
- Kept the architecture decision unchanged: preserve the simplified heuristic baseline and continue to defer learned reranking until a real failure pattern justifies it.

### v0.3.5-v0.3.8

- Simplified the remaining document-level support path by reusing one support-entry builder across overview, routing, listing, justification, and comparison answers.
- Strengthened structure metadata flowing from extraction into chunking and retrieval:
  - section heading levels are inferred more explicitly
  - chunk records now carry `section_content_hints`
  - chunk records now classify `chunk_type` more explicitly for table-heavy/header-like cases
- Tightened chunk boundaries for structure-sensitive content, especially questionnaire-like numbered sections and table-heavy transitions.
- Extended support traces so they carry section summaries, section hints, and answer-shaped document facts that line up better with the rendered document-level answers.
- Added `structure_chunking_core` as a compact regression shard covering manuals, questionnaires, checklist-like appendices, and table-heavy support cases.
- Extended the default `release-check` maintainer regression gate to include `structure_chunking_core`.
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `evaluate-regression --shard structure_chunking_core`
  - `release-check`
  - full `evaluate-mvp --top-k 5 --json`
- Result:
  - broad benchmark still green
  - sampled faithfulness audit green again
  - no evidence that a learned reranker is needed yet

### v0.3.9

- Added an explicit `document_selection` contract to the document-level answer trace so planner, retrieval, and answer rendering now hand off one inspectable selection payload.
- Centralized document-level selection around:
  - `candidate_doc_ids`
  - `ranked_doc_ids`
  - `selected_doc_ids`
  - `primary_doc_id`
  - `strategy`
- Simplified document-level answer assembly so overview, routing, listing, justification, and comparison answers consume the same selected-document payload instead of recomputing their own shortlist decisions.
- Kept the public JSON payload compact while exposing the new `document_selection` trace in the default answer trace contract.
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `evaluate-regression --shard document_pipeline_core --top-k 5 --json`
  - `release-check --json`
- Verified directly on the source checkout that the previously reopened benchmark cases:
  - `transmission`
  - `definition`
  - `causes`
  - `antibiotics`
  - `source_listing_nonmedical_learning_and_incident_response`
  still pass their retrieval and keyword checks under the new handoff contract.
- Maintainer note:
  - after editing `src/`, run broad benchmark commands through `PYTHONPATH=src python -m pdf_to_json_rag ...` or reinstall the package first, otherwise the console script may still point at an older installed build.

### v0.4.0-v0.4.3

- Strengthened extraction-time section reconstruction so saved section records now carry:
  - `parent_section_id`
  - `section_path`
  - `section_kind`
- Tightened section detection for numbered, question-like, checklist-style, appendix-like, and table-oriented headings.
- Pushed the richer structure layer through chunking, indexing, retrieval, and verbose answer traces:
  - chunks now carry `section_parent_id`
  - `section_path`
  - `section_kind`
  - more explicit `checklist` chunk typing for structure-sensitive sections
- Improved structure-aware chunk boundaries for questionnaire-style, checklist-like, and section-transition-heavy content.
- Extended the document-level selection contract with a compact shortlist breakdown and richer support-trace structure fields such as:
  - `section_paths`
  - `section_kinds`
- Added two new maintainer regression shards:
  - `section_reconstruction_core`
  - `document_selection_core`
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `evaluate-regression --shard section_reconstruction_core --top-k 5 --json`
  - `evaluate-regression --shard document_selection_core --top-k 5 --json`
  - `release-check --json`
  - full `evaluate-mvp --top-k 5 --json`
- End state:
  - public release gates: green
  - maintainer shard set: green
  - full 67-case benchmark: green
  - sampled faithfulness audit: green
  - learned reranking remains deferred because the stronger structure-aware baseline is still green without it

### v0.4.4+

- Reduced document-level maintenance cost in `answering.py` by centralizing:
  - candidate-doc resolution
  - ranked-doc selection
  - selected-doc strategy selection
  - final answer assembly for retrieval and non-retrieval paths
- Added a shared answer-chunk filtering helper so retrieval-time document locking and source-anchored filtering no longer live inline in one long answer path.
- Kept the user-facing document-level contract unchanged while making the internal selection/assembly path shorter and easier to inspect.
- Added `document_maintenance_core` as a compact regression shard covering:
  - overview
  - routing
  - source listing
  - source justification
  - cross-document comparison
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `evaluate-regression --shard document_maintenance_core --top-k 5 --json`
  - `release-check --json`
  - full `evaluate-mvp --top-k 5 --json`
- End state:
  - public release gates: green
  - maintainer shard set: green
  - full 67-case benchmark: green
  - sampled faithfulness audit: green
  - learned reranking remains deferred because the lower-maintenance heuristic baseline is still sufficient

### v0.4.6

- Split the remaining large document-level renderer into smaller shared mode-specific helpers for:
  - overview
  - routing
  - source justification
  - source listing
  - cross-document comparison
- Added shared trace/contract helpers so answer-contract construction and comparison support items no longer repeat across multiple answer modes.
- Kept the public answer contract unchanged while making the internals more modular and easier to extend.
- Revalidated:
  - `python -m unittest tests.test_cli_public_surface`
  - `evaluate-regression --shard document_maintenance_core --top-k 5 --json`
  - `release-check --json`
  - full `evaluate-mvp --top-k 5 --json`
- End state:
  - public release gates: green
  - maintainer shard set: green
  - full 67-case benchmark: green
  - sampled faithfulness audit: green
  - learned reranking remains deferred because the simpler mode-split baseline is still sufficient

## Deferred Features

Still intentionally deferred:

- deeper `pdfplumber` table schema normalization beyond supplemental table blocks
- mandatory/default cross-encoder reranking
- automated LLM-as-a-judge execution against a configured judge model
- cloud deployment
- multi-document schema extraction
- visual grounding UI

These stay out of scope unless future architecture work exposes failures the current local stack cannot absorb.
