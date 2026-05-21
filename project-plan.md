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

## 6.10-6.12. Structured-form Hardening

These iterations consolidated the form/checklist path instead of growing more one-off fixes.

Completed scope:

1. Added opioid-appendix and related checklist-style cases to prove the path generalizes beyond a single questionnaire source.
2. Reworked the assist logic into reusable pattern families and shared declarative intent metadata.
3. Added deterministic regression shards and stronger slice-stability checks for high-risk structured-form cases.

Outcome:

- the structured-form path became shorter, more inspectable, and less document-specific
- the benchmark stayed warning-free while `pdfplumber`, cross-encoder reranking, and LLM-as-a-judge remained deferred

## 6.13-6.17. Document Discovery Expansion

These iterations moved the project from single-document evidence lookup toward multi-source discovery and routing.

Completed scope:

1. Added cross-document intents for source listing, comparison, and justification.
2. Added several non-medical, public-safe source families so routing no longer depended on one domain or one benchmark style.
3. Pushed more discovery behavior into extraction-time metadata such as `summary_cues`, `discovery_terms`, and cleaner source labels.
4. Expanded evaluation with mixed-domain routing, ambiguous multi-source cases, and dedicated discovery regressions.

Outcome:

- the benchmark expanded from 48 to 64 cases and stayed warning-free
- the system learned document overview, routing, source listing, and source justification instead of only chunk-level evidence lookup

## 6.18-6.20. Discovery Architecture Hardening

These iterations shifted the project away from benchmark growth and toward a more explicit document-intelligence layer.

Completed scope:

1. Added extraction-time document facets plus reusable metadata summaries for type, purpose, audience, evidence style, and structure style.
2. Built a document inventory and query-planning layer so discovery and routing can shortlist candidate files before chunk retrieval.
3. Added an explicit answer-mode layer for overview, routing, source-listing, source-justification, comparison, and evidence lookup.
4. Improved comparison and routing answers so they can explain differences in purpose, audience, structure style, and evidence style across mixed-domain sources.
5. Added compact architecture-focused regressions and slices such as `document_facets_core`, `query_planning_core`, and `answer_modes_core`.

Outcome:

- the benchmark remains warning-free at 67 cases across 19 indexed documents
- the project now has a stable facet-driven, inventory-first, query-planned path for document overview, routing, source justification, and cross-document comparison

## 6.21. v1.21 Milestones

This iteration hardened the document-intelligence layer itself rather than widening the benchmark.

Completed scope:

1. Added a compact document-family classifier layer so books, guidance notes, model reports, manuals, forms, and clinical references can be reasoned about through one shared abstraction.
2. Normalized answer contracts for evidence lookup, overview, routing, and comparison so downstream evaluation depends less on free-form wording.
3. Added document-level relationship signals so comparison answers can say whether sources complement, overlap, or diverge.
4. Added a compact evaluation slice for document-family reasoning and answer-contract stability without introducing another new source family.
5. Re-checked deferred features and kept the current local retrieval plus lightweight reranking stack because the family/contract pass did not expose failures that justified heavier additions.

Outcome:

- the benchmark remains warning-free at 67 cases across 19 indexed documents
- `document_family_core`, `query_planning_core`, and `answer_modes_core` all pass
- document-level and cross-document answers now expose a smaller, more inspectable contract

## 6.22-6.26. Document-Intelligence Consolidation

These five iterations shifted the project toward a publishable tool architecture instead of wider benchmark churn.

Completed scope:

1. Unified facets, family, inventory-summary logic, and coverage reasoning into a shared document-semantics layer.
2. Strengthened inventory-first routing with coverage-aware, rarity-aware, and distinctive-term shortlist behavior.
3. Tightened document-level and cross-document answers around reusable answer contracts and clearer relationship reasoning.
4. Added architecture-facing evaluation slices and regressions for inventory coverage and relationship behavior.
5. Added user-facing CLI paths for listing documents, inspecting document metadata, planning queries, and exporting JSON answers.

Outcome:

- the benchmark remains stable at 67 cases across 19 indexed sample documents
- the project now has a more coherent document-intelligence layer instead of scattered metadata heuristics
- the local CLI is closer to a publishable tool surface rather than just an internal benchmark harness

## 6.27. v1.27 Milestones

This iteration focused on turning the repo into a more publishable local tool surface instead of extending benchmark logic.

Completed scope:

1. Added a package-first entry path through `pyproject.toml`, `python -m pdf_to_json_rag`, and a console-script target.
2. Unified CLI JSON output contracts across document listing, inspection, planning, retrieval, answering, and evaluation commands.
3. Added an explicit `run-workflow` path for `extract -> chunk -> index -> inspect -> plan -> answer`.
4. Added public-safe `examples/` assets so the user-facing workflow is separated from ignored local benchmark PDFs.
5. Re-validated the core planning and answer-mode regressions after the publication pass and kept heavier deferred features out of scope.

Outcome:

- the project is materially closer to a first public tool release rather than just an internal benchmark harness
- packaging, JSON contracts, and the end-to-end workflow are now explicit parts of the repo surface

## 6.28. v1.28 Milestones

This iteration focused on first-user release polish instead of deeper retrieval changes.

Completed scope:

1. Added a concise quickstart path that uses the packaged CLI entrypoint consistently in docs and examples.
2. Added a small automated `smoke-check` path for the packaged workflow, not just benchmark regressions.
3. Tightened CLI error contracts so common failures now return clearer user-facing diagnostics in both human-readable and JSON modes.
4. Added public-safe example JSON output shapes for document inspection, planning, and answering.
5. Re-checked deferred features and kept them out of scope because the release-polish pass exposed no new architectural gap.

Outcome:

- the project now has a packageable CLI, a first-run smoke path, and stable output/error contracts for the user-facing commands
- the repo is closer to a first public tool release and less dependent on internal benchmark knowledge

## 6.29-6.33. Public Release Hardening

These five iterations focused on making the repo usable as a first public local tool rather than extending the benchmark.

Completed scope:

1. Added an isolated `PDF_TO_JSON_RAG_DATA_DIR` path so the CLI and smoke tests can run outside the repo-local benchmark workspace.
2. Added automated public-surface smoke tests that generate a tiny PDF and validate the packaged CLI flow end to end.
3. Tightened command ergonomics with user-facing help, aliases, `doctor`, `demo-profile`, and optional `--output` JSON export.
4. Split public CLI onboarding docs from internal evaluation/debug docs.
5. Re-ran the public CLI checks, core regression shards, and the full benchmark after the release-facing pass.

Outcome:

- the repo is now materially closer to a first public `v0.1` than to another internal benchmark sprint
- the release surface is cleaner, more testable, and less dependent on the existing local workspace state

## 6.34-6.38. Processing and Semantic-Retrieval Hardening

These five iterations shifted the focus back from CLI release polish to the two largest remaining architectural gaps: robustness in document processing and a retrieval/answering core that depends less on brittle surface cues.

Completed scope:

1. Added extraction-time block metadata so native and OCR paths both persist block kind, line count, token count, and structural flags.
2. Added chunk-level semantic metadata such as `semantic_terms`, `content_hints`, `structural_flags`, and `source_block_kinds`, and persisted them through chunk JSON and index metadata.
3. Extended retrieval scoring with semantic overlap and structural-reference alignment for `Appendix`, `Table`, and `Question` queries instead of relying only on lexical matches and source-family heuristics.
4. Made answer assembly more coverage-aware so multi-point answers pull complementary evidence rather than repeatedly selecting the same high-score fragment.
5. Re-ran regression shards and the full benchmark after the semantic/structural pass and restored a warning-free state without reviving `pdfplumber`, cross-encoder reranking, or LLM-as-a-judge.

Outcome:

- document processing now carries richer structural semantics from extraction into chunking and retrieval
- retrieval and answer assembly are less purely heuristic and less dependent on accidental wording matches
- the benchmark remains warning-free at 67 cases across 19 indexed sample documents, with `MRR = 1.0`, `avg_keyword_coverage = 1.0`, and stable slices

## 6.39. v1.39 Milestones

This iteration focused on the last release-path gaps rather than on core benchmark or retrieval expansion.

Completed scope:

1. Added a `create-demo-pdf` command so the quickstart can run without requiring any user-supplied PDF.
2. Added a single `release-check` command that combines public-surface smoke checks with the highest-value maintainer regression shards.
3. Tightened `doctor` so it separates required public-tool assets from optional OCR capability and internal benchmark assets.
4. Added a minimal release checklist path for the first public pre-release or release candidate.
5. Re-checked the release surface from the perspective of a first public user and kept heavier deferred features out of scope.

Outcome:

- the public onboarding flow is now self-contained
- the maintainer has one release-gate command instead of several manual checks
- the repo is materially closer to a first public `v0.1` / pre-release than to another benchmark sprint

## 6.40. v1.40 Milestones

This iteration focused on the last small polish gaps before the first public pre-release.

Completed scope:

1. Added a public-safe `v0.1` pre-release notes template instead of relying on a repo-tracked maintainer-only checklist.
2. Tightened CLI wording so document-level answers and `doctor` / `release-check` summaries read more like a product surface than an internal tool.
3. Added one more public-surface smoke test around `create-demo-pdf -> smoke-check -> answer-query` as a single scripted path.
4. Added a small `--format text|json` polish pass without breaking the stable JSON contracts.
5. Re-assessed the first public label and kept the recommendation at `v0.1.0-beta` rather than `v0.1.0-rc1`.

Outcome:

- the public release surface is now self-contained, easier to explain, and easier to validate from a clean install path
- the repo is now closer to executing a first public beta than to another release-polish architecture sprint

## 6.41. v1.41 Milestones

This iteration focused on the last packaging and publication gaps before the first public beta.

Completed scope:

1. Added an isolated packaging verification path that builds the project artifact and verifies the packaged CLI from a clean temporary install root.
2. Extended `release-check` with a packaging/distribution gate while keeping the command usable as a normal maintainer release gate.
3. Added concise public release artifacts derived from the version log and pre-release template.
4. Tightened the final public docs so quickstart, command reference, and release notes no longer assume benchmark knowledge.
5. Re-assessed the current release state and kept the recommended first public label at `v0.1.0-beta`.

Outcome:

- the repo now verifies both source-path and packaged-install behavior before a public release
- the remaining gap to a first public beta is mostly release execution rather than missing architecture

## 6.42. v0.1.1 Milestones

The first post-beta patch should fix rough edges exposed by the public release without changing the tool architecture.

Completed scope:

1. Fixed the `document_overview` answer path so document-level metadata answers no longer fall through to unnecessary abstention.
2. Fixed `inspect-document` so it reads from the active `PDF_TO_JSON_RAG_DATA_DIR` instead of assuming repo-local benchmark paths.
3. Tightened `doctor` so retrieval readiness recognizes the public `workflow_smoke` index path and suggests a concrete next step.
4. Added more explicit next-step guidance to `init`, `create-demo-pdf`, `extract-native`, `chunk-document`, and `build-index`.
5. Re-validated the public CLI smoke path after the post-beta bug fixes.

Outcome:

- the public beta flow is less confusing after extraction and first-run indexing
- the CLI is less dependent on implicit repo-local state

## 6.43. v0.1.2 Milestones

The second post-beta patch should reduce onboarding friction and make the public docs align with the easiest successful path through the tool.

Completed scope:

1. Rewrote the quickstart so the shortest path goes through `smoke-check` and `run-workflow` before the manual `extract -> chunk -> build-index` flow.
2. Switched public install examples to `python -m pip install .` and added a clear development fallback using `PYTHONPATH=src python -m pdf_to_json_rag`.
3. Aligned the CLI help epilog and command reference with the public install and demo flow.
4. Updated example/docs wording so public onboarding is clearly separate from internal evaluation usage.
5. Kept the user-facing command set stable while polishing the text output around the public paths.

Outcome:

- the public docs now lead with the shortest successful path instead of the most manual one
- install and first-run guidance are more consistent across the README and CLI docs

## 6.44. v0.2.0 Milestones

The first minor release after the beta line should be an architectural upgrade, not another benchmark-growth cycle.

Completed scope:

1. Added extraction-time document sections so blocks are grouped into stable semantic sections before chunking.
2. Reworked chunk building so section boundaries come from extraction-time structure and semantic chunk metadata inherits section context.
3. Split retrieval scoring into clearer runtime signals for quality, semantic alignment, structural alignment, metadata priors, and rank priors.
4. Tightened document-level overview answers so they render a cleaner section-aware summary instead of mostly stitched inventory strings.
5. Hardened the default local-first embedding path so the public workflow falls back deterministically without requiring model downloads.

Outcome:

- the core architecture now carries more structure from extraction through chunking, retrieval, and document-level synthesis
- the public demo path stays local-first and stable even without external model downloads
- the new architecture is visible in the CLI through section-aware inspect output and retrieval signal payloads

## 6.45. v0.2.1-v0.2.4 Milestones

The first sprint after `v0.2.0` should harden the install and release path around the new architecture before attempting a larger learned layer.

Completed scope:

1. Fixed installed-package path resolution so the CLI no longer defaults to writing data under the Python installation when run outside the repo.
2. Added packaged example assets and install-safe example loading so `doctor`, `demo-profile`, `create-demo-pdf`, and related commands work from a real installed wheel.
3. Tightened `package-check` to build from a maintainer checkout but validate the installed CLI from a clean temporary workspace rather than implicitly relying on the repo.
4. Split `release-check` semantics so public-surface readiness, maintainer package/test gates, and benchmark-only regression gates are reported separately.
5. Re-ran public-surface validation and documented the remaining repo-local regression shard failures exposed by the `v0.2.0` section-aware refactor.

Outcome:

- the packaged CLI now behaves more like a real installed tool than a repo-bound script
- public release validation is green for the install path
- the next internal hardening target is benchmark regression recovery, not more packaging work

## 6.46. v0.2.5 Milestones

The next patch recovered the small set of repo-local regressions exposed by the section-aware architecture and clarified what still remains before the wider benchmark is back in parity.

Completed scope:

1. Recovered the failing core maintainer shards in `query_planning_core`, `answer_modes_core`, `document_family_core`, and `relationship_core`.
2. Tightened section-aware scoring and source preference only where the failing cases showed real drift, instead of adding another broad benchmark-expansion pass.
3. Added an anchor-recovery retrieval path for a few high-risk intents where embedding-first hits were still hiding the correct section or structured-form chunk inside the right document.
4. Revalidated the public release path and maintainer gates after the recovery pass.
5. Re-ran the broad benchmark and made explicit that wide-benchmark parity is still not back to the earlier pre-`v0.2.0` level.

Outcome:

- the public install/release path is green
- the maintainer shard set used by `release-check` is green
- the next step should be `v0.2.6` broad-benchmark recovery and evaluation cleanup, not more packaging work

## 6.47. v0.2.6 Milestones

This patch finished the first full-benchmark recovery pass on top of the section-aware architecture instead of stopping at a small green maintainer shard set.

Completed scope:

1. Recovered the highest-impact broad-benchmark regressions that still affected common evidence lookup, structured-form, and source-listing slices outside the maintainer shard set.
2. Tightened `release-check` so internal benchmark regressions only run when the active data root actually contains full benchmark assets instead of any partial demo/index state.
3. Fixed benchmark-evaluation semantics so document-level cases can use `relevant_doc_ids` without breaking the full rerun.
4. Re-ran the full benchmark and restored a stable post-`v0.2.x` baseline:
   - `precision@5 = 0.533`
   - `recall@5 = 1.0`
   - `MRR = 1.0`
   - `avg_keyword_coverage = 1.0`
   - `negative_success_rate = 1.0`
   - `warning_case_count = 0`
5. Revalidated the release-facing path after the recovery pass and confirmed that the next target is no longer broad retrieval parity, but cleaner document-level support/faithfulness semantics.

Outcome:

- public release gates are green
- maintainer package/test gates are green
- the maintainer shard set is green
- the full 67-case benchmark is green again on the section-aware architecture
- the next architectural gap is document-level support tracing, not broad benchmark churn

## 6.48. v0.2.7 Milestones

This patch hardened document-level support semantics on top of the recovered `v0.2.x` section-aware architecture instead of reopening broad benchmark churn.

Completed scope:

1. Added explicit `support_trace` payloads for document-level answer modes so overview, routing, source listing, source justification, and cross-document comparison expose structured support instead of mostly empty evidence payloads.
2. Extended document-level answer assembly so support can be derived from inventory summaries, document facets, section cues, matched terms, and comparison-level relationship summaries.
3. Adjusted the sampled faithfulness audit so document-level answer modes are judged against their support contract rather than only chunk-evidence sentences.
4. Tightened CLI text rendering so document-level answers show `Support:` rather than a misleading empty `Evidence:` block.
5. Re-ran `release-check`, the maintainer shard set, and the full 67-case benchmark after the support-trace pass.

Outcome:

- public release gates are green
- maintainer package/test gates are green
- the maintainer shard set is green
- the full 67-case benchmark is green
- the sampled faithfulness audit is green again on document-level modes

## 6.49. v0.3.0 Milestones

This iteration chose simplification over a heavier learned layer and reworked the document-level stack around clearer intermediate contracts.

Completed scope:

1. Replaced brittle literal query routing with a feature-based planner that returns:
   - per-mode scores
   - explicit chosen rationale
   - shortlist-aware document metadata
2. Simplified inventory shortlist scoring into four inspectable buckets:
   - title/label overlap
   - semantic/discovery overlap
   - facet/purpose/family fit
   - rarity/distinctive bonus
3. Split document-level retrieval into:
   - candidate-document selection
   - chunk retrieval inside selected documents
4. Unified document-level answer construction around shared support entries and a common intermediate support trace instead of multiple separate hand-built branches.
5. Tightened evaluation and output contracts by:
   - adding `document_pipeline_core`
   - compacting default JSON outputs
   - pushing full retrieval/debug payloads behind `--verbose`

Outcome:

- public release gates remain green
- maintainer shard set remains green
- document-level simplification is now explicit and inspectable
- targeted reruns covering all reopened warning cases are green
- the next decision is no longer whether to simplify; it is whether any learned reranker is justified after this cleaner baseline

## 6.50. v0.3.1 Milestones

This iteration froze the simplified `v0.3.0` baseline and used that cleaner stack to decide whether a learned reranker is justified yet.

Completed scope:

1. Re-ran the full broad benchmark on top of the simplified document-level stack and restored a stable green baseline:
   - `precision@5 = 0.5309`
   - `recall@5 = 1.0`
   - `MRR = 1.0`
   - `avg_keyword_coverage = 1.0`
   - `negative_success_rate = 1.0`
   - `warning_case_count = 0`
2. Audited the remaining source-anchor-sensitive evidence lookup seams and narrowed them only where they were still affecting real benchmark cases:
   - `antibiotics`
   - `wat_antibiotics_review`
   - `vitamin_c_normal_populations`
   - `vitamin_c_cold_stress`
3. Extended the maintainer release gate to include `document_pipeline_core`, so the simplified document-level path is now part of the default release-check regression set.
4. Revalidated the simplified baseline across:
   - `python -m unittest tests.test_cli_public_surface`
   - `release-check`
   - `document_pipeline_core`
   - the full 67-case benchmark
5. Used the fully green simplified baseline to make an explicit architecture decision: a learned reranker is still not justified yet.

Outcome:

- public release gates are green
- maintainer regression gates are green
- the full broad benchmark is green
- the sampled faithfulness audit remains green
- the next step is not to add a heavier model layer by default; it is to preserve this baseline and only revisit learned reranking if future failures justify it

## 6.51. v0.3.2-v0.3.4 Milestones

This sprint kept the `v0.3.1` simplified baseline intact while reducing the remaining maintenance cost around source-sensitive evidence lookups.

Completed scope:

1. Centralized source-anchor resolution so retrieval, answering, and evaluation all reuse the same preferred-source and matched-source helpers instead of carrying separate copies of the same heuristics.
2. Added a compact `evidence_anchor_core` regression shard to hold the highest-risk source-sensitive evidence cases:
   - `antibiotics`
   - `wat_antibiotics_review`
   - `vitamin_c_normal_populations`
   - `vitamin_c_cold_stress`
   - `echinacea_overall_conclusion`
   - `ct_follow_up_improvement`
   - `cmaj_zinc_prevention`
3. Extended the default `release-check` maintainer regression gate to include `evidence_anchor_core` alongside the existing simplified document-level shard set.
4. Fixed release-check metadata so the recommendation points to the current public beta tag, `v0.1.0-beta`, instead of the stale pre-`v0.3.x` tag.
5. Revalidated the simplified baseline across:
   - `python -m unittest tests.test_cli_public_surface`
   - `evaluate-regression --shard evidence_anchor_core`
   - `release-check`

Outcome:

- public release gates are still green
- the maintainer regression gate now covers the critical evidence-anchor seams explicitly
- the architecture decision still holds: a learned reranker remains deferred because the simplified heuristic baseline is still green on release gates and benchmark-critical shards

## 6.52. v0.3.5-v0.3.8 Milestones

This sprint moved one step closer to plan points `1` and `2` by strengthening structure metadata and reducing the remaining document-level support duplication without adding a heavier learned layer.

Completed scope:

1. Reduced the remaining document-level support duplication by reusing one support-entry builder across overview, routing, listing, justification, and comparison answers.
2. Strengthened extraction-to-chunk structure signals:
   - more explicit section heading levels
   - chunk-level `section_content_hints`
   - explicit `chunk_type` for table-heavy and header-like cases
3. Tightened structure-aware chunk boundaries around:
   - questionnaire-like numbered sections
   - table-heavy transitions
   - section-level structure handoffs
4. Extended document-level support traces so they carry section summaries, section hints, and answer-shaped document facts that better match the rendered answer sentences.
5. Added `structure_chunking_core` as a compact regression shard and extended the default `release-check` maintainer gate to include it.
6. Revalidated across:
   - `python -m unittest tests.test_cli_public_surface`
   - `evaluate-regression --shard structure_chunking_core`
   - `release-check`
   - full `evaluate-mvp --top-k 5 --json`

Outcome:

- public release gates remain green
- the broad benchmark remains green
- the sampled faithfulness audit remains green
- plan point `1` improved through stronger structure reconstruction and chunk metadata
- plan point `2` improved through cleaner document-level support assembly
- plan point `3` improved through a tighter structure-sensitive regression gate

## 6.53. v0.3.9 Milestones

This iteration kept the structure-aware `v0.3.x` baseline green while simplifying candidate-document selection and the retrieval-to-answer handoff.

Completed scope:

1. Added an explicit `document_selection` contract so the document-level answer trace now exposes:
   - `candidate_doc_ids`
   - `ranked_doc_ids`
   - `selected_doc_ids`
   - `primary_doc_id`
   - `strategy`
2. Simplified the retrieval-to-answer handoff so document-level answers consume one normalized selection payload instead of recomputing shortlist decisions inside each answer mode.
3. Kept overview, routing, source listing, source justification, and cross-document comparison on the same shared document-selection path.
4. Revalidated the simplified handoff through:
   - `python -m unittest tests.test_cli_public_surface`
   - `evaluate-regression --shard document_pipeline_core --top-k 5 --json`
   - `release-check --json`
5. Preserved the simplified baseline and kept the learned-reranker decision deferred, while noting that broad benchmark reruns from a source checkout should use `PYTHONPATH=src` or a fresh reinstall.

Outcome:

- plan point `1` stays green on the stronger structure-aware baseline
- plan point `2` now has a cleaner, more inspectable candidate-document contract
- plan point `3` remains green through the public release gates and maintainer shards, with direct source-checkout verification on the document-level cases reopened by the handoff refactor
- the next step should keep simplifying document-level maintenance cost before revisiting any heavier learned layer


## 7. Explicitly Deferred

The following are useful, but should not block MVP or any early version:

- `pdfplumber` for difficult table extraction
- cross-encoder reranking
- automated LLM-as-a-judge evaluation
- cloud deployment
- multi-document schema extraction
- visual grounding UI
