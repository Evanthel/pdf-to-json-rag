# PDF-to-JSON RAG

Local-first, domain-agnostic PDF-to-JSON RAG tool for turning PDFs into structured JSON, routing queries across documents, and returning grounded CLI answers.

![Public CLI readiness check](./pdf_json_gh_repo.png)

## Lineage

This repo is a personal implementation inspired by:

- the upstream course repo [https-deeplearning-ai/sc-landingai](https://github.com/https-deeplearning-ai/sc-landingai)
- the course [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information)
- selected architecture ideas from Google's LangExtract project, used as conceptual inspiration for grounded extraction contracts, strict output parsing, provider boundaries, and multipass review patterns

The working course fork and setup/debug history are preserved separately in:

- [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai)

This codebase is intentionally separate from that fork. The fork captures the baseline course reproduction and AWS-side learning path; this repo moves toward a local-first, JSON-first implementation with tighter control over chunking, retrieval, and evaluation.

## Current Status

Current public version: `0.1.0-beta`

Internal development iterations in this repo use `vN.x` labels. Public releases follow semantic versioning starting at `0.1.0-beta`.

Current package metadata version: `0.1.0`

Current internal milestone: `v4.5.0`

The public release label is `0.1.0-beta`; package metadata remains PEP440-compatible `0.1.0` until the first non-beta public cut.

The current baseline includes:

- extraction-time block roles with per-block text provenance and quality signals
- native/OCR page fusion that can merge or switch sources per page instead of one global fallback
- extraction-time layout signals and per-page processing summaries
- explicit multi-column reading-order normalization from extraction through chunking
- relative font-size, bold-font, and TOC-backed heading signals during extraction
- optional `pdfplumber` table probe metadata and supplemental `table_like` blocks when the `tables` extra is installed
- extraction-time sections with `section_path` and `section_kind`
- section roles, layout signals, text-source profiles, and source-block traces carried from extraction into inspection and chunking
- structure-aware chunking and chunk metadata
- chunk-level block provenance, block-role profiles, layout signals, and explicit chunk strategies
- feature-based query planning and explicit `document_selection` traces
- evidence-intent planning for treatment subquestions such as null-effect and subgroup-benefit queries
- shared document-level mode renderers and shared answer finalization helpers
- preserved document-root section context for inline and synthetic section splits
- shared structured-form answer helpers and a dedicated structured-form maintenance gate
- document and chunk `structure_confidence` / `layout_confidence` metadata
- sanity gates for layout robustness and single-document random-PDF behavior
- stronger table-like and form-heavy chunk splitting on unfamiliar layouts
- dedicated table/form layout sanity gates in the maintainer release path
- richer document typing and purpose inference for unfamiliar financial/admin forms
- distinct document-level answers for type, purpose, audience, and overview queries
- a semantic document-understanding gate for source-specific type/purpose/audience questions
- semantic confidence signals and confidence-aware document classification answers
- explicit classification-rationale and classification-limits answers for trust-aware document semantics
- a local corpus sampler over repo-local `pdf/` artifacts and metadata for unknown-document sanity checks
- stronger unknown-document typing for registration forms, court opinions, government bulletins, and inspection-style records
- stronger public-record semantics for statistical tables, web job listings, environmental site records, and institutional correspondence found in unknown-PDF corpus buckets
- corpus-level semantic pass metrics so unfamiliar PDFs are tracked as semantically understood vs only technically processable
- a dedicated `processing_layer_core` maintainer gate for block typing, section roles, and chunk provenance
- a dedicated `processing_strategy_core` maintainer gate for strategy-aware chunking on structure-heavy inputs
- an explicit `retrieval_contract` split for single-document QA, document understanding, and cross-document discovery
- optional cross-encoder reranking behind `PDF_TO_JSON_RAG_USE_CROSS_ENCODER=1`, with lightweight reranking as the stable fallback
- runtime-mode comparison for baseline, sentence-transformers, cross-encoder, and opt-in LLM synthesis
- full-suite runtime comparison with a green promotion gate for optional sentence-transformer embeddings
- explicit `runtime-check` and `runtime-promotion-report` commands for backend selection and promotion readiness
- explicit runtime decision output with `hash` default, recommended opt-in backend, and not-default rationale
- explicit embedding backend policy via `PDF_TO_JSON_RAG_EMBEDDING_BACKEND=hash|sentence-transformers|auto`
- promotion snapshots saved after green full-suite runtime comparisons
- installed-entrypoint verification for the public README flow through `readme-smoke-check`
- aggregated public beta validation through `public-beta-check`
- reranking of the neighbor-expanded context before answer synthesis, with `initial_retrieval_rank` and `expanded_context_rank` signals
- an explicit grounded-only synthesis prompt contract with opt-in local-command LLM execution
- strict local JSON/fence parsing for opt-in LLM outputs and judge diagnostics
- an LLM-as-judge faithfulness prompt contract with opt-in local-command JSON judging
- answer-claim/evidence alignment status in answer traces and faithfulness audit records
- document-level claim alignment now uses support-trace fragments as evidence for metadata claims
- provider abstraction over the current env-command prompt runtime
- prompt/eval contract validation for sampled faithfulness gates
- optional low-confidence semantic multipass behind an env flag, with no default-path change
- a shared `document_synthesis` handoff so document selection, support scope, and answer chunks stay aligned
- compact answer `contract_health` and workflow `quality_profile` blocks for unknown-PDF processing drilldown, retrieval readiness reasons, and answer trust
- explicit quality-profile thresholds and public-smoke quality summary in `public-beta-check`
- `quality_profile.overall_status` and `recommended_next_action` for random-PDF UX
- compact `processing_diagnostics` with failure taxonomy for extraction/layout/chunking issues
- retrieval/synthesis contract status with support coverage and answer source mix
- `assess-pdf` as a compact public acceptance layer for unfamiliar PDFs
- `unknown_document_semantics_core` as a maintainer shard for document type, purpose, audience, and confidence behavior on unfamiliar-document semantics
- a layer-aware evaluation report that separates processing, retrieval, and answer-faithfulness signals
- layer-stability and architecture-gate summaries on top of the layer-aware evaluation report
- corpus-level `processing / semantics / trust` layers and an unknown-document architecture gate
- bucket-level corpus diagnostics and follow-up actions for unknown-document sanity checks
- corpus sample profiles, deterministic sample manifests, saved corpus snapshots, and a contract gate for bucket diagnostics
- saved corpus snapshot comparison through `corpus-profile-compare`
- corpus review workbench output with `pass/review/fail`, top review metrics, and opt-in model experiment scope
- compact corpus snapshots and saved snapshot comparison without reprocessing PDFs
- compact release `product_gate` summary over public path, benchmark, and corpus pass/review state
- compact default workflow JSON output with richer debug state behind `--verbose`
- frozen public compact JSON contracts for `run-workflow`, `smoke-check`, and `assess-pdf`
- explicit backend policy: `hash` default, sentence-transformers recommended opt-in, cross-encoder experimental, LLM synthesis opt-in only
- model decision gates for runtime comparisons and promotion reports, always with `default_change_allowed=false`
- compact `release-check --json` summaries, with full release payloads behind `--verbose`
- deterministic local embeddings by default, with optional `sentence-transformers` or `auto` backend selection

Validation state:

- public CLI tests rerun in the current milestone: green
- public compact workflow contract tests rerun in the current milestone: green
- unknown-document semantics shard rerun in the current milestone: green, `9/9`
- balanced local corpus sanity rerun in the current milestone: green, `12/12` technical and semantic pass
- quick local corpus sanity rerun in the current milestone: green, `4/4` technical and semantic pass
- quick-latest vs balanced-latest corpus profile compare rerun in the current milestone: review due lower average structure confidence on the larger sample
- corpus review and model-decision focused tests rerun in the current milestone: green
- full-suite baseline vs local `all-MiniLM-L6-v2` runtime comparison: green, `77/77` for both modes, sentence-transformer promotion gate green
- processing-layer maintainer shards rerun in the current milestone: green
- retrieval-contract maintainer shard rerun in the current milestone: green
- retrieval-synthesis maintainer shard rerun in the current milestone: green
- evaluation-layer public/unit validation rerun in the current milestone: green
- layer-gate validation rerun in the current milestone: green
- corpus-gate validation rerun in the current milestone: green
- bucket-diagnostics validation rerun in the current milestone: green
- corpus snapshot/profile/contract validation rerun in the current milestone: green
- latest saved full 77-case benchmark: green
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

Current engineering direction:

- keep strengthening the processing layer as the primary baseline for downstream retrieval behavior
- keep retrieval behavior aligned to explicit answer-path contracts instead of one shared fallback path
- keep improving unknown-document semantics on unfamiliar PDFs without hiding uncertainty
- keep using the repo-local `pdf/` corpus as a local-only semantic stress test, not just a layout stress test
- keep heavier reranking optional until it proves value over the structure-aware lightweight baseline

## Capabilities

- Extract native-text PDFs into document-level and chunk-level JSON artifacts.
- Fall back to OCR with `pytesseract` when native extraction is weak or missing.
- Build a local vector index with `ChromaDB`.
- Carry extraction-time document sections into chunking, inspection, and document-level synthesis.
- Answer single-document evidence questions with grounded citations.
- Route document-level and cross-document queries such as:
  - `What does this file cover?`
  - `What kind of document is this?`
  - `Which file is most relevant for X?`
  - `Why is this the best source?`
  - `What do these sources have in common or how do they differ?`
- Expose inspection and planning paths through a packaged CLI:
  - `list-documents`
  - `inspect-document`
  - `plan-query`
  - `answer-query`
  - `run-workflow`
  - `smoke-check`
  - `assess-pdf`
- Expose maintainer-facing release gates through:
  - `package-check`
  - `release-check`
  - `layout-sanity-check`
  - `corpus-sanity-check`

## Workflow

1. Extract a PDF into `*.native.json` and `*.document.json`
2. Convert extracted blocks into chunk JSON files
3. Build a persistent local vector index from chunk text and metadata
4. Plan the query as evidence lookup, document discovery, cross-document, or document-facet behavior
5. Shortlist candidate documents from inventory metadata when the query needs document-level routing
6. Retrieve top-k chunks for the shortlisted sources
7. Expand with adjacent chunks when needed
8. Assemble a grounded answer from the expanded context, or a source-level / document-level answer for source discovery, comparison, overview, routing, and facet questions
9. Evaluate the result on the full benchmark or on a smaller regression shard

## Quickstart

```bash
python -m pip install .
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

Use a fresh `PDF_TO_JSON_RAG_DATA_DIR` for quickstart and release-check runs so old local artifacts do not make `doctor` or `release-check` look greener than the current session really is.

The current baseline is still heuristic-first. It now behaves more sanely on unfamiliar financial/admin forms, carries richer extraction-time block structure into chunking and inspection, and can separate type, purpose, audience, and overview answers, but unfamiliar layouts can still degrade section recovery and answer confidence.

Optional stronger local embeddings:

```bash
export PDF_TO_JSON_RAG_EMBEDDING_BACKEND=sentence-transformers
export PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL=/path/to/local/all-MiniLM-L6-v2
pdf-to-json-rag runtime-check --json
```

`PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1` remains supported as a legacy alias. The default remains `hash`; `auto` selects sentence-transformers only when the local model is already available.

The latest full-suite comparison promotes local `all-MiniLM-L6-v2` as the recommended opt-in embedding backend for retrieval quality, not as a silent default change. Run `pdf-to-json-rag runtime-promotion-report --json` to inspect the saved promotion snapshot and gate decision.

Optional local LLM hooks are provider-agnostic and disabled by default. Commands receive the prompt on stdin and must write the answer or strict judge JSON to stdout:

```bash
export PDF_TO_JSON_RAG_LLM_COMMAND="/path/to/your/synthesis-wrapper"
export PDF_TO_JSON_RAG_JUDGE_COMMAND="/path/to/your/judge-wrapper"
```

Opt-in low-confidence semantic multipass is disabled by default:

```bash
export PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS=1
```

Fastest full workflow:

```bash
pdf-to-json-rag run-workflow --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

For an unfamiliar PDF where you only need a trust/readiness decision:

```bash
pdf-to-json-rag assess-pdf --pdf /path/to/file.pdf --json
```

`assess-pdf` returns a compact acceptance summary: `overall_status`, `processing_status`, `semantic_status`, `retrieval_status`, `answer_trust`, `recommended_next_action`, an `acceptance_profile`, and short diagnostic messages. Use `--verbose` only when you need the full workflow payload behind that assessment.

Maintainer release validation:

```bash
pdf-to-json-rag package-check --json
pdf-to-json-rag release-check --json
pdf-to-json-rag release-check --json --verbose
pdf-to-json-rag readme-smoke-check --json
pdf-to-json-rag public-beta-check --json
pdf-to-json-rag corpus-profile-compare --baseline-profile quick --candidate-profile balanced --json
```

`release-check --json` returns a compact pass/fail/skip summary. Add `--verbose` when you need the full maintainer payload with doctor details, package tails, shard results, and corpus diagnostics.

Release candidate sanity before tagging:

```bash
python -m pip install .
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
pdf-to-json-rag runtime-check --json
```

Maintainers can run the same installed README flow in one command with `pdf-to-json-rag readme-smoke-check --json`. It validates the public installed path only; use `release-check` for benchmark regressions.

For one aggregated pre-tag gate, use `pdf-to-json-rag public-beta-check --json`. It combines the installed README flow, public-smoke quality summary, runtime default decision, corpus quick gate, and compact release summary while keeping `hash` as the default backend.

Local sanity check for unfamiliar PDFs:

```bash
pdf-to-json-rag layout-sanity-check --pdfs /path/a.pdf,/path/b.pdf --json
```

That local sanity path now returns compact overview, type, purpose, audience, confidence, rationale, and limits answers so you can see whether an unfamiliar PDF is only processable or also semantically understood.

Local corpus sanity check over the repo-local `pdf/` directory:

```bash
pdf-to-json-rag corpus-sanity-check --profile quick --json
```

That local-only corpus path now reports both `technical_all_pass` and `semantic_all_pass`, plus rates for specific document typing, specific purpose inference, low-confidence classifications, trust-limited results, and a compact corpus architecture gate over `processing`, `semantics`, and `trust`.

It also returns a deterministic `sample_manifest` with the bucket round-robin algorithm, selected bucket counts, selected digests, and a checksum for the sampled set. Saved compact profile snapshots can be compared later with `corpus-profile-compare` without reprocessing PDFs.

When you want to inspect the new processing layer on one extracted document, `inspect-document --json` now includes compact `processing_diagnostics`, `extraction_summary.block_role_counts`, `extraction_summary.text_source_counts`, `extraction_summary.layout_signal_counts`, and per-section `section_role` / `source_block_roles`. Processing diagnostics use taxonomy codes such as `native_text_low`, `ocr_required`, `weak_sections`, `table_or_form_heavy`, `layout_uncertain`, and `low_text_coverage`.

Document-level answer traces now include compact `retrieval_contract`, `document_synthesis`, `contract_health`, `retrieval_contract_status`, `support_coverage`, and `answer_source_mix` blocks. Default `run-workflow --json` and `smoke-check --json` return compact public payloads with `quality_profile_summary`; use `--verbose` for the full debug payload. Weak or unsupported claims move answer trust to `review`; document-level metadata claims can be supported by `support_trace`.

When you are validating new local code in a source checkout before reinstalling the package, prefer:

```bash
PYTHONPATH=src python -m pdf_to_json_rag release-check --json
PYTHONPATH=src python -m pdf_to_json_rag evaluate-mvp --top-k 5 --json
```

For end users, the main public validation path is still:

```bash
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

More detailed usage lives in:

- [docs/CLI_QUICKSTART.md](./docs/CLI_QUICKSTART.md)
- [docs/CLI_REFERENCE.md](./docs/CLI_REFERENCE.md)

## Key Files

- `src/pdf_to_json_rag/`
  Core extraction, chunking, retrieval, answering, and evaluation code.
- `docs/CLI_QUICKSTART.md`
  Shortest packaged CLI path.
- `docs/CLI_REFERENCE.md`
  User-facing command reference.
- `project-plan.md`
  Master plan and current roadmap.
- `DEVELOPMENT_LOG.md`
  Internal implementation history.
- `examples/`
  Public-safe demo assets and example outputs.
- `data/eval/`
  Benchmark cases and generated reports.

## Evaluation Snapshot

The saved evaluation report includes:

- per-case retrieval, answer-trace, and evidence snapshots
- slice reporting for document discovery, structure-heavy inputs, and source-anchored cases
- compact maintainer shards for planning, structure, selection, anchors, semantics, confidence-aware document understanding, and relationship reasoning
- a layer-aware summary for `processing`, `retrieval`, and `answer_faithfulness`
- a trust-policy shard for classification rationale and classification limits answers
- strict JSON parser and prompt/eval contract diagnostics for opt-in LLM judge output
- a runtime-mode comparison report for baseline vs optional sentence-transformer, cross-encoder, and LLM synthesis paths
- a promotion gate that verifies optional sentence-transformer promotion against full-suite pass count, recall, MRR, and warning count
- a runtime decision block that keeps `hash` as default while recommending sentence-transformers only as an opt-in backend
- processing-layer shards for block typing, section-role recovery, chunk provenance, and strategy-aware chunking
- a retrieval-contract shard that keeps single-doc, doc-understanding, and cross-doc paths separated in regression coverage
- a retrieval-synthesis shard that keeps document selection, support scope, and answer-chunk handoff aligned
- extra sanity shards for layout, single-document, table-like, and form-heavy behavior, plus a sampled faithfulness audit
- a local corpus sanity pass that distinguishes technical success, semantic success, bucket-specific follow-up, and saved snapshot/contract state on repo-local unknown PDFs
- compact release summaries that expose public, maintainer, shard, runtime, and corpus gates without requiring the full JSON payload

Current gate status:

- public release gates are green
- the maintainer shard set used by `release-check` is green
- `release-check` distinguishes public checks, maintainer checks, and benchmark-only regressions
- the current focus is preserving the structure-aware lightweight baseline while testing learned reranking only as an opt-in local path
- the current focus is moving more unfamiliar PDFs out of `document/reference_lookup` while keeping confidence signalling honest

## Limitations

- OCR fallback is still heuristic and not fully layout-aware
- `pdfplumber` table extraction is optional and currently supplements table-like blocks; deeper table schema normalization is still not implemented
- Section detection is improved, but still rule-based and fragile on unfamiliar layouts
- Chunking and document-level reasoning are still heuristic-first rather than learned
- Retrieval still defaults to heuristic scoring, structure cues, and lightweight reranking; cross-encoder reranking is opt-in and model availability depends on the local environment
- Stronger local sentence-transformer embeddings are opt-in; the default public path uses deterministic fallback embeddings
- Document facets, document families, shortlist decisions, and type/purpose/audience inference are still handcrafted metadata layers
- Grounded answers are still extractive by default; opt-in local-command synthesis can replace the final answer when explicitly configured
- Claim/evidence alignment is diagnostic; it is not a replacement for human review on high-risk answers
- The benchmark is still hand-built and not broad enough to prove true generalization
- The scanned and structure-heavy coverage is still modest
- The faithfulness audit emits an LLM-as-judge prompt contract and can run an opt-in local JSON judge command, but does not invoke one by default
- Multilingual robustness is not validated yet

## Notes on Reference Material

This repo was brainstormed with ideas from:

- [DeepLearning.AI Skill Builder](https://skillbuilder.deeplearning.ai/)
- ChatGPT 5.4
- Google's LangExtract, as architecture inspiration only; this repo does not vendor or copy its implementation

Earlier in development, a small set of course notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied into a temporary `references/` folder and used only as design input for OCR fallback planning, reading-order/layout handling, schema design, and grounding-aware RAG flow.

Those reference notebooks were removed from the final repo structure. The current codebase is a separate local implementation rather than a notebook-derived copy.
