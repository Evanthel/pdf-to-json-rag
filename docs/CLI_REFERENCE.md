# CLI Reference

Install from the repo root:

```bash
python -m pip install .
```

Optional table support:

```bash
python -m pip install '.[tables]'
```

For local development without installing the console script:

```bash
PYTHONPATH=src python -m pdf_to_json_rag help
```

Optional stronger local embeddings:

```bash
export PDF_TO_JSON_RAG_EMBEDDING_BACKEND=sentence-transformers
export PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL=/path/to/local/all-MiniLM-L6-v2
pdf-to-json-rag runtime-check --json
```

The default remains deterministic `hash`. `PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1` remains a legacy alias; `PDF_TO_JSON_RAG_EMBEDDING_BACKEND=auto` uses sentence-transformers only when the local model is ready.

`runtime-check --json` also returns `runtime_decision`, including the default backend, recommended opt-in backend when a promotion snapshot is available, and the reason the opt-in backend is not the default.

Optional cross-encoder reranking:

```bash
export PDF_TO_JSON_RAG_USE_CROSS_ENCODER=1
export PDF_TO_JSON_RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder path is opt-in; missing local model/dependency availability falls back to lightweight reranking.

Retrieval payloads include `rerank_backend` plus numeric rank signals in `retrieval_signals`:

- `initial_retrieval_rank`
- `expanded_context_rank`

Evidence planning also exposes treatment sub-intents such as `treatment_null_effect` and `treatment_subgroup_benefit` when the query provides enough cues.

Answer traces include `synthesis_prompt_contract`, which documents the LLM-ready context-only prompt boundary without invoking an LLM by default. Set `PDF_TO_JSON_RAG_LLM_COMMAND` to run an opt-in local synthesis command over stdin/stdout; `synthesis_runtime` reports whether it was configured, invoked, used, and which provider boundary handled it.

Evaluation reports include an `llm_judge_prompt_contract` inside sampled `faithfulness_audit` records. Set `PDF_TO_JSON_RAG_JUDGE_COMMAND` to run an opt-in strict-JSON judge command; judge execution stays disabled by default. Judge output is parsed by the built-in strict JSON/fence parser.

Optional semantic multipass for low-confidence document typing:

```bash
export PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS=1
```

User-facing commands:

- `init`
- `doctor`
- `runtime-check`
- `runtime-promotion-report`
- `demo-profile`
- `create-demo-pdf`
- `extract-native`
- `chunk-document`
- `build-index`
- `inspect-document`
- `list-documents`
- `plan-query`
- `answer-query`
- `run-workflow`
- `smoke-check`
- `assess-pdf`

Maintainer validation commands:

- `package-check`
- `release-check`
- `readme-smoke-check`
- `public-beta-check`
- `layout-sanity-check`
- `corpus-sanity-check`
- `corpus-profile-compare`
- `compare-runtime-modes`

Benchmark/debug commands:

- `retrieve`
- `retrieve-expanded`
- `evaluate-mvp`
- `evaluate-regression`

Helpful aliases:

- `extract` -> `extract-native`
- `chunk` -> `chunk-document`
- `index` -> `build-index`
- `inspect` -> `inspect-document`
- `list` -> `list-documents`
- `plan` -> `plan-query`
- `answer` -> `answer-query`
- `workflow` -> `run-workflow`
- `demo` -> `demo-profile`
- `create-demo` -> `create-demo-pdf`
- `self-check` -> `doctor`
- `layout-check` -> `layout-sanity-check`
- `corpus-check` -> `corpus-sanity-check`
- `corpus-compare` -> `corpus-profile-compare`
- `compare-modes` -> `compare-runtime-modes`
- `readme-smoke` -> `readme-smoke-check`
- `beta-check` -> `public-beta-check`

Focused help:

```bash
pdf-to-json-rag help
pdf-to-json-rag help --topic answer-query
```

JSON output:

- Most public commands support `--json`
- `--format json` is equivalent to `--json`
- JSON can also be written to a file with `--output /path/to/file.json`

Examples:

```bash
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
pdf-to-json-rag run-workflow --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
pdf-to-json-rag assess-pdf --pdf /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag inspect-document --doc-id your-doc-id --json --output inspect.json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag package-check --json
pdf-to-json-rag readme-smoke-check --json
pdf-to-json-rag public-beta-check --json
pdf-to-json-rag corpus-profile-compare --baseline-profile quick --candidate-profile balanced --json
pdf-to-json-rag plan-query --query "Which file is most relevant for drought triggers?" --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
pdf-to-json-rag release-check --json
pdf-to-json-rag release-check --json --verbose
pdf-to-json-rag runtime-check --json
pdf-to-json-rag layout-sanity-check --pdfs /path/a.pdf,/path/b.pdf --json
pdf-to-json-rag compare-runtime-modes --shard evidence_anchor_core --json
pdf-to-json-rag compare-runtime-modes --modes baseline,sentence-transformers --all-cases --json
pdf-to-json-rag runtime-promotion-report --json
pdf-to-json-rag answer-query --query "What does this file cover?" --format json
```

`layout-sanity-check` returns compact overview, type, purpose, audience, and confidence answers for each unfamiliar PDF in addition to structure/layout confidence, semantic confidence, and smoke-style checks.

`inspect-document --json` now also exposes compact `processing_diagnostics` plus processing-layer details such as `extraction_summary.block_role_counts`, `extraction_summary.text_source_counts`, `extraction_summary.layout_signal_counts`, and per-section `section_role` / `source_block_roles`. Processing taxonomy codes include `native_text_low`, `ocr_required`, `weak_sections`, `table_or_form_heavy`, `layout_uncertain`, and `low_text_coverage`.

`assess-pdf --json` is the compact public acceptance layer for unfamiliar PDFs. It returns `overall_status`, `processing_status`, `semantic_status`, `retrieval_status`, `answer_trust`, `recommended_next_action`, `acceptance_profile`, and short diagnostic `messages`. Use `--verbose` to include the full workflow payload behind the assessment.

`answer-query --json`, `run-workflow --json`, and `smoke-check --json` include compact `retrieval_contract`, `document_synthesis`, `claim_alignment`, `contract_health`, `retrieval_contract_status`, `support_coverage`, and `answer_source_mix` blocks so both the retrieval path and answer support sources are explicit. `run-workflow --json` and `smoke-check --json` are compact by default and return `processing_diagnostics` plus `quality_profile_summary`; add `--verbose` for the full workflow `quality_profile`, artifacts, and debug payload. Weak or unsupported claims move answer trust to `review`; document-level metadata claims can be supported by `support_trace`.

Public compact contract freeze:

- `run-workflow --json` result keys: `pdf`, `doc_id`, `document`, `plan`, `index`, `answer`, `processing_diagnostics`, `quality_profile_summary`.
- `smoke-check --json` result keys: all `run-workflow` compact keys plus `checks` and `all_pass`.
- `assess-pdf --json` result keys: `pdf`, `doc_id`, `overall_status`, `processing_status`, `semantic_status`, `retrieval_status`, `answer_trust`, `recommended_next_action`, `acceptance_profile`, `messages`.
- `document` compact keys: `doc_id`, `label`, `title`, `document_family`, `document_type`, `document_purpose`, `audience`, `inventory_summary`, `coverage_terms`, `structure_confidence`, `layout_confidence`, `semantic_confidence`, `semantic_confidence_label`, `page_count`, `section_count`.
- `index` compact keys: `doc_ids`, `chunk_count`, `embedding`.
- `answer` compact keys: `query`, `query_intent`, `answer`, `answer_trace`, `contract_health`, `retrieval_contract_status`, `support_coverage`, `answer_source_mix`.
- Debug-only workflow fields such as `artifacts`, full `quality_profile`, `top_k_hits`, `expanded_hits`, and `evidence` require `--verbose`.

```bash
pdf-to-json-rag corpus-sanity-check --profile quick --json
```

`corpus-sanity-check` samples the repo-local `pdf/` corpus through `pdf/lcwa_gov_pdf_metadata.csv`, runs isolated workflow checks on the sampled PDFs, and returns aggregate rates, deterministic `sample_manifest` bucket/checksum data, bucket-level diagnostics, follow-up actions with concrete failure examples, a saved snapshot path, a corpus contract gate, and a corpus architecture gate over `processing`, `semantics`, and `trust`. Use `--profile quick|balanced|stress` or the equivalent `--sample-profile` to control cost; `--sample-size` still overrides the profile when needed.

`corpus-profile-compare --json` compares saved `corpus-sanity-check` profile snapshots such as quick vs balanced without reprocessing PDFs. It supports aliases such as `quick-latest`, reports metric deltas, sample checksum changes, regression metrics, and a compact `corpus_diff_summary` with pass/fail/skip checks. It also returns `corpus_review` with `pass/review/fail`, top metrics, bucket changes, and `model_experiment_scope`. Treat regressions on changed samples as a corpus review signal, not as proof that one PDF should be special-cased.

`evaluate-mvp --json` now also returns `layer_summary`, `layer_stability`, `architecture_gates`, and sampled faithfulness `contract_validation` blocks so you can separate `processing`, `retrieval`, and `answer_faithfulness` health from the broader benchmark summary and still get an explicit gate decision.

`compare-runtime-modes --json` writes `data/eval/runtime_mode_comparison.json` and compares the same cases across `baseline`, `sentence-transformers`, `cross-encoder`, and `llm-synthesis`. Add `--all-cases` to run the full evaluation suite. Optional models remain offline-safe: if a model or `PDF_TO_JSON_RAG_LLM_COMMAND` is not locally available, the report shows the effective fallback/runtime state instead of treating it as a hidden success. The compact JSON also includes `model_decision_gate`, which can mark a model as recommended or experimental opt-in but keeps `default_change_allowed=false`.

`runtime-check --json` reports install context, the requested embedding backend, effective backend/model, local sentence-transformer availability, fallback reason, runtime decision, cross-encoder opt-in state, LLM synthesis opt-in state, and a unified `backend_policy` that keeps `hash` as default.

`runtime-promotion-report --json` summarizes the latest saved `runtime_mode_comparison.json` without rerunning the benchmark. When the full-suite sentence-transformer gate is green, it also writes `data/eval/runtime_promotion_snapshot.json`. The output includes `default_decision` and `model_decision_gate`, which keep sentence-transformers recommended opt-in only and keep cross-encoder/LLM paths out of the default.

`build-index --json`, `run-workflow --json`, and `smoke-check --json` include an `index.embedding` block with requested backend, effective backend/model, fallback reason, and runtime-check diagnostics.

`release-check --json` returns a compact release summary by default, including public, maintainer, internal-regression, runtime, and local-corpus gates as pass/fail/skip records. It also includes `product_gate`, which summarizes `public_path`, `benchmark`, and `corpus` as `pass`, `fail`, `skip`, or `review`; corpus review includes compact failure examples when available. Use `release-check --json --verbose` for the full payload.

`readme-smoke-check --json` builds and installs the package into a temporary environment, then replays the public README flow: `init`, `doctor`, `create-demo-pdf`, `smoke-check`, and `runtime-check`. It intentionally excludes maintainer benchmark regressions.

`public-beta-check --json` aggregates the installed README flow, public-smoke quality summary, runtime default decision, local corpus quick gate, and compact release summary into one pre-tag gate. It reports `hash` as the default backend and keeps sentence-transformers, cross-encoder, and LLM synthesis as opt-in scopes.
