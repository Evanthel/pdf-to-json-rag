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

Maintainer validation commands:

- `package-check`
- `release-check`
- `readme-smoke-check`
- `layout-sanity-check`
- `corpus-sanity-check`
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
- `compare-modes` -> `compare-runtime-modes`
- `readme-smoke` -> `readme-smoke-check`

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
pdf-to-json-rag inspect-document --doc-id your-doc-id --json --output inspect.json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag package-check --json
pdf-to-json-rag readme-smoke-check --json
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

`inspect-document --json` now also exposes processing-layer details such as `extraction_summary.block_role_counts`, `extraction_summary.text_source_counts`, `extraction_summary.layout_signal_counts`, and per-section `section_role` / `source_block_roles`.

`answer-query --json`, `run-workflow --json`, and `smoke-check --json` now include compact `retrieval_contract`, `document_synthesis`, and `claim_alignment` blocks inside `answer_trace` so both the retrieval path and the answer-time support scope are explicit.

```bash
pdf-to-json-rag corpus-sanity-check --profile quick --json
```

`corpus-sanity-check` samples the repo-local `pdf/` corpus through `pdf/lcwa_gov_pdf_metadata.csv`, runs isolated workflow checks on the sampled PDFs, and returns aggregate rates, deterministic `sample_manifest` bucket/checksum data, bucket-level diagnostics, follow-up actions with concrete failure examples, a saved snapshot path, a corpus contract gate, and a corpus architecture gate over `processing`, `semantics`, and `trust`. Use `--profile quick|balanced|stress` or the equivalent `--sample-profile` to control cost; `--sample-size` still overrides the profile when needed.

`evaluate-mvp --json` now also returns `layer_summary`, `layer_stability`, `architecture_gates`, and sampled faithfulness `contract_validation` blocks so you can separate `processing`, `retrieval`, and `answer_faithfulness` health from the broader benchmark summary and still get an explicit gate decision.

`compare-runtime-modes --json` writes `data/eval/runtime_mode_comparison.json` and compares the same cases across `baseline`, `sentence-transformers`, `cross-encoder`, and `llm-synthesis`. Add `--all-cases` to run the full evaluation suite. Optional models remain offline-safe: if a model or `PDF_TO_JSON_RAG_LLM_COMMAND` is not locally available, the report shows the effective fallback/runtime state instead of treating it as a hidden success.

`runtime-check --json` reports install context, the requested embedding backend, effective backend/model, local sentence-transformer availability, fallback reason, runtime decision, cross-encoder opt-in state, and LLM synthesis opt-in state.

`runtime-promotion-report --json` summarizes the latest saved `runtime_mode_comparison.json` without rerunning the benchmark. When the full-suite sentence-transformer gate is green, it also writes `data/eval/runtime_promotion_snapshot.json`. The comparison includes `promotion_gates.sentence-transformers`, which blocks promotion unless the sentence-transformer backend is active and does not regress pass count, recall, MRR, or warning count relative to baseline.

`build-index --json`, `run-workflow --json`, and `smoke-check --json` include an `index.embedding` block with requested backend, effective backend/model, fallback reason, and runtime-check diagnostics.

`release-check --json` returns a compact release summary by default, including public, maintainer, internal-regression, runtime, and local-corpus gates as pass/fail/skip records. Use `release-check --json --verbose` for the full payload.

`readme-smoke-check --json` builds and installs the package into a temporary environment, then replays the public README flow: `init`, `doctor`, `create-demo-pdf`, `smoke-check`, and `runtime-check`. It intentionally excludes maintainer benchmark regressions.
