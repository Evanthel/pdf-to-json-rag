# CLI Quickstart

This is the shortest public path through the tool.

## Install

```bash
python -m pip install .
```

Optional table support:

```bash
python -m pip install '.[tables]'
```

For local development without installing the console script:

```bash
PYTHONPATH=src python -m pdf_to_json_rag doctor --json
```

Optional stronger local embeddings:

```bash
export PDF_TO_JSON_RAG_EMBEDDING_BACKEND=sentence-transformers
export PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL=/path/to/local/all-MiniLM-L6-v2
pdf-to-json-rag runtime-check --json
```

The default embedding backend remains deterministic `hash`. `PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1` is still accepted as a legacy alias. Use `PDF_TO_JSON_RAG_EMBEDDING_BACKEND=auto` only when you want the CLI to use a local sentence-transformer model if it is already available, otherwise fall back to hash.

`runtime-check --json` includes `runtime_decision.default_backend`, `runtime_decision.recommended_opt_in_backend`, and `runtime_decision.not_default_reason` so the recommended optional backend is visible without changing the public default.

Optional cross-encoder reranking for local environments with a model available:

```bash
export PDF_TO_JSON_RAG_USE_CROSS_ENCODER=1
export PDF_TO_JSON_RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

If the cross-encoder cannot be loaded, retrieval falls back to the default lightweight reranker.

## Initialize local data directories

```bash
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
```

Use a fresh `PDF_TO_JSON_RAG_DATA_DIR` for quickstart and release-check runs so old local artifacts do not affect `doctor` or `release-check`.

## Check tool readiness

```bash
pdf-to-json-rag doctor --json
pdf-to-json-rag runtime-check --json
```

## Create a public-safe demo PDF

```bash
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
```

## Shortest end-to-end path

```bash
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

## Full workflow in one command

```bash
pdf-to-json-rag run-workflow --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

If you want the richer structure/debug payload, add `--verbose` to `plan-query`, `run-workflow`, `retrieve`, `retrieve-expanded`, or `answer-query`.

The default answer trace now includes a compact `document_selection` block for document-level modes so you can see which documents were considered, ranked, and finally selected without needing the full verbose payload.

The same compact trace now also includes `retrieval_contract`, `document_synthesis`, and `claim_alignment`, which show the retrieval path, support scope, and diagnostic claim/evidence alignment.

Retrieved chunks expose `rerank_backend`, `initial_retrieval_rank`, and `expanded_context_rank` signals so you can distinguish the first candidate ranking from the neighbor-expanded context ranking used for answer synthesis.

Answer traces also include `synthesis_prompt_contract`, an LLM-ready grounding contract that says which chunks form the allowed context and requires context-only, chunk-cited answers. The CLI does not invoke an LLM by default; set `PDF_TO_JSON_RAG_LLM_COMMAND` to run an opt-in local synthesis command that reads the prompt from stdin and writes the answer to stdout.

`evaluate-mvp --json` also emits an `llm_judge_prompt_contract` and `contract_validation` in the sampled faithfulness audit so automated judging can score the final answer against the exact source context. Judge execution is not invoked by default; set `PDF_TO_JSON_RAG_JUDGE_COMMAND` to run an opt-in local command that returns strict JSON parsed by the built-in strict JSON/fence parser.

Optional low-confidence semantic multipass is disabled by default:

```bash
export PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS=1
```

To compare the default baseline against optional model/runtime paths without changing defaults:

```bash
pdf-to-json-rag compare-runtime-modes --json
pdf-to-json-rag runtime-promotion-report --json
```

The comparison reports the effective embedding backend, cross-encoder fallback count, LLM usage count, and sentence-transformer promotion gate, so missing local models or answer regressions are visible instead of hidden. Add `--all-cases` when you want the full evaluation suite rather than the quick comparison subset. `runtime-promotion-report` summarizes the latest saved comparison without rerunning it and writes `data/eval/runtime_promotion_snapshot.json` when the full-suite sentence-transformer gate is green.

With `--verbose`, the same document-level commands also expose richer structure fields such as section paths, section kinds, section roles, source-block traces, and shortlist breakdowns.

Structured-form and checklist-style answers also use the same compact default output; use `--verbose` when you want the richer support/debug fields from the current structure-aware baseline.

`inspect-document` and verbose answer traces now also expose `structure_confidence` / `layout_confidence`, extraction-time block/source summaries, and per-section source-block traces so you can see when the pipeline is less certain about an unfamiliar PDF layout.

The current baseline also carries layout signals, section source profiles, and explicit chunk strategies through the processing layer, but unfamiliar layouts are still heuristic-first rather than fully layout-aware.

## Manual step-by-step path

```bash
pdf-to-json-rag extract-native --pdf /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag chunk-document --doc-id your-doc-id --json
pdf-to-json-rag build-index --doc-id your-doc-id --json
pdf-to-json-rag inspect-document --doc-id your-doc-id --json
pdf-to-json-rag plan-query --query "What does this file cover?" --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
```

## Maintainer release validation

```bash
pdf-to-json-rag package-check --json
pdf-to-json-rag release-check --json
pdf-to-json-rag release-check --json --verbose
pdf-to-json-rag readme-smoke-check --json
pdf-to-json-rag public-beta-check --json
```

Use these from a source checkout when you want to validate wheel packaging, public smoke behavior, and the current release gates. `release-check --json` is compact by default; add `--verbose` for the full maintainer payload. `readme-smoke-check` validates only the installed public README flow and does not run benchmark regressions. `public-beta-check` aggregates the installed README flow, runtime decision, corpus quick gate, and compact release summary. End users only need `doctor`, `smoke-check`, and `run-workflow`.

For unfamiliar local PDFs that you do not want to add to the benchmark, use:

```bash
pdf-to-json-rag layout-sanity-check --pdfs /path/a.pdf,/path/b.pdf --json
```

If you are working from the source checkout and want a broader local-only sanity pass over the repo-local PDF corpus:

```bash
pdf-to-json-rag corpus-sanity-check --profile quick --json
```

That command now returns compact overview, type, purpose, audience, confidence, rationale, and limits answers for each PDF, plus corpus-level rates, a deterministic `sample_manifest`, and a corpus architecture gate over `processing`, `semantics`, and `trust`.

`run-workflow --json` and `smoke-check --json` include `quality_profile` and answer `contract_health` blocks so unfamiliar PDFs can be read as processing quality, semantic confidence, retrieval readiness, and answer trust instead of only a final answer string.

If you changed code under `src/` and have not reinstalled the package yet, run maintainer checks from the source checkout like this:

```bash
PYTHONPATH=src python -m pdf_to_json_rag release-check --json
PYTHONPATH=src python -m pdf_to_json_rag evaluate-mvp --top-k 5 --json
```

`evaluate-mvp --json` now exposes `layer_summary`, `layer_stability`, and `architecture_gates` so you can distinguish processing, retrieval, and answer-faithfulness regressions and still get one compact gate decision without opening the full saved report first.

## Public-safe examples

See:

- `examples/public_demo_profile.json`
- `examples/public_workflow.json`
- `examples/public_demo_queries.json`
- `examples/*.example.json`
