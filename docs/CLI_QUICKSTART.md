# CLI Quickstart

This is the shortest public path through the tool.

## Install

```bash
python -m pip install .
```

For local development without installing the console script:

```bash
PYTHONPATH=src python -m pdf_to_json_rag doctor --json
```

Optional stronger local embeddings:

```bash
export PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1
```

## Initialize local data directories

```bash
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
```

Use a fresh `PDF_TO_JSON_RAG_DATA_DIR` for quickstart and release-check runs so old local artifacts do not affect `doctor` or `release-check`.

## Check tool readiness

```bash
pdf-to-json-rag doctor --json
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

With `--verbose`, the same document-level commands also expose richer structure fields such as section paths, section kinds, and shortlist breakdowns.

Structured-form and checklist-style answers also use the same compact default output; use `--verbose` when you want the richer support/debug fields from the current structure-aware baseline.

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
```

Use these from a source checkout when you want to validate wheel packaging, public smoke behavior, and the current release gates. End users only need `doctor`, `smoke-check`, and `run-workflow`.

If you changed code under `src/` and have not reinstalled the package yet, run maintainer checks from the source checkout like this:

```bash
PYTHONPATH=src python -m pdf_to_json_rag release-check --json
PYTHONPATH=src python -m pdf_to_json_rag evaluate-mvp --top-k 5 --json
```

## Public-safe examples

See:

- `examples/public_demo_profile.json`
- `examples/public_workflow.json`
- `examples/public_demo_queries.json`
- `examples/*.example.json`
