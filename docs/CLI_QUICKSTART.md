# CLI Quickstart

This is the shortest public path through the tool.

## Install

```bash
pip install -e .
```

## Initialize local data directories

```bash
pdf-to-json-rag init --json
```

## Check tool readiness

```bash
pdf-to-json-rag doctor --json
```

## Create a public-safe demo PDF

```bash
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
```

## Run one document through the tool

```bash
pdf-to-json-rag extract-native --pdf /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag chunk-document --doc-id your-doc-id --json
pdf-to-json-rag build-index --doc-ids your-doc-id --json
pdf-to-json-rag inspect-document --doc-id your-doc-id --json
pdf-to-json-rag plan-query --query "What does this file cover?" --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
```

## Validate the end-to-end path

```bash
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

## Validate the public release surface

```bash
pdf-to-json-rag package-check --json
pdf-to-json-rag release-check --json
```

This returns a single release gate plus a suggested public pre-release tag when the current surface is ready.

## Public-safe examples

See:

- `examples/public_demo_profile.json`
- `examples/public_workflow.json`
- `examples/public_demo_queries.json`
- `examples/*.example.json`
