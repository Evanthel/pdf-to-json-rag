# CLI Quickstart

This is the shortest path from a source checkout to a processed PDF and a grounded answer.

## Install

```bash
python -m pip install .
```

Optional table extraction support:

```bash
python -m pip install '.[tables]'
```

## Create an isolated workspace

```bash
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
```

A fresh data directory prevents older documents or indexes from affecting the example.

## Run the bundled demo

```bash
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag run-workflow \
  --pdf /tmp/pdf-to-json-rag-demo.pdf \
  --query "What kind of document is this?" \
  --json
```

The workflow extracts the PDF, writes structured document and chunk JSON, builds a local ChromaDB index, plans the query, and returns a grounded answer. Add `--verbose` when you need artifact paths, retrieved chunks, and full diagnostics.

For a compact readiness decision on an unfamiliar PDF:

```bash
pdf-to-json-rag assess-pdf --pdf /path/to/document.pdf --json
pdf-to-json-rag inspect-pdf-quality --pdf /path/to/document.pdf --json
```

## Inspect the pipeline step by step

```bash
pdf-to-json-rag extract-native --pdf /path/to/document.pdf --json
pdf-to-json-rag chunk-document --doc-id your-doc-id --json
pdf-to-json-rag build-index --doc-id your-doc-id --json
pdf-to-json-rag inspect-document --doc-id your-doc-id --json
pdf-to-json-rag answer-query --query "What does this document say about X?" --json
```

## Use the web workspace

```bash
pdf-to-json-rag-web --open
```

The server listens on `127.0.0.1:8765` by default and uses the same `PDF_TO_JSON_RAG_DATA_DIR` as the CLI. It does not require Node.js. See the [web interface guide](./WEB_INTERFACE.md) for storage behavior and HTTP routes.

## Optional local semantic retrieval

The default `auto` backend uses a cached sentence-transformer model when available and otherwise falls back to deterministic hash embeddings without downloading model weights.

```bash
export PDF_TO_JSON_RAG_EMBEDDING_BACKEND=sentence-transformers
export PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL=/path/to/local/all-MiniLM-L6-v2
pdf-to-json-rag runtime-check --json
```

Cross-encoder reranking, LLM synthesis, benchmark commands, environment variables, and maintainer checks are documented in the [CLI reference](./CLI_REFERENCE.md). Architecture, evaluation methodology, and limitations are covered in [project details](./PROJECT_DETAILS.md).
