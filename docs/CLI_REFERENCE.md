# CLI Reference

Install from the repo root:

```bash
python -m pip install .
```

For local development without installing the console script:

```bash
PYTHONPATH=src python -m pdf_to_json_rag help
```

Optional stronger local embeddings:

```bash
export PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1
```

User-facing commands:

- `init`
- `doctor`
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
- `layout-sanity-check`
- `corpus-sanity-check`

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
pdf-to-json-rag plan-query --query "Which file is most relevant for drought triggers?" --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
pdf-to-json-rag release-check --json
pdf-to-json-rag layout-sanity-check --pdfs /path/a.pdf,/path/b.pdf --json
pdf-to-json-rag answer-query --query "What does this file cover?" --format json
```

`layout-sanity-check` returns compact overview, type, purpose, audience, and confidence answers for each unfamiliar PDF in addition to structure/layout confidence, semantic confidence, and smoke-style checks.

`inspect-document --json` now also exposes processing-layer details such as `extraction_summary.block_role_counts`, `extraction_summary.text_source_counts`, and per-section `section_role` / `source_block_roles`.

`answer-query --json`, `run-workflow --json`, and `smoke-check --json` now include a compact `retrieval_contract` block inside `answer_trace` so the retrieval path is explicit.

```bash
pdf-to-json-rag corpus-sanity-check --sample-size 12 --json
```

`corpus-sanity-check` samples the repo-local `pdf/` corpus through `pdf/lcwa_gov_pdf_metadata.csv`, runs isolated workflow checks on the sampled PDFs, and returns aggregate counts for buckets, semantic confidence labels, classification status, trust-policy outcomes, and corpus-level semantic rates such as `semantic_pass_rate`, `specific_document_rate`, `specific_purpose_rate`, `low_confidence_rate`, and `trust_limited_rate`.
