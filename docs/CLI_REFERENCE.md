# CLI Reference

User-facing commands:

- `init`
- `doctor`
- `package-check`
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
- `release-check`

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
pdf-to-json-rag inspect-document --doc-id your-doc-id --json --output inspect.json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag package-check --json
pdf-to-json-rag plan-query --query "Which file is most relevant for drought triggers?" --json
pdf-to-json-rag answer-query --query "What does this file cover?" --json
pdf-to-json-rag release-check --json
pdf-to-json-rag answer-query --query "What does this file cover?" --format json
```
