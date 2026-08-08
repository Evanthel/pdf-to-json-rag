# Local web interface

The web workspace is a thin, dependency-free Python server over the canonical extraction, chunking, indexing, and answering functions. The frontend is packaged as static HTML, CSS, and JavaScript; there is no Node.js build step.

## Run

```bash
python -m pip install .
pdf-to-json-rag-web --open
```

Development from a source checkout:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.web --port 8765
```

Open `http://127.0.0.1:8765`. Use `--host` and `--port` or the `PDF_TO_JSON_RAG_WEB_HOST` and `PDF_TO_JSON_RAG_WEB_PORT` environment variables to change the listener. Keep the default loopback host unless you deliberately want to expose the service to another device; this small local server does not implement accounts or authentication.

The default loopback server validates the request host to reduce DNS-rebinding risk. State-changing requests must be same-origin and use `application/pdf` for uploads or `application/json` for questions. Responses also include a restrictive Content Security Policy, frame blocking, MIME sniffing protection, and a same-origin resource policy.

The interface uses the same `PDF_TO_JSON_RAG_DATA_DIR` as the CLI. Every document gets a document-scoped web index under `data/index/web/<doc_id>` so asking about one PDF does not replace the main CLI index.

## User flow

1. Drop or select a PDF up to 100 MB.
2. The backend stores it under `data/input`, then runs the existing extraction, chunking, and indexing pipeline.
3. Ask a question against the active document.
4. Review the answer, page citations, retrieved fragments, and extraction diagnostics.

Existing saved documents appear in the library. If an older document already has chunks but no web-specific index, the service creates that index lazily when the first question is asked.

## HTTP surface

- `GET /api/health` — readiness response.
- `GET /api/documents` — local document summaries.
- `GET /api/documents/<doc_id>` — one document with diagnostics.
- `POST /api/documents` — raw PDF body with an encoded `X-PDF-Filename` header.
- `POST /api/documents/<doc_id>/query` — JSON body such as `{"query": "What does this file cover?", "k": 5}`.

All responses use `{"ok": true, "result": ...}` or a safe `{"ok": false, "error": ...}` envelope. Internal exceptions are not exposed by the HTTP handler.
