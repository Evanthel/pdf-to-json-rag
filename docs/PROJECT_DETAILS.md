# PDF-to-JSON RAG project details

This document contains the technical depth intentionally kept out of the main README. For runnable setup instructions, use the [CLI quickstart](./CLI_QUICKSTART.md); for every command and output contract, use the [CLI reference](./CLI_REFERENCE.md).

## Capabilities

### Extraction and structure

- Extract native-text PDFs into document-level and chunk-level JSON artifacts.
- Fall back to OCR with `pytesseract` when native extraction is weak or missing.
- Preserve extraction-time sections, page numbers, blocks, coordinates, source roles, and layout signals through chunking and inspection.
- Use `pdf-inspector` for fail-open diagnostics, corroborated OCR routing, and validated missing tables while keeping PyMuPDF canonical.

### Retrieval and answers

- Build a persistent local vector index with ChromaDB.
- Plan evidence lookup, document discovery, document-facet, and cross-document queries separately.
- Shortlist documents from inventory metadata before chunk retrieval when the query requires source routing.
- Expand retrieved context with adjacent chunks when it improves support.
- Answer evidence questions with inspectable page and chunk citations.
- Handle overview, type, purpose, audience, source-selection, and comparison questions.
- Lower answer trust or abstain when the retrieved evidence does not support a claim.

Typical questions include:

- `What does this file cover?`
- `What kind of document is this?`
- `Which file is most relevant for X?`
- `Why is this the best source?`
- `What do these sources have in common or how do they differ?`

### Interfaces and validation

The packaged CLI exposes public inspection and workflow commands such as `list-documents`, `inspect-document`, `plan-query`, `answer-query`, `run-workflow`, `smoke-check`, and `assess-pdf`. Maintainer gates include `package-check`, `release-check`, `layout-sanity-check`, and `corpus-sanity-check`.

The local web workspace uses the same extraction, chunking, indexing, retrieval, and answering functions. Its startup, storage model, user flow, and HTTP routes are documented in [WEB_INTERFACE.md](./WEB_INTERFACE.md).

## End-to-end workflow

1. Extract the PDF into `*.native.json` and `*.document.json`.
2. Route suspicious or missing text through targeted OCR while retaining the canonical extraction path.
3. Convert extracted blocks and sections into chunk JSON files with provenance.
4. Build a persistent local vector index from chunk text and metadata.
5. Plan the query as evidence lookup, document discovery, cross-document comparison, or document-facet behavior.
6. Shortlist candidate documents from inventory metadata when document-level routing is required.
7. Retrieve top-k chunks and expand with adjacent chunks when useful.
8. Assemble a grounded answer from the allowed context, including page and chunk citations, or return a source/document-level answer for discovery and overview questions.
9. Evaluate retrieval, support, faithfulness, and processing behavior against the full suite or a focused regression shard.

## Runtime and extraction behavior

`pdf-inspector` runs as a safe extraction assistant by default. PyMuPDF remains canonical for text, coordinates, reading order, and citations. The assistant can corroborate OCR routing and add only validated missing Markdown tables. Set `PDF_TO_JSON_RAG_PDF_INSPECTOR_MODE=shadow` to record diagnostics without changing extracted content, or `PDF_TO_JSON_RAG_PDF_INSPECTOR_MODE=off` to use the previous extraction path. Loading or processing errors fall back to PyMuPDF and appear in `extraction_summary`.

The baseline remains heuristic-first. It carries block structure and layout signals into chunking and inspection and distinguishes type, purpose, audience, and overview answers, but unfamiliar layouts can still reduce section recovery and answer confidence.

The default embedding backend is `auto`: it uses a cached local sentence-transformer model when available and otherwise falls back to deterministic hash embeddings without downloading a model. To request a specific local model:

```bash
export PDF_TO_JSON_RAG_EMBEDDING_BACKEND=sentence-transformers
export PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL=/path/to/local/all-MiniLM-L6-v2
pdf-to-json-rag runtime-check --json
```

`PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1` remains a legacy alias. `runtime-promotion-report --json` exposes the saved full-suite promotion decision.

Optional local LLM synthesis and judge hooks are provider-agnostic and disabled by default. Each command receives its prompt on stdin and writes an answer or strict judge JSON to stdout:

```bash
export PDF_TO_JSON_RAG_LLM_COMMAND="/path/to/your/synthesis-wrapper"
export PDF_TO_JSON_RAG_JUDGE_COMMAND="/path/to/your/judge-wrapper"
```

Low-confidence semantic multipass is also opt-in:

```bash
export PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS=1
```

For the complete runtime, reranking, output, and environment-variable reference, see [CLI_REFERENCE.md](./CLI_REFERENCE.md).

## Public acceptance paths

The shortest full workflow is:

```bash
pdf-to-json-rag run-workflow --pdf /path/to/file.pdf --query "What does this file cover?" --json
```

For an unfamiliar PDF where only a readiness and trust decision is needed:

```bash
pdf-to-json-rag assess-pdf --pdf /path/to/file.pdf --json
pdf-to-json-rag inspect-pdf-quality --pdf /path/to/file.pdf --json
```

These commands return compact processing, semantic, retrieval, trust, and structure-support summaries. Add `--verbose` when the full workflow payload and diagnostics are needed.

Use a fresh `PDF_TO_JSON_RAG_DATA_DIR` for isolated quickstart and release checks so older local artifacts do not influence the current result. The full installed flow and manual step-by-step commands live in [CLI_QUICKSTART.md](./CLI_QUICKSTART.md).

## Evaluation and release gates

The checked-in broad benchmark currently records 77/77 retrieval cases and 77/77 answer-faithfulness cases, with Recall@5 1.000 and MRR 1.000. These are reproducible regression results on maintained fixtures, not evidence of universal PDF performance. The tracked report is [`data/eval/mvp_eval_report.json`](../data/eval/mvp_eval_report.json).

The evaluation surface includes:

- per-case retrieval, answer-trace, and evidence snapshots;
- slices for document discovery, structure-heavy inputs, and source-anchored cases;
- layer-aware processing, retrieval, and answer-faithfulness summaries;
- trust-policy, retrieval-contract, and retrieval-synthesis regression shards;
- prompt and strict-JSON contract checks for opt-in LLM judging;
- runtime-mode comparison and sentence-transformer promotion gates;
- a hand-built real-PDF gate for forms, tables, scans, irregular layouts, legal documents, and public records;
- local corpus checks that separate technical success from semantic success;
- compact release summaries for public, maintainer, benchmark, runtime, and corpus gates.

Common maintainer commands:

```bash
pdf-to-json-rag package-check --json
pdf-to-json-rag release-check --json
pdf-to-json-rag readme-smoke-check --json
pdf-to-json-rag public-beta-check --json
pdf-to-json-rag corpus-sanity-check --profile quick --json
pdf-to-json-rag corpus-profile-compare --baseline-profile quick --candidate-profile balanced --json
```

`release-check --json` returns a compact pass/fail/skip summary; add `--verbose` for the complete maintainer payload. `readme-smoke-check` validates the installed public flow without benchmark regressions. `public-beta-check` aggregates the installed flow, public-smoke quality, runtime policy, local corpus quick gate, and release summary.

For source-checkout validation before reinstalling:

```bash
PYTHONPATH=src python -m pdf_to_json_rag release-check --json
PYTHONPATH=src python -m pdf_to_json_rag evaluate-mvp --top-k 5 --json
```

The [CLI reference](./CLI_REFERENCE.md) documents focused shards, runtime comparisons, real-PDF checks, corpus sampling, saved snapshots, and compact output contracts.

## Key files

- `src/pdf_to_json_rag/` — extraction, chunking, retrieval, answering, web, and evaluation code.
- `docs/CLI_QUICKSTART.md` — shortest packaged CLI path.
- `docs/CLI_REFERENCE.md` — user-facing command and output reference.
- `docs/WEB_INTERFACE.md` — local web workspace and HTTP surface.
- `project-plan.md` — master plan and current roadmap.
- `DEVELOPMENT_LOG.md` — implementation history and checked milestones.
- `examples/` — public-safe demo inputs and outputs.
- `data/eval/` — benchmark cases and generated reports.

## Limitations

- OCR fallback is heuristic and not fully layout-aware.
- Optional `pdfplumber` table extraction supplements table-like blocks; deeper table schema normalization is not implemented.
- Section detection is rule-based and can be fragile on unfamiliar layouts.
- Chunking and document-level reasoning remain heuristic-first rather than learned.
- Retrieval defaults to heuristic scoring, structure cues, and lightweight reranking; cross-encoder reranking is opt-in and depends on local model availability.
- The `auto` embedding policy uses a local sentence-transformer only when it is already cached; otherwise it uses deterministic fallback embeddings.
- Document facets, families, shortlist decisions, and type/purpose/audience inference are handcrafted metadata layers.
- Grounded answers are extractive by default; an explicitly configured local synthesis command can replace the final answer.
- Claim/evidence alignment is diagnostic and does not replace human review for high-risk answers.
- The benchmark is hand-built and is not broad enough to prove generalization.
- Scanned and structure-heavy coverage is still modest.
- The faithfulness audit can run an opt-in local JSON judge but does not invoke one by default.
- Multilingual robustness has not been validated.

## Reference material

This repository was brainstormed with ideas from [DeepLearning.AI Skill Builder](https://skillbuilder.deeplearning.ai/), ChatGPT 5.4, and Google's LangExtract architecture. It does not vendor or copy LangExtract.

Earlier in development, a small set of notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied into a temporary `references/` directory and used only as design input for OCR fallback planning, reading order and layout handling, schema design, and grounding-aware RAG flow. Those notebooks were removed from the final repository.

The working course fork and its setup/debug history remain in [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai). PDF-to-JSON RAG is a separate local implementation.
