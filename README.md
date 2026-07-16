# PDF-to-JSON RAG

Local-first, domain-agnostic PDF-to-JSON RAG tool for turning PDFs into structured JSON, routing queries across documents, and returning grounded CLI answers.

## One Command

```bash
pdf-to-json-rag run-workflow --pdf /path/to/file.pdf --query "What does this file cover?" --json
```

Example grounded answer:

> This file is a procedural safety guide for operations staff. It covers preparation, incident response, reporting steps, and follow-up work. Citations: `pdf-to-json-rag-demo`, p. 1, chunks `0001` and `0002`.

Example retrieval snapshot:

| Rank | Source | Pages | Retrieval path | Evidence |
| --- | --- | --- | --- | --- |
| 1 | `pdf-to-json-rag-demo` | 1 | `document_understanding` | safety checks, incident response, reporting steps |
| 2 | `pdf-to-json-rag-demo` | 1 | `single_document_qa` | preparation, response, follow-up |

The longer implementation status and maintainer notes live in [DEVELOPMENT_LOG.md](./DEVELOPMENT_LOG.md).

![Public CLI readiness check](./pdf_json_gh_repo.png)

## Lineage

This repo is a personal implementation inspired by:

- the upstream course repo [https-deeplearning-ai/sc-landingai](https://github.com/https-deeplearning-ai/sc-landingai)
- the course [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information)
- selected architecture ideas from Google's LangExtract project, used as conceptual inspiration for grounded extraction contracts, strict output parsing, provider boundaries, and multipass review patterns

The working course fork and setup/debug history are preserved separately in:

- [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai)

This codebase is intentionally separate from that fork. The fork captures the baseline course reproduction and AWS-side learning path; this repo moves toward a local-first, JSON-first implementation with tighter control over chunking, retrieval, and evaluation.

## Capabilities

- Extract native-text PDFs into document-level and chunk-level JSON artifacts.
- Fall back to OCR with `pytesseract` when native extraction is weak or missing.
- Build a local vector index with `ChromaDB`.
- Carry extraction-time document sections into chunking, inspection, and document-level synthesis.
- Answer single-document evidence questions with grounded citations.
- Route document-level and cross-document queries such as:
  - `What does this file cover?`
  - `What kind of document is this?`
  - `Which file is most relevant for X?`
  - `Why is this the best source?`
  - `What do these sources have in common or how do they differ?`
- Expose inspection and planning paths through a packaged CLI:
  - `list-documents`
  - `inspect-document`
  - `plan-query`
  - `answer-query`
  - `run-workflow`
  - `smoke-check`
  - `assess-pdf`
- Expose maintainer-facing release gates through:
  - `package-check`
  - `release-check`
  - `layout-sanity-check`
  - `corpus-sanity-check`

## Workflow

1. Extract a PDF into `*.native.json` and `*.document.json`
2. Convert extracted blocks into chunk JSON files
3. Build a persistent local vector index from chunk text and metadata
4. Plan the query as evidence lookup, document discovery, cross-document, or document-facet behavior
5. Shortlist candidate documents from inventory metadata when the query needs document-level routing
6. Retrieve top-k chunks for the shortlisted sources
7. Expand with adjacent chunks when needed
8. Assemble a grounded answer from the expanded context, or a source-level / document-level answer for source discovery, comparison, overview, routing, and facet questions
9. Evaluate the result on the full benchmark or on a smaller regression shard

## Quickstart

```bash
python -m pip install .
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

Use a fresh `PDF_TO_JSON_RAG_DATA_DIR` for quickstart and release-check runs so old local artifacts do not make `doctor` or `release-check` look greener than the current session really is.

The current baseline is still heuristic-first. It now behaves more sanely on unfamiliar financial/admin forms, carries richer extraction-time block structure into chunking and inspection, and can separate type, purpose, audience, and overview answers, but unfamiliar layouts can still degrade section recovery and answer confidence.

Optional stronger local embeddings:

```bash
export PDF_TO_JSON_RAG_EMBEDDING_BACKEND=sentence-transformers
export PDF_TO_JSON_RAG_SENTENCE_TRANSFORMERS_MODEL=/path/to/local/all-MiniLM-L6-v2
pdf-to-json-rag runtime-check --json
```

`PDF_TO_JSON_RAG_USE_SENTENCE_TRANSFORMERS=1` remains supported as a legacy alias. The default is now `auto`: it selects sentence-transformers only when the local model is already available, otherwise it uses the deterministic hash fallback without downloading models.

The latest full-suite comparison promotes local `all-MiniLM-L6-v2` as the preferred backend inside `auto` when it is cached locally. Run `pdf-to-json-rag runtime-promotion-report --json` to inspect the saved promotion snapshot and gate decision.

Optional local LLM hooks are provider-agnostic and disabled by default. Commands receive the prompt on stdin and must write the answer or strict judge JSON to stdout:

```bash
export PDF_TO_JSON_RAG_LLM_COMMAND="/path/to/your/synthesis-wrapper"
export PDF_TO_JSON_RAG_JUDGE_COMMAND="/path/to/your/judge-wrapper"
```

Opt-in low-confidence semantic multipass is disabled by default:

```bash
export PDF_TO_JSON_RAG_SEMANTIC_MULTIPASS=1
```

Fastest full workflow:

```bash
pdf-to-json-rag run-workflow --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

For an unfamiliar PDF where you only need a trust/readiness decision:

```bash
pdf-to-json-rag assess-pdf --pdf /path/to/file.pdf --json
pdf-to-json-rag inspect-pdf-quality --pdf /path/to/file.pdf --json
```

`assess-pdf` and `inspect-pdf-quality` return compact acceptance summaries: `overall_status`, processing/semantic/retrieval status, `answer_trust`, `acceptance_profile`, `structure_support`, and short diagnostic messages. Use `--verbose` only when you need the full workflow payload behind that assessment.

Maintainer release validation:

```bash
pdf-to-json-rag package-check --json
pdf-to-json-rag release-check --json
pdf-to-json-rag release-check --json --verbose
pdf-to-json-rag readme-smoke-check --json
pdf-to-json-rag public-beta-check --json
pdf-to-json-rag corpus-profile-compare --baseline-profile quick --candidate-profile balanced --json
```

`release-check --json` returns a compact pass/fail/skip summary. Add `--verbose` when you need the full maintainer payload with doctor details, package tails, shard results, and corpus diagnostics.

Release candidate sanity before tagging:

```bash
python -m pip install .
export PDF_TO_JSON_RAG_DATA_DIR="$(mktemp -d)"
pdf-to-json-rag init --json
pdf-to-json-rag doctor --json
pdf-to-json-rag create-demo-pdf --path /tmp/pdf-to-json-rag-demo.pdf --json
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
pdf-to-json-rag runtime-check --json
```

Maintainers can run the same installed README flow in one command with `pdf-to-json-rag readme-smoke-check --json`. It validates the public installed path only; use `release-check` for benchmark regressions.

For one aggregated pre-tag gate, use `pdf-to-json-rag public-beta-check --json`. It combines the installed README flow, public-smoke quality summary, runtime default decision, corpus quick gate, and compact release summary while checking the `auto` default backend policy.

Local sanity check for unfamiliar PDFs:

```bash
pdf-to-json-rag layout-sanity-check --pdfs /path/a.pdf,/path/b.pdf --json
```

That local sanity path now returns compact overview, type, purpose, audience, confidence, rationale, and limits answers so you can see whether an unfamiliar PDF is only processable or also semantically understood.

Local corpus sanity check over the repo-local `pdf/` directory:

```bash
pdf-to-json-rag corpus-sanity-check --profile quick --json
```

That local-only corpus path now reports both `technical_all_pass` and `semantic_all_pass`, plus rates for specific document typing, specific purpose inference, low-confidence classifications, trust-limited results, and a compact corpus architecture gate over `processing`, `semantics`, and `trust`.

It also returns a deterministic `sample_manifest` with the bucket round-robin algorithm, selected bucket counts, selected digests, and a checksum for the sampled set. Saved compact profile snapshots can be compared later with `corpus-profile-compare` without reprocessing PDFs.

When you want to inspect the new processing layer on one extracted document, `inspect-document --json` now includes compact `processing_diagnostics`, `extraction_summary.block_role_counts`, `extraction_summary.text_source_counts`, `extraction_summary.layout_signal_counts`, and per-section `section_role` / `source_block_roles`. Processing diagnostics use taxonomy codes such as `native_text_low`, `ocr_required`, `weak_sections`, `table_or_form_heavy`, `layout_uncertain`, and `low_text_coverage`.

Document-level answer traces now include compact `retrieval_contract`, `document_synthesis`, `contract_health`, `retrieval_contract_status`, `support_coverage`, and `answer_source_mix` blocks. Default `run-workflow --json` and `smoke-check --json` return compact public payloads with `quality_profile_summary`; use `--verbose` for the full debug payload. Weak or unsupported claims move answer trust to `review`; document-level metadata claims can be supported by `support_trace`.

When you are validating new local code in a source checkout before reinstalling the package, prefer:

```bash
PYTHONPATH=src python -m pdf_to_json_rag release-check --json
PYTHONPATH=src python -m pdf_to_json_rag evaluate-mvp --top-k 5 --json
```

For end users, the main public validation path is still:

```bash
pdf-to-json-rag smoke-check --pdf /tmp/pdf-to-json-rag-demo.pdf --query "What does this file cover?" --json
```

More detailed usage lives in:

- [docs/CLI_QUICKSTART.md](./docs/CLI_QUICKSTART.md)
- [docs/CLI_REFERENCE.md](./docs/CLI_REFERENCE.md)

## Key Files

- `src/pdf_to_json_rag/`
  Core extraction, chunking, retrieval, answering, and evaluation code.
- `docs/CLI_QUICKSTART.md`
  Shortest packaged CLI path.
- `docs/CLI_REFERENCE.md`
  User-facing command reference.
- `project-plan.md`
  Master plan and current roadmap.
- `DEVELOPMENT_LOG.md`
  Internal implementation history.
- `examples/`
  Public-safe demo assets and example outputs.
- `data/eval/`
  Benchmark cases and generated reports.

## Evaluation Snapshot

The saved evaluation report includes:

- per-case retrieval, answer-trace, and evidence snapshots
- slice reporting for document discovery, structure-heavy inputs, and source-anchored cases
- compact maintainer shards for planning, structure, selection, anchors, semantics, confidence-aware document understanding, and relationship reasoning
- a layer-aware summary for `processing`, `retrieval`, and `answer_faithfulness`
- a trust-policy shard for classification rationale and classification limits answers
- strict JSON parser and prompt/eval contract diagnostics for opt-in LLM judge output
- a runtime-mode comparison report for baseline vs optional sentence-transformer, cross-encoder, and LLM synthesis paths
- a promotion gate that verifies optional sentence-transformer promotion against full-suite pass count, recall, MRR, and warning count
- a runtime decision block that uses `auto` as the public default while keeping deterministic hash as the offline fallback
- a real-PDF ground-truth gate for form-like, financial/table, scan/layout, legal, public-record, and occupational documents
- processing-layer shards for block typing, section-role recovery, chunk provenance, and strategy-aware chunking
- a retrieval-contract shard that keeps single-doc, doc-understanding, and cross-doc paths separated in regression coverage
- a retrieval-synthesis shard that keeps document selection, support scope, and answer-chunk handoff aligned
- extra sanity shards for layout, single-document, table-like, and form-heavy behavior, plus a sampled faithfulness audit
- a local corpus sanity pass that distinguishes technical success, semantic success, bucket-specific follow-up, and saved snapshot/contract state on repo-local unknown PDFs
- compact release summaries that expose public, maintainer, shard, runtime, and corpus gates without requiring the full JSON payload

Current gate status:

- public release gates are green
- the maintainer shard set used by `release-check` is green
- `release-check` distinguishes public checks, maintainer checks, and benchmark-only regressions
- the current focus is preserving the structure-aware lightweight baseline while making runtime promotion decisions from real-PDF evidence
- the current focus is moving more unfamiliar PDFs out of `document/reference_lookup` while keeping confidence signalling honest

## Limitations

- OCR fallback is still heuristic and not fully layout-aware
- `pdfplumber` table extraction is optional and currently supplements table-like blocks; deeper table schema normalization is still not implemented
- Section detection is improved, but still rule-based and fragile on unfamiliar layouts
- Chunking and document-level reasoning are still heuristic-first rather than learned
- Retrieval still defaults to heuristic scoring, structure cues, and lightweight reranking; cross-encoder reranking is opt-in and model availability depends on the local environment
- Local sentence-transformer embeddings are used by the default `auto` policy only when already cached; otherwise the public path uses deterministic fallback embeddings
- Document facets, document families, shortlist decisions, and type/purpose/audience inference are still handcrafted metadata layers
- Grounded answers are still extractive by default; opt-in local-command synthesis can replace the final answer when explicitly configured
- Claim/evidence alignment is diagnostic; it is not a replacement for human review on high-risk answers
- The benchmark is still hand-built and not broad enough to prove true generalization
- The scanned and structure-heavy coverage is still modest
- The faithfulness audit emits an LLM-as-judge prompt contract and can run an opt-in local JSON judge command, but does not invoke one by default
- Multilingual robustness is not validated yet

## Notes on Reference Material

This repo was brainstormed with ideas from:

- [DeepLearning.AI Skill Builder](https://skillbuilder.deeplearning.ai/)
- ChatGPT 5.4
- Google's LangExtract, as architecture inspiration only; this repo does not vendor or copy its implementation

Earlier in development, a small set of course notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied into a temporary `references/` folder and used only as design input for OCR fallback planning, reading-order/layout handling, schema design, and grounding-aware RAG flow.

Those reference notebooks were removed from the final repo structure. The current codebase is a separate local implementation rather than a notebook-derived copy.
