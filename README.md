# PDF-to-JSON RAG

Local-first MVP for turning PDFs into structured JSON, indexing the resulting chunks locally, and answering questions with explicit grounding.

## Lineage

This repo is a personal implementation inspired by:

- the upstream course repo [https-deeplearning-ai/sc-landingai](https://github.com/https-deeplearning-ai/sc-landingai)
- the course [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information)

The working course fork and setup/debug history are preserved separately in:

- [Evanthel/sc-landingai](https://github.com/Evanthel/sc-landingai)

This codebase is intentionally separate from that fork and focuses on a local-first JSON pipeline rather than the original AWS/LandingAI lab structure.

## Current Status

This repo is currently at:

- completed end-to-end MVP
- one post-MVP quality iteration over chunking, retrieval, and answer heuristics
- still intentionally local-first and heuristic-heavy

The pipeline already works on a real sample medical PDF and includes a small local evaluation workflow.

## What Works

- Native PDF extraction with `PyMuPDF`
- Document-level JSON output in `data/documents/`
- Chunk generation with reading-order preservation and lightweight section detection
- Local vector indexing with `ChromaDB`
- Retrieval from the local index
- Adjacent-chunk expansion for context reconstruction
- Grounded answer assembly with explicit evidence citations
- Small benchmark-style evaluation workflow with saved JSON reports

## Workflow

1. Extract a native-text PDF into `*.native.json` and `*.document.json`
2. Convert extracted blocks into chunk JSON files
3. Build a persistent local vector index from chunk text and metadata
4. Retrieve top-k chunks for a query
5. Expand with adjacent chunks
6. Assemble a grounded answer from the expanded context
7. Evaluate retrieval and answer quality on a small hand-built benchmark

## How to Run

Minimal local flow:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pdf_to_json_rag.cli extract-native --pdf /path/to/file.pdf
PYTHONPATH=src python -m pdf_to_json_rag.cli chunk-document --doc-id your-doc-id
PYTHONPATH=src python -m pdf_to_json_rag.cli build-index --doc-id your-doc-id
PYTHONPATH=src python -m pdf_to_json_rag.cli answer-query --query "What are common cold symptoms?"
```

To run the small local benchmark:

```bash
PYTHONPATH=src python -m pdf_to_json_rag.cli evaluate-mvp --k 5
```

## Key Files

- `project-plan.md`
  The current scope, MVP definition, and deferred items.
- `WORK_LOG.md`
  High-level implementation log and smoke-test history.
- `src/pdf_to_json_rag/`
  Core extraction, chunking, indexing, retrieval, answering, and evaluation code.
- `data/eval/mvp_eval_cases.json`
  Small local evaluation set.
- `data/eval/mvp_eval_report.json`
  Generated locally by the evaluation workflow and ignored by default.

## Evaluation Snapshot

After the first quality pass, the local benchmark currently reports:

- `precision@5`: `0.35`
- `recall@5`: `1.0`
- `MRR`: `1.0`
- average answer keyword coverage: `1.0`

This means the benchmark’s relevant chunks are currently found within top-5 for all cases, and the first relevant result is ranked first for each benchmark query.

## Limitations

- OCR fallback is not implemented yet; only detection heuristics exist
- Chunking is still heuristic rather than fully semantic
- Section detection is improved, but still rule-based and fragile on other document layouts
- Retrieval quality still depends on lightweight reranking heuristics
- Grounded answers are extractive, not LLM-synthesized
- The evaluation set is very small and currently centered on one sample document
- Table extraction and harder layout handling are still deferred
- Multi-document and multilingual robustness are not validated yet

## Notes on Reference Material

This repo was brainstormed with ideas from:

- [DeepLearning.AI Skill Builder](https://skillbuilder.deeplearning.ai/)
- ChatGPT 5.4

Earlier in development, a small set of course notebooks from [Document AI: From OCR to Agentic Doc Extraction](https://learn.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction/information) was copied into a temporary `references/` folder and used only as design input for:

- OCR fallback planning
- reading-order and layout handling
- schema design
- grounding-aware RAG flow

Those reference notebooks were not kept as part of the final repo structure. The current codebase is a separate local implementation rather than a notebook-derived copy.

## Near-Term Next Steps

- expand the evaluation set beyond the first small benchmark
- improve answer selection for definition and transmission edge cases
- implement real OCR fallback
- decide whether to keep the extractive answer path or replace it with LLM synthesis over grounded context
