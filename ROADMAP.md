# Roadmap

PDF-to-JSON RAG is a portfolio-scale, local-first Document AI project. The current `0.2.x` line demonstrates structured PDF extraction, inspectable JSON artifacts, local retrieval, grounded answers, a browser workspace, and reproducible evaluation.

## Near-term priorities

1. Expand the public-PDF ground truth beyond retrieval into document type, purpose, form/table structure, and answer faithfulness.
2. Add a project-owned synthetic scan fixture so OCR routing can be tested in CI without relying on ambiguously licensed documents.
3. Extend processing-level assertions to more difficult scanned and layout-heavy PDFs before adding new retrieval heuristics.
4. Grow strict type checking module by module, prioritizing stable contracts and newly extracted helpers.
5. Raise coverage through tests for high-risk behavior rather than line-only tests, while preserving the existing non-regression gate.

## Model decision policy

- Keep the default offline-safe and deterministic.
- Promote a learned embedding or reranking path only when it improves quality on the same public cases without hidden fallback.
- Record quality, latency, model size, and hardware requirements together.
- Treat benchmark gains as evidence for the measured corpus, not as proof of universal PDF performance.

## Explicit non-goals for the current portfolio scope

- production cloud hosting, accounts, or multi-tenant security;
- mandatory hosted LLM dependencies;
- default cross-encoder reranking without a measured quality gain;
- broad multilingual claims without a dedicated evaluation corpus;
- full visual grounding or arbitrary document-schema extraction.

The [project details](./docs/PROJECT_DETAILS.md) describe the current architecture and limitations. Public release changes are recorded in the [changelog](./CHANGELOG.md).
