# Internal Evaluation

This repo contains a broader internal benchmark and regression harness in addition to the public tool surface.

Main commands:

- `pdf-to-json-rag evaluate-mvp --json`
- `pdf-to-json-rag evaluate-regression --shard query_planning_core --json`

Artifacts:

- `data/eval/mvp_eval_cases.json`
- `data/eval/faithfulness_audit_cases.json`
- generated local reports in `data/eval/`

Use this layer for:

- regression checking while changing retrieval or answering
- slice-level stability checks
- faithfulness spot checks

Do not treat this document as the primary onboarding path for first-time users. For that, start with:

- `README.md`
- `docs/CLI_QUICKSTART.md`
- `examples/`
