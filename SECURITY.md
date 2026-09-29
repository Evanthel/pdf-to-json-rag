# Security policy

## Supported versions

Security fixes are applied to the current `main` branch and the latest published release. Older releases are not maintained as separate security branches.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/Evanthel/pdf-to-json-rag/security/advisories/new) when it is available. Include the affected command or component, reproduction steps, impact, and any suggested mitigation. Do not put secrets, private PDFs, or working exploit details in a public issue. If private reporting is unavailable, open a minimal public issue asking the maintainer to establish a private reporting channel.

## Security boundaries

- The project is local-first. It does not require API keys, cloud credentials, or a hosted model.
- The web workspace binds to loopback by default and does not implement accounts or authentication. Do not expose it to an untrusted network.
- PDF extraction and OCR process files supplied by the local user. Treat PDFs from unknown sources as untrusted input and keep resource limits enabled.
- `PDF_TO_JSON_RAG_LLM_COMMAND` and `PDF_TO_JSON_RAG_JUDGE_COMMAND` execute a command configured by the local user without a shell. Configure only commands and wrappers you trust.
- ChromaDB is used only through the embedded `PersistentClient`. This project does not run or expose Chroma's HTTP server, authentication providers, or multi-tenant APIs.

## Accepted ChromaDB advisories

The dependency audit currently excludes the following upstream ChromaDB advisories:

- [CVE-2026-45830 / GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) — authenticated cross-tenant data access.
- [CVE-2026-45831 / GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr) — tenant and collection scope is not enforced by `SimpleRBACAuthorizationProvider`.
- [CVE-2026-45833 / GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf) — code injection through an authenticated server-side collection update.

As reviewed on 2026-09-29, the advisories have no patched release and their documented attack paths require Chroma server or authorization functionality that this project does not expose. The CI exceptions are intentionally limited to these CVE identifiers; every other advisory still fails the dependency audit.

This exception must be reviewed before each release and removed as soon as a compatible patched ChromaDB version is available. Any change from `PersistentClient` to `HttpClient`, server mode, authentication, or multi-tenant use blocks release until these advisories are reassessed.
