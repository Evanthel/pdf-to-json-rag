---
title: PDF-to-JSON RAG Demo
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: 3.11
app_file: app.py
pinned: false
license: mit
short_description: Structured PDF extraction and grounded answers with page citations.
---

# PDF-to-JSON RAG Demo

Upload a public, non-confidential PDF, inspect its structured JSON, and ask grounded questions with page-level citations.

The Space is a thin deployment adapter for [Evanthel/pdf-to-json-rag](https://github.com/Evanthel/pdf-to-json-rag). It installs the stable `v0.3.0` release and uses the same extraction, chunking, indexing, retrieval, and answer contracts as the local CLI and web workspace.

## Demo limits

- PDF files up to 10 MiB and 50 pages.
- One extraction or query runs at a time on the public CPU demo.
- Deterministic hash retrieval; no model weights or API keys are required.
- Session data expires after one hour and is stored only on the Space's temporary filesystem.
- Upload only public or non-confidential documents. Hugging Face, not the local-first project boundary, hosts this deployment.

## Local verification

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Tesseract is installed by Hugging Face from `packages.txt`. Install it separately when testing OCR locally. Native-text PDFs work without it.

## Publish

Create a public Gradio Space, then copy the four files from this directory into its repository:

```bash
hf auth login
hf repos create YOUR_USERNAME/pdf-to-json-rag-demo --repo-type space --sdk gradio
git clone https://huggingface.co/spaces/YOUR_USERNAME/pdf-to-json-rag-demo
```

Commit and push `README.md`, `app.py`, `requirements.txt`, and `packages.txt`. The Space rebuilds automatically.
