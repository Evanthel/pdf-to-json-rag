"""Public Hugging Face Space adapter for the local-first PDF RAG pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import gradio as gr
import spaces

# Keep the public demo deterministic and prevent implicit model downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PDF_TO_JSON_RAG_ALLOW_MODEL_DOWNLOAD", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pdf_to_json_rag.config import ProjectPaths  # noqa: E402
from pdf_to_json_rag.web_service import RagWebService, WebServiceError  # noqa: E402


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGE_COUNT = 50
SESSION_TTL_SECONDS = 60 * 60
SESSION_PREFIX = "pdf-rag-space-"

SessionState = dict[str, str]


def _session_paths(session_root: Path) -> ProjectPaths:
    return ProjectPaths.from_data_dir(
        root=Path.cwd(),
        data_dir=session_root / "data",
    )


def _service(paths: ProjectPaths) -> RagWebService:
    return RagWebService(
        paths=paths,
        max_upload_bytes=MAX_UPLOAD_BYTES,
        max_page_count=MAX_PAGE_COUNT,
    )


def _session_root_from_state(state: object) -> Path | None:
    if not isinstance(state, dict):
        return None
    value = state.get("session_root")
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def cleanup_session(state: object) -> None:
    """Delete only adapter-owned temporary session directories."""
    session_root = _session_root_from_state(state)
    if session_root is None:
        return
    try:
        resolved = session_root.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
    except OSError:
        return
    if resolved.parent != temp_root or not resolved.name.startswith(SESSION_PREFIX):
        return
    shutil.rmtree(resolved, ignore_errors=True)


def _structured_preview(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": document.get("doc_id"),
        "title": document.get("title"),
        "page_count": document.get("page_count"),
        "detected_language": document.get("detected_language"),
        "document_type": document.get("document_type"),
        "document_purpose": document.get("document_purpose"),
        "structure_confidence": document.get("structure_confidence"),
        "layout_confidence": document.get("layout_confidence"),
        "extraction_summary": document.get("extraction_summary"),
        "sections_preview": list(document.get("sections") or [])[:3],
        "chunks_preview": list(document.get("chunks") or [])[:3],
    }


def _document_status(summary: dict[str, Any]) -> str:
    label = str(
        summary.get("label")
        or summary.get("title")
        or summary.get("doc_id")
        or "Document"
    )
    page_count = int(summary.get("page_count") or 0)
    chunk_count = int(summary.get("chunk_count") or 0)
    document_type = str(
        summary.get("document_type") or "unclassified document"
    ).replace("_", " ")
    return (
        f"{label}\n"
        f"{page_count} page{'s' if page_count != 1 else ''} · "
        f"{chunk_count} searchable chunk{'s' if chunk_count != 1 else ''} · "
        f"{document_type}"
    )


@spaces.GPU(duration=120)
def process_pdf(
    pdf_path: str | None,
    previous_state: SessionState | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, dict[str, Any], dict[str, Any], str, SessionState]:
    """Extract, structure, chunk, and index one PDF in an isolated session."""
    if not pdf_path:
        raise gr.Error("Select a PDF first.")

    source = Path(pdf_path)
    try:
        file_size = source.stat().st_size
    except OSError as exc:
        raise gr.Error("The selected file is no longer available.") from exc
    if file_size <= 0:
        raise gr.Error("The selected file is empty.")
    if file_size > MAX_UPLOAD_BYTES:
        raise gr.Error("The public demo accepts PDFs up to 10 MiB.")

    cleanup_session(previous_state)
    session_root = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX))
    paths = _session_paths(session_root)
    service = _service(paths)

    try:
        progress(0.1, desc="Validating PDF")
        content = source.read_bytes()
        progress(0.3, desc="Extracting structured content")
        summary = service.ingest_pdf(source.name, content)
        doc_id = str(summary["doc_id"])
        document_path = paths.data_documents / f"{doc_id}.document.json"
        document = json.loads(document_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Structured document is not a JSON object")
        progress(1.0, desc="Document ready")
    except WebServiceError as exc:
        cleanup_session({"session_root": str(session_root)})
        raise gr.Error(exc.message) from exc
    except Exception as exc:
        cleanup_session({"session_root": str(session_root)})
        raise gr.Error("The PDF could not be processed.") from exc

    state: SessionState = {
        "session_root": str(session_root),
        "doc_id": doc_id,
    }
    return (
        _document_status(summary),
        summary,
        _structured_preview(document),
        str(document_path),
        state,
    )


def _citation_rows(result: dict[str, Any]) -> list[list[object]]:
    rows: list[list[object]] = []
    for source in list(result.get("sources") or [])[:8]:
        if not isinstance(source, dict):
            continue
        page_start = int(source.get("page_start") or 0)
        page_end = int(source.get("page_end") or page_start)
        pages = (
            str(page_start) if page_start == page_end else f"{page_start}–{page_end}"
        )
        rows.append(
            [
                pages,
                str(source.get("section_title") or "Untitled section"),
                round(float(source.get("quality_score") or 0.0), 3),
                str(source.get("excerpt") or ""),
            ]
        )
    return rows


def ask_question(
    question: str,
    state: SessionState | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, list[list[object]], dict[str, Any]]:
    """Answer one question against the session-scoped document index."""
    session_root = _session_root_from_state(state)
    if session_root is None or not isinstance(state, dict):
        raise gr.Error("Process a PDF first.")
    doc_id = state.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        raise gr.Error("Process a PDF first.")

    paths = _session_paths(session_root)
    service = _service(paths)
    try:
        progress(0.2, desc="Planning retrieval")
        result = service.ask(doc_id, question, k=5)
        progress(1.0, desc="Answer ready")
    except WebServiceError as exc:
        raise gr.Error(exc.message) from exc
    except Exception as exc:
        raise gr.Error("The question could not be answered.") from exc

    return (
        str(result.get("answer") or ""),
        str(result.get("trust") or "limited"),
        _citation_rows(result),
        result,
    )


CSS = """
:root {
  --ink: #172033;
  --muted: #5d687c;
  --line: #dce2ea;
  --paper: #f7f8fb;
  --accent: #2855d9;
}
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 30px 24px 64px !important;
  color: var(--ink);
}
#brand {
  padding: 18px 0 26px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 28px;
  animation: rise-in 420ms ease-out both;
}
#brand .eyebrow {
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}
#brand h1 {
  margin: 8px 0 8px;
  max-width: 760px;
  font-size: clamp(34px, 5vw, 62px);
  line-height: .98;
  letter-spacing: -.045em;
}
#brand p {
  max-width: 720px;
  margin: 0;
  color: var(--muted);
  font-size: 16px;
  line-height: 1.55;
}
.section-label {
  margin: 4px 0 10px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
}
#process-button, #ask-button {
  transition: transform 150ms ease, filter 150ms ease;
}
#process-button:hover, #ask-button:hover {
  filter: brightness(.96);
  transform: translateY(-1px);
}
#workspace {
  animation: rise-in 520ms 80ms ease-out both;
}
#privacy-note {
  margin-top: 24px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 13px;
}
@keyframes rise-in {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
  #brand, #workspace { animation: none; }
  #process-button, #ask-button { transition: none; }
}
"""

THEME = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="sm",
)


with gr.Blocks(
    title="PDF-to-JSON RAG",
    delete_cache=(SESSION_TTL_SECONDS, SESSION_TTL_SECONDS),
) as demo:
    gr.HTML(
        """
        <header id="brand">
          <div class="eyebrow">Local-first Document AI · public demo</div>
          <h1>PDF to structured evidence.</h1>
          <p>Extract an inspectable JSON document, build a local index, and ask a grounded question with page-level citations.</p>
        </header>
        """
    )

    session = gr.State(
        value=None,
        time_to_live=SESSION_TTL_SECONDS,
        delete_callback=cleanup_session,
    )

    with gr.Column(elem_id="workspace"):
        gr.HTML('<div class="section-label">01 · Structure the document</div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                pdf = gr.File(
                    label="PDF document",
                    file_types=[".pdf"],
                    type="filepath",
                    height=210,
                )
                process_button = gr.Button(
                    "Process PDF",
                    variant="primary",
                    elem_id="process-button",
                )
                document_status = gr.Textbox(
                    label="Processing result",
                    lines=3,
                    interactive=False,
                    placeholder="The document summary will appear here.",
                )
            with gr.Column(scale=3, min_width=420):
                with gr.Tabs():
                    with gr.Tab("JSON preview"):
                        json_preview = gr.JSON(label="First sections and chunks")
                    with gr.Tab("Diagnostics"):
                        document_summary = gr.JSON(label="Document summary")
                json_download = gr.File(label="Download full document JSON")

        gr.HTML(
            '<div class="section-label" style="margin-top:28px">02 · Ask against the evidence</div>'
        )
        with gr.Row(equal_height=True):
            question = gr.Textbox(
                label="Question",
                placeholder="What does this document say about…?",
                lines=2,
                scale=5,
            )
            ask_button = gr.Button(
                "Ask",
                variant="primary",
                scale=1,
                min_width=120,
                elem_id="ask-button",
            )

        with gr.Row(equal_height=False):
            answer = gr.Textbox(
                label="Grounded answer",
                lines=7,
                interactive=False,
                scale=4,
            )
            trust = gr.Textbox(
                label="Trust status",
                lines=2,
                interactive=False,
                scale=1,
            )

        citations = gr.Dataframe(
            headers=["Pages", "Section", "Quality", "Evidence excerpt"],
            datatype=["str", "str", "number", "str"],
            interactive=False,
            label="Citations",
        )
        with gr.Accordion("Complete answer contract", open=False):
            answer_contract = gr.JSON()

    gr.HTML(
        """
        <div id="privacy-note">
          Public demo limits: 10 MiB · 50 pages · one queued task at a time. Do not upload confidential documents.
          Session files expire after one hour. Source and benchmarks:
          <a href="https://github.com/Evanthel/pdf-to-json-rag" target="_blank" rel="noreferrer">GitHub</a>.
        </div>
        """
    )

    process_button.click(
        process_pdf,
        inputs=[pdf, session],
        outputs=[
            document_status,
            document_summary,
            json_preview,
            json_download,
            session,
        ],
        api_name="process_pdf",
        concurrency_id="document-ai",
        concurrency_limit=1,
    )
    ask_button.click(
        ask_question,
        inputs=[question, session],
        outputs=[answer, trust, citations, answer_contract],
        api_name="ask_question",
        concurrency_id="document-ai",
        concurrency_limit=1,
    )
    question.submit(
        ask_question,
        inputs=[question, session],
        outputs=[answer, trust, citations, answer_contract],
        concurrency_id="document-ai",
        concurrency_limit=1,
    )

demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=THEME,
        max_file_size=MAX_UPLOAD_BYTES,
    )
