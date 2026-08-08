from __future__ import annotations

import json
from pathlib import Path
import tempfile
from threading import Thread
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pdf_to_json_rag.config import ProjectPaths
from pdf_to_json_rag.web import create_server
from pdf_to_json_rag.web_service import RagWebService, WebServiceError, answer_view


class FakeWebService:
    max_upload_bytes = 1024 * 1024

    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.questions: list[tuple[str, str, int]] = []
        self.document = {
            "doc_id": "demo-document",
            "label": "Demo document",
            "page_count": 2,
            "chunk_count": 4,
            "diagnostics": {"status": "ready"},
        }

    def list_documents(self) -> list[dict[str, object]]:
        return [self.document]

    def get_document(self, doc_id: str) -> dict[str, object]:
        if doc_id != "demo-document":
            raise WebServiceError("document_not_found", "Document not found.", status=404)
        return self.document

    def ingest_pdf(self, filename: str, content: bytes) -> dict[str, object]:
        self.uploads.append((filename, content))
        return self.document

    def ask(self, doc_id: str, query: str, *, k: int = 5) -> dict[str, object]:
        self.questions.append((doc_id, query, k))
        return {
            "query": query,
            "answer": "Grounded answer",
            "trust": "supported",
            "evidence": [],
            "sources": [],
        }


class WebHttpContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeWebService()
        self.server = create_server("127.0.0.1", 0, self.service)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _json(self, path: str, *, data: bytes | None = None, headers: dict[str, str] | None = None):
        request = Request(self.base_url + path, data=data, headers=headers or {})
        with urlopen(request, timeout=5) as response:
            return response.status, response.headers, json.loads(response.read())

    def test_static_workspace_and_security_headers(self) -> None:
        with urlopen(self.base_url + "/", timeout=5) as response:
            body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("PDF RAG", body)
            self.assertIn("content-security-policy", {key.lower() for key in response.headers.keys()})
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")
            self.assertEqual(response.headers["Cross-Origin-Resource-Policy"], "same-origin")
            self.assertIn("camera=()", response.headers["Permissions-Policy"])

        with urlopen(self.base_url + "/app.css", timeout=5) as response:
            self.assertEqual(response.headers.get_content_type(), "text/css")
            self.assertIn(b"--accent", response.read())

    def test_health_and_document_routes(self) -> None:
        status, _headers, health = self._json("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["result"]["status"], "ok")

        status, _headers, documents = self._json("/api/documents")
        self.assertEqual(status, 200)
        self.assertEqual(documents["result"][0]["doc_id"], "demo-document")

        status, _headers, document = self._json("/api/documents/demo-document")
        self.assertEqual(status, 200)
        self.assertEqual(document["result"]["chunk_count"], 4)

    def test_raw_pdf_upload_contract(self) -> None:
        status, _headers, payload = self._json(
            "/api/documents",
            data=b"%PDF-1.7\nexample",
            headers={
                "Content-Type": "application/pdf",
                "X-PDF-Filename": "report%20final.pdf",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["result"]["doc_id"], "demo-document")
        self.assertEqual(self.service.uploads, [("report final.pdf", b"%PDF-1.7\nexample")])

    def test_query_contract(self) -> None:
        body = json.dumps({"query": "What is covered?", "k": 6}).encode("utf-8")
        status, _headers, payload = self._json(
            "/api/documents/demo-document/query",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": self.base_url,
                "Sec-Fetch-Site": "same-origin",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["result"]["answer"], "Grounded answer")
        self.assertEqual(self.service.questions, [("demo-document", "What is covered?", 6)])

    def test_rejects_cross_origin_and_simple_content_types(self) -> None:
        query_body = json.dumps({"query": "Expensive query"}).encode("utf-8")
        with self.assertRaises(HTTPError) as raised:
            self._json(
                "/api/documents/demo-document/query",
                data=query_body,
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://attacker.example",
                },
            )
        self.assertEqual(raised.exception.code, 403)
        self.assertEqual(self.service.questions, [])

        with self.assertRaises(HTTPError) as raised:
            self._json(
                "/api/documents",
                data=b"%PDF-1.7\nexample",
                headers={"Content-Type": "text/plain"},
            )
        self.assertEqual(raised.exception.code, 415)
        self.assertEqual(self.service.uploads, [])

        with self.assertRaises(HTTPError) as raised:
            self._json(
                "/api/documents/demo-document/query",
                data=query_body,
                headers={"Content-Type": "text/plain"},
            )
        self.assertEqual(raised.exception.code, 415)
        self.assertEqual(self.service.questions, [])

    def test_rejects_dns_rebinding_host_on_loopback(self) -> None:
        request = Request(
            self.base_url + "/api/health",
            headers={"Host": "attacker.example"},
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 400)
        payload = json.loads(raised.exception.read())
        self.assertEqual(payload["error"]["code"], "invalid_host")

    def test_missing_resource_uses_safe_error_envelope(self) -> None:
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base_url + "/api/missing", timeout=5)
        self.assertEqual(raised.exception.code, 404)
        payload = json.loads(raised.exception.read())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "not_found")

    def test_internal_error_details_are_not_exposed(self) -> None:
        def fail_with_internal_detail(_doc_id: str) -> dict[str, object]:
            raise WebServiceError(
                "invalid_document_artifact",
                "The saved document has an invalid format.",
                status=500,
                details={"path": "/private/path", "reason": "parser detail"},
            )

        self.service.get_document = fail_with_internal_detail
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base_url + "/api/documents/demo-document", timeout=5)
        payload = json.loads(raised.exception.read())
        self.assertEqual(payload["error"]["details"], {})


class RagWebServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.paths = ProjectPaths.from_data_dir(root, root / "data")
        self.service = RagWebService(paths=self.paths, max_upload_bytes=32)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_rejects_empty_non_pdf_and_large_uploads(self) -> None:
        with self.assertRaisesRegex(WebServiceError, "empty"):
            self.service.ingest_pdf("empty.pdf", b"")
        with self.assertRaisesRegex(WebServiceError, "valid PDF"):
            self.service.ingest_pdf("notes.txt", b"plain text")
        with self.assertRaisesRegex(WebServiceError, "limit"):
            self.service.ingest_pdf("large.pdf", b"%PDF-" + b"x" * 40)

    def test_lists_saved_documents_with_compact_diagnostics(self) -> None:
        payload = {
            "doc_id": "saved-report",
            "source_pdf": "saved-report.pdf",
            "page_count": 3,
            "title": "Saved report",
            "document_type": "report",
            "detected_language": "en",
            "structure_confidence": 0.82,
            "layout_confidence": 0.74,
            "sections": [{"section_id": "intro"}],
            "chunks": [{"chunk_id": "a"}, {"chunk_id": "b"}],
            "extraction_summary": {
                "ocr_used": False,
                "pdf_inspector": {
                    "status": "ok",
                    "effective_mode": "assist",
                    "confidence": 0.9,
                    "pages_with_tables": [1],
                },
            },
        }
        document_path = self.paths.data_documents / "saved-report.document.json"
        document_path.write_text(json.dumps(payload), encoding="utf-8")

        documents = self.service.list_documents()

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["label"], "Saved report")
        self.assertEqual(documents[0]["chunk_count"], 2)
        self.assertEqual(documents[0]["diagnostics"]["status"], "ready")
        self.assertEqual(documents[0]["diagnostics"]["tables"]["pages"], [1])

    def test_invalid_document_id_never_resolves_outside_data_root(self) -> None:
        with self.assertRaisesRegex(WebServiceError, "document ID"):
            self.service.get_document("../../secret")


class AnswerViewTests(unittest.TestCase):
    def test_preserves_human_page_numbers_and_limits_public_fields(self) -> None:
        evidence = SimpleNamespace(
            chunk_id="demo-0001",
            page_start=1,
            page_end=2,
            section_title="Summary",
            sentence="A grounded sentence.",
            score=0.87654,
            matched_terms=["grounded"],
        )
        chunk = SimpleNamespace(
            chunk_id="demo-0001",
            page_start=1,
            page_end=2,
            section_title="Summary",
            chunk_type="text",
            extraction_method="native",
            quality_score=0.9,
            text="A source excerpt.",
        )
        result = SimpleNamespace(
            query="What is covered?",
            query_intent="grounded_evidence",
            answer="A grounded answer.",
            evidence=[evidence],
            top_k_hits=[chunk],
            answer_trace={"claim_alignment": {"weak_claim_count": 0, "unsupported_claim_count": 0}},
        )

        payload = answer_view(result)

        self.assertEqual(payload["evidence"][0]["page_start"], 1)
        self.assertEqual(payload["evidence"][0]["page_end"], 2)
        self.assertEqual(payload["sources"][0]["page_start"], 1)
        self.assertEqual(payload["trust"], "supported")
        self.assertNotIn("answer_trace", payload)


if __name__ == "__main__":
    unittest.main()
