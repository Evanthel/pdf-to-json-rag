"""Dependency-free HTTP server for the local PDF RAG workspace."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from importlib import resources
import json
import os
from pathlib import PurePosixPath
import re
import sys
from typing import Any
from urllib.parse import unquote, urlparse
import webbrowser

from .web_service import DEFAULT_MAX_UPLOAD_BYTES, RagWebService, WebServiceError


ASSET_PACKAGE = "pdf_to_json_rag"
ASSET_DIRECTORY = "assets/web"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
QUERY_PATH_RE = re.compile(r"^/api/documents/([^/]+)/query$")
DOCUMENT_PATH_RE = re.compile(r"^/api/documents/([^/]+)$")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class WebServer(ThreadingHTTPServer):
    """HTTP server carrying the application service dependency."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], service: Any) -> None:
        self.configured_host = server_address[0].strip().lower()
        super().__init__(server_address, WebRequestHandler)
        self.service = service


class WebRequestHandler(BaseHTTPRequestHandler):
    """Serve static assets and a compact same-origin JSON API."""

    server: WebServer

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[web] {self.address_string()} - {format % args}\n")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )

    def _send_bytes(
        self,
        payload: bytes,
        *,
        status: int = 200,
        content_type: str = "application/octet-stream",
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, payload: dict[str, object], *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _send_result(self, result: object, *, status: int = 200) -> None:
        self._send_json({"ok": True, "result": result}, status=status)

    def _send_service_error(self, error: WebServiceError) -> None:
        public_details = error.details if error.status < 500 else {}
        self._send_json(
            {
                "ok": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": public_details,
                },
            },
            status=error.status,
        )

    def _path(self) -> str:
        return urlparse(self.path).path

    def _request_hostname(self) -> str | None:
        host_header = self.headers.get("Host", "").strip()
        if not host_header:
            return None
        try:
            return urlparse(f"//{host_header}").hostname
        except ValueError:
            return None

    def _validate_request_source(self, *, state_changing: bool = False) -> None:
        request_hostname = self._request_hostname()
        if self.server.configured_host in LOOPBACK_HOSTS and request_hostname not in LOOPBACK_HOSTS:
            raise WebServiceError(
                "invalid_host",
                "The request host is not allowed for this local server.",
                status=400,
            )
        if not state_changing:
            return

        fetch_site = self.headers.get("Sec-Fetch-Site", "").strip().lower()
        if fetch_site == "cross-site":
            raise WebServiceError(
                "cross_origin_request",
                "Cross-origin requests are not allowed.",
                status=403,
            )
        origin = self.headers.get("Origin")
        if not origin:
            return
        try:
            parsed_origin = urlparse(origin)
        except ValueError as exc:
            raise WebServiceError(
                "cross_origin_request",
                "Cross-origin requests are not allowed.",
                status=403,
            ) from exc
        request_host = self.headers.get("Host", "").strip().lower()
        if (
            parsed_origin.scheme not in {"http", "https"}
            or not parsed_origin.netloc
            or parsed_origin.netloc.lower() != request_host
        ):
            raise WebServiceError(
                "cross_origin_request",
                "Cross-origin requests are not allowed.",
                status=403,
            )

    def _require_content_type(self, expected: str) -> None:
        received = self.headers.get_content_type().lower()
        if received != expected:
            raise WebServiceError(
                "unsupported_media_type",
                f"Content-Type must be {expected}.",
                status=415,
            )

    def _asset_bytes(self, asset_name: str) -> bytes:
        safe_name = PurePosixPath(asset_name).name
        asset = resources.files(ASSET_PACKAGE).joinpath(ASSET_DIRECTORY, safe_name)
        return asset.read_bytes()

    def _serve_static(self, path: str) -> bool:
        spec = STATIC_FILES.get(path)
        if spec is None:
            return False
        asset_name, content_type = spec
        try:
            payload = self._asset_bytes(asset_name)
        except (FileNotFoundError, OSError):
            self._send_json(
                {
                    "ok": False,
                    "error": {
                        "code": "asset_not_found",
                        "message": "A web interface asset could not be found.",
                        "details": {},
                    },
                },
                status=500,
            )
            return True
        self._send_bytes(
            payload,
            content_type=content_type,
            cache_control="public, max-age=300",
        )
        return True

    def _read_json(self, *, limit: int = 64 * 1024) -> dict[str, object]:
        self._require_content_type("application/json")
        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length or "0")
        except ValueError as exc:
            raise WebServiceError("invalid_content_length", "Invalid request length.") from exc
        if length <= 0 or length > limit:
            raise WebServiceError("invalid_request_body", "Invalid request size.", status=413)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebServiceError("invalid_json", "The request does not contain valid JSON.") from exc
        if not isinstance(payload, dict):
            raise WebServiceError("invalid_json", "The request must be a JSON object.")
        return payload

    def _read_upload(self) -> tuple[str, bytes]:
        self._require_content_type("application/pdf")
        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length or "0")
        except ValueError as exc:
            raise WebServiceError("invalid_content_length", "Invalid file length.") from exc
        limit = int(getattr(self.server.service, "max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES))
        if length <= 0:
            raise WebServiceError("empty_upload", "The selected file is empty.")
        if length > limit:
            raise WebServiceError(
                "upload_too_large",
                f"The file exceeds the {limit // (1024 * 1024)} MB limit.",
                status=413,
            )
        filename = unquote(self.headers.get("X-PDF-Filename", "document.pdf"))
        return filename, self.rfile.read(length)

    def do_HEAD(self) -> None:
        try:
            self._validate_request_source()
            path = self._path()
            if not self._serve_static(path):
                self._send_json(
                    {"ok": False, "error": {"code": "not_found", "message": "Resource not found.", "details": {}}},
                    status=404,
                )
        except WebServiceError as error:
            self._send_service_error(error)

    def do_GET(self) -> None:
        path = self._path()
        try:
            self._validate_request_source()
            if self._serve_static(path):
                return
            if path == "/api/health":
                self._send_result({"status": "ok", "service": "pdf-to-json-rag-web"})
                return
            if path == "/api/documents":
                self._send_result(self.server.service.list_documents())
                return
            match = DOCUMENT_PATH_RE.fullmatch(path)
            if match:
                self._send_result(self.server.service.get_document(unquote(match.group(1))))
                return
            self._send_json(
                {"ok": False, "error": {"code": "not_found", "message": "Resource not found.", "details": {}}},
                status=404,
            )
        except WebServiceError as error:
            self._send_service_error(error)
        except Exception:
            self._send_json(
                {"ok": False, "error": {"code": "internal_error", "message": "An internal server error occurred.", "details": {}}},
                status=500,
            )

    def do_POST(self) -> None:
        path = self._path()
        try:
            self._validate_request_source(state_changing=True)
            if path == "/api/documents":
                filename, content = self._read_upload()
                self._send_result(
                    self.server.service.ingest_pdf(filename, content),
                    status=HTTPStatus.CREATED,
                )
                return
            match = QUERY_PATH_RE.fullmatch(path)
            if match:
                payload = self._read_json()
                query = payload.get("query")
                if not isinstance(query, str):
                    raise WebServiceError("invalid_query", "The query field must be a string.")
                k_value = payload.get("k", 5)
                if not isinstance(k_value, int) or isinstance(k_value, bool):
                    raise WebServiceError("invalid_top_k", "The k field must be an integer.")
                self._send_result(
                    self.server.service.ask(unquote(match.group(1)), query, k=k_value)
                )
                return
            self._send_json(
                {"ok": False, "error": {"code": "not_found", "message": "Resource not found.", "details": {}}},
                status=404,
            )
        except WebServiceError as error:
            self._send_service_error(error)
        except Exception:
            self._send_json(
                {"ok": False, "error": {"code": "internal_error", "message": "An internal server error occurred.", "details": {}}},
                status=500,
            )


def create_server(host: str, port: int, service: Any | None = None) -> WebServer:
    """Create the local server, allowing an ephemeral port in tests."""
    return WebServer((host, port), service or RagWebService())


def serve(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False) -> None:
    """Run the local web workspace until interrupted."""
    server = create_server(host, port)
    actual_host, actual_port = server.server_address[:2]
    try:
        is_unspecified_host = ip_address(actual_host).is_unspecified
    except ValueError:
        is_unspecified_host = False
    browser_host = "127.0.0.1" if is_unspecified_host else actual_host
    url = f"http://{browser_host}:{actual_port}"
    print(f"PDF RAG workspace: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping web workspace.")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local PDF RAG web workspace.")
    parser.add_argument("--host", default=os.environ.get("PDF_TO_JSON_RAG_WEB_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PDF_TO_JSON_RAG_WEB_PORT", "8765")),
    )
    parser.add_argument("--open", action="store_true", help="Open the workspace in the default browser.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    serve(host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
