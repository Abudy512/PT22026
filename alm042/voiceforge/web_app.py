"""
Dependency-free web interface for VoiceForge.

The user interface is a separate static application in ``frontend/`` (HTML,
CSS and ES modules).  This module only serves those files and the JSON API:

* ``GET  /``                 the application shell (``frontend/index.html``)
* ``GET  /css/...``, ``/js/...``, ``/assets/...``  static assets
* ``GET  /health``           engine capabilities and limits
* ``GET  /audio/<file>``     a generated WAV (playback and download)
* ``POST /generate``         JSON ``{"text": ...}`` -> generated WAV metadata
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from voiceforge.app_core import (
    ENGINE_LABELS,
    MAX_TEXT_LENGTH,
    AppError,
    VoiceForgeApp,
)

#: Static frontend root (``alm042/frontend``).
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

#: URL prefixes that may be served from :data:`FRONTEND_DIR`.
STATIC_PREFIXES = ("css/", "js/", "assets/")

#: Only these extensions are served, with an explicit content type.
STATIC_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}

#: Refuse request bodies larger than this (text is limited to 400 chars).
MAX_BODY_BYTES = 64 * 1024


class VoiceForgeServer(ThreadingHTTPServer):
    """HTTP server that carries the application instance."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        app: VoiceForgeApp,
    ):
        super().__init__(server_address, VoiceForgeHandler)
        self.app = app


class VoiceForgeHandler(BaseHTTPRequestHandler):
    """Handles the interface page, speech generation and audio files."""

    server_version = "VoiceForge/1.0"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def app(self) -> VoiceForgeApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")

        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)

        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise AppError("Invalid request length.") from None

        if length <= 0:
            raise AppError("The request was empty.")

        if length > MAX_BODY_BYTES:
            raise AppError("The request is too large.")

        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AppError("The request was not valid JSON.") from None

        if not isinstance(payload, dict):
            raise AppError("The request was not valid JSON.")

        return payload

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (standard library naming)
        # Unquote so that encoded traversal attempts are inspected as paths.
        path = unquote(self.path.split("?", 1)[0])

        if path in ("/", "/index.html"):
            self._send_file(
                FRONTEND_DIR / "index.html",
                "text/html; charset=utf-8",
            )
            return

        if path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "model_loaded": self.app.is_loaded,
                    "checkpoint": str(self.app.checkpoint_path),
                    "limits": {"max_text_length": MAX_TEXT_LENGTH},
                    "engines": self.app.describe_engines(),
                },
            )
            return

        if path.startswith("/audio/"):
            self._send_audio(path[len("/audio/"):])
            return

        relative = path.lstrip("/")

        if relative.startswith(STATIC_PREFIXES):
            self._send_static(relative)
            return

        self._send_error_json(404, "Page not found.")

    def _send_file(self, path: Path, content_type: str) -> None:
        """Send one file from the frontend directory."""

        try:
            data = path.read_bytes()
        except OSError:
            self._send_error_json(404, "Page not found.")
            return

        self._send_bytes(200, data, content_type)

    def _send_static(self, relative: str) -> None:
        """
        Serve a CSS/JS/asset file, refusing anything outside ``frontend/``.

        Traversal is blocked three ways: explicit ``..`` segments, a resolved
        path that must stay inside the frontend root, and an allow-list of
        extensions (so a stray .py or .pt file can never be served).
        """

        parts = Path(relative).parts

        if ".." in parts or relative.startswith("/") or ":" in parts[0]:
            self._send_error_json(404, "Page not found.")
            return

        root = FRONTEND_DIR.resolve()
        candidate = (root / relative).resolve()

        if root not in candidate.parents or not candidate.is_file():
            self._send_error_json(404, "Page not found.")
            return

        content_type = STATIC_MIME_TYPES.get(candidate.suffix.lower())

        if content_type is None:
            self._send_error_json(404, "Page not found.")
            return

        self._send_file(candidate, content_type)

    def _send_audio(self, filename: str) -> None:
        try:
            path = self.app.resolve_audio(filename)
        except AppError as error:
            self._send_error_json(404, str(error))
            return

        data = path.read_bytes()

        self._send_bytes(
            200,
            data,
            "audio/wav",
            {"Content-Disposition": f'inline; filename="{path.name}"'},
        )

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 (standard library naming)
        path = self.path.split("?", 1)[0]

        if path != "/generate":
            self._send_error_json(404, "Page not found.")
            return

        try:
            payload = self._read_json_body()
            result = self.app.synthesize(
                payload.get("text"),
                engine=payload.get("engine") or None,
                voice=payload.get("voice") or None,
            )
        except AppError as error:
            self._send_error_json(400, str(error))
            return

        self._send_json(
            200,
            {
                "ok": True,
                "text": result.text,
                "filename": result.filename,
                "audio_url": f"/audio/{result.filename}",
                "duration_seconds": result.duration_seconds,
                "sample_rate": result.sample_rate,
                "engine": result.engine,
                "padded": result.padded,
                "truncated": result.truncated,
            },
        )

    # Keep the console readable: one short line per request.
    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}", flush=True)


def create_server(
    app: VoiceForgeApp,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> VoiceForgeServer:
    """Create (but do not start) the VoiceForge HTTP server."""

    return VoiceForgeServer((host, port), app)


def run(
    app: VoiceForgeApp,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = False,
) -> None:
    """Serve the interface until interrupted."""

    server = create_server(app, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}"

    print(f"VoiceForge is running at {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VoiceForge.")
    finally:
        server.server_close()
