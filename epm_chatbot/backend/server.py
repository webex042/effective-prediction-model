"""
EPM Chat HTTP server (standard library only).

Run with ``python -m epm_chatbot`` (see README). It serves the chat page, the
optional embed loader, and a JSON/SSE API. It is independent of the main EPM
website and its ``run.py`` server.
"""

from __future__ import annotations

import json
import logging
import math
import mimetypes
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from epm_chatbot import __version__
from epm_chatbot.backend.controller import Controller, _cell
from epm_chatbot.backend.jobs import JobManager, QueueFull
from epm_chatbot.backend.llm import LLMService
from epm_chatbot.backend.logging_setup import setup_logging
from epm_chatbot.backend.security import SECURITY_HEADERS, AuthError, Guard, page_csp
from epm_chatbot.config import PACKAGE_ROOT, Settings
from epm_chatbot.ml_service.datasets import UploadStore
from epm_chatbot.ml_service.model_store import ModelStore

log = logging.getLogger("epm_chat.http")
FRONTEND = PACKAGE_ROOT / "frontend"
STATIC_FILES = {"app.js", "app.css", "favicon.svg"}


class App:
    """Wires settings, security, storage, jobs and the controller together."""

    def __init__(self, settings: Settings, llm: Optional[LLMService] = None) -> None:
        self.s = settings
        self.guard = Guard(settings)
        self.store = ModelStore(settings.model_dir)
        self.uploads = UploadStore(settings.upload_dir, settings.upload_ttl_s, settings.max_upload_bytes)
        self.jobs = JobManager(settings.job_workers, settings.max_queued_jobs)
        self.llm = llm or LLMService(settings)
        self.controller = Controller(settings, self.store, self.uploads, self.jobs, self.llm)
        self.controller.train_limiter = self.guard.train_limiter
        self.started = time.time()
        self._sweeper = threading.Thread(target=self._sweep_loop, daemon=True, name="epm-chat-sweeper")
        self._sweeper.start()

    def _sweep_loop(self) -> None:
        while True:
            time.sleep(300)
            try:
                n1 = self.uploads.sweep()
                n2 = self.store.sweep_private(self.s.upload_ttl_s)
                if n1 or n2:
                    log.info("sweep", extra={"uploads_removed": n1, "private_models_removed": n2})
            except Exception:  # pragma: no cover
                log.exception("sweep failed")


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return _cell(o)


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(_clean(obj), allow_nan=False, separators=(",", ":")).encode()


def make_handler(app: App):
    s = app.s

    class Handler(BaseHTTPRequestHandler):
        server_version = "EPMChat/" + __version__
        sys_version = ""
        protocol_version = "HTTP/1.0"

        # ---------------------------------------------------------------- utils
        def log_message(self, fmt: str, *args: Any) -> None:  # silence default access log (we log JSON)
            pass

        def _client_ip(self) -> str:
            if s.trust_proxy:
                fwd = self.headers.get("X-Forwarded-For", "")
                if fwd:
                    return fwd.split(",")[0].strip()
            return self.client_address[0]

        def _host_origin(self) -> str:
            if s.public_origin:
                return s.public_origin.lower()
            proto = "http"
            if s.trust_proxy and self.headers.get("X-Forwarded-Proto", "").lower() == "https":
                proto = "https"
            return f"{proto}://{(self.headers.get('Host') or '').lower()}"

        def _hdrs(self) -> Dict[str, str]:
            return {k.lower(): v for k, v in self.headers.items()}

        def _send(self, status: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            self._status = status

        def _json(self, status: int, obj: Any, extra: Optional[Dict[str, str]] = None) -> None:
            h = {"Cache-Control": "no-store"}
            h.update(self._cors)
            h.update(extra or {})
            self._send(status, _json_bytes(obj), "application/json; charset=utf-8", h)

        def _error(self, status: int, message: str, extra: Optional[Dict[str, str]] = None) -> None:
            self._json(status, {"error": message}, extra)

        def _read_body(self, limit: int) -> bytes:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise AuthError("Invalid Content-Length.", 400) from None
            if n < 0:
                raise AuthError("Invalid Content-Length.", 400)
            if n > limit:
                raise AuthError(f"Request too large (limit {limit // 1024} KB).", 413)
            return self.rfile.read(n) if n else b""

        def _read_json(self) -> Dict[str, Any]:
            raw = self._read_body(s.max_body_bytes)
            if not raw:
                return {}
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                raise AuthError("Content-Type must be application/json.", 415)
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                raise AuthError("Body must be valid JSON.", 400) from None
            if not isinstance(obj, dict):
                raise AuthError("Body must be a JSON object.", 400)
            return obj

        def _auth(self):
            q = parse_qs(urlparse(self.path).query)
            claims = app.guard.authenticate(self._hdrs(), self._host_origin(), (q.get("token") or [""])[0])
            return claims

        # ---------------------------------------------------------------- dispatch
        def _handle(self, method: str) -> None:
            t0 = time.perf_counter()
            self._status = 0
            path = urlparse(self.path).path
            hdrs = self._hdrs()
            origin = app.guard.request_origin(hdrs, self._host_origin())
            self._cors = app.guard.cors_headers(origin, self._host_origin()) if path.startswith(("/api/", "/embed/")) else {}
            try:
                if path.startswith("/api/") and not app.guard.ip_limiter.allow(f"ip:{self._client_ip()}"):
                    return self._error(429, "Too many requests. Please slow down.", {"Retry-After": "30"})
                if method == "OPTIONS":
                    if origin and not self._cors.get("Access-Control-Allow-Origin") and origin != self._host_origin():
                        return self._send(403, b"", "text/plain", {"Vary": "Origin"})
                    return self._send(204, b"", "text/plain", self._cors)
                # cross-origin browser calls to the API must come from an allowlisted site
                if path.startswith("/api/") and origin and not app.guard.origin_permitted(origin, self._host_origin()):
                    return self._error(403, "This website is not permitted to use EPM Chat.")
                self._route(method, path)
            except AuthError as e:
                self._error(e.status, str(e))
            except QueueFull as e:
                self._error(429, str(e))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                log.exception("unhandled error", extra={"path": path})
                if not self._status:
                    self._error(500, "Internal error. The request was not completed.")
            finally:
                log.info("request", extra={"method": method, "path": _safe_path(path), "status": self._status,
                                           "ms": round((time.perf_counter() - t0) * 1000, 1)})

        def do_GET(self):  # noqa: N802
            self._handle("GET")

        def do_HEAD(self):  # noqa: N802
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

        def do_DELETE(self):  # noqa: N802
            self._handle("DELETE")

        def do_OPTIONS(self):  # noqa: N802
            self._handle("OPTIONS")

        # ---------------------------------------------------------------- routes
        def _route(self, method: str, path: str) -> None:
            if method == "GET" and path in ("/", "/index.html"):
                return self._page()
            if method == "GET" and path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if method == "GET" and path == "/embed.js":
                return self._file(FRONTEND / "embed.js", "application/javascript; charset=utf-8",
                                  {"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"})
            if method == "GET" and path == "/embed/status":
                origin = app.guard.request_origin(self._hdrs(), self._host_origin())
                if not origin or not s.is_allowed_origin(origin):
                    return self._send(403, b'{"allowed":false}', "application/json", {"Vary": "Origin"})
                return self._json(200, {"allowed": True, "version": __version__})
            if method == "GET" and path == "/api/health":
                return self._json(200, {"status": "ok", "version": __version__,
                                        "models": len(app.store.list(owner=None)), "llm": app.llm.enabled,
                                        "uptime_s": round(time.time() - app.started)})
            if method == "GET" and path == "/api/config":
                return self._json(200, {"max_upload_bytes": s.max_upload_bytes, "max_message_chars": s.max_message_chars,
                                        "access_code_required": bool(s.access_code), "llm": app.llm.enabled,
                                        "embed_origins": list(s.allowed_origins), "version": __version__})
            if method == "POST" and path == "/api/session":
                return self._open_session()

            claims = self._auth()
            sid = claims.sid
            if method == "DELETE" and path == "/api/session":
                return self._json(200, {"message": app.controller.clear(sid)})
            if method == "GET" and path == "/api/history":
                return self._json(200, {"messages": app.controller.history(sid)})
            if method == "POST" and path == "/api/chat":
                return self._chat(sid)
            if method == "POST" and path == "/api/upload":
                return self._upload(sid)
            m = re.fullmatch(r"/api/jobs/(j_[0-9a-f]{12})(/cancel|/events)?", path)
            if m:
                return self._job(sid, m.group(1), m.group(2) or "", method)
            m = re.fullmatch(r"/api/download/(d_[0-9a-f]{12})", path)
            if m and method == "GET":
                f = app.controller.download(sid, m.group(1))
                if not f:
                    return self._error(404, "That download has expired.")
                name, data = f
                safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)[:80]
                return self._send(200, data, "text/csv; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{safe}"', "Cache-Control": "no-store",
                                   **self._cors})
            return self._error(404, "Not found.")

        def _page(self) -> None:
            body = (FRONTEND / "index.html").read_bytes()
            self._send(200, body, "text/html; charset=utf-8",
                       {"Content-Security-Policy": page_csp(s), "Cache-Control": "no-cache"})

        def _static(self, name: str) -> None:
            if name not in STATIC_FILES:
                return self._error(404, "Not found.")
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._file(FRONTEND / name, ctype, {"Cache-Control": "no-cache"})

        def _file(self, p: Path, ctype: str, extra: Dict[str, str]) -> None:
            try:
                data = p.read_bytes()
            except FileNotFoundError:
                return self._error(404, "Not found.")
            self._send(200, data, ctype, extra)

        def _open_session(self) -> None:
            body = self._read_json()
            hdrs = self._hdrs()
            # resume an existing, still-valid session (page reload) instead of opening a new one
            if hdrs.get("authorization"):
                try:
                    claims = app.guard.authenticate(hdrs, self._host_origin())
                    token = hdrs["authorization"][7:].strip()
                    hist = app.controller.history(claims.sid)
                    return self._json(200, {"token": token, "expires_at": claims.exp, "resumed": True,
                                            "messages": hist or [app.controller.welcome()]})
                except AuthError:
                    pass
            token, claims = app.guard.open_session(hdrs, self._host_origin(), self._client_ip(),
                                                   str(body.get("access_code") or ""))
            welcome = app.controller.welcome()
            app.controller._log(app.controller.session(claims.sid), welcome)
            self._json(200, {"token": token, "expires_at": claims.exp, "resumed": False, "messages": [welcome]})

        def _chat(self, sid: str) -> None:
            if not app.guard.msg_limiter.allow(f"msg:{sid}"):
                return self._error(429, "You're sending messages too quickly. Wait a few seconds and retry.",
                                   {"Retry-After": "10"})
            body = self._read_json()
            msg = body.get("message")
            if not isinstance(msg, str) or not msg.strip():
                return self._error(400, "Message must be a non-empty string.")
            if len(msg) > s.max_message_chars:
                return self._error(400, f"Messages are limited to {s.max_message_chars} characters.")
            msg = _sanitize(msg)
            q = parse_qs(urlparse(self.path).query)
            stream = (q.get("stream") or ["0"])[0] == "1" or "text/event-stream" in (self.headers.get("Accept") or "")
            if not stream:
                return self._json(200, {"message": app.controller.handle(sid, msg)})
            self._sse_start()
            reply = app.controller.handle(sid, msg, on_delta=lambda d: self._sse("delta", {"text": d}))
            self._sse("message", {"message": reply})
            if reply.get("job"):
                self._follow_job(sid, reply["job"]["id"])
            self._sse("done", {})

        def _upload(self, sid: str) -> None:
            if not app.guard.msg_limiter.allow(f"msg:{sid}", cost=2):
                return self._error(429, "Too many uploads; wait a moment.", {"Retry-After": "20"})
            fname = (self.headers.get("X-Filename") or "upload.csv")[:120]
            raw = self._read_body(s.max_upload_bytes)
            self._json(200, {"message": app.controller.handle_upload(sid, fname, raw)})

        def _job(self, sid: str, job_id: str, action: str, method: str) -> None:
            if action == "/cancel" and method == "POST":
                ok = app.jobs.cancel(job_id, sid)
                return self._json(200 if ok else 404, {"cancelled": ok})
            if action == "/events" and method == "GET":
                if app.jobs.get(job_id, sid) is None:
                    return self._error(404, "Unknown job.")
                self._sse_start()
                self._follow_job(sid, job_id)
                self._sse("done", {})
                return
            st = app.controller.job_status(sid, job_id)
            if st is None:
                return self._error(404, "Unknown job.")
            self._json(200, st)

        # ---------------------------------------------------------------- SSE
        def _sse_start(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            for k, v in {**SECURITY_HEADERS, **self._cors}.items():
                self.send_header(k, v)
            self.end_headers()
            self.close_connection = True
            self._status = 200

        def _sse(self, event: str, data: Dict[str, Any]) -> None:
            self.wfile.write(f"event: {event}\ndata: ".encode() + _json_bytes(data) + b"\n\n")
            self.wfile.flush()

        def _follow_job(self, sid: str, job_id: str) -> None:
            job = app.jobs.get(job_id, sid)
            if job is None:
                return
            last = None
            deadline = time.monotonic() + s.train_timeout_s + 60
            while time.monotonic() < deadline:
                pub = job.public()
                key = (pub["state"], pub["progress"], pub["message"])
                if key != last:
                    self._sse("job", {"job": pub})
                    last = key
                if job.done_event.wait(0.4):
                    break
            self._sse("job", {"job": job.public()})
            if job.reply:
                self._sse("reply", {"message": job.reply})

    return Handler


def _safe_path(path: str) -> str:
    return re.sub(r"/(j|d)_[0-9a-f]{12}", r"/\1_…", path)[:80]


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    """Strip control characters; the UI renders text with textContent, so no HTML is ever interpreted."""
    return _CTRL.sub("", text).strip()


def create_server(settings: Optional[Settings] = None, llm: Optional[LLMService] = None) -> Tuple[ThreadingHTTPServer, App]:
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level)
    app = App(settings, llm)
    server = ThreadingHTTPServer((settings.host, settings.port), make_handler(app))
    server.daemon_threads = True
    return server, app


def serve(settings: Optional[Settings] = None) -> None:
    server, app = create_server(settings)
    host, port = server.server_address[:2]
    log.info("EPM Chat listening", extra={"url": f"http://{host}:{port}", "allowed_origins": list(app.s.allowed_origins),
                                          "models": len(app.store.list(owner=None)), "llm": app.llm.enabled})
    print(f"EPM Chat running at http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
