"""HTTP API: auth, origin enforcement, CORS, limits, SSE streaming, jobs, uploads, static files, logging."""

import http.client
import json
import logging
import threading
from http.server import ThreadingHTTPServer

import pytest

from epm_chatbot.backend.server import App, make_handler

PARTNER = "https://partner.example"


@pytest.fixture()
def srv(settings):
    app = App(settings)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]
    yield app, port
    server.shutdown()
    server.server_close()


def req(port, method, path, body=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    h = dict(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    payload = r.read()
    c.close()
    return r, payload


def session(port, origin=PARTNER):
    r, b = req(port, "POST", "/api/session", {}, {"Origin": origin})
    assert r.status == 200, b
    return json.loads(b)["token"]


def auth(tok, origin=PARTNER):
    return {"Authorization": "Bearer " + tok, "Origin": origin}


def test_health_config_and_page_headers(srv):
    _, port = srv
    r, b = req(port, "GET", "/api/health")
    from epm_chatbot.tests.conftest import HAS_TORCH

    assert r.status == 200 and json.loads(b)["models"] == (3 if HAS_TORCH else 2)
    r, b = req(port, "GET", "/api/config")
    assert json.loads(b)["embed_origins"] == [PARTNER]
    r, b = req(port, "GET", "/")
    assert r.status == 200 and b"<title>EPM Chat</title>" in b
    assert "frame-ancestors 'self' https://partner.example" in r.getheader("Content-Security-Policy")
    assert r.getheader("X-Content-Type-Options") == "nosniff"


def test_static_whitelist_blocks_traversal(srv):
    _, port = srv
    assert req(port, "GET", "/static/app.js")[0].status == 200
    assert req(port, "GET", "/static/../backend/server.py")[0].status == 404
    assert req(port, "GET", "/static/%2e%2e/config/settings.py")[0].status == 404


def test_session_origin_rules(srv):
    _, port = srv
    assert req(port, "POST", "/api/session", {}, {"Origin": "https://evil.example"})[0].status == 403
    assert req(port, "POST", "/api/session", {}, {})[0].status == 403          # no origin, no API key
    tok = session(port)
    r, _ = req(port, "GET", "/api/history", headers=auth(tok, "https://evil.example"))
    assert r.status == 403
    r, _ = req(port, "GET", "/api/history")
    assert r.status == 401


def test_cors_preflight(srv):
    _, port = srv
    r, _ = req(port, "OPTIONS", "/api/chat", headers={"Origin": PARTNER, "Access-Control-Request-Method": "POST"})
    assert r.status == 204 and r.getheader("Access-Control-Allow-Origin") == PARTNER
    r, _ = req(port, "OPTIONS", "/api/chat", headers={"Origin": "https://evil.example"})
    assert r.status == 403 and r.getheader("Access-Control-Allow-Origin") is None


def test_embed_status(srv):
    _, port = srv
    r, _ = req(port, "GET", "/embed/status", headers={"Origin": PARTNER})
    assert r.status == 200 and r.getheader("Access-Control-Allow-Origin") == PARTNER
    r, _ = req(port, "GET", "/embed/status", headers={"Origin": "https://evil.example"})
    assert r.status == 403 and r.getheader("Access-Control-Allow-Origin") is None
    r, b = req(port, "GET", "/embed.js")
    assert r.status == 200 and b"data-epm-chat" in b


def test_chat_json_and_validation(srv):
    _, port = srv
    tok = session(port)
    r, b = req(port, "POST", "/api/chat", {"message": "predict iris with sepal_length=5.1, sepal_width=3.5, "
                                                      "petal_length=1.4, petal_width=0.2"}, auth(tok))
    m = json.loads(b)["message"]
    assert r.status == 200 and any(x and x["type"] == "prediction" for x in m["blocks"])
    assert req(port, "POST", "/api/chat", {"message": ""}, auth(tok))[0].status == 400
    assert req(port, "POST", "/api/chat", {"message": "x" * 5000}, auth(tok))[0].status == 400
    assert req(port, "POST", "/api/chat", {"message": 3}, auth(tok))[0].status == 400
    r, _ = req(port, "POST", "/api/chat", raw=b"message=hi", headers={**auth(tok), "Content-Type": "text/plain"})
    assert r.status == 415
    r, _ = req(port, "POST", "/api/chat", raw=b"{" * 70000, headers={**auth(tok), "Content-Type": "application/json"})
    assert r.status == 413


def parse_sse(b):
    out = []
    for chunk in b.decode().split("\n\n"):
        if not chunk.strip():
            continue
        ev = data = None
        for line in chunk.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        out.append((ev, data))
    return out


def test_sse_stream_with_training_job(srv):
    app, port = srv
    tok = session(port)
    r, b = req(port, "POST", "/api/chat?stream=1", {"message": "train a decision tree on iris"},
               {**auth(tok), "Accept": "text/event-stream"})
    assert r.getheader("Content-Type").startswith("text/event-stream")
    events = parse_sse(b)
    kinds = [e for e, _ in events]
    assert kinds[0] == "message" and "job" in kinds and "reply" in kinds and kinds[-1] == "done"
    reply = next(d for e, d in events if e == "reply")["message"]
    assert reply["job"]["state"] == "done"
    assert any(x and x["type"] == "model" for x in reply["blocks"])
    progress = [d["job"]["progress"] for e, d in events if e == "job"]
    assert progress == sorted(progress)


def test_upload_endpoint_and_download(srv):
    app, port = srv
    tok = session(port)
    from epm_chatbot.ml_service.datasets import load_registry

    csv = load_registry("iris", False).features.head(30).to_csv(index=False).encode()
    r, b = req(port, "POST", "/api/upload", raw=csv, headers={**auth(tok), "Content-Type": "text/csv", "X-Filename": "x.csv"})
    assert r.status == 200 and "Received" in json.dumps(json.loads(b))
    r, b = req(port, "POST", "/api/chat", {"message": "predict with my file"}, auth(tok))
    m = json.loads(b)["message"]
    fid = next(x for x in m["blocks"] if x and x["type"] == "download")["file_id"]
    r, b = req(port, "GET", f"/api/download/{fid}", headers=auth(tok))
    assert r.status == 200 and b.startswith(b"sepal_length") and "attachment" in r.getheader("Content-Disposition")
    other = session(port)
    assert req(port, "GET", f"/api/download/{fid}", headers=auth(other))[0].status == 404
    big = b"a,b\n" + b"1,2\n" * 2_000_000
    try:   # the server answers 413 from Content-Length without reading the body, then closes
        r, _ = req(port, "POST", "/api/upload", raw=big, headers={**auth(tok), "Content-Type": "text/csv"})
        assert r.status == 413
    except (BrokenPipeError, ConnectionResetError):
        pass
    assert len(list(app.uploads.root.glob("*/*.csv"))) == 1      # nothing stored for the oversized file


def test_rate_limit(srv):
    app, port = srv
    app.guard.msg_limiter.capacity = 2.0
    app.guard.msg_limiter.rate = 2 / 60
    tok = session(port)
    codes = [req(port, "POST", "/api/chat", {"message": "hello"}, auth(tok))[0].status for _ in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes[2:]


def test_clear_session(srv):
    _, port = srv
    tok = session(port)
    req(port, "POST", "/api/chat", {"message": "list datasets"}, auth(tok))
    r, b = req(port, "DELETE", "/api/session", headers=auth(tok))
    assert r.status == 200
    r, b = req(port, "GET", "/api/history", headers=auth(tok))
    assert len(json.loads(b)["messages"]) == 1


def test_logs_never_contain_message_text(srv, caplog):
    _, port = srv
    tok = session(port)
    secret = "my-private-patient-note-123"
    with caplog.at_level(logging.INFO, logger="epm_chat"):
        logging.getLogger("epm_chat").propagate = True
        req(port, "POST", "/api/chat", {"message": "describe iris " + secret}, auth(tok))
    assert secret not in caplog.text
    assert tok not in caplog.text


def test_llm_explanations_stream_and_are_verified(settings):
    """With an LLM configured, explanation tokens stream as SSE 'delta' events; the final text is checked."""
    from epm_chatbot.backend.llm import LLMService
    from epm_chatbot.tests.test_llm import FakeClient

    llm = LLMService(settings, client=FakeClient("The model predicted species = setosa because of the petal sizes."))
    app = App(settings, llm=llm)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        tok = session(port)
        req(port, "POST", "/api/chat", {"message": "predict iris with sepal_length=5.1, sepal_width=3.5, "
                                                   "petal_length=1.4, petal_width=0.2"}, auth(tok))
        r, b = req(port, "POST", "/api/chat?stream=1", {"message": "explain the prediction"},
                   {**auth(tok), "Accept": "text/event-stream"})
        ev = parse_sse(b)
        assert sum(1 for e, _ in ev if e == "delta") >= 3
        final = next(d for e, d in ev if e == "message")["message"]
        assert any(x and x.get("type") == "source" and "checked" in x["text"] for x in final["blocks"])
        # a fabricated number makes the server fall back to the deterministic explanation
        llm.client = FakeClient("This is 99.97% certain across 12,345 flowers.")
        r, b = req(port, "POST", "/api/chat", {"message": "explain the prediction"}, auth(tok))
        text = json.dumps(json.loads(b))
        assert "12,345" not in text
        assert "99.97" not in text
    finally:
        server.shutdown()
        server.server_close()
