"""Static checks on the chat UI and embed loader, plus an optional real-browser test (EPM_CHAT_E2E=1)."""

import http.server
import os
import re
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

FRONT = Path(__file__).resolve().parents[1] / "frontend"
HTML = (FRONT / "index.html").read_text(encoding="utf-8")
APP = (FRONT / "app.js").read_text(encoding="utf-8")
EMBED = (FRONT / "embed.js").read_text(encoding="utf-8")


def test_no_inline_script_or_handlers():
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", HTML)
    assert not re.search(r"\son[a-z]+\s*=", HTML)


def test_every_element_used_by_app_exists():
    ids = set(re.findall(r'\$\("([\w-]+)"\)', APP))
    missing = [i for i in ids if f'id="{i}"' not in HTML]
    assert not missing, missing


def test_no_html_injection_sinks_in_app():
    assert "innerHTML" not in APP and "eval(" not in APP and "new Function" not in APP
    assert "insertAdjacentHTML" not in APP


def test_controls_have_accessible_names():
    for b in re.findall(r"<button[^>]*>", HTML):
        assert "aria-label" in b or ">" in b, b
    assert 'role="log"' in HTML and 'aria-live="polite"' in HTML
    assert '<label for="input"' in HTML


def test_embed_is_opt_in_and_origin_checked():
    assert 'hasAttribute("data-epm-chat")' in EMBED
    assert "e.origin !== chatOrigin" in EMBED and "e.source !== frame.contentWindow" in EMBED
    assert 'postMessage(' in EMBED and ', "*")' not in EMBED and ", '*')" not in EMBED
    assert 'attachShadow({ mode: "closed" })' in EMBED
    assert "e.origin !== parentOrigin" in APP and "postMessage(msg, parentOrigin)" in APP


# --------------------------------------------------------------------------- real browser (optional)

def _playwright_ok():
    if os.environ.get("EPM_CHAT_E2E") != "1":
        return False
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _playwright_ok(), reason="set EPM_CHAT_E2E=1 with Playwright + Chromium installed")
def test_browser_end_to_end(settings, tmp_path):
    from playwright.sync_api import sync_playwright

    from epm_chatbot.backend.server import App, make_handler

    settings.allowed_origins = ("http://localhost:18790",)
    app = App(settings)
    chat = ThreadingHTTPServer(("127.0.0.1", 18765), make_handler(app))
    (tmp_path / "host.html").write_text(
        '<!doctype html><html><body><style>button{background:red!important}</style><p>Host</p>'
        '<script src="http://127.0.0.1:18765/embed.js" data-epm-chat defer></script></body></html>')
    host_handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    hosts = [ThreadingHTTPServer(("0.0.0.0", p), host_handler) for p in (18790, 18791)]
    for s in [chat, *hosts]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto("http://127.0.0.1:18765/")
            pg.wait_for_selector(".chip")
            for chip in ["Explain how this prediction works.", "List available models"]:
                pg.click(f'.chip:has-text("{chip}")')
                pg.wait_for_timeout(300)
                pg.locator(".typing").wait_for(state="detached")
            pg.fill("#input", "predict iris with sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
            pg.press("#input", "Enter")
            pg.wait_for_selector(".readout")
            assert pg.locator(".readout .value").inner_text() in ("setosa", "versicolor", "virginica")
            assert not errors
            # allowed host gets the widget; a non-allowlisted host does not
            pg.goto("http://localhost:18790/host.html")
            pg.wait_for_timeout(1500)
            assert pg.locator("[data-epm-chat-host]").count() == 1
            pg.goto("http://127.0.0.1:18791/host.html")
            pg.wait_for_timeout(1500)
            assert pg.locator("[data-epm-chat-host]").count() == 0
            b.close()
    finally:
        for s in [chat, *hosts]:
            s.shutdown()
            s.server_close()
