"""Origin allowlist, signed session tokens, rate limiting, CSP and settings validation."""

import pytest

from epm_chatbot.backend.security import AuthError, Guard, RateLimiter, TokenSigner, page_csp
from epm_chatbot.config import Settings

HOST = "http://chat.local"


def test_token_roundtrip_and_tamper():
    s = TokenSigner("k" * 40, 60)
    tok, claims = s.issue("https://partner.example")
    assert s.verify(tok).sid == claims.sid
    payload, sig = tok.split(".")
    with pytest.raises(AuthError):
        s.verify(payload + "." + sig[:-2] + ("AA" if sig[-2:] != "AA" else "BB"))
    with pytest.raises(AuthError):
        TokenSigner("other" * 10, 60).verify(tok)          # different secret
    with pytest.raises(AuthError):
        s.verify("garbage")


def test_token_expiry():
    s = TokenSigner("k" * 40, 10)
    tok, _ = s.issue("https://a.example", now=1000)
    s.verify(tok, now=1005)
    with pytest.raises(AuthError, match="expired"):
        s.verify(tok, now=1011)


def test_open_session_requires_permitted_origin(settings):
    g = Guard(settings)
    tok, c = g.open_session({"origin": "https://partner.example"}, HOST, "1.1.1.1")
    assert c.origin == "https://partner.example"
    with pytest.raises(AuthError) as e:
        g.open_session({"origin": "https://evil.example"}, HOST, "1.1.1.2")
    assert e.value.status == 403
    # the chat's own page (same origin) may always open a session
    g.open_session({"origin": HOST}, HOST, "1.1.1.3")


def test_no_origin_needs_server_api_key(settings):
    with pytest.raises(AuthError):
        Guard(settings).open_session({}, HOST, "2.2.2.2")
    settings.api_keys = ("server-key-123",)
    tok, c = Guard(settings).open_session({"x-api-key": "server-key-123"}, HOST, "2.2.2.3")
    assert c.origin == "server"


def test_token_bound_to_origin(settings):
    g = Guard(settings)
    tok, _ = g.open_session({"origin": "https://partner.example"}, HOST, "3.3.3.3")
    g.authenticate({"authorization": "Bearer " + tok, "origin": "https://partner.example"}, HOST)
    with pytest.raises(AuthError) as e:
        g.authenticate({"authorization": "Bearer " + tok, "origin": "https://other.example"}, HOST)
    assert e.value.status == 403
    with pytest.raises(AuthError):
        g.authenticate({}, HOST)


def test_access_code(settings):
    settings.access_code = "letmein"
    g = Guard(settings)
    with pytest.raises(AuthError):
        g.open_session({"origin": HOST}, HOST, "4.4.4.4", access_code="wrong")
    g.open_session({"origin": HOST}, HOST, "4.4.4.5", access_code="letmein")


def test_cors_only_for_allowlist(settings):
    g = Guard(settings)
    assert g.cors_headers("https://partner.example", HOST)["Access-Control-Allow-Origin"] == "https://partner.example"
    assert "Access-Control-Allow-Origin" not in g.cors_headers("https://evil.example", HOST)


def test_rate_limiter():
    r = RateLimiter(3, 60)
    assert [r.allow("k", now=0) for _ in range(4)] == [True, True, True, False]
    assert r.allow("k", now=21)          # refilled one token after 20 s
    assert r.allow("other", now=0)


def test_csp_frame_ancestors(settings):
    csp = page_csp(settings)
    assert "frame-ancestors 'self' https://partner.example" in csp
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp


def test_settings_reject_wildcard_and_weak_secret(monkeypatch):
    monkeypatch.setenv("EPM_CHAT_ALLOWED_ORIGINS", "*")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("EPM_CHAT_ALLOWED_ORIGINS", "https://ok.example, http://localhost:3000/")
    monkeypatch.setenv("EPM_CHAT_SECRET_KEY", "short")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("EPM_CHAT_SECRET_KEY", "s" * 40)
    s = Settings.from_env()
    assert s.allowed_origins == ("https://ok.example", "http://localhost:3000")
    assert not s.llm_enabled
