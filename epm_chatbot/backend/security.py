"""
Security primitives: origin allowlist, signed short-lived session tokens,
rate limiting and response security headers.

Design notes
------------
* CORS only tells *browsers* which pages may read responses. It is not
  authentication, so every API route (except health/config) also requires a
  signed session token, and token issuance is gated on the Origin allowlist
  (plus an optional access code).
* Tokens are HMAC-SHA256 signed ``{sid, origin, exp}`` payloads. A token minted
  for one origin is rejected when presented from another origin.
* Rate limits are token buckets keyed by client IP and by session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from epm_chatbot.config import Settings


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


@dataclass
class TokenClaims:
    sid: str
    origin: str
    exp: int


class TokenSigner:
    def __init__(self, secret: str, ttl_s: int) -> None:
        self._key = hashlib.sha256(("epm-chat-token:" + secret).encode()).digest()
        self.ttl_s = ttl_s

    def issue(self, origin: str, sid: Optional[str] = None, now: Optional[float] = None) -> Tuple[str, TokenClaims]:
        claims = TokenClaims(sid=sid or secrets.token_hex(12), origin=origin, exp=int((now or time.time()) + self.ttl_s))
        payload = _b64(json.dumps(claims.__dict__, separators=(",", ":")).encode())
        sig = _b64(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())
        return f"{payload}.{sig}", claims

    def verify(self, token: str, now: Optional[float] = None) -> TokenClaims:
        try:
            payload, sig = token.split(".", 1)
        except ValueError:
            raise AuthError("Malformed session token.") from None
        expected = _b64(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise AuthError("Invalid session token.")
        try:
            data = json.loads(_unb64(payload))
            claims = TokenClaims(sid=str(data["sid"]), origin=str(data["origin"]), exp=int(data["exp"]))
        except Exception:
            raise AuthError("Malformed session token.") from None
        if claims.exp < (now or time.time()):
            raise AuthError("Session expired. Reload the chat to start a new session.")
        return claims


class RateLimiter:
    """Token bucket per key. ``capacity`` events per ``period_s`` with steady refill."""

    def __init__(self, capacity: int, period_s: float, max_keys: int = 20000) -> None:
        self.capacity = float(capacity)
        self.rate = capacity / period_s
        self._buckets: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_keys = max_keys

    def allow(self, key: str, cost: float = 1.0, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            ok = tokens >= cost
            if ok:
                tokens -= cost
            self._buckets[key] = (tokens, now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
            return ok

    def retry_after(self, key: str, cost: float = 1.0) -> int:
        with self._lock:
            tokens, _ = self._buckets.get(key, (self.capacity, 0))
        return max(1, int((cost - tokens) / self.rate) + 1)


class Guard:
    """Origin / token / rate checks used by the HTTP layer."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.signer = TokenSigner(settings.secret_key, settings.token_ttl_s)
        self.ip_limiter = RateLimiter(settings.rate_ip_per_min, 60)
        self.msg_limiter = RateLimiter(settings.rate_per_min, 60)
        self.train_limiter = RateLimiter(settings.train_per_hour, 3600)
        self.session_limiter = RateLimiter(10, 60)  # new sessions per IP per minute

    # -- origins ----------------------------------------------------------------
    def request_origin(self, headers: Dict[str, str], host_origin: str) -> Optional[str]:
        """The browser-reported origin, or None for non-browser clients."""
        o = (headers.get("origin") or "").strip().rstrip("/").lower()
        if o and o != "null":
            return o
        return None

    def origin_permitted(self, origin: Optional[str], host_origin: str) -> bool:
        if origin is None:
            return False
        return origin == host_origin.lower() or self.s.is_allowed_origin(origin)

    def cors_headers(self, origin: Optional[str], host_origin: str) -> Dict[str, str]:
        if origin and origin != host_origin.lower() and self.s.is_allowed_origin(origin):
            return {"Access-Control-Allow-Origin": origin, "Vary": "Origin",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Access-Code",
                    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                    "Access-Control-Max-Age": "600"}
        return {"Vary": "Origin"}

    # -- sessions -----------------------------------------------------------------
    def open_session(self, headers: Dict[str, str], host_origin: str, client_ip: str,
                     access_code: str = "") -> Tuple[str, TokenClaims]:
        origin = self.request_origin(headers, host_origin)
        if origin is None:
            key = (headers.get("x-api-key") or "").strip()
            if not key or not any(hmac.compare_digest(key, k) for k in self.s.api_keys):
                raise AuthError("Sessions can only be opened from a permitted website (or with a server API key).", 403)
            origin = "server"
        elif not self.origin_permitted(origin, host_origin):
            raise AuthError("This website is not permitted to use EPM Chat.", 403)
        if self.s.access_code and not hmac.compare_digest(access_code or "", self.s.access_code):
            raise AuthError("A valid access code is required.", 401)
        if not self.session_limiter.allow(f"sess:{client_ip}"):
            raise AuthError("Too many new sessions; try again in a minute.", 429)
        return self.signer.issue(origin)

    def authenticate(self, headers: Dict[str, str], host_origin: str, query_token: str = "") -> TokenClaims:
        auth = headers.get("authorization") or ""
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else query_token
        if not token:
            raise AuthError("Missing session token.")
        claims = self.signer.verify(token)
        origin = self.request_origin(headers, host_origin)
        if claims.origin == "server":
            if origin is not None:
                raise AuthError("Server tokens cannot be used from a browser.", 403)
        elif origin is not None and origin != claims.origin:
            raise AuthError("Session token was issued for a different website.", 403)
        return claims


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cross-Origin-Resource-Policy": "cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def page_csp(settings: Settings) -> str:
    """CSP for the chat page: no inline script, no third-party code, framing only by allowlisted sites."""
    return "; ".join([
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "font-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors " + " ".join(settings.frame_ancestors()),
    ])
