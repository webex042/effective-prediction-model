"""
Environment-driven configuration for the standalone EPM chatbot.

Every setting can be overridden with an environment variable (see
``epm_chatbot/.env.example``). Nothing here is read by the main EPM website.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[1]          # epm_chatbot/
PROJECT_ROOT = PACKAGE_ROOT.parent                           # repository root (imports src/)


def load_env_file(path: Optional[Path] = None) -> int:
    """Read KEY=VALUE lines from ``epm_chatbot/.env`` (or EPM_CHAT_ENV_FILE) into os.environ.

    Real environment variables always win. Returns the number of keys loaded.
    """
    p = Path(path or os.environ.get("EPM_CHAT_ENV_FILE") or PACKAGE_ROOT / ".env")
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #", 1)[0].strip().strip("'\"")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = val
            n += 1
    return n


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _origins(raw: str) -> Tuple[str, ...]:
    out = []
    for part in raw.replace(";", ",").split(","):
        o = part.strip().rstrip("/")
        if not o:
            continue
        if o != "*" and not (o.startswith("http://") or o.startswith("https://")):
            raise ValueError(f"EPM_CHAT_ALLOWED_ORIGINS entries must start with http(s)://, got {o!r}")
        if o == "*":
            raise ValueError("EPM_CHAT_ALLOWED_ORIGINS does not accept '*'; list each permitted origin explicitly")
        out.append(o.lower())
    return tuple(dict.fromkeys(out))


@dataclass
class Settings:
    # --- network -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8765
    public_origin: str = ""                  # e.g. https://chat.example.com (used in embed snippets)
    allowed_origins: Tuple[str, ...] = ()    # websites allowed to embed / call the API
    trust_proxy: bool = False                # honour X-Forwarded-For for rate limiting (behind a proxy only)

    # --- security ------------------------------------------------------------
    secret_key: str = ""                     # HMAC key for session tokens (random per process if unset)
    token_ttl_s: int = 1800
    access_code: str = ""                    # optional shared code required to open a session
    api_keys: Tuple[str, ...] = ()           # optional server-to-server keys (no Origin header)
    max_body_bytes: int = 64 * 1024          # JSON requests
    max_upload_bytes: int = 5 * 1024 * 1024  # CSV uploads
    max_message_chars: int = 1000
    rate_per_min: int = 40                   # chat messages per session token per minute
    rate_ip_per_min: int = 120               # all requests per client IP per minute
    train_per_hour: int = 12                 # training jobs per session per hour
    max_sessions: int = 500

    # --- ML / jobs -----------------------------------------------------------
    model_dir: Path = PACKAGE_ROOT / "var" / "models"
    upload_dir: Path = PACKAGE_ROOT / "var" / "uploads"
    upload_ttl_s: int = 3600
    job_workers: int = 1
    max_queued_jobs: int = 4
    train_timeout_s: int = 300
    predict_timeout_s: int = 20
    max_train_rows: int = 20000
    allow_downloads: bool = False            # let the bot fetch uncached public datasets
    allow_neural: bool = True                # LSTM/GRU/Transformer/MLP training
    neural_epochs: int = 25

    # --- optional LLM ----------------------------------------------------------
    llm_provider: str = ""                   # "" (off) | "anthropic"
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_timeout_s: float = 20.0
    llm_base_url: str = "https://api.anthropic.com"

    # --- logging ---------------------------------------------------------------
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_provider and self.llm_api_key and self.llm_model)

    def is_allowed_origin(self, origin: Optional[str]) -> bool:
        if not origin:
            return False
        o = origin.strip().rstrip("/").lower()
        return o in self.allowed_origins or (bool(self.public_origin) and o == self.public_origin.lower())

    def frame_ancestors(self) -> List[str]:
        return ["'self'", *self.allowed_origins]

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file()
        s = cls(
            host=_env("EPM_CHAT_HOST", "127.0.0.1"),
            port=_env_int("EPM_CHAT_PORT", _env_int("PORT", 8765)),
            public_origin=_env("EPM_CHAT_PUBLIC_ORIGIN").rstrip("/"),
            allowed_origins=_origins(_env("EPM_CHAT_ALLOWED_ORIGINS")),
            trust_proxy=_env_bool("EPM_CHAT_TRUST_PROXY", False),
            secret_key=_env("EPM_CHAT_SECRET_KEY"),
            token_ttl_s=_env_int("EPM_CHAT_TOKEN_TTL_S", 1800),
            access_code=_env("EPM_CHAT_ACCESS_CODE"),
            api_keys=tuple(k for k in _env("EPM_CHAT_API_KEYS").split(",") if k.strip()),
            max_body_bytes=_env_int("EPM_CHAT_MAX_BODY_BYTES", 64 * 1024),
            max_upload_bytes=_env_int("EPM_CHAT_MAX_UPLOAD_BYTES", 5 * 1024 * 1024),
            max_message_chars=_env_int("EPM_CHAT_MAX_MESSAGE_CHARS", 1000),
            rate_per_min=_env_int("EPM_CHAT_RATE_PER_MIN", 40),
            rate_ip_per_min=_env_int("EPM_CHAT_RATE_IP_PER_MIN", 120),
            train_per_hour=_env_int("EPM_CHAT_TRAIN_PER_HOUR", 12),
            max_sessions=_env_int("EPM_CHAT_MAX_SESSIONS", 500),
            model_dir=Path(_env("EPM_CHAT_MODEL_DIR") or PACKAGE_ROOT / "var" / "models"),
            upload_dir=Path(_env("EPM_CHAT_UPLOAD_DIR") or PACKAGE_ROOT / "var" / "uploads"),
            upload_ttl_s=_env_int("EPM_CHAT_UPLOAD_TTL_S", 3600),
            job_workers=_env_int("EPM_CHAT_JOB_WORKERS", 1),
            max_queued_jobs=_env_int("EPM_CHAT_MAX_QUEUED_JOBS", 4),
            train_timeout_s=_env_int("EPM_CHAT_TRAIN_TIMEOUT_S", 300),
            predict_timeout_s=_env_int("EPM_CHAT_PREDICT_TIMEOUT_S", 20),
            max_train_rows=_env_int("EPM_CHAT_MAX_TRAIN_ROWS", 20000),
            allow_downloads=_env_bool("EPM_CHAT_ALLOW_DOWNLOADS", False),
            allow_neural=_env_bool("EPM_CHAT_ALLOW_NEURAL", True),
            neural_epochs=_env_int("EPM_CHAT_NEURAL_EPOCHS", 25),
            llm_provider=_env("EPM_CHAT_LLM_PROVIDER").lower(),
            llm_api_key=_env("EPM_CHAT_LLM_API_KEY") or _env("ANTHROPIC_API_KEY"),
            llm_model=_env("EPM_CHAT_LLM_MODEL"),
            llm_timeout_s=_env_float("EPM_CHAT_LLM_TIMEOUT_S", 20.0),
            llm_base_url=_env("EPM_CHAT_LLM_BASE_URL", "https://api.anthropic.com").rstrip("/"),
            log_level=_env("EPM_CHAT_LOG_LEVEL", "INFO").upper(),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if not self.secret_key:
            # Tokens stay valid only for this process; fine for a single local instance.
            self.secret_key = secrets.token_hex(32)
        elif len(self.secret_key) < 32:
            raise ValueError("EPM_CHAT_SECRET_KEY must be at least 32 characters")
        if self.llm_provider and self.llm_provider != "anthropic":
            raise ValueError("EPM_CHAT_LLM_PROVIDER supports only 'anthropic' (or leave it empty)")
        if self.public_origin and not self.public_origin.startswith(("http://", "https://")):
            raise ValueError("EPM_CHAT_PUBLIC_ORIGIN must start with http(s)://")
        for name in ("max_body_bytes", "max_upload_bytes", "rate_per_min", "rate_ip_per_min", "job_workers",
                     "max_queued_jobs", "train_timeout_s", "predict_timeout_s", "max_train_rows", "token_ttl_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
