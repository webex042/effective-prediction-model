"""
Structured JSON logging.

Log records carry request metadata only (route, status, latency, intent, job
state). Message text, uploaded data, tokens and API keys are never logged.
"""

from __future__ import annotations

import json
import logging
import sys
import time

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {"message", "asctime"}
_REDACT = {"text", "message_text", "token", "authorization", "api_key", "csv", "body", "values"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
               "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k, v in record.__dict__.items():
            if k in _RESERVED or k.startswith("_"):
                continue
            out[k] = "[redacted]" if k in _REDACT else v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)[-2000:]
        return json.dumps(out, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger("epm_chat")
    if any(getattr(h, "_epm_chat", False) for h in root.handlers):
        root.setLevel(level)
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    h._epm_chat = True  # type: ignore[attr-defined]
    root.addHandler(h)
    root.setLevel(level)
    root.propagate = False
