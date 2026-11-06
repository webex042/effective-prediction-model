"""
Optional LLM provider (server-side only).

The LLM never produces predictions, metrics or importances. It is used for two
narrow jobs, and both are checked:

1. ``classify_intent`` — map an ambiguous message to one of the controller's
   supported intents. Output must be one of a fixed list or it is ignored.
2. ``rephrase`` — turn a grounded fact sheet (the JSON the model actually
   returned) into a short plain-language explanation. Every number in the draft
   must appear in the fact sheet; otherwise the draft is discarded and the
   deterministic explanation is used.

When no provider is configured the chatbot is fully functional without it.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from epm_chatbot.config import Settings

log = logging.getLogger("epm_chat.llm")
_NUMBER = re.compile(r"(?<![A-Za-z_])[-−]?\d+(?:[.,]\d+)*(?:\.\d+)?%?")


class LLMError(RuntimeError):
    pass


# =============================================================================
# Grounding guard
# =============================================================================

def _walk_numbers(obj: Any, out: Set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in _NUMBER.finditer(obj):
            v = _to_float(m.group(0))
            if v is not None:
                out.add(v)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(k, out)
            _walk_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, out)


def _to_float(s: str) -> Optional[float]:
    s = s.replace("−", "-").replace(",", "")
    pct = s.endswith("%")
    try:
        v = float(s.rstrip("%"))
    except ValueError:
        return None
    return v / 100 if pct else v


def ungrounded_numbers(text: str, facts: Any) -> List[str]:
    """Numbers in ``text`` that are not a fact rounded to the precision written in the text.

    "0.93" matches a fact 0.9333 (2 decimals); "97.8%" matches 0.9782; "99%" does not match 0.9782.
    Small integers 0-10 (counts like "3 features") are always allowed.
    """
    allowed: Set[float] = set()
    _walk_numbers(facts, allowed)
    bad = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0).replace("−", "-").replace(",", "")
        is_pct = raw.endswith("%")
        num = raw.rstrip("%")
        try:
            v = float(num)
        except ValueError:
            continue
        decimals = len(num.split(".")[1]) if "." in num else 0
        if not is_pct and decimals == 0 and 0 <= v <= 10:
            continue
        scales = (100.0, 1.0) if is_pct else (1.0,)
        if not any(round(a * k, decimals) == round(v, decimals) for a in allowed for k in scales):
            bad.append(m.group(0))
    return bad


# =============================================================================
# Provider
# =============================================================================

class AnthropicClient:
    """Minimal Messages API client using the standard library (no SDK dependency)."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def _request(self, payload: Dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self.s.llm_base_url + "/v1/messages", data=json.dumps(payload).encode(), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.s.llm_api_key,
                     "anthropic-version": "2023-06-01"})

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        payload = {"model": self.s.llm_model, "max_tokens": max_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]}
        try:
            with urllib.request.urlopen(self._request(payload), timeout=self.s.llm_timeout_s) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise LLMError(f"LLM provider returned HTTP {e.code}") from None
        except Exception as e:
            raise LLMError(f"LLM provider unavailable ({type(e).__name__})") from None
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()

    def stream(self, system: str, user: str, max_tokens: int = 400) -> Iterable[str]:
        payload = {"model": self.s.llm_model, "max_tokens": max_tokens, "system": system, "stream": True,
                   "messages": [{"role": "user", "content": user}]}
        try:
            resp = urllib.request.urlopen(self._request(payload), timeout=self.s.llm_timeout_s)
        except urllib.error.HTTPError as e:
            raise LLMError(f"LLM provider returned HTTP {e.code}") from None
        except Exception as e:
            raise LLMError(f"LLM provider unavailable ({type(e).__name__})") from None
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                    yield ev["delta"].get("text", "")


SYSTEM_EXPLAIN = (
    "You explain machine-learning results from the EPM platform to non-experts. "
    "Use ONLY the facts in the provided JSON. Do not add numbers that are not in it, do not speculate about "
    "causes beyond what the facts say, and keep stated limitations. Write 3-5 short sentences of plain prose, "
    "no headings, no lists."
)

SYSTEM_INTENT = (
    "Classify the user's message for an ML chatbot. Reply with exactly one label from this list and nothing else: "
)


class LLMService:
    def __init__(self, settings: Settings, client: Optional[Any] = None) -> None:
        self.s = settings
        self.client = client or (AnthropicClient(settings) if settings.llm_enabled else None)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def classify_intent(self, message: str, labels: List[str]) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            out = self.client.complete(SYSTEM_INTENT + ", ".join(labels), message[:500], max_tokens=10)
        except LLMError as e:
            log.warning("llm intent failed", extra={"reason": str(e)})
            return None
        label = out.strip().split()[0].strip(".,'\"").lower() if out.strip() else ""
        return label if label in labels else None

    def rephrase(self, facts: Dict[str, Any], fallback: str,
                 on_delta: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Return {"text", "source": "llm"|"deterministic", "rejected": [...]}."""
        if not self.enabled:
            return {"text": fallback, "source": "deterministic"}
        user = "Facts (JSON):\n" + json.dumps(facts, default=str)[:6000] + "\n\nExplain these results."
        try:
            if on_delta is not None:
                parts: List[str] = []
                for d in self.client.stream(SYSTEM_EXPLAIN, user):
                    parts.append(d)
                    on_delta(d)
                draft = "".join(parts).strip()
            else:
                draft = self.client.complete(SYSTEM_EXPLAIN, user)
        except LLMError as e:
            log.warning("llm explain failed", extra={"reason": str(e)})
            return {"text": fallback, "source": "deterministic", "error": str(e)}
        bad = ungrounded_numbers(draft, facts)
        if not draft or bad:
            log.warning("llm draft rejected", extra={"ungrounded_count": len(bad)})
            return {"text": fallback, "source": "deterministic", "rejected": bad}
        return {"text": draft, "source": "llm"}
