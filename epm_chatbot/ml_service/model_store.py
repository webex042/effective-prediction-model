"""
Model artifact store.

Each trained model is saved as ``<id>.joblib`` plus a ``<id>.json`` metadata
card (schema, held-out metrics, importance...). The card stores the artifact's
SHA-256, which is verified before every load, so a corrupted or swapped file is
refused instead of being unpickled. Only files written by this service are ever
loaded; user uploads are never unpickled.

Models trained on a registry dataset by the bootstrap command are *shared*
(``owner = None``). Models trained during a chat on an uploaded file are
*private* to that session and deleted with it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ID = re.compile(r"^m_[0-9a-f]{12}$")


class ModelStoreError(RuntimeError):
    pass


class ModelStore:
    def __init__(self, root: Path, cache_size: int = 8) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write
    def save(self, artifact: Path, meta: Dict[str, Any], owner: Optional[str] = None) -> Dict[str, Any]:
        """Move a finished artifact file into the store and write its card."""
        mid = "m_" + secrets.token_hex(6)
        data = Path(artifact).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        dst = self.root / f"{mid}.joblib"
        tmp = self.root / f"{mid}.joblib.part"
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        try:
            Path(artifact).unlink()
        except OSError:
            pass
        card = dict(meta)
        card.update({"id": mid, "owner": owner, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "artifact_sha256": sha, "artifact_kb": round(len(data) / 1024, 2)})
        self._write_card(mid, card)
        return card

    def _write_card(self, mid: str, card: Dict[str, Any]) -> None:
        tmp = self.root / f"{mid}.json.part"
        tmp.write_text(json.dumps(card, indent=1, default=_json_default), encoding="utf-8")
        os.replace(tmp, self.root / f"{mid}.json")

    # ------------------------------------------------------------------ read
    def _card(self, mid: str) -> Dict[str, Any]:
        if not _ID.match(mid or ""):
            raise ModelStoreError("Unknown model id.")
        p = self.root / f"{mid}.json"
        if not p.exists():
            raise ModelStoreError("That model no longer exists (it may have expired with its session).")
        return json.loads(p.read_text(encoding="utf-8"))

    def card(self, mid: str, owner: Optional[str]) -> Dict[str, Any]:
        c = self._card(mid)
        if c.get("owner") not in (None, owner):
            raise ModelStoreError("That model belongs to another session.")
        return c

    def list(self, owner: Optional[str] = None, dataset_id: Optional[str] = None,
             task: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("m_*.json")):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if c.get("owner") not in (None, owner):
                continue
            if dataset_id and c.get("dataset_id") != dataset_id:
                continue
            if task and c.get("task") != task:
                continue
            out.append(c)
        out.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return out

    def load(self, mid: str, owner: Optional[str]) -> Any:
        c = self.card(mid, owner)
        key = f"{mid}:{c['artifact_sha256']}"
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        path = self.root / f"{mid}.joblib"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise ModelStoreError("The model file is missing; retrain the model.") from None
        if hashlib.sha256(data).hexdigest() != c["artifact_sha256"]:
            raise ModelStoreError("The model file failed its integrity check and was not loaded; retrain the model.")
        import io

        import joblib

        try:
            model = joblib.load(io.BytesIO(data))
        except Exception as exc:
            raise ModelStoreError(f"The model could not be loaded ({type(exc).__name__}); retrain it.") from None
        with self._lock:
            self._cache[key] = model
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return model

    # ------------------------------------------------------------------ delete
    def delete_owner(self, owner: str) -> int:
        n = 0
        for c in self.list(owner=owner):
            if c.get("owner") == owner:
                self._delete(c["id"])
                n += 1
        return n

    def _delete(self, mid: str) -> None:
        for suffix in (".json", ".joblib"):
            try:
                (self.root / f"{mid}{suffix}").unlink()
            except FileNotFoundError:
                pass
        with self._lock:
            for k in [k for k in self._cache if k.startswith(mid + ":")]:
                del self._cache[k]

    def sweep_private(self, max_age_s: int, now: Optional[float] = None) -> int:
        """Remove session-private models older than ``max_age_s``."""
        now = now or time.time()
        n = 0
        for p in self.root.glob("m_*.json"):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if c.get("owner") and now - p.stat().st_mtime > max_age_s:
                self._delete(c["id"])
                n += 1
        return n

    def wipe(self) -> None:  # tests only
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(o)
