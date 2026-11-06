"""
Dataset access for the chatbot.

* Registry datasets come from EPM's ``src.data`` loaders (download only when
  ``EPM_CHAT_ALLOW_DOWNLOADS`` is on; large datasets are never used).
* Uploaded CSVs are validated with EPM's ``validate_uploaded_csv`` and stored in
  the chatbot's *own* per-session temporary folder (``EPM_CHAT_UPLOAD_DIR``),
  never in the main project's ``data/`` tree, and are deleted after
  ``EPM_CHAT_UPLOAD_TTL_S`` or when the session is cleared.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.loaders import DatasetLoadError, LoadedDataset, dataset_status, load_dataset
from src.data.registry import (
    AD_METRICS, CAT_PREP, CLF_METRICS, CLU_METRICS, DATASETS, FC_METRICS, REG_METRICS, TAB_PREP, TS_PREP,
    DatasetSpec, get_dataset_spec,
)
from src.data.validation import UploadValidationError, validate_uploaded_csv

PREDICTABLE_TASKS = ("classification", "regression", "forecasting")
_TASK_METRICS = {"regression": REG_METRICS, "classification": CLF_METRICS + ("pr_auc",), "clustering": CLU_METRICS,
                 "forecasting": FC_METRICS, "anomaly_detection": AD_METRICS}


class DataAccessError(ValueError):
    """User-facing dataset problem (not cached, too large, invalid upload...)."""


def session_key(sid: str) -> str:
    """Directory-safe, non-reversible key for a session id."""
    return hashlib.sha256(("epm-chat:" + sid).encode()).hexdigest()[:20]


# =============================================================================
# Registry datasets
# =============================================================================

def registry_datasets(task: Optional[str] = None) -> List[DatasetSpec]:
    out = []
    for spec in DATASETS.values():
        if spec.size_class == "large":
            continue
        if task and not spec.supports(task):
            continue
        if not any(spec.supports(t) for t in PREDICTABLE_TASKS):
            continue
        out.append(spec)
    return sorted(out, key=lambda s: s.n_samples or 0)


def availability(spec: DatasetSpec, allow_downloads: bool) -> str:
    st = dataset_status(spec.id)["state"]
    if st in ("available", "cached"):
        return "ready"
    return "downloadable" if allow_downloads else "not_cached"


def load_registry(dataset_id: str, allow_downloads: bool) -> LoadedDataset:
    spec = get_dataset_spec(dataset_id)
    if spec.size_class == "large":
        raise DataAccessError(f"{spec.name} is a large dataset and is disabled in the chatbot to protect resources.")
    ready = availability(spec, allow_downloads) == "ready"
    try:
        return load_dataset(dataset_id, allow_download=allow_downloads)
    except DatasetLoadError as exc:
        missing = re.search(r"No module named '([\w.]+)'", str(exc))
        if missing:
            raise DataAccessError(f"{spec.name} needs the optional package '{missing.group(1)}', which is not "
                                  "installed on this server.") from None
        if not allow_downloads and not ready:
            raise DataAccessError(
                f"{spec.name} is not cached on this server and dataset downloads are disabled "
                "(EPM_CHAT_ALLOW_DOWNLOADS=false). Try a bundled dataset such as Iris, Wine, Breast Cancer, "
                "Diabetes, Digits or the EPM synthetic datasets.") from None
        raise DataAccessError(f"Could not load {spec.name}: {exc}") from None


# =============================================================================
# Session uploads
# =============================================================================

_ID = re.compile(r"^up_[0-9a-f]{12}$")


class UploadStore:
    def __init__(self, root: Path, ttl_s: int, max_bytes: int) -> None:
        self.root = Path(root)
        self.ttl_s = ttl_s
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def _dir(self, sid: str) -> Path:
        return self.root / session_key(sid)

    # -- write ------------------------------------------------------------------
    def save(self, sid: str, raw: bytes, filename: str) -> Dict[str, Any]:
        """Validate and store a CSV. The purpose (train vs predict) is decided later by the user."""
        if len(raw) > self.max_bytes:
            raise DataAccessError(f"File is {len(raw) / 1e6:.1f} MB; the limit is {self.max_bytes / 1e6:.0f} MB.")
        if not str(filename).lower().endswith((".csv", ".txt")):
            raise DataAccessError("Only .csv files are accepted.")
        try:
            # task='clustering' parses + sanitises without requiring a target column
            df, report = validate_uploaded_csv(raw, "clustering", None)
        except UploadValidationError as exc:
            raise DataAccessError(str(exc)) from None
        if df.shape[1] > 200:
            raise DataAccessError("Too many columns (limit 200 in the chatbot).")
        if len(df) > 50000:
            raise DataAccessError("Too many rows (limit 50,000 in the chatbot).")
        upload_id = "up_" + hashlib.sha256(raw + sid.encode()).hexdigest()[:12]
        d = self._dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{upload_id}.csv.part"
        df.to_csv(tmp, index=False)
        os.replace(tmp, d / f"{upload_id}.csv")
        meta = {"id": upload_id, "name": Path(str(filename)).name[:80], "created": time.time(),
                "rows": int(len(df)), "columns": profile_columns(df), "report": report}
        (d / f"{upload_id}.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    # -- read -------------------------------------------------------------------
    def meta(self, sid: str, upload_id: str) -> Dict[str, Any]:
        if not _ID.match(upload_id or ""):
            raise DataAccessError("Unknown upload.")
        p = self._dir(sid) / f"{upload_id}.json"
        if not p.exists():
            raise DataAccessError("That upload has expired or was cleared. Please upload the file again.")
        return json.loads(p.read_text(encoding="utf-8"))

    def frame(self, sid: str, upload_id: str) -> pd.DataFrame:
        self.meta(sid, upload_id)
        return pd.read_csv(self._dir(sid) / f"{upload_id}.csv")

    def as_dataset(self, sid: str, upload_id: str, task: str, target: str,
                   time_column: Optional[str] = None) -> LoadedDataset:
        """Build an EPM LoadedDataset from an upload once the user chose task + target."""
        from src.data.loaders import _from_frame, _series_dataset  # EPM's own frame adapters

        m = self.meta(sid, upload_id)
        df = self.frame(sid, upload_id)
        if target not in df.columns:
            raise DataAccessError(f"Column '{target}' is not in the file. Columns: {', '.join(map(str, df.columns[:25]))}")
        if task == "forecasting":
            if not time_column:
                raise DataAccessError("Forecasting needs a time column (e.g. 'time column timestamp').")
        raw = df.to_csv(index=False).encode()
        try:
            df, report = validate_uploaded_csv(raw, task, target, time_column)
        except UploadValidationError as exc:
            raise DataAccessError(str(exc)) from None
        feat_cols = [c for c in df.columns if c not in (target, time_column)]
        if not feat_cols:
            raise DataAccessError("The file needs at least one feature column besides the target.")
        num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in feat_cols if c not in num]
        if task == "classification":
            ttype = "binary" if df[target].nunique() == 2 else "multiclass"
            if ttype == "binary" and set(pd.unique(df[target].dropna())) - {0, 1}:
                ttype = "multiclass"  # EPM's binary schema requires 0/1; non-0/1 labels are encoded as classes
        elif task == "forecasting":
            ttype = "series"
            num = [c for c in df.columns if c != time_column and pd.api.types.is_numeric_dtype(df[c])]
            cat = []
        else:
            ttype = "continuous"
        n = len(df)
        spec = DatasetSpec(
            id=f"upload:{upload_id}", name=f"Your file ({m['name']})",
            description=f"User-uploaded CSV ({n} rows) used for {task}.", tasks=(task,), source="User upload",
            source_url="local file", download_url=None, download_method="upload",
            license="Provided by the user", usage_notes="Validated on upload; column names were sanitised.",
            numeric_features=tuple(num), categorical_features=tuple(cat), target=target, target_type=ttype,
            n_samples=n, n_features=len(num) + len(cat), missing_values=f"{report['missing_cells']} missing cells",
            approx_memory_mb=round(n * max(1, len(num) + len(cat)) * 8 / 1e6, 2),
            size_class="small" if n <= 5000 else "medium",
            preprocessing=TS_PREP if task == "forecasting" else (CAT_PREP if cat else TAB_PREP),
            metrics={task: _TASK_METRICS[task]},
            split_strategy="chronological" if task == "forecasting" else ("stratified" if task == "classification" else "random"),
            time_column=time_column, seasonal_period=24 if task == "forecasting" else None,
            default_window=24 if task == "forecasting" else None, default_horizon=1 if task == "forecasting" else None,
        )
        if task == "forecasting":
            df[time_column] = pd.to_datetime(df[time_column])
            return _series_dataset(spec, df.set_index(time_column)[num])
        return _from_frame(spec, df, target)

    # -- cleanup ----------------------------------------------------------------
    def clear(self, sid: str) -> int:
        d = self._dir(sid)
        if not d.exists():
            return 0
        n = len(list(d.glob("*.csv")))
        shutil.rmtree(d, ignore_errors=True)
        return n

    def sweep(self, now: Optional[float] = None) -> int:
        """Delete uploads older than the TTL. Returns the number of files removed."""
        now = now or time.time()
        removed = 0
        if not self.root.exists():
            return 0
        with self._lock:
            for meta in self.root.glob("*/*.json"):
                try:
                    created = json.loads(meta.read_text(encoding="utf-8")).get("created", 0)
                except Exception:
                    created = 0
                if now - created > self.ttl_s:
                    for p in (meta, meta.with_suffix(".csv")):
                        try:
                            p.unlink()
                            removed += 1
                        except FileNotFoundError:
                            pass
            for d in self.root.iterdir():
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        return removed


def profile_columns(df: pd.DataFrame, max_cols: int = 60) -> List[Dict[str, Any]]:
    """Per-column type, missing share and a short summary (no raw rows)."""
    out = []
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        entry: Dict[str, Any] = {"name": str(c), "missing": int(s.isna().sum()),
                                 "missing_pct": round(float(s.isna().mean() * 100), 1), "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            entry["type"] = "numeric"
            if s.notna().any():
                entry.update({"min": _r(s.min()), "max": _r(s.max()), "mean": _r(s.mean())})
        else:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed") if s.notna().any() else s
            if s.notna().any() and parsed.notna().mean() > 0.95:
                entry["type"] = "datetime"
            else:
                entry["type"] = "categorical"
                entry["top"] = [str(v) for v in s.value_counts().head(3).index]
        out.append(entry)
    return out


def _r(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 6) if np.isfinite(f) else None
