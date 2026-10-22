"""
EPM Local Experiment API
========================

Served by ``python run.py --serve`` next to the static dashboard. Unlike the
stdlib-only Vercel function in ``api/index.py`` (read-only artifacts), this API
can load datasets and train models, so it only runs locally.

* Training and downloads run in a background worker pool (``EPM_MAX_JOBS``,
  default 1) so requests never block the UI; the UI polls ``/api/jobs/<id>``.
* POST bodies must be JSON (forces a CORS pre-flight from other origins) and are
  size-capped; ids are validated before touching the filesystem.
* Unknown routes fall through to the existing ``api/index.py`` router, so all
  earlier endpoints keep working unchanged.
"""

from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.benchmarking.catalog import (
    ALGORITHMS, check_compatibility, get_algorithm, list_algorithms, recommend_algorithms, recommend_datasets,
)
from src.benchmarking.runner import BenchmarkConfig, BenchmarkError, load_history, load_run, run_benchmark
from src.data.loaders import DatasetLoadError, dataset_status, download_dataset, load_dataset
from src.data.registry import TASKS, get_dataset_spec, list_datasets
from src.data.uploads import list_uploads, save_upload, upload_spec
from src.data.validation import UploadValidationError

MAX_BODY_BYTES = 30 * 1024 * 1024
MAX_QUEUED_JOBS = 5

Response = Tuple[int, Any]


# =============================================================================
# Background jobs
# =============================================================================

class JobCancelled(BaseException):  # not Exception: per-model handlers must not swallow it
    """Raised inside a job's progress callback once cancellation was requested."""


class JobManager:
    def __init__(self, max_workers: int = 1, keep: int = 50) -> None:
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="epm-job")
        self.jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.lock = threading.Lock()
        self.keep = keep

    def active(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["status"] in ("queued", "running"))

    def latest_active(self, session_id: Optional[str] = None) -> Optional[str]:
        with self.lock:
            for jid, j in reversed(self.jobs.items()):
                if j["status"] in ("queued", "running") and (
                        session_id is None or (j.get("meta") or {}).get("session") == session_id):
                    return jid
        return None

    def submit(self, kind: str, fn: Callable[[Callable[[float, str], None]], Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        if self.active() >= MAX_QUEUED_JOBS:
            raise BenchmarkError(f"Too many jobs queued (limit {MAX_QUEUED_JOBS}); wait for one to finish.")
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "kind": kind, "status": "queued", "progress": 0.0, "message": "Queued",
               "created": time.time(), "meta": meta, "result": None, "error": None, "cancel_requested": False}
        with self.lock:
            self.jobs[jid] = job
            while len(self.jobs) > self.keep:
                oldest = next(iter(self.jobs))
                if self.jobs[oldest]["status"] in ("queued", "running"):
                    break
                self.jobs.popitem(last=False)

        def progress(frac: float, msg: str) -> None:
            with self.lock:
                if job["cancel_requested"]:
                    raise JobCancelled()
                job["progress"] = round(float(frac), 3)
                job["message"] = msg

        def work() -> None:
            with self.lock:
                if job["cancel_requested"]:
                    job.update(status="cancelled", message="Cancelled before start", finished=time.time())
                    return
                job["status"] = "running"
                job["started"] = time.time()
            try:
                out = fn(progress)
                with self.lock:
                    job.update(status="done", progress=1.0, message="Done", result=out)
            except JobCancelled:
                with self.lock:
                    job.update(status="cancelled", message="Cancelled (stopped at the next model boundary)")
            except (BenchmarkError, DatasetLoadError, KeyError, ValueError) as exc:
                with self.lock:
                    job.update(status="error", error=str(exc).strip("'\""), message="Failed")
            except Exception as exc:  # pragma: no cover - unexpected
                with self.lock:
                    job.update(status="error", error=f"{type(exc).__name__}: {exc}", message="Failed",
                               traceback=traceback.format_exc(limit=4))
            finally:
                with self.lock:
                    job["finished"] = time.time()

        self.pool.submit(work)
        return self.public(job)

    def cancel(self, jid: str) -> Optional[Dict[str, Any]]:
        """Request cancellation. Queued jobs never start; running jobs stop at the next
        progress checkpoint (between models / training phases), not mid-fit."""
        with self.lock:
            j = self.jobs.get(jid)
            if j is None:
                return None
            if j["status"] in ("queued", "running"):
                j["cancel_requested"] = True
                if j["status"] == "running":
                    j["message"] = "Cancelling…"
            return self.public(j)

    def get(self, jid: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            j = self.jobs.get(jid)
            return None if j is None else self.public(j, full=True)

    def list(self) -> list:
        with self.lock:
            return [self.public(j) for j in reversed(self.jobs.values())]

    @staticmethod
    def public(job: Dict[str, Any], full: bool = False) -> Dict[str, Any]:
        out = {k: job.get(k) for k in ("id", "kind", "status", "progress", "message", "error", "meta", "created",
                                        "started", "finished")}
        if full:
            out["result"] = job.get("result")
        return out


JOBS = JobManager(max_workers=int(os.environ.get("EPM_MAX_JOBS", "1")))

_CHAT = None
_CHAT_LOCK = threading.Lock()


def chat_engine():
    """The chatbot shares the job queue with the Experiment Lab (same limits, same cancellation)."""
    global _CHAT
    with _CHAT_LOCK:
        if _CHAT is None:
            from src.chatbot.engine import ChatEngine

            _CHAT = ChatEngine(submit=JOBS.submit, cancel=JOBS.cancel,
                               active_job=lambda sid: JOBS.latest_active(sid))
        return _CHAT


def _chat_payload(session_id: str, resp) -> Dict[str, Any]:
    d = resp.to_dict()
    d["session_id"] = session_id
    d["reply"] = d["text"]
    return _clean(d)


# =============================================================================
# JSON helpers
# =============================================================================

def _clean(o: Any) -> Any:
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else f
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def _qbool(q: Dict[str, str], key: str) -> bool:
    return str(q.get(key, "")).lower() in ("1", "true", "yes", "on")


def _qint(q: Dict[str, str], key: str) -> Optional[int]:
    try:
        return int(q[key]) if q.get(key) not in (None, "") else None
    except ValueError:
        raise BenchmarkError(f"'{key}' must be an integer") from None


# =============================================================================
# Dataset views
# =============================================================================

def _dataset_entry(spec) -> Dict[str, Any]:
    d = spec.to_dict()
    d["status"] = dataset_status(spec.id) if not spec.id.startswith("upload_") else {"state": "available"}
    d["algorithms_by_task"] = {}
    for t in spec.tasks:
        ids = [a.id for a in list_algorithms(t)
               if check_compatibility(a, spec, t, allow_large=True)[0]]
        d["algorithms_by_task"][t] = ids
    return d


def _column_stats(df: pd.DataFrame, max_cols: int = 80) -> list:
    out = []
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        e: Dict[str, Any] = {"name": str(c), "dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean() * 100), 2),
                             "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            e.update({"kind": "numeric", "mean": desc.get("mean"), "std": desc.get("std"), "min": desc.get("min"),
                      "max": desc.get("max"), "median": float(s.median()) if s.notna().any() else None})
        else:
            e.update({"kind": "categorical", "top": {str(k): int(v) for k, v in s.value_counts().head(5).items()}})
        out.append(e)
    return out


def dataset_preview(dataset_id: str, rows: int = 15) -> Response:
    spec = get_dataset_spec(dataset_id)
    st = dataset_status(dataset_id) if not dataset_id.startswith("upload_") else {"state": "available"}
    if st["state"] not in ("available", "cached"):
        return 409, {"error": "Dataset is not downloaded yet.", "state": st["state"],
                     "last_error": st.get("last_error"), "action": f"POST /api/datasets/{dataset_id}/download"}
    if spec.size_class == "large" and not st.get("in_memory"):
        pass  # cached large datasets may still be previewed (loaded from the local cache)
    ds = load_dataset(dataset_id, allow_download=False, allow_large=True)
    rows = max(1, min(rows, 50))
    head = ds.features.head(rows).copy()
    if ds.target is not None and spec.target and spec.target not in head.columns:
        head[spec.target] = ds.target.head(rows).values
    if ds.time_index is not None:
        head.insert(0, spec.time_column or "time", [str(t) for t in list(ds.time_index[:rows])])
    target = None
    if ds.target is not None:
        t = ds.target
        if spec.target_type in ("continuous", "series"):
            target = {"kind": "numeric", "role": ds.label_role, "describe": {k: float(v) for k, v in t.describe().items()}}
        else:
            target = {"kind": "categorical", "role": ds.label_role,
                      "counts": {str(k): int(v) for k, v in t.value_counts().head(20).items()}}
    return 200, _clean({
        "dataset_id": dataset_id, "n_rows": ds.n_samples, "n_columns": int(ds.features.shape[1]),
        "memory_mb": round(ds.memory_mb, 3), "duplicate_rows": int(ds.features.duplicated().sum()),
        "missing_cells": int(ds.features.isna().sum().sum()),
        "columns": list(map(str, head.columns)), "rows": head.astype(object).where(head.notna(), None).values.tolist(),
        "column_stats": _column_stats(ds.features), "target": target, "info": ds.info,
    })


# =============================================================================
# Router
# =============================================================================

def route(method: str, path: str, query: Dict[str, str], body: Dict[str, Any]) -> Optional[Response]:
    """Return (status, payload) for local-API routes, or None to fall through."""
    parts = [p for p in path.split("?")[0].strip("/").split("/") if p]
    if not parts or parts[0] != "api":
        return None
    p = parts[1:]
    try:
        if method == "GET":
            if p == ["capabilities"]:
                return 200, {"mode": "local", "training": True, "uploads": True, "tasks": list(TASKS),
                             "max_jobs": JOBS.pool._max_workers, "active_jobs": JOBS.active()}
            if p == ["datasets"]:
                task = query.get("task") or None
                specs = list_datasets(task)
                ups = [upload_spec(u["id"]) for u in list_uploads()]
                if task:
                    ups = [s for s in ups if s.supports(task)]
                return 200, _clean({"datasets": [_dataset_entry(s) for s in specs + ups]})
            if len(p) == 2 and p[0] == "datasets":
                spec = get_dataset_spec(p[1])
                return 200, _clean(_dataset_entry(spec))
            if len(p) == 3 and p[0] == "datasets" and p[2] == "preview":
                return dataset_preview(p[1], _qint(query, "rows") or 15)
            if p == ["algorithms"]:
                task = query.get("task") or None
                return 200, {"algorithms": [a.to_dict() for a in list_algorithms(task)]}
            if p == ["compatibility"]:
                spec = get_dataset_spec(query.get("dataset", ""))
                task = query.get("task") or spec.tasks[0]
                rec = recommend_algorithms(spec.id, task, n_train=_qint(query, "max_train_rows"),
                                           window=_qint(query, "window"), allow_large=_qbool(query, "allow_large"))
                return 200, rec
            if p == ["recommend"]:
                return 200, recommend_datasets(query.get("algorithm", ""), query.get("budget", "quick"),
                                               allow_large=_qbool(query, "allow_large"))
            if p == ["jobs"]:
                return 200, {"jobs": JOBS.list()}
            if len(p) == 2 and p[0] == "jobs":
                j = JOBS.get(p[1])
                return (200, _clean(j)) if j else (404, {"error": "job not found"})
            if p == ["benchmarks", "history"]:
                return 200, {"runs": load_history()}
            if p == ["chat", "sessions"]:
                sid = query.get("session_id", "")
                if not sid:
                    return 400, {"error": "session_id is required"}
                return 200, _clean(chat_engine().session_info(sid))
            if p == ["chat", "job"]:
                sid, jid = query.get("session_id", ""), query.get("job_id", "")
                j = JOBS.get(jid)
                if not sid or j is None:
                    return 404, {"error": "job not found"}
                return 200, _chat_payload(sid, chat_engine().job_update(sid, j))
            if p == ["chat", "samples"]:
                return _chat_samples(query.get("run_id", ""))
            if len(p) == 2 and p[0] == "benchmarks" and p[1] != "history":
                r = load_run(p[1])
                return (200, r) if r else (404, {"error": "run not found"})
            return None

        if method == "POST":
            if p == ["chat"]:
                message = body.get("message")
                if not isinstance(message, str) or not message.strip():
                    return 400, {"error": "message is required"}
                if len(message) > 1000:
                    return 400, {"error": "message is too long (max 1000 characters)"}
                eng = chat_engine()
                sess = eng.session(str(body.get("session_id") or ""))
                return 200, _chat_payload(sess.id, eng.handle(sess.id, message))
            if p == ["benchmarks", "run"]:
                cfg = BenchmarkConfig.from_dict(body.get("config", body))
                spec = get_dataset_spec(cfg.dataset_id)
                task = cfg.task or spec.tasks[0]
                if not spec.supports(task):
                    raise BenchmarkError(f"'{spec.id}' does not support task '{task}'.")
                if cfg.epochs is not None and not 1 <= cfg.epochs <= 300:
                    raise BenchmarkError("epochs must be between 1 and 300")
                if cfg.cv_folds and not 2 <= cfg.cv_folds <= 10:
                    raise BenchmarkError("cv_folds must be 0 or between 2 and 10")
                for aid in cfg.algorithms or []:
                    get_algorithm(aid)
                job = JOBS.submit("benchmark", lambda prog: _clean(run_benchmark(cfg, prog)),
                                  {"dataset_id": cfg.dataset_id, "task": task, "algorithms": cfg.algorithms})
                return 202, job
            if len(p) == 3 and p[0] == "datasets" and p[2] == "download":
                spec = get_dataset_spec(p[1])
                allow_large = bool(body.get("allow_large", False))
                job = JOBS.submit("download", lambda prog: (prog(0.1, f"Downloading {spec.name}"),
                                                            _clean(download_dataset(spec.id, allow_large)))[1],
                                  {"dataset_id": spec.id})
                return 202, job
            if len(p) == 3 and p[0] == "jobs" and p[2] == "cancel":
                j = JOBS.cancel(p[1])
                return (200, j) if j else (404, {"error": "job not found"})
            if p == ["datasets", "upload"]:
                content = body.get("content_b64")
                if not content:
                    raise UploadValidationError("content_b64 is required")
                try:
                    raw = base64.b64decode(content, validate=True)
                except Exception:
                    raise UploadValidationError("content_b64 is not valid base64") from None
                meta = save_upload(raw, str(body.get("filename", "upload.csv")), str(body.get("task", "")),
                                   body.get("target") or None, body.get("time_column") or None, body.get("name"))
                return 201, _clean(meta)
            return None
    except (BenchmarkError, UploadValidationError, DatasetLoadError, ValueError) as exc:
        return 400, {"error": str(exc)}
    except KeyError as exc:
        return 404, {"error": str(exc).strip("'\"")}
    return None


def _chat_samples(run_id: str) -> Response:
    """CSV of synthetic samples produced by a chat 'generate' run (served as text)."""
    from src.chatbot.workflows import workflows_dir

    safe = "".join(c for c in run_id if c.isalnum() or c in "_-")
    path = workflows_dir() / f"{safe}_samples.csv"
    if not safe or not path.is_file():
        return 404, {"error": "samples not found"}
    if path.stat().st_size > 5 * 1024 * 1024:
        return 413, {"error": "samples file too large to return inline"}
    return 200, {"run_id": safe, "filename": path.name, "csv": path.read_text(encoding="utf-8")}


def parse_body(raw: bytes, content_type: str) -> Dict[str, Any]:
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError("request body too large")
    if not raw:
        return {}
    if "application/json" not in (content_type or ""):
        raise ValueError("POST bodies must be application/json")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data
