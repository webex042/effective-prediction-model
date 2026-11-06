"""
Sandboxed execution of training / benchmark jobs.

Each job runs in a fresh ``spawn`` subprocess so that a slow model can be
stopped at a hard timeout (Python threads cannot be killed) and a crash in a
native library cannot take the chat server down. Progress messages flow back
over a multiprocessing queue; the trained artifact is written to a temporary
file that the parent verifies and moves into the :class:`ModelStore`.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_CTX = mp.get_context("spawn")


class JobTimeout(RuntimeError):
    pass


class JobCancelledError(RuntimeError):
    pass


class JobFailed(RuntimeError):
    pass


def _load_dataset(spec: Dict[str, Any]):
    from epm_chatbot.ml_service.datasets import UploadStore, load_registry

    if spec["kind"] == "registry":
        return load_registry(spec["dataset_id"], spec.get("allow_downloads", False))
    store = UploadStore(Path(spec["upload_root"]), ttl_s=10 ** 9, max_bytes=10 ** 9)
    return store.as_dataset(spec["sid"], spec["upload_id"], spec["task"], spec["target"], spec.get("time_column"))


def _child(q, job: Dict[str, Any]) -> None:  # pragma: no cover - runs in a subprocess (covered by integration tests)
    root = job["project_root"]
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import warnings

        warnings.filterwarnings("ignore")

        def progress(p: float, msg: str) -> None:
            q.put(("progress", float(p), str(msg)[:160]))

        progress(0.02, "Loading dataset")
        ds = _load_dataset(job["dataset"])
        if job["kind"] == "train":
            from epm_chatbot.ml_service.training import serialize, train

            model, meta = train(ds, job["task"], job["algorithm_id"], max_train_rows=job["max_train_rows"],
                                allow_neural=job["allow_neural"], epochs=job["epochs"], horizon=job.get("horizon"),
                                window=job.get("window"), progress=progress)
            Path(job["artifact_path"]).write_bytes(serialize(model))
            q.put(("done", meta))
        elif job["kind"] == "benchmark":
            from src.benchmarking.runner import BenchmarkConfig, run_benchmark

            cfg = BenchmarkConfig(dataset_id=ds.spec.id, task=job["task"], algorithms=job["algorithms"],
                                  seed=42, cv_folds=0, max_train_rows=job["max_train_rows"],
                                  epochs=job["epochs"], save=False, allow_download=job["dataset"].get("allow_downloads", False))
            if job["dataset"]["kind"] != "registry":
                raise ValueError("Comparisons run on registry datasets; train individual models on uploaded files.")
            res = run_benchmark(cfg, progress=progress)
            for m in res.get("models", []):
                m.pop("traceback", None)
            q.put(("done", res))
        else:
            raise ValueError(f"unknown job kind {job['kind']}")
    except Exception as exc:
        from epm_chatbot.ml_service.datasets import DataAccessError
        from epm_chatbot.ml_service.training import TrainingError

        if isinstance(exc, (DataAccessError, TrainingError, ValueError, KeyError)):
            q.put(("error", str(exc).strip("'\"")))
        else:
            q.put(("error", f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=4)))


def run_sandboxed(job: Dict[str, Any], timeout_s: float, progress: Callable[[float, str], None],
                  cancelled: Callable[[], bool]) -> Any:
    """Run ``job`` in a subprocess. Raises JobTimeout / JobCancelledError / JobFailed."""
    q = _CTX.Queue()
    proc = _CTX.Process(target=_child, args=(q, job), daemon=True)
    proc.start()
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            if cancelled():
                raise JobCancelledError("Cancelled.")
            if time.monotonic() > deadline:
                raise JobTimeout(f"The job exceeded the {int(timeout_s)} s time limit and was stopped.")
            try:
                msg = q.get(timeout=0.25)
            except queue_mod.Empty:
                if not proc.is_alive():
                    # drain a final message that may have raced with process exit
                    try:
                        msg = q.get(timeout=0.5)
                    except queue_mod.Empty:
                        raise JobFailed(f"The training process exited unexpectedly (code {proc.exitcode}).") from None
                else:
                    continue
            kind = msg[0]
            if kind == "progress":
                progress(msg[1], msg[2])
            elif kind == "done":
                return msg[1]
            elif kind == "error":
                err = JobFailed(msg[1])
                err.trace = msg[2] if len(msg) > 2 else None  # type: ignore[attr-defined]
                raise err
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():  # pragma: no cover
                proc.kill()
        q.close()
