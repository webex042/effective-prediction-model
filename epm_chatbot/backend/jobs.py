"""
Bounded background job queue for training / comparison jobs.

* ``max_queued`` caps queued + running jobs (resource exhaustion guard).
* Each job's work function receives ``progress(p, msg)`` and ``cancelled()``.
* The work itself runs in a sandboxed subprocess (see ``ml_service.runner``),
  so timeouts and cancellation are enforced by terminating that process.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from epm_chatbot.ml_service.runner import JobCancelledError, JobFailed, JobTimeout

log = logging.getLogger("epm_chat.jobs")


class QueueFull(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    sid: str
    kind: str
    label: str
    fn: Callable[[Callable[[float, str], None], Callable[[], bool]], Any]
    state: str = "queued"            # queued | running | done | failed | cancelled | timeout
    progress: float = 0.0
    message: str = "Queued"
    result: Any = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    on_done: Optional[Callable[["Job"], None]] = None
    reply: Optional[Dict[str, Any]] = None   # assistant message built when the job finishes

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    def public(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label, "state": self.state,
                "progress": round(self.progress, 3), "message": self.message, "error": self.error,
                "elapsed_s": round((self.finished or time.time()) - (self.started or self.created), 1)}


class JobManager:
    def __init__(self, workers: int = 1, max_queued: int = 4, keep: int = 200) -> None:
        self.max_queued = max_queued
        self.keep = keep
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        for i in range(max(1, workers)):
            t = threading.Thread(target=self._worker, name=f"epm-chat-job-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def submit(self, sid: str, kind: str, label: str, fn, on_done=None) -> Job:
        with self._lock:
            active = sum(1 for j in self._jobs.values() if j.active)
            if active >= self.max_queued:
                raise QueueFull("The server is busy with other training jobs. Please try again in a minute.")
            if any(j.active and j.sid == sid for j in self._jobs.values()):
                raise QueueFull("You already have a job running. Wait for it to finish or say 'cancel'.")
            job = Job(id="j_" + secrets.token_hex(6), sid=sid, kind=kind, label=label, fn=fn, on_done=on_done)
            self._jobs[job.id] = job
            while len(self._jobs) > self.keep:
                old_id, old = next(iter(self._jobs.items()))
                if old.active:
                    break
                self._jobs.pop(old_id)
        self._q.put(job)
        return job

    def get(self, job_id: str, sid: str) -> Optional[Job]:
        j = self._jobs.get(job_id)
        return j if j is not None and j.sid == sid else None

    def active_for(self, sid: str) -> Optional[Job]:
        for j in reversed(list(self._jobs.values())):
            if j.sid == sid and j.active:
                return j
        return None

    def cancel(self, job_id: str, sid: str) -> bool:
        j = self.get(job_id, sid)
        if j is None or not j.active:
            return False
        j.cancel_flag.set()
        if j.state == "queued":
            self._finish(j, "cancelled", error="Cancelled before it started.")
        return True

    def cancel_session(self, sid: str) -> int:
        n = 0
        for j in list(self._jobs.values()):
            if j.sid == sid and j.active:
                n += int(self.cancel(j.id, sid))
        return n

    def _finish(self, j: Job, state: str, result: Any = None, error: Optional[str] = None) -> None:
        if j.done_event.is_set():
            return
        j.state, j.result, j.error = state, result, error
        j.finished = time.time()
        j.progress = 1.0 if state == "done" else j.progress
        j.message = {"done": "Finished", "failed": "Failed", "cancelled": "Cancelled", "timeout": "Timed out"}[state]
        if j.on_done:
            try:
                j.on_done(j)
            except Exception:  # pragma: no cover - defensive
                log.exception("job on_done failed", extra={"job": j.id})
        j.done_event.set()

    def _worker(self) -> None:
        while True:
            j = self._q.get()
            if j.state != "queued":
                continue
            j.state, j.started, j.message = "running", time.time(), "Starting"

            def progress(p: float, msg: str, j=j) -> None:
                j.progress = max(j.progress, min(0.99, float(p)))
                j.message = msg

            try:
                res = j.fn(progress, j.cancel_flag.is_set)
                self._finish(j, "done", result=res)
            except JobCancelledError:
                self._finish(j, "cancelled", error="Cancelled.")
            except JobTimeout as e:
                self._finish(j, "timeout", error=str(e))
            except JobFailed as e:
                if getattr(e, "trace", None):
                    log.error("job failed", extra={"job": j.id, "kind": j.kind, "trace": e.trace})
                self._finish(j, "failed", error=str(e))
            except Exception as e:  # pragma: no cover - defensive
                log.exception("job crashed", extra={"job": j.id})
                self._finish(j, "failed", error="an internal error occurred (details are in the server log)")
            log.info("job finished", extra={"job": j.id, "kind": j.kind, "state": j.state,
                                            "seconds": round(j.finished - j.started, 2)})
