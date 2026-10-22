"""
EPM ChatEngine — natural-language front end over the existing EPM pipeline.

``ChatEngine.handle(session_id, message) -> ChatResponse``

Conversation state machine (per session)::

    idle ──intent──▶ slot_filling ──all slots──▶ (confirming) ──▶ executing ──▶ showing_results ─▶ idle
                         ▲   │ ask for missing slot                    │ job failed
                         └───┘                                         ▼
                                                                      error ─▶ idle

The engine contains no ML logic. It resolves the request against the dataset
registry and algorithm catalogue, validates it, and then calls one of:
``run_benchmark``, ``recommend_algorithms`` / ``recommend_datasets``,
``load_dataset``, ``download_dataset`` or the Phase 3 workflows in
``src.chatbot.workflows`` (which compose EPM's generative / compression modules).

Execution is synchronous by default (terminal, tests). The web API passes
``submit`` / ``cancel`` callables so long jobs run on the existing JobManager
and the UI polls for completion.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.benchmarking.catalog import (
    ALGORITHMS, check_compatibility, compatible_algorithms, list_algorithms, recommend_algorithms, recommend_datasets,
)
from src.benchmarking.runner import BenchmarkConfig, BenchmarkError, load_history, load_run, run_benchmark
from src.chatbot import responses as R
from src.chatbot.intents import INTENTS, TASK_INTENTS, classify_intent
from src.chatbot.slots import ChatSlots, algorithm_name, extract_slots, resolve_algorithms
from src.data.loaders import DatasetLoadError, dataset_status, download_dataset, load_dataset
from src.data.registry import DATASETS, TASKS, get_dataset_spec, list_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_MESSAGE_CHARS = 1000
MAX_SESSIONS = 500
MAX_HISTORY = 200
RUN_INTENTS = {"classify", "regress", "cluster", "forecast", "detect_anomalies", "benchmark", "train", "compare",
               "uncertainty"}
WORKFLOW_INTENTS = {"generate", "compare_generative", "compress"}
# short messages with these intents are treated as a change of topic, never as a slot answer
EXPLICIT_TOPIC_INTENTS = {"describe_dataset", "preview_data", "list_datasets", "list_algorithms", "explain", "help",
                          "status", "download", "show_results", "generate", "compress", "compare_generative"}
NEEDS_DATASET = RUN_INTENTS | {"generate", "compress", "describe_dataset", "preview_data", "download"}


class ChatError(Exception):
    """User-facing error with a helpful suggestion."""

    def __init__(self, message: str, suggestion: str = "", suggestions: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.suggestions = suggestions or []


@dataclass
class Session:
    id: str
    state: str = "idle"
    context: ChatSlots = field(default_factory=ChatSlots)       # remembered across turns
    pending: Optional[Dict[str, Any]] = None                    # {"intent", "slots", "awaiting", ...}
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    last_kind: Optional[str] = None
    last_request: Optional[Dict[str, Any]] = None
    last_job_id: Optional[str] = None
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_intent: str = "unknown"

    def log(self, role: str, text: str, **extra: Any) -> None:
        self.history.append({"role": role, "text": text, "ts": round(time.time(), 3), **extra})
        del self.history[:-MAX_HISTORY]


SubmitFn = Callable[[str, Callable[[Callable[[float, str], None]], Any], Dict[str, Any]], Dict[str, Any]]


class ChatEngine:
    def __init__(self, submit: Optional[SubmitFn] = None, cancel: Optional[Callable[[str], Any]] = None,
                 active_job: Optional[Callable[[str], Optional[str]]] = None,
                 progress_printer: Optional[Callable[[float, str], None]] = None) -> None:
        self.submit = submit
        self.cancel_job = cancel
        self.active_job = active_job
        self.progress_printer = progress_printer
        self.sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ sessions
    def session(self, session_id: Optional[str]) -> Session:
        sid = re.sub(r"[^A-Za-z0-9_\-]", "", session_id or "")[:64] or uuid.uuid4().hex[:12]
        with self._lock:
            s = self.sessions.get(sid)
            if s is None:
                s = self.sessions[sid] = Session(sid)
            self.sessions.move_to_end(sid)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        return s

    def session_info(self, session_id: str) -> Dict[str, Any]:
        s = self.session(session_id)
        return {"session_id": s.id, "current_state": s.state, "history": s.history,
                "last_job_id": s.last_job_id, "context": s.context.to_dict(),
                "pending": None if not s.pending else {"intent": s.pending["intent"], "awaiting": s.pending.get("awaiting")}}

    # ------------------------------------------------------------------ entry
    def handle(self, session_id: Optional[str], message: str) -> R.ChatResponse:
        s = self.session(session_id)
        msg = (message or "").strip()[:MAX_MESSAGE_CHARS]
        with s.lock:
            s.log("user", msg)
            try:
                resp = self._dispatch(s, msg)
            except ChatError as e:
                resp = R.ChatResponse(e.message + (f"\n{e.suggestion}" if e.suggestion else ""),
                                      state=s.state if s.pending else "error", kind="error",
                                      intent=getattr(s, "current_intent", "unknown"),
                                      suggestions=e.suggestions or ["help", "list datasets"])
                if not s.pending:
                    s.state = "idle"
            except (BenchmarkError, DatasetLoadError, KeyError, ValueError) as e:
                s.state = "idle"
                resp = R.ChatResponse(f"I couldn't do that: {str(e).strip(chr(39) + chr(34))}", state="error",
                                      kind="error", intent=getattr(s, "current_intent", "unknown"),
                                      suggestions=["help", "list datasets"])
            resp.slots = resp.slots or s.context.to_dict()
            s.log("bot", resp.text, intent=resp.intent, job_id=resp.job_id, state=resp.state)
            return resp

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, s: Session, msg: str) -> R.ChatResponse:
        intent, conf, source = classify_intent(msg)
        slots = extract_slots(msg)

        # 1) confirmation / slot answers for a pending request
        if s.pending:
            p = s.pending
            if p.get("awaiting") == "confirm":
                if intent == "confirm":
                    s.pending = None
                    p["slots"].allow_large = True
                    return self._run_intent(s, p["intent"], p["slots"], confirmed=True)
                if intent in ("deny", "cancel"):
                    s.pending, s.state = None, "idle"
                    return R.ChatResponse("OK, cancelled — nothing was run.", intent="cancel",
                                          suggestions=["list datasets", "help"])
            elif source != "rule" or intent in ("confirm", "train", "benchmark", "recommend") or intent == p["intent"] \
                    or (len(msg.split()) <= 4 and intent not in EXPLICIT_TOPIC_INTENTS):
                answered = self._fill_pending(p, slots, msg, intent)
                if answered:
                    s.pending = None
                    return self._run_intent(s, p["intent"], p["slots"], asked=answered)
            if intent in ("deny", "cancel"):
                s.pending, s.state = None, "idle"
                return R.ChatResponse("OK, I've dropped that request.", intent="cancel", suggestions=["help"])
            if source != "rule":
                return self._ask_again(s)
            s.pending = None  # user changed topic

        if intent == "confirm":
            return R.ChatResponse("There's nothing waiting for confirmation.", intent="confirm",
                                  suggestions=["help", "show results"])
        if intent == "deny":
            return R.ChatResponse("OK.", intent="deny", suggestions=["help"])

        # 2) follow-ups: "run again with 5-fold cv" / "now with seed 7" re-run the last request
        if s.last_request and re.search(r"\b(again|re-?run|same (thing|setup|run)|instead)\b", msg, re.I) \
                and intent in ("train", "unknown", "benchmark", s.last_request["intent"]) and not slots.dataset_id:
            return self._rerun(s, slots)
        if intent == "unknown":
            if s.last_request and self._has_settings(slots):
                return self._rerun(s, slots)
            if slots.dataset_id:
                intent = "train"
            elif slots.algorithm_families or slots.algorithm_ids:
                intent = "recommend"
            else:
                return R.ChatResponse("I didn't understand that. " + "Try one of the examples below, or say **help**.",
                                      intent="unknown", kind="error",
                                      suggestions=["classify iris with random forest", "list datasets", "help"])
        return self._run_intent(s, intent, slots)

    def _has_settings(self, slots: ChatSlots) -> bool:
        return any(v is not None for v in (slots.cv_folds, slots.seed, slots.horizon, slots.window,
                                           slots.contamination, slots.max_train_rows, slots.n_clusters, slots.epochs)) \
            or bool(slots.algorithm_families or slots.algorithm_ids or slots.all_algorithms or slots.hyperparams)

    def _rerun(self, s: Session, slots: ChatSlots) -> R.ChatResponse:
        req = s.last_request
        merged = ChatSlots()
        merged.merge(req["slots"])
        if slots.algorithm_families or slots.algorithm_ids or slots.all_algorithms:
            merged.algorithm_families, merged.algorithm_ids, merged.all_algorithms = [], [], False
        merged.merge(slots)
        return self._run_intent(s, req["intent"], merged)

    def _fill_pending(self, p: Dict[str, Any], slots: ChatSlots, msg: str, intent: str):
        """Apply an answer to the pending question.

        Returns False (not an answer), "dataset" (dataset chosen interactively — the
        run step may ask for algorithms next) or True (request complete).
        """
        target: ChatSlots = p["slots"]
        awaiting = p.get("awaiting")
        if awaiting == "dataset":
            if not slots.dataset_id:
                return False
            target.merge(slots)
            return "dataset"
        if awaiting == "algorithms":
            low = msg.lower().strip()
            if slots.all_algorithms or low in ("all", "everything", "all of them", "every one"):
                target.all_algorithms = True
            elif re.search(r"\b(recommended|default|defaults|you choose|any|whatever|suggested)\b", low):
                pass
            elif slots.algorithm_families or slots.algorithm_ids:
                target.algorithm_families, target.algorithm_ids = slots.algorithm_families, slots.algorithm_ids
            else:
                return False
            target.merge(ChatSlots(**{k: v for k, v in slots.__dict__.items()
                                      if k not in ("algorithm_families", "algorithm_ids", "all_algorithms")}))
            return True
        if awaiting == "task":
            if not slots.task:
                return False
            target.task = slots.task
            return True
        return False

    def _ask_again(self, s: Session) -> R.ChatResponse:
        p = s.pending
        q = {"dataset": "Which dataset? Name one from the list (e.g. `iris`).",
             "algorithms": "Which algorithms? Name them, or say **all** / **recommended**.",
             "task": "Which task: classification, regression, clustering, forecasting or anomaly detection?",
             "confirm": "Please answer **yes** or **no**."}.get(p.get("awaiting"), "Could you rephrase?")
        return R.ChatResponse(q, state=s.state, intent=p["intent"], kind="question",
                              suggestions=p.get("suggestions", []))

    # ------------------------------------------------------------------ routing
    def _run_intent(self, s: Session, intent: str, slots: ChatSlots, asked: bool = False,
                    confirmed: bool = False) -> R.ChatResponse:
        s.current_intent = intent
        handler = getattr(self, f"_i_{intent}", None)
        if handler is None:
            raise ChatError(f"I can't handle '{intent}' yet.")
        resp = handler(s, slots) if intent not in RUN_INTENTS | WORKFLOW_INTENTS else handler(s, slots, asked=asked)
        resp.intent = resp.intent if resp.intent != "unknown" else intent
        return resp

    def _remember(self, s: Session, slots: ChatSlots) -> None:
        for k in ("dataset_id", "task"):
            v = getattr(slots, k)
            if v:
                setattr(s.context, k, v)

    def _ask(self, s: Session, intent: str, slots: ChatSlots, awaiting: str, text: str,
             suggestions: List[str], tables: Optional[List[Dict[str, Any]]] = None) -> R.ChatResponse:
        s.pending = {"intent": intent, "slots": slots, "awaiting": awaiting, "suggestions": suggestions}
        s.state = "confirming" if awaiting == "confirm" else "slot_filling"
        return R.ChatResponse(text, state=s.state, intent=intent, kind="question", suggestions=suggestions,
                              tables=tables or [], slots=slots.to_dict())

    # ------------------------------------------------------------------ helpers
    def _dataset_for(self, s: Session, intent: str, slots: ChatSlots, task: Optional[str],
                     use_context: bool) -> Optional[str]:
        if slots.dataset_id:
            return slots.dataset_id
        if use_context and s.context.dataset_id:
            ctx = get_dataset_spec(s.context.dataset_id)
            if task is None or ctx.supports(task):
                slots.dataset_id = s.context.dataset_id
                return slots.dataset_id
        return None

    def _ask_dataset(self, s: Session, intent: str, slots: ChatSlots, task: Optional[str]) -> R.ChatResponse:
        specs = list_datasets(task) if task else list_datasets()
        listing = ", ".join(f"`{d.id}` ({R.fmt(d.n_samples)} rows)" for d in specs[:10])
        what = f"for {R.human_task(task)}" if task else ""
        return self._ask(s, intent, slots, "dataset", f"Which dataset would you like to use {what}?\nAvailable: {listing}",
                         [d.id for d in specs[:4]])

    @staticmethod
    def _check_dataset_task(spec, task: str) -> None:
        if not spec.supports(task):
            alt = [d.id for d in list_datasets(task)][:5]
            raise ChatError(
                f"**{spec.name}** is a {', '.join(R.human_task(t) for t in spec.tasks)} dataset, not {R.human_task(task)}.",
                f"{R.human_task(task).capitalize()} datasets: {', '.join(alt)}.",
                [f"{_verb(task)} {alt[0]}" if alt else "list datasets", f"{_verb(spec.tasks[0])} {spec.id}"],
            )

    def _validate_settings(self, slots: ChatSlots, spec, task: str) -> List[str]:
        notes: List[str] = []
        if slots.cv_folds is not None and not (slots.cv_folds == 0 or 2 <= slots.cv_folds <= 10):
            raise ChatError("Cross-validation folds must be between 2 and 10.")
        if slots.n_clusters is not None and not 2 <= slots.n_clusters <= 50:
            raise ChatError("The number of clusters must be between 2 and 50.")
        if slots.epochs is not None and not 1 <= slots.epochs <= 300:
            raise ChatError("Epochs must be between 1 and 300.")
        if slots.max_train_rows is not None and slots.max_train_rows < 50:
            raise ChatError("Max train rows must be at least 50.")
        if slots.contamination is not None and not 0 < slots.contamination < 0.5:
            raise ChatError("Contamination must be between 0 and 0.5 (e.g. 0.05 for 5%).")
        if task == "forecasting":
            h = slots.horizon or spec.default_horizon or 1
            if not 1 <= h <= 168:
                raise ChatError("The forecast horizon must be between 1 and 168 steps.")
            if slots.window is not None:
                if slots.window < 2 * h:
                    raise ChatError(f"The lookback window ({slots.window}) must be at least 2 × the horizon ({h}), "
                                    f"i.e. ≥ {2 * h}.", suggestions=[f"window {2 * h}"])
                if slots.window > 512:
                    raise ChatError("The lookback window must be ≤ 512.")
            elif (spec.default_window or 24) < 2 * h:
                slots.window = 2 * h
                notes.append(f"Lookback window set to {slots.window} (2 × horizon {h}); the default "
                             f"{spec.default_window} would be too short.")
        elif slots.horizon is not None or slots.window is not None:
            if task != "anomaly_detection":
                notes.append("Horizon/window only apply to forecasting and were ignored.")
        return notes

    # ------------------------------------------------------------------ run intents
    def _i_classify(self, s, slots, asked=False):
        return self._run_task(s, "classify", slots, "classification", asked)

    def _i_regress(self, s, slots, asked=False):
        return self._run_task(s, "regress", slots, "regression", asked)

    def _i_cluster(self, s, slots, asked=False):
        return self._run_task(s, "cluster", slots, "clustering", asked)

    def _i_forecast(self, s, slots, asked=False):
        return self._run_task(s, "forecast", slots, "forecasting", asked)

    def _i_detect_anomalies(self, s, slots, asked=False):
        return self._run_task(s, "detect_anomalies", slots, "anomaly_detection", asked)

    def _i_train(self, s, slots, asked=False):
        return self._run_task(s, "train", slots, None, asked)

    def _i_benchmark(self, s, slots, asked=False):
        slots.all_algorithms = True
        return self._run_task(s, "benchmark", slots, None, asked)

    def _i_uncertainty(self, s, slots, asked=False):
        ds = self._dataset_for(s, "uncertainty", slots, None, use_context=True)
        if not ds:
            return self._ask_dataset(s, "uncertainty", slots, "classification")
        spec = get_dataset_spec(ds)
        if not spec.supports("classification"):
            return self._uncertainty_other(spec)
        slots.task = "classification"
        if not (slots.algorithm_ids or slots.algorithm_families or slots.all_algorithms):
            slots.algorithm_ids = [a for a in ("logreg", "rf_clf", "gnb", "svc", "knn_clf") if
                                   check_compatibility(ALGORITHMS[a], spec, "classification")[0]]
        return self._run_task(s, "uncertainty", slots, "classification", asked, calibration=True)

    def _uncertainty_other(self, spec) -> R.ChatResponse:
        lines = [f"Calibration (ECE / Brier) is measured for classifiers; **{spec.name}** is a "
                 f"{', '.join(R.human_task(t) for t in spec.tasks)} dataset.",
                 "The registry's point forecasters and regressors don't produce predictive distributions."]
        art = _read_json(PROJECT_ROOT / "benchmarks" / "stage9_timeseries_diffusion.json")
        if art and art.get("coverage_80") is not None:
            lines.append(f"Recorded Stage 9 artifact (probabilistic diffusion forecaster, synthetic telemetry, horizon "
                         f"{art.get('horizon')}): CRPS {R.fmt(art.get('crps'))}, empirical coverage of the 80% interval "
                         f"**{R.fmt(100 * art['coverage_80'], 1)}%** — far below nominal, i.e. the intervals are over-confident. "
                         "(Read from benchmarks/stage9_timeseries_diffusion.json, not re-run.)")
        return R.ChatResponse("\n".join(lines), intent="uncertainty",
                              suggestions=["how calibrated is the classifier on wine?", "explain diffusion"])

    def _i_compare(self, s, slots, asked=False):
        ds = self._dataset_for(s, "compare", slots, None, use_context=True)
        last = s.last_result if s.last_kind == "benchmark" else None
        if not (slots.algorithm_ids or slots.algorithm_families) or slots.all_algorithms:
            slots.all_algorithms = True
            return self._run_task(s, "compare", slots, None, asked)
        if last and (not ds or ds == last["dataset_id"]):
            ids, problems = resolve_algorithms(slots, last["task"])
            present = [m["algorithm"] for m in last["models"] if m.get("status") == "ok"]
            if len(ids) >= 2 and all(i in present for i in ids):
                return self._compare_from(last, ids)
        if len(slots.algorithm_ids) + len(slots.algorithm_families) < 2 and not last:
            raise ChatError("Name at least two algorithms to compare, e.g. \"compare random forest and svm on iris\".")
        return self._run_task(s, "compare", slots, None, asked)

    def _compare_from(self, result: Dict[str, Any], ids: List[str]) -> R.ChatResponse:
        pm = result["primary_metric"]
        ms = [m for m in result["models"] if m["algorithm"] in ids and m.get("status") == "ok"]
        ms.sort(key=lambda m: m.get("rank", 99))
        a = ms[0]
        lines = [f"From run `{result['run_id']}` on **{result['dataset_name']}** (same split, seed {result['config']['seed']}):"]
        for b in ms[1:]:
            diff = (a["metrics"][pm["name"]] or 0) - (b["metrics"][pm["name"]] or 0)
            lat = (b.get("latency_us_per_sample") or 0) / max(a.get("latency_us_per_sample") or 1e-9, 1e-9)
            speed = (f"{R.fmt(lat, 2)}× slower" if lat >= 1 else f"{R.fmt(1 / lat, 2)}× faster") if lat > 0 else "n/a"
            lines.append(f"- **{a['name']}** beats {b['name']} on {pm['name'].upper()} by {R.fmt(abs(diff))}; "
                         f"{b['name']} is {speed} per prediction.")
            if a.get("cv") and b.get("cv"):
                gap, spread = abs(a["cv"]["mean"] - b["cv"]["mean"]), max(a["cv"]["std"], b["cv"]["std"])
                lines.append("  The cross-validated difference is within one std, so it may not be meaningful."
                             if gap <= spread else "  The gap holds up under cross-validation.")
            elif not a.get("cv"):
                lines.append("  Single split only — add cross-validation to check the gap is real.")
        sugg = ([] if result["config"].get("cv_folds") else ["run again with 5-fold cv"]) + \
            ["show results", "benchmark all algorithms"]
        return R.ChatResponse("\n".join(lines), state="showing_results", kind="results", intent="compare",
                              tables=[R.benchmark_table(result, only=ids)], suggestions=sugg)

    def _run_task(self, s: Session, intent: str, slots: ChatSlots, task: Optional[str], asked: bool,
                  calibration: bool = False) -> R.ChatResponse:
        # ---------------- dataset
        ds = self._dataset_for(s, intent, slots, task, use_context=True)
        if not ds:
            return self._ask_dataset(s, intent, slots, task or slots.task)
        spec = get_dataset_spec(ds)
        # ---------------- task
        if task is None:
            task = slots.task if slots.task and spec.supports(slots.task) else None
            if task is None and (slots.algorithm_families or slots.algorithm_ids):
                for t in spec.tasks:  # pick the dataset task that the named algorithm supports
                    if resolve_algorithms(slots, t)[0]:
                        task = t
                        break
            task = task or (s.context.task if s.context.task and spec.supports(s.context.task)
                            and s.context.dataset_id == spec.id else None) or spec.tasks[0]
        self._check_dataset_task(spec, task)
        slots.task = task
        # ---------------- algorithms
        problems: List[str] = []
        if slots.all_algorithms:
            n_train = min(int((spec.n_samples or 0) * 0.7), slots.max_train_rows or 10 ** 9)
            ok, _ = compatible_algorithms(spec, task, n_train=n_train, window=slots.window,
                                          allow_large=True)
            algos = [a.id for a in ok]
        elif slots.algorithm_ids or slots.algorithm_families:
            algos, problems = resolve_algorithms(slots, task)
            if not algos:
                raise ChatError(" ".join(problems), suggestions=[f"list {R.human_task(task)} algorithms",
                                                                  f"{_verb(task)} {spec.id}"])
        else:
            if asked == "dataset" and intent in RUN_INTENTS - {"uncertainty", "benchmark", "compare"}:
                rec = recommend_algorithms(spec.id, task, window=slots.window, allow_large=True)["recommended"]
                return self._ask(s, intent, slots, "algorithms",
                                 "Which algorithms? Say **all** for the full benchmark, **recommended** for "
                                 f"the default set, or name some.\nRecommended: {', '.join(algorithm_name(a) for a in rec)}",
                                 ["recommended", "all", rec[0] if rec else "help"])
            algos = None  # runner uses the recommended set
        # ---------------- validation & size
        notes = self._validate_settings(slots, spec, task)
        if spec.size_class == "large" and not slots.allow_large:
            return self._ask(s, intent, slots, "confirm",
                             f"**{spec.name}** is a large dataset (~{R.fmt(spec.approx_download_mb, 0)} MB download, "
                             f"~{R.fmt(spec.approx_memory_mb, 0)} MB in memory, {R.fmt(spec.n_samples)} rows). "
                             "Download (if needed) and run it?", ["yes", "no"])
        hyper = {}
        if slots.hyperparams:
            if algos and len(algos) == 1:
                hyper = {algos[0]: _safe_params(slots.hyperparams)}
            else:
                notes.append("Hyper-parameters are applied only when exactly one algorithm is named; ignored.")
        cfg = BenchmarkConfig(
            dataset_id=spec.id, task=task, algorithms=algos, seed=slots.seed if slots.seed is not None else 42,
            cv_folds=slots.cv_folds or 0, max_train_rows=slots.max_train_rows, allow_large=slots.allow_large,
            window=slots.window, horizon=slots.horizon, n_clusters=slots.n_clusters,
            contamination=slots.contamination, epochs=slots.epochs, algorithm_params=hyper,
        )
        self._remember(s, slots)
        s.last_request = {"intent": intent, "slots": slots}
        names = ", ".join(algorithm_name(a) for a in algos) if algos else "the recommended algorithms"
        extra = []
        if task == "forecasting":
            extra.append(f"horizon={cfg.horizon or spec.default_horizon}, window={cfg.window or spec.default_window}")
        if task == "clustering":
            extra.append(f"k={cfg.n_clusters or spec.n_clusters_hint}")
        if cfg.cv_folds:
            extra.append(f"{cfg.cv_folds}-fold CV")
        st = dataset_status(spec.id)
        if st["state"] in ("not_downloaded", "error"):
            extra.append(f"downloads ~{R.mb(spec.approx_download_mb)} first")
        start = (f"Running {R.human_task(task)} on **{spec.name}**" + (f" ({'; '.join(extra)})" if extra else "")
                 + f" with {names}.")
        start += "".join(f"\n- {p}" for p in problems + notes)
        kind = "calibration" if calibration else "benchmark"
        return self._execute(s, kind, intent, lambda prog: run_benchmark(cfg, prog), start,
                             {"dataset_id": spec.id, "task": task})

    # ------------------------------------------------------------------ Phase 3 workflows
    def _i_generate(self, s, slots, asked=False):
        from src.chatbot.workflows import run_generation

        ds = self._dataset_for(s, "generate", slots, None, use_context=asked)
        if not ds:
            return self._ask_dataset(s, "generate", slots, None)
        spec = get_dataset_spec(ds)
        if spec.size_class == "large" and not slots.allow_large:
            raise ChatError(f"{spec.name} is too large for the interactive generator; pick a small or medium dataset.")
        note = ""
        if slots.generative_model == "ae":
            note = "\n- A plain autoencoder has no prior to sample from, so I'm generating with a VAE and showing the AE's reconstruction for comparison."
        n = slots.num_samples or 100
        if not 1 <= n <= 10000:
            raise ChatError("Number of samples must be between 1 and 10,000.")
        epochs = slots.epochs or 80
        beta = slots.hyperparams.get("beta")
        self._remember(s, slots)
        s.last_request = {"intent": "generate", "slots": slots}
        start = f"Training a VAE on **{spec.name}** and generating {n} synthetic rows ({epochs} epochs).{note}"
        return self._execute(s, "generate", "generate",
                             lambda prog: run_generation(spec.id, num_samples=n, latent_dim=slots.latent_dim, epochs=epochs,
                                                         seed=slots.seed if slots.seed is not None else 42,
                                                         beta=float(beta) if isinstance(beta, (int, float)) else None,
                                                         progress=prog),
                             start, {"dataset_id": spec.id})

    def _i_compare_generative(self, s, slots, asked=False):
        from src.chatbot.workflows import run_generative_comparison

        ds = slots.dataset_id or "epm_sensor_telemetry"
        spec = get_dataset_spec(ds)
        if spec.size_class == "large":
            raise ChatError(f"{spec.name} is too large for the interactive comparison; try `epm_sensor_telemetry` or `wine`.")
        epochs = slots.epochs or 40
        s.last_request = {"intent": "compare_generative", "slots": slots}
        start = (f"Comparing OT-CFM flow matching (10/20 Euler steps) with DDPM (100 steps) and DDIM (20/10 steps) "
                 f"on **{spec.name}**, {epochs} epochs each, scored on held-out rows.")
        return self._execute(s, "compare_generative", "compare_generative",
                             lambda prog: run_generative_comparison(spec.id, epochs=epochs,
                                                                    seed=slots.seed if slots.seed is not None else 42,
                                                                    progress=prog),
                             start, {"dataset_id": spec.id})

    def _i_compress(self, s, slots, asked=False):
        from src.chatbot.workflows import run_compression

        ds = self._dataset_for(s, "compress", slots, "classification", use_context=True)
        if not ds:
            return self._ask_dataset(s, "compress", slots, "classification")
        spec = get_dataset_spec(ds)
        self._check_dataset_task(spec, "classification")
        epochs = slots.epochs or 30
        self._remember(s, slots)
        s.last_request = {"intent": "compress", "slots": slots}
        start = (f"Training a teacher MLP on **{spec.name}**, then a distilled student, INT8 quantisation and "
                 f"pruning ({epochs} epochs).")
        return self._execute(s, "compress", "compress",
                             lambda prog: run_compression(spec.id, epochs=epochs,
                                                          seed=slots.seed if slots.seed is not None else 42,
                                                          max_rows=slots.max_train_rows or 30000, progress=prog),
                             start, {"dataset_id": spec.id})

    # ------------------------------------------------------------------ execution
    def _execute(self, s: Session, kind: str, intent: str, fn: Callable, start_text: str,
                 meta: Dict[str, Any]) -> R.ChatResponse:
        if self.submit is not None:
            job = self.submit(f"chat_{kind}", fn, {**meta, "session": s.id, "chat_kind": kind})
            s.jobs[job["id"]] = {"kind": kind, "intent": intent, "reply": None}
            s.last_job_id = job["id"]
            s.state = "executing"
            return R.ChatResponse(start_text, state="executing", intent=intent, kind="progress", job_id=job["id"],
                                  suggestions=["status", "cancel"])
        # synchronous (terminal / tests)
        s.state = "executing"
        prog = self.progress_printer or (lambda f, m: None)
        try:
            result = fn(prog)
        except Exception as exc:
            s.state = "idle"
            raise ChatError(f"The run failed: {exc}") from exc
        resp = self._format_result(s, kind, intent, result)
        resp.text = start_text + "\n\n" + resp.text
        return resp

    def job_update(self, session_id: str, job: Dict[str, Any]) -> R.ChatResponse:
        """Turn a JobManager record into a chat reply (cached once the job is finished)."""
        s = self.session(session_id)
        with s.lock:
            info = s.jobs.get(job["id"])
            if info is None:
                return R.ChatResponse("That job doesn't belong to this chat session.", kind="error", state=s.state)
            if info.get("reply") is not None:
                return R.ChatResponse(**info["reply"])
            status = job.get("status")
            if status in ("queued", "running"):
                return R.ChatResponse(job.get("message") or "Working…", state="executing", kind="progress",
                                      job_id=job["id"], intent=info["intent"], data={"progress": job.get("progress", 0)})
            if status == "done" and job.get("result") is not None:
                resp = self._format_result(s, info["kind"], info["intent"], job["result"])
            elif status == "cancelled":
                s.state = "idle"
                resp = R.ChatResponse("Cancelled. Completed models were discarded; nothing was saved for this run.",
                                      intent="cancel", suggestions=["show results", "help"])
            else:
                s.state = "error"
                resp = R.ChatResponse(f"The run failed: {job.get('error') or 'unknown error'}", state="error", kind="error",
                                      intent=info["intent"], suggestions=["list datasets", "help"])
                s.state = "idle"
            resp.job_id = job["id"]
            info["reply"] = resp.to_dict()
            s.log("bot", resp.text, intent=resp.intent, job_id=job["id"], state=resp.state)
            return resp

    def _format_result(self, s: Session, kind: str, intent: str, result: Dict[str, Any]) -> R.ChatResponse:
        s.state = "showing_results"
        if kind in ("benchmark", "calibration"):
            s.last_result, s.last_kind = result, "benchmark"
            text = R.benchmark_summary(result, calibration=kind == "calibration")
            tables = [R.benchmark_table(result, calibration=kind == "calibration")]
            if kind == "calibration":
                tables.append(R.benchmark_table(result))
            t = result["task"]
            sugg = (["run again with 5-fold cv"] if not result["config"].get("cv_folds") else []) + \
                   [f"compare {_short(result, 0)} and {_short(result, 1)}" if len(result["models"]) > 1 else "show results",
                    "benchmark all algorithms" if t != "clustering" else "cluster with k=4",
                    "try another dataset"]
            data = {"run_id": result["run_id"], "dataset_id": result["dataset_id"], "task": t}
            ok_ids = [m["algorithm"] for m in result["models"] if m.get("status") == "ok"]
            if intent == "compare" and 2 <= len(ok_ids) <= 4 and kind == "benchmark":
                text += "\n" + self._compare_from(result, ok_ids).text
        elif kind == "generate":
            s.last_result, s.last_kind = result, "generate"
            text, tables = R.generation_reply(result)
            sugg = ["generate 500 samples with 100 epochs", "compare diffusion vs flow matching", "explain vae"]
            data = {"run_id": result["run_id"], "samples_csv": result["samples_csv"]}
        elif kind == "compare_generative":
            s.last_result, s.last_kind = result, "compare_generative"
            text, tables = R.gen_compare_reply(result)
            sugg = ["explain flow matching", "explain diffusion", "generate samples from wine with a vae"]
            data = {"run_id": result["run_id"]}
        elif kind == "compress":
            s.last_result, s.last_kind = result, "compress"
            text, tables = R.compression_reply(result)
            sugg = ["explain knowledge distillation", "compress a model on digits", "show results"]
            data = {"run_id": result["run_id"]}
        elif kind == "download":
            s.state = "idle"
            return R.ChatResponse(f"Downloaded and verified **{result['dataset_id']}** (state: {result['state']}).",
                                  intent="download", kind="results",
                                  suggestions=[f"preview {result['dataset_id']}", f"describe {result['dataset_id']}"])
        else:  # pragma: no cover
            text, tables, sugg, data = json.dumps(result)[:500], [], [], {}
        return R.ChatResponse(text, state="showing_results", intent=intent, kind="results", tables=tables,
                              suggestions=sugg, data=data)

    # ------------------------------------------------------------------ exploratory intents
    def _i_list_datasets(self, s, slots):
        task = slots.task
        last_msg = s.history[-1]["text"].lower() if s.history else ""
        if not task and re.search(r"\b(another|other|different)\b", last_msg):
            task = s.context.task
        specs = list_datasets(task)
        status = {d.id: dataset_status(d.id)["state"].replace("_", " ") for d in specs}
        what = f"{R.human_task(task)} datasets" if task else "datasets"
        return R.ChatResponse(f"{len(specs)} {what} in the registry (small first). Status 'not downloaded' means it is "
                              "fetched from its official source on first use.", intent="list_datasets",
                              tables=[R.dataset_rows(specs, status)],
                              suggestions=[f"describe {specs[0].id}" if specs else "help",
                                           f"{_verb(task or 'classification')} {specs[0].id}" if specs else "help"])

    def _i_list_algorithms(self, s, slots):
        if slots.dataset_id:
            spec = get_dataset_spec(slots.dataset_id)
            task = slots.task if slots.task and spec.supports(slots.task) else spec.tasks[0]
            rec = recommend_algorithms(spec.id, task, allow_large=True)
            rows = [[a, algorithm_name(a), ALGORITHMS[a].framework, "yes" if a in rec["recommended"] else ""]
                    for a in rec["compatible"]]
            text = f"{len(rows)} algorithms can run {R.human_task(task)} on **{spec.name}**."
            if rec["excluded"]:
                text += "\nNot compatible: " + "; ".join(f"{algorithm_name(a)} ({why[0]})" for a, why in list(rec["excluded"].items())[:4])
            return R.ChatResponse(text, intent="list_algorithms",
                                  tables=[R.table("Compatible algorithms", ["id", "Name", "Framework", "Recommended"], rows)],
                                  suggestions=[f"benchmark {spec.id}", f"recommend a model for {spec.id}"])
        algos = list_algorithms(slots.task)
        rows = [[a.id, a.name, R.human_task(a.task), a.framework, a.cost, "yes" if a.available() else "no"] for a in algos]
        return R.ChatResponse(f"{len(rows)} algorithms" + (f" for {R.human_task(slots.task)}" if slots.task else "") + ".",
                              intent="list_algorithms",
                              tables=[R.table("Algorithms", ["id", "Name", "Task", "Framework", "Cost", "Installed"], rows)],
                              suggestions=["list classification algorithms", "which algorithms work on iris?"])

    def _i_describe_dataset(self, s, slots):
        ds = self._dataset_for(s, "describe_dataset", slots, None, use_context=True)
        if not ds:
            return self._ask_dataset(s, "describe_dataset", slots, None)
        spec = get_dataset_spec(ds)
        self._remember(s, slots)
        return R.ChatResponse(R.describe_spec(spec, dataset_status(spec.id)), intent="describe_dataset",
                              suggestions=[f"preview {spec.id}", f"recommend a model for {spec.id}",
                                           f"{_verb(spec.tasks[0])} {spec.id}"])

    def _i_preview_data(self, s, slots):
        ds = self._dataset_for(s, "preview_data", slots, None, use_context=True)
        if not ds:
            return self._ask_dataset(s, "preview_data", slots, None)
        spec = get_dataset_spec(ds)
        st = dataset_status(spec.id)
        if st["state"] not in ("available", "cached"):
            raise ChatError(f"**{spec.name}** hasn't been downloaded yet ({R.mb(spec.approx_download_mb)} from {spec.source}).",
                            suggestions=[f"download {spec.id}", f"describe {spec.id}"])
        ld = load_dataset(spec.id, allow_download=False, allow_large=True)
        n = max(1, min(slots.n_rows or 5, 20))
        head = ld.features.head(n).copy()
        if ld.target is not None and spec.target not in head.columns:
            head[spec.target] = ld.target.head(n).values
        cols = list(map(str, head.columns))[:12]
        rows = [[(v if isinstance(v, str) else (None if v != v else v)) for v in r] for r in head[cols].values.tolist()]
        self._remember(s, slots)
        text = (f"First {n} of {R.fmt(ld.n_samples)} rows of **{spec.name}** "
                f"({ld.features.shape[1]} feature columns{'; showing 12' if len(head.columns) > 12 else ''}). "
                f"Missing cells: {int(ld.features.isna().sum().sum())}.")
        return R.ChatResponse(text, intent="preview_data", tables=[R.table("Preview", cols, rows)],
                              suggestions=[f"describe {spec.id}", f"{_verb(spec.tasks[0])} {spec.id}"])

    def _i_download(self, s, slots):
        ds = self._dataset_for(s, "download", slots, None, use_context=False)
        if not ds:
            return self._ask_dataset(s, "download", slots, None)
        spec = get_dataset_spec(ds)
        if not spec.requires_network:
            return R.ChatResponse(f"**{spec.name}** is built in — nothing to download.", intent="download",
                                  suggestions=[f"preview {spec.id}"])
        if spec.size_class == "large" and not slots.allow_large:
            return self._ask(s, "download", slots, "confirm",
                             f"**{spec.name}** is ~{R.fmt(spec.approx_download_mb, 0)} MB. Download it?", ["yes", "no"])
        return self._execute(s, "download", "download", lambda prog: download_dataset(spec.id, slots.allow_large),
                             f"Downloading **{spec.name}** from {spec.source}…", {"dataset_id": spec.id})

    def _i_recommend(self, s, slots):
        ds = slots.dataset_id or None
        if ds is None and (slots.algorithm_families or slots.algorithm_ids):
            ids = list(slots.algorithm_ids)
            for fam in slots.algorithm_families:
                from src.chatbot.slots import family_default

                pick = family_default(fam, slots.task)
                if pick:
                    ids.append(pick)
            if not ids:
                raise ChatError("I don't know that algorithm.", suggestions=["list algorithms"])
            aid = ids[0]
            quick = recommend_datasets(aid, "quick")
            real = recommend_datasets(aid, "realistic")
            text = (f"Datasets for **{algorithm_name(aid)}** ({R.human_task(ALGORITHMS[aid].task)}):\n"
                    f"- Quick experiments: {', '.join(quick['datasets'][:4]) or '—'}\n"
                    f"- Realistic benchmarks: {', '.join(real['datasets'][:4]) or '—'}")
            if quick["rejected"]:
                text += "\n- Excluded: " + "; ".join(f"{k} ({v[0]})" for k, v in list(quick["rejected"].items())[:3])
            first = quick["datasets"][0] if quick["datasets"] else None
            return R.ChatResponse(text, intent="recommend",
                                  suggestions=[f"{_verb(ALGORITHMS[aid].task)} {first} with {algorithm_name(aid).lower()}"] if first else ["list datasets"])
        ds = ds or self._dataset_for(s, "recommend", slots, None, use_context=True)
        if not ds:
            return self._ask_dataset(s, "recommend", slots, slots.task)
        spec = get_dataset_spec(ds)
        task = slots.task if slots.task and spec.supports(slots.task) else spec.tasks[0]
        rec = recommend_algorithms(spec.id, task, allow_large=True)
        self._remember(s, slots)
        lines = [f"For {R.human_task(task)} on **{spec.name}** ({R.fmt(spec.n_samples)} rows) I'd start with:"]
        for a in rec["recommended"][:8]:
            al = ALGORITHMS[a]
            lines.append(f"- {al.name} — {al.family}, cost {al.cost}, interpretability {al.interpretability}")
        lines.append("These are chosen by compatibility, cost and size, not predicted accuracy. Benchmark them to see which actually wins.")
        last = s.last_result if s.last_kind == "benchmark" else None
        if last and last["dataset_id"] == spec.id and last["task"] == task and (last.get("summary") or {}).get("best_by_metric"):
            lines.append(f"Measured earlier in this chat: **{algorithm_name(last['summary']['best_by_metric'])}** was best "
                         f"({last['primary_metric']['name']} {R.fmt(last['summary']['best_value'])}).")
        return R.ChatResponse("\n".join(lines), intent="recommend",
                              suggestions=[f"benchmark {spec.id}", f"{_verb(task)} {spec.id} with 5-fold cv"])

    def _i_show_results(self, s, slots):
        if s.last_result:
            kind = {"benchmark": "benchmark"}.get(s.last_kind, s.last_kind)
            resp = self._format_result(s, kind, "show_results", s.last_result)
            return resp
        hist = load_history()
        if hist:
            r = load_run(hist[0]["run_id"])
            if r:
                resp = self._format_result(s, "benchmark", "show_results", r)
                resp.text = "Nothing has run in this chat yet; here is the most recent saved run.\n" + resp.text
                return resp
        return R.ChatResponse("No results yet. Try \"classify iris\".", intent="show_results",
                              suggestions=["classify iris", "help"])

    def _i_status(self, s, slots):
        cached = sum(1 for d in DATASETS if dataset_status(d)["state"] in ("available", "cached"))
        lines = [f"- Datasets ready locally: {cached}/{len(DATASETS)} (others download on first use)",
                 f"- Algorithms in the catalogue: {len(ALGORITHMS)} "
                 f"({sum(1 for a in ALGORITHMS.values() if a.available())} installed)"]
        if self.active_job:
            jid = self.active_job(s.id)
            lines.append(f"- Active job in this chat: `{jid}`" if jid else "- No job running in this chat")
        hist = load_history()
        if hist:
            h = hist[0]
            lines.append(f"- Last saved run: {h.get('dataset_name')} / {R.human_task(h.get('task'))} — best "
                         f"{h.get('best')} ({R.fmt(h.get('best_value'))})")
        ts = _read_json(PROJECT_ROOT / "benchmarks" / "test_status.json")
        if ts and ts.get("available"):
            lines.append(f"- Last recorded test run: {ts.get('passed')}/{ts.get('collected')} passed ({ts.get('generated_at')})")
        lines.append(f"- Conversation state: {s.state}")
        return R.ChatResponse("**EPM status**\n" + "\n".join(lines), intent="status", suggestions=["show results", "help"])

    def _i_cancel(self, s, slots):
        if s.pending:
            s.pending, s.state = None, "idle"
            return R.ChatResponse("Dropped the pending request.", intent="cancel")
        jid = self.active_job(s.id) if self.active_job else None
        if jid and self.cancel_job:
            self.cancel_job(jid)
            return R.ChatResponse(f"Cancellation requested for job `{jid}`. It stops at the next model boundary "
                                  "(a model already training finishes first).", intent="cancel", job_id=jid,
                                  state="executing")
        return R.ChatResponse("Nothing is running in this chat.", intent="cancel", suggestions=["status"])

    def _i_help(self, s, slots):
        s.state = "idle"
        return R.ChatResponse(R.HELP_TEXT, intent="help",
                              suggestions=["classify iris with random forest", "list datasets",
                                           "forecast air quality 12 steps ahead", "explain flow matching"])

    def _i_explain(self, s, slots):
        from src.chatbot.docs_index import get_index

        query = s.history[-1]["text"] if s.history else ""
        topic = re.sub(r"^\s*(please\s+)?(explain|define|what (is|are)|what's|tell me about|how does|how do|why does|teach me)\s+",
                       "", query, flags=re.I).strip(" ?.")
        hits = get_index().search(topic or query)
        if slots.dataset_id and (not hits or hits[0]["score"] < 0.25):
            return self._i_describe_dataset(s, slots)
        if not hits:
            topics = get_index().topics()
            return R.ChatResponse(f"I couldn't find \"{topic}\" in EPM's documentation. Topics I can explain: "
                                  + ", ".join(t.split(":")[0] for t in topics[:14]) + ".", intent="explain",
                                  suggestions=["explain diffusion", "explain attention", "explain backpropagation"])
        h = hits[0]
        return R.ChatResponse(f"**{h['title']}**\n{h['excerpt']}\n\nSource: `docs/{h['file']}`", intent="explain",
                              data={"source": f"docs/{h['file']}"},
                              suggestions=_explain_followups(h["file"]))


# ------------------------------------------------------------------ module helpers

def _verb(task: Optional[str]) -> str:
    return {"classification": "classify", "regression": "regression on", "clustering": "cluster",
            "forecasting": "forecast", "anomaly_detection": "detect anomalies in"}.get(task or "", "train on")


def _short(result: Dict[str, Any], i: int) -> str:
    ms = sorted([m for m in result["models"] if m.get("status") == "ok"], key=lambda m: m.get("rank", 99))
    return ms[i]["algorithm"] if i < len(ms) else ""


def _safe_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in params.items():
        if not re.fullmatch(r"[a-z][a-z_]{0,24}", k):
            continue
        if isinstance(v, (int, float, bool)) or v is None or (isinstance(v, str) and re.fullmatch(r"[a-z_]{1,16}", v)):
            out[k] = v
    return out


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _explain_followups(file: str) -> List[str]:
    return {
        "12_flow_matching.md": ["compare diffusion vs flow matching", "explain diffusion"],
        "11_diffusion.md": ["compare diffusion vs flow matching", "explain flow matching"],
        "10_generative_models.md": ["generate samples from wine with a vae", "explain diffusion"],
        "13_model_compression.md": ["compress a model on breast cancer", "explain quantization"],
        "09_time_series.md": ["forecast the sensor telemetry with lstm", "explain attention"],
    }.get(file, ["help", "list datasets"])
