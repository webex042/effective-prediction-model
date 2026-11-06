"""
Conversation controller for EPM Chat.

It decides what a message is asking for, asks follow-up questions when inputs
are missing, validates them, and calls the ML service. It never fabricates
results: every number in a reply comes from a model card, a prediction, an
explanation computed on the loaded model, or a benchmark result.

Message categories
------------------
* general questions about EPM / how predictions work  → ``about``, ``how_it_works``
* questions about ML models and concepts              → ``concept``, ``list_models``
* prediction requests                                  → ``predict`` (single row or uploaded CSV)
* upload / dataset analysis                            → ``upload`` (file endpoint), ``analyze``, ``upload_setup``
* explaining previous results                          → ``explain_result``, ``importance``
* training / comparison (explicit only)                → ``train``, ``compare``
* unsupported or ambiguous                             → ``unknown`` (helpful options; optional LLM routing)
"""

from __future__ import annotations

import io
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from epm_chatbot.backend import knowledge as K
from epm_chatbot.backend.jobs import Job, JobManager, QueueFull
from epm_chatbot.backend.llm import LLMService
from epm_chatbot.config import PROJECT_ROOT, Settings
from epm_chatbot.ml_service import inference as I
from epm_chatbot.ml_service.datasets import (
    PREDICTABLE_TASKS, DataAccessError, UploadStore, availability, load_registry, registry_datasets, session_key,
)
from epm_chatbot.ml_service.model_store import ModelStore, ModelStoreError
from epm_chatbot.ml_service.runner import run_sandboxed
from epm_chatbot.ml_service.training import TrainingError, check_trainable
from src.benchmarking.catalog import ALGORITHMS, recommend_algorithms
from src.chatbot.slots import extract_slots, resolve_algorithms
from src.data.registry import DATASETS, get_dataset_spec

log = logging.getLogger("epm_chat.controller")

MAX_HISTORY = 120
MAX_PRIVATE_MODELS = 8
PREDICT_SEM = threading.BoundedSemaphore(4)

WELCOME_CHIPS = [
    "Explain how this prediction works.",
    "Run a prediction using my data.",
    "Compare available ML models.",
    "Explain the prediction results.",
    "Show me important features.",
]
INTENT_LABELS = ["predict", "train", "compare", "importance", "explain_result", "how_it_works", "about",
                 "list_models", "list_algorithms", "list_datasets", "describe_dataset", "concept", "help", "unknown"]


# =============================================================================
# Message building
# =============================================================================

def text(t: str) -> Dict[str, Any]:
    return {"type": "text", "text": t}


def notice(t: str, level: str = "info") -> Dict[str, Any]:
    return {"type": "notice", "level": level, "text": t}


def table(columns: List[str], rows: List[List[Any]], title: str = "") -> Dict[str, Any]:
    return {"type": "table", "title": title, "columns": columns, "rows": rows}


def bars(title: str, items: List[Dict[str, Any]], note: str = "", signed: bool = False) -> Dict[str, Any]:
    return {"type": "bars", "title": title, "items": items, "note": note, "signed": signed}


def details(title: str, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "details", "title": title, "blocks": blocks}


def fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not np.isfinite(f):
            return "—"
        a = abs(f)
        if a != 0 and (a < 1e-3 or a >= 1e6):
            return f"{f:.3e}"
        return f"{f:,.{d}f}".rstrip("0").rstrip(".") if d else f"{f:,.0f}"
    return str(v)


def pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{100 * float(v):.1f}%"


def task_name(t: str) -> str:
    return t.replace("_", " ")


@dataclass
class Session:
    sid: str
    created: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)
    dataset: Optional[Dict[str, Any]] = None        # {"kind": "registry"|"upload", "id", "name", "task", ...}
    model_id: Optional[str] = None
    upload_id: Optional[str] = None
    pending: Optional[Dict[str, Any]] = None
    last_prediction: Optional[Dict[str, Any]] = None
    last_compare: Optional[Dict[str, Any]] = None
    downloads: "OrderedDict[str, Tuple[str, bytes]]" = field(default_factory=OrderedDict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def owner(self) -> str:
        return session_key(self.sid)


class Controller:
    def __init__(self, settings: Settings, store: ModelStore, uploads: UploadStore, jobs: JobManager,
                 llm: LLMService) -> None:
        self.s = settings
        self.store = store
        self.uploads = uploads
        self.jobs = jobs
        self.llm = llm
        self.sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ sessions
    def session(self, sid: str) -> Session:
        with self._lock:
            s = self.sessions.get(sid)
            if s is None:
                s = self.sessions[sid] = Session(sid)
            self.sessions.move_to_end(sid)
            while len(self.sessions) > self.s.max_sessions:
                old_sid, _ = self.sessions.popitem(last=False)
                self._purge(old_sid)
        return s

    def _purge(self, sid: str) -> None:
        self.jobs.cancel_session(sid)
        self.uploads.clear(sid)
        self.store.delete_owner(session_key(sid))

    def clear(self, sid: str) -> Dict[str, Any]:
        self._purge(sid)
        with self._lock:
            self.sessions.pop(sid, None)
        s = self.session(sid)
        msg = self.welcome()
        self._log(s, msg)
        return msg

    def welcome(self) -> Dict[str, Any]:
        shared = self.store.list(owner=None)
        blocks = [text("Hi, I'm **EPM Chat**. I run predictions with EPM's trained machine-learning models, explain "
                       "the results, and compare models on held-out data. Ask in plain English, or upload a CSV with "
                       "the paperclip button.")]
        if shared:
            names = sorted({c["dataset_name"] for c in shared})
            blocks.append(text(f"Trained models are ready for: **{', '.join(names)}**."))
        else:
            blocks.append(notice("No trained models are installed on this server yet. You can ask me to train one "
                                 "(e.g. “train a random forest on iris”), or the operator can run the bootstrap "
                                 "command described in the README.", "info"))
        return self._msg(blocks, WELCOME_CHIPS, intent="welcome")

    def history(self, sid: str) -> List[Dict[str, Any]]:
        return list(self.session(sid).history)

    def _log(self, s: Session, msg: Dict[str, Any]) -> None:
        s.history.append(msg)
        del s.history[:-MAX_HISTORY]

    def _msg(self, blocks: List[Dict[str, Any]], suggestions: Optional[List[str]] = None, intent: str = "",
             job: Optional[Job] = None, state: str = "idle") -> Dict[str, Any]:
        return {"id": "a_" + secrets.token_hex(5), "role": "assistant", "ts": round(time.time(), 3),
                "blocks": blocks, "suggestions": (suggestions or [])[:6], "intent": intent,
                "state": "executing" if job else state, "job": job.public() if job else None}

    # ------------------------------------------------------------------ entry
    def handle(self, sid: str, message: str, on_delta: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        s = self.session(sid)
        msg = (message or "").strip()
        if len(msg) > self.s.max_message_chars:
            return self._msg([notice(f"Messages are limited to {self.s.max_message_chars} characters.", "error")],
                             intent="error", state="error")
        with s.lock:
            self._log(s, {"id": "u_" + secrets.token_hex(5), "role": "user", "ts": round(time.time(), 3),
                          "blocks": [text(msg)]})
            t0 = time.perf_counter()
            try:
                intent = self.detect_intent(s, msg)
                out = self._dispatch(s, intent, msg, on_delta)
            except (DataAccessError, TrainingError, ModelStoreError, I.InputError) as e:
                out = self._msg([notice(str(e), "error")] + ([text("\n".join("- " + d for d in e.details))]
                                                            if getattr(e, "details", None) else []),
                                ["What can you do?", "List available models"], intent="error", state="error")
            except QueueFull as e:
                out = self._msg([notice(str(e), "warning")], ["Cancel"], intent="busy", state="error")
            except Exception:
                log.exception("controller error")
                out = self._msg([notice("Something went wrong on our side while handling that request. "
                                        "Nothing was changed; please retry, or rephrase the request.", "error")],
                                ["What can you do?"], intent="error", state="error")
            log.info("message handled", extra={"intent": out.get("intent"), "ms": round((time.perf_counter() - t0) * 1000, 1)})
            self._log(s, out)
            return out

    # ------------------------------------------------------------------ intent detection
    def detect_intent(self, s: Session, msg: str) -> str:
        t = msg.lower().strip()
        if not t:
            return "help"
        p = s.pending
        if p and re.fullmatch(r"(y|yes|yeah|yep|sure|ok|okay|go( ahead)?|do it|proceed|confirm|start|train it|run it)[.! ]*", t):
            return "confirm"
        if p and re.fullmatch(r"(n|no|nope|nah|never ?mind|don'?t|not now|no thanks)[.! ]*", t):
            return "deny"
        if re.search(r"\b(cancel|abort|stop)\b", t) and len(t) < 40:
            return "cancel"
        if re.fullmatch(r"(hi|hello|hey|help|\?|menu|start over|what can you do\??|how do i use (this|you)\??)[.!? ]*", t):
            return "help"
        if p and p.get("kind") == "missing_features":
            if re.search(r"\b(typical|default|median|rest|remaining|fill)\b", t):
                return "fill_missing"
            card = self._card_or_none(s, p["model_id"])
            if card and I.parse_feature_values(msg, card)[0]:
                return "provide_values"
        if s.upload_id and (re.search(r"\b(target|label)\b", t) or
                            re.search(r"\b(train|fit|build)\b.*\b(my|this|the|uploaded)\s+(file|data|csv|dataset)\b", t)):
            return "upload_setup"
        if re.search(r"\bhow\b.*\b(this|the|a|my|your|it|prediction|model)?\b.*\b(work|works|made|computed|calculated|produced)\b", t):
            return "how_it_works"
        if re.search(r"\b(feature importance|important features?|which features|most important|key drivers|what matters|drivers?)\b", t):
            return "importance"
        if re.search(r"\b(explain|interpret|why|understand|reason)\b.*\b(prediction|result|output|forecast|answer|probabilit\w*|that|this|it)\b"
                     r"|^\s*why\b", t):
            return "explain_result"
        if re.search(r"\b(what is|about|tell me about)\s+epm\b|\bwhat (does|can) epm\b", t):
            return "about"
        if re.search(r"\b(compare|comparison|versus|vs\.?|benchmark|leaderboard)\b|\bwhich (model|algorithm)s? (is|are|works?) (best|better)\b|\bbest model\b", t):
            return "compare"
        if re.search(r"\b(list|show|which|what)\b.*\bmodels?\b.*\b(trained|available|saved|ready|have|installed)\b"
                     r"|\b(available|trained|my) models\b|^\s*models\??$", t):
            return "list_models"
        if re.search(r"\b(list|show|which|what)\b.*\bdatasets?\b|^\s*datasets\??$", t):
            return "list_datasets"
        if re.search(r"\b(list|show|which|what)\b.*\balgorithms?\b|^\s*algorithms\??$", t):
            return "list_algorithms"
        if re.search(r"\b(preview|first rows|head of|sample rows)\b", t):
            return "preview"
        if re.search(r"\b(describe|details? (of|about)|info(rmation)? (about|on)|schema|columns of|features of)\b", t):
            return "describe_dataset"
        if re.search(r"\b(train|fit|build|create|retrain)\b", t) and not re.search(r"\bpredict", t):
            return "train"
        if re.search(r"\b(predict\w*|forecast\w*|estimate|classify|score)\b|\bwhat (would|will)\b|\bmy data\b", t):
            return "predict"
        if re.search(r"\b(upload|csv|my file|analy[sz]e)\b", t):
            return "analyze"
        if re.search(r"^\s*(what is|what are|what's|explain|define|how does|tell me about)\b", t):
            return "concept"
        slots = extract_slots(msg)
        if slots.dataset_id:
            return "select_dataset"
        if self.llm.enabled:
            label = self.llm.classify_intent(msg, INTENT_LABELS)
            if label and label != "unknown":
                return label
        return "unknown"

    def _dispatch(self, s: Session, intent: str, msg: str, on_delta) -> Dict[str, Any]:
        if intent not in ("confirm", "deny", "provide_values", "fill_missing", "upload_setup", "cancel"):
            s.pending = None if intent not in ("help",) else s.pending
        fn = getattr(self, f"_i_{intent}")
        res = fn(s, msg, on_delta) if intent in ("explain_result",) else fn(s, msg)
        res["intent"] = res.get("intent") or intent
        return res

    # ------------------------------------------------------------------ helpers: dataset / model resolution
    def _card_or_none(self, s: Session, mid: Optional[str]) -> Optional[Dict[str, Any]]:
        if not mid:
            return None
        try:
            return self.store.card(mid, s.owner)
        except ModelStoreError:
            return None

    @staticmethod
    def _mentions_file(msg: str) -> bool:
        return bool(re.search(r"\b(my|this|the|uploaded)\s+(file|csv|upload|data|dataset)\b", msg.lower()))

    def _dataset_from(self, s: Session, msg: str, use_context: bool = True) -> Optional[Dict[str, Any]]:
        slots = extract_slots(msg)
        if slots.dataset_id:
            spec = get_dataset_spec(slots.dataset_id)
            tasks = [t for t in spec.tasks if t in PREDICTABLE_TASKS]
            if spec.size_class == "large":
                raise DataAccessError(f"{spec.name} is a large dataset and is disabled in the chatbot.")
            if not tasks:
                raise DataAccessError(f"{spec.name} is a {', '.join(map(task_name, spec.tasks))} dataset; the chatbot "
                                      "makes predictions for classification, regression and forecasting.")
            ds = {"kind": "registry", "id": spec.id, "name": spec.name, "task": tasks[0]}
            s.dataset = ds
            return ds
        if self._mentions_file(msg) and s.upload_id:
            return s.dataset if s.dataset and s.dataset.get("kind") == "upload" else None
        return s.dataset if use_context else None

    def _requested_algos(self, msg: str, task: str) -> Tuple[List[str], List[str]]:
        slots = extract_slots(msg)
        ids, problems = resolve_algorithms(slots, task)
        return ids, problems

    def _models_for(self, s: Session, ds: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.store.list(owner=s.owner, dataset_id=self._ds_key(ds), task=ds["task"])

    @staticmethod
    def _ds_key(ds: Dict[str, Any]) -> str:
        return ds["id"] if ds["kind"] == "registry" else f"upload:{ds['upload_id']}"

    def _pick_model(self, s: Session, ds: Dict[str, Any], algos: List[str], horizon: Optional[int] = None
                    ) -> Optional[Dict[str, Any]]:
        models = self._models_for(s, ds)
        if algos:
            models = [m for m in models if m["algorithm_id"] in algos]
        if horizon and ds["task"] == "forecasting":
            models = [m for m in models if m["forecast"]["horizon"] >= horizon]
        if not models:
            return None
        for m in models:
            if m["id"] == s.model_id:
                return m
        return sorted(models, key=_rank_key)[0]   # best held-out score when the user did not choose

    # ------------------------------------------------------------------ cards → blocks
    def model_block(self, c: Dict[str, Any]) -> Dict[str, Any]:
        m = c.get("metrics", {})
        if c["task"] == "classification":
            metrics = [["Accuracy", fmt(m.get("accuracy"))], [f"F1 ({m.get('averaging', '')})", fmt(m.get("f1"))],
                       ["ROC-AUC", fmt(m.get("roc_auc"))], ["Calibration error (ECE)", fmt(m.get("ece"))]]
        elif c["task"] == "regression":
            metrics = [["RMSE", fmt(m.get("rmse"))], ["MAE", fmt(m.get("mae"))], ["R²", fmt(m.get("r2"))]]
            if c.get("conformal"):
                metrics.append([f"{int(c['conformal']['level'] * 100)}% interval half-width", fmt(c["conformal"]["halfwidth"])])
        else:
            fc = c["forecast"]
            metrics = [["RMSE", fmt(m.get("rmse"))], ["MAE", fmt(m.get("mae"))], ["MASE", fmt(m.get("mase"))],
                       ["Naive baseline RMSE", fmt(fc.get("naive_test_rmse"))],
                       ["Skill vs naive", fmt(m.get("skill_vs_naive"))]]
        sp = c.get("split", {})
        if c["task"] == "forecasting":
            split = (f"chronological · {fmt(sp.get('n_train_windows'))} train / {fmt(sp.get('n_val_windows'))} val / "
                     f"{fmt(sp.get('n_test_windows'))} test windows")
        else:
            split = (f"{sp.get('strategy', '?')} · {fmt(sp.get('n_train'))} train / {fmt(sp.get('n_val'))} val / "
                     f"{fmt(sp.get('n_test'))} test rows")
        rows = [["Model", c["algorithm_name"]], ["Framework", c.get("framework", "")], ["Dataset", c["dataset_name"]],
                ["Task", task_name(c["task"])], ["Target", c["target"]["name"]], ["Split", split],
                ["Trained", c.get("created_at", "")], ["Model id", c["id"]]]
        if c["task"] == "forecasting":
            rows.insert(5, ["Window → horizon", f"{c['forecast']['window']} → {c['forecast']['horizon']} steps"])
        return {"type": "model", "title": f"{c['algorithm_name']} · {c['dataset_name']}", "rows": rows,
                "metrics": metrics, "metrics_note": "Measured on the held-out test split.",
                "preprocessing": c.get("preprocessing", [])}

    # ------------------------------------------------------------------ intents: general
    def _i_help(self, s, msg):
        return self.welcome()

    def _i_about(self, s, msg):
        return self._msg([text(K.ABOUT_EPM)], ["List available models", "Explain how this prediction works.",
                                               "Run a prediction using my data."])

    def _i_unknown(self, s, msg):
        return self._msg([text("I'm not sure what you'd like to do. I can make predictions with trained models, train "
                               "a model when you ask, compare models, show important features, explain a prediction, "
                               "or analyse a CSV you upload.")],
                         ["List available models", "Predict iris species", "What can you do?",
                          "Run a prediction using my data."], intent="unknown")

    def _i_cancel(self, s, msg):
        n = self.jobs.cancel_session(s.sid)
        had = s.pending is not None
        s.pending = None
        if n:
            return self._msg([text("Stopping the running job. Nothing from it will be saved.")], intent="cancel")
        return self._msg([text("Okay, cancelled." if had else "There's nothing running to cancel.")], intent="cancel")

    def _i_deny(self, s, msg):
        s.pending = None
        return self._msg([text("Okay, I won't do that.")], ["List available models", "What can you do?"])

    def _i_how_it_works(self, s, msg):
        c = self._card_or_none(s, s.model_id)
        blocks = [text(K.HOW_PREDICTIONS_WORK)]
        if c:
            blocks.insert(0, text(f"For the current model, **{c['algorithm_name']}** on **{c['dataset_name']}**:"))
            blocks.append(text(K.algorithm_text(c["algorithm_id"], c["algorithm_name"])))
            blocks.append(details("Preprocessing fitted on the training split",
                                  [text("\n".join("- " + p for p in c.get("preprocessing", [])))]))
            blocks.append(self.model_block(c))
            sugg = ["Show me important features.", "Explain the prediction results.", "Compare available ML models."]
        else:
            sugg = ["Predict iris species", "List available models", "Run a prediction using my data."]
        return self._msg(blocks, sugg)

    def _i_concept(self, s, msg):
        slots = extract_slots(msg)
        task = slots.task or "classification"
        ids = list(slots.algorithm_ids)
        for fam in slots.algorithm_families:
            from src.chatbot.slots import family_default
            a = family_default(fam, None)
            if a:
                ids.append(a)
        if ids:
            blocks = [text(K.algorithm_text(a, ALGORITHMS[a].name)) for a in dict.fromkeys(ids) if a in ALGORITHMS]
            a0 = ALGORITHMS[ids[0]]
            blocks.append(text(f"In EPM it is available for **{task_name(a0.task)}** "
                               f"({a0.framework}, interpretability: {a0.interpretability}, cost: {a0.cost})."))
            return self._msg(blocks, [f"Compare available ML models.", "List available models"])
        from src.chatbot.docs_index import get_index

        topic = re.sub(r"^\s*(please\s+)?(explain|define|what (is|are)|what's|tell me about|how does)\s+", "", msg,
                       flags=re.I).strip(" ?.")
        hits = get_index().search(topic or msg)
        if not hits:
            return self._msg([text(f"I couldn't find “{topic}” in EPM's documentation. I can explain algorithms "
                                   "(e.g. “what is gradient boosting”) and ML concepts covered in the EPM docs.")],
                             ["What is a random forest?", "Explain data leakage", "Explain how this prediction works."])
        h = hits[0]
        return self._msg([text(f"**{h['title']}**\n{h['excerpt']}"),
                          {"type": "source", "text": f"Source: EPM documentation, docs/{h['file']}"}],
                         ["Explain how this prediction works.", "List available models"])

    # ------------------------------------------------------------------ intents: catalogue
    def _i_list_models(self, s, msg):
        models = self.store.list(owner=s.owner)
        if not models:
            return self._msg([text("No trained models are available yet. Ask me to train one, e.g. "
                                   "“train a random forest on iris”.")],
                             ["Train a random forest on iris", "List datasets"])
        rows = []
        for c in models[:30]:
            m = c["metrics"]
            key, val = (("F1", m.get("f1")) if c["task"] == "classification" else ("RMSE", m.get("rmse")))
            rows.append([c["dataset_name"], c["algorithm_name"], task_name(c["task"]), f"{key} {fmt(val)}",
                         "your session" if c.get("owner") else "shared"])
        sugg = []
        for c in models[:3]:
            sugg.append(self._predict_chip(c))
        return self._msg([text(f"**{len(models)}** trained model(s) can make predictions right now. Metrics are from "
                               "each model's held-out test split."),
                          table(["Dataset", "Model", "Task", "Held-out score", "Scope"], rows)], sugg)

    def _predict_chip(self, c: Dict[str, Any]) -> str:
        if c["task"] == "forecasting":
            return f"Forecast {c['dataset_name']} with {c['algorithm_name']}"
        return f"Predict with {c['algorithm_name']} on {c['dataset_name']}"

    def _i_list_algorithms(self, s, msg):
        from src.benchmarking.catalog import list_algorithms

        slots = extract_slots(msg)
        rows = []
        for t in ([slots.task] if slots.task in PREDICTABLE_TASKS else PREDICTABLE_TASKS):
            for a in list_algorithms(t):
                if a.framework == "baseline" and t != "forecasting":
                    continue
                if a.framework == "pytorch" and not self.s.allow_neural:
                    continue
                rows.append([a.name, task_name(t), a.framework, "yes" if a.supports_proba else "no",
                             a.interpretability, a.cost])
        return self._msg([text("Algorithms the chatbot can train for predictions (EPM catalogue):"),
                          table(["Algorithm", "Task", "Framework", "Probabilities", "Interpretability", "Cost"], rows)],
                         ["What is gradient boosting?", "Train XGBoost on breast cancer", "List available models"])

    def _i_list_datasets(self, s, msg):
        slots = extract_slots(msg)
        rows = []
        for spec in registry_datasets(slots.task):
            st = availability(spec, self.s.allow_downloads)
            n_models = len(self.store.list(owner=s.owner, dataset_id=spec.id))
            rows.append([spec.name, ", ".join(task_name(t) for t in spec.tasks if t in PREDICTABLE_TASKS),
                         fmt(spec.n_samples), {"ready": "ready", "downloadable": "downloads on first use",
                                               "not_cached": "not on this server"}[st], str(n_models)])
        return self._msg([text("Datasets the chatbot can train on and predict with (large datasets are disabled):"),
                          table(["Dataset", "Task", "Rows", "Status", "Trained models"], rows)],
                         ["Describe iris", "Train a random forest on breast cancer", "List available models"])

    def _i_describe_dataset(self, s, msg):
        if self._mentions_file(msg) and s.upload_id:
            return self._analysis(s, self.uploads.meta(s.sid, s.upload_id))
        slots = extract_slots(msg)
        if slots.dataset_id and get_dataset_spec(slots.dataset_id).size_class == "large":
            from src.chatbot.responses import describe_spec
            from src.data.loaders import dataset_status

            spec = get_dataset_spec(slots.dataset_id)
            return self._msg([text(describe_spec(spec, dataset_status(spec.id))),
                              notice("Large datasets are described here but disabled for training in the chatbot.")],
                             ["List datasets"])
        ds = self._dataset_from(s, msg)
        if not ds:
            return self._msg([text("Which dataset? For example “describe iris” or “describe diabetes”.")],
                             ["List datasets"], state="slot_filling")
        if ds["kind"] == "upload":
            return self._analysis(s, self.uploads.meta(s.sid, ds["upload_id"]))
        from src.chatbot.responses import describe_spec
        from src.data.loaders import dataset_status

        spec = get_dataset_spec(ds["id"])
        return self._msg([text(describe_spec(spec, dataset_status(spec.id)))],
                         [f"Preview {spec.id}", self._train_or_predict_chip(s, ds)])

    def _train_or_predict_chip(self, s, ds) -> str:
        models = self._models_for(s, ds)
        if models:
            return self._predict_chip(models[0])
        algo = K.DEFAULT_ALGO[ds["task"]][0]
        return f"Train {ALGORITHMS[algo].name} on {ds['name']}"

    def _i_preview(self, s, msg):
        ds = self._dataset_from(s, msg)
        if self._mentions_file(msg) and s.upload_id:
            m = self.uploads.meta(s.sid, s.upload_id)
            ds = {"kind": "upload", "upload_id": s.upload_id, "name": m["name"]}
        if not ds:
            return self._msg([text("Which dataset should I preview?")], ["Preview iris", "List datasets"])
        if ds["kind"] == "upload":
            df = self.uploads.frame(s.sid, ds["upload_id"])
            name = ds["name"]
        else:
            ld = load_registry(ds["id"], self.s.allow_downloads)
            df = ld.features.copy()
            if ld.target is not None and ld.target.name not in df.columns:
                df[ld.target.name] = ld.target.values
            name = ld.spec.name
        head = df.head(8)
        cols = list(map(str, head.columns))[:10]
        rows = [[_cell(v) for v in r] for r in head[cols].values.tolist()]
        return self._msg([text(f"First {len(head)} of {fmt(len(df))} rows of **{name}**"
                               + (f" (first 10 of {df.shape[1]} columns)" if df.shape[1] > 10 else "") + "."),
                          table(cols, rows)], [self._train_or_predict_chip(s, ds)])

    def _i_select_dataset(self, s, msg):
        ds = self._dataset_from(s, msg)
        models = self._models_for(s, ds)
        spec = get_dataset_spec(ds["id"])
        blocks = [text(f"**{spec.name}** — {spec.description}")]
        if models:
            s.model_id = models[0]["id"]
            blocks.append(text(f"{len(models)} trained model(s) are available for it; I'll use "
                               f"**{models[0]['algorithm_name']}** unless you name another."))
            sugg = [self._predict_chip(models[0]), "Show me important features.", "Compare available ML models."]
        else:
            sugg = [self._train_or_predict_chip(s, ds), f"Describe {spec.id}", "Compare available ML models."]
        return self._msg(blocks, sugg)

    # ------------------------------------------------------------------ intents: predict
    def _i_predict(self, s, msg):
        low = msg.lower()
        ds = self._dataset_from(s, msg)
        wants_file = bool(re.search(r"\b(my|this|uploaded|the)\s+(file|csv|upload)\b|\bmy data\b", low))
        if wants_file and s.upload_id and (not ds or ds["kind"] == "registry"):
            return self._batch_predict(s, msg, ds)
        if wants_file and not s.upload_id and not extract_slots(msg).dataset_id:
            return self._predict_options(s, True)
        if not ds:
            return self._predict_options(s, wants_file)
        algos, problems = self._requested_algos(msg, ds["task"])
        if problems:
            return self._msg([notice(" ".join(problems), "warning")],
                             [self._train_or_predict_chip(s, ds)], intent="predict")
        horizon = _horizon(msg) if ds["task"] == "forecasting" else None
        card = self._pick_model(s, ds, algos, horizon)
        if card is None:
            return self._offer_training(s, ds, algos, horizon, reason="predict")
        s.model_id = card["id"]
        if ds["task"] == "forecasting":
            return self._forecast(s, card, ds, horizon)
        values, notes = I.parse_feature_values(msg, card)
        if not values:
            return self._ask_features(s, card, {}, intro=True)
        return self._predict_one(s, card, values, notes, fill_missing=False)

    def _predict_options(self, s, wants_file: bool):
        models = self.store.list(owner=s.owner)
        blocks = [text("You can run a prediction in two ways:\n"
                       "- **Type the values**, e.g. “predict iris with petal length 1.4, petal width 0.2, sepal "
                       "length 5.1, sepal width 3.5”.\n"
                       "- **Upload a CSV** with the paperclip button: I'll check its columns and either score it with "
                       "a matching model or, if you ask, train a new model on it.")]
        if models:
            names = sorted({c["dataset_name"] for c in models})
            blocks.append(text("Models are ready for: " + ", ".join(f"**{n}**" for n in names) + "."))
        sugg = [self._predict_chip(c) for c in models[:3]] or ["Train a random forest on iris"]
        return self._msg(blocks, sugg + ["List available models"], intent="predict")

    def _ask_features(self, s, card, have: Dict[str, Any], intro: bool = False, missing: Optional[List[str]] = None):
        s.pending = {"kind": "missing_features", "model_id": card["id"], "values": dict(have)}
        feats = card["features"]
        need = missing or [f["name"] for f in feats]
        rows = []
        for f in feats:
            if f["name"] not in need:
                continue
            if f["type"] == "numeric":
                rows.append([f["name"], "number", f"{fmt(f['min'])} – {fmt(f['max'])}", fmt(f["median"])])
            else:
                rows.append([f["name"], "category", ", ".join(f.get("categories", [])[:6]), f.get("mode") or "—"])
        example = self._example_message(card)
        head = (f"To predict **{card['target']['name']}** with **{card['algorithm_name']}** "
                f"(trained on {card['dataset_name']}), I need {len(need)} value(s)."
                if intro else f"I still need {len(need)} value(s): " + ", ".join(need) + ".")
        blocks = [text(head)]
        if len(rows) <= 30:
            blocks.append(table(["Feature", "Type", "Training range / values", "Typical (median)"], rows))
        else:
            blocks.append(details(f"All {len(rows)} features", [table(["Feature", "Type", "Training range / values",
                                                                       "Typical (median)"], rows)]))
        blocks.append(text("Type them like `name = value`, separated by commas, or upload a CSV with these columns."))
        sugg = []
        if len(feats) <= 12:
            sugg.append(example)
        sugg.append("Use typical values for the rest" if have else "Use typical values for all features")
        sugg.append("Cancel")
        return self._msg(blocks, sugg, intent="predict", state="slot_filling")

    @staticmethod
    def _example_message(card) -> str:
        parts = []
        lead = "predict " + (card["dataset_id"].replace("_", " ") + " with " if not card["dataset_id"].startswith("upload:") else "")
        for f in card["features"][:12]:
            v = f.get("median") if f["type"] == "numeric" else f.get("mode")
            parts.append(f"{f['name']}={_plain(v)}" if f["type"] == "numeric" else f"{f['name']}={v}")
        return lead + ", ".join(parts)

    def _i_provide_values(self, s, msg):
        p = s.pending
        card = self.store.card(p["model_id"], s.owner)
        values, notes = I.parse_feature_values(msg, card)
        merged = {**p.get("values", {}), **values}
        return self._predict_one(s, card, merged, notes, fill_missing=False)

    def _i_fill_missing(self, s, msg):
        p = s.pending
        card = self.store.card(p["model_id"], s.owner)
        values, notes = I.parse_feature_values(msg, card)
        merged = {**p.get("values", {}), **values}
        return self._predict_one(s, card, merged, notes, fill_missing=True)

    def _predict_one(self, s, card, values, notes, fill_missing: bool):
        try:
            X, warnings = I.validate_row(values, card, fill_missing=fill_missing)
        except I.InputError as e:
            if e.details:
                s.pending = {"kind": "missing_features", "model_id": card["id"],
                             "values": {k: v for k, v in values.items() if k in {f['name'] for f in card['features']}}}
                return self._msg([notice("I couldn't use some of those inputs:", "warning"),
                                  text("\n".join("- " + d for d in e.details))]
                                 + ([text("Still missing: " + ", ".join(e.missing) + ".")] if e.missing else []),
                                 ["Cancel"], intent="predict", state="slot_filling")
            return self._ask_features(s, card, values, missing=e.missing)
        model = self.store.load(card["id"], s.owner)
        res = self._timed_call(lambda: I.predict_rows(model, card, X))
        s.pending = None
        s.model_id = card["id"]
        inputs = {k: _cell(v) for k, v in X.iloc[0].to_dict().items()}
        s.last_prediction = {"model_id": card["id"], "inputs": inputs, "result": res, "warnings": warnings}
        shown_inputs = [[k, "— (imputed)" if v is None else v] for k, v in inputs.items()]
        blocks = self._prediction_blocks(card, res, 0)
        for w in notes + warnings:
            blocks.append(notice(w, "warning" if "outside" in w or "not seen" in w or "filled" in w else "info"))
        blocks.append(details("Inputs used", [table(["Feature", "Value"], shown_inputs)]))
        blocks.append(self.model_block(card))
        return self._msg(blocks, ["Explain the prediction results.", "Show me important features.",
                                  "Compare available ML models."], intent="predict")

    def _prediction_blocks(self, card, res, i: int) -> List[Dict[str, Any]]:
        if card["task"] == "classification":
            label = res["labels"][i]
            block = {"type": "prediction", "task": "classification", "target": card["target"]["name"],
                     "label": label}
            out = []
            if res.get("probabilities"):
                probs = sorted(res["probabilities"][i].items(), key=lambda kv: -kv[1])
                block["probabilities"] = [{"label": k, "p": v} for k, v in probs]
                block["confidence"] = probs[0][1]
                ece = card["metrics"].get("ece")
                cal = (f"On held-out data this model's calibration error (ECE) is {fmt(ece, 3)}, so read the "
                       "probability as approximate rather than exact." if ece is not None else
                       "The probability is the model's own estimate; its calibration was not measured.")
                out.append(text(f"**{label}** with model probability {pct(probs[0][1])}. {cal}"))
            else:
                out.append(text(f"Predicted **{card['target']['name']}**: **{label}**. This model does not output "
                                "probabilities, so no confidence score is shown."))
            out.insert(0, block)
            return out
        v = res["values"][i]
        block = {"type": "prediction", "task": "regression", "target": card["target"]["name"], "value": v}
        line = f"Predicted **{card['target']['name']}**: **{fmt(v)}**."
        if res.get("intervals"):
            lo, hi = res["intervals"][i]
            block["interval"] = [lo, hi]
            block["interval_level"] = res["interval_level"]
            line += (f" The {int(res['interval_level'] * 100)}% interval ({fmt(lo)} to {fmt(hi)}) comes from the "
                     "model's errors on validation rows it was not trained on (split conformal), so about "
                     f"{int(res['interval_level'] * 100)}% of similar cases should fall inside it.")
        else:
            line += f" Typical held-out error (RMSE): {fmt(card['metrics'].get('rmse'))}."
        return [block, text(line)]

    def _timed_call(self, fn):
        if not PREDICT_SEM.acquire(timeout=self.s.predict_timeout_s):
            raise DataAccessError("The server is busy; please retry in a moment.")
        box: Dict[str, Any] = {}

        def run():
            try:
                box["v"] = fn()
            except Exception as e:  # pragma: no cover - surfaced below
                box["e"] = e
            finally:
                PREDICT_SEM.release()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(self.s.predict_timeout_s)
        if th.is_alive():
            raise DataAccessError(f"The prediction took longer than {self.s.predict_timeout_s} s and was abandoned.")
        if "e" in box:
            if isinstance(box["e"], I.InputError):
                raise box["e"]
            log.error("prediction failed", extra={"error": type(box["e"]).__name__})
            raise DataAccessError("The model failed while predicting on this input. Check the values, or retrain "
                                  "the model if the problem persists.")
        return box["v"]

    def _forecast(self, s, card, ds, horizon: Optional[int]):
        model = self.store.load(card["id"], s.owner)
        if ds["kind"] == "registry":
            ld = load_registry(ds["id"], self.s.allow_downloads)
            hist = ld.features.copy()
            if ld.time_index is not None:
                hist.index = ld.time_index
        else:
            df = self.uploads.frame(s.sid, ds["upload_id"])
            tc = ds["time_column"]
            df[tc] = pd.to_datetime(df[tc])
            hist = df.set_index(tc)
        res = self._timed_call(lambda: I.forecast_next(model, card, hist))
        H = min(horizon or res["horizon"], res["horizon"])
        steps, vals = res["steps"][:H], res["values"][:H]
        ints = res.get("intervals", [])[:H]
        s.last_prediction = {"model_id": card["id"], "inputs": {"history_rows": len(hist)}, "result": res,
                             "warnings": []}
        fc = card["forecast"]
        block = {"type": "forecast", "target": fc["target"], "steps": steps, "values": vals, "intervals": ints,
                 "level": res.get("interval_level"), "last_observed": res["last_observed"],
                 "history": [_cell(v) for v in hist[fc["target"]].tail(48).tolist()]}
        m = card["metrics"]
        line = (f"Forecast of **{fc['target']}** for the next {H} step(s) after the last observation "
                f"({fmt(res['last_observed'])}), using the latest {fc['window']} rows.")
        blocks = [text(line), block,
                  table(["Step", "Forecast"] + ([f"{int(res['interval_level'] * 100)}% interval"] if ints else []),
                        [[st, fmt(v)] + ([f"{fmt(a)} – {fmt(b)}"] if ints else [])
                         for st, v, (a, b) in zip(steps, vals, ints or [(None, None)] * H)])]
        if m.get("skill_vs_naive") is not None and m["skill_vs_naive"] < 0:
            blocks.append(notice(f"On held-out data this model's RMSE ({fmt(m.get('rmse'))}) is worse than the naive "
                                 f"“repeat the last value” baseline ({fmt(fc.get('naive_test_rmse'))}). Consider "
                                 "comparing models before relying on it.", "warning"))
        blocks.append(self.model_block(card))
        return self._msg(blocks, ["Explain the prediction results.", "Compare available ML models.",
                                  "Explain how this prediction works."], intent="predict")

    def _batch_predict(self, s, msg, ds):
        meta = self.uploads.meta(s.sid, s.upload_id)
        cols = {c["name"] for c in meta["columns"]}
        candidates = []
        pool = self._models_for(s, ds) if ds else self.store.list(owner=s.owner)
        for c in pool:
            if c["task"] == "forecasting":
                continue
            need = {f["name"] for f in c["features"]}
            if need <= cols:
                candidates.append(c)
        if not candidates:
            return self._msg([text(f"None of the trained models uses the columns in **{meta['name']}**, so I can't "
                                   "score it without training. If the file contains a target column, I can train a "
                                   "model on it — say e.g. “train on my file, target <column>”.")],
                             ["Describe my file", "List available models"], intent="predict")
        algos, _ = self._requested_algos(msg, candidates[0]["task"])
        card = next((c for c in candidates if c["algorithm_id"] in algos), None) or next(
            (c for c in candidates if c["id"] == s.model_id), candidates[0])
        df = self.uploads.frame(s.sid, s.upload_id)
        X, warnings = I.validate_batch(df, card)
        model = self.store.load(card["id"], s.owner)
        res = self._timed_call(lambda: I.predict_rows(model, card, X))
        out = df.copy()
        tname = f"predicted_{card['target']['name']}"
        if card["task"] == "classification":
            out[tname] = res["labels"]
            if res.get("probabilities"):
                out["probability"] = [max(p.values()) for p in res["probabilities"]]
        else:
            out[tname] = res["values"]
            if res.get("intervals"):
                out["interval_low"] = [a for a, _ in res["intervals"]]
                out["interval_high"] = [b for _, b in res["intervals"]]
        fid = "d_" + secrets.token_hex(6)
        s.downloads[fid] = (f"predictions_{Path(meta['name']).stem}.csv", out.to_csv(index=False).encode())
        while len(s.downloads) > 3:
            s.downloads.popitem(last=False)
        s.model_id = card["id"]
        s.last_prediction = {"model_id": card["id"], "inputs": {k: _cell(v) for k, v in X.iloc[0].to_dict().items()},
                             "result": {k: (v[:1] if isinstance(v, list) else v) for k, v in res.items()},
                             "warnings": warnings, "batch_rows": len(X)}
        show = [c for c in out.columns if c in (tname, "probability", "interval_low", "interval_high")]
        prev_cols = [c for c in list(df.columns)[:4]] + show
        rows = [[_cell(v) for v in r] for r in out[prev_cols].head(10).values.tolist()]
        blocks = [text(f"Scored **{fmt(len(X))}** row(s) of **{meta['name']}** with **{card['algorithm_name']}** "
                       f"(trained on {card['dataset_name']}). First 10 rows:"),
                  table(prev_cols, rows),
                  {"type": "download", "file_id": fid, "label": f"Download all {fmt(len(X))} predictions (CSV)"}]
        if card["task"] == "classification":
            counts = pd.Series(res["labels"]).value_counts()
            blocks.append(bars("Predicted classes", [{"label": str(k), "value": int(v)} for k, v in counts.items()]))
        blocks += [notice(w, "warning") for w in warnings]
        blocks.append(notice("“Explain the prediction results” will explain row 1 of your file.", "info"))
        blocks.append(self.model_block(card))
        return self._msg(blocks, ["Explain the prediction results.", "Show me important features."], intent="predict")

    # ------------------------------------------------------------------ intents: explain
    def _resolve_model_for_question(self, s, msg) -> Optional[Dict[str, Any]]:
        ds = self._dataset_from(s, msg, use_context=False)
        if ds:
            algos, _ = self._requested_algos(msg, ds["task"])
            c = self._pick_model(s, ds, algos)
            if c:
                s.model_id = c["id"]
            return c
        return self._card_or_none(s, s.model_id)

    def _i_importance(self, s, msg):
        c = self._resolve_model_for_question(s, msg)
        if not c:
            models = self.store.list(owner=s.owner)
            names = list(dict.fromkeys(m["dataset_name"] for m in models if m["task"] != "forecasting"))[:3]
            return self._msg([text("Which model? Feature importance is computed per trained model. Pick a dataset, or "
                                   "make a prediction first and I'll use that model.")],
                             [f"Show important features for {n}" for n in names] or ["Train a random forest on iris"],
                             intent="importance")
        if c["task"] == "forecasting":
            return self._msg([text(f"**{c['algorithm_name']}** forecasts from a window of past values of "
                                   f"{', '.join(c['forecast']['channels'])}. Per-feature permutation importance isn't "
                                   "computed for forecasting models in this version; the model card shows how it "
                                   "compares with the naive baseline."), self.model_block(c)],
                             ["Compare available ML models."], intent="importance")
        imp = c["importance"]
        items = [{"label": it["feature"], "value": it["mean"], "err": it["std"]} for it in imp["items"][:12]]
        top = [it for it in imp["items"] if (it["mean"] or 0) > 0][:3]
        summary = (f"For **{c['algorithm_name']}** on **{c['dataset_name']}**, the features the model relies on most are "
                   + ", ".join(f"**{it['feature']}**" for it in top) + "." if top else
                   f"No single feature changes **{c['algorithm_name']}**'s held-out score much when shuffled.")
        blocks = [text(summary),
                  bars(f"Permutation importance (drop in {imp['metric']})", items, note=imp["meaning"]),
                  notice("SHAP is not installed on this server. These importances are measured directly: each "
                         "feature is shuffled on the held-out test split and the loss in quality is recorded.", "info")]
        if c.get("native_importance"):
            ni = c["native_importance"]
            blocks.append(details(f"Model-internal view: {ni['method']}",
                                  [table(["Transformed feature", "Value"],
                                         [[it["feature"], fmt(it["value"])] for it in ni["items"]])]))
        blocks.append(self.model_block(c))
        return self._msg(blocks, [self._predict_chip(c), "Explain how this prediction works."], intent="importance")

    def _i_explain_result(self, s, msg, on_delta=None):
        lp = s.last_prediction
        if not lp:
            if s.last_compare:
                return self._explain_compare(s, on_delta)
            return self._msg([text("There's no prediction in this conversation yet, so there's nothing to explain. "
                                   "Make one first — I'll then show how each input pushed the result.")],
                             self._starter_chips(s), intent="explain_result")
        c = self._card_or_none(s, lp["model_id"])
        if not c:
            return self._msg([notice("The model behind the last prediction is no longer available.", "warning")],
                             ["List available models"])
        if c["task"] == "forecasting":
            res = lp["result"]
            m = c["metrics"]
            fact = {"model": c["algorithm_name"], "dataset": c["dataset_name"], "target": c["forecast"]["target"],
                    "last_observed": res["last_observed"], "forecast": res["values"], "horizon": res["horizon"],
                    "window": res["window"], "heldout_rmse": m.get("rmse"),
                    "naive_rmse": c["forecast"].get("naive_test_rmse"), "skill_vs_naive": m.get("skill_vs_naive")}
            base = (f"{c['algorithm_name']} read the last {res['window']} observations of every channel and produced "
                    f"{res['horizon']} future value(s) of {c['forecast']['target']}, starting from the last observed "
                    f"value {fmt(res['last_observed'])}. On the held-out (most recent) segment its RMSE was "
                    f"{fmt(m.get('rmse'))}, against {fmt(c['forecast'].get('naive_test_rmse'))} for the naive "
                    "“repeat the last value” forecast" + (" — so the naive baseline was more accurate there." if
                                                          (m.get("skill_vs_naive") or 0) < 0 else "."))
            out = self.llm.rephrase(fact, base, on_delta)
            return self._msg([text(out["text"]), self._llm_source(out), self.model_block(c)],
                             ["Compare available ML models."], intent="explain_result")
        if lp.get("batch_rows"):
            X = pd.DataFrame([lp["inputs"]])[c["feature_order"]]
        else:
            X = pd.DataFrame([lp["inputs"]])[c["feature_order"]]
        model = self.store.load(c["id"], s.owner)
        expl = self._timed_call(lambda: I.local_explanation(model, c, X))
        items = expl["items"][:10]
        res = lp["result"]
        if c["task"] == "classification":
            outcome = f"{c['target']['name']} = {res['labels'][0]}"
        else:
            outcome = f"{c['target']['name']} = {fmt(res['values'][0])}"
        ups = [it for it in items if it["effect"] > 0 and it["value"] is not None][:3]
        downs = [it for it in items if it["effect"] < 0 and it["value"] is not None][:2]
        base = (f"The model predicted {outcome}. " +
                ("The inputs that pushed the " + expl["output"] + " up the most were " +
                 ", ".join(f"{it['feature']} = {it['value']} (+{fmt(it['effect'], 3)} vs. its typical value "
                           f"{it['baseline']})" for it in ups) + ". " if ups else "") +
                ("Inputs that pulled it down: " + ", ".join(f"{it['feature']} = {it['value']} ({fmt(it['effect'], 3)})"
                                                          for it in downs) + ". " if downs else "") +
                "Each number is how much the output changes when only that feature is reset to its training median "
                "(or most frequent value).")
        facts = {"prediction": outcome, "output_measured": expl["output"], "current_output": expl["current"],
                 "effects": items, "heldout_metrics": {k: v for k, v in c["metrics"].items()
                                                       if k in ("accuracy", "f1", "rmse", "mae", "r2", "ece")},
                 "warnings": lp.get("warnings", [])}
        out = self.llm.rephrase(facts, base, on_delta)
        imputed = [it["feature"] for it in items if it["value"] is None]
        shown = [it for it in items if it["value"] is not None]
        blocks = [text(out["text"]), self._llm_source(out),
                  bars(f"Effect on {expl['output']} (vs. typical value)",
                       [{"label": f"{it['feature']} = {it['value']}", "value": it["effect"]} for it in shown],
                       note=expl["caveat"], signed=True),
                  notice("Not shown (value was missing and imputed with the training median, so it has no effect "
                         "here): " + ", ".join(imputed), "info") if imputed else None,
                  details("Method", [text(expl["method"] + ". " + expl["caveat"])])]
        if lp.get("warnings"):
            blocks.append(notice("Caveats from the input check: " + " ".join(lp["warnings"]), "warning"))
        blocks.append(self.model_block(c))
        return self._msg([b for b in blocks if b], ["Show me important features.", "Explain how this prediction works."],
                         intent="explain_result")

    def _llm_source(self, out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if out.get("source") == "llm":
            return {"type": "source", "text": "Wording by the configured language model; every number was checked "
                                              "against the model's actual output."}
        return None

    def _explain_compare(self, s, on_delta):
        lc = s.last_compare
        rows = lc["rows"]
        best = rows[0]
        base = (f"On {lc['dataset']}, {len(rows)} models were trained on the same split and scored on the same "
                f"held-out rows. {best['name']} ranked first by {lc['metric']} ({fmt(best['score'])}). Differences "
                "between models smaller than a few hundredths can come from the particular split, so also weigh speed, "
                "size and interpretability.")
        out = self.llm.rephrase({"comparison": lc}, base, on_delta)
        return self._msg([text(out["text"]), self._llm_source(out)], ["Compare available ML models."],
                         intent="explain_result")

    def _starter_chips(self, s) -> List[str]:
        models = self.store.list(owner=s.owner)
        return [self._predict_chip(c) for c in models[:3]] or ["Train a random forest on iris", "List datasets"]

    # ------------------------------------------------------------------ intents: train / compare (explicit)
    def _offer_training(self, s, ds, algos, horizon, reason: str):
        task = ds["task"]
        algo = algos[0] if algos else K.DEFAULT_ALGO[task][0]
        why = "" if algos else f" {K.DEFAULT_ALGO[task][1]}."
        s.pending = {"kind": "confirm_train", "dataset": ds, "algorithm_id": algo, "horizon": horizon}
        what = f"**{ALGORITHMS[algo].name}**" + (f" (horizon {horizon})" if horizon else "")
        kind = f"{ALGORITHMS[algo].name} model" if algos else "model" + (f" with a {horizon}-step horizon" if horizon else "")
        lead = (f"There's no trained {kind} for **{ds['name']}** yet" if reason == "predict"
                else f"Ready to train on **{ds['name']}**")
        return self._msg([text(f"{lead}. I only train when you ask. "
                               f"Shall I train {what} on it now?{why} Training uses the training split only and the "
                               "model is scored on held-out data; it is kept for this session.")],
                         ["Yes", "No", "List available models"], state="confirming", intent="predict")

    def _i_confirm(self, s, msg):
        p, s.pending = s.pending, None
        if not p:
            return self._msg([text("There's nothing waiting for confirmation.")])
        if p["kind"] == "confirm_train":
            return self._start_training(s, p["dataset"], p["algorithm_id"], p.get("horizon"))
        if p["kind"] == "confirm_compare":
            return self._start_compare(s, p["dataset"], p["algorithms"])
        if p["kind"] == "missing_features":
            s.pending = p
            return self._i_fill_missing(s, "use typical values for the rest")
        return self._msg([text("Okay.")])

    def _i_train(self, s, msg):
        ds = self._dataset_from(s, msg)
        if not ds:
            if s.upload_id:
                return self._upload_setup_prompt(s, self.uploads.meta(s.sid, s.upload_id))
            return self._msg([text("Which dataset should I train on? Name one (e.g. “train a random forest on iris”) or "
                                   "upload a CSV.")], ["List datasets", "Train a random forest on iris"],
                             state="slot_filling")
        algos, problems = self._requested_algos(msg, ds["task"])
        if problems:
            return self._msg([notice(" ".join(problems), "warning")], ["List algorithms"], intent="train")
        algo = algos[0] if algos else K.DEFAULT_ALGO[ds["task"]][0]
        horizon = _horizon(msg) if ds["task"] == "forecasting" else None
        blocks = [] if algos else [text(f"No algorithm named, so I'll use **{ALGORITHMS[algo].name}**. "
                                        f"{K.DEFAULT_ALGO[ds['task']][1]}.")]
        out = self._start_training(s, ds, algo, horizon)
        out["blocks"] = blocks + out["blocks"]
        return out

    def _dataset_job_spec(self, s, ds) -> Dict[str, Any]:
        if ds["kind"] == "registry":
            return {"kind": "registry", "dataset_id": ds["id"], "allow_downloads": self.s.allow_downloads}
        return {"kind": "upload", "upload_root": str(self.s.upload_dir), "sid": s.sid, "upload_id": ds["upload_id"],
                "task": ds["task"], "target": ds["target"], "time_column": ds.get("time_column")}

    def _start_training(self, s, ds, algo_id: str, horizon: Optional[int]):
        if not self.jobs_allowed(s, train=True):
            raise DataAccessError(f"Training is limited to {self.s.train_per_hour} jobs per hour per session.")
        if len([c for c in self.store.list(owner=s.owner) if c.get("owner") == s.owner]) >= MAX_PRIVATE_MODELS:
            raise DataAccessError(f"This session already has {MAX_PRIVATE_MODELS} private models. Clear the chat to "
                                  "start fresh.")
        # validate in-process before queuing (fast; gives immediate, specific errors)
        if ds["kind"] == "registry":
            ld = load_registry(ds["id"], self.s.allow_downloads)
        else:
            ld = self.uploads.as_dataset(s.sid, ds["upload_id"], ds["task"], ds["target"], ds.get("time_column"))
        algo = check_trainable(algo_id, ld, ds["task"], self.s.max_train_rows, self.s.allow_neural)
        workdir = self.s.model_dir / "_tmp"
        workdir.mkdir(parents=True, exist_ok=True)
        art = workdir / f"{secrets.token_hex(8)}.bin"
        job_spec = {"kind": "train", "project_root": str(PROJECT_ROOT), "dataset": self._dataset_job_spec(s, ds),
                    "task": ds["task"], "algorithm_id": algo.id, "max_train_rows": self.s.max_train_rows,
                    "allow_neural": self.s.allow_neural, "epochs": self.s.neural_epochs, "horizon": horizon,
                    "artifact_path": str(art)}
        owner = s.owner
        timeout = self.s.train_timeout_s

        def work(progress, cancelled):
            try:
                meta = run_sandboxed(job_spec, timeout, progress, cancelled)
                if ds["kind"] == "upload":
                    meta["dataset_id"] = self._ds_key(ds)
                    meta["dataset_name"] = ds["name"]
                return self.store.save(art, meta, owner=owner)
            finally:
                if art.exists():
                    art.unlink()

        job = self.jobs.submit(s.sid, "train", f"Training {algo.name} on {ds['name']}", work,
                               on_done=lambda j: self._job_done(s, j))
        return self._msg([text(f"Training **{algo.name}** on **{ds['name']}**. I'll post the held-out results here "
                               "when it finishes.")], ["Cancel"], intent="train", job=job)

    def jobs_allowed(self, s: Session, train: bool = False) -> bool:
        return True if not train else self._train_limiter_ok(s)

    def _train_limiter_ok(self, s: Session) -> bool:
        lim = getattr(self, "train_limiter", None)
        return True if lim is None else lim.allow(f"train:{s.sid}")

    def _i_compare(self, s, msg):
        ds = self._dataset_from(s, msg)
        if not ds:
            models = self.store.list(owner=s.owner)
            if not models:
                return self._msg([text("There are no trained models yet. Name a dataset and I can train and compare "
                                       "several algorithms on it, e.g. “compare models on breast cancer”.")],
                                 ["Compare models on iris", "List datasets"], intent="compare")
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for c in models:
                groups.setdefault(c["dataset_name"], []).append(c)
            rows = []
            for name, cs in groups.items():
                for c in cs:
                    rows.append(self._compare_row(c))
            return self._msg([text("Trained models by dataset (held-out test metrics). Scores are only comparable "
                                   "within the same dataset:"),
                              table(["Dataset", "Model", "Metric", "Score", "Train time"], rows)],
                             [f"Compare models on {n}" for n in list(groups)[:3]], intent="compare")
        models = self._models_for(s, ds)
        if len(models) >= 2 and not re.search(r"\b(benchmark|all|train|run|full)\b", msg.lower()):
            fps = {(c["split"].get("fingerprint") or str(c["split"].get("n_test_windows"))) for c in models}
            rows = [self._compare_row(c)[1:] for c in models]
            rows.sort(key=lambda r: r[2] if isinstance(r[2], float) else 0, reverse=ds["task"] == "classification")
            note = ("All models share the same split, so the scores are directly comparable." if len(fps) == 1 else
                    "These models were trained on different splits, so small differences are not meaningful.")
            return self._msg([text(f"Trained models for **{ds['name']}** (held-out test metrics):"),
                              table(["Model", "Metric", "Score", "Train time"], [[r[0], r[1], fmt(r[2]), r[3]] for r in rows]),
                              notice(note, "info")],
                             [f"Benchmark all recommended models on {ds['name']}", "Show me important features."],
                             intent="compare")
        if ds["kind"] != "registry":
            return self._msg([text("Comparisons run on registry datasets. For your file, train models one at a time "
                                   "(e.g. “train ridge on my file, target price”) and then ask to compare them.")],
                             intent="compare")
        rec = recommend_algorithms(ds["id"], ds["task"], allow_large=False)
        algos = [a for a in rec["recommended"] if ALGORITHMS[a].framework != "pytorch" or self.s.allow_neural]
        algos = [a for a in algos if ALGORITHMS[a].cost != "high"][:8]
        s.pending = {"kind": "confirm_compare", "dataset": ds, "algorithms": algos}
        names = ", ".join(ALGORITHMS[a].name for a in algos)
        return self._msg([text(f"I can train and compare **{len(algos)}** algorithms on **{ds['name']}** using one "
                               f"shared split and seed: {names}. This trains models (they're evaluated, not saved). "
                               "Run it?")], ["Yes", "No"], intent="compare", state="confirming")

    def _compare_row(self, c) -> List[Any]:
        m = c["metrics"]
        if c["task"] == "classification":
            key, val = "F1", m.get("f1")
        else:
            key, val = "RMSE", m.get("rmse")
        return [c["dataset_name"], c["algorithm_name"], key, val, f"{fmt(c.get('train_time_s'), 2)} s"]

    def _start_compare(self, s, ds, algos: List[str]):
        if not self._train_limiter_ok(s):
            raise DataAccessError(f"Training is limited to {self.s.train_per_hour} jobs per hour per session.")
        job_spec = {"kind": "benchmark", "project_root": str(PROJECT_ROOT), "dataset": self._dataset_job_spec(s, ds),
                    "task": ds["task"], "algorithms": algos, "max_train_rows": self.s.max_train_rows,
                    "epochs": self.s.neural_epochs}
        timeout = self.s.train_timeout_s

        def work(progress, cancelled):
            return run_sandboxed(job_spec, timeout, progress, cancelled)

        job = self.jobs.submit(s.sid, "benchmark", f"Comparing {len(algos)} models on {ds['name']}", work,
                               on_done=lambda j: self._job_done(s, j))
        return self._msg([text(f"Comparing {len(algos)} models on **{ds['name']}**…")], ["Cancel"], intent="compare",
                         job=job)

    # ------------------------------------------------------------------ job completion
    def _job_done(self, s: Session, j: Job) -> None:
        with s.lock:
            if j.state != "done":
                reason = j.error or j.state
                msg = self._msg([notice(f"{j.label} did not finish: {reason}", "error" if j.state == "failed" else "warning")],
                                ["List available models", "What can you do?"], intent=j.kind, state="error")
            elif j.kind == "train":
                c = j.result
                s.model_id = c["id"]
                msg = self._trained_message(c)
            else:
                msg = self._benchmark_message(s, j.result)
            msg["job"] = j.public()
            msg["job_final"] = True
            j.reply = msg
            self._log(s, msg)

    def _trained_message(self, c) -> Dict[str, Any]:
        m = c["metrics"]
        if c["task"] == "classification":
            line = f"held-out accuracy {fmt(m.get('accuracy'))}, F1 {fmt(m.get('f1'))}"
        elif c["task"] == "regression":
            line = f"held-out RMSE {fmt(m.get('rmse'))}, R² {fmt(m.get('r2'))}"
        else:
            line = (f"held-out RMSE {fmt(m.get('rmse'))} vs. naive baseline {fmt(c['forecast'].get('naive_test_rmse'))}")
        blocks = [text(f"Trained **{c['algorithm_name']}** on **{c['dataset_name']}** in {fmt(c.get('train_time_s'), 2)} s "
                       f"— {line}.")]
        if c["task"] == "forecasting" and (m.get("skill_vs_naive") or 0) < 0:
            blocks.append(notice("This model is less accurate than the naive baseline on held-out data.", "warning"))
        blocks.append(self.model_block(c))
        if c["task"] == "forecasting":
            sugg = [self._predict_chip(c), "Explain how this prediction works.", "Compare available ML models."]
        else:
            sugg = [self._example_message(c) if len(c["features"]) <= 12 else self._predict_chip(c),
                    "Show me important features.", "Compare available ML models."]
        return self._msg(blocks, sugg, intent="train")

    def _benchmark_message(self, s, res) -> Dict[str, Any]:
        from src.evaluation.task_metrics import PRIMARY_METRIC

        task = res["task"]
        key, mode = PRIMARY_METRIC[task]
        ok = [m for m in res["models"] if m.get("status") == "ok" and m["metrics"].get(key) is not None]
        ok.sort(key=lambda m: m["metrics"][key], reverse=(mode == "max"))
        failed = [m for m in res["models"] if m.get("status") != "ok"]
        rows = []
        for i, m in enumerate(ok, 1):
            mm = m["metrics"]
            extra = (fmt(mm.get("accuracy")) if task == "classification" else
                     fmt(mm.get("r2")) if task == "regression" else fmt(mm.get("mase")))
            rows.append([i, m["name"], fmt(mm[key]), extra, f"{fmt(m.get('latency_us_per_sample'), 1)} µs",
                         f"{fmt(m.get('model_size_kb'), 1)} KB"])
        second = {"classification": "Accuracy", "regression": "R²", "forecasting": "MASE"}.get(task, "—")
        name = get_dataset_spec(res["dataset_id"]).name
        s.last_compare = {"dataset": name, "metric": key, "rows": [{"name": m["name"], "score": m["metrics"][key]}
                                                                   for m in ok]}
        blocks = [text(f"Compared **{len(ok)}** models on **{name}** with one shared split "
                       f"({fmt(res['split'].get('n_train'))} train / {fmt(res['split'].get('n_test'))} test rows). "
                       f"Ranked by held-out **{key.upper()}** ({'higher' if mode == 'max' else 'lower'} is better)."),
                  table(["#", "Model", key.upper(), second, "Latency", "Size"], rows),
                  bars(f"Held-out {key.upper()}", [{"label": m["name"], "value": m["metrics"][key]} for m in ok]),
                  notice("The most accurate model isn't automatically the best: differences of a few hundredths can "
                         "come from this particular split, and speed, size and interpretability matter too.", "info")]
        if failed:
            blocks.append(notice("Failed: " + "; ".join(f"{m['name']} ({m.get('error', '')[:80]})" for m in failed),
                                 "warning"))
        top = ok[0]["algorithm"] if ok else None
        sugg = ["Explain the prediction results."]
        if top:
            sugg.insert(0, f"Train {ALGORITHMS[top].name} on {name}")
        return self._msg(blocks, sugg, intent="compare")

    def job_status(self, sid: str, job_id: str) -> Optional[Dict[str, Any]]:
        j = self.jobs.get(job_id, sid)
        if j is None:
            return None
        return {"job": j.public(), "reply": j.reply}

    # ------------------------------------------------------------------ uploads
    def handle_upload(self, sid: str, filename: str, raw: bytes) -> Dict[str, Any]:
        s = self.session(sid)
        with s.lock:
            self._log(s, {"id": "u_" + secrets.token_hex(5), "role": "user", "ts": round(time.time(), 3),
                          "blocks": [text(f"📎 {Path(filename).name[:80]}")]})
            try:
                meta = self.uploads.save(sid, raw, filename)
            except DataAccessError as e:
                out = self._msg([notice(f"I couldn't use that file: {e}", "error"),
                                 text("Upload a UTF-8 CSV with a header row, at least 20 rows, at most "
                                      f"{self.s.max_upload_bytes // (1024 * 1024)} MB.")], intent="upload", state="error")
                self._log(s, out)
                return out
            s.upload_id = meta["id"]
            s.pending = {"kind": "upload_setup", "upload_id": meta["id"]}
            out = self._analysis(s, meta)
            self._log(s, out)
            return out

    def _analysis(self, s, meta) -> Dict[str, Any]:
        cols = meta["columns"]
        rows = []
        for c in cols[:40]:
            summary = (f"{fmt(c.get('min'))} – {fmt(c.get('max'))}" if c["type"] == "numeric" else
                       ", ".join(c.get("top", [])) if c["type"] == "categorical" else "dates")
            rows.append([c["name"], c["type"], f"{c['missing_pct']}%", fmt(c["unique"]), summary])
        rep = meta["report"]
        issues = []
        if rep.get("missing_cells"):
            issues.append(f"{fmt(rep['missing_cells'])} missing cell(s) — they will be imputed using training rows only.")
        if rep.get("duplicate_rows"):
            issues.append(f"{fmt(rep['duplicate_rows'])} duplicate row(s) — dropped before any split.")
        if rep.get("renamed_columns"):
            issues.append(f"{len(rep['renamed_columns'])} column name(s) were sanitised.")
        names = {c["name"] for c in cols}
        matches = [c for c in self.store.list(owner=s.owner)
                   if c["task"] != "forecasting" and {f["name"] for f in c["features"]} <= names]
        blocks = [text(f"Received **{meta['name']}**: {fmt(meta['rows'])} rows × {len(cols)} columns. Nothing has been "
                       "trained or predicted yet."),
                  table(["Column", "Type", "Missing", "Unique", "Range / top values"], rows)]
        blocks += [notice(i, "info") for i in issues]
        sugg = []
        if matches:
            c = matches[0]
            blocks.append(text(f"Its columns match **{c['algorithm_name']}** trained on **{c['dataset_name']}**, so I can "
                               "score every row with it."))
            sugg.append(f"Predict with my file using {c['algorithm_name']}")
        numeric = [c["name"] for c in cols if c["type"] == "numeric"]
        guess = cols[-1]["name"] if cols else None
        blocks.append(text("To **train** on it, tell me the target column, e.g. "
                           f"“train on my file, target {guess}”. Add “as classification” or “as regression” to "
                           "override my guess of the task. For forecasting, also give the time column."))
        if guess:
            sugg.append(f"Train on my file, target {guess}")
        sugg.append("Preview my file")
        del numeric
        return self._msg(blocks, sugg, intent="upload", state="slot_filling")

    def _i_analyze(self, s, msg):
        if s.upload_id:
            return self._analysis(s, self.uploads.meta(s.sid, s.upload_id))
        return self._msg([text("Use the 📎 button next to the message box to upload a CSV (UTF-8, header row, up to "
                               f"{self.s.max_upload_bytes // (1024 * 1024)} MB). I'll profile its columns, check missing "
                               "values and types, and tell you whether a trained model can score it.")],
                         ["What can you do?"], intent="analyze")

    def _upload_setup_prompt(self, s, meta):
        return self._analysis(s, meta)

    def _i_upload_setup(self, s, msg):
        uid = (s.pending or {}).get("upload_id") or s.upload_id
        if not uid:
            return self._i_analyze(s, msg)
        meta = self.uploads.meta(s.sid, uid)
        cols = [c["name"] for c in meta["columns"]]
        target = _match_column(msg, cols, r"(?:target|label|predict(?:ing)?)\s*(?:column)?\s*(?:is|=|:)?\s*")
        time_col = _match_column(msg, cols, r"(?:time|date|timestamp)\s*(?:column)?\s*(?:is|=|:)?\s*")
        if not target:
            return self._msg([text("Which column should the model predict? Say e.g. “target "
                                   f"{cols[-1]}”. Columns: " + ", ".join(cols[:30]) + ".")],
                             [f"Train on my file, target {c}" for c in cols[-3:]], state="slot_filling")
        info = next(c for c in meta["columns"] if c["name"] == target)
        low = msg.lower()
        if "forecast" in low or time_col:
            task = "forecasting"
            if not time_col:
                dt = [c["name"] for c in meta["columns"] if c["type"] == "datetime"]
                time_col = dt[0] if dt else None
            if not time_col:
                return self._msg([text("Forecasting needs a time column. Say e.g. “time column timestamp”.")],
                                 state="slot_filling")
        elif "classif" in low:
            task = "classification"
        elif "regress" in low:
            task = "regression"
        else:
            task = ("regression" if info["type"] == "numeric" and info["unique"] > 20 else "classification")
        ds = {"kind": "upload", "id": uid, "upload_id": uid, "name": f"your file ({meta['name']})", "task": task,
              "target": target, "time_column": time_col}
        s.dataset = ds
        s.pending = None
        slots_algos, problems = self._requested_algos(msg, task)
        if problems:
            return self._msg([notice(" ".join(problems), "warning")], [])
        algo = slots_algos[0] if slots_algos else K.DEFAULT_ALGO[task][0]
        guess_note = "" if ("classif" in low or "regress" in low or task == "forecasting") else (
            f" I treated this as **{task}** because '{target}' "
            + ("is numeric with many distinct values." if task == "regression" else "has few distinct values / is categorical.")
            + " Say “as regression” or “as classification” to change it.")
        out = self._start_training(s, ds, algo, _horizon(msg) if task == "forecasting" else None)
        out["blocks"].insert(0, text(f"Target: **{target}**.{guess_note}"))
        return out

    def download(self, sid: str, file_id: str) -> Optional[Tuple[str, bytes]]:
        s = self.session(sid)
        return s.downloads.get(file_id)


# =============================================================================
# Module helpers
# =============================================================================

def _cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v if isinstance(v, (int, str, bool)) else str(v)


def _plain(v: Any) -> str:
    if v is None:
        return "0"
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:.6g}"


def _rank_key(c: Dict[str, Any]) -> float:
    m = c.get("metrics", {})
    if c["task"] == "classification":
        v = m.get("f1")
        return -(v if v is not None else -1)
    v = m.get("rmse")
    return v if v is not None else float("inf")


def _horizon(msg: str) -> Optional[int]:
    m = re.search(r"(\d{1,3})\s*(?:steps?|hours?|periods?|points?)\b(?:\s*ahead)?|horizon\s*(?:=|of)?\s*(\d{1,3})|next\s+(\d{1,3})",
                  msg.lower())
    if not m:
        return None
    v = int(next(g for g in m.groups() if g))
    if not 1 <= v <= 48:
        raise DataAccessError("The forecast horizon must be between 1 and 48 steps.")
    return v


def _match_column(msg: str, cols: List[str], prefix: str) -> Optional[str]:
    low = msg.lower()
    norm = {re.sub(r"[^a-z0-9]+", " ", c.lower()).strip(): c for c in cols}
    m = re.search(prefix + r"[\"'`]?([^,;\"'`]+)", low)
    if not m:
        return None
    tail = re.sub(r"[^a-z0-9]+", " ", m.group(1)).strip()
    best = None
    for n, c in sorted(norm.items(), key=lambda kv: -len(kv[0])):
        if n and (tail == n or tail.startswith(n + " ") or tail.startswith(n)):
            best = c
            break
    return best
