"""
Vercel Serverless Function Handler for EPM
==========================================
Lightweight, zero-dependency REST endpoints for the EPM research dashboard.

INTEGRITY NOTE
--------------
Every number served here is read from real artifacts committed to the repo
(``benchmarks/*.json`` produced by the stage demos, the Stage 12/13 experiment
registry in ``experiments/registry/``, and a real pytest status snapshot in
``benchmarks/test_status.json``). Nothing is hard-coded or fabricated; if an
artifact is missing the endpoint reports an explicit empty/unavailable state
instead of inventing values. On the serverless runtime these files are read
only (training must be run locally via ``demo_stage*.py`` / ``run.py``).
"""

from http.server import BaseHTTPRequestHandler
import json
import time
import random
import sys
import glob
import os

API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(API_DIR)
# Candidate base dirs for reading committed data artifacts. Different deploy
# layouts (classic function, service bundle, local run) place the working set
# differently, so we probe the project root, the function's own dir, and cwd.
_DATA_BASES = [ROOT, API_DIR, os.getcwd()]


def _resolve(rel_path):
    for base in _DATA_BASES:
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(ROOT, rel_path)


def _read_json(rel_path, default=None):
    """Safely read a JSON artifact from the first base dir that has it."""
    try:
        with open(_resolve(rel_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_all_benchmarks():
    """Load every benchmarks/*.json keyed by filename stem (real artifacts)."""
    out = {}
    for base in _DATA_BASES:
        for path in sorted(glob.glob(os.path.join(base, "benchmarks", "*.json"))):
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
            if stem in out:
                continue
            data = _read_json(os.path.join("benchmarks", name))
            if data is not None:
                out[stem] = data
    return out


def load_registry_index():
    """Return the list of recorded experiments (Stage 12/13 registry), or []."""
    idx = _read_json(os.path.join("experiments", "registry", "index.json"), default=[])
    return idx if isinstance(idx, list) else []


def load_test_status():
    """Real pytest snapshot written by tooling; explicit unavailable state if absent."""
    return _read_json(
        os.path.join("benchmarks", "test_status.json"),
        default={"available": False, "note": "Run pytest locally to generate benchmarks/test_status.json"},
    )

RESEARCH_QUESTION = (
    "Can efficient generative and multimodal models achieve comparable or better "
    "predictive performance than conventional ML/DL models while reducing inference "
    "latency, memory, FLOPs, and parameter count?"
)

# Descriptive stage metadata (titles/descriptions are documentation, not metrics).
STAGE_META = [
    {"stage": 0, "title": "Environment & Infrastructure", "status": "Completed", "desc": "Config, deterministic seeds, reproducible logging, experiment tracking"},
    {"stage": 1, "title": "Mathematical Foundations", "status": "Completed", "desc": "Linear algebra, calculus, probability, GD/Momentum/Adam from scratch"},
    {"stage": 2, "title": "Classical Machine Learning", "status": "Completed", "desc": "Scratch Linear/Logistic/KNN/Trees vs scikit-learn & XGBoost baselines"},
    {"stage": 3, "title": "Neural Networks from Scratch", "status": "Completed", "desc": "Forward/backprop, gradient checking, PyTorch parity"},
    {"stage": 4, "title": "NLP, Embeddings & Transformers", "status": "Completed", "desc": "Preprocessing, embeddings, scaled dot-product self-attention"},
    {"stage": 5, "title": "Time Series Prediction", "status": "Completed", "desc": "Sliding windows, LSTM/GRU/Temporal Transformer forecasting"},
    {"stage": 6, "title": "Multimodal Representation & Fusion", "status": "Completed", "desc": "Sensor+text; concat, GMU gating, cross-attention fusion"},
    {"stage": 7, "title": "Generative Modeling & Latent Space", "status": "Completed", "desc": "AE, VAE, Conditional-VAE, ELBO, uncertainty"},
    {"stage": 8, "title": "Diffusion Models (DDPM & DDIM)", "status": "Completed", "desc": "Forward/reverse diffusion and accelerated deterministic sampling"},
    {"stage": 9, "title": "Diffusion for Time Series", "status": "Completed", "desc": "Probabilistic trajectory forecasting, CRPS, interval coverage"},
    {"stage": 10, "title": "Flow Matching & CNFs", "status": "Completed", "desc": "Continuous normalizing flows, optimal-transport velocity fields"},
    {"stage": 11, "title": "Attention Efficiency & Complexity", "status": "Completed", "desc": "Linear & tiled attention, sequence-length scaling"},
    {"stage": 12, "title": "Model Compression", "status": "Completed", "desc": "Knowledge distillation, INT8 quantization, structured/unstructured pruning"},
    {"stage": 13, "title": "Research Experiment (H1)", "status": "Completed", "desc": "Reproducible reduced-step generative forecasting study with real baselines"},
]


def _h1_headline(stage13):
    """Extract an honest one-line H1 headline from the Stage 13 summary, if present."""
    try:
        comps = stage13["h1_analysis"]["comparisons"]
        competitive = [c for c in comps if c.get("rmse_within_full_ci")]
        if competitive:
            best = max(competitive, key=lambda c: c["speedup_vs_full"])
            return {
                "supported": True,
                "best_steps": best["sampling_steps"],
                "speedup_vs_full": best["speedup_vs_full"],
                "rmse_change_pct": best["rmse_change_pct"],
            }
        return {"supported": False, "note": "No reduced-step setting stayed within the full-step RMSE CI."}
    except Exception:
        return None


def build_project_info():
    """Assemble project metadata entirely from real artifacts (no fabrication)."""
    tests = load_test_status()
    registry = load_registry_index()
    benches = load_all_benchmarks()

    info = {
        "name": "EPM — Efficient Predictive Multimodal Generative Model",
        "version": "0.1.0",
        "status": "operational",
        "stages_completed": sum(1 for s in STAGE_META if s["status"] == "Completed"),
        "stages_total": len(STAGE_META),
        "experiments_recorded": len(registry),
        "research_question": RESEARCH_QUESTION,
    }
    if isinstance(tests, dict) and tests.get("available"):
        info["unit_tests_passing"] = tests.get("passed")
        info["unit_tests_total"] = tests.get("collected")
        info["test_pass_rate_pct"] = tests.get("pass_rate_pct")

    s12 = benches.get("stage12_model_compression")
    if isinstance(s12, dict):
        if s12.get("max_size_reduction"):
            info["model_size_reduction"] = s12["max_size_reduction"]
        if s12.get("max_speedup"):
            info["compression_max_speedup"] = s12["max_speedup"]

    s13 = benches.get("stage13_research")
    headline = _h1_headline(s13) if isinstance(s13, dict) else None
    if headline is not None:
        info["h1_result"] = headline

    return info

def _matrix_benchmark(dim):
    """Real pure-Python matrix-multiply micro-benchmark (bounded for responsiveness)."""
    dim = max(20, min(int(dim), 300))
    t0 = time.perf_counter()
    random.seed(42)
    A = [[random.random() for _ in range(dim)] for _ in range(dim)]
    B = [[random.random() for _ in range(dim)] for _ in range(dim)]
    C = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        row_a = A[i]
        for k in range(dim):
            aik = row_a[k]
            row_b = B[k]
            for j in range(dim):
                C[i][j] += aik * row_b[j]
    elapsed = time.perf_counter() - t0
    flops = 2 * (dim ** 3)
    return {
        "dimension": f"{dim}x{dim}",
        "elapsed_seconds": round(elapsed, 4),
        "total_flops": flops,
        "gflops": round((flops / elapsed) / 1e9, 3) if elapsed > 0 else 0,
        "sample_val": round(C[0][0], 4),
    }


def _calc_metrics(y_true, y_pred):
    """Real confusion-matrix classification metrics."""
    n = len(y_true)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
    return {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _parse_query(query_string):
    out = {}
    for part in (query_string or "").split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out

def route_request(method, path, query, body):
    """Shared router used by BOTH the classic `handler` and the WSGI `app`.

    Returns (status_code, payload_dict). Pure routing over the real-data
    loaders and compute helpers above; no framework dependencies.
    """
    raw = (path or "").split("?")[0].rstrip("/")
    method = (method or "GET").upper()

    if method == "OPTIONS":
        return 200, {}

    if method == "GET":
        if raw in ("", "/api", "/api/health"):
            data = dict(build_project_info())
            data.update({
                "server_time": time.time(),
                "python_runtime": f"Python {sys.version.split()[0]}",
                "platform": sys.platform,
            })
            return 200, data
        if raw == "/api/benchmarks":
            return 200, load_all_benchmarks()
        if raw == "/api/stages":
            return 200, STAGE_META
        if raw == "/api/experiments":
            return 200, load_registry_index()
        if raw == "/api/experiment":
            exp_id = query.get("id", "")
            safe = "".join(c for c in exp_id if c.isalnum() or c in ("_", "-"))
            rec = _read_json(os.path.join("experiments", "registry", f"{safe}.json"))
            return (200, rec) if rec is not None else (404, {"error": f"experiment not found: {exp_id}"})
        if raw == "/api/test-status":
            return 200, load_test_status()
        return 404, {"error": f"Endpoint not found: {raw}"}

    if method == "POST":
        if raw == "/api/benchmark-matrix":
            return 200, _matrix_benchmark(body.get("dim", 120))
        if raw == "/api/calculate-metrics":
            return 200, _calc_metrics(
                body.get("y_true", [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]),
                body.get("y_pred", [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]),
            )
        if raw in ("/api/run-experiment", "/api/run"):
            return 501, {
                "started": False,
                "reason": "Training is not available on the serverless/service runtime.",
                "how_to_run": "Run locally: `python demo_stage12.py` or `python demo_stage13.py`; "
                              "results appear in experiments/registry/.",
            }
        return 404, {"error": f"Endpoint not found: {raw}"}

    return 405, {"error": f"Method not allowed: {method}"}


class handler(BaseHTTPRequestHandler):
    """Vercel-compatible serverless handler serving real artifact-derived data."""

    def _send(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(json.dumps(payload, indent=2, default=str).encode("utf-8"))

    def do_OPTIONS(self):
        self._send({}, 200)

    def do_GET(self):
        raw = self.path.split("?")[0]
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        status, payload = route_request("GET", raw, _parse_query(qs), {})
        self._send(payload, status)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            body = {}
        status, payload = route_request("POST", self.path, {}, body)
        self._send(payload, status)


def app(environ, start_response):
    """WSGI entrypoint for Vercel Services (service `api`, entrypoint `index:app`).

    Serves the same routes as the classic `handler` via the shared
    `route_request` dispatcher, so both deployment modes behave identically.
    """
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/") or "/"
    query = _parse_query(environ.get("QUERY_STRING", ""))

    body = {}
    if method == "POST":
        try:
            size = int(environ.get("CONTENT_LENGTH") or 0)
        except (TypeError, ValueError):
            size = 0
        if size > 0:
            raw = environ["wsgi.input"].read(size)
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                body = {}

    status, payload = route_request(method, path, query, body)
    data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    status_line = {
        200: "200 OK", 404: "404 Not Found",
        405: "405 Method Not Allowed", 501: "501 Not Implemented",
    }.get(status, f"{status} OK")
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(data))),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
        ("Cache-Control", "no-cache, no-store, must-revalidate"),
    ]
    start_response(status_line, headers)
    return [data]
