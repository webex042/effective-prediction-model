"""
Vercel Serverless Function Handler for EPM
==========================================
Provides lightweight, zero-dependency REST endpoints for the EPM
Interactive Research Dashboard. Fully compatible with Vercel Serverless
Python Runtime (@vercel/python).
"""

from http.server import BaseHTTPRequestHandler
import json
import time
import math
import random
import sys
import os

# Project Metadata
PROJECT_INFO = {
    "name": "EPM — Efficient Predictive Multimodal Generative Model",
    "version": "0.1.0",
    "status": "operational",
    "stages_completed": 8,
    "stages_total": 14,
    "unit_tests_passing": 205,
    "numerical_gradient_error": 1.4158e-09,
    "research_question": "Can efficient generative and multimodal models achieve comparable or better predictive performance than conventional ML/DL models while reducing inference latency and memory?",
}

BENCHMARK_DATA = {
    "stage2_classification": [
        {"model": "Logistic Regression (Scratch)", "accuracy": 0.9125, "f1": 0.8293, "auroc": 0.9667, "latency_us": 0.33, "params": "Linear"},
        {"model": "Logistic Regression (Scikit)", "accuracy": 0.9125, "f1": 0.8293, "auroc": 0.9664, "latency_us": 1.45, "params": "Linear"},
        {"model": "SVC (RBF)", "accuracy": 0.8917, "f1": 0.7797, "auroc": 0.9541, "latency_us": 36.35, "params": "Kernel"},
        {"model": "XGBoost", "accuracy": 0.8792, "f1": 0.7680, "auroc": 0.9353, "latency_us": 10.29, "params": "Ensemble"},
        {"model": "KNN (k=5)", "accuracy": 0.8750, "f1": 0.7619, "auroc": 0.9348, "latency_us": 27.60, "params": "Non-parametric"},
        {"model": "Decision Tree", "accuracy": 0.8542, "f1": 0.7250, "auroc": 0.8279, "latency_us": 0.35, "params": "Tree (depth=5)"}
    ],
    "stage5_timeseries": [
        {"model": "Linear Regression", "rmse": 2.833, "mae": 2.246, "r2": 0.926, "latency_us": 12.38, "parameters": 726},
        {"model": "Random Forest", "rmse": 6.719, "mae": 4.835, "r2": 0.584, "latency_us": 41.45, "parameters": 25000},
        {"model": "XGBoost", "rmse": 7.970, "mae": 5.927, "r2": 0.414, "latency_us": 26.89, "parameters": 18000},
        {"model": "Transformer Forecaster", "rmse": 8.973, "mae": 6.908, "r2": 0.257, "latency_us": 64.47, "parameters": 17478}
    ],
    "stage6_multimodal": [
        {"model": "Sensor-Only", "parameters": 136003, "clean_f1": 1.0, "noisy_sensor_f1": 0.5631, "missing_text_f1": 1.0, "missing_sensor_f1": 0.1728},
        {"model": "Text-Only", "parameters": 69315, "clean_f1": 1.0, "noisy_sensor_f1": 1.0, "missing_text_f1": 0.1667, "missing_sensor_f1": 1.0},
        {"model": "Multimodal-Concat", "parameters": 215811, "clean_f1": 1.0, "noisy_sensor_f1": 0.5654, "missing_text_f1": 1.0, "missing_sensor_f1": 1.0},
        {"model": "Multimodal-Gated (GMU)", "parameters": 219843, "clean_f1": 1.0, "noisy_sensor_f1": 0.5420, "missing_text_f1": 1.0, "missing_sensor_f1": 1.0},
        {"model": "Multimodal-CrossAttn", "parameters": 249347, "clean_f1": 1.0, "noisy_sensor_f1": 0.5401, "missing_text_f1": 1.0, "missing_sensor_f1": 1.0}
    ],
    "stage7_generative": [
        {"model": "Standard-AE", "recon_mse": 0.03076, "recon_mae": 0.11798, "auroc": 1.0, "latency_ms": 4.68, "parameters": 20672},
        {"model": "Conditional-VAE", "recon_mse": 0.03265, "recon_mae": 0.12435, "auroc": 1.0, "latency_ms": 3.02, "parameters": 21224},
        {"model": "Variational-VAE", "recon_mse": 0.16204, "recon_mae": 0.34906, "auroc": 0.927, "latency_ms": 4.55, "parameters": 20936}
    ]
}

STAGES_INFO = [
    {"stage": 0, "title": "Environment & Infrastructure", "status": "Completed", "desc": "Config system, deterministic seed control, reproducible logging, scratch metrics"},
    {"stage": 1, "title": "Mathematical Foundations", "status": "Completed", "desc": "Linear algebra, calculus, probability distributions, GD, Momentum, Adam from scratch"},
    {"stage": 2, "title": "Classical Machine Learning", "status": "Completed", "desc": "Linear/Logistic Regression, KNN, Decision Trees from scratch vs Scikit/XGBoost baseline"},
    {"stage": 3, "title": "Neural Networks from Scratch", "status": "Completed", "desc": "Forward/backprop, custom activations, XOR non-linear decision boundary, PyTorch parity"},
    {"stage": 4, "title": "NLP, Embeddings & Transformers", "status": "Completed", "desc": "Text preprocessing, vocabulary, scratch embeddings, Scaled Dot-Product Attention"},
    {"stage": 5, "title": "Time Series Prediction", "status": "Completed", "desc": "Multivariate telemetry, sliding windows, LSTM, GRU, Temporal Transformer forecasting"},
    {"stage": 6, "title": "Multimodal Representation & Fusion", "status": "Completed", "desc": "Continuous telemetry + discrete text, Concat, GMU gating, Cross-Attention fusion"},
    {"stage": 7, "title": "Generative Modeling & Latent Space", "status": "Completed", "desc": "AE, VAE, Conditional-VAE, ELBO derivation, epistemic uncertainty quantification"},
    {"stage": 8, "title": "Diffusion Models (DDPM)", "status": "In Development", "desc": "Forward/reverse SDE diffusion process on controlled datasets"},
    {"stage": 9, "title": "Diffusion for Time Series", "status": "Planned", "desc": "Probabilistic trajectory forecasting with conditional noise schedule"},
    {"stage": 10, "title": "Flow Matching", "status": "Planned", "desc": "Continuous normalizing flows and optimal transport velocity fields"},
    {"stage": 11, "title": "Efficiency Optimization", "status": "Planned", "desc": "Linear attention, FlashAttention benchmarks, computational complexity profiling"},
    {"stage": 12, "title": "Model Compression", "status": "Planned", "desc": "Knowledge distillation, INT8/INT4 quantization, structured pruning"},
    {"stage": 13, "title": "Research Synthesis & Paper", "status": "Planned", "desc": "Hypothesis testing, statistical ablation study, empirical paper"}
]

class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function Handler"""

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        
        if path in ("", "/api", "/api/health"):
            data = {
                **PROJECT_INFO,
                "server_time": time.time(),
                "python_runtime": f"Python {sys.version.split()[0]}",
                "platform": sys.platform
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

        elif path == "/api/benchmarks":
            self._set_headers(200)
            self.wfile.write(json.dumps(BENCHMARK_DATA, indent=2).encode("utf-8"))

        elif path == "/api/stages":
            self._set_headers(200)
            self.wfile.write(json.dumps(STAGES_INFO, indent=2).encode("utf-8"))

        elif path == "/api/test-status":
            test_summary = {
                "total_collected": 205,
                "passed": 205,
                "failed": 0,
                "skipped": 0,
                "pass_rate_pct": 100.0,
                "modules": [
                    {"name": "test_math_foundations.py", "tests": 74, "status": "PASSED"},
                    {"name": "test_classical_ml.py", "tests": 24, "status": "PASSED"},
                    {"name": "test_neural_networks.py", "tests": 12, "status": "PASSED"},
                    {"name": "test_nlp.py", "tests": 14, "status": "PASSED"},
                    {"name": "test_timeseries.py", "tests": 11, "status": "PASSED"},
                    {"name": "test_multimodal.py", "tests": 13, "status": "PASSED"},
                    {"name": "test_generative.py", "tests": 15, "status": "PASSED"},
                    {"name": "test_metrics.py", "tests": 16, "status": "PASSED"},
                    {"name": "test_reproducibility.py", "tests": 7, "status": "PASSED"},
                    {"name": "test_config.py", "tests": 10, "status": "PASSED"},
                    {"name": "test_experiment.py", "tests": 9, "status": "PASSED"}
                ]
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(test_summary, indent=2).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Endpoint not found: {path}"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
        
        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        path = self.path.split("?")[0].rstrip("/")

        if path == "/api/benchmark-matrix":
            # Real matrix multiply benchmark in pure Python
            dim = int(body.get("dim", 120))
            dim = max(20, min(dim, 300))  # bound for serverless responsiveness
            
            t0 = time.perf_counter()
            # Generate deterministic matrices
            random.seed(42)
            A = [[random.random() for _ in range(dim)] for _ in range(dim)]
            B = [[random.random() for _ in range(dim)] for _ in range(dim)]
            
            # Multiply A x B
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
            gflops = (flops / elapsed) / 1e9 if elapsed > 0 else 0

            response = {
                "dimension": f"{dim}x{dim}",
                "elapsed_seconds": round(elapsed, 4),
                "total_flops": flops,
                "gflops": round(gflops, 3),
                "sample_val": round(C[0][0], 4)
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(response).encode("utf-8"))

        elif path == "/api/calculate-metrics":
            y_true = body.get("y_true", [1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
            y_pred = body.get("y_pred", [1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
            
            n = len(y_true)
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            
            acc = (tp + tn) / n if n > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            res = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Endpoint not found: {path}"}).encode("utf-8"))
