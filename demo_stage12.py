"""
EPM Stage 12 — Model Compression Demo
======================================

Demonstrates all Stage 12 modules:
    1. Knowledge Distillation: Hinton Soft Targets & Temperature Scaling (T=4.0)
    2. Quantization: Uniform Affine SQNR & Dynamic INT8 Post-Training Quantization
    3. Pruning: Unstructured Magnitude Pruning vs Structured Neuron Contraction
    4. Tri-Compression Pareto Frontier: Distilled + Structured Pruned + INT8
    5. Publication-Quality Figures & Structured Benchmark Artifacts

Run:
    python demo_stage12.py
"""

import sys
import os
import time
import json
import shutil
import copy
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reconfigure stdout for UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.reproducibility import set_seed
from src.logger import setup_logging
from src.compression.distillation import (
    KnowledgeDistillationLoss,
    DistillationTrainer,
)
from src.compression.quantization import (
    quantize_dequantize,
    compute_sqnr,
    get_model_size_kb,
    quantize_model_dynamic_int8,
)
from src.compression.pruning import (
    calculate_sparsity,
    prune_unstructured_magnitude,
    StructuredMLPCompressor,
)
from src.compression.benchmarks import (
    BenchmarkClassifier,
    generate_synthetic_industrial_data,
    measure_inference_latency,
    evaluate_accuracy,
    count_parameters,
    run_stage12_benchmarks,
)

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage12")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
WEB_PLOTS_DIR = Path("web/plots/stage12")
WEB_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR = Path("benchmarks")
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("  EPM - Stage 12: Model Compression (Distillation, Quantization, Pruning)")
print("  Knowledge Transfer, Dynamic INT8 PTQ & Physical Structured Contraction")
print("=" * 78)

# Monochrome retro styling
plt.rcParams.update({
    "font.family": "monospace",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#000000",
    "axes.linewidth": 1.2,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.color": "#cccccc",
    "grid.linestyle": ":",
})


# =====================================================================
# Step 1: Execute Full Compression Benchmark Suite
# =====================================================================
print("\n[Step 1] Executing Compression Pipeline Benchmark Across All Variants...")
benchmark_res = run_stage12_benchmarks(
    in_dim=32, num_classes=4, epochs_teacher=20, epochs_student=20, seed=42
)
records = benchmark_res["benchmark_records"]

print("\n" + "=" * 106)
print(f"{'Variant':<34} | {'Params':<8} | {'Size (KB)':<10} | {'Reduct':<8} | {'Accuracy':<10} | {'Retention':<10} | {'Latency (us)':<12} | {'Speedup'}")
print("-" * 106)
for r in records:
    print(
        f"{r['variant']:<34} | {r['parameters']:<8} | {r['size_kb']:<10.1f} | {r['size_reduction']:<8} | "
        f"{r['accuracy']:<10.4f} | {r['acc_retention']:<10} | {r['latency_us']:<12.2f} | {r['speedup']}"
    )
print("=" * 106)


# =====================================================================
# Step 2: Detailed Temperature Distillation Analysis
# =====================================================================
print("\n[Step 2] Profiling Soft Probability Entropy across Temperatures T in [1, 2, 4, 8, 16]...")
example_logits = torch.tensor([[4.5, 1.2, -0.8, -3.2]])
temperatures = [1.0, 2.0, 4.0, 8.0, 16.0]
soft_probs = []
entropies = []

for T in temperatures:
    p = F.softmax(example_logits / T, dim=-1).squeeze(0).numpy()
    soft_probs.append(p)
    ent = -np.sum(p * np.log(p + 1e-12))
    entropies.append(ent)
    print(f"  • Temperature T = {T:<4.1f} -> Probabilities: {np.round(p, 3)} | Entropy: {ent:.3f}")


# =====================================================================
# Step 3: Quantization Bit-Width SQNR Sweep
# =====================================================================
print("\n[Step 3] Profiling Signal-to-Quantization-Noise Ratio (SQNR) across Bit-Widths...")
test_weight = torch.randn(1000, 32)
bit_widths = [2, 3, 4, 6, 8, 10, 12, 16]
sqnrs = []
for b in bit_widths:
    _, sqnr_val = quantize_dequantize(test_weight, num_bits=b, symmetric=True)
    sqnrs.append(sqnr_val)
    print(f"  • Bit-Width = {b:>2} bits -> SQNR = {sqnr_val:>6.2f} dB (Theoretical ~ {6.02 * b + 1.76:.1f} dB)")


# =====================================================================
# Step 4: Pruning Sparsity Sweep (Unstructured vs Structured)
# =====================================================================
print("\n[Step 4] Sweeping Sparsity Levels from 0% to 85% on Student Model...")
train_ds, test_ds = generate_synthetic_industrial_data(num_samples=1000, in_dim=32, num_classes=4, seed=42)
base_model = BenchmarkClassifier(32, [32, 16], 4)
# Train base student quickly
opt = torch.optim.Adam(base_model.parameters(), lr=0.005)
loader = DataLoader(train_ds, batch_size=32, shuffle=True)
for _ in range(12):
    for x, y in loader:
        opt.zero_grad()
        out, _ = base_model(x)
        F.cross_entropy(out, y).backward()
        opt.step()

sparsity_levels = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85]
unstructured_accs = []
for sp in sparsity_levels:
    m = copy.deepcopy(base_model)
    if sp > 0.0:
        prune_unstructured_magnitude(m, sparsity=sp)
    acc = evaluate_accuracy(m, test_ds)
    unstructured_accs.append(acc)
    print(f"  • Sparsity = {sp * 100:>4.1f}% -> Accuracy = {acc:.4f}")


# =====================================================================
# Step 5: Generate Publication-Quality Monochrome Plots
# =====================================================================
print("\n[Step 5] Generating Publication-Quality Figures...")

# -------------------------------------------------------------
# Figure 1: Knowledge Distillation Temperature Softening
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# Soft probabilities bar chart
x_indices = np.arange(4)
width = 0.16
for i, (T, p) in enumerate(zip(temperatures, soft_probs)):
    offset = (i - 2) * width
    ax1.bar(x_indices + offset, p, width=width, label=f"T={T}", edgecolor="#000000", alpha=0.8 - i * 0.12)

ax1.set_title("Hinton Soft Targets across Temperature T", fontsize=10, fontweight="bold")
ax1.set_xlabel("Class Index")
ax1.set_ylabel("Softmax Probability")
ax1.set_xticks(x_indices)
ax1.set_xticklabels([f"Class {c}" for c in range(4)])
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True)

# Shannon Entropy vs Temperature
ax2.plot(temperatures, entropies, marker="s", color="#000000", linewidth=1.5, markersize=6)
ax2.set_title("Probability Distribution Entropy vs Temperature T", fontsize=10, fontweight="bold")
ax2.set_xlabel("Distillation Temperature (T)")
ax2.set_ylabel("Shannon Entropy (nats)")
for t_val, ent_val in zip(temperatures, entropies):
    ax2.annotate(f"{ent_val:.2f}", (t_val, ent_val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax2.grid(True)

plt.tight_layout()
fig1_path = PLOTS_DIR / "knowledge_distillation_temperature_and_loss.png"
fig.savefig(fig1_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig1_path}")

# -------------------------------------------------------------
# Figure 2: Quantization Weight Distribution & SQNR
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# Histogram of weights (Original vs INT8 Reconstructed)
w_flat = test_weight.numpy().flatten()
w_recon, _ = quantize_dequantize(test_weight, num_bits=8, symmetric=True)
w_recon_flat = w_recon.numpy().flatten()

ax1.hist(w_flat, bins=40, density=True, color="#888888", alpha=0.5, label="Original FP32", edgecolor="#000000")
ax1.hist(w_recon_flat, bins=40, density=True, histtype="step", linewidth=1.5, color="#000000", label="Quantized INT8 Reconstructed")
ax1.set_title("Weight Distribution: FP32 vs Uniform INT8", fontsize=10, fontweight="bold")
ax1.set_xlabel("Weight Value")
ax1.set_ylabel("Probability Density")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True)

# SQNR vs Bit-Width
ax2.plot(bit_widths, sqnrs, marker="o", color="#000000", linewidth=1.5, markersize=6, label="Measured SQNR")
theo_sqnr = [6.02 * b + 1.76 for b in bit_widths]
ax2.plot(bit_widths, theo_sqnr, linestyle="--", color="#666666", label="Theoretical Uniform Limit (6.02b + 1.76)")
ax2.set_title("Signal-to-Quantization-Noise Ratio vs Bit-Width", fontsize=10, fontweight="bold")
ax2.set_xlabel("Quantization Bit-Width (b)")
ax2.set_ylabel("SQNR (dB)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True)

plt.tight_layout()
fig2_path = PLOTS_DIR / "quantization_weight_distribution_and_sqnr.png"
fig.savefig(fig2_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig2_path}")

# -------------------------------------------------------------
# Figure 3: Pruning Sparsity vs Accuracy
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
sp_pct = [s * 100 for s in sparsity_levels]
ax.plot(sp_pct, unstructured_accs, marker="^", color="#000000", linewidth=1.5, markersize=7, label="Unstructured Magnitude Pruning")
ax.axhline(benchmark_res["teacher_accuracy"], color="#666666", linestyle=":", label=f"Teacher Accuracy ({benchmark_res['teacher_accuracy']:.3f})")
ax.set_title("Model Accuracy Degradation vs Parameter Sparsity", fontsize=10, fontweight="bold")
ax.set_xlabel("Pruning Sparsity (%)")
ax.set_ylabel("Test Accuracy")
ax.set_ylim(0.0, 1.05)
for x_val, y_val in zip(sp_pct, unstructured_accs):
    ax.annotate(f"{y_val:.3f}", (x_val, y_val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.legend(loc="lower left", fontsize=8)
ax.grid(True)

plt.tight_layout()
fig3_path = PLOTS_DIR / "pruning_sparsity_vs_accuracy_tradeoff.png"
fig.savefig(fig3_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig3_path}")

# -------------------------------------------------------------
# Figure 4: Model Compression Pareto Frontier
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# Memory Size (KB) vs Accuracy
sizes = [r["size_kb"] for r in records]
accs = [r["accuracy"] for r in records]
names = [r["variant"].split(" (")[0] for r in records]

ax1.scatter(sizes, accs, color="#000000", s=60, edgecolor="#000000", zorder=3)
ax1.set_title("Pareto Trade-off: Model Size vs Accuracy", fontsize=10, fontweight="bold")
ax1.set_xlabel("Model Checkpoint Size (KB) — Log Scale")
ax1.set_ylabel("Test Accuracy")
ax1.set_xscale("log")
for s_val, a_val, n_val in zip(sizes, accs, names):
    ax1.annotate(n_val, (s_val, a_val), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax1.grid(True)

# Latency vs Accuracy
lats = [r["latency_us"] for r in records]
ax2.scatter(lats, accs, color="#000000", s=60, edgecolor="#000000", zorder=3)
ax2.set_title("Pareto Trade-off: Inference Latency vs Accuracy", fontsize=10, fontweight="bold")
ax2.set_xlabel("Latency per Sample (us)")
ax2.set_ylabel("Test Accuracy")
for l_val, a_val, n_val in zip(lats, accs, names):
    ax2.annotate(n_val, (l_val, a_val), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax2.grid(True)

plt.tight_layout()
fig4_path = PLOTS_DIR / "model_compression_pareto_frontier.png"
fig.savefig(fig4_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig4_path}")

# Copy all plots to web directory
for p in [fig1_path, fig2_path, fig3_path, fig4_path]:
    shutil.copy(p, WEB_PLOTS_DIR / p.name)


# =====================================================================
# Step 6: Save Benchmark & Experiment Artifacts
# =====================================================================
benchmark_path = BENCHMARKS_DIR / "stage12_model_compression.json"
with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_res, f, indent=2)
print(f"\n[Step 6] Saved Benchmark Metrics -> {benchmark_path}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_path = EXPERIMENTS_DIR / f"{timestamp}_stage12_model_compression.json"
exp_record = {
    "timestamp": timestamp,
    "stage": 12,
    "name": "Model Compression & Edge Deployment Evaluation",
    "benchmark_records": records,
    "max_size_reduction": benchmark_res["max_size_reduction"],
    "max_speedup": benchmark_res["max_speedup"],
    "plots_generated": [str(p.name) for p in [fig1_path, fig2_path, fig3_path, fig4_path]],
}
with open(exp_path, "w", encoding="utf-8") as f:
    json.dump(exp_record, f, indent=2)
print(f"[Step 7] Logged Experiment Record -> {exp_path}")

print("\n" + "=" * 78)
print("  STAGE 12 EXECUTION COMPLETE — MODEL COMPRESSION BENCHMARKED")
print("  Distillation, Quantization, Pruning & Edge Deployment Analyzed.")
print("=" * 78)
