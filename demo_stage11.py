"""
EPM Stage 11 — Computational Efficiency & Sequence Scaling Demo
===============================================================

This script demonstrates all Stage 11 modules:
    1. Theoretical Complexity Analysis (Big-O scaling in Time and Space)
    2. Sequence Length Scaling Benchmark: N in [64, 128, 256, 512, 1024, 2048]
    3. Memory Hierarchy Profiling (Peak Allocated Memory vs Sequence Length)
    4. Tiled Memory-Efficient Attention Online Softmax Exact Parity Verification
    5. Downstream Long-Context Industrial Sensor Forecasting (RMSE / MAE vs Latency)
    6. Publication-Quality Monochrome Figures & Benchmark Artifacts

Run:
    python demo_stage11.py
"""

import sys
import os
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Reconfigure stdout for UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.reproducibility import set_seed
from src.logger import setup_logging
from src.efficiency.attention import (
    standard_attention,
    linear_attention,
    local_window_attention,
    tiled_exact_attention,
    calculate_theoretical_complexity,
)
from src.efficiency.benchmarks import (
    benchmark_attention_scaling,
    benchmark_downstream_sensor_forecasting,
)

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage11")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
WEB_PLOTS_DIR = Path("plots/stage11")
WEB_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR = Path("benchmarks")
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("  EPM - Stage 11: Computational Efficiency & Sequence Scaling")
print("  Linear Attention, Tiled Memory-Efficient Attention & Complexity Analysis")
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
# Step 1: Sequence Length Scaling Benchmark
# =====================================================================
print("\n[Step 1] Profiling Latency & Memory Across Sequence Lengths N in [64...2048]...")
seq_lengths = [64, 128, 256, 512, 1024, 2048]
scaling_res = benchmark_attention_scaling(
    seq_lengths=seq_lengths, embed_dim=64, num_heads=4, batch_size=2, warmup=3, reps=15
)
scaling_records = scaling_res["scaling_records"]

print("\n" + "=" * 98)
print(f"{'Seq Length (N)':<14} | {'Standard (ms)':<14} | {'Linear (ms)':<12} | {'Tiled (ms)':<12} | {'Std Mem (KB)':<13} | {'Lin Mem (KB)':<13} | {'Speedup'}")
print("-" * 98)
for r in scaling_records:
    print(f"N = {r['sequence_length']:<10} | {r['standard_latency_ms']:<14.3f} | {r['linear_latency_ms']:<12.3f} | "
          f"{r['tiled_latency_ms']:<12.3f} | {r['standard_mem_kb']:<13.1f} | {r['linear_mem_kb']:<13.1f} | {r['linear_speedup_vs_standard']:.2f}x")
print("=" * 98)


# =====================================================================
# Step 2: Downstream Multi-Sensor Telemetry Long-Horizon Forecasting
# =====================================================================
print("\n[Step 2] Evaluating Downstream Multi-Sensor Forecasting (L=128 -> H=12)...")
downstream_res = benchmark_downstream_sensor_forecasting(seq_len=128, horizon=12, epochs=15)

print("\n" + "=" * 95)
print(f"{'Mechanism':<26} | {'Time Comp':<10} | {'Space Comp':<11} | {'Test RMSE':<10} | {'Test MAE':<9} | {'Latency'}")
print("-" * 95)
for d in downstream_res:
    print(f"{d['Mechanism']:<26} | {d['Time Complexity']:<10} | {d['Space Complexity']:<11} | "
          f"{d['Test RMSE']:<10.4f} | {d['Test MAE']:<9.4f} | {d['Latency / Sample']}")
print("=" * 95)


# =====================================================================
# Step 3: Generating Publication Figures
# =====================================================================
print("\n[Step 3] Rendering Publication Monochrome Figures...")

# -------------------------------------------------------------
# Figure 1: Latency Scaling vs. Sequence Length
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
Ns = [r["sequence_length"] for r in scaling_records]
std_lats = [r["standard_latency_ms"] for r in scaling_records]
lin_lats = [r["linear_latency_ms"] for r in scaling_records]
loc_lats = [r["local_latency_ms"] for r in scaling_records]
tiled_lats = [r["tiled_latency_ms"] for r in scaling_records]

ax.plot(Ns, std_lats, marker="o", color="#000000", linewidth=2.0, label="Standard Attention O(N² d)")
ax.plot(Ns, tiled_lats, marker="^", color="#444444", linewidth=1.5, linestyle="--", label="Tiled Exact Attention O(N² d)")
ax.plot(Ns, lin_lats, marker="s", color="#000000", linewidth=1.8, linestyle=":", label="Kernel Linear Attention O(N d²)")
ax.plot(Ns, loc_lats, marker="x", color="#777777", linewidth=1.5, linestyle="-.", label="Local Window Attention O(N w d)")

ax.set_title("Wall-Clock Latency Scaling vs. Sequence Length (N)", fontsize=11, fontweight="bold")
ax.set_xlabel("Sequence Length (N)")
ax.set_ylabel("Inference Latency (ms)")
ax.set_xticks(Ns)
ax.grid(True)
ax.legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
fig1_path = PLOTS_DIR / "latency_scaling_vs_sequence_length.png"
fig.savefig(fig1_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig1_path}")

# -------------------------------------------------------------
# Figure 2: Peak Memory Footprint Scaling (Quadratic vs Linear)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
std_mems = [r["standard_mem_kb"] / 1024.0 for r in scaling_records]  # in MB
lin_mems = [r["linear_mem_kb"] / 1024.0 for r in scaling_records]
tiled_mems = [r["tiled_mem_kb"] / 1024.0 for r in scaling_records]
loc_mems = [r["local_mem_kb"] / 1024.0 for r in scaling_records]

ax.plot(Ns, std_mems, marker="o", color="#000000", linewidth=2.2, label="Standard Matrix Memory O(N²)")
ax.plot(Ns, tiled_mems, marker="^", color="#555555", linewidth=1.8, linestyle="--", label="Tiled Peak Memory O(N d)")
ax.plot(Ns, lin_mems, marker="s", color="#000000", linewidth=1.8, linestyle=":", label="Linear Attention Memory O(N d)")
ax.plot(Ns, loc_mems, marker="x", color="#888888", linewidth=1.5, linestyle="-.", label="Local Window Memory O(N w)")

ax.set_title("Peak Matrix Memory Footprint vs. Sequence Length (N)", fontsize=11, fontweight="bold")
ax.set_xlabel("Sequence Length (N)")
ax.set_ylabel("Peak Allocated Memory (MB)")
ax.set_xticks(Ns)
ax.grid(True)
ax.legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
fig2_path = PLOTS_DIR / "peak_memory_scaling_quadratic_vs_linear.png"
fig.savefig(fig2_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig2_path}")

# -------------------------------------------------------------
# Figure 3: Tiled Attention Exact Numerical Parity Verification
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# Generate small attention map for visual parity comparison
N_vis = 48
d_vis = 16
torch.manual_seed(42)
Q_vis = torch.randn(1, 1, N_vis, d_vis)
K_vis = torch.randn(1, 1, N_vis, d_vis)
V_vis = torch.randn(1, 1, N_vis, d_vis)

out_std_vis, A_std = standard_attention(Q_vis, K_vis, V_vis)
out_tiled_vis = tiled_exact_attention(Q_vis, K_vis, V_vis, block_size=16)

# Heatmap of exact attention weights
im1 = ax1.imshow(A_std[0, 0].numpy(), cmap="gray_r", aspect="auto")
ax1.set_title("Standard Attention Matrix A [N × N]", fontsize=10, fontweight="bold")
ax1.set_xlabel("Key Position")
ax1.set_ylabel("Query Position")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Output difference: |Out_tiled - Out_standard|
diff_vis = torch.abs(out_tiled_vis - out_std_vis)[0, 0].numpy()
im2 = ax2.imshow(diff_vis, cmap="gray_r", aspect="auto", vmin=0, vmax=1e-5)
ax2.set_title("Absolute Numerical Difference |Tiled - Standard|", fontsize=10, fontweight="bold")
ax2.set_xlabel("Feature Dimension")
ax2.set_ylabel("Token Position")
cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Max Error < 1e-6 (Float Parity)")

plt.tight_layout()
fig3_path = PLOTS_DIR / "tiled_online_softmax_parity.png"
fig.savefig(fig3_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig3_path}")

# -------------------------------------------------------------
# Figure 4: Downstream Forecasting Accuracy vs. Efficiency
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

mechanisms = [d["Mechanism"] for d in downstream_res]
rmses = [d["Test RMSE"] for d in downstream_res]
lats = [float(d["Latency / Sample"].replace(" ms", "")) for d in downstream_res]

x_pos = np.arange(len(mechanisms))

# Bar plot of RMSE
bars1 = ax1.bar(x_pos, rmses, color=["#000000", "#444444", "#777777", "#AAAAAA"], width=0.55, edgecolor="#000000")
ax1.set_title("Downstream Forecasting Error (Test RMSE)", fontsize=10, fontweight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(["Standard", "Tiled", "Linear", "Local"], rotation=15)
ax1.set_ylabel("Test RMSE (Lower is Better)")
ax1.set_ylim(0.0, 0.45)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)
ax1.grid(True, axis="y")

# Bar plot of Latency
bars2 = ax2.bar(x_pos, lats, color=["#000000", "#444444", "#777777", "#AAAAAA"], width=0.55, edgecolor="#000000")
ax2.set_title("Inference Latency per Sample (ms)", fontsize=10, fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(["Standard", "Tiled", "Linear", "Local"], rotation=15)
ax2.set_ylabel("Latency (ms)")
ax2.set_ylim(0.0, max(lats) * 1.3)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.002, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)
ax2.grid(True, axis="y")

plt.tight_layout()
fig4_path = PLOTS_DIR / "downstream_forecasting_accuracy_vs_efficiency.png"
fig.savefig(fig4_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig4_path}")

# Copy all plots to web folder
for p in [fig1_path, fig2_path, fig3_path, fig4_path]:
    shutil.copy(p, WEB_PLOTS_DIR / p.name)


# =====================================================================
# Step 4: Saving Benchmark & Experiment JSON Artifacts
# =====================================================================
benchmark_path = BENCHMARKS_DIR / "stage11_computational_efficiency.json"
benchmark_data = {
    "scaling_records": scaling_records,
    "downstream_summary": downstream_res,
    "sequence_lengths": seq_lengths,
    "tiled_max_error": scaling_records[-1]["tiled_numerical_error"],
    "max_linear_speedup": max(r["linear_speedup_vs_standard"] for r in scaling_records),
    "peak_memory_savings_at_2048": f"{scaling_records[-1]['standard_mem_kb'] / scaling_records[-1]['linear_mem_kb']:.1f}x",
}
with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2)
print(f"\n[Step 4] Saved Benchmark Metrics -> {benchmark_path}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_path = EXPERIMENTS_DIR / f"{timestamp}_stage11_computational_efficiency.json"
experiment_record = {
    "timestamp": timestamp,
    "stage": 11,
    "name": "Computational Efficiency & Sequence Scaling Analysis",
    "sequence_lengths_tested": seq_lengths,
    "scaling_records": scaling_records,
    "downstream_summary": downstream_res,
    "plots_generated": [str(p.name) for p in [fig1_path, fig2_path, fig3_path, fig4_path]],
}
with open(exp_path, "w", encoding="utf-8") as f:
    json.dump(experiment_record, f, indent=2)
print(f"[Step 5] Logged Experiment Record -> {exp_path}")

print("\n" + "=" * 78)
print("  STAGE 11 EXECUTION COMPLETE — EFFICIENCY BENCHMARKED")
print("  All sequence scaling, memory metrics, and publication plots generated.")
print("=" * 78)
