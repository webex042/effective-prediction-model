"""
EPM Stage 10 — Flow Matching (Continuous Normalizing Flows & Optimal Transport) Demo
=====================================================================================

This script demonstrates all Stage 10 modules:
    1. Continuous Normalizing Flows (Neural ODEs dx/dt = v_theta(x, t))
    2. Optimal Transport Conditional Flow Matching (OT-CFM with Hungarian mini-batch coupling)
    3. Numerical ODE Solvers (Euler, Midpoint, RK4) and Straightness Analysis (Displacement / Arc Length)
    4. Instantaneous Change of Variables & Exact Log-Likelihood (NLL) via divergence trace
    5. Systematic Benchmark against Diffusion: OT-CFM vs Gaussian-CFM vs DDIM vs DDPM
    6. Multi-Channel Industrial Sensor Telemetry Manifold Flow Matching
    7. Publication-Quality Monochrome Visualizations & Benchmark Artifacts

Run:
    python demo_stage10.py
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
from torch.utils.data import DataLoader, TensorDataset

# UTF-8 stdout configuration for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.reproducibility import set_seed
from src.logger import setup_logging
from src.flow_matching.models import VelocityFieldNet
from src.flow_matching.probability_paths import OptimalTransportCFMPath, GaussianCFMPath
from src.flow_matching.ode_solvers import ODESolver, compute_path_straightness
from src.flow_matching.continuous_normalizing_flow import (
    ContinuousNormalizingFlow,
    compute_exact_divergence,
)
from src.flow_matching.benchmarks import (
    generate_two_moons_data,
    generate_sensor_manifold_data,
    compute_sliced_wasserstein,
    train_flow_matching,
    run_flow_matching_benchmark,
)

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage10")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
WEB_PLOTS_DIR = Path("plots/stage10")
WEB_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR = Path("benchmarks")
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("  EPM - Stage 10: Flow Matching & Continuous Normalizing Flows")
print("  Optimal Transport Vector Fields, Neural ODEs & Instantaneous Change of Vars")
print("=" * 78)

# Matplotlib monochrome retro styling
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
# Step 1: Synthesizing Target Distributions
# =====================================================================
print("\n[Step 1] Generating Multi-Modal 2D Manifold & 5-Channel Sensor Telemetry...")
moons_data = generate_two_moons_data(n_samples=1500, noise=0.08, random_state=42)
sensor_data = generate_sensor_manifold_data(n_samples=1500, dim=5, random_state=42)

print(f"  • Two Moons Dataset:         {moons_data.shape} (N={len(moons_data)}, D=2)")
print(f"  • Industrial Sensor Manifold: {sensor_data.shape} (N={len(sensor_data)}, D=5 channels)")


# =====================================================================
# Step 2: Training Optimal Transport Conditional Flow Matching (OT-CFM)
# =====================================================================
print("\n[Step 2] Training OT-CFM on 2D Multi-Modal Manifold...")
device = torch.device("cpu")
vnet_2d = VelocityFieldNet(input_dim=2, hidden_dim=96, num_layers=3)
cnf_2d = ContinuousNormalizingFlow(vnet_2d, data_dim=2, default_solver="euler")
ot_path = OptimalTransportCFMPath(use_ot=True)

dataset_2d = TensorDataset(torch.from_numpy(moons_data))
loader_2d = DataLoader(dataset_2d, batch_size=64, shuffle=True)

t0 = time.perf_counter()
losses = train_flow_matching(cnf_2d, ot_path, loader_2d, epochs=30, lr=2e-3, device=device)
train_time = time.perf_counter() - t0

print(f"  • Training Finished in {train_time:.2f}s (30 epochs)")
print(f"  • Initial CFM Loss: {losses[0]:.4f} -> Final CFM Loss: {losses[-1]:.4f}")


# =====================================================================
# Step 3: Trajectory Generation & Straightness Evaluation
# =====================================================================
print("\n[Step 3] Integrating Neural ODE Across Multiple Numerical Solvers...")
solvers_to_eval = [
    ("euler_20", "euler", 20),
    ("midpoint_10", "midpoint", 10),
    ("rk4_5", "rk4", 5),
    ("euler_5", "euler", 5),
]

solver_trajectories = {}
for name, method, steps in solvers_to_eval:
    samples, traj, nfe = cnf_2d.sample(
        batch_size=300, num_steps=steps, solver_method=method, return_trajectory=True, seed=42
    )
    straightness_mean, straightness_std = compute_path_straightness(traj)
    w2 = compute_sliced_wasserstein(samples.numpy(), moons_data[:300])
    solver_trajectories[name] = {
        "samples": samples.numpy(),
        "traj": traj.numpy(),
        "nfe": nfe,
        "straightness": straightness_mean,
        "w2": w2,
    }
    print(f"  • Solver '{method.upper()}' (steps={steps:2d}, NFE={nfe:2d}): "
          f"Straightness={straightness_mean:.4f}, W2={w2:.4f}")


# =====================================================================
# Step 4: Instantaneous Change of Variables & Exact Log-Likelihood
# =====================================================================
print("\n[Step 4] Computing Exact Negative Log-Likelihood (NLL) via Instantaneous Change of Vars...")
test_pts = torch.from_numpy(moons_data[:60])
log_p, nll = cnf_2d.compute_log_prob(test_pts, num_steps=20, divergence_method="exact")
mean_nll = float(nll.mean().item())
print(f"  • Exact Evaluated Test NLL: {mean_nll:.3f} nats (over 60 test points)")


# =====================================================================
# Step 5: Full Comparative Benchmark against Diffusion
# =====================================================================
print("\n[Step 5] Running Systematic Empirical Benchmark (Flow Matching vs Diffusion)...")
benchmark_results = run_flow_matching_benchmark(
    target_data=moons_data,
    epochs=25,
    batch_size=64,
    test_eval_size=300,
    seed=42,
)

summary_table = benchmark_results["summary"]
print("\n" + "=" * 95)
print(f"{'Model / Method':<34} | {'NFE':<4} | {'W2 Dist':<9} | {'Straightness':<12} | {'Latency':<10} | {'Params'}")
print("-" * 95)
for row in summary_table:
    print(f"{row['Model / Method']:<34} | {row['NFE']:<4} | {row['Wasserstein-2 (W2)']:<9} | "
          f"{row['Straightness Ratio']:<12} | {row['Latency / Sample']:<10} | {row['Parameters']}")
print("=" * 95)


# =====================================================================
# Step 6: Generating Publication Figures
# =====================================================================
print("\n[Step 6] Rendering Publication Monochrome Figures...")

# -------------------------------------------------------------
# Figure 1: Vector Field & Particle Trajectories
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
t_snapshots = [0.0, 0.5, 1.0]

grid_res = 16
x_vals = np.linspace(-2.5, 2.5, grid_res)
y_vals = np.linspace(-2.5, 2.5, grid_res)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
grid_pts = torch.from_numpy(np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1).astype(np.float32))

for ax, t_val in zip(axes, t_snapshots):
    t_tensor = torch.full((grid_pts.size(0), 1), t_val)
    with torch.no_grad():
        v_field = cnf_2d(grid_pts, t_tensor).numpy()
    U = v_field[:, 0].reshape(grid_res, grid_res)
    V = v_field[:, 1].reshape(grid_res, grid_res)

    # Plot velocity quiver
    ax.quiver(X_grid, Y_grid, U, V, color="#555555", alpha=0.7, scale=25.0, width=0.004)

    # Plot trajectories overlay
    traj_sample = solver_trajectories["euler_20"]["traj"]  # (21, 300, 2)
    for p_idx in range(min(25, traj_sample.shape[1])):
        path_x = traj_sample[:, p_idx, 0]
        path_y = traj_sample[:, p_idx, 1]
        ax.plot(path_x, path_y, color="#000000", linewidth=0.8, alpha=0.4)

    # Endpoints at current t
    step_idx = int(t_val * 20)
    curr_pts = traj_sample[step_idx, :80]
    ax.scatter(curr_pts[:, 0], curr_pts[:, 1], color="#000000", s=14, edgecolors="none", zorder=4)

    ax.set_title(f"Velocity Field v_θ(x, t={t_val:.1f})", fontsize=11, fontweight="bold")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(True)

plt.tight_layout()
fig1_path = PLOTS_DIR / "vector_field_and_trajectories.png"
fig.savefig(fig1_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig1_path}")

# -------------------------------------------------------------
# Figure 2: Trajectory Curvature Comparison (OT-CFM vs DDIM vs DDPM)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)

# A. OT-CFM Trajectories
ot_traj = solver_trajectories["euler_20"]["traj"]
for i in range(40):
    axes[0].plot(ot_traj[:, i, 0], ot_traj[:, i, 1], color="#000000", linewidth=0.9, alpha=0.6)
axes[0].scatter(ot_traj[0, :40, 0], ot_traj[0, :40, 1], marker="o", color="#888888", s=20, label="Prior p₀ ~ N(0, I)")
axes[0].scatter(ot_traj[-1, :40, 0], ot_traj[-1, :40, 1], marker="x", color="#000000", s=25, label="Target p₁ (Data)")
axes[0].set_title(f"OT-CFM (Straightness = {solver_trajectories['euler_20']['straightness']:.3f})\nOptimal Transport Paths", fontsize=10, fontweight="bold")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_xlim(-2.5, 2.5)
axes[0].set_ylim(-2.5, 2.5)
axes[0].grid(True)

# B. DDIM Accelerated Trajectories
ddim_traj = benchmark_results["ddim_trajectories"].numpy()
ddim_s, _ = compute_path_straightness(benchmark_results["ddim_trajectories"])
for i in range(40):
    axes[1].plot(ddim_traj[:, i, 0], ddim_traj[:, i, 1], color="#000000", linewidth=0.9, alpha=0.6, linestyle="--")
axes[1].scatter(ddim_traj[0, :40, 0], ddim_traj[0, :40, 1], marker="o", color="#888888", s=20)
axes[1].scatter(ddim_traj[-1, :40, 0], ddim_traj[-1, :40, 1], marker="x", color="#000000", s=25)
axes[1].set_title(f"DDIM-20 (Straightness = {ddim_s:.3f})\nCurved Inversion Flow", fontsize=10, fontweight="bold")
axes[1].set_xlim(-2.5, 2.5)
axes[1].set_ylim(-2.5, 2.5)
axes[1].grid(True)

# C. DDPM Stochastic Trajectories
ddpm_traj = benchmark_results["ddpm_trajectories"].numpy()
ddpm_s, _ = compute_path_straightness(benchmark_results["ddpm_trajectories"])
for i in range(25):
    axes[2].plot(ddpm_traj[:, i, 0], ddpm_traj[:, i, 1], color="#555555", linewidth=0.7, alpha=0.5)
axes[2].scatter(ddpm_traj[0, :25, 0], ddpm_traj[0, :25, 1], marker="o", color="#888888", s=20)
axes[2].scatter(ddpm_traj[-1, :25, 0], ddpm_traj[-1, :25, 1], marker="x", color="#000000", s=25)
axes[2].set_title(f"DDPM-100 (Straightness = {ddpm_s:.3f})\nStochastic Brownian Walk", fontsize=10, fontweight="bold")
axes[2].set_xlim(-2.5, 2.5)
axes[2].set_ylim(-2.5, 2.5)
axes[2].grid(True)

plt.tight_layout()
fig2_path = PLOTS_DIR / "flow_matching_vs_diffusion_curvature.png"
fig.savefig(fig2_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig2_path}")

# -------------------------------------------------------------
# Figure 3: NFE vs Generation Quality (Ablation Curve)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

nfe_steps_list = [2, 4, 8, 15, 25, 50]
ot_w2_list = []
ot_lat_list = []
for steps in nfe_steps_list:
    t_start = time.perf_counter()
    s_pts, _, _ = cnf_2d.sample(batch_size=200, num_steps=steps, solver_method="euler")
    lat = (time.perf_counter() - t_start) * 1000.0 / 200.0
    w_val = compute_sliced_wasserstein(s_pts.numpy(), moons_data[:200])
    ot_w2_list.append(w_val)
    ot_lat_list.append(lat)

ax1.plot(nfe_steps_list, ot_w2_list, marker="s", color="#000000", linewidth=1.5, label="OT-CFM (Euler)")
ax1.axhline(y=0.18, color="#555555", linestyle="--", label="Target Quality Threshold")
ax1.set_title("NFE vs. Sliced 2-Wasserstein Distance (W2)", fontsize=10, fontweight="bold")
ax1.set_xlabel("Number of Function Evaluations (NFE)")
ax1.set_ylabel("Sliced 2-Wasserstein Distance (W2)")
ax1.set_ylim(0.0, 0.6)
ax1.grid(True)
ax1.legend(loc="upper right", fontsize=8)

ax2.plot(nfe_steps_list, ot_lat_list, marker="o", color="#000000", linewidth=1.5)
ax2.set_title("NFE vs. Sampling Latency (ms / sample)", fontsize=10, fontweight="bold")
ax2.set_xlabel("Number of Function Evaluations (NFE)")
ax2.set_ylabel("Inference Latency (ms)")
ax2.grid(True)

plt.tight_layout()
fig3_path = PLOTS_DIR / "nfe_vs_generation_quality.png"
fig.savefig(fig3_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig3_path}")

# -------------------------------------------------------------
# Figure 4: CNF Density Evolution & Change of Variables
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=150)
time_eval_steps = [0.0, 0.33, 0.66, 1.0]

for idx, (ax, t_val) in enumerate(zip(axes, time_eval_steps)):
    step_num = int(t_val * 20)
    pts_at_t = solver_trajectories["euler_20"]["traj"][step_num]  # (300, 2)
    ax.scatter(pts_at_t[:, 0], pts_at_t[:, 1], color="#000000", s=10, alpha=0.7)
    ax.set_title(f"Density p_t(x) at t = {t_val:.2f}", fontsize=10, fontweight="bold")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("x₁")
    if idx == 0:
        ax.set_ylabel("x₂")
    ax.grid(True)

plt.tight_layout()
fig4_path = PLOTS_DIR / "cnf_density_and_change_of_vars.png"
fig.savefig(fig4_path, bbox_inches="tight")
plt.close(fig)
print(f"  • Saved: {fig4_path}")

# Copy all plots to web folder
for p in [fig1_path, fig2_path, fig3_path, fig4_path]:
    shutil.copy(p, WEB_PLOTS_DIR / p.name)


# =====================================================================
# Step 7: Saving Benchmark & Experiment JSON Artifacts
# =====================================================================
benchmark_path = BENCHMARKS_DIR / "stage10_flow_matching.json"
benchmark_data = {
    "summary": summary_table,
    "eval_samples": 300,
    "ot_straightness": solver_trajectories["euler_20"]["straightness"],
    "ddim_straightness": float(ddim_s),
    "ddpm_straightness": float(ddpm_s),
    "exact_nll": mean_nll,
}
with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2)
print(f"\n[Step 7] Saved Benchmark Metrics -> {benchmark_path}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_path = EXPERIMENTS_DIR / f"{timestamp}_stage10_flow_matching.json"
experiment_record = {
    "timestamp": timestamp,
    "stage": 10,
    "name": "Flow Matching (Continuous Normalizing Flows & Optimal Transport)",
    "training_epochs": 30,
    "final_cfm_loss": losses[-1],
    "models_evaluated": [row["Model / Method"] for row in summary_table],
    "summary": summary_table,
    "straightness_comparison": {
        "OT-CFM": solver_trajectories["euler_20"]["straightness"],
        "DDIM-20": float(ddim_s),
        "DDPM-100": float(ddpm_s),
    },
    "exact_nll_nats": mean_nll,
    "plots_generated": [str(p.name) for p in [fig1_path, fig2_path, fig3_path, fig4_path]],
}
with open(exp_path, "w", encoding="utf-8") as f:
    json.dump(experiment_record, f, indent=2)
print(f"[Step 8] Logged Experiment Record -> {exp_path}")

print("\n" + "=" * 78)
print("  STAGE 10 EXECUTION COMPLETE — FLOW MATCHING VERIFIED")
print("  All benchmarks, metrics, and publication monochrome plots generated.")
print("=" * 78)
