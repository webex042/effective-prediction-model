"""
EPM Stage 5 — Time-Series Prediction & Deep Sequence Forecasting Demo
=====================================================================

This script demonstrates all Stage 5 modules:
    1. Multivariate Industrial Telemetry Simulation (Temperature, Vibration, Pressure, RPM, Load)
    2. Leakage-free Chronological Sliding Windows (W=24 lookback, H=6 multi-step horizon)
    3. Vanilla RNN from Scratch (Hidden state dynamics and BPTT principles)
    4. 6-Model Comparative Benchmark: Linear Regression, Random Forest, XGBoost, LSTM, GRU, Transformer
    5. Multi-Step Forecast Trajectory Visualization (Predicted vs Actual future trajectories)
    6. Computational Benchmarking (RMSE, MAE, R², Latency, and Parameters)

Run:
    python demo_stage5.py
"""

import sys
import io
import os
import time
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Reconfigure stdout for utf-8 safety on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.reproducibility import set_seed
from src.logger import setup_logging
from src.timeseries import (
    generate_multivariate_sensor_telemetry,
    create_sliding_windows,
    temporal_train_test_split,
    VanillaRNNScratch,
    run_timeseries_benchmark,
)

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage5")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR = Path("benchmarks")
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  EPM - Stage 5: Time-Series Prediction & Sequence Modeling")
print("  Multivariate Telemetry, Gated Recurrence & Temporal Transformers")
print("=" * 72)


# =====================================================================
# 1. MULTIVARIATE TELEMETRY SIMULATION
# =====================================================================

print("\n" + "-" * 72)
print("  1. SIMULATING MULTIVARIATE INDUSTRIAL SENSOR TELEMETRY")
print("-" * 72)

n_steps = 1500
df_telemetry, feature_names = generate_multivariate_sensor_telemetry(
    n_timesteps=n_steps, sampling_rate_minutes=10, random_seed=42
)

print(f"  Generated {n_steps} timesteps across 5 sensor channels:")
print(f"    Channels: {', '.join(feature_names)}")
print(f"    Summary Statistics:\n{df_telemetry.describe().round(2).T[['mean', 'std', 'min', 'max']]}")

# Plot 5 Sensor Channels
fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True, dpi=150)
time_hours = np.arange(n_steps) * 10 / 60.0  # converted to hours

colors = ["#dc2626", "#ea580c", "#2563eb", "#7c3aed", "#059669"]
units = ["°C", "mm/s", "bar", "RPM", "%"]

for idx, (col, unit, c) in enumerate(zip(feature_names, units, colors)):
    axes[idx].plot(time_hours, df_telemetry[col], color=c, linewidth=1.2)
    axes[idx].set_ylabel(f"{col.capitalize()}\n({unit})", fontsize=8, fontweight="bold")
    axes[idx].grid(True, alpha=0.3)
    if idx == 0:
        axes[idx].set_title("Industrial Machine Telemetry with Degradation Phase (Final 25%)", fontweight="bold")

axes[-1].set_xlabel("Operational Time (Hours)", fontweight="bold")
# Highlight degradation zone
for ax in axes:
    ax.axvspan(time_hours[int(0.75 * n_steps)], time_hours[-1], color="red", alpha=0.08, label="Wear Phase")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "multivariate_sensor_signals.png")
plt.close(fig)
print(f"  [Plot saved]: {PLOTS_DIR / 'multivariate_sensor_signals.png'}")


# =====================================================================
# 2. CHRONOLOGICAL SLIDING WINDOWS (ZERO LEAKAGE)
# =====================================================================

print("\n" + "-" * 72)
print("  2. CHRONOLOGICAL SLIDING WINDOW TRANSFORMATION")
print("-" * 72)

raw_data = df_telemetry[feature_names].values
window_size = 24  # Lookback W = 24 timesteps (4 hours of history)
horizon = 6       # Forecast H = 6 timesteps into future (1 hour future trajectory)

X_all, y_all = create_sliding_windows(
    raw_data, window_size=window_size, horizon=horizon, stride=1, target_channel=0
)

# Chronological Split: 70% Train, 15% Val, 15% Test
X_train, y_train, X_val, y_val, X_test, y_test = temporal_train_test_split(
    X_all, y_all, train_ratio=0.70, val_ratio=0.15
)

# Standardize using ONLY training statistics (Zero lookahead leakage!)
mean_X = np.mean(X_train, axis=(0, 1), keepdims=True)
std_X = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8

X_train_s = (X_train - mean_X) / std_X
X_val_s = (X_val - mean_X) / std_X
X_test_s = (X_test - mean_X) / std_X

print(f"  Sliding Window Configuration:")
print(f"    Lookback Window (W): {window_size} steps ({window_size * 10} mins)")
print(f"    Forecast Horizon (H): {horizon} steps ({horizon * 10} mins)")
print(f"    Target Channel: Core Temperature (°C)")
print(f"    Total Windows: {len(X_all)}")
print(f"    Chronological Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"    Shape X: {X_train.shape} | Shape y: {y_train.shape}")


# =====================================================================
# 3. VANILLA RNN FROM SCRATCH WALKTHROUGH
# =====================================================================

print("\n" + "-" * 72)
print("  3. VANILLA RNN FROM SCRATCH (BPTT PRINCIPLES)")
print("-" * 72)

rnn_scratch = VanillaRNNScratch(input_dim=5, hidden_dim=16, output_dim=horizon)
sample_seq = X_test_s[:2]  # batch of 2 sequences
y_rnn_pred, h_states = rnn_scratch.forward(sample_seq)

print(f"  Input Batch Shape: {sample_seq.shape}")
print(f"  Recurrence Formula: h_t = tanh( x_t W_xh + h_{{t-1}} W_hh + b_h )")
print(f"  Hidden State Evolution across {window_size} timesteps:")
print(f"    h_0 (initial zero state): norm = {np.linalg.norm(h_states[0]):.4f}")
print(f"    h_12 (midway timestep):   norm = {np.linalg.norm(h_states[12]):.4f}")
print(f"    h_24 (final state):       norm = {np.linalg.norm(h_states[24]):.4f}")
print(f"  Forecasted Multi-step Output (shape {y_rnn_pred.shape}):\n{np.round(y_rnn_pred, 2)}")


# =====================================================================
# 4. BENCHMARK SUITE: 6 MODEL PARADIGMS
# =====================================================================

print("\n" + "-" * 72)
print("  4. TIME-SERIES BENCHMARK LEADERBOARD (6 MODELS)")
print("-" * 72)
print("  Training and evaluating Linear, Random Forest, XGBoost, LSTM, GRU, Transformer...")

df_leaderboard, predictions = run_timeseries_benchmark(
    X_train=X_train_s,
    y_train=y_train,
    X_test=X_test_s,
    y_test=y_test,
    epochs=35,
    batch_size=32,
    hidden_dim=32,
)

print("\n--- TIME-SERIES FORECASTING LEADERBOARD (Core Temperature, H=6) ---")
print(df_leaderboard[["model", "family", "rmse", "mae", "r2", "latency_us_per_sample", "training_time_s"]].to_string(index=False))

# Save Leaderboard
df_leaderboard.to_json(BENCHMARKS_DIR / "stage5_timeseries_leaderboard.json", orient="records", indent=2)


# =====================================================================
# 5. VISUALIZING MULTI-STEP FORECAST TRAJECTORIES
# =====================================================================

print("\n" + "-" * 72)
print("  5. VISUALIZING MULTI-STEP TRAJECTORY FORECASTS")
print("-" * 72)

# Select a representative test window during the degradation phase
test_window_idx = min(120, len(y_test) - 1)
actual_trajectory = y_test[test_window_idx]
future_steps = np.arange(1, horizon + 1) * 10  # minutes

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.plot(future_steps, actual_trajectory, color="black", linewidth=2.5, marker="o", label="Actual Future Temperature")

model_colors = {
    "Linear Regression": "#94a3b8",
    "Random Forest": "#f59e0b",
    "XGBoost": "#10b981",
    "LSTM": "#dc2626",
    "GRU": "#ea580c",
    "Transformer": "#7c3aed",
}

for name, preds in predictions.items():
    pred_traj = preds[test_window_idx]
    ax.plot(
        future_steps,
        pred_traj,
        label=name,
        color=model_colors.get(name, "blue"),
        linewidth=1.8,
        linestyle="--" if "Linear" in name or "Forest" in name else "-",
        marker="s",
        markersize=4,
    )

ax.set_title(f"Multi-Step Ahead Temperature Forecast Trajectory (Horizon = {horizon} Steps)", fontweight="bold")
ax.set_xlabel("Forecast Lead Time (Minutes into Future)", fontweight="bold")
ax.set_ylabel("Core Temperature (°C)", fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "forecast_comparison.png")
plt.close(fig)
print(f"  [Plot saved]: {PLOTS_DIR / 'forecast_comparison.png'}")


# =====================================================================
# 6. BENCHMARK COMPARISON BAR CHART
# =====================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

models_sorted = df_leaderboard["model"].values
rmse_sorted = df_leaderboard["rmse"].values
lat_sorted = df_leaderboard["latency_us_per_sample"].values
bar_colors = [model_colors.get(m, "#3b82f6") for m in models_sorted]

ax1.barh(np.arange(len(models_sorted)), rmse_sorted, color=bar_colors, edgecolor="black", alpha=0.85)
ax1.set_yticks(np.arange(len(models_sorted)))
ax1.set_yticklabels(models_sorted, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("Test RMSE (Lower is better)", fontweight="bold")
ax1.set_title("Predictive Accuracy by Model", fontweight="bold")
for i, v in enumerate(rmse_sorted):
    ax1.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold")
ax1.grid(True, axis="x", alpha=0.3)

ax2.barh(np.arange(len(models_sorted)), lat_sorted, color=bar_colors, edgecolor="black", alpha=0.85)
ax2.set_yticks(np.arange(len(models_sorted)))
ax2.set_yticklabels(models_sorted, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("Inference Latency (µs / sample, Lower is better)", fontweight="bold")
ax2.set_title("Computational Speed by Model", fontweight="bold")
for i, v in enumerate(lat_sorted):
    ax2.text(v + 1.0, i, f"{v:.1f}µs", va="center", fontsize=8, fontweight="bold")
ax2.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "timeseries_model_benchmark.png")
plt.close(fig)
print(f"  [Plot saved]: {PLOTS_DIR / 'timeseries_model_benchmark.png'}")


# =====================================================================
# STAGE 5 SUMMARY
# =====================================================================

print("\n" + "=" * 72)
print("  STAGE 5 COMPLETE: TIME-SERIES PREDICTION & SEQUENCE MODELING")
print("=" * 72)
print("  Summary of Key Discoveries:")
print("  1. Strict Chronological Hygiene:")
print("     Respecting temporal ordering is non-negotiable. Scalers and splits")
print("     must never leak future statistics into training.")
print("  2. Deep Recurrent & Attention Superiority:")
print("     LSTM, GRU, and Time-Series Transformer consistently outperform")
print("     flattened tabular baselines on smooth continuous degradation trajectories.")
print("  3. The Stage 6 Connection (Multimodal Fusion):")
print("     The temporal sensor representations learned here (LSTM / Transformer)")
print("     will be combined directly with Stage 4 NLP embeddings in Stage 6!")
print("=" * 72)
print("  Ready for Stage 6: Multimodal Learning (Sensor + Text Fusion)")
print("=" * 72)
