"""
EPM Stage 6 — Multimodal Representation Learning & Fusion Demo
==============================================================

This script demonstrates all Stage 6 modules:
    1. Paired Multimodal Equipment Dataset (Multivariate Telemetry + Free-form Technician Text Logs)
    2. Modality Encoders: SensorEncoder (1D-Conv + BiGRU) & TextEncoder (Embedding + BiGRU)
    3. Multimodal Fusion Mechanisms:
       - Baseline Concatenation Fusion
       - Gated Multimodal Unit (GMU — Adaptive Modal Highway)
       - Bidirectional Cross-Modal Attention
    4. Unimodal vs Multimodal Ablation Study (Sensor-Only vs Text-Only vs Concat vs Gated vs Cross-Attn)
    5. Stress-Testing Under Missing & Corrupted Modalities (Missing Text, Noisy Sensor, Missing Sensor)
    6. Modal Interpretability & Gating Weight Distributions
    7. Publication-Quality Visualizations & Benchmark Leaderboard

Run:
    python demo_stage6.py
"""

import sys
import io
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
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
from src.multimodal import (
    MultimodalEquipmentDataset,
    generate_paired_multimodal_dataset,
    SensorEncoder,
    TextEncoder,
    ConcatenationFusion,
    GatedMultimodalFusion,
    CrossAttentionFusion,
    MultimodalClassifier,
    SensorOnlyClassifier,
    TextOnlyClassifier,
    run_multimodal_ablation_study,
)

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage6")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR = Path("benchmarks")
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("  EPM - Stage 6: Multimodal Representation Learning & Fusion")
print("  Continuous Sensor Telemetry + Discrete Text Logs + Gated Fusion (GMU)")
print("=" * 75)


# =====================================================================
# Step 1: Paired Multimodal Equipment Dataset Generation
# =====================================================================
print("\n[Step 1] Synthesizing Paired Multimodal Diagnostic Dataset...")
n_samples = 600
window_size = 24
max_text_len = 16

sensors_raw, text_ids, labels, raw_texts, vocab = generate_paired_multimodal_dataset(
    n_samples=n_samples,
    window_size=window_size,
    max_text_len=max_text_len,
    normalize_sensors=False,
    random_seed=42,
)

class_names = ["Nominal", "Warning", "Critical Fault"]
print(f"  Total samples generated: {n_samples}")
print(f"  Sensor window shape:     {sensors_raw.shape[1:]} (24 timesteps x 5 physical channels)")
print(f"  Text token length:       {text_ids.shape[1]} tokens per incident report")
print(f"  Vocabulary size:         {len(vocab)} unique technical terms")

print("\n  Sample Paired Diagnostic Observations (Physical Units):")
for cls_idx, cname in enumerate(class_names):
    idx = int(np.where(labels == cls_idx)[0][0])
    mean_temp = float(np.mean(sensors_raw[idx, :, 0]))
    mean_vib = float(np.mean(sensors_raw[idx, :, 1]))
    print(f"  - [{cname:^14}] Temp: {mean_temp:.1f} C, Vib: {mean_vib:.2f} g | Report: \"{raw_texts[idx]}\"")

# Standardize sensor telemetry (zero mean, unit variance) for neural training
ch_mean = np.mean(sensors_raw, axis=(0, 1), keepdims=True)
ch_std = np.std(sensors_raw, axis=(0, 1), keepdims=True) + 1e-6
sensors = ((sensors_raw - ch_mean) / ch_std).astype(np.float32)

# Chronological Train / Test Split (80% train, 20% test)
split_idx = int(n_samples * 0.8)
train_sensors, test_sensors = sensors[:split_idx], sensors[split_idx:]
train_text_ids, test_text_ids = text_ids[:split_idx], text_ids[split_idx:]
train_labels, test_labels = labels[:split_idx], labels[split_idx:]

train_dataset = MultimodalEquipmentDataset(train_sensors, train_text_ids, train_labels)
test_dataset = MultimodalEquipmentDataset(test_sensors, test_text_ids, test_labels)
print(f"  Data partitions:         Train={len(train_dataset)}, Test={len(test_dataset)}")

# =====================================================================
# Step 2: Running Comprehensive Multimodal Ablation Study
# =====================================================================
print("\n" + "=" * 75)
print("[Step 2] Executing Systematic Multimodal Ablation Study...")
print("  Evaluating 5 Architectures across Clean and Degraded Modality Regimes:")
print("    1. Sensor-Only Model (Unimodal baseline)")
print("    2. Text-Only Model (Unimodal baseline)")
print("    3. Multimodal Concat (Early/Mid concatenation)")
print("    4. Multimodal Gated GMU (Dynamic adaptive channel gating)")
print("    5. Multimodal Cross-Attention (Bidirectional query interaction)")
print("=" * 75)

summary_df, detailed_results = run_multimodal_ablation_study(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    vocab_size=len(vocab),
    sensor_dim=5,
    latent_dim=64,
    epochs=15,
    batch_size=32,
    device="cpu",
)

print("\nMultimodal Benchmark Leaderboard:")
print(summary_df.to_string(index=False))


# =====================================================================
# Step 3: Modality Interpretability & Gating Dynamics (GMU)
# =====================================================================
print("\n" + "=" * 75)
print("[Step 3] Modal Interpretability: GMU Gating Weight Analysis...")
print("=" * 75)

gated_info = detailed_results["models"]["Multimodal-Gated"]
if "sensor_gates" in gated_info["aux"]:
    test_gates = gated_info["aux"]["sensor_gates"]
    y_test_arr = np.array(gated_info["predictions"]["targets"])

    print("  Average Reliance on Sensor Telemetry (z_sensor in [0, 1]) by Health State:")
    for cls_idx, cname in enumerate(class_names):
        mask = (y_test_arr == cls_idx)
        if np.any(mask):
            cls_gates = test_gates[mask]
            mean_gate = float(np.mean(cls_gates))
            std_gate = float(np.std(cls_gates))
            print(f"    - Class {cls_idx} ({cname}): mean z_sensor = {mean_gate:.3f} +/- {std_gate:.3f} "
                  f"(Sensor: {mean_gate*100:.1f}%, Text: {(1-mean_gate)*100:.1f}%)")


# =====================================================================
# Step 4: Generating Publication-Grade Visualizations
# =====================================================================
print("\n" + "=" * 75)
print("[Step 4] Generating Publication-Grade Visualizations...")
print("=" * 75)

# ---------------------------------------------------------------------
# Visualization 1: Multimodal Ablation & Robustness Comparison
# ---------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

models = summary_df["Model"].tolist()
x = np.arange(len(models))
width = 0.2

rects1 = ax.bar(x - 1.5 * width, summary_df["Clean Macro F1"], width, label="Clean (Full Modalities)", color="#2563EB", alpha=0.9)
rects2 = ax.bar(x - 0.5 * width, summary_df["Missing Text F1"], width, label="Missing Text (No Reports)", color="#F59E0B", alpha=0.9)
rects3 = ax.bar(x + 0.5 * width, summary_df["Noisy Sensor F1"], width, label="Noisy Sensor (High Noise)", color="#EF4444", alpha=0.9)
rects4 = ax.bar(x + 1.5 * width, summary_df["Missing Sensor F1"], width, label="Missing Sensor (Offline)", color="#6B7280", alpha=0.9)

ax.set_ylabel("Macro F1-Score", fontsize=12, fontweight="bold")
ax.set_title("EPM Stage 6: Multimodal Ablation & Robustness under Degraded Regimes", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight="medium")
ax.set_ylim(0.0, 1.08)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower left", fontsize=10)

# Add value labels on top of clean bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plot1_path = PLOTS_DIR / "multimodal_ablation_comparison.png"
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"  [Saved] {plot1_path}")


# ---------------------------------------------------------------------
# Visualization 2: GMU Dynamic Gating Distributions
# ---------------------------------------------------------------------
if "sensor_gates" in gated_info["aux"]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, dpi=150)
    colors = ["#10B981", "#F59E0B", "#EF4444"]

    for i, cname in enumerate(class_names):
        mask = (y_test_arr == i)
        cls_gates = test_gates[mask] if np.any(mask) else np.array([0.5])
        axes[i].hist(cls_gates, bins=12, range=(0, 1), color=colors[i], edgecolor="black", alpha=0.85)
        axes[i].axvline(np.mean(cls_gates), color="darkblue", linestyle="--", linewidth=2, label=f"Mean: {np.mean(cls_gates):.2f}")
        axes[i].set_title(f"State: {cname}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Sensor Modality Gate (z_sensor)", fontsize=11)
        axes[i].set_xlim(0, 1)
        axes[i].legend(loc="upper left", fontsize=10)

    axes[0].set_ylabel("Sample Frequency", fontsize=11, fontweight="bold")
    fig.suptitle("EPM Stage 6: GMU Modality Reliance Gating Weights by Operational State", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot2_path = PLOTS_DIR / "modality_gating_weights.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"  [Saved] {plot2_path}")


# ---------------------------------------------------------------------
# Visualization 3: Confusion Matrices (Sensor-Only vs Text-Only vs Gated)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
models_to_plot = ["Sensor-Only", "Text-Only", "Multimodal-Gated"]

for i, m_name in enumerate(models_to_plot):
    cm = np.array(detailed_results["models"][m_name]["clean"]["confusion_matrix"])
    ax = axes[i]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{m_name}\n(Macro F1: {detailed_results['models'][m_name]['clean']['macro_f1']:.3f})", fontsize=12, fontweight="bold")
    
    # Tick marks
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    # Text annotations
    thresh = cm.max() / 2.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, f"{cm[r, c]}",
                    ha="center", va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontweight="bold", fontsize=11)

    ax.set_xlabel("Predicted Diagnosis", fontsize=10)
    if i == 0:
        ax.set_ylabel("True Health State", fontsize=10, fontweight="bold")

plt.tight_layout()
plot3_path = PLOTS_DIR / "confusion_matrices_multimodal.png"
plt.savefig(plot3_path, dpi=150)
plt.close()
print(f"  [Saved] {plot3_path}")


# =====================================================================
# Step 5: Persisting Leaderboard & Benchmark Artifacts
# =====================================================================
benchmark_file = BENCHMARKS_DIR / "stage6_multimodal_leaderboard.json"
with open(benchmark_file, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary_df.to_dict(orient="records"),
        "detailed_results": {
            m: {
                "parameters": res["parameters"],
                "train_time_sec": res["train_time_sec"],
                "clean": res["clean"],
                "missing_text": res["missing_text"],
                "noisy_sensor": res["noisy_sensor"],
                "missing_sensor": res["missing_sensor"],
            }
            for m, res in detailed_results["models"].items()
        }
    }, f, indent=2)

print(f"\n[Artifact Saved] Benchmark results written to {benchmark_file}")
print("\n" + "=" * 75)
print("  Stage 6 Execution Complete: Multimodal Representation Learning Verified!")
print("=" * 75)
