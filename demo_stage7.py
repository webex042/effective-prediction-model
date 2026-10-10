"""
EPM Stage 7 — Generative Modeling & Latent Representation Learning Demo
========================================================================

This script demonstrates all Stage 7 modules:
    1. NumPy VAE from Scratch & Numerical Gradient Checking (ELBO + Reparameterization)
    2. PyTorch Generative Suite: Standard-AE, Variational-VAE, and Conditional-VAE (C-VAE)
    3. Systematic Generative Benchmark: Reconstruction MSE/MAE, Latency, and Anomaly AUROC
    4. Controlled & Conditional Telemetry Generation (Synthesizing Nominal vs Fault trajectories)
    5. Latent Space Manifold Analysis (Smooth representation topology)
    6. Monte-Carlo Epistemic Uncertainty Estimation (Decoded confidence bounds)
    7. Publication-Quality Visualizations & Benchmark Artifacts

Run:
    python demo_stage7.py
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
import torch
from torch.utils.data import TensorDataset

# Reconfigure stdout for utf-8 safety on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.reproducibility import set_seed
from src.logger import setup_logging
from src.multimodal.dataset import generate_paired_multimodal_dataset
from src.generative.vae_scratch import VAEScratch, check_vae_numerical_gradients
from src.generative.models import (
    StandardAutoencoder,
    VariationalAutoencoder,
    ConditionalVAE,
    train_generative_model,
)
from src.generative.benchmarks import run_generative_benchmark

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage7")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR = Path("benchmarks")
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("  EPM - Stage 7: Generative Modeling & Probabilistic Latent Representations")
print("  Autoencoders, VAEs, ELBO, Reparameterization Trick & Uncertainty Estimation")
print("=" * 78)


# =====================================================================
# Step 1: NumPy VAE from Scratch & Numerical Gradient Checking
# =====================================================================
print("\n[Step 1] Verifying Mathematical Foundations (Pure NumPy VAE)...")
vae_scratch = VAEScratch(input_dim=10, hidden_dim=16, latent_dim=4, beta=1.0, random_seed=42)
dummy_x = np.random.randn(4, 10)
x_rec, mu_s, logvar_s, z_s = vae_scratch.forward(dummy_x)
loss_tot, loss_rec, loss_kl = vae_scratch.compute_loss(dummy_x, x_rec, mu_s, logvar_s)

print(f"  - Forward pass completed:  Input=(4, 10) -> Latent z=(4, 4) -> Recon=(4, 10)")
print(f"  - ELBO Loss components:    Total={loss_tot:.4f} (Recon={loss_rec:.4f}, KL={loss_kl:.4f})")

print("  - Running two-sided numerical gradient checking for VAE loss...")
max_rel_error = check_vae_numerical_gradients(input_dim=6, hidden_dim=8, latent_dim=3, h=1e-5)
print(f"  - Maximum relative error:  {max_rel_error:.2e} (Threshold: < 1.00e-04)")
assert max_rel_error < 1e-4, "Gradient check failed!"
print("  - [VERIFIED] Analytical gradients match numerical finite-differences perfectly!")


# =====================================================================
# Step 2: Preparing Industrial Telemetry Dataset
# =====================================================================
print("\n" + "=" * 78)
print("[Step 2] Synthesizing Multi-Channel Industrial Sensor Telemetry...")
n_samples = 600
window_size = 24
n_channels = 5
input_dim = window_size * n_channels  # 24 * 5 = 120

sensors_raw, _, labels, _, _ = generate_paired_multimodal_dataset(
    n_samples=n_samples,
    window_size=window_size,
    normalize_sensors=False,
    random_seed=42,
)

# Standardize sensor telemetry
ch_mean = np.mean(sensors_raw, axis=(0, 1), keepdims=True)
ch_std = np.std(sensors_raw, axis=(0, 1), keepdims=True) + 1e-6
sensors_norm = ((sensors_raw - ch_mean) / ch_std).astype(np.float32)

# Flatten temporal sequence to feature vector for dense generative models
x_flat = sensors_norm.reshape(n_samples, input_dim)

split_idx = int(n_samples * 0.8)
x_train, x_test = torch.from_numpy(x_flat[:split_idx]), torch.from_numpy(x_flat[split_idx:])
y_train, y_test = torch.from_numpy(labels[:split_idx]), torch.from_numpy(labels[split_idx:])

train_dataset = TensorDataset(x_train, x_train, y_train)
test_dataset = TensorDataset(x_test, x_test, y_test)

class_names = ["Nominal", "Warning", "Critical Fault"]
print(f"  Dataset partitions: Train={len(train_dataset)}, Test={len(test_dataset)}")
print(f"  Input dimension:    {input_dim} (24 timesteps x 5 channels: Temp, Vib, Press, RPM, Load)")


# =====================================================================
# Step 3: Generative Models Benchmark Suite
# =====================================================================
print("\n" + "=" * 78)
print("[Step 3] Executing Generative Benchmark Suite...")
print("  Comparing 3 Architectures:")
print("    1. Standard-AE:    Deterministic Autoencoder (Information bottleneck)")
print("    2. Variational-VAE: Probabilistic Gaussian Latent Space (ELBO)")
print("    3. Conditional-VAE: Class-Conditioned Latent Generation (C-VAE)")
print("=" * 78)

summary_df, detailed_results = run_generative_benchmark(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    input_dim=input_dim,
    latent_dim=8,
    epochs=25,
    batch_size=32,
    device="cpu",
)

print("\nGenerative Model Leaderboard:")
print(summary_df.to_string(index=False))


# =====================================================================
# Step 4: Latent Space Geometry & Manifold Extraction
# =====================================================================
print("\n" + "=" * 78)
print("[Step 4] Extracting Latent Space Manifold & Topology...")
print("=" * 78)

vae_model = detailed_results["Variational-VAE"]["model"]
vae_model.eval()
with torch.no_grad():
    mu_test, _ = vae_model.encode(x_test)
    z_test = mu_test.cpu().numpy()

# 2D PCA projection of the 8D latent space for visualization
cov = np.cov(z_test, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
top2_vecs = eigenvectors[:, -2:]
z_2d = z_test @ top2_vecs

print(f"  Latent representations extracted: shape={z_test.shape} -> projected to 2D manifold")


# =====================================================================
# Step 5: Controlled Class-Conditional Waveform Generation
# =====================================================================
print("\n" + "=" * 78)
print("[Step 5] Controlled Synthetic Telemetry Synthesis (Conditional VAE)...")
print("=" * 78)

cvae_model = detailed_results["Conditional-VAE"]["model"]
cvae_model.eval()

generated_by_class = {}
with torch.no_grad():
    for c in range(3):
        samples_flat = cvae_model.sample_conditional(class_label=c, num_samples=5).cpu().numpy()
        # Unflatten to (N, 24, 5) and unnormalize back to physical units
        samples_unflat = samples_flat.reshape(5, window_size, n_channels)
        samples_phys = samples_unflat * ch_std + ch_mean
        generated_by_class[class_names[c]] = samples_phys

        mean_temp = float(np.mean(samples_phys[:, :, 0]))
        mean_vib = float(np.mean(samples_phys[:, :, 1]))
        print(f"  Generated {class_names[c]:^14}: Temp={mean_temp:.1f} C, Vib={mean_vib:.2f} g")


# =====================================================================
# Step 6: Monte-Carlo Posterior Uncertainty Estimation
# =====================================================================
print("\n" + "=" * 78)
print("[Step 6] Estimating Predictive Epistemic Uncertainty via Monte-Carlo Draws...")
print("=" * 78)

test_sample = x_test[0:1]  # (1, 120)
with torch.no_grad():
    mc_mean_flat, mc_std_flat = vae_model.sample_with_uncertainty(test_sample, num_mc_samples=30)

mc_mean = mc_mean_flat.numpy().reshape(window_size, n_channels) * ch_std[0] + ch_mean[0]
mc_std = mc_std_flat.numpy().reshape(window_size, n_channels) * ch_std[0]

print(f"  Input sample test target: Class = {class_names[int(y_test[0])]}")
print(f"  Average Decoded Uncertainty (+/- 1 std): Temperature = +/- {float(np.mean(mc_std[:, 0])):.2f} C")
print(f"                                          Vibration   = +/- {float(np.mean(mc_std[:, 1])):.3f} g")


# =====================================================================
# Step 7: Generating Publication-Grade Visualizations
# =====================================================================
print("\n" + "=" * 78)
print("[Step 7] Generating Publication-Grade Visualizations...")
print("=" * 78)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# ---------------------------------------------------------------------
# Visualization 1: VAE Latent Space Manifold (2D Projection)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
colors = ["#10B981", "#F59E0B", "#EF4444"]
markers = ["o", "s", "^"]
y_test_np = y_test.numpy()

for c in range(3):
    mask = (y_test_np == c)
    ax.scatter(
        z_2d[mask, 0], z_2d[mask, 1],
        c=colors[c], label=class_names[c],
        marker=markers[c], alpha=0.85, s=60, edgecolors="black", linewidth=0.5
    )

ax.set_title("EPM Stage 7: VAE Continuous Latent Manifold Topology (d_z = 8 -> 2D)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Latent Principal Dimension 1", fontsize=11)
ax.set_ylabel("Latent Principal Dimension 2", fontsize=11)
ax.legend(title="Health State", frameon=True, facecolor="white", fontsize=10)
plt.tight_layout()
plot1_path = PLOTS_DIR / "vae_latent_space_manifold.png"
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"  [Saved] {plot1_path}")


# ---------------------------------------------------------------------
# Visualization 2: Reconstruction Fidelity Comparison (Original vs AE vs VAE)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=150)
sample_idx = 5
timesteps = np.arange(window_size)

orig_wave = sensors_raw[split_idx + sample_idx]  # (24, 5)

# Standard-AE reconstruction
ae_model = detailed_results["Standard-AE"]["model"]
with torch.no_grad():
    ae_rec_flat, _ = ae_model(x_test[sample_idx:sample_idx+1])
ae_wave = ae_rec_flat.numpy().reshape(window_size, n_channels) * ch_std[0] + ch_mean[0]

# VAE reconstruction
with torch.no_grad():
    vae_rec_flat, _, _, _ = vae_model(x_test[sample_idx:sample_idx+1])
vae_wave = vae_rec_flat.numpy().reshape(window_size, n_channels) * ch_std[0] + ch_mean[0]

# Channel 0: Temperature
axes[0].plot(timesteps, orig_wave[:, 0], "k-", linewidth=2.0, label="Ground Truth Telemetry")
axes[0].plot(timesteps, ae_wave[:, 0], "--", color="#2563EB", linewidth=1.8, label="Standard-AE Reconstruction")
axes[0].plot(timesteps, vae_wave[:, 0], ":", color="#DC2626", linewidth=2.0, label="Variational-VAE Reconstruction")
axes[0].set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
axes[0].set_title(f"Telemetry Reconstruction Comparison (Equipment State: {class_names[int(y_test[sample_idx])]})", fontsize=12, fontweight="bold")
axes[0].legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)

# Channel 1: Vibration
axes[1].plot(timesteps, orig_wave[:, 1], "k-", linewidth=2.0, label="Ground Truth Telemetry")
axes[1].plot(timesteps, ae_wave[:, 1], "--", color="#2563EB", linewidth=1.8, label="Standard-AE")
axes[1].plot(timesteps, vae_wave[:, 1], ":", color="#DC2626", linewidth=2.0, label="Variational-VAE")
axes[1].set_ylabel("Vibration (g)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Window Timesteps (Hours)", fontsize=11)
axes[1].legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)

plt.tight_layout()
plot2_path = PLOTS_DIR / "vae_reconstruction_vs_original.png"
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"  [Saved] {plot2_path}")


# ---------------------------------------------------------------------
# Visualization 3: Controlled Synthetic Generation (C-VAE)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, dpi=150)
for i, cname in enumerate(class_names):
    gen_samples = generated_by_class[cname]  # (5, 24, 5)
    for s_idx in range(len(gen_samples)):
        axes[i].plot(timesteps, gen_samples[s_idx, :, 0], color=colors[i], alpha=0.6, linewidth=1.5)
    # Mean of generated
    axes[i].plot(timesteps, np.mean(gen_samples[:, :, 0], axis=0), color="black", linewidth=2.5, linestyle="--", label="Mean Trajectory")
    axes[i].set_title(f"Generated: {cname}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Timesteps (Hours)", fontsize=10)
    axes[i].legend(loc="upper left", fontsize=9)

axes[0].set_ylabel("Generated Temperature (°C)", fontsize=11, fontweight="bold")
fig.suptitle("EPM Stage 7: Conditional VAE (C-VAE) Targeted Trajectory Generation", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plot3_path = PLOTS_DIR / "cvae_conditional_generation.png"
plt.savefig(plot3_path, dpi=150)
plt.close()
print(f"  [Saved] {plot3_path}")


# ---------------------------------------------------------------------
# Visualization 4: Monte-Carlo Posterior Uncertainty Bounds
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
ax.plot(timesteps, mc_mean[:, 0], color="#2563EB", linewidth=2.0, label="Expected Posterior Decoding μ(t)")
ax.fill_between(
    timesteps,
    mc_mean[:, 0] - 2 * mc_std[:, 0],
    mc_mean[:, 0] + 2 * mc_std[:, 0],
    color="#93C5FD",
    alpha=0.5,
    label="95% Epistemic Credible Band (μ ± 2σ)",
)
ax.plot(timesteps, orig_wave[:, 0], "k--", linewidth=1.5, label="Actual Observed Telemetry")
ax.set_title("EPM Stage 7: Monte-Carlo Latent Uncertainty Quantification", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Window Timesteps (Hours)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
ax.legend(frameon=True, facecolor="white", fontsize=10, loc="upper left")
plt.tight_layout()
plot4_path = PLOTS_DIR / "monte_carlo_uncertainty_bands.png"
plt.savefig(plot4_path, dpi=150)
plt.close()
print(f"  [Saved] {plot4_path}")


# =====================================================================
# Step 8: Persisting Benchmark Artifacts
# =====================================================================
benchmark_file = BENCHMARKS_DIR / "stage7_generative_leaderboard.json"
with open(benchmark_file, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary_df.to_dict(orient="records"),
        "numerical_gradient_check": {
            "max_relative_error": float(max_rel_error),
            "status": "PASSED",
        },
    }, f, indent=2)

print(f"\n[Artifact Saved] Benchmark results written to {benchmark_file}")
print("\n" + "=" * 78)
print("  Stage 7 Execution Complete: Generative Modeling Foundations Verified!")
print("=" * 78)
