"""
EPM Diffusion Benchmark Suite
=============================

Benchmarks diffusion models against deterministic and VAE baselines:
- Training convergence & loss profiling
- Sampling latency: DDPM (Full steps) vs DDIM (20 steps) vs DDIM (10 steps)
- Quality metrics: Reconstruction fidelity, MMD / distribution coverage
- Epistemic uncertainty quantification across stochastic diffusion trajectories
"""

from __future__ import annotations

import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.diffusion.schedules import DiffusionSchedule
from src.diffusion.models import DenoisingMLP
from src.diffusion.gaussian_diffusion import GaussianDiffusion


def train_diffusion_model(
    diffusion: GaussianDiffusion,
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    print_every: int = 5,
) -> List[float]:
    """
    Trains a Denoising Score Network using Adam optimizer.

    Returns
    -------
    List[float]
        Epoch average training loss curve.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_history = []

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0].to(device)
            c = batch[1].to(device) if len(batch) > 1 else None

            optimizer.zero_grad()
            loss = diffusion.compute_training_loss(model, x, c=c)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        loss_history.append(avg_loss)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print(f"    Epoch {epoch:02d}/{epochs:02d} | Diffusion Loss (MSE): {avg_loss:.5f}")

    return loss_history


def compute_multivariate_mmd(x_real: np.ndarray, x_gen: np.ndarray, gamma: float = 1.0) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD) with RBF kernel between real and generated distributions.
    """
    # Sample subset if arrays are large for efficiency
    n = min(len(x_real), len(x_gen), 200)
    x1 = x_real[:n]
    x2 = x_gen[:n]

    def rbf(a, b):
        dist_sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * dist_sq)

    k_xx = rbf(x1, x1)
    k_yy = rbf(x2, x2)
    k_xy = rbf(x1, x2)

    mmd_sq = np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy)
    return float(np.sqrt(max(0.0, mmd_sq)))


def run_diffusion_benchmark(
    diffusion: GaussianDiffusion,
    model: DenoisingMLP,
    test_tensor: torch.Tensor,
    labels_tensor: Optional[torch.Tensor] = None,
    num_samples: int = 64,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes comparative benchmark across sampling methods:
    1. DDPM (Full schedule: 100 steps)
    2. DDIM (20 steps, deterministic eta=0.0)
    3. DDIM (10 steps, deterministic eta=0.0)
    4. DDIM (20 steps, stochastic eta=0.5)
    """
    model.eval()
    model.to(device)

    input_dim = test_tensor.shape[1]
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x_real_np = test_tensor.cpu().numpy()
    c_sample = None
    if labels_tensor is not None:
        c_sample = labels_tensor[:num_samples].to(device)

    results = []
    generated_samples_dict = {}

    # Configurations to test
    configs = [
        {"name": "DDPM (Full SDE)", "method": "ddpm", "steps": diffusion.timesteps, "eta": 1.0},
        {"name": "DDIM-20 (Deterministic ODE)", "method": "ddim", "steps": 20, "eta": 0.0},
        {"name": "DDIM-10 (Fast ODE)", "method": "ddim", "steps": 10, "eta": 0.0},
        {"name": "DDIM-20 (Stochastic)", "method": "ddim", "steps": 20, "eta": 0.5},
    ]

    for cfg in configs:
        shape = (num_samples, input_dim)
        torch.manual_seed(42)

        start_t = time.perf_counter()
        if cfg["method"] == "ddpm":
            samples = diffusion.sample_ddpm(model, shape=shape, c=c_sample, device=device)
        else:
            samples = diffusion.sample_ddim(
                model, shape=shape, c=c_sample, num_steps=cfg["steps"], eta=cfg["eta"], device=device
            )
        elapsed_s = time.perf_counter() - start_t

        samples_np = samples.cpu().numpy()
        generated_samples_dict[cfg["name"]] = samples_np

        latency_total_ms = elapsed_s * 1000.0
        latency_per_sample_ms = latency_total_ms / num_samples
        latency_per_step_ms = latency_total_ms / cfg["steps"]

        # Compute MMD against real test telemetry
        mmd_score = compute_multivariate_mmd(x_real_np, samples_np)
        mean_abs_val = float(np.mean(np.abs(samples_np)))
        sample_std = float(np.std(samples_np))

        results.append({
            "Method": cfg["name"],
            "Steps": cfg["steps"],
            "Eta (Stochasticity)": cfg["eta"],
            "Total Latency (ms)": round(latency_total_ms, 2),
            "Latency / Sample (ms)": round(latency_per_sample_ms, 2),
            "Latency / Step (ms)": round(latency_per_step_ms, 2),
            "MMD vs Real Telemetry": round(mmd_score, 4),
            "Mean Amplitude": round(mean_abs_val, 3),
            "Output Variance": round(sample_std, 3),
            "Model Parameters": f"{param_count:,}",
        })

    summary_df = pd.DataFrame(results)
    detailed_data = {
        "summary": results,
        "input_dim": input_dim,
        "parameters": param_count,
        "num_eval_samples": num_samples,
    }

    return summary_df, detailed_data
