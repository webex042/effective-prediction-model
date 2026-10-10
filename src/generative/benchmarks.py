"""
EPM Generative Modeling Benchmarking Suite
==========================================

Systematic evaluation comparing Deterministic Autoencoder vs Variational Autoencoder
vs Conditional VAE on multivariate industrial telemetry:
  1. Reconstruction Error (MSE, MAE, R²)
  2. Latent Compression Ratio & Parameter Count
  3. Generative Anomaly Detection (AUROC on fault detection via reconstruction residuals)
  4. Inference Latency & Computational Efficiency
"""

from __future__ import annotations

import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.generative.models import (
    StandardAutoencoder,
    VariationalAutoencoder,
    ConditionalVAE,
    train_generative_model,
)
from src.evaluation.metrics import roc_auc, mae, mse


def evaluate_generative_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate generative model on test partition for reconstruction accuracy,
    anomaly detection ability, and inference speed.
    """
    model.to(device)
    model.eval()

    all_x = []
    all_recon = []
    all_scores = []
    all_labels = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                y = batch[2].to(device) if len(batch) >= 3 else None
            else:
                x = batch.to(device)
                y = None

            if isinstance(model, ConditionalVAE):
                x_recon, _, _, _ = model(x, y)
                diff = (x - x_recon).view(x.size(0), -1)
                scores = torch.mean(diff ** 2, dim=-1)
            elif isinstance(model, VariationalAutoencoder):
                x_recon, _, _, _ = model(x)
                scores = model.anomaly_score(x)
            elif isinstance(model, StandardAutoencoder):
                x_recon, _ = model(x)
                scores = model.anomaly_score(x)
            else:
                raise TypeError(f"Unknown model: {type(model)}")

            all_x.append(x.cpu().numpy())
            all_recon.append(x_recon.cpu().numpy())
            all_scores.extend(scores.cpu().numpy().tolist())
            if y is not None:
                all_labels.extend(y.cpu().numpy().tolist())

    elapsed = time.perf_counter() - start_time

    x_true = np.concatenate(all_x, axis=0)
    x_pred = np.concatenate(all_recon, axis=0)

    x_true_flat = x_true.reshape(x_true.shape[0], -1)
    x_pred_flat = x_pred.reshape(x_pred.shape[0], -1)

    rec_mse = float(np.mean((x_true_flat - x_pred_flat) ** 2))
    rec_mae = float(np.mean(np.abs(x_true_flat - x_pred_flat)))

    # Compute AUROC for anomaly detection: Nominal (0) vs Critical Fault (2)
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    auroc = 0.5
    if len(labels_arr) > 0:
        mask = (labels_arr == 0) | (labels_arr == 2)
        if np.any(labels_arr == 0) and np.any(labels_arr == 2):
            bin_true = (labels_arr[mask] == 2).astype(int)
            bin_scores = scores_arr[mask]
            auroc = roc_auc(bin_true, bin_scores)

    latency_ms = (elapsed / len(test_loader)) * 1000.0

    return {
        "reconstruction_mse": rec_mse,
        "reconstruction_mae": rec_mae,
        "anomaly_auroc": auroc,
        "latency_ms": latency_ms,
        "x_true": x_true,
        "x_pred": x_pred,
        "anomaly_scores": scores_arr,
        "labels": labels_arr,
    }


def run_generative_benchmark(
    train_dataset: Dataset,
    test_dataset: Dataset,
    input_dim: int = 120,
    latent_dim: int = 8,
    epochs: int = 20,
    batch_size: int = 32,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train and benchmark AE, VAE, and C-VAE models.

    Returns
    -------
    summary_df : pd.DataFrame
        Ranked comparison table.
    results : Dict[str, Any]
        Complete results including reconstructions and anomaly scores.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    factories = {
        "Standard-AE": lambda: StandardAutoencoder(
            input_dim=input_dim, latent_dim=latent_dim
        ),
        "Variational-VAE": lambda: VariationalAutoencoder(
            input_dim=input_dim, latent_dim=latent_dim, beta=1.0
        ),
        "Conditional-VAE": lambda: ConditionalVAE(
            input_dim=input_dim, num_classes=3, latent_dim=latent_dim, beta=1.0
        ),
    }

    trained_models = {}
    train_times = {}
    param_counts = {}

    for name, factory in factories.items():
        torch.manual_seed(42)
        model = factory()
        param_counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        start = time.perf_counter()
        model, _ = train_generative_model(
            model=model,
            train_loader=train_loader,
            epochs=epochs,
            lr=1e-3,
            device=device,
        )
        train_times[name] = time.perf_counter() - start
        trained_models[name] = model

    records = []
    detailed_results = {}

    for name, model in trained_models.items():
        eval_dict = evaluate_generative_model(model, test_loader, device=device)
        detailed_results[name] = {
            "parameters": param_counts[name],
            "train_time_sec": train_times[name],
            "metrics": eval_dict,
            "model": model,
        }

        records.append({
            "Model": name,
            "Latent Dim": latent_dim,
            "Parameters": param_counts[name],
            "Train Time (s)": round(train_times[name], 3),
            "Latency (ms)": round(eval_dict["latency_ms"], 2),
            "Recon MSE": round(eval_dict["reconstruction_mse"], 5),
            "Recon MAE": round(eval_dict["reconstruction_mae"], 5),
            "Anomaly AUROC": round(eval_dict["anomaly_auroc"], 4),
        })

    summary_df = pd.DataFrame(records).sort_values(by="Recon MSE", ascending=True).reset_index(drop=True)
    return summary_df, detailed_results
