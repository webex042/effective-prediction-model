"""
EPM Multimodal Ablation & Robustness Study Suite
================================================

Conducts systematic ablation studies across unimodal baselines and multimodal fusion
architectures (Concatenation, Gated Multimodal Unit, and Cross-Attention).

EVALUATION MODES:
  1. Clean Evaluation: Both sensor telemetry and textual logs are complete.
  2. Missing Text: Simulates scenarios where operators did not file incident logs.
  3. Noisy Sensors: Simulates sensor degradation / electrical noise.
  4. Missing Sensors: Simulates complete hardware communication failure.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.multimodal.models import (
    SensorEncoder,
    TextEncoder,
    ConcatenationFusion,
    GatedMultimodalFusion,
    CrossAttentionFusion,
    MultimodalClassifier,
    SensorOnlyClassifier,
    TextOnlyClassifier,
)


# =============================================================================
# 1. Multiclass Evaluation Utilities
# =============================================================================

def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> Dict[str, Any]:
    """
    Compute multiclass classification metrics (Accuracy, Macro Precision, Recall, F1)
    and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    n = len(y_true)

    # Accuracy
    acc = float(np.mean(y_true == y_pred))

    # Per-class & Macro metrics
    precisions = []
    recalls = []
    f1s = []

    # Confusion matrix: rows = true, cols = pred
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": [float(x) for x in f1s],
        "confusion_matrix": cm.tolist(),
    }


# =============================================================================
# 2. Training and Evaluation Routines
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
    """
    Train a multimodal or unimodal classifier using AdamW and Cross-Entropy loss.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    for epoch in range(epochs):
        for batch in train_loader:
            sensor_x, text_x, labels = batch
            sensor_x = sensor_x.to(device)
            text_x = text_x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if isinstance(model, SensorOnlyClassifier):
                logits, _ = model(sensor_x)
            elif isinstance(model, TextOnlyClassifier):
                logits, _ = model(text_x=text_x)
            else:
                logits, _ = model(sensor_x, text_x)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    train_time = time.perf_counter() - start_time
    return model, train_time


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    missing_modality: Optional[Literal["sensor", "text"]] = None,
    sensor_noise_std: float = 0.0,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Evaluate a model under normal or corrupted modal regimes.

    Parameters
    ----------
    model : nn.Module
    data_loader : DataLoader
    missing_modality : Optional[str]
        'sensor' or 'text' to zero out that modality.
    sensor_noise_std : float
        Additive Gaussian noise standard deviation applied to sensor inputs.
    device : str

    Returns
    -------
    metrics : Dict[str, Any]
    all_preds : np.ndarray
    all_targets : np.ndarray
    aux_info : Dict[str, Any] (e.g. gates or attention weights)
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    gate_values = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in data_loader:
            sensor_x, text_x, labels = batch
            sensor_x = sensor_x.to(device)
            text_x = text_x.to(device)

            # Apply sensor noise if specified
            if sensor_noise_std > 0.0:
                noise = torch.randn_like(sensor_x) * sensor_noise_std
                sensor_x = sensor_x + noise

            if isinstance(model, SensorOnlyClassifier):
                if missing_modality == "sensor":
                    sensor_x = torch.zeros_like(sensor_x)
                logits, aux = model(sensor_x)
            elif isinstance(model, TextOnlyClassifier):
                if missing_modality == "text":
                    text_x = torch.zeros_like(text_x)
                logits, aux = model(text_x=text_x)
            else:
                logits, aux = model(
                    sensor_x, text_x, missing_modality=missing_modality
                )
                if "sensor_weight" in aux:
                    gate_values.append(aux["sensor_weight"].cpu().numpy())

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    infer_time = time.perf_counter() - start_time

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = compute_multiclass_metrics(y_true, y_pred, num_classes=3)
    metrics["latency_ms"] = (infer_time / len(data_loader)) * 1000.0

    aux_info = {}
    if gate_values:
        aux_info["sensor_gates"] = np.vstack(gate_values).ravel()

    return metrics, y_pred, y_true, aux_info


# =============================================================================
# 3. Comprehensive Multimodal Ablation Study
# =============================================================================

def run_multimodal_ablation_study(
    train_dataset: Dataset,
    test_dataset: Dataset,
    vocab_size: int,
    sensor_dim: int = 5,
    latent_dim: int = 64,
    epochs: int = 25,
    batch_size: int = 32,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run end-to-end ablation comparing 5 model architectures across 4 modal regimes:
      Models:
        1. Sensor-Only (Unimodal baseline)
        2. Text-Only (Unimodal baseline)
        3. Multimodal Concat
        4. Multimodal Gated (GMU)
        5. Multimodal Cross-Attention

      Regimes:
        - Clean (Full data)
        - Missing Text (Text logs absent)
        - Noisy Sensor (Sensor corrupted with severe noise)
        - Missing Sensor (Sensors absent)

    Returns
    -------
    summary_df : pd.DataFrame
        Ranked comparison of models across clean and corrupted conditions.
    study_results : Dict[str, Any]
        Complete performance dictionary, gates, and predictions.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    model_factories = {
        "Sensor-Only": lambda: SensorOnlyClassifier(
            sensor_dim=sensor_dim, latent_dim=latent_dim, num_classes=3
        ),
        "Text-Only": lambda: TextOnlyClassifier(
            vocab_size=vocab_size, latent_dim=latent_dim, num_classes=3
        ),
        "Multimodal-Concat": lambda: MultimodalClassifier(
            vocab_size=vocab_size,
            sensor_dim=sensor_dim,
            latent_dim=latent_dim,
            fusion_type="concat",
        ),
        "Multimodal-Gated": lambda: MultimodalClassifier(
            vocab_size=vocab_size,
            sensor_dim=sensor_dim,
            latent_dim=latent_dim,
            fusion_type="gated",
        ),
        "Multimodal-CrossAttn": lambda: MultimodalClassifier(
            vocab_size=vocab_size,
            sensor_dim=sensor_dim,
            latent_dim=latent_dim,
            fusion_type="cross_attention",
        ),
    }

    trained_models = {}
    train_times = {}
    param_counts = {}

    # 1. Train all models
    for name, factory in model_factories.items():
        torch.manual_seed(42)
        model = factory()
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[name] = param_count

        print(f"    - Training {name} ({epochs} epochs)...", flush=True)
        model, train_time = train_model(
            model=model,
            train_loader=train_loader,
            epochs=epochs,
            lr=1e-3,
            device=device,
        )
        trained_models[name] = model
        train_times[name] = train_time

    # 2. Evaluate all models across regimes
    records = []
    study_results = {
        "models": {},
        "regimes": ["Clean", "Missing Text", "Noisy Sensor", "Missing Sensor"],
    }

    for name, model in trained_models.items():
        # Clean
        clean_m, clean_p, clean_t, clean_aux = evaluate_model(
            model, test_loader, device=device
        )
        # Missing Text
        mis_text_m, mis_text_p, _, _ = evaluate_model(
            model, test_loader, missing_modality="text", device=device
        )
        # Noisy Sensor
        noise_sensor_m, noise_sensor_p, _, _ = evaluate_model(
            model, test_loader, sensor_noise_std=3.0, device=device
        )
        # Missing Sensor
        mis_sensor_m, mis_sensor_p, _, _ = evaluate_model(
            model, test_loader, missing_modality="sensor", device=device
        )

        study_results["models"][name] = {
            "parameters": param_counts[name],
            "train_time_sec": train_times[name],
            "clean": clean_m,
            "missing_text": mis_text_m,
            "noisy_sensor": noise_sensor_m,
            "missing_sensor": mis_sensor_m,
            "predictions": {
                "clean": clean_p.tolist(),
                "targets": clean_t.tolist(),
            },
            "aux": clean_aux,
        }

        records.append({
            "Model": name,
            "Parameters": param_counts[name],
            "Train Time (s)": round(train_times[name], 3),
            "Latency (ms)": round(clean_m["latency_ms"], 2),
            "Clean Acc": round(clean_m["accuracy"], 4),
            "Clean Macro F1": round(clean_m["macro_f1"], 4),
            "Noisy Sensor F1": round(noise_sensor_m["macro_f1"], 4),
            "Missing Text F1": round(mis_text_m["macro_f1"], 4),
            "Missing Sensor F1": round(mis_sensor_m["macro_f1"], 4),
        })

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values(by="Clean Macro F1", ascending=False).reset_index(drop=True)

    return summary_df, study_results
