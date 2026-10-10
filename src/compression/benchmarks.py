"""
EPM Stage 12 — Compression Benchmarks
======================================

Comprehensive comparative evaluation across compression pipelines:
- Teacher Model (FP32 Full Capacity)
- Uncompressed Baseline Student (FP32 Scratch)
- Distilled Student (FP32 Knowledge Distilled)
- Dynamic INT8 Quantized Student (PTQ)
- Unstructured Pruned Student (50% & 75% Sparsity)
- Structured Physically Compact Student
- Tri-Compressed Edge Deployable Model (Distilled + Structured Pruned + INT8)
"""

from __future__ import annotations

import copy
import io
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .distillation import (
    KnowledgeDistillationLoss,
    DistillationTrainer,
    FeatureHintProjector,
)
from .quantization import (
    quantize_tensor,
    dequantize_tensor,
    quantize_dequantize,
    compute_sqnr,
    get_model_size_kb,
    quantize_model_dynamic_int8,
    profile_layer_quantization_errors,
)
from .pruning import (
    calculate_sparsity,
    prune_unstructured_magnitude,
    StructuredMLPCompressor,
)


class BenchmarkClassifier(nn.Module):
    """Standard configurable multi-layer neural classifier for compression experiments."""

    def __init__(self, in_dim: int, hidden_dims: List[int], num_classes: int) -> None:
        super().__init__()
        layers = []
        curr = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            curr = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(curr, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features


def generate_synthetic_industrial_data(
    num_samples: int = 1500,
    in_dim: int = 32,
    num_classes: int = 4,
    noise_std: float = 0.15,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """Generates synthetic multivariate telemetry with distinct class centroids."""
    rng = np.random.RandomState(seed)
    centroids = rng.randn(num_classes, in_dim) * 2.0

    labels = rng.randint(0, num_classes, size=num_samples)
    features = np.zeros((num_samples, in_dim), dtype=np.float32)

    for i in range(num_samples):
        c = labels[i]
        features[i] = centroids[c] + rng.randn(in_dim) * noise_std

    # Split train (70%) and test (30%)
    split_idx = int(num_samples * 0.7)
    X_train = torch.from_numpy(features[:split_idx])
    y_train = torch.from_numpy(labels[:split_idx]).long()

    X_test = torch.from_numpy(features[split_idx:])
    y_test = torch.from_numpy(labels[split_idx:]).long()

    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)


def measure_inference_latency(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int = 20,
    reps: int = 200,
) -> Dict[str, float]:
    """Measures mean latency per sample in microseconds and throughput (samples/sec)."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)

        start = time.perf_counter()
        for _ in range(reps):
            _ = model(sample)
        elapsed = time.perf_counter() - start

    batch_size = sample.size(0)
    total_samples = reps * batch_size
    latency_us_per_sample = (elapsed / total_samples) * 1e6
    throughput = total_samples / elapsed

    return {
        "latency_us_per_sample": float(latency_us_per_sample),
        "latency_ms_per_sample": float(latency_us_per_sample / 1000.0),
        "throughput_samples_sec": float(throughput),
    }


def evaluate_accuracy(model: nn.Module, dataset: TensorDataset) -> float:
    """Computes overall classification accuracy."""
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return float(correct / max(total, 1))


def count_parameters(model: nn.Module) -> int:
    """Counts total learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_stage12_benchmarks(
    in_dim: int = 32,
    num_classes: int = 4,
    epochs_teacher: int = 25,
    epochs_student: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete Stage 12 Model Compression benchmark suite.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, test_ds = generate_synthetic_industrial_data(
        num_samples=2000, in_dim=in_dim, num_classes=num_classes, seed=seed
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    sample_x = test_ds.tensors[0][:1]  # Single sample for latency test

    # 1. Train Teacher Model (Wide & Deep: [128, 128, 64])
    teacher = BenchmarkClassifier(in_dim, [128, 128, 64], num_classes)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=0.005)
    for _ in range(epochs_teacher):
        teacher.train()
        for x, y in train_loader:
            opt_t.zero_grad()
            logits, _ = teacher(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt_t.step()

    teacher_acc = evaluate_accuracy(teacher, test_ds)
    teacher_size_kb = get_model_size_kb(teacher)
    teacher_lat = measure_inference_latency(teacher, sample_x)
    teacher_params = count_parameters(teacher)

    # 2. Train Uncompressed Student Baseline from Scratch (Compact: [32, 16])
    student_base = BenchmarkClassifier(in_dim, [32, 16], num_classes)
    opt_sb = torch.optim.Adam(student_base.parameters(), lr=0.005)
    for _ in range(epochs_student):
        student_base.train()
        for x, y in train_loader:
            opt_sb.zero_grad()
            logits, _ = student_base(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt_sb.step()

    base_acc = evaluate_accuracy(student_base, test_ds)
    base_size_kb = get_model_size_kb(student_base)
    base_lat = measure_inference_latency(student_base, sample_x)
    base_params = count_parameters(student_base)

    # 3. Train Distilled Student via Knowledge Distillation
    student_distilled = BenchmarkClassifier(in_dim, [32, 16], num_classes)
    opt_sd = torch.optim.Adam(student_distilled.parameters(), lr=0.005)
    kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.3, task="classification")
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student_distilled,
        optimizer=opt_sd,
        criterion=kd_loss,
    )
    for _ in range(epochs_student):
        trainer.train_epoch(train_loader)

    dist_acc = evaluate_accuracy(student_distilled, test_ds)
    dist_size_kb = get_model_size_kb(student_distilled)
    dist_lat = measure_inference_latency(student_distilled, sample_x)
    dist_params = count_parameters(student_distilled)

    # 4. INT8 Dynamic Post-Training Quantization of Distilled Student
    # Create single-output wrapper for standard quantize_dynamic
    class SingleOutputClassifier(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = copy.deepcopy(model)

        def forward(self, x):
            out = self.model(x)
            return out[0] if isinstance(out, tuple) else out

    wrapped_dist = SingleOutputClassifier(student_distilled)
    quant_int8 = quantize_model_dynamic_int8(wrapped_dist)
    quant_acc = evaluate_accuracy(quant_int8, test_ds)
    quant_size_kb = get_model_size_kb(quant_int8)
    quant_lat = measure_inference_latency(quant_int8, sample_x)

    # 5. Unstructured Magnitude Pruning (50% and 75% on Student)
    pruned_50 = copy.deepcopy(student_distilled)
    prune_unstructured_magnitude(pruned_50, sparsity=0.50)
    p50_acc = evaluate_accuracy(pruned_50, test_ds)
    p50_sp = calculate_sparsity(pruned_50)["global_sparsity"]

    pruned_75 = copy.deepcopy(student_distilled)
    prune_unstructured_magnitude(pruned_75, sparsity=0.75)
    p75_acc = evaluate_accuracy(pruned_75, test_ds)
    p75_sp = calculate_sparsity(pruned_75)["global_sparsity"]

    # 6. Structured Physical Pruning (Contract hidden layer [32 -> 20])
    struct_student = copy.deepcopy(student_distilled)
    fc0 = struct_student.backbone[0]  # Linear(in_dim, 32)
    fc1 = struct_student.backbone[2]  # Linear(32, 16)
    new_fc0, new_fc1 = StructuredMLPCompressor.prune_mlp_hidden_layer(fc0, fc1, keep_ratio=0.625)
    struct_student.backbone[0] = new_fc0
    struct_student.backbone[2] = new_fc1
    struct_acc = evaluate_accuracy(struct_student, test_ds)
    struct_size_kb = get_model_size_kb(struct_student)
    struct_lat = measure_inference_latency(struct_student, sample_x)
    struct_params = count_parameters(struct_student)

    # 7. Tri-Compressed Edge Deployable Model (Distilled + Structured + INT8)
    wrapped_struct = SingleOutputClassifier(struct_student)
    tri_edge_model = quantize_model_dynamic_int8(wrapped_struct)
    tri_acc = evaluate_accuracy(tri_edge_model, test_ds)
    tri_size_kb = get_model_size_kb(tri_edge_model)
    tri_lat = measure_inference_latency(tri_edge_model, sample_x)

    # Compile benchmark records
    records = [
        {
            "variant": "Teacher (Full FP32)",
            "parameters": teacher_params,
            "size_kb": round(teacher_size_kb, 2),
            "size_reduction": "1.0x",
            "accuracy": round(teacher_acc, 4),
            "acc_retention": "100.0%",
            "latency_us": round(teacher_lat["latency_us_per_sample"], 2),
            "speedup": "1.00x",
            "sparsity": "0.0%",
        },
        {
            "variant": "Student Baseline (Scratch FP32)",
            "parameters": base_params,
            "size_kb": round(base_size_kb, 2),
            "size_reduction": f"{teacher_size_kb / max(base_size_kb, 1e-3):.1f}x",
            "accuracy": round(base_acc, 4),
            "acc_retention": f"{(base_acc / teacher_acc) * 100:.1f}%",
            "latency_us": round(base_lat["latency_us_per_sample"], 2),
            "speedup": f"{teacher_lat['latency_us_per_sample'] / max(base_lat['latency_us_per_sample'], 1e-3):.2f}x",
            "sparsity": "0.0%",
        },
        {
            "variant": "Distilled Student (KD FP32)",
            "parameters": dist_params,
            "size_kb": round(dist_size_kb, 2),
            "size_reduction": f"{teacher_size_kb / max(dist_size_kb, 1e-3):.1f}x",
            "accuracy": round(dist_acc, 4),
            "acc_retention": f"{(dist_acc / teacher_acc) * 100:.1f}%",
            "latency_us": round(dist_lat["latency_us_per_sample"], 2),
            "speedup": f"{teacher_lat['latency_us_per_sample'] / max(dist_lat['latency_us_per_sample'], 1e-3):.2f}x",
            "sparsity": "0.0%",
        },
        {
            "variant": "Dynamic INT8 Quantized (PTQ)",
            "parameters": dist_params,
            "size_kb": round(quant_size_kb, 2),
            "size_reduction": f"{teacher_size_kb / max(quant_size_kb, 1e-3):.1f}x",
            "accuracy": round(quant_acc, 4),
            "acc_retention": f"{(quant_acc / teacher_acc) * 100:.1f}%",
            "latency_us": round(quant_lat["latency_us_per_sample"], 2),
            "speedup": f"{teacher_lat['latency_us_per_sample'] / max(quant_lat['latency_us_per_sample'], 1e-3):.2f}x",
            "sparsity": "0.0%",
        },
        {
            "variant": "Unstructured Pruned (50% Sparse)",
            "parameters": dist_params,
            "size_kb": round(dist_size_kb, 2),
            "size_reduction": f"{teacher_size_kb / max(dist_size_kb, 1e-3):.1f}x",
            "accuracy": round(p50_acc, 4),
            "acc_retention": f"{(p50_acc / teacher_acc) * 100:.1f}%",
            "latency_us": round(dist_lat["latency_us_per_sample"], 2),
            "speedup": f"{teacher_lat['latency_us_per_sample'] / max(dist_lat['latency_us_per_sample'], 1e-3):.2f}x",
            "sparsity": f"{p50_sp * 100:.1f}%",
        },
        {
            "variant": "Unstructured Pruned (75% Sparse)",
            "parameters": dist_params,
            "size_kb": round(dist_size_kb, 2),
            "size_reduction": f"{teacher_size_kb / max(dist_size_kb, 1e-3):.1f}x",
            "accuracy": round(p75_acc, 4),
            "acc_retention": f"{(p75_acc / teacher_acc) * 100:.1f}%",
            "latency_us": round(dist_lat["latency_us_per_sample"], 2),
            "speedup": f"{teacher_lat['latency_us_per_sample'] / max(dist_lat['latency_us_per_sample'], 1e-3):.2f}x",
            "sparsity": f"{p75_sp * 100:.1f}%",
        },
        {
            "variant": "Structured Compact (Physical Pruning)",
            "parameters": struct_params,
            "size_kb": round(struct_size_kb, 2),
            "size_reduction": f"{teacher_size_kb / max(struct_size_kb, 1e-3):.1f}x",
            "accuracy": round(struct_acc, 4),
            "acc_retention": f"{(struct_acc / teacher_acc) * 100:.1f}%",
            "latency_us": round(struct_lat["latency_us_per_sample"], 2),
            "speedup": f"{teacher_lat['latency_us_per_sample'] / max(struct_lat['latency_us_per_sample'], 1e-3):.2f}x",
            "sparsity": "0.0%",
        },
        {
            "variant": "Tri-Compressed Edge Model",
            "parameters": struct_params,
            "size_kb": round(tri_size_kb, 2),
            "size_reduction": f"{teacher_size_kb / max(tri_size_kb, 1e-3):.1f}x",
            "accuracy": round(tri_acc, 4),
            "acc_retention": f"{(tri_acc / teacher_acc) * 100:.1f}%",
            "latency_us": round(tri_lat["latency_us_per_sample"], 2),
            "speedup": f"{teacher_lat['latency_us_per_sample'] / max(tri_lat['latency_us_per_sample'], 1e-3):.2f}x",
            "sparsity": "Structured",
        },
    ]

    return {
        "benchmark_records": records,
        "teacher_accuracy": teacher_acc,
        "student_distilled_accuracy": dist_acc,
        "max_size_reduction": f"{teacher_size_kb / max(tri_size_kb, 1e-3):.1f}x",
        "max_speedup": f"{teacher_lat['latency_us_per_sample'] / max(tri_lat['latency_us_per_sample'], 1e-3):.2f}x",
    }
