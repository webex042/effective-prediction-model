"""
Unit Tests for Stage 12 — Model Compression
============================================
Tests Knowledge Distillation, Quantization, and Pruning modules.
"""

import math
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.compression.distillation import (
    KnowledgeDistillationLoss,
    DistillationTrainer,
    FeatureHintProjector,
)
from src.compression.quantization import (
    calculate_scale_and_zero_point,
    quantize_tensor,
    dequantize_tensor,
    quantize_dequantize,
    compute_sqnr,
    get_model_size_kb,
    quantize_model_dynamic_int8,
    profile_layer_quantization_errors,
)
from src.compression.pruning import (
    calculate_sparsity,
    prune_unstructured_magnitude,
    StructuredMLPCompressor,
    iterative_prune_and_finetune,
)
from src.compression.benchmarks import (
    BenchmarkClassifier,
    generate_synthetic_industrial_data,
    run_stage12_benchmarks,
)


class TestDistillation:
    """Verifies knowledge distillation loss, temperature scaling, and gradient flow."""

    def test_distillation_loss_classification_shapes_and_values(self):
        B, C = 8, 4
        student_logits = torch.randn(B, C, requires_grad=True)
        teacher_logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        criterion = KnowledgeDistillationLoss(temperature=3.0, alpha=0.4, task="classification")
        loss_dict = criterion(student_logits, teacher_logits, targets)

        assert "total_loss" in loss_dict
        assert "hard_loss" in loss_dict
        assert "soft_loss" in loss_dict
        assert loss_dict["total_loss"].item() > 0.0

        # Gradient flow check: student_logits receives gradient, teacher receives none
        loss_dict["total_loss"].backward()
        assert student_logits.grad is not None
        assert not torch.isnan(student_logits.grad).any()

    def test_distillation_temperature_scaling_softening(self):
        logits = torch.tensor([[10.0, 0.0, -10.0]])
        # At T=1, probabilities are very peaked
        p_t1 = torch.softmax(logits / 1.0, dim=-1)
        # At T=10, probabilities are much softer / flatter
        p_t10 = torch.softmax(logits / 10.0, dim=-1)

        entropy_t1 = -(p_t1 * torch.log(p_t1 + 1e-9)).sum().item()
        entropy_t10 = -(p_t10 * torch.log(p_t10 + 1e-9)).sum().item()

        assert entropy_t10 > entropy_t1

    def test_distillation_regression_task(self):
        B, D = 10, 5
        student_preds = torch.randn(B, D, requires_grad=True)
        teacher_preds = torch.randn(B, D)
        targets = torch.randn(B, D)

        criterion = KnowledgeDistillationLoss(alpha=0.5, task="regression")
        loss_dict = criterion(student_preds, teacher_preds, targets)

        assert loss_dict["total_loss"].item() >= 0.0
        loss_dict["total_loss"].backward()
        assert student_preds.grad is not None

    def test_feature_hint_distillation(self):
        B, S_dim, T_dim = 4, 16, 32
        s_feat = torch.randn(B, S_dim, requires_grad=True)
        t_feat = torch.randn(B, T_dim)
        projector = FeatureHintProjector(student_dim=S_dim, teacher_dim=T_dim)

        s_logits = torch.randn(B, 3, requires_grad=True)
        t_logits = torch.randn(B, 3)
        targets = torch.randint(0, 3, (B,))

        criterion = KnowledgeDistillationLoss(
            temperature=2.0, alpha=0.3, task="classification", feature_weight=0.5
        )
        loss_dict = criterion(
            s_logits,
            t_logits,
            targets,
            student_features=s_feat,
            teacher_features=t_feat,
            feature_projector=projector,
        )

        assert loss_dict["hint_loss"].item() > 0.0
        loss_dict["total_loss"].backward()
        assert s_feat.grad is not None


class TestQuantization:
    """Verifies affine quantization arithmetic, SQNR math, and PTQ dynamic INT8."""

    def test_scale_and_zero_point_calculation(self):
        x_min, x_max = -2.0, 6.0
        scale, zp, q_min, q_max = calculate_scale_and_zero_point(
            x_min, x_max, num_bits=8, symmetric=False
        )
        assert scale > 0
        assert q_min <= zp <= q_max
        assert q_min == 0 and q_max == 255

    def test_quantize_dequantize_roundtrip_fidelity(self):
        x = torch.linspace(-3.0, 3.0, 1000)
        x_recon, sqnr = quantize_dequantize(x, num_bits=8, symmetric=True)

        assert x_recon.shape == x.shape
        # 8-bit uniform quantization should have SQNR > 35 dB
        assert sqnr > 35.0
        # Max reconstruction error should be very small (< 0.05)
        max_err = torch.max(torch.abs(x - x_recon)).item()
        assert max_err < 0.05

    def test_sqnr_increases_with_bit_width(self):
        x = torch.randn(2000)
        _, sqnr_4 = quantize_dequantize(x, num_bits=4, symmetric=True)
        _, sqnr_8 = quantize_dequantize(x, num_bits=8, symmetric=True)
        _, sqnr_16 = quantize_dequantize(x, num_bits=16, symmetric=True)

        assert sqnr_4 < sqnr_8 < sqnr_16

    def test_dynamic_quantize_int8_model(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        fp32_size = get_model_size_kb(model)
        quant_model = quantize_model_dynamic_int8(model)

        x = torch.randn(4, 32)
        out_fp32 = model(x)
        out_quant = quant_model(x)

        assert out_quant.shape == (4, 10)
        # Numerical predictions should be very close
        diff = torch.mean(torch.abs(out_fp32 - out_quant)).item()
        assert diff < 0.05


class TestPruning:
    """Verifies magnitude pruning, structured contraction, and iterative schedules."""

    def test_unstructured_magnitude_pruning_sparsity(self):
        model = nn.Sequential(
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
        )
        # Apply 60% unstructured magnitude pruning
        prune_unstructured_magnitude(model, sparsity=0.60)
        info = calculate_sparsity(model)

        assert abs(info["global_sparsity"] - 0.60) < 0.02
        assert info["zero_weight_elements"] > 0

    def test_structured_mlp_compressor(self):
        fc1 = nn.Linear(16, 32)
        fc2 = nn.Linear(32, 8)

        new_fc1, new_fc2 = StructuredMLPCompressor.prune_mlp_hidden_layer(
            fc1, fc2, keep_ratio=0.5
        )

        # 32 hidden units contracted to 16
        assert new_fc1.out_features == 16
        assert new_fc2.in_features == 16
        assert new_fc1.in_features == 16
        assert new_fc2.out_features == 8

        # Test forward pass with contracted layers
        x = torch.randn(2, 16)
        h = torch.relu(new_fc1(x))
        out = new_fc2(h)
        assert out.shape == (2, 8)

    def test_iterative_prune_and_finetune_routine(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        x_dummy = torch.randn(50, 10)
        y_dummy = torch.randint(0, 2, (50,))

        def dummy_train(m, epochs=1):
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            for _ in range(epochs):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(m(x_dummy), y_dummy)
                loss.backward()
                opt.step()

        def dummy_eval(m):
            with torch.no_grad():
                preds = m(x_dummy).argmax(dim=-1)
                return (preds == y_dummy).float().mean().item()

        res = iterative_prune_and_finetune(
            model,
            train_fn=dummy_train,
            eval_fn=dummy_eval,
            target_sparsity=0.5,
            num_steps=2,
            finetune_epochs=1,
        )

        assert len(res["history"]) == 3
        assert res["final_sparsity"] > 0.45


class TestCompressionIntegration:
    """Verifies benchmark execution and end-to-end compression pipeline."""

    def test_synthetic_data_generation(self):
        train_ds, test_ds = generate_synthetic_industrial_data(
            num_samples=200, in_dim=16, num_classes=3, seed=123
        )
        assert len(train_ds) == 140
        assert len(test_ds) == 60
        assert train_ds.tensors[0].shape == (140, 16)

    def test_run_stage12_benchmarks_quick(self):
        # Quick run with small dimensions and epochs
        results = run_stage12_benchmarks(
            in_dim=16,
            num_classes=3,
            epochs_teacher=5,
            epochs_student=5,
            seed=42,
        )
        assert "benchmark_records" in results
        records = results["benchmark_records"]
        assert len(records) == 8

        variants = [r["variant"] for r in records]
        assert "Teacher (Full FP32)" in variants
        assert "Distilled Student (KD FP32)" in variants
        assert "Dynamic INT8 Quantized (PTQ)" in variants
        assert "Tri-Compressed Edge Model" in variants
