"""
Unit Tests for Stage 6 — Multimodal Representation Learning & Fusion
=====================================================================
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.multimodal.dataset import (
    MultimodalEquipmentDataset,
    generate_paired_multimodal_dataset,
)
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
from src.multimodal.ablation import (
    compute_multiclass_metrics,
    train_model,
    evaluate_model,
    run_multimodal_ablation_study,
)


class TestMultimodalDataset:
    """Verifies multimodal dataset generation, alignment, and PyTorch dataset behavior."""

    def test_paired_dataset_generation(self):
        n_samples = 60
        window_size = 24
        max_text_len = 16
        sensors, text_ids, labels, raw_texts, vocab = generate_paired_multimodal_dataset(
            n_samples=n_samples,
            window_size=window_size,
            max_text_len=max_text_len,
            random_seed=42,
        )

        assert sensors.shape == (n_samples, window_size, 5)
        assert text_ids.shape == (n_samples, max_text_len)
        assert labels.shape == (n_samples,)
        assert len(raw_texts) == n_samples
        assert len(vocab) > 10
        # Ensure balanced classes
        for c in [0, 1, 2]:
            assert np.sum(labels == c) == n_samples // 3

    def test_pytorch_dataset_indexing(self):
        sensors, text_ids, labels, _, _ = generate_paired_multimodal_dataset(
            n_samples=30, random_seed=123
        )
        ds = MultimodalEquipmentDataset(sensors, text_ids, labels)
        assert len(ds) == 30

        s_tensor, t_tensor, y_tensor = ds[0]
        assert isinstance(s_tensor, torch.Tensor)
        assert isinstance(t_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)
        assert s_tensor.shape == (24, 5)
        assert t_tensor.shape == (16,)
        assert y_tensor.dim() == 0


class TestModalityEncoders:
    """Verifies unimodal feature encoders for continuous sensors and discrete text."""

    def test_sensor_encoder_shapes_and_gradients(self):
        batch_size, window_size, sensor_dim = 4, 24, 5
        latent_dim = 32
        encoder = SensorEncoder(
            sensor_dim=sensor_dim, hidden_dim=32, latent_dim=latent_dim
        )
        x = torch.randn(batch_size, window_size, sensor_dim, requires_grad=True)

        latent, seq = encoder(x)
        assert latent.shape == (batch_size, latent_dim)
        assert seq.shape == (batch_size, window_size, latent_dim)

        loss = latent.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_text_encoder_shapes_and_gradients(self):
        batch_size, seq_len = 4, 16
        vocab_size = 50
        latent_dim = 32
        encoder = TextEncoder(
            vocab_size=vocab_size, embed_dim=32, hidden_dim=32, latent_dim=latent_dim
        )
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        latent, seq = encoder(token_ids)
        assert latent.shape == (batch_size, latent_dim)
        assert seq.shape == (batch_size, seq_len, latent_dim)

        loss = latent.sum()
        loss.backward()
        # Verify encoder parameters received gradients
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestFusionMechanisms:
    """Verifies Concatenation, Gated GMU, and Cross-Attention fusion blocks."""

    def test_concatenation_fusion(self):
        batch_size, latent_dim = 4, 32
        fusion = ConcatenationFusion(latent_dim=latent_dim, fused_dim=latent_dim)
        h_s = torch.randn(batch_size, latent_dim)
        h_t = torch.randn(batch_size, latent_dim)

        fused, aux = fusion(h_s, h_t)
        assert fused.shape == (batch_size, latent_dim)
        assert aux["fusion_type"] == "concat"

    def test_gated_multimodal_fusion(self):
        batch_size, latent_dim = 4, 32
        fusion = GatedMultimodalFusion(latent_dim=latent_dim, fused_dim=latent_dim)
        h_s = torch.randn(batch_size, latent_dim)
        h_t = torch.randn(batch_size, latent_dim)

        fused, aux = fusion(h_s, h_t)
        assert fused.shape == (batch_size, latent_dim)
        assert aux["fusion_type"] == "gated"

        gate = aux["gate"]
        assert gate.shape == (batch_size, latent_dim)
        assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
        assert aux["sensor_weight"].shape == (batch_size, 1)

    def test_cross_attention_fusion(self):
        batch_size, seq_s, seq_t, latent_dim = 4, 24, 16, 32
        fusion = CrossAttentionFusion(
            latent_dim=latent_dim, num_heads=4, fused_dim=latent_dim
        )
        h_s = torch.randn(batch_size, latent_dim)
        h_t = torch.randn(batch_size, latent_dim)
        seq_sensor = torch.randn(batch_size, seq_s, latent_dim)
        seq_text = torch.randn(batch_size, seq_t, latent_dim)

        fused, aux = fusion(h_s, h_t, seq_sensor=seq_sensor, seq_text=seq_text)
        assert fused.shape == (batch_size, latent_dim)
        assert aux["fusion_type"] == "cross_attention"


class TestClassifiersAndAblation:
    """Verifies unimodal and multimodal models, missing modality ablations, and metrics."""

    @pytest.mark.parametrize("fusion_type", ["concat", "gated", "cross_attention"])
    def test_multimodal_classifier_forward_and_backward(self, fusion_type):
        batch_size, window_size, sensor_dim = 4, 24, 5
        seq_len, vocab_size = 16, 50
        num_classes = 3

        model = MultimodalClassifier(
            vocab_size=vocab_size,
            sensor_dim=sensor_dim,
            latent_dim=32,
            num_classes=num_classes,
            fusion_type=fusion_type,
        )
        sensor_x = torch.randn(batch_size, window_size, sensor_dim)
        text_x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Clean forward pass
        logits, aux = model(sensor_x, text_x)
        assert logits.shape == (batch_size, num_classes)
        loss = logits.sum()
        loss.backward()

        # Missing modality passes
        logits_mis_text, _ = model(sensor_x, text_x, missing_modality="text")
        assert logits_mis_text.shape == (batch_size, num_classes)

        logits_mis_sensor, _ = model(sensor_x, text_x, missing_modality="sensor")
        assert logits_mis_sensor.shape == (batch_size, num_classes)

    def test_unimodal_classifiers(self):
        batch_size, window_size, sensor_dim = 4, 24, 5
        seq_len, vocab_size = 16, 50

        sensor_model = SensorOnlyClassifier(
            sensor_dim=sensor_dim, latent_dim=32, num_classes=3
        )
        text_model = TextOnlyClassifier(
            vocab_size=vocab_size, latent_dim=32, num_classes=3
        )

        sensor_x = torch.randn(batch_size, window_size, sensor_dim)
        text_x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits_s, _ = sensor_model(sensor_x)
        logits_t, _ = text_model(text_x=text_x)

        assert logits_s.shape == (batch_size, 3)
        assert logits_t.shape == (batch_size, 3)

    def test_multiclass_metrics_computation(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_multiclass_metrics(y_true, y_pred, num_classes=3)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

        y_wrong = np.array([1, 2, 0, 1, 2, 0])
        metrics_wrong = compute_multiclass_metrics(y_true, y_wrong, num_classes=3)
        assert metrics_wrong["accuracy"] == 0.0
        assert metrics_wrong["macro_f1"] == 0.0

    def test_mini_ablation_study_execution(self):
        sensors, text_ids, labels, _, vocab = generate_paired_multimodal_dataset(
            n_samples=48, window_size=24, max_text_len=16, random_seed=42
        )
        train_ds = MultimodalEquipmentDataset(sensors[:36], text_ids[:36], labels[:36])
        test_ds = MultimodalEquipmentDataset(sensors[36:], text_ids[36:], labels[36:])

        summary_df, results = run_multimodal_ablation_study(
            train_dataset=train_ds,
            test_dataset=test_ds,
            vocab_size=len(vocab),
            sensor_dim=5,
            latent_dim=16,
            epochs=2,
            batch_size=12,
        )

        assert len(summary_df) == 5
        expected_cols = [
            "Model",
            "Parameters",
            "Train Time (s)",
            "Latency (ms)",
            "Clean Acc",
            "Clean Macro F1",
            "Noisy Sensor F1",
            "Missing Text F1",
            "Missing Sensor F1",
        ]
        for col in expected_cols:
            assert col in summary_df.columns
        assert "models" in results
