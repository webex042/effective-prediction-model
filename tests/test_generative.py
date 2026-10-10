"""
Unit Tests for Stage 7 — Generative Models (AE, VAE, C-VAE)
===========================================================
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.generative.vae_scratch import (
    VAEScratch,
    check_vae_numerical_gradients,
)
from src.generative.models import (
    StandardAutoencoder,
    VariationalAutoencoder,
    ConditionalVAE,
    train_generative_model,
)
from src.generative.benchmarks import (
    evaluate_generative_model,
    run_generative_benchmark,
)


class TestVAEScratch:
    """Verifies from-scratch NumPy VAE, ELBO formulation, and numerical gradients."""

    def test_vae_scratch_forward_shapes(self):
        batch_size, input_dim, latent_dim = 8, 20, 4
        vae = VAEScratch(input_dim=input_dim, hidden_dim=16, latent_dim=latent_dim, random_seed=42)
        x = np.random.randn(batch_size, input_dim)

        x_recon, mu, log_var, z = vae.forward(x)
        assert x_recon.shape == (batch_size, input_dim)
        assert mu.shape == (batch_size, latent_dim)
        assert log_var.shape == (batch_size, latent_dim)
        assert z.shape == (batch_size, latent_dim)

    def test_vae_scratch_loss_and_backward_shapes(self):
        batch_size, input_dim, latent_dim = 6, 12, 3
        vae = VAEScratch(input_dim=input_dim, hidden_dim=10, latent_dim=latent_dim, random_seed=42)
        x = np.random.randn(batch_size, input_dim)

        x_recon, mu, log_var, z = vae.forward(x)
        total_loss, recon_loss, kl_loss = vae.compute_loss(x, x_recon, mu, log_var)

        assert total_loss > 0.0
        assert recon_loss >= 0.0
        assert kl_loss >= 0.0
        assert abs(total_loss - (recon_loss + kl_loss)) < 1e-6

        grads = vae.backward()
        for name in ["W1", "b1", "W_mu", "b_mu", "W_logvar", "b_logvar", "W2", "b2", "W3", "b3"]:
            param = getattr(vae, name)
            assert grads[name].shape == param.shape

    def test_vae_scratch_numerical_gradient_checking(self):
        """Verifies analytical backpropagation matches finite-difference numerical gradients."""
        max_rel_error = check_vae_numerical_gradients(
            input_dim=6,
            hidden_dim=8,
            latent_dim=3,
            h=1e-5,
        )
        # Relative error should be tiny (< 1e-4) for analytical backprop correctness
        assert max_rel_error < 1e-4, f"Numerical gradient check failed with rel error: {max_rel_error}"

    def test_vae_scratch_prior_sampling(self):
        vae = VAEScratch(input_dim=15, hidden_dim=12, latent_dim=4, random_seed=42)
        samples = vae.sample_prior(n_samples=5)
        assert samples.shape == (5, 15)


class TestPyTorchGenerativeModels:
    """Verifies PyTorch production implementations of AE, VAE, and Conditional VAE."""

    def test_standard_autoencoder(self):
        batch_size, input_dim, latent_dim = 8, 30, 4
        ae = StandardAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        x = torch.randn(batch_size, input_dim)

        x_recon, z = ae(x)
        assert x_recon.shape == (batch_size, input_dim)
        assert z.shape == (batch_size, latent_dim)

        score = ae.anomaly_score(x)
        assert score.shape == (batch_size,)
        assert torch.all(score >= 0.0)

        # Gradient flow
        loss = F.mse_loss(x_recon, x)
        loss.backward()
        for p in ae.parameters():
            assert p.grad is not None

    def test_variational_autoencoder(self):
        batch_size, input_dim, latent_dim = 8, 30, 4
        vae = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim, beta=1.0)
        x = torch.randn(batch_size, input_dim)

        x_recon, mu, log_var, z = vae(x)
        assert x_recon.shape == (batch_size, input_dim)
        assert mu.shape == (batch_size, latent_dim)
        assert log_var.shape == (batch_size, latent_dim)
        assert z.shape == (batch_size, latent_dim)

        loss_dict = vae.compute_loss(x, x_recon, mu, log_var)
        assert "loss" in loss_dict
        assert "recon_loss" in loss_dict
        assert "kl_loss" in loss_dict

        loss = loss_dict["loss"]
        loss.backward()
        for p in vae.parameters():
            assert p.grad is not None

    def test_vae_prior_sampling_and_uncertainty(self):
        input_dim, latent_dim = 25, 4
        vae = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        samples = vae.sample_prior(num_samples=7)
        assert samples.shape == (7, input_dim)

        x = torch.randn(3, input_dim)
        mean_recon, std_recon = vae.sample_with_uncertainty(x, num_mc_samples=15)
        assert mean_recon.shape == (3, input_dim)
        assert std_recon.shape == (3, input_dim)
        assert torch.all(std_recon >= 0.0)

    def test_conditional_vae(self):
        batch_size, input_dim, num_classes, latent_dim = 8, 25, 3, 4
        cvae = ConditionalVAE(input_dim=input_dim, num_classes=num_classes, latent_dim=latent_dim)
        x = torch.randn(batch_size, input_dim)
        y = torch.randint(0, num_classes, (batch_size,))

        x_recon, mu, log_var, z = cvae(x, y)
        assert x_recon.shape == (batch_size, input_dim)
        assert mu.shape == (batch_size, latent_dim)

        loss = F.mse_loss(x_recon, x)
        loss.backward()
        for p in cvae.parameters():
            assert p.grad is not None

        # Test class-conditional generation
        cond_samples = cvae.sample_conditional(class_label=2, num_samples=5)
        assert cond_samples.shape == (5, input_dim)


class TestGenerativeBenchmark:
    """Verifies end-to-end training and benchmarking harness."""

    def test_mini_generative_benchmark(self):
        n_samples, input_dim, latent_dim = 32, 20, 4
        x = torch.randn(n_samples, input_dim)
        y = torch.randint(0, 3, (n_samples,))

        train_ds = TensorDataset(x[:24], x[:24], y[:24])
        test_ds = TensorDataset(x[24:], x[24:], y[24:])

        summary_df, results = run_generative_benchmark(
            train_dataset=train_ds,
            test_dataset=test_ds,
            input_dim=input_dim,
            latent_dim=latent_dim,
            epochs=2,
            batch_size=12,
        )

        assert len(summary_df) == 3
        expected_cols = [
            "Model",
            "Latent Dim",
            "Parameters",
            "Train Time (s)",
            "Latency (ms)",
            "Recon MSE",
            "Recon MAE",
            "Anomaly AUROC",
        ]
        for col in expected_cols:
            assert col in summary_df.columns
        assert "Standard-AE" in results
        assert "Variational-VAE" in results
        assert "Conditional-VAE" in results
