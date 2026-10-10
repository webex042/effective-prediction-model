"""
Unit Tests for Stage 8 — Diffusion Models (DDPM & DDIM)
=======================================================
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.diffusion.schedules import DiffusionSchedule
from src.diffusion.models import SinusoidalTimeEmbedding, DenoisingMLP
from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.diffusion.benchmarks import compute_multivariate_mmd


class TestDiffusionSchedules:
    """Verifies variance schedules and analytical diffusion constants."""

    def test_linear_schedule_monotonicity_and_bounds(self):
        timesteps = 50
        schedule = DiffusionSchedule(timesteps=timesteps, schedule_type="linear", beta_start=1e-4, beta_end=0.02)
        assert schedule.betas.shape == (timesteps,)
        assert schedule.alphas.shape == (timesteps,)
        assert schedule.alphas_cumprod.shape == (timesteps,)

        # Betas must be strictly positive and increasing
        assert torch.all(schedule.betas > 0.0)
        assert torch.all(schedule.betas[:-1] <= schedule.betas[1:])

        # Alphas cumprod must monotonically decrease from near 1.0 towards 0.0
        assert torch.all(schedule.alphas_cumprod[:-1] >= schedule.alphas_cumprod[1:])
        assert 0.0 < schedule.alphas_cumprod[-1] < 1.0

    def test_cosine_schedule_properties(self):
        timesteps = 60
        schedule = DiffusionSchedule(timesteps=timesteps, schedule_type="cosine")
        assert schedule.betas.shape == (timesteps,)
        # Betas clamped <= 0.999
        assert torch.all(schedule.betas <= 0.999)
        assert torch.all(schedule.betas >= 1e-5)

        # SNR must be monotonically decreasing
        snr = schedule.get_snr()
        assert snr[0] > snr[-1]

    def test_schedule_device_movement(self):
        schedule = DiffusionSchedule(timesteps=20, schedule_type="cosine")
        schedule.to("cpu")
        assert schedule.alphas_cumprod.device == torch.device("cpu")


class TestDenoisingModels:
    """Verifies Denoising Score Networks and sinusoidal time embeddings."""

    def test_sinusoidal_time_embedding(self):
        batch_size = 8
        embed_dim = 32
        ste = SinusoidalTimeEmbedding(embed_dim=embed_dim)
        timesteps = torch.arange(batch_size)

        emb = ste(timesteps)
        assert emb.shape == (batch_size, embed_dim)
        assert not torch.isnan(emb).any()

        # Different timesteps must have different embeddings
        cos_sim = F.cosine_similarity(emb[0:1], emb[1:2])
        assert cos_sim.item() < 0.999

    def test_denoising_mlp_unconditional(self):
        batch_size, input_dim = 4, 24
        model = DenoisingMLP(input_dim=input_dim, hidden_dim=64, time_embed_dim=32, num_blocks=2)
        x = torch.randn(batch_size, input_dim)
        t = torch.tensor([0, 10, 20, 30])

        out = model(x, t)
        assert out.shape == (batch_size, input_dim)

        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_denoising_mlp_class_conditional(self):
        batch_size, input_dim, num_classes = 6, 20, 3
        model = DenoisingMLP(input_dim=input_dim, hidden_dim=64, time_embed_dim=32, num_classes=num_classes)
        x = torch.randn(batch_size, input_dim)
        t = torch.randint(0, 50, (batch_size,))
        c = torch.tensor([0, 1, 2, 0, 1, 2])

        out = model(x, t, c=c)
        assert out.shape == (batch_size, input_dim)


class TestGaussianDiffusionProcess:
    """Verifies closed-form forward sampling and reverse score matching."""

    def test_q_sample_forward_noising(self):
        schedule = DiffusionSchedule(timesteps=50, schedule_type="cosine")
        diffusion = GaussianDiffusion(schedule)

        x_0 = torch.ones(10, 8)
        # At t=0, x_t should be very close to x_0 (minimal noise)
        t_0 = torch.zeros(10, dtype=torch.long)
        x_at_0 = diffusion.q_sample(x_0, t_0, noise=torch.zeros_like(x_0))
        assert torch.allclose(x_at_0, x_0, atol=1e-2)

        # At t = T-1, x_t should be dominated by noise
        t_last = torch.full((10,), 49, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_at_last = diffusion.q_sample(x_0, t_last, noise=noise)
        assert not torch.allclose(x_at_last, x_0)

    def test_training_loss_computation(self):
        schedule = DiffusionSchedule(timesteps=30, schedule_type="linear")
        diffusion = GaussianDiffusion(schedule)
        model = DenoisingMLP(input_dim=12, hidden_dim=32, time_embed_dim=16, num_blocks=1)

        x_0 = torch.randn(8, 12)
        loss = diffusion.compute_training_loss(model, x_0)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_predict_start_from_noise_inversion(self):
        schedule = DiffusionSchedule(timesteps=20, schedule_type="cosine")
        diffusion = GaussianDiffusion(schedule)

        x_0 = torch.randn(4, 10)
        t = torch.tensor([5, 10, 15, 18], dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = diffusion.q_sample(x_0, t, noise=noise)

        # Reconstructed x_0 with perfect ground-truth noise should equal original x_0
        recon_x0 = diffusion.predict_start_from_noise(x_t, t, noise)
        assert torch.allclose(recon_x0, x_0, atol=1e-4)


class TestSamplingEngines:
    """Verifies DDPM and DDIM generation algorithms."""

    def test_ddpm_sampling_shape(self):
        schedule = DiffusionSchedule(timesteps=15, schedule_type="cosine")
        diffusion = GaussianDiffusion(schedule)
        model = DenoisingMLP(input_dim=8, hidden_dim=32, time_embed_dim=16, num_blocks=1)

        samples = diffusion.sample_ddpm(model, shape=(4, 8), device="cpu")
        assert samples.shape == (4, 8)
        assert not torch.isnan(samples).any()

    def test_ddim_accelerated_sampling(self):
        schedule = DiffusionSchedule(timesteps=30, schedule_type="cosine")
        diffusion = GaussianDiffusion(schedule)
        model = DenoisingMLP(input_dim=8, hidden_dim=32, time_embed_dim=16, num_blocks=1)

        # Sample with only 5 accelerated steps instead of 30
        samples_ddim = diffusion.sample_ddim(
            model, shape=(4, 8), num_steps=5, eta=0.0, device="cpu"
        )
        assert samples_ddim.shape == (4, 8)
        assert not torch.isnan(samples_ddim).any()

    def test_mmd_computation(self):
        x = np.random.randn(50, 10)
        # MMD between same distribution sample should be zero
        mmd_zero = compute_multivariate_mmd(x, x)
        assert abs(mmd_zero) < 1e-5

        y = np.random.randn(50, 10) + 5.0
        mmd_diff = compute_multivariate_mmd(x, y)
        assert mmd_diff > 0.1
