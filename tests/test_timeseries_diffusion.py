"""
Unit Tests for Stage 9 — Diffusion for Time Series Forecasting
==============================================================
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.diffusion.schedules import DiffusionSchedule
from src.diffusion.timeseries_diffusion import (
    HistoricalContextEncoder,
    TrajectoryDenoisingNet,
    TimeSeriesDiffusionForecaster,
)
from src.diffusion.timeseries_benchmarks import (
    compute_crps,
    compute_interval_coverage,
)


class TestHistoricalContextEncoder:
    """Verifies past telemetry context encoding."""

    def test_context_encoder_shapes(self):
        batch_size, lookback, in_channels = 8, 24, 5
        context_dim = 32
        encoder = HistoricalContextEncoder(
            in_channels=in_channels,
            lookback_window=lookback,
            hidden_dim=32,
            context_dim=context_dim,
        )
        x_past = torch.randn(batch_size, lookback, in_channels)
        c = encoder(x_past)
        assert c.shape == (batch_size, context_dim)
        assert not torch.isnan(c).any()

    def test_context_encoder_gradients(self):
        encoder = HistoricalContextEncoder(in_channels=4, lookback_window=16, hidden_dim=24, context_dim=24)
        x = torch.randn(4, 16, 4)
        out = encoder(x)
        out.sum().backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestTrajectoryDenoisingNet:
    """Verifies horizon denoiser."""

    def test_trajectory_denoiser_shapes(self):
        batch_size, horizon, context_dim = 6, 12, 32
        denoiser = TrajectoryDenoisingNet(
            horizon=horizon,
            context_dim=context_dim,
            hidden_dim=48,
            time_embed_dim=16,
            num_blocks=2,
        )
        y_noisy = torch.randn(batch_size, horizon)
        t = torch.randint(0, 50, (batch_size,))
        context = torch.randn(batch_size, context_dim)

        eps_pred = denoiser(y_noisy, t, context)
        assert eps_pred.shape == (batch_size, horizon)
        assert not torch.isnan(eps_pred).any()


class TestTimeSeriesDiffusionForecaster:
    """Verifies end-to-end conditional trajectory forecasting model."""

    def test_loss_computation(self):
        schedule = DiffusionSchedule(timesteps=30, schedule_type="cosine")
        model = TimeSeriesDiffusionForecaster(
            schedule=schedule,
            in_channels=3,
            lookback_window=16,
            horizon=8,
            hidden_dim=32,
            context_dim=24,
        )
        x_past = torch.randn(4, 16, 3)
        y_future = torch.randn(4, 8)

        loss = model.compute_loss(x_past, y_future)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_sample_trajectories_and_quantiles(self):
        schedule = DiffusionSchedule(timesteps=20, schedule_type="cosine")
        model = TimeSeriesDiffusionForecaster(
            schedule=schedule,
            in_channels=4,
            lookback_window=12,
            horizon=6,
            hidden_dim=32,
            context_dim=24,
        )
        x_past = torch.randn(3, 12, 4)

        # Generate 5 stochastic trajectory samples
        samples = model.sample_trajectories(x_past, n_samples=5, num_steps=5, eta=0.0)
        assert samples.shape == (3, 5, 6)

        quantiles = model.predict_quantiles(
            x_past, quantiles=(0.1, 0.5, 0.9), n_samples=10, num_steps=5, eta=0.0
        )
        assert "median" in quantiles
        assert "q10" in quantiles
        assert "q90" in quantiles
        assert quantiles["median"].shape == (3, 6)


class TestProbabilisticMetrics:
    """Verifies CRPS and prediction interval metrics."""

    def test_crps_perfect_prediction(self):
        y_true = np.array([[1.0, 2.0, 3.0]])
        # K identical samples matching ground truth
        samples = np.array([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
        crps = compute_crps(y_true, samples)
        assert abs(crps) < 1e-5

    def test_interval_coverage(self):
        y_true = np.array([[2.0, 3.0]])
        q_lower = np.array([[1.0, 2.5]])
        q_upper = np.array([[3.0, 3.5]])
        coverage, width = compute_interval_coverage(y_true, q_lower, q_upper)
        assert coverage == 1.0
        assert width > 0.0
