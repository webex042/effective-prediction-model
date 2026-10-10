"""
EPM Time-Series Diffusion Forecaster
====================================

Conditional Generative Diffusion for Probabilistic Trajectory Forecasting:
- HistoricalContextEncoder: Encodes past multivariate telemetry window (L x C)
- TrajectoryDenoisingNet: Denoiser predicting noise on future trajectory horizon (H)
- TimeSeriesDiffusionForecaster: Forward SDE & reverse conditional sampling (DDPM/DDIM)
- Quantile estimation, CRPS, and Prediction Interval Coverage (PICP)
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion.schedules import DiffusionSchedule
from src.diffusion.models import SinusoidalTimeEmbedding


class HistoricalContextEncoder(nn.Module):
    """
    Encodes past multivariate telemetry window X in R^{B x L x C}
    into a dense historical context representation c in R^{B x d_c}.

    Parameters
    ----------
    in_channels : int
        Number of sensor channels (e.g. 5: Temp, Vib, Press, RPM, Load).
    lookback_window : int
        Number of past timesteps (e.g. 24).
    hidden_dim : int
        Hidden representation dimension (e.g. 64).
    context_dim : int
        Final context output dimension (e.g. 64).
    """

    def __init__(
        self,
        in_channels: int = 5,
        lookback_window: int = 24,
        hidden_dim: int = 64,
        context_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.lookback_window = lookback_window
        self.context_dim = context_dim

        # Temporal convolution feature extractor
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        # GRU for temporal sequential integration
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        # Projection head to context_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_past : torch.Tensor of shape (B, L, C)

        Returns
        -------
        torch.Tensor of shape (B, context_dim)
        """
        # (B, L, C) -> (B, C, L) for Conv1d
        x_conv = x_past.transpose(1, 2)
        h = F.gelu(self.norm1(self.conv1(x_conv)))
        h = F.gelu(self.norm2(self.conv2(h)))

        # Back to (B, L, hidden_dim) for GRU
        h = h.transpose(1, 2)
        _, h_last = self.gru(h)  # (1, B, hidden_dim)
        context = self.proj(h_last.squeeze(0))
        return context


class TrajectoryDenoisingNet(nn.Module):
    """
    Residual Denoising MLP that predicts noise on future trajectory horizon
    y_t in R^{B x H} conditioned on timestep t and historical context c.

    Parameters
    ----------
    horizon : int
        Forecast horizon length (e.g. 12 timesteps).
    context_dim : int
        Historical context dimension from encoder (e.g. 64).
    hidden_dim : int
        Denoising network hidden dimension (e.g. 128).
    time_embed_dim : int
        Sinusoidal time embedding dimension (e.g. 32).
    num_blocks : int
        Number of residual blocks (e.g. 3).
    """

    def __init__(
        self,
        horizon: int = 12,
        context_dim: int = 64,
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # Sinusoidal time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Context projection: context_dim -> hidden_dim
        self.context_proj = nn.Linear(context_dim, hidden_dim)

        # Input trajectory projection: horizon -> hidden_dim
        self.in_proj = nn.Linear(horizon, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_blocks)
        ])

        # Final projection: hidden_dim -> horizon (predicts noise epsilon)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_noisy : torch.Tensor of shape (B, H)
        t : torch.Tensor of shape (B,)
        context : torch.Tensor of shape (B, context_dim)

        Returns
        -------
        torch.Tensor of shape (B, H) (predicted noise)
        """
        t_emb = self.time_embed(t)
        c_emb = self.context_proj(context)

        # Fuse noisy trajectory with time and past context
        h = self.in_proj(y_noisy) + t_emb + c_emb

        for block in self.blocks:
            h = h + block(h)

        eps_pred = self.out_proj(h)
        return eps_pred


class TimeSeriesDiffusionForecaster(nn.Module):
    """
    Complete Conditional Diffusion Forecaster for continuous time-series trajectories.

    Parameters
    ----------
    schedule : DiffusionSchedule
        Diffusion schedule (e.g. cosine or linear).
    in_channels : int
        Number of historical sensor channels (default: 5).
    lookback_window : int
        Number of historical lookback timesteps L (default: 24).
    horizon : int
        Forecast horizon timesteps H (default: 12).
    hidden_dim : int
        Internal hidden dimension (default: 128).
    """

    def __init__(
        self,
        schedule: DiffusionSchedule,
        in_channels: int = 5,
        lookback_window: int = 24,
        horizon: int = 12,
        hidden_dim: int = 128,
        context_dim: int = 64,
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self.timesteps = schedule.timesteps
        self.in_channels = in_channels
        self.lookback_window = lookback_window
        self.horizon = horizon

        self.context_encoder = HistoricalContextEncoder(
            in_channels=in_channels,
            lookback_window=lookback_window,
            hidden_dim=hidden_dim // 2,
            context_dim=context_dim,
        )

        self.denoiser = TrajectoryDenoisingNet(
            horizon=horizon,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=32,
            num_blocks=3,
        )

    def compute_loss(
        self,
        x_past: torch.Tensor,
        y_future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes conditional trajectory denoising MSE loss.
        """
        batch_size = x_past.shape[0]
        device = x_past.device

        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        if noise is None:
            noise = torch.randn_like(y_future)

        # Encode historical context
        context = self.context_encoder(x_past)

        # Forward noising q(y_t | y_0)
        sqrt_alphas_t = self.schedule.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_t = self.schedule.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        y_noisy = sqrt_alphas_t * y_future + sqrt_one_minus_t * noise

        # Denoising prediction
        eps_pred = self.denoiser(y_noisy, t, context)
        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def sample_trajectories(
        self,
        x_past: torch.Tensor,
        n_samples: int = 10,
        num_steps: int = 20,
        eta: float = 0.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generates n_samples future stochastic trajectories for each sample in batch.

        Parameters
        ----------
        x_past : torch.Tensor of shape (B, L, C)
        n_samples : int, default=10
            Number of stochastic Monte-Carlo paths per sequence.
        num_steps : int, default=20
            Accelerated DDIM steps.
        eta : float, default=0.0
            Stochasticity parameter (0 = deterministic ODE).

        Returns
        -------
        torch.Tensor of shape (B, n_samples, horizon)
        """
        self.eval()
        x_past = x_past.to(device)
        B = x_past.shape[0]

        # Encode context once and repeat for n_samples
        context = self.context_encoder(x_past)  # (B, context_dim)
        context_rep = context.repeat_interleave(n_samples, dim=0)  # (B * n_samples, context_dim)

        total_batch = B * n_samples
        step_indices = torch.linspace(0, self.timesteps - 1, num_steps, dtype=torch.long, device=device).tolist()

        y = torch.randn((total_batch, self.horizon), device=device)

        for i in reversed(range(len(step_indices))):
            cur_t = step_indices[i]
            prev_t = step_indices[i - 1] if i > 0 else -1

            t_tensor = torch.full((total_batch,), cur_t, device=device, dtype=torch.long)
            eps_pred = self.denoiser(y, t_tensor, context_rep)

            alpha_cumprod_cur = self.schedule.alphas_cumprod[cur_t]
            alpha_cumprod_prev = (
                self.schedule.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            )

            # Predict y_0 with standard dynamic thresholding / clipping
            alpha_clamped = torch.clamp(alpha_cumprod_cur, min=1e-4)
            y_0_pred = (y - torch.sqrt(1.0 - alpha_cumprod_cur) * eps_pred) / torch.sqrt(alpha_clamped)
            y_0_pred = torch.clamp(y_0_pred, min=-4.0, max=4.0)

            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_cur))
                * torch.sqrt(torch.clamp(1.0 - alpha_cumprod_cur / alpha_cumprod_prev, min=0.0))
            )

            dir_yt = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_prev - sigma**2, min=0.0)) * eps_pred
            noise = torch.randn_like(y) if sigma > 0 else 0.0
            y = torch.sqrt(alpha_cumprod_prev) * y_0_pred + dir_yt + sigma * noise

        # Reshape to (B, n_samples, horizon)
        return y.view(B, n_samples, self.horizon)

    @torch.no_grad()
    def predict_quantiles(
        self,
        x_past: torch.Tensor,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        n_samples: int = 20,
        num_steps: int = 20,
        eta: float = 0.5,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Computes probabilistic quantile predictions across stochastic diffusion runs.

        Returns
        -------
        Dict with keys like 'median', 'q10', 'q90', 'all_samples'
        """
        samples = self.sample_trajectories(
            x_past, n_samples=n_samples, num_steps=num_steps, eta=eta, device=device
        )  # (B, n_samples, H)

        results = {}
        for q in quantiles:
            q_name = f"q{int(q*100):02d}" if q != 0.5 else "median"
            results[q_name] = torch.quantile(samples, q, dim=1)

        results["mean"] = torch.mean(samples, dim=1)
        results["std"] = torch.std(samples, dim=1)
        results["all_samples"] = samples
        return results
