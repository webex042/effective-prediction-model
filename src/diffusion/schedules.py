"""
EPM Diffusion Schedules
=======================

Mathematical formulations of forward diffusion variance schedules:
- Linear schedule (Ho et al., 2020: DDPM)
- Cosine schedule (Nichol & Dhariwal, 2021: Improved DDPM)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple
import torch
import numpy as np


class DiffusionSchedule:
    """
    Computes and caches precomputed constants for Gaussian diffusion forward and reverse processes.

    Parameters
    ----------
    timesteps : int
        Total discrete diffusion steps T (default: 100).
    schedule_type : str
        Schedule type: 'linear' or 'cosine' (default: 'cosine').
    beta_start : float
        Initial beta for linear schedule (default: 1e-4).
    beta_end : float
        Final beta for linear schedule (default: 0.02).
    """

    def __init__(
        self,
        timesteps: int = 100,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif schedule_type == "cosine":
            # Cosine schedule from Nichol & Dhariwal (2021)
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=1e-5, max=0.999)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}. Choose 'linear' or 'cosine'.")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clamped log variance because posterior_variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device | str) -> DiffusionSchedule:
        """Move all schedule tensors to specified device."""
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod", "sqrt_recip_alphas",
            "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
            "posterior_variance", "posterior_log_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2"
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def get_snr(self) -> torch.Tensor:
        """Signal-to-Noise Ratio at each timestep: alpha_bar / (1 - alpha_bar)."""
        return self.alphas_cumprod / (1.0 - self.alphas_cumprod)
