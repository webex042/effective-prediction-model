"""
EPM Gaussian Diffusion Engine
=============================

Core Gaussian Diffusion implementation for DDPM (Ho et al., 2020) and
DDIM (Song et al., 2020) accelerated sampling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion.schedules import DiffusionSchedule


def extract(tensor: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Extract values from 1D schedule tensor for given batch timesteps and broadcast."""
    batch_size = timesteps.shape[0]
    out = tensor.gather(0, timesteps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


class GaussianDiffusion:
    """
    Gaussian Diffusion framework containing forward diffusion, training loss,
    and DDPM / DDIM reverse sampling engines.

    Parameters
    ----------
    schedule : DiffusionSchedule
        Variance schedule instance (linear or cosine).
    """

    def __init__(self, schedule: DiffusionSchedule) -> None:
        self.schedule = schedule
        self.timesteps = schedule.timesteps

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward process: sample from q(x_t | x_0) in closed form.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(
            self.schedule.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes simplified MSE denoising score matching loss:
        L_simple = E_{t, x_0, eps} [ || eps - eps_theta(x_t, t, c) ||^2 ]
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t, c)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate x_0 given x_t and predicted noise epsilon."""
        sqrt_recip_alphas_cumprod_t = extract(
            self.schedule.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = extract(
            self.schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.schedule.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.schedule.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.schedule.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(
            self.schedule.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_ddpm(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_{t-1} from p_theta(x_{t-1} | x_t) via standard DDPM step."""
        eps_pred = model(x, t, c)
        x_recon = self.predict_start_from_noise(x, t, eps_pred)
        model_mean, _, posterior_log_variance = self.q_posterior_mean_variance(
            x_start=x_recon, x_t=x, t=t
        )

        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        c: Optional[torch.Tensor] = None,
        device: str = "cpu",
        return_trajectory: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sample full trajectory from pure noise using DDPM (all T steps).
        """
        model.eval()
        x = torch.randn(shape, device=device)
        trajectory = [x.clone()] if return_trajectory else []

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample_ddpm(model, x, t, c=c)
            if return_trajectory and (i % max(1, self.timesteps // 10) == 0 or i == 0):
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        c: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        eta: float = 0.0,
        device: str = "cpu",
        return_trajectory: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Accelerated generation via Denoising Diffusion Implicit Models (DDIM).
        Subsamples discrete timesteps (e.g., 20 steps instead of 100).
        eta=0.0 corresponds to completely deterministic ODE reverse trajectory.
        """
        model.eval()
        batch_size = shape[0]

        # Uniformly spaced subsequence of timesteps
        step_indices = torch.linspace(0, self.timesteps - 1, num_steps, dtype=torch.long, device=device)
        step_indices = step_indices.tolist()

        x = torch.randn(shape, device=device)
        trajectory = [x.clone()] if return_trajectory else []

        for i in reversed(range(len(step_indices))):
            cur_t = step_indices[i]
            prev_t = step_indices[i - 1] if i > 0 else -1

            t_tensor = torch.full((batch_size,), cur_t, device=device, dtype=torch.long)
            eps_pred = model(x, t_tensor, c)

            alpha_cumprod_cur = self.schedule.alphas_cumprod[cur_t]
            alpha_cumprod_prev = (
                self.schedule.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            )

            # Predict x_0
            x_0_pred = (x - torch.sqrt(1.0 - alpha_cumprod_cur) * eps_pred) / torch.sqrt(alpha_cumprod_cur)

            # DDIM sigma variance parameter
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_cur))
                * torch.sqrt(1.0 - alpha_cumprod_cur / alpha_cumprod_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_prev - sigma**2, min=0.0)) * eps_pred

            # Next sample
            noise = torch.randn_like(x) if sigma > 0 else 0.0
            x = torch.sqrt(alpha_cumprod_prev) * x_0_pred + dir_xt + sigma * noise

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x
