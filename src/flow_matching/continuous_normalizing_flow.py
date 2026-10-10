"""
Continuous Normalizing Flow (CNF) & Instantaneous Change of Variables
====================================================================
Implements:
    - Continuous Normalizing Flow generative model
    - Instantaneous change of variables theorem for exact log-likelihoods:
        d/dt log p_t(x_t) = -Tr(dv_theta / dx_t) = -div(v_theta)
    - Exact analytical Jacobian trace computation (for d <= 32)
    - Hutchinson stochastic trace estimator (Rademacher / Gaussian) for arbitrary d
    - Flow Matching training loss (OT-CFM and Gaussian-CFM)
"""

import math
from typing import Optional, Tuple, Callable, Dict, Any
import torch
import torch.nn as nn
import numpy as np

from .models import VelocityFieldNet, ConditionalVelocityFieldNet
from .probability_paths import ProbabilityPath, OptimalTransportCFMPath
from .ode_solvers import ODESolver, euler_step, rk4_step, midpoint_step


def compute_exact_divergence(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact analytical divergence (trace of Jacobian) using autograd:
        div(v) = sum_{i=1}^D (dv_i / dx_i)

    Args:
        v_fn: Velocity callable taking (x, t) -> v.
        x: Input state of shape (B, D).
        t: Time tensor of shape (B, 1) or (B,).

    Returns:
        Tensor of shape (B, 1) with exact divergence for each batch item.
    """
    x_input = x.clone().requires_grad_(True)
    v = v_fn(x_input, t)  # (B, D)
    dim = x.size(-1)
    batch_size = x.size(0)

    divergence = torch.zeros((batch_size, 1), device=x.device, dtype=x.dtype)

    for i in range(dim):
        v_i = v[:, i]  # (B,)
        grad_outputs = torch.ones_like(v_i)
        grad_i = torch.autograd.grad(
            outputs=v_i,
            inputs=x_input,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, D)
        divergence = divergence + grad_i[:, i : i + 1]

    return divergence


def compute_hutchinson_divergence(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_samples: int = 1,
    noise_dist: str = "rademacher",
) -> torch.Tensor:
    """
    Compute unbiased stochastic trace estimator (Hutchinson, 1989):
        Tr(J) = E_eps [ eps^T J eps ]

    Args:
        v_fn: Velocity callable taking (x, t) -> v.
        x: Input state of shape (B, D).
        t: Time tensor of shape (B, 1).
        num_samples: Number of stochastic noise projections.
        noise_dist: 'rademacher' (values in {-1, +1}) or 'gaussian' N(0, I).

    Returns:
        Tensor of shape (B, 1) with estimated divergence.
    """
    x_input = x.clone().requires_grad_(True)
    v = v_fn(x_input, t)  # (B, D)
    batch_size = x.size(0)
    dim = x.size(-1)

    div_estimates = []
    for _ in range(num_samples):
        if noise_dist == "rademacher":
            eps = torch.randint(0, 2, size=(batch_size, dim), device=x.device, dtype=x.dtype) * 2.0 - 1.0
        else:
            eps = torch.randn_like(x)

        # eps^T v
        v_dot_eps = (v * eps).sum()
        # grad is eps^T J
        grad = torch.autograd.grad(
            outputs=v_dot_eps,
            inputs=x_input,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, D)
        # eps^T J eps
        div_est = (grad * eps).sum(dim=-1, keepdim=True)  # (B, 1)
        div_estimates.append(div_est)

    return torch.stack(div_estimates, dim=0).mean(dim=0)


class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow (CNF) model parameterized by a velocity field net.

    Supports:
        - Exact ODE-based sample generation from prior p_0 = N(0, I)
        - Exact Negative Log-Likelihood (NLL) via instantaneous change of variables
        - Conditional Flow Matching training loss (OT-CFM / Gaussian-CFM)
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        data_dim: int,
        default_solver: str = "euler",
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.data_dim = data_dim
        self.default_solver = default_solver
        self.solver = ODESolver(method=default_solver)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate velocity field v_theta(x, t, c)."""
        if condition is not None and isinstance(self.velocity_net, ConditionalVelocityFieldNet):
            return self.velocity_net(x, t, condition)
        elif hasattr(self.velocity_net, "forward") and "condition" in self.velocity_net.forward.__code__.co_varnames:
            return self.velocity_net(x, t, condition=condition)
        return self.velocity_net(x, t)

    def compute_cfm_loss(
        self,
        x1: torch.Tensor,
        probability_path: ProbabilityPath,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Conditional Flow Matching (CFM) objective:
            L_CFM(theta) = E_{t, x0, x1} [ || v_theta(x_t, t, c) - u_t(x | x0, x1) ||^2 ]

        Args:
            x1: Batch of target training data points of shape (B, D).
            probability_path: OptimalTransportCFMPath or GaussianCFMPath.
            condition: Optional condition tensor.

        Returns:
            Scalar MSE loss.
        """
        batch_size = x1.size(0)
        device = x1.device

        # 1. Sample source noise x0 ~ N(0, I)
        x0 = probability_path.sample_prior(batch_size, self.data_dim, device)

        # 2. Mini-batch Optimal Transport pairing if supported
        if hasattr(probability_path, "pair_samples"):
            x0, x1_paired = probability_path.pair_samples(x0, x1)
        else:
            x1_paired = x1

        # 3. Sample continuous time t ~ U[eps, 1 - eps]
        t = probability_path.sample_time(batch_size, device)

        # 4. Interpolate along path x_t
        x_t = probability_path.interpolate(x0, x1_paired, t)

        # 5. Compute analytical conditional target velocity u_t
        u_t = probability_path.target_velocity(x0, x1_paired, t)

        # 6. Predict velocity field
        v_pred = self.forward(x_t, t, condition=condition)

        # 7. Regress MSE
        loss = torch.mean((v_pred - u_t) ** 2)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_steps: int = 20,
        solver_method: Optional[str] = None,
        condition: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Generate samples by integrating the ODE from t=0.0 (prior) to t=1.0 (data).

        Args:
            batch_size: Number of samples to generate.
            num_steps: Discretization step count.
            solver_method: 'euler', 'midpoint', or 'rk4' (defaults to self.default_solver).
            condition: Conditioning state if model is conditional.
            return_trajectory: Whether to return all intermediate steps.
            seed: Optional random seed for reproducible sampling.

        Returns:
            Tuple of:
                - x_samples: Generated samples of shape (B, D).
                - trajectory: Tensor of shape (num_steps + 1, B, D) if return_trajectory, else None.
                - total_nfe: Total function evaluations performed.
        """
        device = next(self.velocity_net.parameters()).device
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            x0 = torch.randn((batch_size, self.data_dim), device=device, generator=g)
        else:
            x0 = torch.randn((batch_size, self.data_dim), device=device)

        solver = ODESolver(method=solver_method or self.default_solver)

        def v_fn(x_curr, t_curr):
            return self.forward(x_curr, t_curr, condition=condition)

        x_final, trajectory, nfe = solver.integrate(
            v_fn=v_fn,
            x0=x0,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            return_trajectory=return_trajectory,
        )
        return x_final, trajectory, nfe

    def compute_log_prob(
        self,
        x1: torch.Tensor,
        num_steps: int = 20,
        divergence_method: str = "exact",
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate exact Negative Log-Likelihood via instantaneous change of variables:
            log p_1(x_1) = log p_0(x_0) - int_1^0 div(v_theta(x_t, t)) dt
                         = log p_0(x_0) + int_0^1 div(v_theta(x_t, t)) dt

        Args:
            x1: Batch of target samples of shape (B, D).
            num_steps: Discretization steps for numerical integration backward.
            divergence_method: 'exact' or 'hutchinson'.
            condition: Optional condition tensor.

        Returns:
            Tuple of:
                - log_p: Log-probability of shape (B, 1).
                - nll: Negative log-likelihood = -log_p of shape (B, 1).
        """
        device = x1.device
        batch_size = x1.size(0)
        dt = 1.0 / num_steps

        current_x = x1.clone()
        delta_log_p = torch.zeros((batch_size, 1), device=device, dtype=x1.dtype)

        # Backward integration from t = 1.0 down to t = 0.0
        for step in range(num_steps):
            # Time advances from 1.0 - step*dt down to 1.0 - (step+1)*dt
            t_val = 1.0 - step * dt
            t_tensor = torch.full((batch_size, 1), t_val, device=device, dtype=x1.dtype)

            def v_fn(x_in, t_in):
                return self.forward(x_in, t_in, condition=condition)

            # Compute divergence at current state
            if divergence_method == "exact":
                div = compute_exact_divergence(v_fn, current_x, t_tensor)
            else:
                div = compute_hutchinson_divergence(v_fn, current_x, t_tensor)

            # In backward integration: dx = -dt * v, d(log p) = -dt * (-div) = +dt * div
            # So log p_0 = log p_1 - int_1^0 div dt  =>  log p_1 = log p_0 - (-int_1^0 div dt)
            with torch.no_grad():
                v = v_fn(current_x, t_tensor)
                current_x = current_x - dt * v
                # Accumulated integral of div(v) from 0 to 1
                delta_log_p = delta_log_p + dt * div.detach()

        # Prior log-likelihood under standard Gaussian N(0, I)
        # log p_0(x_0) = -0.5 * D * log(2*pi) - 0.5 * ||x_0||^2
        dim = self.data_dim
        log_p0 = -0.5 * dim * math.log(2.0 * math.pi) - 0.5 * torch.sum(current_x ** 2, dim=-1, keepdim=True)

        log_p1 = log_p0 + delta_log_p
        nll = -log_p1
        return log_p1, nll
