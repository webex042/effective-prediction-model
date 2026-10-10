"""
Probability Paths for Flow Matching
===================================
Constructs continuous-time probability paths p_t(x) connecting base distributions
p_0 = N(0, I) to target data distributions p_1 = q(x), alongside analytical
conditional target velocity fields u_t(x | x_0, x_1).

Implements:
    - Optimal Transport Conditional Flow Matching (OT-CFM)
    - Standard Gaussian Conditional Flow Matching (Gaussian-CFM)
    - Mini-Batch Optimal Transport pairing via the Hungarian algorithm (scipy)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_minibatch_ot(
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2-Wasserstein mini-batch optimal transport pairing between x0 and x1.

    Pairs samples in x0 with samples in x1 that minimize the sum of squared
    Euclidean distances:
        min_{pi} sum_i ||x0[i] - x1[pi(i)]||^2

    Args:
        x0: Source noise batch of shape (B, D).
        x1: Target data batch of shape (B, D).

    Returns:
        Tuple of (x0, x1_matched) where x1_matched is reordered by optimal permutation.
    """
    if x0.size(0) != x1.size(0):
        raise ValueError(f"Batch sizes must match: x0={x0.size(0)}, x1={x1.size(0)}")

    batch_size = x0.size(0)
    # Compute pairwise squared Euclidean distance matrix C in (B, B)
    # ||x0 - x1||^2 = ||x0||^2 + ||x1||^2 - 2 <x0, x1>
    with torch.no_grad():
        dist_matrix = torch.cdist(x0, x1, p=2.0).pow(2).cpu().numpy()

    # Solve linear sum assignment problem via Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    col_ind_tensor = torch.as_tensor(col_ind, dtype=torch.long, device=x1.device)
    x1_matched = x1[col_ind_tensor]
    return x0, x1_matched


class ProbabilityPath(ABC):
    """Abstract base class for probability paths in Flow Matching."""

    @abstractmethod
    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state x_t at time t given boundary states x0 and x1."""
        pass

    @abstractmethod
    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional target velocity u_t(x | x0, x1) = dx_t / dt."""
        pass

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Sample continuous time t uniformly from [eps, 1 - eps]."""
        return torch.rand((batch_size, 1), device=device) * (1.0 - 2 * eps) + eps

    def sample_prior(
        self,
        batch_size: int,
        dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample source states x0 ~ N(0, I)."""
        return torch.randn((batch_size, dim), device=device)


class OptimalTransportCFMPath(ProbabilityPath):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) Path.

    Linear interpolation path:
        x_t = (1 - t) * x_0 + t * x_1

    Target velocity field:
        u_t(x | x_0, x_1) = d/dt x_t = x_1 - x_0

    When used with mini-batch OT coupling, the trajectories are straight lines
    that minimize kinetic transport energy:
        min int_0^1 ||v_t||^2 dt
    """

    def __init__(self, use_ot: bool = True):
        super().__init__()
        self.use_ot = use_ot

    def pair_samples(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pair x0 and x1 using mini-batch optimal transport if enabled."""
        if self.use_ot:
            return compute_minibatch_ot(x0, x1)
        return x0, x1

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_t = (1 - t) * x_0 + t * x_1
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * x1

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        u_t(x | x_0, x_1) = x_1 - x_0 (constant velocity along straight ray)
        """
        return x1 - x0


class GaussianCFMPath(ProbabilityPath):
    """
    Standard Gaussian Conditional Flow Matching Path (Lipman et al. 2023).

    Path:
        x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        u_t(x | x_0, x_1) = x_1 - (1 - sigma_min) * x_0

    Guarantees standard Gaussian distribution N(0, I) at t=0, and
    concentrated Gaussian N(x_1, sigma_min^2 I) at t=1.
    """

    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        sigma_t = 1.0 - (1.0 - self.sigma_min) * t
        mu_t = t * x1
        return sigma_t * x0 + mu_t

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return x1 - (1.0 - self.sigma_min) * x0
