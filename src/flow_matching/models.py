"""
Flow Matching Velocity Field Models
====================================
Neural network architectures parameterizing continuous-time velocity fields:
    v_theta(x, t) in R^d
and class-conditioned velocity fields:
    v_theta(x, t, c) in R^d

Includes sinusoidal time embeddings and residual MLP vector field regressors.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTimeEmbedding(nn.Module):
    """
    Fourier / Sinusoidal embedding for continuous time scalar t in [0, 1].

    Transforms scalar t -> vector of shape (B, embed_dim) using sinusoidal
    basis frequencies with learnable linear projection.
    """

    def __init__(self, embed_dim: int = 64, max_period: float = 10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (B,) or (B, 1) with continuous time values in [0, 1].

        Returns:
            Embedding tensor of shape (B, embed_dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        elif t.dim() > 2:
            t = t.view(t.size(0), -1)

        half_dim = self.embed_dim // 2
        # Frequencies
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        args = t * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return self.proj(emb)


class VelocityResidualBlock(nn.Module):
    """Residual block with time conditioning and LayerNorm."""

    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.act(self.norm1(x))
        h = self.fc1(h)
        h = h + self.time_proj(t_emb)
        h = self.act(self.norm2(h))
        h = self.fc2(h)
        return residual + h


class VelocityFieldNet(nn.Module):
    """
    Velocity Field Parameterization v_theta(x, t): R^d x [0, 1] -> R^d.

    Parameterizes continuous vector field for Continuous Normalizing Flows
    and Conditional Flow Matching.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.time_embed = ContinuousTimeEmbedding(embed_dim=time_embed_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            VelocityResidualBlock(hidden_dim=hidden_dim, time_dim=time_embed_dim)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity vector v_theta(x, t).

        Args:
            x: Input state tensor of shape (B, input_dim).
            t: Continuous time tensor of shape (B,) or (B, 1) in [0, 1].

        Returns:
            Velocity tensor of shape (B, input_dim).
        """
        t_emb = self.time_embed(t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.act(self.out_norm(h))
        v = self.out_proj(h)
        return v


class ConditionalVelocityFieldNet(nn.Module):
    """
    Conditional Velocity Field v_theta(x, t, c): R^d x [0, 1] x R^c -> R^d.

    Supports class labels (discrete operating state) or continuous context vectors
    (e.g., historical sensor summary).
    """

    def __init__(
        self,
        input_dim: int = 5,
        num_classes: Optional[int] = None,
        context_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.time_embed = ContinuousTimeEmbedding(embed_dim=time_embed_dim)

        # Condition handling
        self.num_classes = num_classes
        self.context_dim = context_dim
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        elif context_dim is not None:
            self.context_proj = nn.Linear(context_dim, time_embed_dim)
        else:
            self.class_embed = None
            self.context_proj = None

        self.in_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            VelocityResidualBlock(hidden_dim=hidden_dim, time_dim=time_embed_dim)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute conditional velocity v_theta(x, t, c).

        Args:
            x: Input state tensor of shape (B, input_dim).
            t: Time scalar tensor of shape (B,) or (B, 1) in [0, 1].
            condition: Class labels (B,) or context vectors (B, context_dim).

        Returns:
            Velocity tensor of shape (B, input_dim).
        """
        t_emb = self.time_embed(t)

        if condition is not None:
            if self.class_embed is not None:
                c_emb = self.class_embed(condition.long())
                t_emb = t_emb + c_emb
            elif self.context_proj is not None:
                c_emb = self.context_proj(condition.float())
                t_emb = t_emb + c_emb

        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.act(self.out_norm(h))
        v = self.out_proj(h)
        return v
