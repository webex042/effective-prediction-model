"""
EPM Denoising Score Networks
============================

Denoising neural networks for diffusion models:
- SinusoidalTimeEmbedding (Fourier / Transformer-style positional encoding)
- DenoisingMLP: Residual score network with time and class conditioning
"""

from __future__ import annotations

import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Computes sinusoidal / Fourier embeddings for discrete timesteps t:
    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timesteps : torch.Tensor
            1D tensor of timesteps (batch_size,).

        Returns
        -------
        torch.Tensor
            Embedded timesteps of shape (batch_size, embed_dim).
        """
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.float()

        half_dim = self.embed_dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        ) / half_dim
        freqs = torch.exp(exponent)
        args = timesteps[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):
    """Residual MLP block with time projection injection and LayerNorm."""

    def __init__(self, dim: int, time_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim)
        )
        self.fc1 = nn.Linear(dim, dim * 2)
        self.norm1 = nn.LayerNorm(dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act(h)
        # Inject time conditioning
        t_proj = self.time_proj(t_emb)
        # Expand time projection to dim*2 for gating or add to dim
        h = self.fc2(h) + t_proj
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        return residual + h


class DenoisingMLP(nn.Module):
    """
    Denoising Score Network epsilon_theta(x_t, t, c).

    Parameters
    ----------
    input_dim : int
        Dimension of noisy telemetry vector (e.g. 120).
    hidden_dim : int
        Hidden dimension for internal residual layers (e.g. 128).
    time_embed_dim : int
        Dimension of sinusoidal time embeddings (e.g. 64).
    num_blocks : int
        Number of residual denoising blocks (e.g. 3).
    num_classes : Optional[int]
        Number of classes for conditional generation (e.g. 3 for Nominal, Warning, Fault).
    """

    def __init__(
        self,
        input_dim: int = 120,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_blocks: int = 3,
        num_classes: Optional[int] = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_classes = num_classes

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Optional class embedding
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None

        # Input projection: input_dim -> hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Residual denoising blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_embed_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Output projection: hidden_dim -> input_dim (predicts noise epsilon)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Noisy input batch (batch_size, input_dim).
        t : torch.Tensor
            Timesteps (batch_size,).
        c : Optional[torch.Tensor]
            Class labels for conditional diffusion (batch_size,).

        Returns
        -------
        torch.Tensor
            Predicted noise epsilon_hat (batch_size, input_dim).
        """
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        t_emb = self.time_embed(t)

        if self.class_embed is not None and c is not None:
            c_emb = self.class_embed(c)
            t_emb = t_emb + c_emb

        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)

        eps_pred = self.out_proj(h)

        if len(orig_shape) > 2:
            eps_pred = eps_pred.view(orig_shape)

        return eps_pred
