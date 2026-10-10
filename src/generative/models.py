"""
EPM Generative Modeling Architectures in PyTorch
================================================

Implements production-grade generative models for continuous telemetry trajectories:
  1. StandardAutoencoder (Deterministic baseline with information bottleneck)
  2. VariationalAutoencoder (Probabilistic Gaussian latent space + ELBO)
  3. ConditionalVAE (Class-conditional trajectory synthesis)
  4. Anomaly Detection & Monte-Carlo Uncertainty Quantification
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Deterministic Autoencoder Baseline
# =============================================================================

class StandardAutoencoder(nn.Module):
    """
    Deterministic Autoencoder baseline with information bottleneck.

    Parameters
    ----------
    input_dim : int
        Flattened input dimension (e.g. window_size * channels = 24 * 5 = 120).
    hidden_dims : List[int]
        Encoder hidden layer dimensions.
    latent_dim : int
        Bottleneck code dimension (d_z).
    """

    def __init__(
        self,
        input_dim: int = 120,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        hidden_dims = hidden_dims or [64, 32]

        # Encoder: x -> z
        enc_layers = []
        in_d = input_dim
        for h in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        enc_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: z -> x_recon
        dec_layers = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        if x.dim() > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
        z = self.encoder(x_flat)
        x_recon = self.decoder(z)
        if len(orig_shape) > 2:
            x_recon = x_recon.view(orig_shape)
        return x_recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction MSE per sample as anomaly score."""
        x_recon, _ = self.forward(x)
        diff = (x - x_recon).view(x.size(0), -1)
        return torch.mean(diff ** 2, dim=-1)


# =============================================================================
# 2. Variational Autoencoder (VAE)
# =============================================================================

class VariationalAutoencoder(nn.Module):
    """
    Probabilistic Variational Autoencoder with Gaussian latent distribution.

    Optimizes the Evidence Lower Bound (ELBO):
        log p(x) >= E_{q_phi}[log p_theta(x | z)] - beta * D_{KL}(q_phi(z | x) || p(z))

    Parameters
    ----------
    input_dim : int
        Flattened input dimension.
    hidden_dims : List[int]
        Hidden dimensions for encoder and decoder.
    latent_dim : int
        Latent Gaussian space dimension.
    beta : float
        Weight on KL divergence (beta-VAE).
    """

    def __init__(
        self,
        input_dim: int = 120,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        beta: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        hidden_dims = hidden_dims or [64, 32]

        # Shared Encoder Trunk
        enc_trunk = []
        in_d = input_dim
        for h in hidden_dims:
            enc_trunk.extend([
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        self.encoder_trunk = nn.Sequential(*enc_trunk)

        # Latent heads: Mean and Log-Variance
        self.fc_mu = nn.Linear(in_d, latent_dim)
        self.fc_logvar = nn.Linear(in_d, latent_dim)

        # Decoder
        dec_layers = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.encoder_trunk(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # During deterministic evaluation use mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        if x.dim() > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x

        mu, log_var = self.encode(x_flat)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)

        if len(orig_shape) > 2:
            x_recon = x_recon.view(orig_shape)

        return x_recon, mu, log_var, z

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss, reconstruction loss, and analytical KL divergence.
        """
        x_flat = x.view(x.size(0), -1)
        recon_flat = x_recon.view(x_recon.size(0), -1)

        # Reconstruction MSE loss per batch
        recon_loss = F.mse_loss(recon_flat, x_flat, reduction="mean")

        # Analytical KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_div = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        kl_loss = torch.mean(kl_div)

        total_loss = recon_loss + self.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def sample_prior(self, num_samples: int = 10, device: str = "cpu") -> torch.Tensor:
        """Sample synthetic instances directly from prior p(z) ~ N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def sample_with_uncertainty(
        self, x: torch.Tensor, num_mc_samples: int = 25
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte-Carlo posterior sampling for predictive uncertainty estimation.

        Returns
        -------
        recon_mean : (B, ...) expected trajectory
        recon_std : (B, ...) epistemic standard deviation across stochastic latent draws
        """
        orig_shape = x.shape
        x_flat = x.view(x.size(0), -1) if x.dim() > 2 else x
        mu, log_var = self.encode(x_flat)
        std = torch.exp(0.5 * log_var)

        draws = []
        for _ in range(num_mc_samples):
            eps = torch.randn_like(std)
            z = mu + std * eps
            rec = self.decode(z)
            if len(orig_shape) > 2:
                rec = rec.view(orig_shape)
            draws.append(rec.unsqueeze(0))

        # Stack over MC samples: (num_mc, B, ...)
        all_draws = torch.cat(draws, dim=0)
        mean_recon = torch.mean(all_draws, dim=0)
        std_recon = torch.std(all_draws, dim=0)
        return mean_recon, std_recon

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction error for anomaly identification."""
        x_recon, _, _, _ = self.forward(x)
        diff = (x - x_recon).view(x.size(0), -1)
        return torch.mean(diff ** 2, dim=-1)


# =============================================================================
# 3. Conditional Variational Autoencoder (C-VAE)
# =============================================================================

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (C-VAE).

    Generates sensor telemetry trajectories conditioned on operational health states:
        q_phi(z | x, y) ~ N(mu(x, y), sigma^2(x, y))
        p_theta(x | z, y)

    Enables controlled generative forecasting of Nominal vs Critical degradation.
    """

    def __init__(
        self,
        input_dim: int = 120,
        num_classes: int = 3,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        beta: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta
        hidden_dims = hidden_dims or [64, 32]

        # Encoder accepts [x || one_hot(y)]
        enc_in_dim = input_dim + num_classes
        enc_trunk = []
        in_d = enc_in_dim
        for h in hidden_dims:
            enc_trunk.extend([
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        self.encoder_trunk = nn.Sequential(*enc_trunk)
        self.fc_mu = nn.Linear(in_d, latent_dim)
        self.fc_logvar = nn.Linear(in_d, latent_dim)

        # Decoder accepts [z || one_hot(y)]
        dec_in_dim = latent_dim + num_classes
        dec_layers = []
        in_d = dec_in_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def _condition_vector(self, y: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Convert integer labels to one-hot vectors."""
        if y.dim() == 1:
            return F.one_hot(y, num_classes=self.num_classes).float().to(device)
        return y.float().to(device)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        y_cond = self._condition_vector(y, x.size(0), x.device)
        x_cond = torch.cat([x, y_cond], dim=-1)
        h = self.encoder_trunk(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_cond = self._condition_vector(y, z.size(0), z.device)
        z_cond = torch.cat([z, y_cond], dim=-1)
        return self.decoder(z_cond)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_flat = x.view(x.size(0), -1) if x.dim() > 2 else x

        mu, log_var = self.encode(x_flat, y)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, y)

        if len(orig_shape) > 2:
            x_recon = x_recon.view(orig_shape)

        return x_recon, mu, log_var, z

    def sample_conditional(self, class_label: int, num_samples: int = 10, device: str = "cpu") -> torch.Tensor:
        """Sample new telemetry specifically conditioned on class_label."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        return self.decode(z, y)


# =============================================================================
# 4. Training Harness
# =============================================================================

def train_generative_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[nn.Module, List[float]]:
    """
    Train AE, VAE, or C-VAE with Adam optimizer.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        epoch_losses = []
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                y = batch[2].to(device) if len(batch) >= 3 else None
            else:
                x = batch.to(device)
                y = None

            optimizer.zero_grad()

            if isinstance(model, ConditionalVAE):
                if y is None:
                    raise ValueError("Target labels y required for ConditionalVAE.")
                x_recon, mu, log_var, z = model(x, y)
                loss_dict = {
                    "loss": F.mse_loss(x_recon.view(x.size(0), -1), x.view(x.size(0), -1))
                    - model.beta * 0.5 * torch.mean(torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
                }
                loss = loss_dict["loss"]
            elif isinstance(model, VariationalAutoencoder):
                x_recon, mu, log_var, z = model(x)
                loss_dict = model.compute_loss(x, x_recon, mu, log_var)
                loss = loss_dict["loss"]
            elif isinstance(model, StandardAutoencoder):
                x_recon, _ = model(x)
                loss = F.mse_loss(x_recon.view(x.size(0), -1), x.view(x.size(0), -1))
            else:
                raise TypeError(f"Unsupported model type: {type(model)}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        loss_history.append(float(np.mean(epoch_losses)))

    return model, loss_history
