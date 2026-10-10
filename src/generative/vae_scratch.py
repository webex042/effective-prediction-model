"""
EPM Variational Autoencoder (VAE) from Scratch in NumPy
======================================================

Mathematical implementation of a Variational Autoencoder from first principles:
  - Probabilistic Latent Space: q_phi(z | x) ~ N(mu(x), diag(sigma^2(x)))
  - Prior: p(z) ~ N(0, I)
  - Reparameterization Trick: z = mu + sigma * epsilon, with epsilon ~ N(0, I)
  - Objective: Evidence Lower Bound (ELBO)
      ELBO = E_{q_phi}[log p_theta(x | z)] - beta * D_{KL}(q_phi(z | x) || p(z))
  - Analytical Backpropagation through the stochastic computational graph
  - Two-sided numerical finite-difference gradient checking verification
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np


class VAEScratch:
    """
    Pure NumPy Variational Autoencoder with analytical backpropagation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features (D).
    hidden_dim : int
        Dimensionality of hidden encoder/decoder layers.
    latent_dim : int
        Dimensionality of bottleneck latent space (d_z).
    beta : float, default=1.0
        Weighting factor on KL divergence (beta-VAE formulation).
    random_seed : int, default=42
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 4,
        beta: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.rng = np.random.default_rng(random_seed)

        # Initialize parameters with Xavier / He scaling
        # Encoder: x -> h_enc
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # Latent heads: h_enc -> mu, h_enc -> log_var
        self.W_mu = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, latent_dim))
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, latent_dim))
        self.b_logvar = np.zeros(latent_dim)

        # Decoder: z -> h_dec -> x_recon
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / latent_dim), (latent_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)

        self.W3 = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, input_dim))
        self.b3 = np.zeros(input_dim)

        # Cache for backpropagation
        self.cache: Dict[str, np.ndarray] = {}

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map input x to posterior parameters (mu, log_var).
        """
        h_enc = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        mu = h_enc @ self.W_mu + self.b_mu
        log_var = h_enc @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(
        self, mu: np.ndarray, log_var: np.ndarray, eps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply reparameterization trick: z = mu + sigma * epsilon.
        """
        std = np.exp(0.5 * log_var)
        if eps is None:
            eps = self.rng.normal(0, 1, size=mu.shape)
        z = mu + std * eps
        return z, eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from latent code z.
        """
        h_dec = np.maximum(0, z @ self.W2 + self.b2)  # ReLU
        x_recon = h_dec @ self.W3 + self.b3
        return x_recon

    def forward(
        self, x: np.ndarray, eps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full forward pass caching intermediate activations.

        Returns
        -------
        x_recon : (B, D)
        mu : (B, d_z)
        log_var : (B, d_z)
        z : (B, d_z)
        """
        # Encoder forward
        a1 = x @ self.W1 + self.b1
        h_enc = np.maximum(0, a1)
        mu = h_enc @ self.W_mu + self.b_mu
        log_var = h_enc @ self.W_logvar + self.b_logvar

        # Reparameterization
        z, eps_used = self.reparameterize(mu, log_var, eps=eps)

        # Decoder forward
        a2 = z @ self.W2 + self.b2
        h_dec = np.maximum(0, a2)
        x_recon = h_dec @ self.W3 + self.b3

        self.cache = {
            "x": x,
            "a1": a1,
            "h_enc": h_enc,
            "mu": mu,
            "log_var": log_var,
            "eps": eps_used,
            "z": z,
            "a2": a2,
            "h_dec": h_dec,
            "x_recon": x_recon,
        }
        return x_recon, mu, log_var, z

    def compute_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute negative ELBO loss: Total Loss = Recon Loss (MSE) + beta * KL Divergence.

        Returns
        -------
        total_loss : float
        recon_loss : float
        kl_loss : float
        """
        batch_size = x.shape[0]

        # 1. Reconstruction Loss: Mean Squared Error
        recon_loss = np.mean(np.sum((x_recon - x) ** 2, axis=1))

        # 2. Analytical KL Divergence: D_KL(N(mu, sigma^2) || N(0, I))
        # D_KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_per_sample = -0.5 * np.sum(1.0 + log_var - mu ** 2 - np.exp(log_var), axis=1)
        kl_loss = np.mean(kl_per_sample)

        total_loss = recon_loss + self.beta * kl_loss
        return float(total_loss), float(recon_loss), float(kl_loss)

    def backward(self) -> Dict[str, np.ndarray]:
        """
        Analytical backpropagation computing exact gradients for all parameters.

        Returns
        -------
        grads : Dict of parameter gradients
        """
        x = self.cache["x"]
        x_recon = self.cache["x_recon"]
        h_dec = self.cache["h_dec"]
        a2 = self.cache["a2"]
        z = self.cache["z"]
        mu = self.cache["mu"]
        log_var = self.cache["log_var"]
        eps = self.cache["eps"]
        h_enc = self.cache["h_enc"]
        a1 = self.cache["a1"]
        batch_size = x.shape[0]

        # 1. Gradient of Reconstruction Loss w.r.t x_recon
        # dRecon/dx_recon = 2 * (x_recon - x) / batch_size
        dx_recon = (2.0 / batch_size) * (x_recon - x)

        # Decoder gradients
        dW3 = h_dec.T @ dx_recon
        db3 = np.sum(dx_recon, axis=0)

        dh_dec = dx_recon @ self.W3.T
        da2 = dh_dec * (a2 > 0)  # ReLU grad

        dW2 = z.T @ da2
        db2 = np.sum(da2, axis=0)

        # Gradient at latent z from reconstruction
        dz = da2 @ self.W2.T

        # 2. Backprop through Reparameterization Trick: z = mu + std * eps
        std = np.exp(0.5 * log_var)
        dmu_recon = dz
        dlogvar_recon = dz * eps * (0.5 * std)

        # 3. Gradients of KL divergence term
        # D_KL = -0.5 * (1 + log_var - mu^2 - exp(log_var))
        # dKL/dmu = mu / batch_size
        # dKL/dlogvar = 0.5 * (exp(log_var) - 1) / batch_size
        dmu_kl = (mu / batch_size) * self.beta
        dlogvar_kl = (0.5 * (np.exp(log_var) - 1.0) / batch_size) * self.beta

        # Total gradients for latent heads
        dmu = dmu_recon + dmu_kl
        dlogvar = dlogvar_recon + dlogvar_kl

        dW_mu = h_enc.T @ dmu
        db_mu = np.sum(dmu, axis=0)

        dW_logvar = h_enc.T @ dlogvar
        db_logvar = np.sum(dlogvar, axis=0)

        # Backprop into encoder hidden activations
        dh_enc = dmu @ self.W_mu.T + dlogvar @ self.W_logvar.T
        da1 = dh_enc * (a1 > 0)  # ReLU grad

        dW1 = x.T @ da1
        db1 = np.sum(da1, axis=0)

        return {
            "W1": dW1,
            "b1": db1,
            "W_mu": dW_mu,
            "b_mu": db_mu,
            "W_logvar": dW_logvar,
            "b_logvar": db_logvar,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    def sample_prior(self, n_samples: int = 10) -> np.ndarray:
        """
        Generate novel synthetic samples by decoding random vectors from prior p(z) ~ N(0, I).
        """
        z_prior = self.rng.normal(0, 1, size=(n_samples, self.latent_dim))
        return self.decode(z_prior)


def check_vae_numerical_gradients(
    input_dim: int = 6,
    hidden_dim: int = 8,
    latent_dim: int = 3,
    h: float = 1e-5,
) -> float:
    """
    Two-sided numerical gradient checking for VAE loss.
    Returns maximum relative error between analytical and numerical gradients.
    """
    vae = VAEScratch(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        beta=1.0,
        random_seed=42,
    )
    rng = np.random.default_rng(101)
    x = rng.normal(0, 1, size=(4, input_dim))
    eps = rng.normal(0, 1, size=(4, latent_dim))

    # Analytical gradients
    x_recon, mu, log_var, z = vae.forward(x, eps=eps)
    _ = vae.compute_loss(x, x_recon, mu, log_var)
    analytical_grads = vae.backward()

    max_rel_error = 0.0

    # Verify each parameter
    param_names = ["W1", "b1", "W_mu", "b_mu", "W_logvar", "b_logvar", "W2", "b2", "W3", "b3"]

    for name in param_names:
        param = getattr(vae, name)
        grad_ana = analytical_grads[name]

        # Check random sample of coordinates to be efficient
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]

            # f(x + h)
            param[idx] = orig_val + h
            xr_pos, mu_pos, lv_pos, _ = vae.forward(x, eps=eps)
            loss_pos, _, _ = vae.compute_loss(x, xr_pos, mu_pos, lv_pos)

            # f(x - h)
            param[idx] = orig_val - h
            xr_neg, mu_neg, lv_neg, _ = vae.forward(x, eps=eps)
            loss_neg, _, _ = vae.compute_loss(x, xr_neg, mu_neg, lv_neg)

            # Restore
            param[idx] = orig_val

            grad_num = (loss_pos - loss_neg) / (2.0 * h)
            rel_error = abs(grad_ana[idx] - grad_num) / (max(abs(grad_ana[idx]), abs(grad_num)) + 1e-8)
            if rel_error > max_rel_error:
                max_rel_error = float(rel_error)

            it.iternext()

    return max_rel_error
