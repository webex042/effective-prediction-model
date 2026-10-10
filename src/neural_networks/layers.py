"""
EPM Neural Network Layers & Losses from Scratch
===============================================

Pure NumPy modular layers implementing forward and backward passes (analytical gradients)
with full mathematical derivations.

MATHEMATICAL FOUNDATION
-----------------------
1. Linear (Affine / Dense) Layer:
   Forward:
     Z = X W + b
     Where:
       - X has shape (N, D_in)  [N = batch size, D_in = input features]
       - W has shape (D_in, D_out)
       - b has shape (D_out,)
       - Z has shape (N, D_out)
   
   Backward (given upstream gradient dL/dZ of shape (N, D_out)):
     dL/dX = (dL/dZ) @ W^T           shape (N, D_in)   [propagates back to previous layer]
     dL/dW = X^T @ (dL/dZ)           shape (D_in, D_out) [gradient for weight update]
     dL/db = Σ_{i=1}^N (dL/dZ)_i     shape (D_out,)     [gradient for bias update]

2. Non-Linear Activations:
   Why are activations essential?
     Without non-linear activations f(Z), stacking linear layers collapses into a single matrix:
       Z₂ = (X W₁ + b₁) W₂ + b₂ = X (W₁ W₂) + (b₁ W₂ + b₂) = X W_equiv + b_equiv
     Thus, a 100-layer linear network has the exact same representational power as a 1-layer network!
     Non-linear activations break this linearity, giving networks the Universal Approximation property.
   
   ReLU (Rectified Linear Unit):
     Forward:  a = max(0, z)
     Backward: da/dz = 1 if z > 0 else 0
     Vectorized gradient: dL/dZ = dL/dA ⊙ 𝟙(Z > 0)
     Advantage: Does not saturate for large positive inputs (avoids vanishing gradients).
   
   Sigmoid:
     Forward:  a = 1 / (1 + e^(-z))
     Backward: da/dz = a * (1 - a)
     Gradient: dL/dZ = dL/dA ⊙ a ⊙ (1 - a)

3. Softmax + Cross-Entropy (Fused for Numerical Stability):
   Logits Z of shape (N, C):
     p_{i, c} = exp(z_{i, c} - max_k z_{i, k}) / Σ_j exp(z_{i, j} - max_k z_{i, k})
   Loss:
     L = - (1/N) * Σ_{i=1}^N ln(p_{i, y_i})
   Gradient with respect to logits Z:
     dL/dZ = (1/N) * (P - Y_onehot)
   This elegant result avoids computing the full Jacobian of Softmax directly!
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class Layer(ABC):
    """Abstract base class for all neural network layers."""

    @abstractmethod
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute layer output given input activations X."""
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute upstream gradient dL/dX given downstream dL/dOutput."""
        pass

    def get_params(self) -> list[np.ndarray]:
        """Return list of trainable parameters (e.g. weights, biases)."""
        return []

    def get_grads(self) -> list[np.ndarray]:
        """Return list of parameter gradients corresponding to get_params()."""
        return []


class Linear(Layer):
    """
    Fully-connected (Dense / Affine) Layer: Z = X W + b.

    Parameters
    ----------
    in_features : int
        Number of input dimensions (D_in).
    out_features : int
        Number of output dimensions (D_out).
    weight_init : {"kaiming", "xavier", "normal"}, default="kaiming"
        Weight initialization strategy.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: str = "kaiming",
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        if weight_init == "kaiming":
            # He / Kaiming Normal: optimal for ReLU activations
            # std = sqrt(2 / in_features)
            std = np.sqrt(2.0 / in_features)
            self.W = np.random.randn(in_features, out_features) * std
        elif weight_init == "xavier":
            # Xavier / Glorot: optimal for Sigmoid/Tanh
            # std = sqrt(2 / (in_features + out_features))
            std = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * std
        else:
            self.W = np.random.randn(in_features, out_features) * 0.01

        # Initialize biases to zeros
        self.b = np.zeros(out_features, dtype=np.float64)

        # Gradient storage
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backprop
        self.X_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: Z = X W + b."""
        if training:
            self.X_cache = X
        return X @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass:
          dL/dX = grad_output @ W^T
          dL/dW = X^T @ grad_output
          dL/db = sum(grad_output, axis=0)
        """
        assert self.X_cache is not None, "forward() must be called before backward()"

        # Compute parameter gradients
        self.dW = self.X_cache.T @ grad_output
        self.db = np.sum(grad_output, axis=0)

        # Propagate gradient to preceding layer
        grad_input = grad_output @ self.W.T
        return grad_input

    def get_params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def get_grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU(Layer):
    """
    Rectified Linear Unit: A = max(0, Z).
    Derivative: dA/dZ = 1 if Z > 0 else 0.
    """

    def __init__(self) -> None:
        self.Z_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.Z_cache = X
        return np.maximum(0.0, X)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.Z_cache is not None
        # Subgradient at 0 is taken as 0
        return grad_output * (self.Z_cache > 0).astype(np.float64)


class Sigmoid(Layer):
    """
    Logistic Sigmoid: A = 1 / (1 + exp(-Z)).
    Derivative: dA/dZ = A * (1 - A).
    """

    def __init__(self) -> None:
        self.A_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        X_clipped = np.clip(X, -500.0, 500.0)
        A = np.where(
            X_clipped >= 0,
            1.0 / (1.0 + np.exp(-X_clipped)),
            np.exp(X_clipped) / (1.0 + np.exp(X_clipped)),
        )
        if training:
            self.A_cache = A
        return A

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.A_cache is not None
        return grad_output * self.A_cache * (1.0 - self.A_cache)


class Tanh(Layer):
    """
    Hyperbolic Tangent: A = tanh(Z).
    Derivative: dA/dZ = 1 - A^2.
    """

    def __init__(self) -> None:
        self.A_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        A = np.tanh(X)
        if training:
            self.A_cache = A
        return A

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.A_cache is not None
        return grad_output * (1.0 - self.A_cache ** 2)


# =============================================================================
# Loss Functions
# =============================================================================

class SoftmaxCrossEntropyLoss:
    """
    Fused Softmax and Categorical Cross-Entropy Loss.
    Combines them analytically for numerical stability and zero vanishing gradients.

    Forward:
      p_{i, c} = exp(z_{i, c} - max(z_i)) / Σ_j exp(z_{i, j} - max(z_i))
      loss = - (1/N) * Σ ln(p_{i, y_i})

    Backward:
      grad = (1/N) * (P - Y_onehot)
    """

    def __init__(self) -> None:
        self.probs_cache: Optional[np.ndarray] = None
        self.y_true_cache: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Parameters
        ----------
        logits : np.ndarray of shape (N, C)
            Unnormalized network output.
        y_true : np.ndarray of shape (N,)
            Integer ground truth class indices {0, ..., C-1}.

        Returns
        -------
        loss : float
        """
        N = logits.shape[0]
        # Shift logits by max for numerical stability (prevents exp overflow)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        self.probs_cache = probs
        self.y_true_cache = y_true.astype(int)

        # Negative log-likelihood of target class
        # Add epsilon to prevent log(0)
        eps = 1e-15
        correct_class_probs = probs[np.arange(N), self.y_true_cache]
        loss = - np.mean(np.log(correct_class_probs + eps))
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Returns dL/dLogits of shape (N, C):
          (1/N) * (P - Y_onehot)
        """
        assert self.probs_cache is not None and self.y_true_cache is not None
        N, C = self.probs_cache.shape
        grad = self.probs_cache.copy()

        # Subtract 1 at ground truth index: P - Y_onehot
        grad[np.arange(N), self.y_true_cache] -= 1.0

        # Average over batch
        grad /= N
        return grad


class MSELoss:
    """
    Mean Squared Error Loss for regression:
      L = (1 / (2N)) * ||Y_pred - Y_true||²
      dL/dY_pred = (1/N) * (Y_pred - Y_true)
    """

    def __init__(self) -> None:
        self.diff_cache: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
        y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
        diff = y_pred - y_true
        self.diff_cache = diff
        return float(0.5 * np.mean(diff ** 2))

    def backward(self) -> np.ndarray:
        assert self.diff_cache is not None
        N = self.diff_cache.shape[0]
        return self.diff_cache / N
