"""
EPM Vanilla Recurrent Neural Network (RNN) from Scratch
=======================================================

A pure NumPy implementation of a Recurrent Neural Network with
Backpropagation Through Time (BPTT).

MATHEMATICAL FOUNDATION
-----------------------
1. Recurrence Relation:
   At each timestep t:
     h_t = tanh( x_t W_{xh} + h_{t-1} W_{hh} + b_h )
     y_t = h_t W_{hy} + b_y
   
   Where:
     - x_t is input of shape (N, D_in)
     - h_t is hidden state of shape (N, D_hidden)
     - W_{xh} has shape (D_in, D_hidden)
     - W_{hh} has shape (D_hidden, D_hidden)  [Recurrent weight matrix]
     - W_{hy} has shape (D_hidden, D_out)

2. Backpropagation Through Time (BPTT):
   To compute dL/dW_{hh}, error gradients must be propagated backwards
   through ALL historical timesteps:
     dL/dh_t = (dL/dy_t) @ W_{hy}^T + (dL/dh_{t+1}) @ W_{hh}^T ⊙ (1 - h_{t+1}²)
   
3. The Vanishing / Exploding Gradient Dilemma:
   When propagating backwards across T timesteps, the gradient involves:
     Π_{k=t}^T W_{hh}^T
   - If the largest eigenvalue λ_max(W_{hh}) < 1: Gradients vanish exponentially (0.9^50 ≈ 0.005)
   - If λ_max(W_{hh}) > 1: Gradients explode exponentially (1.2^50 ≈ 9100)
   This fundamental mathematical failure led directly to gated architectures:
   LSTM (Hochreiter & Schmidhuber, 1997) and GRU (Cho et al., 2014).
"""

from __future__ import annotations

from typing import Tuple, List, Optional
import numpy as np


class VanillaRNNScratch:
    """
    Vanilla Recurrent Neural Network implemented in pure NumPy.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    hidden_dim : int
        Number of hidden state units.
    output_dim : int
        Number of output targets.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier / Glorot initialization
        std_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        std_hh = np.sqrt(2.0 / (2 * hidden_dim))
        std_hy = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W_xh = np.random.randn(input_dim, hidden_dim) * std_xh
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * std_hh
        self.b_h = np.zeros(hidden_dim)

        self.W_hy = np.random.randn(hidden_dim, output_dim) * std_hy
        self.b_y = np.zeros(output_dim)

        # Gradient storage
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)

    def forward(
        self, X: np.ndarray, h_0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass over sequence of shape (batch_size, seq_len, input_dim).

        Returns:
          last_output : shape (batch_size, output_dim)
          hidden_states : list of hidden states at each timestep
        """
        batch_size, seq_len, _ = X.shape
        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_dim))

        h_states = [h_0]
        h_current = h_0

        for t in range(seq_len):
            x_t = X[:, t, :]
            # pre-activation
            z_t = x_t @ self.W_xh + h_current @ self.W_hh + self.b_h
            h_current = np.tanh(z_t)
            h_states.append(h_current)

        # Final prediction from the last hidden state
        last_h = h_states[-1]
        y_pred = last_h @ self.W_hy + self.b_y

        return y_pred, h_states
