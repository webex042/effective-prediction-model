"""
EPM Deep Learning Models for Time-Series Forecasting
====================================================

PyTorch implementations of LSTM, GRU, and Time-Series Transformer models
supporting single-step and multi-step forecasting horizons.

MATHEMATICAL FOUNDATION
-----------------------
1. Long Short-Term Memory (LSTM):
   Introduces an additive Cell State c_t gated by non-linear sigmoid gates:
     - Forget Gate : f_t = σ( W_f [h_{t-1}, x_t] + b_f )
     - Input Gate  : i_t = σ( W_i [h_{t-1}, x_t] + b_i )
     - Candidate   : č_t = tanh( W_c [h_{t-1}, x_t] + b_c )
     - Cell State  : c_t = f_t ⊙ c_{t-1} + i_t ⊙ č_t
     - Output Gate : o_t = σ( W_o [h_{t-1}, x_t] + b_o )
     - Hidden State: h_t = o_t ⊙ tanh(c_t)
   
   Why it stops vanishing gradients:
     The path from c_{t-1} to c_t is an ADDITIVE linear highway: ∂c_t/∂c_{t-1} = f_t.
     If f_t = 1, gradients flow unhindered backwards through 100+ timesteps without decay!

2. Gated Recurrent Unit (GRU):
   Merges cell and hidden state into a streamlined 2-gate architecture:
     - Reset Gate  : r_t = σ( W_r [h_{t-1}, x_t] + b_r )
     - Update Gate : z_t = σ( W_z [h_{t-1}, x_t] + b_z )
     - Candidate   : h̃_t = tanh( W [r_t ⊙ h_{t-1}, x_t] + b )
     - Hidden State: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
   Fewer parameters (~25% less than LSTM) with comparable predictive power.

3. Time-Series Transformer:
   Replaces recurrence with direct attention across all historical timesteps:
     Input (B, W, D) -> Linear Projection (d_model) + Positional Encoding ->
     Multi-Head Self-Attention -> Temporal Pooling -> Linear Head (horizon H).
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.transformers_.transformer_block import PositionalEncoding, TransformerBlock


class LSTMForecaster(nn.Module):
    """
    LSTM sequence-to-vector forecaster for time series.

    Parameters
    ----------
    input_dim : int
        Number of sensor channels per timestep.
    hidden_dim : int, default=32
        Hidden state size.
    horizon : int, default=1
        Number of future steps to forecast.
    num_layers : int, default=2
        Number of stacked LSTM layers.
    dropout : float, default=0.1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        horizon: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, seq_len, input_dim)
        Output: (batch, horizon)
        """
        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Use final timestep hidden state for forecasting
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


class GRUForecaster(nn.Module):
    """
    GRU sequence-to-vector forecaster for time series.

    Parameters
    ----------
    input_dim : int
        Number of sensor channels per timestep.
    hidden_dim : int, default=32
    horizon : int, default=1
    num_layers : int, default=2
    dropout : float, default=0.1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        horizon: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


class TimeSeriesTransformer(nn.Module):
    """
    Time-Series Transformer Encoder for temporal forecasting.

    Parameters
    ----------
    input_dim : int
        Number of sensor channels per timestep.
    d_model : int, default=32
        Model embedding dimension.
    num_heads : int, default=4
        Number of self-attention heads.
    horizon : int, default=1
    d_ff : int, default=64
    num_layers : int, default=2
    dropout : float, default=0.1
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        num_heads: int = 4,
        horizon: int = 1,
        d_ff: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.horizon = horizon

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=500)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project sensor channels to d_model and add sinusoidal positions
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        for layer in self.layers:
            h, _ = layer(h)

        # Average pool across time dimension
        pooled = torch.mean(h, dim=1)
        out = self.head(pooled)
        return out


def train_forecasting_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.005,
) -> Dict[str, list[float]]:
    """Train a PyTorch forecasting model with MSE loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        history["train_loss"].append(avg_loss)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                v_loss = criterion(val_pred, y_val_t).item()
                history["val_loss"].append(v_loss)

    return history
