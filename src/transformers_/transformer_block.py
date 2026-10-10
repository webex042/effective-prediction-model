"""
EPM Transformer Block & Text Classifier
=======================================

Implementation of Sinusoidal Positional Encoding, Transformer Encoder Layer,
and TextTransformerClassifier in PyTorch.

MATHEMATICAL FOUNDATION
-----------------------
1. Permutation Equivariance of Self-Attention:
   Self-Attention computes dot products pairwise regardless of word order:
     Attention(P X) = P Attention(X) for any permutation matrix P.
   Without position information, the model cannot distinguish between:
     "temperature rising before vibration" vs "vibration rising before temperature".

2. Sinusoidal Positional Encoding (Vaswani et al., 2017):
   PE(pos, 2i)     = sin( pos / 10000^(2i / d_model) )
   PE(pos, 2i + 1) = cos( pos / 10000^(2i / d_model) )
   
   Geometric Intuition:
     - Each frequency corresponds to a sinusoid.
     - Allows the model to easily learn relative positions because:
       PE(pos + k) can be represented as a linear function of PE(pos).

3. Transformer Encoder Block:
   x₁ = LayerNorm( x + Dropout( MultiHeadAttention(x, x, x) ) )
   x₂ = LayerNorm( x₁ + Dropout( MLP(x₁) ) )
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike

from src.transformers_.attention import MultiHeadAttention
from src.evaluation.metrics import accuracy


class PositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encodings added to token embeddings.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int, default=500
        Maximum sequence length supported.
    """

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a trainable parameter, but part of state dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x of shape (batch, seq_len, d_model)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    """
    Standard Transformer Encoder Block:
      Self-Attention -> Residual & LayerNorm -> Feedforward MLP -> Residual & LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward MLP
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sublayer 1: Multi-Head Self-Attention with residual connection
        attn_out, attn_weights = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sublayer 2: Feedforward MLP with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class TextTransformerClassifier(nn.Module):
    """
    Complete Lightweight Transformer for Sequence Classification.

    Architecture:
      Token IDs -> Embedding -> Positional Encoding -> Transformer Block ->
      Masked Average Pooling -> Linear Head -> Logits
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 3,
        d_model: int = 32,
        num_heads: int = 4,
        d_ff: int = 64,
        max_seq_len: int = 64,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.transformer_block = TransformerBlock(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, attention_weights).
        """
        mask = (token_ids != self.pad_idx)  # (batch, seq_len)
        x = self.embedding(token_ids)       # (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        # Pass through Transformer encoder block
        x, attn_weights = self.transformer_block(x, mask=mask)

        # Masked average pooling across tokens
        mask_expanded = mask.unsqueeze(-1).float()
        sum_x = torch.sum(x * mask_expanded, dim=1)
        lengths = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        pooled = sum_x / lengths  # (batch, d_model)

        logits = self.classifier(pooled)
        return logits, attn_weights

    def predict_proba(self, token_ids: ArrayLike) -> np.ndarray:
        self.eval()
        t_ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64))
        with torch.no_grad():
            logits, _ = self(t_ids)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, token_ids: ArrayLike) -> np.ndarray:
        probs = self.predict_proba(token_ids)
        return np.argmax(probs, axis=1)

    def score(self, token_ids: ArrayLike, y: ArrayLike) -> float:
        preds = self.predict(token_ids)
        return accuracy(y, preds)
