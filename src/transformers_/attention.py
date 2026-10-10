"""
EPM Scaled Dot-Product & Multi-Head Self-Attention from Scratch
==============================================================

Pure NumPy and PyTorch implementations of the core Attention Mechanism
introduced in "Attention Is All You Need" (Vaswani et al., 2017).

MATHEMATICAL FOUNDATION
-----------------------
1. Intuition (Database Query Analogy):
   In traditional retrieval:
     - Query Q searches for exact matching Key K and retrieves associated Value V.
   In Soft Attention:
     - Query Q is compared against ALL Keys K simultaneously via dot products.
     - Normalized similarities form a probability distribution (Attention Weights A).
     - The output is the expected Value: a weighted average of all Values.

2. Scaled Dot-Product Attention:
   Attention(Q, K, V) = softmax( (Q K^T) / √d_k ) V
   
   Dimensions:
     - Q has shape (B, L_q, d_k)   [B: batch, L: sequence length, d_k: key dimension]
     - K has shape (B, L_k, d_k)
     - V has shape (B, L_v, d_v)   (usually L_k = L_v = L)
     - Q K^T has shape (B, L_q, L_k)  [Pairwise similarity scores between tokens]
     - Attention Weights A = softmax(Scores, axis=-1)  [A[b, i, j] is how much token i attends to token j]
     - Output has shape (B, L_q, d_v)

3. Why the √d_k Scaling Factor?
   Assume components of q and k are independent random variables with zero mean and variance 1:
     E[q · k] = 0
     Var(q · k) = Σ_{i=1}^{d_k} Var(q_i k_i) = d_k
   As d_k grows large (e.g. 64, 512), dot products grow proportionally to √d_k.
   Large inputs to Softmax push gradients into saturation regions (derivative ≈ 0),
   causing severe vanishing gradients during training.
   Dividing by √d_k forces the variance back to 1.0, preserving healthy gradients:
     Var( (q · k) / √d_k ) = (1 / d_k) * d_k = 1.0

4. Multi-Head Attention:
   Instead of performing a single attention function with d_model-dimensional Q, K, V,
   multi-head attention projects Q, K, V linearly h times with different learned matrices:
     head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
     MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_h) W^O
   This allows the model to jointly attend to information from different representation
   subspaces (e.g., one head tracks syntax, another tracks subject-verb relations,
   another tracks vibration-temperature physical correlations).
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def scaled_dot_product_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Scaled Dot-Product Attention using pure NumPy.

    Parameters
    ----------
    Q : np.ndarray of shape (B, L_q, d_k) or (L_q, d_k)
    K : np.ndarray of shape (B, L_k, d_k) or (L_k, d_k)
    V : np.ndarray of shape (B, L_v, d_v) or (L_v, d_v)
    mask : np.ndarray, optional
        Boolean mask where True/1 indicates valid token and False/0 indicates token to ignore.

    Returns
    -------
    output : np.ndarray of shape (B, L_q, d_v)
    attention_weights : np.ndarray of shape (B, L_q, L_k)
    """
    has_batch = Q.ndim == 3
    if not has_batch:
        Q = Q[np.newaxis, ...]
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]

    d_k = Q.shape[-1]

    # 1. Similarity Scores: Q @ K^T / sqrt(d_k)
    # Shape: (B, L_q, L_k)
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    # 2. Apply Mask if provided (e.g. padding mask)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, np.newaxis, :]  # broadcast over queries
        # Where mask is False, fill with large negative value
        scores = np.where(mask, scores, -1e9)

    # 3. Softmax across key dimension (last axis)
    # Numerically stable softmax
    shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Context Output: Weights @ Values
    output = attention_weights @ V

    if not has_batch:
        output = output[0]
        attention_weights = attention_weights[0]

    return output, attention_weights


def scaled_dot_product_attention_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention in PyTorch.
    Returns (output, attention_weights).
    """
    d_k = Q.size(-1)
    # Scores: (..., L_q, L_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-2)
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module implemented in PyTorch.

    Parameters
    ----------
    d_model : int
        Total model embedding dimension.
    num_heads : int
        Number of attention heads (d_model must be divisible by num_heads).
    dropout : float, default=0.0
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        Q, K, V : torch.Tensor of shape (batch, seq_len, d_model)
        mask : torch.Tensor, optional

        Returns
        -------
        output : torch.Tensor of shape (batch, seq_len, d_model)
        attention_weights : torch.Tensor of shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)

        # 1. Linear projections and reshape into (batch, num_heads, seq_len, d_k)
        q = self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention across all heads in parallel
        # Scores: (batch, num_heads, L_q, L_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                # (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Context: (batch, num_heads, L_q, d_k)
        context = torch.matmul(attn_weights_drop, v)

        # 3. Concatenate heads: (batch, L_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final linear projection
        output = self.w_o(context)

        return output, attn_weights
