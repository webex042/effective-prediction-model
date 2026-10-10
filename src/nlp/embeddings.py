"""
EPM Word Embeddings & Neural Text Classifier
============================================

Dense learned token representations and Bag-of-Embeddings classification architectures
implemented in NumPy and PyTorch.

MATHEMATICAL FOUNDATION
-----------------------
1. Limitations of Discrete One-Hot Encodings:
   - High Dimensionality: Orthogonal vectors of size |V| (often 10,000+).
   - Zero Semantic Metric: Any two distinct words have dot product 0 (cosine similarity = 0):
       similarity("vibration", "shaking") = 0
       similarity("vibration", "banana") = 0
   - No concept of proximity or geometric manifolds.

2. Dense Embeddings:
   Embeddings map discrete token index i into a continuous low-dimensional vector space:
     e_i = E[i, :] ∈ ℝ^D_embed (typically D = 16 to 128 for small models)
   Words with similar contexts converge geometrically (cosine similarity > 0):
     cos(e_vibration, e_shaking) ≈ 1.0

3. Bag of Embeddings (Average Pooling):
   For a sentence of tokens (t₁, t₂, ..., t_L):
     h_doc = (1/L) * Σ_{i=1}^L E[tᵢ, :]
   Sentence embedding h_doc is fed to a classification head:
     logits = W h_doc + b
"""

from __future__ import annotations

from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from numpy.typing import ArrayLike

from src.neural_networks.layers import Layer, Linear, ReLU, SoftmaxCrossEntropyLoss
from src.evaluation.metrics import accuracy


class EmbeddingLayerScratch(Layer):
    """
    Embedding lookup table implemented from scratch in NumPy.

    Parameters
    ----------
    vocab_size : int
        Size of vocabulary |V|.
    embedding_dim : int
        Dimension of dense vector representations.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Standard normal initialization scaled by sqrt(1 / embedding_dim)
        std = np.sqrt(1.0 / embedding_dim)
        self.W = np.random.randn(vocab_size, embedding_dim) * std
        self.dW = np.zeros_like(self.W)

        self.last_input: Optional[np.ndarray] = None

    def forward(self, token_ids: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward lookup:
          token_ids: shape (N, L)
          output: shape (N, L, D_embed)
        """
        token_ids_arr = np.asarray(token_ids, dtype=int)
        if training:
            self.last_input = token_ids_arr
        return self.W[token_ids_arr]

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Accumulate gradients into vocabulary rows corresponding to each token.
          grad_output: shape (N, L, D_embed)
        """
        assert self.last_input is not None
        self.dW.fill(0.0)

        # Accumulate gradients into indexed rows (np.add.at handles duplicate tokens cleanly)
        np.add.at(self.dW, self.last_input, grad_output)

    def get_params(self) -> list[np.ndarray]:
        return [self.W]

    def get_grads(self) -> list[np.ndarray]:
        return [self.dW]


class PyTorchEmbeddingClassifier(nn.Module):
    """
    PyTorch Bag-of-Embeddings Neural Text Classifier.

    Architecture:
      Token IDs -> nn.Embedding -> Mean Pooling -> Linear -> ReLU -> Linear
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 32,
        num_classes: int = 3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.pad_idx = pad_idx

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : torch.Tensor of shape (batch_size, seq_len)
        """
        # Embeddings: (batch, seq_len, embed_dim)
        embeds = self.embedding(token_ids)

        # Mask out padding tokens from mean pooling
        mask = (token_ids != self.pad_idx).unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_embeds = torch.sum(embeds * mask, dim=1)
        lengths = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = sum_embeds / lengths  # (batch, embed_dim)

        h = self.relu(self.fc1(pooled))
        logits = self.fc2(h)
        return logits

    def predict_proba(self, token_ids: ArrayLike) -> np.ndarray:
        self.eval()
        t_ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64))
        with torch.no_grad():
            logits = self(t_ids)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, token_ids: ArrayLike) -> np.ndarray:
        probs = self.predict_proba(token_ids)
        return np.argmax(probs, axis=1)

    def score(self, token_ids: ArrayLike, y: ArrayLike) -> float:
        preds = self.predict(token_ids)
        return accuracy(y, preds)
