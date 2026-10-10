"""
Efficient Attention Mechanisms & Computational Complexity
==========================================================
First-principles implementation of exact and efficient attention algorithms:
    1. Standard Full Quadratic Attention: O(N^2 d) time, O(N^2) space
    2. Kernel Linear Attention (Katharopoulos et al.): O(N d^2) time, O(N d) space
       via associative property: (phi(Q) phi(K)^T) V = phi(Q) (phi(K)^T V)
    3. Local Sliding-Window Attention: O(N w d) time, O(N w) space
    4. Tiled Memory-Efficient Exact Attention: O(N^2 d) time, O(N d) space
       via FlashAttention-style Online Softmax running statistics
    5. MultiHeadEfficientAttention modular PyTorch wrapper
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def phi_elu(x: torch.Tensor) -> torch.Tensor:
    """Non-negative kernel feature map for Linear Attention: phi(x) = elu(x) + 1."""
    return F.elu(x) + 1.0


def standard_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard Scaled Dot-Product Attention (Vaswani et al., 2017).

    Args:
        Q: Queries of shape (B, H, N, d).
        K: Keys of shape (B, H, N, d).
        V: Values of shape (B, H, N, d).
        mask: Optional mask of shape (N, N) or (B, 1, N, N).

    Returns:
        Tuple of (output, attention_matrix) where output is (B, H, N, d) and
        attention_matrix is (B, H, N, N) requiring O(N^2) memory.
    """
    d_k = Q.size(-1)
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
    out = torch.matmul(attn_weights, V)  # (B, H, N, d)
    return out, attn_weights


def linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Linear Attention via Kernel Feature Map (Katharopoulos et al., 2020).

    Computes:
        phi(Q) * (phi(K)^T * V) / (phi(Q) * sum(phi(K)))
    By changing multiplication order from (Q K^T) V to Q (K^T V), computational
    complexity drops from O(N^2 d) to O(N d^2) in time and O(N d) in space.

    Args:
        Q: Queries of shape (B, H, N, d).
        K: Keys of shape (B, H, N, d).
        V: Values of shape (B, H, N, d).
        causal: Whether to apply causal autoregressive masking.

    Returns:
        Output tensor of shape (B, H, N, d).
    """
    # 1. Non-negative kernel mapping
    Q_phi = phi_elu(Q)  # (B, H, N, d)
    K_phi = phi_elu(K)  # (B, H, N, d)

    if not causal:
        # Bidirectional Linear Attention:
        # KV summary matrix: (B, H, d, d)
        # phi(K)^T @ V -> sum over sequence N
        KV = torch.matmul(K_phi.transpose(-2, -1), V)  # (B, H, d, d)

        # Normalization denominator: phi(Q) @ (sum_j phi(K)_j)
        K_sum = K_phi.sum(dim=-2, keepdim=True)  # (B, H, 1, d)
        normalizer = torch.matmul(Q_phi, K_sum.transpose(-2, -1))  # (B, H, N, 1)

        # Numerator: phi(Q) @ KV -> (B, H, N, d)
        num = torch.matmul(Q_phi, KV)  # (B, H, N, d)
        out = num / (normalizer + eps)
    else:
        # Causal Linear Attention via cumulative sum (O(N d^2) time):
        # Outer product: K_phi_i * V_i^T for each position i
        # Shape: (B, H, N, d, 1) x (B, H, N, 1, d) -> (B, H, N, d, d)
        outer = torch.einsum("b h n i, b h n j -> b h n i j", K_phi, V)
        S = torch.cumsum(outer, dim=2)  # (B, H, N, d, d)

        # Cumulative normalizer
        Z = torch.cumsum(K_phi, dim=2)  # (B, H, N, d)
        normalizer = torch.einsum("b h n d, b h n d -> b h n", Q_phi, Z).unsqueeze(-1)  # (B, H, N, 1)

        # Numerator
        num = torch.einsum("b h n i, b h n i j -> b h n j", Q_phi, S)  # (B, H, N, d)
        out = num / (normalizer + eps)

    return out


def local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int = 32,
) -> torch.Tensor:
    """
    Local Sliding-Window Attention (Beltagy et al., Longformer 2020).

    Each query at position i only attends to keys in [i - window_size//2, i + window_size//2].
    Complexity: O(N * w * d) time and O(N * w) space.

    Args:
        Q: Queries of shape (B, H, N, d).
        K: Keys of shape (B, H, N, d).
        V: Values of shape (B, H, N, d).
        window_size: Window bandwidth w.

    Returns:
        Output tensor of shape (B, H, N, d).
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    half_w = window_size // 2

    # Unfold K and V into sliding local blocks
    # Pad sequence on both sides
    K_padded = F.pad(K, (0, 0, half_w, half_w), mode="constant", value=0.0)
    V_padded = F.pad(V, (0, 0, half_w, half_w), mode="constant", value=0.0)

    # Unfold K and V: (B, H, N, window_size + 1, d)
    # Using torch.as_strided or unfold
    # For PyTorch tensor unfold along sequence dim (dim=2):
    K_unfolded = K_padded.unfold(dimension=2, size=window_size + 1, step=1)  # (B, H, N, d, w+1)
    K_unfolded = K_unfolded.transpose(-2, -1)  # (B, H, N, w+1, d)

    V_unfolded = V_padded.unfold(dimension=2, size=window_size + 1, step=1)  # (B, H, N, d, w+1)
    V_unfolded = V_unfolded.transpose(-2, -1)  # (B, H, N, w+1, d)

    # Pairwise local scores: Q_i dot K_{i, j} for j in local window
    # Q: (B, H, N, 1, d), K_unfolded: (B, H, N, w+1, d)
    scores = torch.sum(Q.unsqueeze(-2) * K_unfolded, dim=-1) * scale  # (B, H, N, w+1)

    # Softmax over local window
    attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, H, N, w+1, 1)
    out = torch.sum(attn_weights * V_unfolded, dim=-2)  # (B, H, N, d)
    return out


def tiled_exact_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Tiled Memory-Efficient Exact Attention (Dao et al., FlashAttention / Milakov & Gimelshteyn).

    Computes mathematically EXACT scaled dot-product attention without ever allocating
    or materializing the N x N attention matrix in memory. Uses the Online Softmax algorithm
    with running max m_i and running sum l_i.

    Peak memory: O(N d + block_size^2) instead of O(N^2).

    Args:
        Q: Queries of shape (B, H, N, d).
        K: Keys of shape (B, H, N, d).
        V: Values of shape (B, H, N, d).
        block_size: Discretization block size for cache tiling.

    Returns:
        Exact attention output of shape (B, H, N, d).
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    device = Q.device
    dtype = Q.dtype

    out = torch.zeros_like(Q)
    # Running max and normalizer for online softmax: (B, H, N, 1)
    m = torch.full((B, H, N, 1), float("-inf"), device=device, dtype=dtype)
    l = torch.zeros((B, H, N, 1), device=device, dtype=dtype)

    # Outer loop over key/value blocks (columns)
    num_blocks = math.ceil(N / block_size)

    for block_idx in range(num_blocks):
        start_kv = block_idx * block_size
        end_kv = min(start_kv + block_size, N)

        K_block = K[:, :, start_kv:end_kv, :]  # (B, H, B_kv, d)
        V_block = V[:, :, start_kv:end_kv, :]  # (B, H, B_kv, d)

        # Compute block scores between all Q and current K_block: (B, H, N, B_kv)
        scores_block = torch.matmul(Q, K_block.transpose(-2, -1)) * scale

        # Online softmax update:
        # 1. Block max
        m_block = scores_block.max(dim=-1, keepdim=True).values  # (B, H, N, 1)
        m_new = torch.maximum(m, m_block)  # (B, H, N, 1)

        # 2. Rescale previous sum and output by exp(m_old - m_new)
        alpha = torch.exp(m - m_new)
        # Handle -inf in initial step
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Exponentiate block scores with new max
        P_block = torch.exp(scores_block - m_new)  # (B, H, N, B_kv)
        l_block = P_block.sum(dim=-1, keepdim=True)  # (B, H, N, 1)

        # 4. Update running normalizer
        l_new = l * alpha + l_block  # (B, H, N, 1)

        # 5. Update running output: out = out * alpha + P_block @ V_block
        out = out * alpha + torch.matmul(P_block, V_block)  # (B, H, N, d)

        m = m_new
        l = l_new

    # Final normalization
    out = out / (l + 1e-8)
    return out


class MultiHeadEfficientAttention(nn.Module):
    """
    Modular Multi-Head Attention supporting selectable efficiency paradigms:
        - 'standard': Classical O(N^2) Softmax Attention
        - 'linear': O(N) Kernel Linear Attention (Katharopoulos et al.)
        - 'local': O(N) Sliding Window Attention (Longformer)
        - 'tiled': Exact Online Softmax Attention with O(N) peak memory
    """

    SUPPORTED_TYPES = ("standard", "linear", "local", "tiled")

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        attn_type: str = "standard",
        window_size: int = 32,
        block_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if attn_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unknown attn_type '{attn_type}'. Choose from {self.SUPPORTED_TYPES}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_type = attn_type
        self.window_size = window_size
        self.block_size = block_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward multi-head attention pass.

        Args:
            x: Input token sequence of shape (B, N, embed_dim).
            mask: Optional attention mask.
            causal: Whether to apply causal masking.

        Returns:
            Output tensor of shape (B, N, embed_dim).
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project Q, K, V and reshape to (B, H, N, d)
        Q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        K = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        if self.attn_type == "standard":
            attn_out, _ = standard_attention(Q, K, V, mask=mask)
        elif self.attn_type == "linear":
            attn_out = linear_attention(Q, K, V, causal=causal)
        elif self.attn_type == "local":
            attn_out = local_window_attention(Q, K, V, window_size=self.window_size)
        elif self.attn_type == "tiled":
            attn_out = tiled_exact_attention(Q, K, V, block_size=self.block_size)
        else:
            raise RuntimeError(f"Unsupported attn_type: {self.attn_type}")

        # Concatenate heads: (B, H, N, d) -> (B, N, H * d)
        concat = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(concat)
        return self.dropout(out)


def calculate_theoretical_complexity(N: int, d: int, H: int = 4, w: int = 32) -> Dict[str, Any]:
    """
    Calculate theoretical FLOPs and peak matrix memory complexity for comparison.

    Args:
        N: Sequence length.
        d: Model dimension.
        H: Number of heads.
        w: Local window size.

    Returns:
        Dictionary of Big-O formulas, FLOPs, and peak memory allocations.
    """
    d_k = d // H
    # Standard:
    # Q K^T: 2 * H * N * N * d_k = 2 * N^2 * d FLOPs
    # Attn * V: 2 * N^2 * d FLOPs
    # Total attention matrix bytes: H * N^2 * 4 bytes
    standard_flops = 4 * (N ** 2) * d
    standard_matrix_bytes = H * (N ** 2) * 4

    # Linear:
    # K^T V: 2 * H * d_k * N * d_k = 2 * N * d * d_k FLOPs
    # Q @ (KV): 2 * N * d * d_k FLOPs
    # Total state bytes: H * (d_k ** 2) * 4 bytes + N * d * 4 bytes
    linear_flops = 4 * N * d * d_k
    linear_matrix_bytes = (H * (d_k ** 2) + N * d) * 4

    # Local:
    local_flops = 4 * N * w * d
    local_matrix_bytes = H * N * w * 4

    # Tiled:
    tiled_flops = standard_flops
    tiled_matrix_bytes = (N * d + (64 ** 2)) * 4

    return {
        "N": N,
        "d": d,
        "standard_flops": standard_flops,
        "standard_memory_bytes": standard_matrix_bytes,
        "linear_flops": linear_flops,
        "linear_memory_bytes": linear_matrix_bytes,
        "local_flops": local_flops,
        "local_memory_bytes": local_matrix_bytes,
        "tiled_flops": tiled_flops,
        "tiled_memory_bytes": tiled_matrix_bytes,
    }
