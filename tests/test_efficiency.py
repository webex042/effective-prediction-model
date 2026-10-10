"""
Unit Tests for Stage 11 — Computational Efficiency & Attention Scaling
======================================================================
"""

import math
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.efficiency.attention import (
    standard_attention,
    linear_attention,
    local_window_attention,
    tiled_exact_attention,
    MultiHeadEfficientAttention,
    calculate_theoretical_complexity,
)


class TestAttentionImplementations:
    """Verifies output shapes, numerical stability, and mathematical correctness."""

    def test_standard_attention_shapes_and_softmax_sum(self):
        B, H, N, d = 2, 4, 32, 16
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        out, attn_weights = standard_attention(Q, K, V)
        assert out.shape == (B, H, N, d)
        assert attn_weights.shape == (B, H, N, N)

        # Softmax weights must sum to 1.0 along the key dimension
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_linear_attention_bidirectional_shapes_and_finite(self):
        B, H, N, d = 2, 4, 48, 16
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        out = linear_attention(Q, K, V, causal=False)
        assert out.shape == (B, H, N, d)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_linear_attention_causal_shapes_and_no_future_leakage(self):
        B, H, N, d = 1, 1, 8, 4
        # Create base input
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        out_base = linear_attention(Q, K, V, causal=True)

        # Perturb token at future position index 6 and 7
        K_perturbed = K.clone()
        V_perturbed = V.clone()
        K_perturbed[:, :, 6:, :] += 50.0
        V_perturbed[:, :, 6:, :] += 50.0

        out_perturbed = linear_attention(Q, K_perturbed, V_perturbed, causal=True)

        # Outputs at positions 0 through 5 must remain IDENTICAL (no lookahead leakage!)
        assert torch.allclose(out_base[:, :, :6, :], out_perturbed[:, :, :6, :], atol=1e-5)
        # Positions 6 and 7 will differ
        assert not torch.allclose(out_base[:, :, 6:, :], out_perturbed[:, :, 6:, :], atol=1e-3)

    def test_tiled_exact_attention_mathematical_parity(self):
        """Tiled Online Softmax attention must match Standard Attention within float tolerance."""
        B, H, N, d = 2, 2, 64, 16
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        # Ground truth standard attention
        out_standard, _ = standard_attention(Q, K, V)

        # Test tiled attention across multiple block sizes
        for b_size in [16, 32, 64]:
            out_tiled = tiled_exact_attention(Q, K, V, block_size=b_size)
            assert out_tiled.shape == out_standard.shape
            max_err = torch.max(torch.abs(out_tiled - out_standard)).item()
            assert max_err < 1e-4, f"Block size {b_size} exceeded tolerance: max error {max_err}"

    def test_local_window_attention_shapes(self):
        B, H, N, d = 2, 2, 40, 16
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        out_local = local_window_attention(Q, K, V, window_size=16)
        assert out_local.shape == (B, H, N, d)
        assert not torch.isnan(out_local).any()


class TestMultiHeadEfficientAttention:
    """Verifies PyTorch module gradients and backward passes across all attention types."""

    @pytest.mark.parametrize("attn_type", ["standard", "linear", "local", "tiled"])
    def test_multi_head_attention_forward_and_backward(self, attn_type):
        B, N, D = 2, 32, 32
        mha = MultiHeadEfficientAttention(embed_dim=D, num_heads=4, attn_type=attn_type, window_size=16, block_size=16)
        x = torch.randn(B, N, D, requires_grad=True)

        out = mha(x)
        assert out.shape == (B, N, D)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        for p in mha.parameters():
            assert p.grad is not None


class TestComplexityAndScaling:
    """Verifies theoretical scaling laws."""

    def test_theoretical_complexity_ratios(self):
        # When N quadruples (e.g. 512 -> 2048):
        c_512 = calculate_theoretical_complexity(N=512, d=64)
        c_2048 = calculate_theoretical_complexity(N=2048, d=64)

        # Standard Attention FLOPs scale quadratically: (2048/512)^2 = 16x
        ratio_std_flops = c_2048["standard_flops"] / c_512["standard_flops"]
        assert np.isclose(ratio_std_flops, 16.0)

        # Linear Attention FLOPs scale linearly: 2048/512 = 4x
        ratio_lin_flops = c_2048["linear_flops"] / c_512["linear_flops"]
        assert np.isclose(ratio_lin_flops, 4.0)

        # Memory matrix footprint for standard attention quadruples squared (16x)
        ratio_std_mem = c_2048["standard_memory_bytes"] / c_512["standard_memory_bytes"]
        assert np.isclose(ratio_std_mem, 16.0)
