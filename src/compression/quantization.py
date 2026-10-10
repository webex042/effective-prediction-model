"""
EPM Stage 12 — Quantization Module
===================================

Implements scratch uniform affine / symmetric quantization, SQNR analysis,
and PyTorch Post-Training Dynamic Quantization (PTQ) to INT8.

MATHEMATICAL FOUNDATION
-----------------------
1. Uniform Affine Quantization (Asymmetric):
   Maps continuous real values x in [x_min, x_max] to b-bit discrete integers q in [q_min, q_max]:
     Scale S = (x_max - x_min) / (q_max - q_min)
     Zero-point Z = round(-x_min / S) + q_min
     Quantize:   x_q = clamp(round(x / S) + Z, q_min, q_max)
     Dequantize: x_hat = S * (x_q - Z)

2. Symmetric Quantization:
   Zero-point Z = 0 (exact zero mapped to integer 0):
     Scale S = max(|x_min|, |x_max|) / (2^(b - 1) - 1)
     Quantize:   x_q = clamp(round(x / S), -2^(b-1) + 1, 2^(b-1) - 1)
     Dequantize: x_hat = S * x_q

3. Signal-to-Quantization-Noise Ratio (SQNR):
   Measures signal fidelity after quantization in decibels (dB):
     SQNR = 10 * log10( sum(x^2) / (sum((x - x_hat)^2) + eps) )
   Theoretical limit for uniform distribution: SQNR ~ 6.02 * b + 1.76 dB.
"""

from __future__ import annotations

import io
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def calculate_scale_and_zero_point(
    x_min: float,
    x_max: float,
    num_bits: int = 8,
    symmetric: bool = False,
) -> Tuple[float, int, int, int]:
    """
    Compute quantization scale S and zero-point integer Z.

    Returns
    -------
    scale : float
    zero_point : int
    q_min : int
    q_max : int
    """
    if symmetric:
        q_min = -(2 ** (num_bits - 1) - 1)
        q_max = 2 ** (num_bits - 1) - 1
        max_val = max(abs(x_min), abs(x_max))
        scale = max_val / q_max if q_max > 0 else 1.0
        scale = max(scale, 1e-8)
        zero_point = 0
    else:
        q_min = 0
        q_max = 2 ** num_bits - 1
        # Avoid division by zero when min == max
        range_val = max(x_max - x_min, 1e-8)
        scale = range_val / (q_max - q_min)
        zero_point = int(round(-x_min / scale)) + q_min
        zero_point = max(q_min, min(q_max, zero_point))

    return scale, zero_point, q_min, q_max


def quantize_tensor(
    x: torch.Tensor,
    num_bits: int = 8,
    symmetric: bool = False,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
) -> Tuple[torch.Tensor, float, int]:
    """
    Quantize floating point tensor x into integer representation.

    Returns
    -------
    x_q : torch.Tensor (int dtype)
    scale : float
    zero_point : int
    """
    if scale is None or zero_point is None:
        x_min = x.min().item()
        x_max = x.max().item()
        s, z, q_min, q_max = calculate_scale_and_zero_point(
            x_min, x_max, num_bits=num_bits, symmetric=symmetric
        )
    else:
        s, z = scale, zero_point
        if symmetric:
            q_min = -(2 ** (num_bits - 1) - 1)
            q_max = 2 ** (num_bits - 1) - 1
        else:
            q_min = 0
            q_max = 2 ** num_bits - 1

    x_div = x / s
    x_q = torch.clamp(torch.round(x_div) + z, q_min, q_max)
    if num_bits <= 8:
        dtype = torch.int8 if symmetric else torch.uint8
    elif num_bits <= 16:
        dtype = torch.int16 if symmetric else torch.int32
    else:
        dtype = torch.int32 if symmetric else torch.int64
    return x_q.to(dtype), s, z


def dequantize_tensor(
    x_q: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    """Reconstruct floating point tensor from integer quantized values."""
    return scale * (x_q.float() - float(zero_point))


def quantize_dequantize(
    x: torch.Tensor,
    num_bits: int = 8,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Simulate quantization error via round-trip quantize -> dequantize.

    Returns
    -------
    x_recon : torch.Tensor
    sqnr_db : float
    """
    x_q, scale, zero_point = quantize_tensor(x, num_bits=num_bits, symmetric=symmetric)
    x_recon = dequantize_tensor(x_q, scale, zero_point)
    sqnr = compute_sqnr(x, x_recon)
    return x_recon, sqnr


def compute_sqnr(x: torch.Tensor, x_recon: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB).
    """
    signal_power = torch.sum(x ** 2).item()
    noise_power = torch.sum((x - x_recon) ** 2).item()
    if noise_power <= eps:
        return 100.0  # Essentially lossless
    ratio = max(signal_power / noise_power, eps)
    return float(10.0 * math.log10(ratio))


def get_model_size_kb(model: nn.Module) -> float:
    """
    Calculate model state_dict storage size in Kilobytes (KB).
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_bytes = buf.tell()
    return size_bytes / 1024.0


def quantize_model_dynamic_int8(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch Post-Training Dynamic Quantization (PTQ) to Linear layers.
    Quantizes weights to INT8 and activations dynamically at runtime.
    """
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def profile_layer_quantization_errors(model: nn.Module, num_bits: int = 8) -> Dict[str, Dict[str, float]]:
    """
    Profiles weight quantization errors (MSE and SQNR) across each linear/conv layer.
    """
    results = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            w = param.detach()
            w_recon, sqnr = quantize_dequantize(w, num_bits=num_bits, symmetric=True)
            mse = torch.mean((w - w_recon) ** 2).item()
            results[name] = {
                "sqnr_db": sqnr,
                "mse": mse,
                "shape": list(w.shape),
                "num_elements": w.numel(),
            }
    return results
