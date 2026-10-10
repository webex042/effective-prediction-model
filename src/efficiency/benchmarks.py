"""
Computational Efficiency & Sequence Scaling Benchmarks
======================================================
Systematic benchmarking of Attention mechanisms across sequence lengths:
    N in [64, 128, 256, 512, 1024, 2048]

Evaluates:
    - Wall-clock execution time (ms)
    - Memory allocation & scaling slope (quadratic vs linear)
    - Peak memory reduction factor
    - Output reconstruction parity vs exact attention
    - Downstream multi-sensor telemetry forecasting RMSE / MAE
"""

import time
import math
import gc
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .attention import (
    standard_attention,
    linear_attention,
    local_window_attention,
    tiled_exact_attention,
    MultiHeadEfficientAttention,
    calculate_theoretical_complexity,
)
from src.timeseries.dataset import generate_multivariate_sensor_telemetry, create_sliding_windows


def measure_peak_memory_and_latency(
    fn,
    *args,
    warmup: int = 3,
    reps: int = 10,
    **kwargs,
) -> Tuple[float, float]:
    """
    Measure average wall-clock latency (ms) and peak memory delta (KB).

    Args:
        fn: Function to profile.
        warmup: Warmup iterations.
        reps: Measurement iterations.

    Returns:
        Tuple of (mean_latency_ms, estimated_memory_kb).
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)

    gc.collect()
    t_start = time.perf_counter()
    for _ in range(reps):
        res = fn(*args, **kwargs)
    t_end = time.perf_counter()

    latency_ms = (t_end - t_start) * 1000.0 / reps

    # Estimate memory footprint of tensors in result and internal buffers
    if isinstance(res, tuple):
        out_tensor = res[0]
    else:
        out_tensor = res
    mem_kb = (out_tensor.numel() * out_tensor.element_size()) / 1024.0

    return float(latency_ms), float(mem_kb)


def benchmark_attention_scaling(
    seq_lengths: List[int] = [64, 128, 256, 512, 1024, 2048],
    embed_dim: int = 64,
    num_heads: int = 4,
    batch_size: int = 2,
    warmup: int = 3,
    reps: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Profile execution latency and memory scaling across sequence lengths.
    """
    torch.manual_seed(seed)
    d_k = embed_dim // num_heads

    records = []

    for N in seq_lengths:
        Q = torch.randn(batch_size, num_heads, N, d_k)
        K = torch.randn(batch_size, num_heads, N, d_k)
        V = torch.randn(batch_size, num_heads, N, d_k)

        # 1. Standard Quadratic Attention
        lat_std, _ = measure_peak_memory_and_latency(
            standard_attention, Q, K, V, warmup=warmup, reps=reps
        )
        # Memory matrix footprint: B * H * N^2 * 4 bytes
        matrix_mem_kb = (batch_size * num_heads * (N ** 2) * 4) / 1024.0

        out_std, _ = standard_attention(Q, K, V)

        # 2. Linear Attention
        lat_lin, _ = measure_peak_memory_and_latency(
            linear_attention, Q, K, V, warmup=warmup, reps=reps
        )
        out_lin = linear_attention(Q, K, V)
        lin_mem_kb = (batch_size * num_heads * (d_k ** 2) * 4 + batch_size * num_heads * N * d_k * 4) / 1024.0

        # 3. Local Window Attention (w=32)
        window_size = min(32, N)
        lat_loc, _ = measure_peak_memory_and_latency(
            local_window_attention, Q, K, V, window_size=window_size, warmup=warmup, reps=reps
        )
        loc_mem_kb = (batch_size * num_heads * N * window_size * 4) / 1024.0

        # 4. Tiled Exact Attention
        block_size = min(64, N)
        lat_tiled, _ = measure_peak_memory_and_latency(
            tiled_exact_attention, Q, K, V, block_size=block_size, warmup=warmup, reps=reps
        )
        out_tiled = tiled_exact_attention(Q, K, V, block_size=block_size)
        tiled_mem_kb = (batch_size * num_heads * N * d_k * 4 + (block_size ** 2) * 4) / 1024.0

        # Verify exact numerical equivalence of Tiled Attention
        tiled_max_err = float(torch.max(torch.abs(out_tiled - out_std)).item())

        complexity = calculate_theoretical_complexity(N=N, d=embed_dim, H=num_heads, w=window_size)

        records.append({
            "sequence_length": N,
            "standard_latency_ms": round(lat_std, 3),
            "linear_latency_ms": round(lat_lin, 3),
            "local_latency_ms": round(lat_loc, 3),
            "tiled_latency_ms": round(lat_tiled, 3),
            "standard_mem_kb": round(matrix_mem_kb, 1),
            "linear_mem_kb": round(lin_mem_kb, 1),
            "local_mem_kb": round(loc_mem_kb, 1),
            "tiled_mem_kb": round(tiled_mem_kb, 1),
            "tiled_numerical_error": round(tiled_max_err, 6),
            "linear_speedup_vs_standard": round(lat_std / (lat_lin + 1e-6), 2),
            "standard_flops": complexity["standard_flops"],
            "linear_flops": complexity["linear_flops"],
        })

    return {"scaling_records": records}


class SequenceForecaster(nn.Module):
    """Temporal Sequence Forecaster parameterized by an Attention Module."""

    def __init__(self, in_features: int = 5, embed_dim: int = 64, num_heads: int = 4, attn_type: str = "standard", horizon: int = 12):
        super().__init__()
        self.in_proj = nn.Linear(in_features, embed_dim)
        self.attn = MultiHeadEfficientAttention(embed_dim=embed_dim, num_heads=num_heads, attn_type=attn_type)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_features)
        h = self.in_proj(x)
        h = h + self.attn(h)
        h = self.norm(h)
        # Pool across sequence and predict horizon
        h_pool = h.mean(dim=1)
        out = self.head(h_pool)
        return out


def benchmark_downstream_sensor_forecasting(
    seq_len: int = 128,
    horizon: int = 12,
    epochs: int = 15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Evaluates downstream forecasting RMSE across Attention paradigms on long telemetry windows.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Synthesize telemetry
    df, _ = generate_multivariate_sensor_telemetry(n_timesteps=1200, random_seed=seed)
    data = df.values.astype(np.float32)
    # Standardize
    train_len = int(len(data) * 0.7)
    mean_val = np.mean(data[:train_len], axis=0, keepdims=True)
    std_val = np.std(data[:train_len], axis=0, keepdims=True) + 1e-6
    data_norm = (data - mean_val) / std_val

    X, y = create_sliding_windows(data_norm, window_size=seq_len, horizon=horizon, stride=2, target_channel=0)
    split_idx = int(len(X) * 0.75)
    X_train, y_train = torch.from_numpy(X[:split_idx]), torch.from_numpy(y[:split_idx])
    X_test, y_test = torch.from_numpy(X[split_idx:]), torch.from_numpy(y[split_idx:])

    dataset_train = TensorDataset(X_train, y_train)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    attn_types = [
        ("Standard Attention", "standard", "O(N² d)", "O(N²)"),
        ("Tiled Exact Attention", "tiled", "O(N² d)", "O(N d)"),
        ("Kernel Linear Attention", "linear", "O(N d²)", "O(N d)"),
        ("Local Window Attention", "local", "O(N w d)", "O(N w)"),
    ]

    results = []

    for name, a_type, time_comp, space_comp in attn_types:
        model = SequenceForecaster(in_features=5, embed_dim=64, num_heads=4, attn_type=a_type, horizon=horizon)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        criterion = nn.MSELoss()

        t0 = time.perf_counter()
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader_train:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        train_time = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            t_eval_0 = time.perf_counter()
            test_preds = model(X_test)
            eval_latency = (time.perf_counter() - t_eval_0) * 1000.0 / len(X_test)
            test_mse = float(criterion(test_preds, y_test).item())
            test_rmse = float(math.sqrt(test_mse))
            test_mae = float(torch.mean(torch.abs(test_preds - y_test)).item())

        results.append({
            "Mechanism": name,
            "Attn Type": a_type,
            "Time Complexity": time_comp,
            "Space Complexity": space_comp,
            "Test RMSE": round(test_rmse, 4),
            "Test MAE": round(test_mae, 4),
            "Latency / Sample": f"{eval_latency:.3f} ms",
            "Training Time": f"{train_time:.2f} s",
            "Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        })

    return results


def run_stage11_benchmarks() -> Dict[str, Any]:
    """Execute all Stage 11 efficiency benchmarks."""
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    scaling_data = benchmark_attention_scaling(seq_lengths=seq_lengths)
    forecasting_data = benchmark_downstream_sensor_forecasting(seq_len=128, horizon=12, epochs=15)

    return {
        "scaling_records": scaling_data["scaling_records"],
        "forecasting_summary": forecasting_data,
        "sequence_lengths_tested": seq_lengths,
    }
