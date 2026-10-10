"""
EPM Computational Efficiency Module
====================================
Linear Attention, Tiled Exact Attention, Local Sliding Windows, and Complexity Analysis.
"""

from .attention import (
    standard_attention,
    linear_attention,
    local_window_attention,
    tiled_exact_attention,
    MultiHeadEfficientAttention,
    calculate_theoretical_complexity,
    phi_elu,
)
from .benchmarks import (
    benchmark_attention_scaling,
    benchmark_downstream_sensor_forecasting,
    run_stage11_benchmarks,
)

__all__ = [
    "standard_attention",
    "linear_attention",
    "local_window_attention",
    "tiled_exact_attention",
    "MultiHeadEfficientAttention",
    "calculate_theoretical_complexity",
    "phi_elu",
    "benchmark_attention_scaling",
    "benchmark_downstream_sensor_forecasting",
    "run_stage11_benchmarks",
]
