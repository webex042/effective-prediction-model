"""
EPM Stage 13 — Research Metrics
===============================

Self-contained, documented metric implementations used by the Stage 13
pipeline. Keeping them here (rather than importing from the Stage 9 benchmark
module) means the research report can cite exactly how each number is computed.

Forecasting (point)
    MAE  = mean |y - ŷ|
    RMSE = sqrt(mean (y - ŷ)²)

Probabilistic (ensemble forecasts, K Monte-Carlo trajectories)
    CRPS = E|X - y| - ½ E|X - X'|   (energy form; lower is better)
    PICP = fraction of targets inside [q_low, q_high]  (should ≈ nominal)
    MPIW = mean (q_high - q_low)     (sharpness; lower is better at equal PICP)

Efficiency
    latency: median & IQR of per-call wall-clock over `reps` timed runs
    throughput: items processed per second at the median latency

Uncertainty of the estimate
    bootstrap_ci: non-parametric percentile confidence interval
    aggregate_seeds: mean ± std (+ bootstrap CI) across repeated seeds
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Point forecasting
# --------------------------------------------------------------------------- #
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff ** 2)))


# --------------------------------------------------------------------------- #
# Probabilistic forecasting
# --------------------------------------------------------------------------- #
def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> float:
    """CRPS for an ensemble forecast.

    Parameters
    ----------
    y_true : (N, H) ground-truth future values.
    samples : (N, K, H) K Monte-Carlo trajectories per window.
    """
    y_true = np.asarray(y_true)
    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError(f"samples must be (N, K, H); got {samples.shape}")
    term1 = np.mean(np.abs(samples - y_true[:, None, :]), axis=1)          # (N, H)
    diffs = np.abs(samples[:, :, None, :] - samples[:, None, :, :])        # (N,K,K,H)
    term2 = 0.5 * np.mean(diffs, axis=(1, 2))                              # (N, H)
    return float(np.mean(term1 - term2))


def interval_coverage(
    y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray
) -> Tuple[float, float]:
    """Return (PICP, MPIW) for a predictive interval."""
    y_true = np.asarray(y_true)
    inside = (y_true >= np.asarray(q_lower)) & (y_true <= np.asarray(q_upper))
    picp = float(np.mean(inside))
    mpiw = float(np.mean(np.asarray(q_upper) - np.asarray(q_lower)))
    return picp, mpiw


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #
def measure_latency(
    fn: Callable[[], Any],
    warmup: int = 10,
    reps: int = 50,
    n_items: int = 1,
) -> Dict[str, float]:
    """Time ``fn`` repeatedly and summarise latency robustly (median + IQR).

    ``n_items`` is the number of logical items processed per call, used to
    report throughput (items / second).
    """
    for _ in range(max(0, warmup)):
        fn()
    times_ms: List[float] = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times_ms)
    median = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    throughput = (n_items / (median / 1000.0)) if median > 0 else 0.0
    return {
        "latency_ms_median": median,
        "latency_ms_iqr": float(q3 - q1),
        "latency_ms_mean": float(np.mean(arr)),
        "latency_ms_std": float(np.std(arr)),
        "throughput_per_sec": float(throughput),
    }


# --------------------------------------------------------------------------- #
# Statistical variability
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Non-parametric percentile bootstrap CI for a statistic of ``values``."""
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples, dtype=float)
    n = vals.size
    for i in range(resamples):
        stats[i] = statistic(vals[rng.integers(0, n, n)])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def aggregate_seeds(
    per_seed: List[Dict[str, float]],
    resamples: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of per-seed metric dicts into mean/std/CI per metric."""
    if not per_seed:
        return {}
    keys = set().union(*[d.keys() for d in per_seed])
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(keys):
        vals = [float(d[k]) for d in per_seed if k in d and _is_number(d[k])]
        if not vals:
            continue
        ci_lo, ci_hi = bootstrap_ci(vals, resamples=resamples)
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n": len(vals),
        }
    return out


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)
