"""
EPM Stage 13 — Research Experiment Pipeline
===========================================

An honest, reproducible pipeline for the primary hypothesis:

    H1: a reduced-step generative (diffusion) forecaster can retain competitive
        forecasting accuracy while cutting inference cost versus many-step
        diffusion sampling.

How integrity is preserved
--------------------------
* Real models only. Baselines (linear least-squares, LSTM, Transformer) are
  genuinely trained on the same data as the diffusion forecaster — nothing is
  "ground truth + noise".
* Leakage-free: chronological split (no shuffling) and the standardizer is fit
  on **training windows only**, then applied to val/test.
* H1 is tested by sampling the *same* trained diffusion weights at different
  DDIM step counts (NFE), so only inference cost varies.
* Multiple seeds → mean ± std and bootstrap confidence intervals.

All metrics are de-normalized to physical units (e.g. °C for temperature)
before reporting, so errors are interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.reproducibility import set_seed
from src.timeseries.dataset import (
    generate_multivariate_sensor_telemetry,
    create_sliding_windows,
    temporal_train_test_split,
)
from src.timeseries.models import (
    LSTMForecaster,
    GRUForecaster,
    TimeSeriesTransformer,
    train_forecasting_model,
)
from src.diffusion.schedules import DiffusionSchedule
from src.diffusion.timeseries_diffusion import TimeSeriesDiffusionForecaster
from src.compression.quantization import get_model_size_kb
from src.research.config import ResearchConfig
from src.research.registry import ExperimentRegistry, ExperimentRecord
from src.research import metrics as M

@dataclass
class PreparedData:
    """Leakage-free, standardized windows for one seed."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    target_mean: float
    target_std: float
    feature_names: List[str]

    @property
    def in_channels(self) -> int:
        return self.X_train.shape[-1]


def _fit_channel_stats(x_train: np.ndarray):
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return mean, std


def prepare_data(cfg: ResearchConfig, seed: int) -> PreparedData:
    """Generate telemetry, window it, split chronologically, standardize on train."""
    set_seed(seed)
    df, names = generate_multivariate_sensor_telemetry(
        n_timesteps=cfg.n_timesteps,
        sampling_rate_minutes=cfg.sampling_rate_minutes,
        random_seed=seed,
    )
    values = df.values.astype(np.float32)
    X, y = create_sliding_windows(
        values, window_size=cfg.window_size, horizon=cfg.horizon,
        target_channel=cfg.target_channel,
    )
    X_tr, y_tr, X_va, y_va, X_te, y_te = temporal_train_test_split(
        X, y, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
    )
    mean, std = _fit_channel_stats(X_tr)  # TRAIN ONLY — no leakage
    t_mean = float(mean[cfg.target_channel])
    t_std = float(std[cfg.target_channel])

    def nx(a: np.ndarray) -> np.ndarray:
        return ((a - mean) / std).astype(np.float32)

    def ny(a: np.ndarray) -> np.ndarray:
        return ((a - t_mean) / t_std).astype(np.float32)

    return PreparedData(
        nx(X_tr), ny(y_tr), nx(X_va), ny(y_va), nx(X_te), ny(y_te),
        t_mean, t_std, list(names),
    )


def _denorm(arr: np.ndarray, data: PreparedData) -> np.ndarray:
    return arr * data.target_std + data.target_mean


class LinearForecaster:
    """Closed-form least-squares autoregressor: flattened window -> horizon."""

    def __init__(self, weight: np.ndarray) -> None:
        self.weight = weight  # (n_features + 1, horizon)

    @property
    def num_params(self) -> int:
        return int(self.weight.size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        flat = X.reshape(X.shape[0], -1)
        aug = np.concatenate([flat, np.ones((len(flat), 1), dtype=flat.dtype)], axis=1)
        return aug @ self.weight


def _train_linear(data: PreparedData) -> LinearForecaster:
    flat = data.X_train.reshape(data.X_train.shape[0], -1)
    aug = np.concatenate([flat, np.ones((len(flat), 1), dtype=flat.dtype)], axis=1)
    weight, *_ = np.linalg.lstsq(aug, data.y_train, rcond=None)
    return LinearForecaster(weight)


def _predict_torch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).cpu().numpy()


def train_baseline(name: str, cfg: ResearchConfig, data: PreparedData, seed: int):
    """Train one real baseline. Returns an object exposing ``predict`` + param count."""
    set_seed(seed)
    name = name.lower()
    if name == "linear":
        return _train_linear(data)

    in_dim = data.in_channels
    rnn_hidden = max(16, cfg.hidden_dim // 4)
    if name == "lstm":
        model = LSTMForecaster(in_dim, hidden_dim=rnn_hidden, horizon=cfg.horizon)
    elif name == "gru":
        model = GRUForecaster(in_dim, hidden_dim=rnn_hidden, horizon=cfg.horizon)
    elif name == "transformer":
        model = TimeSeriesTransformer(in_dim, d_model=32, num_heads=4, horizon=cfg.horizon)
    else:
        raise ValueError(f"unknown baseline {name!r}")

    train_forecasting_model(
        model, data.X_train, data.y_train, data.X_val, data.y_val,
        epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
    )
    return model


def train_diffusion(cfg: ResearchConfig, data: PreparedData, seed: int) -> TimeSeriesDiffusionForecaster:
    """Train the conditional diffusion forecaster (trained once; sampled at many NFE)."""
    set_seed(seed)
    schedule = DiffusionSchedule(timesteps=cfg.diffusion_timesteps, schedule_type=cfg.schedule_type)
    model = TimeSeriesDiffusionForecaster(
        schedule=schedule, in_channels=data.in_channels,
        lookback_window=cfg.window_size, horizon=cfg.horizon,
        hidden_dim=cfg.hidden_dim, context_dim=cfg.context_dim,
    )
    ds = TensorDataset(
        torch.from_numpy(data.X_train).float(), torch.from_numpy(data.y_train).float()
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = model.compute_loss(xb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def eval_point_model(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    data: PreparedData,
    cfg: ResearchConfig,
) -> Dict[str, float]:
    """Evaluate a deterministic forecaster (MAE/RMSE in physical units + latency)."""
    pred = _denorm(predict_fn(data.X_test), data)
    y = _denorm(data.y_test, data)
    out: Dict[str, float] = {"mae": M.mae(y, pred), "rmse": M.rmse(y, pred)}
    n_items = min(64, len(data.X_test))
    batch = data.X_test[:n_items]
    lat = M.measure_latency(
        lambda: predict_fn(batch),
        warmup=cfg.latency_warmup, reps=cfg.latency_reps, n_items=n_items,
    )
    out["latency_ms"] = lat["latency_ms_median"] / max(n_items, 1)
    out["throughput"] = lat["throughput_per_sec"]
    return out


def eval_diffusion_stepsweep(
    model: TimeSeriesDiffusionForecaster,
    data: PreparedData,
    cfg: ResearchConfig,
    seed: int,
) -> List[Dict[str, float]]:
    """Evaluate the SAME diffusion weights across the configured DDIM step counts."""
    set_seed(seed)
    n_eval = min(len(data.X_test), 100)
    x_eval = torch.from_numpy(data.X_test[:n_eval]).float()
    y_eval = _denorm(data.y_test[:n_eval], data)
    single = x_eval[:1]
    rows: List[Dict[str, float]] = []

    for steps in cfg.sampling_steps:
        with torch.no_grad():
            samples = model.sample_trajectories(
                x_eval, n_samples=cfg.n_mc_samples, num_steps=steps,
                eta=cfg.eta, device=cfg.device,
            ).cpu().numpy()  # (N, K, H)
        samples = samples * data.target_std + data.target_mean
        median = np.median(samples, axis=1)
        q05 = np.percentile(samples, 5, axis=1)
        q95 = np.percentile(samples, 95, axis=1)
        picp, mpiw = M.interval_coverage(y_eval, q05, q95)
        lat = M.measure_latency(
            lambda: model.sample_trajectories(
                single, n_samples=cfg.n_mc_samples, num_steps=steps,
                eta=cfg.eta, device=cfg.device,
            ),
            warmup=2, reps=max(5, cfg.latency_reps // 5), n_items=1,
        )
        rows.append({
            "sampling_steps": int(steps),
            "mae": M.mae(y_eval, median),
            "rmse": M.rmse(y_eval, median),
            "crps": M.crps_ensemble(y_eval, samples),
            "picp_90": picp,
            "mpiw_90": mpiw,
            "latency_ms": lat["latency_ms_median"],
            "throughput": lat["throughput_per_sec"],
        })
    return rows


def _h1_analysis(diffusion_agg: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Compare reduced-step DDIM against the full-step reference (highest NFE)."""
    steps_desc = sorted(diffusion_agg.keys(), reverse=True)
    if not steps_desc:
        return {}
    full = steps_desc[0]
    full_rmse = diffusion_agg[full]["rmse"]
    full_lat = diffusion_agg[full]["latency_ms"]
    comparisons = []
    for s in steps_desc[1:]:
        r = diffusion_agg[s]["rmse"]
        lat = diffusion_agg[s]["latency_ms"]
        speedup = full_lat["mean"] / lat["mean"] if lat["mean"] > 0 else float("nan")
        rmse_change = (
            (r["mean"] - full_rmse["mean"]) / full_rmse["mean"] * 100.0
            if full_rmse["mean"] > 0 else float("nan")
        )
        comparisons.append({
            "sampling_steps": s,
            "speedup_vs_full": round(speedup, 2),
            "rmse_change_pct": round(rmse_change, 2),
            "rmse_within_full_ci": bool(r["mean"] <= full_rmse["ci_high"]),
            "nfe_reduction": round(full / s, 2) if s > 0 else float("nan"),
        })
    return {
        "full_steps": full,
        "full_rmse_mean": round(full_rmse["mean"], 4),
        "full_latency_ms_mean": round(full_lat["mean"], 3),
        "comparisons": comparisons,
    }


def run_experiment(
    cfg: ResearchConfig,
    registry: Optional[ExperimentRegistry] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full multi-seed H1 experiment; persist registry records; return summary."""
    cfg.validate()
    reg = registry or ExperimentRegistry()

    per_seed_baseline: Dict[str, List[Dict[str, float]]] = {b: [] for b in cfg.baselines}
    per_seed_diffusion: Dict[int, List[Dict[str, float]]] = {s: [] for s in cfg.sampling_steps}
    final_baseline_models: Dict[str, Any] = {}
    final_diffusion: Optional[TimeSeriesDiffusionForecaster] = None
    split_info: Dict[str, int] = {}

    for i, seed in enumerate(cfg.seeds):
        if verbose:
            print(f"  [seed {seed}] preparing data + training ({i+1}/{len(cfg.seeds)})")
        data = prepare_data(cfg, seed)
        split_info = {
            "train": len(data.X_train), "val": len(data.X_val), "test": len(data.X_test),
        }
        for b in cfg.baselines:
            model = train_baseline(b, cfg, data, seed)
            predict = model.predict if hasattr(model, "predict") else (lambda X, m=model: _predict_torch(m, X))
            per_seed_baseline[b].append(eval_point_model(predict, data, cfg))
            final_baseline_models[b] = model
        dmodel = train_diffusion(cfg, data, seed)
        for row in eval_diffusion_stepsweep(dmodel, data, cfg, seed):
            per_seed_diffusion[row.pop("sampling_steps")].append(row)
        final_diffusion = dmodel

    from src.research.registry import _software_env

    baseline_agg = {
        b: M.aggregate_seeds(v, resamples=cfg.bootstrap_resamples)
        for b, v in per_seed_baseline.items()
    }
    diffusion_agg = {
        s: M.aggregate_seeds(v, resamples=cfg.bootstrap_resamples)
        for s, v in per_seed_diffusion.items()
    }

    def _psize(model) -> tuple[int, float]:
        if hasattr(model, "num_params"):
            n = int(model.num_params)
            return n, round(n * 4 / 1024.0, 2)
        n = _count_params(model)
        return n, round(get_model_size_kb(model), 2)

    record_ids: List[str] = []
    results_baselines: Dict[str, Any] = {}
    for b in cfg.baselines:
        n, size = _psize(final_baseline_models[b])
        agg = baseline_agg[b]
        results_baselines[b] = {"metrics": agg, "parameter_count": n, "model_size_kb": size}
        rec = ExperimentRecord(
            experiment_id=reg.new_id(f"s13_{b}", cfg.fingerprint()),
            name=f"{cfg.name}:{b}", stage=13, hypothesis=cfg.hypothesis,
            model_architecture=f"baseline_{b}", method="deterministic_baseline",
            config=cfg.to_dict(), dataset=cfg.dataset, dataset_version=cfg.dataset_version,
            split=split_info, seeds=list(cfg.seeds), parameter_count=n, model_size_kb=size,
            inference_latency_ms=agg.get("latency_ms", {}).get("mean"),
            throughput_per_sec=agg.get("throughput", {}).get("mean"),
            predictive_metrics={k: agg[k] for k in ("mae", "rmse") if k in agg},
            limitations=["CPU-only timing", "synthetic telemetry dataset"],
        )
        reg.save(rec)
        record_ids.append(rec.experiment_id)

    results_diffusion: Dict[int, Any] = {}
    dn, dsize = _psize(final_diffusion)
    for s in cfg.sampling_steps:
        agg = diffusion_agg[s]
        results_diffusion[s] = {
            "metrics": agg, "parameter_count": dn, "model_size_kb": dsize, "sampling_steps": s,
        }
        rec = ExperimentRecord(
            experiment_id=reg.new_id(f"s13_ddim{s}", cfg.fingerprint()),
            name=f"{cfg.name}:ddim_{s}", stage=13, hypothesis=cfg.hypothesis,
            model_architecture="ts_diffusion", method=f"ddim_{s}steps",
            config=cfg.to_dict(), dataset=cfg.dataset, dataset_version=cfg.dataset_version,
            split=split_info, seeds=list(cfg.seeds), parameter_count=dn, model_size_kb=dsize,
            sampling_steps=s,
            inference_latency_ms=agg.get("latency_ms", {}).get("mean"),
            throughput_per_sec=agg.get("throughput", {}).get("mean"),
            predictive_metrics={k: agg[k] for k in ("mae", "rmse") if k in agg},
            probabilistic_metrics={k: agg[k] for k in ("crps", "picp_90", "mpiw_90") if k in agg},
            limitations=["CPU-only timing", "synthetic telemetry dataset"],
        )
        reg.save(rec)
        record_ids.append(rec.experiment_id)

    return {
        "config": cfg.to_dict(),
        "fingerprint": cfg.fingerprint(),
        "split": split_info,
        "baselines": results_baselines,
        "diffusion": results_diffusion,
        "h1_analysis": _h1_analysis(diffusion_agg),
        "record_ids": record_ids,
        "environment": _software_env(),
        "seeds": list(cfg.seeds),
    }
