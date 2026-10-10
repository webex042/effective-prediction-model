"""
EPM Stage 13 — Research Experiment Configuration
================================================

A reproducible, serializable configuration object for Stage 13 research
experiments. Unlike scattered hyperparameters, every setting that affects a
result lives here so it can be hashed, logged, and reproduced exactly.

Design goals
------------
* Pure stdlib dataclass (no external deps) → trivially JSON-serializable.
* ``fingerprint()`` gives a short deterministic hash of the *scientifically
  relevant* fields, so two configs that would produce the same experiment
  share an id prefix.
* ``validate()`` fails fast on contradictory settings (e.g. ratios > 1).
* Preset constructors encode the hypotheses from the project roadmap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional


@dataclass
class ResearchConfig:
    """Full specification of a Stage 13 research experiment."""

    # --- Identity / hypothesis ---
    name: str = "stage13_experiment"
    hypothesis: str = "H1"  # H1 | H2 | H3 | H4
    hypothesis_statement: str = (
        "A reduced-step generative model can maintain competitive forecasting "
        "performance while reducing inference latency and computational cost "
        "compared with conventional (many-step) diffusion sampling."
    )

    # --- Dataset ---
    dataset: str = "synthetic_industrial_telemetry"
    dataset_version: str = "v1"
    n_timesteps: int = 2000
    sampling_rate_minutes: int = 10
    target_channel: int = 0  # 0 = temperature

    # --- Windowing / splits (time-aware, leakage-free) ---
    window_size: int = 24
    horizon: int = 12
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train_ratio - val_ratio

    # --- Model (conditional diffusion forecaster) ---
    model_arch: str = "ts_diffusion"
    hidden_dim: int = 128
    context_dim: int = 64
    schedule_type: str = "cosine"
    diffusion_timesteps: int = 100

    # --- Baselines to train & compare (real models only) ---
    baselines: List[str] = field(
        default_factory=lambda: ["linear", "lstm", "transformer"]
    )

    # --- Training ---
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1.0e-3

    # --- Sampling / efficiency (H1: vary NFE on the *same* trained weights) ---
    sampling_steps: List[int] = field(default_factory=lambda: [100, 50, 20, 10, 5])
    n_mc_samples: int = 16
    eta: float = 0.3  # DDIM stochasticity (0 = deterministic ODE)

    # --- Reproducibility / benchmarking ---
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    latency_warmup: int = 10
    latency_reps: int = 50
    bootstrap_resamples: int = 1000
    device: str = "cpu"

    # --- Optional compression hook (H4) ---
    compression_method: Optional[str] = None  # None | "dynamic_int8"

    # ------------------------------------------------------------------ #
    def validate(self) -> "ResearchConfig":
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0,1), got {self.train_ratio}")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0,1), got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                "train_ratio + val_ratio must leave room for a test split "
                f"(got {self.train_ratio} + {self.val_ratio})"
            )
        if self.horizon < 1 or self.window_size < 1:
            raise ValueError("window_size and horizon must be >= 1")
        if any(s < 1 or s > self.diffusion_timesteps for s in self.sampling_steps):
            raise ValueError(
                f"every sampling_steps value must be in [1, {self.diffusion_timesteps}]"
            )
        if self.n_mc_samples < 2:
            raise ValueError("n_mc_samples must be >= 2 for probabilistic metrics")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.hypothesis not in ("H1", "H2", "H3", "H4"):
            raise ValueError(f"unknown hypothesis {self.hypothesis!r}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def fingerprint(self) -> str:
        """Deterministic 8-char hash of the scientifically relevant fields."""
        relevant = {
            k: getattr(self, k)
            for k in (
                "hypothesis", "dataset", "dataset_version", "n_timesteps",
                "window_size", "horizon", "target_channel", "train_ratio",
                "val_ratio", "model_arch", "hidden_dim", "context_dim",
                "schedule_type", "diffusion_timesteps", "baselines", "epochs",
                "batch_size", "lr", "sampling_steps", "n_mc_samples", "eta",
                "seeds", "compression_method",
            )
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]

    # ------------------------------------------------------------------ #
    # Presets
    # ------------------------------------------------------------------ #
    @classmethod
    def h1_reduced_step_forecasting(cls, **overrides: Any) -> "ResearchConfig":
        """Primary hypothesis: reduced-step DDIM vs full-step diffusion."""
        cfg = cls(name="h1_reduced_step_forecasting", hypothesis="H1")
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg.validate()

    @classmethod
    def quick(cls, **overrides: Any) -> "ResearchConfig":
        """Fast variant for CI / smoke tests (tiny data, 1 seed, few epochs)."""
        cfg = cls(
            name="h1_quick_smoke",
            hypothesis="H1",
            n_timesteps=400,
            window_size=12,
            horizon=6,
            hidden_dim=32,
            context_dim=16,
            diffusion_timesteps=40,
            baselines=["linear", "lstm"],
            epochs=2,
            sampling_steps=[40, 10, 5],
            n_mc_samples=4,
            seeds=[0],
            latency_reps=5,
            bootstrap_resamples=100,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg.validate()
