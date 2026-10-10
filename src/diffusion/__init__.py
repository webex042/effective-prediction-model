"""
EPM Diffusion Module
====================

Probabilistic Generative Modeling via Denoising Diffusion:
- Linear and Cosine Variance Schedules (DiffusionSchedule)
- Sinusoidal Timestep Embeddings and Residual Denoising MLP (DenoisingMLP)
- Gaussian Diffusion Process (DDPM & DDIM Sampling)
- Systematic Generation Benchmarks & Uncertainty Analysis
"""

from src.diffusion.schedules import DiffusionSchedule
from src.diffusion.models import SinusoidalTimeEmbedding, ResidualBlock, DenoisingMLP
from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.diffusion.benchmarks import (
    train_diffusion_model,
    run_diffusion_benchmark,
    compute_multivariate_mmd,
)
from src.diffusion.timeseries_diffusion import (
    HistoricalContextEncoder,
    TrajectoryDenoisingNet,
    TimeSeriesDiffusionForecaster,
)
from src.diffusion.timeseries_benchmarks import (
    compute_crps,
    compute_interval_coverage,
    train_timeseries_diffusion,
    run_timeseries_diffusion_benchmark,
)

__all__ = [
    "DiffusionSchedule",
    "SinusoidalTimeEmbedding",
    "ResidualBlock",
    "DenoisingMLP",
    "GaussianDiffusion",
    "train_diffusion_model",
    "run_diffusion_benchmark",
    "compute_multivariate_mmd",
    "HistoricalContextEncoder",
    "TrajectoryDenoisingNet",
    "TimeSeriesDiffusionForecaster",
    "compute_crps",
    "compute_interval_coverage",
    "train_timeseries_diffusion",
    "run_timeseries_diffusion_benchmark",
]
