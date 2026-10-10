"""
EPM Research Module — Stage 13
==============================
Reproducible research-experiment pipeline: configuration, a shared experiment
registry, documented metrics, real baseline-vs-generative comparison, and an
automated Markdown report. See ``src/research/pipeline.py`` for the H1 runner.
"""

from src.research.config import ResearchConfig
from src.research.registry import ExperimentRecord, ExperimentRegistry
from src.research import metrics
from src.research.pipeline import (
    PreparedData,
    LinearForecaster,
    prepare_data,
    train_baseline,
    train_diffusion,
    eval_point_model,
    eval_diffusion_stepsweep,
    run_experiment,
)
from src.research.report import generate_report

__all__ = [
    "ResearchConfig",
    "ExperimentRecord",
    "ExperimentRegistry",
    "metrics",
    "PreparedData",
    "LinearForecaster",
    "prepare_data",
    "train_baseline",
    "train_diffusion",
    "eval_point_model",
    "eval_diffusion_stepsweep",
    "run_experiment",
    "generate_report",
]
