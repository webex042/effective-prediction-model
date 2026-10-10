"""
EPM Stage 0 — Demo Script
==========================
Demonstrates all core infrastructure components working together.
"""

import numpy as np

print("=" * 60)
print("EPM - Stage 0 Demo")
print("=" * 60)

# 1. Config System
from src.config import load_config

config = load_config()
print(f"\n[CONFIG]")
print(f"  Project:  {config.project.name} v{config.project.version}")
print(f"  Seed:     {config.reproducibility.seed}")
print(f"  LR:       {config.training.learning_rate}")
print(f"  Epochs:   {config.training.epochs}")
print(f"  Batch:    {config.training.batch_size}")

# 2. Reproducibility
from src.reproducibility import set_seed, get_environment_info

set_seed(config.reproducibility.seed)
env = get_environment_info()
print(f"\n[ENVIRONMENT]")
print(f"  Python:   {env['python_version'].split()[0]}")
print(f"  NumPy:    {env['packages']['numpy']}")
print(f"  OS:       {env['os']}")

# 3. Metrics (implemented from scratch)
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    Timer,
)

y_true_cls = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred_cls = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)
print(f"\n[CLASSIFICATION METRICS]")
for k, v in cls_metrics.items():
    print(f"  {k:>10s}: {v:.4f}")

y_true_reg = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8, 4.0])
reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)
print(f"\n[REGRESSION METRICS]")
for k, v in reg_metrics.items():
    print(f"  {k:>10s}: {v:.4f}")

# 4. Experiment Tracking
from src.experiment import Experiment

exp = Experiment(
    "stage0_demo",
    config=config,
    save_dir="experiments",
    tags=["demo", "stage0"],
)
exp.log_params({"model": "dummy", "features": 5})

# Simulate training loop
for step, (loss, acc) in enumerate(
    [(0.9, 0.5), (0.6, 0.7), (0.3, 0.85), (0.15, 0.92)]
):
    exp.log_metrics({"loss": loss, "accuracy": acc}, step=step)

print(f"\n[EXPERIMENT]")
print(f"  Name:          {exp.name}")
print(f"  Tags:          {exp.tags}")
print(f"  Final metrics: {exp.get_final_metrics()}")
print(f"  Best loss:     {exp.get_best_metric('loss', mode='min')}")
print(f"  Best accuracy: {exp.get_best_metric('accuracy', mode='max')}")

save_path = exp.end()
print(f"  Saved to:      {save_path}")

# 5. Timer
print(f"\n[TIMER]")
with Timer("matrix_multiply_500x500") as t:
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)
    C = A @ B
print(f"  Result shape:  {C.shape}")

# 6. Seed reproducibility proof
print(f"\n[REPRODUCIBILITY PROOF]")
set_seed(42)
sample_a = np.random.rand(5).round(6).tolist()
set_seed(42)
sample_b = np.random.rand(5).round(6).tolist()
print(f"  Run 1: {sample_a}")
print(f"  Run 2: {sample_b}")
print(f"  Match: {sample_a == sample_b}")

print("\n" + "=" * 60)
print("Stage 0 fully operational!")
print("=" * 60)
