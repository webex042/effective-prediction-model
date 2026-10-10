"""
EPM Classical ML Baselines & Leaderboard
========================================

Comprehensive benchmarking suite comparing from-scratch NumPy implementations
against production-grade libraries (scikit-learn and XGBoost).

Tracks all research metrics:
  - Predictive: Accuracy, Precision, Recall, F1, AUROC, MAE, MSE, RMSE, R²
  - Computational: Training time, Inference latency (per sample & batch), Peak memory (MB), Model size
"""

from __future__ import annotations

import os
import sys
import time
import json
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union
import numpy as np
import pandas as pd

# Scikit-learn models
from sklearn.linear_model import LinearRegression as SklearnLinearRegression, LogisticRegression as SklearnLogisticRegression, Ridge as SklearnRidge
from sklearn.neighbors import KNeighborsClassifier as SklearnKNNClassifier, KNeighborsRegressor as SklearnKNNRegressor
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree, DecisionTreeRegressor as SklearnDecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# XGBoost
import xgboost as xgb

# EPM Scratch implementations
from src.classical_ml.linear_regression import LinearRegressionScratch
from src.classical_ml.logistic_regression import LogisticRegressionScratch
from src.classical_ml.knn import KNNClassifierScratch, KNNRegressorScratch
from src.classical_ml.decision_tree import DecisionTreeClassifierScratch, DecisionTreeRegressorScratch
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    roc_auc,
    r2_score,
)
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Synthetic Industrial Sensor Datasets (Controlled Research Setting)
# =============================================================================

def generate_sensor_regression_dataset(
    n_samples: int = 1000,
    noise_std: float = 0.5,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic industrial machine sensor data for temperature prediction.
    Features:
      1. Vibration (mm/s)
      2. Pressure (bar)
      3. Motor RPM
      4. Mechanical Load (%)
      5. Ambient Temperature (°C)
    
    Target:
      Operating Core Temperature (°C) with physical nonlinear relations.
    """
    rng = np.random.default_rng(random_seed)

    vibration = rng.uniform(0.5, 5.0, n_samples)
    pressure = rng.uniform(20.0, 80.0, n_samples)
    rpm = rng.uniform(1000.0, 3600.0, n_samples)
    load = rng.uniform(10.0, 100.0, n_samples)
    ambient = rng.uniform(15.0, 35.0, n_samples)

    # Physical interaction target
    target = (
        ambient
        + 0.15 * load
        + 0.008 * rpm
        + 1.8 * vibration
        + 0.05 * pressure
        + 0.002 * (load * vibration)
        + rng.normal(0, noise_std, n_samples)
    )

    X = np.column_stack([vibration, pressure, rpm, load, ambient])
    feature_names = ["Vibration", "Pressure", "RPM", "Load", "Ambient_Temp"]
    return X, target, feature_names


def generate_sensor_failure_dataset(
    n_samples: int = 1000,
    failure_ratio: float = 0.25,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic industrial machine sensor data for binary equipment failure prediction.
    Target:
      0 = Normal operating condition
      1 = Impending component failure (abnormal temperature/vibration/pressure)
    """
    rng = np.random.default_rng(random_seed)

    vibration = rng.normal(2.5, 0.8, n_samples)
    pressure = rng.normal(50.0, 10.0, n_samples)
    rpm = rng.normal(2500.0, 400.0, n_samples)
    load = rng.normal(55.0, 18.0, n_samples)
    ambient = rng.normal(25.0, 5.0, n_samples)

    # Risk score based on high vibration, high pressure, high load
    risk_score = (
        0.8 * (vibration - 2.5) / 0.8
        + 0.6 * (pressure - 50.0) / 10.0
        + 0.7 * (load - 55.0) / 18.0
        + 0.3 * (rpm - 2500.0) / 400.0
        + rng.normal(0, 0.5, n_samples)
    )

    threshold = np.percentile(risk_score, (1.0 - failure_ratio) * 100.0)
    y = (risk_score >= threshold).astype(int)

    X = np.column_stack([vibration, pressure, rpm, load, ambient])
    feature_names = ["Vibration", "Pressure", "RPM", "Load", "Ambient_Temp"]
    return X, y, feature_names


# =============================================================================
# Benchmarking Engine
# =============================================================================

def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: Literal["classification", "regression"],
    model_name: str,
    framework: Literal["scratch", "scikit-learn", "xgboost"],
) -> dict[str, Any]:
    """
    Train and rigorously evaluate a model on both predictive and computational metrics.
    """
    n_test = len(X_test)

    # 1. Measure Training Time and Memory
    tracemalloc.start()
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    # 2. Measure Inference Time (Latency)
    # Warm-up run
    _ = model.predict(X_test[:min(10, n_test)])

    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_time = time.perf_counter() - start_infer
    latency_per_sample_us = (infer_time / n_test) * 1_000_000.0  # in microseconds

    # 3. Model size estimate
    import pickle
    try:
        model_size_kb = len(pickle.dumps(model)) / 1024.0
    except Exception:
        model_size_kb = 0.0

    result: dict[str, Any] = {
        "model": model_name,
        "framework": framework,
        "task": task,
        "training_time_s": round(train_time, 5),
        "inference_time_s": round(infer_time, 5),
        "latency_us_per_sample": round(latency_per_sample_us, 2),
        "peak_memory_mb": round(peak_mem_mb, 4),
        "model_size_kb": round(model_size_kb, 2),
    }

    # 4. Task-Specific Predictive Metrics
    if task == "classification":
        # Check if predict_proba is available
        y_score = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    y_score = proba[:, 1]
            except Exception:
                pass
        elif hasattr(model, "decision_function"):
            try:
                y_score = model.decision_function(X_test)
            except Exception:
                pass

        class_metrics = compute_classification_metrics(y_test, y_pred, y_score=y_score)
        result.update({
            "accuracy": round(class_metrics["accuracy"], 4),
            "precision": round(class_metrics["precision"], 4),
            "recall": round(class_metrics["recall"], 4),
            "f1": round(class_metrics["f1"], 4),
            "auroc": round(class_metrics.get("auroc", 0.0), 4) if y_score is not None else None,
        })
    else:  # regression
        reg_metrics = compute_regression_metrics(y_test, y_pred)
        result.update({
            "mae": round(reg_metrics["mae"], 4),
            "mse": round(reg_metrics["mse"], 4),
            "rmse": round(reg_metrics["rmse"], 4),
            "r2": round(reg_metrics["r2"], 4),
        })

    return result


def get_classical_regression_models() -> list[tuple[str, str, Any]]:
    """Return list of (model_name, framework, model_instance) for regression."""
    return [
        (
            "Linear Regression (Normal Eq)",
            "scratch",
            LinearRegressionScratch(method="normal_equation"),
        ),
        (
            "Linear Regression (GD)",
            "scratch",
            LinearRegressionScratch(method="gradient_descent", learning_rate=0.01, n_iterations=1000),
        ),
        (
            "Linear Regression (OLS)",
            "scikit-learn",
            SklearnLinearRegression(),
        ),
        (
            "Ridge Regression (L2)",
            "scikit-learn",
            SklearnRidge(alpha=1.0),
        ),
        (
            "KNN Regressor (k=5)",
            "scratch",
            KNNRegressorScratch(n_neighbors=5),
        ),
        (
            "KNN Regressor (k=5)",
            "scikit-learn",
            SklearnKNNRegressor(n_neighbors=5),
        ),
        (
            "Decision Tree (Depth=5)",
            "scratch",
            DecisionTreeRegressorScratch(max_depth=5),
        ),
        (
            "Decision Tree (Depth=5)",
            "scikit-learn",
            SklearnDecisionTreeRegressor(max_depth=5, random_state=42),
        ),
        (
            "Random Forest (100 trees)",
            "scikit-learn",
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        ),
        (
            "Support Vector Regressor (SVR)",
            "scikit-learn",
            SVR(kernel="rbf", C=10.0),
        ),
        (
            "Gradient Boosting Regressor",
            "scikit-learn",
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        ),
        (
            "XGBoost Regressor",
            "xgboost",
            xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, verbosity=0),
        ),
    ]


def get_classical_classification_models() -> list[tuple[str, str, Any]]:
    """Return list of (model_name, framework, model_instance) for classification."""
    return [
        (
            "Logistic Regression",
            "scratch",
            LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000),
        ),
        (
            "Logistic Regression",
            "scikit-learn",
            SklearnLogisticRegression(max_iter=1000, random_state=42),
        ),
        (
            "KNN Classifier (k=5)",
            "scratch",
            KNNClassifierScratch(n_neighbors=5),
        ),
        (
            "KNN Classifier (k=5)",
            "scikit-learn",
            SklearnKNNClassifier(n_neighbors=5),
        ),
        (
            "Decision Tree (Depth=5)",
            "scratch",
            DecisionTreeClassifierScratch(max_depth=5),
        ),
        (
            "Decision Tree (Depth=5)",
            "scikit-learn",
            SklearnDecisionTree(max_depth=5, random_state=42),
        ),
        (
            "Random Forest (100 trees)",
            "scikit-learn",
            RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        ),
        (
            "Support Vector Classifier (SVC)",
            "scikit-learn",
            SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        ),
        (
            "Gradient Boosting Classifier",
            "scikit-learn",
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        ),
        (
            "XGBoost Classifier",
            "xgboost",
            xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric="logloss", verbosity=0),
        ),
    ]


def run_classical_ml_benchmarks(
    save_dir: str = "benchmarks",
    n_samples: int = 1200,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full classical ML benchmark suite on both regression and classification tasks.
    Saves leaderboard as CSV, Markdown, and JSON in benchmarks/.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Regression Dataset
    X_reg, y_reg, _ = generate_sensor_regression_dataset(n_samples=n_samples, random_seed=random_seed)
    split_idx = int(0.8 * len(X_reg))
    X_reg_train, X_reg_test = X_reg[:split_idx], X_reg[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

    # Scale features for fair comparison across linear/SVM/KNN models
    scaler_reg = StandardScaler()
    X_reg_train_s = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_s = scaler_reg.transform(X_reg_test)

    # 2. Prepare Classification Dataset
    X_cls, y_cls, _ = generate_sensor_failure_dataset(n_samples=n_samples, random_seed=random_seed)
    X_cls_train, X_cls_test = X_cls[:split_idx], X_cls[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]

    scaler_cls = StandardScaler()
    X_cls_train_s = scaler_cls.fit_transform(X_cls_train)
    X_cls_test_s = scaler_cls.transform(X_cls_test)

    # 3. Benchmark Regression Models
    regression_results = []
    for name, framework, model in get_classical_regression_models():
        res = evaluate_model(
            model=model,
            X_train=X_reg_train_s,
            y_train=y_reg_train,
            X_test=X_reg_test_s,
            y_test=y_reg_test,
            task="regression",
            model_name=name,
            framework=framework,
        )
        regression_results.append(res)

    df_reg = pd.DataFrame(regression_results)
    # Sort regression by lowest RMSE
    df_reg = df_reg.sort_values(by="rmse", ascending=True).reset_index(drop=True)

    # 4. Benchmark Classification Models
    classification_results = []
    for name, framework, model in get_classical_classification_models():
        res = evaluate_model(
            model=model,
            X_train=X_cls_train_s,
            y_train=y_cls_train,
            X_test=X_cls_test_s,
            y_test=y_cls_test,
            task="classification",
            model_name=name,
            framework=framework,
        )
        classification_results.append(res)

    df_cls = pd.DataFrame(classification_results)
    # Sort classification by highest F1 score
    df_cls = df_cls.sort_values(by="f1", ascending=False).reset_index(drop=True)

    # 5. Save Leaderboards
    df_reg.to_csv(save_path / "stage2_regression_leaderboard.csv", index=False)
    df_cls.to_csv(save_path / "stage2_classification_leaderboard.csv", index=False)

    with open(save_path / "stage2_regression_leaderboard.json", "w") as f:
        json.dump(df_reg.to_dict(orient="records"), f, indent=2)

    with open(save_path / "stage2_classification_leaderboard.json", "w") as f:
        json.dump(df_cls.to_dict(orient="records"), f, indent=2)

    def _to_md_table(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
        return "\n".join(lines)

    # Markdown Leaderboard
    md_content = f"""# EPM Stage 2: Classical Machine Learning Leaderboard

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Dataset: Industrial Sensor Suite ({n_samples} samples, 80/20 train-test split)

---

## 1. Regression Leaderboard (Predicting Core Temperature)

{_to_md_table(df_reg)}

---

## 2. Classification Leaderboard (Predicting Equipment Failure)

{_to_md_table(df_cls)}
"""
    with open(save_path / "stage2_leaderboard.md", "w") as f:
        f.write(md_content)

    return df_reg, df_cls
