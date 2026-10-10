"""
EPM Evaluation Metrics
======================

A unified set of evaluation metrics for classification, regression,
and computational benchmarking.

WHY IMPLEMENT OUR OWN?
    scikit-learn provides all these metrics, but:
    1. It's not installed yet (Stage 0 has minimal dependencies)
    2. Implementing metrics from scratch teaches the mathematics
    3. We need custom computational metrics (timing, memory)

    In later stages, we'll validate our implementations against sklearn.

METRICS IMPLEMENTED:
    Classification: accuracy, precision, recall, F1
    Regression: MAE, MSE, RMSE
    Computational: timing decorator, memory tracking

USAGE:
    from src.evaluation.metrics import accuracy, mae

    acc = accuracy(y_true, y_pred)
    error = mae(y_true, y_pred)
"""

from __future__ import annotations

import time
import functools
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import ArrayLike


# =============================================================================
# Classification Metrics
# =============================================================================

def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Classification accuracy: fraction of correct predictions.

    MATH:
        accuracy = (number of correct predictions) / (total predictions)
        accuracy = (1/N) * Σ 𝟙(ŷᵢ = yᵢ)

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Accuracy in [0, 1].

    Examples
    --------
    >>> accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    0.75
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary"
) -> float:
    """
    Precision: of all positive predictions, how many were correct?

    MATH:
        precision = TP / (TP + FP)

    WHY IT MATTERS:
        High precision means few false alarms. Critical when the cost
        of a false positive is high (e.g., "machine is failing" when it's not).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str
        "binary" for binary classification.

    Returns
    -------
    float
        Precision in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary"
) -> float:
    """
    Recall (sensitivity): of all actual positives, how many did we find?

    MATH:
        recall = TP / (TP + FN)

    WHY IT MATTERS:
        High recall means we miss few actual positives. Critical when the cost
        of missing a true positive is high (e.g., failing to predict a failure).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str
        "binary" for binary classification.

    Returns
    -------
    float
        Recall in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary"
) -> float:
    """
    F1 Score: harmonic mean of precision and recall.

    MATH:
        F1 = 2 * (precision * recall) / (precision + recall)

    WHY HARMONIC MEAN?
        The harmonic mean penalizes extreme imbalances. If precision=1.0
        but recall=0.01, arithmetic mean would give 0.505 (misleadingly high).
        Harmonic mean gives 0.0198 (correctly low).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str
        "binary" for binary classification.

    Returns
    -------
    float
        F1 score in [0, 1].
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """
    Area Under Receiver Operating Characteristic Curve (AUROC).

    MATH:
        Computed via Wilcoxon-Mann-Whitney U statistic:
        AUC = (sum(ranks of positive instances) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels {0, 1}.
    y_score : array-like
        Predicted probabilities or continuous decision scores.

    Returns
    -------
    float
        AUROC in [0, 1].
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank-based calculation
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Handle ties
    unique_scores, counts = np.unique(y_score, return_counts=True)
    for val, count in zip(unique_scores, counts):
        if count > 1:
            tie_mask = (y_score == val)
            ranks[tie_mask] = np.mean(ranks[tie_mask])

    sum_ranks_pos = np.sum(ranks[pos_mask])
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_score: Optional[ArrayLike] = None
) -> dict[str, float]:
    """
    Compute all classification metrics at once.

    Returns
    -------
    dict
        Dictionary with keys: accuracy, precision, recall, f1, and optionally auroc.
    """
    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_score is not None:
        metrics["auroc"] = roc_auc(y_true, y_score)
    return metrics


# =============================================================================
# Regression Metrics
# =============================================================================

def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error.

    MATH:
        MAE = (1/N) * Σ |yᵢ - ŷᵢ|

    WHY MAE?
        Intuitive: "on average, predictions are off by X units."
        Robust to outliers (unlike MSE).

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean absolute error (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Squared Error.

    MATH:
        MSE = (1/N) * Σ (yᵢ - ŷᵢ)²

    WHY MSE?
        Penalizes large errors more heavily than MAE (quadratic penalty).
        Mathematically convenient: differentiable everywhere, and its
        gradient points toward the optimal solution.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean squared error (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error.

    MATH:
        RMSE = √MSE = √((1/N) * Σ (yᵢ - ŷᵢ)²)

    WHY RMSE?
        Same units as the target variable (unlike MSE which is in squared units).
        More interpretable: "predictions are off by ~X units on average."

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Root mean squared error (lower is better).
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of Determination (R² Score).

    MATH:
        R² = 1 - (SS_res / SS_tot) = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)

    Interpretation:
        1.0: Perfect prediction
        0.0: Predicts the baseline mean
        <0 : Worse than simply predicting the mean

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        R² score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_regression_metrics(
    y_true: ArrayLike, y_pred: ArrayLike
) -> dict[str, float]:
    """
    Compute all regression metrics at once.

    Returns
    -------
    dict
        Dictionary with keys: mae, mse, rmse, r2.
    """
    return {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# =============================================================================
# Computational Metrics
# =============================================================================

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    USAGE:
        @measure_time
        def train_model(X, y):
            ...

        result = train_model(X, y)
        # Prints: "train_model took 12.34s"

    The measured time is also stored as an attribute on the returned result
    (if the result is not None) and printed to stdout.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIME] {func.__name__} took {elapsed:.4f}s")
        # Store timing info for programmatic access
        wrapper.last_elapsed = elapsed
        return result

    wrapper.last_elapsed = 0.0
    return wrapper


class Timer:
    """
    Context manager for timing code blocks.

    USAGE:
        with Timer("training") as t:
            model.fit(X, y)
        print(f"Training took {t.elapsed:.2f}s")
    """

    def __init__(self, name: str = "block") -> None:
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start
        print(f"[TIME] {self.name} took {self.elapsed:.4f}s")
