"""
Tests for the EPM Evaluation Metrics
=====================================

These tests verify that:
1. Classification metrics compute correctly
2. Regression metrics compute correctly
3. Edge cases are handled (all correct, all wrong, empty, etc.)
4. Timer utilities work
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    mae,
    mse,
    rmse,
    compute_classification_metrics,
    compute_regression_metrics,
    Timer,
)


# =============================================================================
# Classification Metrics
# =============================================================================

class TestAccuracy:
    def test_perfect_predictions(self):
        assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0

    def test_all_wrong(self):
        assert accuracy([1, 1, 1], [0, 0, 0]) == 0.0

    def test_partial(self):
        assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75

    def test_works_with_numpy_arrays(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        assert accuracy(y_true, y_pred) == 0.75


class TestPrecision:
    def test_perfect_precision(self):
        # Predicted 2 positives, both correct
        assert precision([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_zero_precision(self):
        # Predicted 2 positives, both wrong
        assert precision([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0

    def test_no_positive_predictions(self):
        # No positives predicted → precision is 0 (no TP+FP)
        assert precision([1, 1, 1], [0, 0, 0]) == 0.0

    def test_mixed(self):
        # TP=1, FP=1 → precision = 0.5
        assert precision([1, 0, 0], [1, 1, 0]) == 0.5


class TestRecall:
    def test_perfect_recall(self):
        assert recall([1, 1, 0], [1, 1, 0]) == 1.0

    def test_zero_recall(self):
        # All positives missed
        assert recall([1, 1, 1], [0, 0, 0]) == 0.0

    def test_mixed(self):
        # TP=1, FN=1 → recall = 0.5
        assert recall([1, 1, 0], [1, 0, 0]) == 0.5


class TestF1Score:
    def test_perfect_f1(self):
        assert f1_score([1, 0, 1], [1, 0, 1]) == 1.0

    def test_zero_f1(self):
        assert f1_score([1, 1, 1], [0, 0, 0]) == 0.0

    def test_harmonic_mean_property(self):
        """F1 should be the harmonic mean of precision and recall."""
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0
        assert abs(f1_score(y_true, y_pred) - expected) < 1e-10


class TestClassificationMetrics:
    def test_returns_all_metrics(self):
        result = compute_classification_metrics([1, 0, 1], [1, 0, 1])
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result


# =============================================================================
# Regression Metrics
# =============================================================================

class TestMAE:
    def test_perfect(self):
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self):
        # |1-2| + |2-3| + |3-4| = 1+1+1 = 3, mean = 1.0
        assert mae([1, 2, 3], [2, 3, 4]) == 1.0

    def test_symmetric(self):
        """MAE should be symmetric: mae(a,b) == mae(b,a)."""
        a = [1, 2, 3]
        b = [1.5, 2.5, 3.5]
        assert abs(mae(a, b) - mae(b, a)) < 1e-10


class TestMSE:
    def test_perfect(self):
        assert mse([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_known_value(self):
        # (1-2)^2 + (2-3)^2 = 1+1 = 2, mean = 1.0
        assert mse([1, 2], [2, 3]) == 1.0

    def test_penalizes_large_errors(self):
        """MSE should penalize large errors more than MAE does."""
        y_true = [0, 0, 0]
        y_pred_small = [1, 1, 1]   # errors: [1, 1, 1]
        y_pred_large = [0, 0, 3]   # errors: [0, 0, 3]

        # Same MAE (1.0), but MSE differs
        assert abs(mae(y_true, y_pred_small) - mae(y_true, y_pred_large)) < 1e-10
        assert mse(y_true, y_pred_large) > mse(y_true, y_pred_small)


class TestRMSE:
    def test_perfect(self):
        assert rmse([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_is_sqrt_of_mse(self):
        y_true = [1, 2, 3, 4]
        y_pred = [1.1, 2.2, 2.8, 4.5]
        assert abs(rmse(y_true, y_pred) - np.sqrt(mse(y_true, y_pred))) < 1e-10


class TestRegressionMetrics:
    def test_returns_all_metrics(self):
        result = compute_regression_metrics([1, 2, 3], [1.1, 2.1, 3.1])
        assert "mae" in result
        assert "mse" in result
        assert "rmse" in result


# =============================================================================
# Timer Tests
# =============================================================================

class TestTimer:
    def test_timer_measures_time(self):
        """Timer should record elapsed time > 0 for non-trivial work."""
        import time as time_mod
        with Timer("test") as t:
            time_mod.sleep(0.05)
        assert t.elapsed >= 0.04  # Allow small timing imprecision
