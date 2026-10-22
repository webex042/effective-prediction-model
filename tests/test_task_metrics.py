"""Task-specific metrics (src/evaluation/task_metrics.py) incl. undefined edge cases."""

import numpy as np
import pytest
from sklearn import metrics as skm

from src.evaluation import task_metrics as TM


class TestRegression:
    def test_matches_definitions(self):
        y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 2.0, 2.0, 5.0])
        m, notes = TM.regression_metrics(y, p)
        assert m["mae"] == pytest.approx(np.mean(np.abs(y - p)))
        assert m["rmse"] == pytest.approx(np.sqrt(np.mean((y - p) ** 2)))
        assert m["r2"] == pytest.approx(skm.r2_score(y, p))
        assert notes == []

    def test_r2_undefined_for_constant_target(self):
        m, notes = TM.regression_metrics([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
        assert m["r2"] is None and notes

    def test_nan_predictions_reported_not_hidden(self):
        m, notes = TM.regression_metrics([1.0, 2.0], [np.nan, 1.0])
        assert m["rmse"] is None and "NaN" in notes[0]


class TestClassification:
    def test_binary_positive_class(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        p = np.array([0, 1, 1, 1, 0, 0])
        proba = np.column_stack([1 - np.linspace(0, 1, 6), np.linspace(0, 1, 6)])
        m, _ = TM.classification_metrics(y, p, proba, classes=[0, 1])
        assert m["precision"] == pytest.approx(skm.precision_score(y, p))
        assert m["recall"] == pytest.approx(skm.recall_score(y, p))
        assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(y, proba[:, 1]))
        assert m["pr_auc"] == pytest.approx(skm.average_precision_score(y, proba[:, 1]))
        assert m["confusion_matrix"]["matrix"] == skm.confusion_matrix(y, p).tolist()

    def test_multiclass_uses_macro_average(self):
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        p = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2])
        proba = np.eye(3)[p] * 0.8 + 0.2 / 3
        m, _ = TM.classification_metrics(y, p, proba, classes=[0, 1, 2])
        assert m["f1"] == pytest.approx(skm.f1_score(y, p, average="macro"))
        assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        assert "macro" in m["averaging"]

    def test_auc_undefined_with_single_class(self):
        m, notes = TM.classification_metrics([1, 1, 1], [1, 0, 1], np.array([[0.2, 0.8]] * 3), classes=[0, 1])
        assert m["roc_auc"] is None and any("one class" in n for n in notes)

    def test_auc_not_computed_without_scores(self):
        m, notes = TM.classification_metrics([0, 1], [0, 1], None, classes=[0, 1])
        assert m["roc_auc"] is None and notes

    def test_group_fairness_gaps(self):
        y = np.array([1, 1, 0, 0] * 20)
        p = np.array([1, 1, 0, 0] * 10 + [0, 0, 0, 0] * 10)
        g = np.array(["a"] * 40 + ["b"] * 40)
        f = TM.group_fairness(y, p, g, min_group=10)
        assert f["groups"]["a"]["tpr"] == 1.0 and f["groups"]["b"]["tpr"] == 0.0
        assert f["equal_opportunity_diff"] == pytest.approx(1.0)
        assert f["demographic_parity_diff"] == pytest.approx(0.5)


class TestClustering:
    def test_internal_indices(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (30, 2))])
        lab = np.r_[np.zeros(30), np.ones(30)].astype(int)
        m, _ = TM.clustering_metrics(X, lab, true_labels=lab)
        assert m["silhouette"] > 0.9 and m["ari_external"] == pytest.approx(1.0)
        assert m["davies_bouldin"] == pytest.approx(skm.davies_bouldin_score(X, lab), abs=1e-6)

    def test_single_cluster_is_undefined_not_zero(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        m, notes = TM.clustering_metrics(X, np.zeros(20, int))
        assert m["silhouette"] is None and notes

    def test_noise_points_excluded(self):
        X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 5, [[100, 100]]])
        lab = np.r_[np.zeros(10), np.ones(10), [-1]].astype(int)
        m, notes = TM.clustering_metrics(X, lab)
        assert m["n_clusters"] == 2 and m["noise_fraction"] == pytest.approx(1 / 21)


class TestForecasting:
    def test_mase_and_smape(self):
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scale = TM.mase_scale(train, 1)
        assert scale == pytest.approx(1.0)
        m, _ = TM.forecasting_metrics(np.array([[6.0], [7.0]]), np.array([[6.5], [6.0]]), scale, naive_rmse=1.0)
        assert m["mae"] == pytest.approx(0.75) and m["mase"] == pytest.approx(0.75)
        assert m["smape"] == pytest.approx(TM.smape([6, 7], [6.5, 6]))
        assert m["skill_vs_naive"] == pytest.approx(1 - m["rmse"])

    def test_mase_undefined_for_flat_series(self):
        assert TM.mase_scale(np.ones(50), 1) is None

    def test_per_horizon_rmse(self):
        y = np.zeros((4, 3))
        p = np.tile([1.0, 2.0, 3.0], (4, 1))
        m, _ = TM.forecasting_metrics(y, p, 1.0)
        assert m["rmse_per_horizon"] == [1.0, 2.0, 3.0] and m["horizon"] == 3


class TestAnomaly:
    def test_detection_delay(self):
        y = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
        f = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
        d = TM.detection_delays(y, f)
        assert d == {"windows": 2, "detected": 1, "missed": 1, "mean_delay_steps": 2.0, "delays_steps": [2]}

    def test_metrics_with_labels(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
        m, _ = TM.anomaly_metrics(y, s, (s > 0.5).astype(int))
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["pr_auc"] == pytest.approx(1.0)

    def test_unlabelled_data_gives_diagnostics_only(self):
        m, notes = TM.anomaly_metrics(None, np.arange(10.0), np.zeros(10))
        assert "precision" not in m and notes
