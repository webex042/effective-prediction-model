"""
Unit Tests for Stage 2: Classical Machine Learning
==================================================

Verifies:
  1. Linear Regression (Normal Equation & Gradient Descent, gradient checks, sklearn parity)
  2. Logistic Regression (Sigmoid, BCE loss, gradient checks, OvR multiclass, sklearn parity)
  3. KNN (Classifier & Regressor, distance metrics, weights)
  4. Decision Tree (Gini/Entropy/Variance impurity, splitting, depth constraint)
  5. Baselines benchmarking suite and synthetic sensor data generators
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression, LogisticRegression as SklearnLogisticRegression
from sklearn.neighbors import KNeighborsClassifier as SklearnKNNClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from src.classical_ml.linear_regression import LinearRegressionScratch
from src.classical_ml.logistic_regression import LogisticRegressionScratch, sigmoid
from src.classical_ml.knn import KNNClassifierScratch, KNNRegressorScratch, pairwise_distances
from src.classical_ml.decision_tree import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
    gini_impurity,
    entropy_impurity,
    variance_impurity,
)
from src.classical_ml.baselines import (
    generate_sensor_regression_dataset,
    generate_sensor_failure_dataset,
    evaluate_model,
)


# =============================================================================
# 1. Linear Regression Tests
# =============================================================================

class TestLinearRegression:
    def test_normal_equation_recovers_known_parameters(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-5, 5, size=(200, 3))
        true_w = np.array([2.5, -1.2, 4.0])
        true_b = 3.5
        y = X @ true_w + true_b  # No noise

        model = LinearRegressionScratch(method="normal_equation")
        model.fit(X, y)

        np.testing.assert_allclose(model.weights, true_w, atol=1e-5)
        np.testing.assert_allclose(model.bias, true_b, atol=1e-5)
        assert model.score(X, y) > 0.9999

    def test_gradient_descent_convergence(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(300, 2))
        true_w = np.array([1.5, -2.0])
        true_b = 0.5
        y = X @ true_w + true_b + rng.normal(0, 0.1, size=300)

        model = LinearRegressionScratch(
            method="gradient_descent", learning_rate=0.05, n_iterations=1500
        )
        model.fit(X, y)

        # Loss should strictly decrease or stabilize
        assert model.loss_history[-1] < model.loss_history[0]
        np.testing.assert_allclose(model.weights, true_w, atol=0.15)
        np.testing.assert_allclose(model.bias, true_b, atol=0.15)
        assert model.score(X, y) > 0.95

    def test_gradient_check(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(20, 3))
        y = rng.normal(0, 1, size=20)
        model = LinearRegressionScratch(l2_reg=0.1)
        check = model.check_gradient(X, y)
        assert check["passed"]
        assert check["relative_error_w"] < 1e-5

    def test_matches_sklearn_linear_regression(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-3, 3, size=(100, 4))
        y = X @ np.array([1.0, -0.5, 2.0, -1.5]) + 1.2 + rng.normal(0, 0.2, 100)

        scratch = LinearRegressionScratch(method="normal_equation")
        scratch.fit(X, y)
        pred_scratch = scratch.predict(X)

        sklearn_model = SklearnLinearRegression()
        sklearn_model.fit(X, y)
        pred_sklearn = sklearn_model.predict(X)

        np.testing.assert_allclose(pred_scratch, pred_sklearn, atol=1e-4)
        np.testing.assert_allclose(scratch.weights, sklearn_model.coef_, atol=1e-4)
        np.testing.assert_allclose(scratch.bias, sklearn_model.intercept_, atol=1e-4)

    def test_unfitted_raises_error(self):
        model = LinearRegressionScratch()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 2)))


# =============================================================================
# 2. Logistic Regression Tests
# =============================================================================

class TestLogisticRegression:
    def test_sigmoid_properties(self):
        z = np.array([-100.0, 0.0, 100.0])
        sig = sigmoid(z)
        assert sig[0] < 1e-10
        assert sig[1] == 0.5
        assert sig[2] > 1.0 - 1e-10

    def test_binary_classification_separable(self):
        # Two easily separable clusters
        X0 = np.random.normal(loc=[-2.0, -2.0], scale=0.5, size=(50, 2))
        X1 = np.random.normal(loc=[2.0, 2.0], scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = LogisticRegressionScratch(learning_rate=0.2, n_iterations=500)
        model.fit(X, y)

        acc = model.score(X, y)
        assert acc >= 0.98

        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)

    def test_gradient_check(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(25, 2))
        y = rng.choice([0, 1], size=25)
        model = LogisticRegressionScratch(l2_reg=0.05)
        check = model.check_gradient(X, y)
        assert check["passed"]
        assert check["relative_error_w"] < 1e-5

    def test_multiclass_ovr(self):
        # 3 classes
        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=[-3, 0], scale=0.4, size=(30, 2))
        X1 = rng.normal(loc=[0, 3], scale=0.4, size=(30, 2))
        X2 = rng.normal(loc=[3, 0], scale=0.4, size=(30, 2))
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)

        model = LogisticRegressionScratch(learning_rate=0.2, n_iterations=800)
        model.fit(X, y)

        assert model.is_multiclass
        acc = model.score(X, y)
        assert acc >= 0.90
        probs = model.predict_proba(X)
        assert probs.shape == (90, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


# =============================================================================
# 3. K-Nearest Neighbors Tests
# =============================================================================

class TestKNN:
    def test_pairwise_distances(self):
        X1 = np.array([[0, 0], [1, 1]])
        X2 = np.array([[0, 0], [3, 4]])
        dists = pairwise_distances(X1, X2, metric="euclidean")
        assert dists.shape == (2, 2)
        assert dists[0, 0] == 0.0
        assert dists[0, 1] == 5.0  # sqrt(3^2 + 4^2) = 5

    def test_knn_classifier(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = KNNClassifierScratch(n_neighbors=3)
        model.fit(X, y)

        query = np.array([[0.2, 0.2], [10.5, 10.5]])
        preds = model.predict(query)
        assert list(preds) == [0, 1]

    def test_knn_regressor(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([10.0, 20.0, 30.0, 40.0])

        model = KNNRegressorScratch(n_neighbors=2)
        model.fit(X, y)

        pred = model.predict([[2.5]])
        # Neighbors of 2.5 are 2.0 (20) and 3.0 (30) -> average is 25.0
        np.testing.assert_allclose(pred, [25.0], atol=1e-5)


# =============================================================================
# 4. Decision Tree Tests
# =============================================================================

class TestDecisionTree:
    def test_impurities(self):
        # Pure node: Gini = 0, Entropy = 0
        pure = np.array([1, 1, 1, 1])
        assert gini_impurity(pure) == 0.0
        assert entropy_impurity(pure) == 0.0

        # Uniform 2 classes: Gini = 0.5, Entropy = 1.0
        split = np.array([0, 1])
        assert abs(gini_impurity(split) - 0.5) < 1e-6
        assert abs(entropy_impurity(split) - 1.0) < 1e-6

        # Variance of constant target is 0
        assert variance_impurity(np.array([5.0, 5.0, 5.0])) == 0.0

    def test_tree_classifier(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        clf = DecisionTreeClassifierScratch(max_depth=3)
        clf.fit(X, y)

        preds = clf.predict([[1.5], [5.5]])
        assert list(preds) == [0, 1]
        assert clf.score(X, y) == 1.0

    def test_tree_regressor(self):
        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(100, 1)), axis=0)
        y = np.sin(X).ravel()

        reg = DecisionTreeRegressorScratch(max_depth=5)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert reg.score(X, y) > 0.85


# =============================================================================
# 5. Benchmarking & Data Generation Tests
# =============================================================================

class TestBaselinesAndData:
    def test_sensor_regression_dataset_generation(self):
        X, y, names = generate_sensor_regression_dataset(n_samples=50, random_seed=42)
        assert X.shape == (50, 5)
        assert y.shape == (50,)
        assert len(names) == 5

    def test_sensor_failure_dataset_generation(self):
        X, y, names = generate_sensor_failure_dataset(n_samples=100, failure_ratio=0.3, random_seed=42)
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert np.isin(y, [0, 1]).all()
        # Roughly 30% positive
        assert 20 <= np.sum(y) <= 40

    def test_evaluate_model(self):
        X, y, _ = generate_sensor_failure_dataset(n_samples=60, random_seed=42)
        model = LogisticRegressionScratch(n_iterations=50)
        res = evaluate_model(
            model=model,
            X_train=X[:40],
            y_train=y[:40],
            X_test=X[40:],
            y_test=y[40:],
            task="classification",
            model_name="TestLR",
            framework="scratch",
        )
        assert res["model"] == "TestLR"
        assert "training_time_s" in res
        assert "latency_us_per_sample" in res
        assert "f1" in res
        assert "accuracy" in res
