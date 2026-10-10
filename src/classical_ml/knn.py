"""
EPM K-Nearest Neighbors (KNN) from Scratch
==========================================

A pure NumPy implementation of K-Nearest Neighbors for classification and regression.

MATHEMATICAL FOUNDATION
-----------------------
1. Instance-Based Learning:
   KNN is a non-parametric, lazy learning algorithm:
     - Training: Simply stores the training dataset (O(1) time).
     - Inference: For each test query x*, computes distance to ALL training samples,
       sorts distances, and selects the k closest neighbors.

2. Distance Metrics:
   - Euclidean (L2):
     d(x, x') = √ ( Σ_{j=1}^D (x_j - x'_j)² )
   - Manhattan (L1):
     d(x, x') = Σ_{j=1}^D |x_j - x'_j|

3. Vectorized Pairwise Distance:
   For Euclidean distance between queries X_test (M, D) and database X_train (N, D):
     ||x - y||² = ||x||² + ||y||² - 2 (x · y)
   In matrix form:
     D² = sum(X_test², axis=1)[:, None] + sum(X_train², axis=1)[None, :] - 2 (X_test @ X_train^T)

4. Prediction Rule:
   - Classification: Mode (majority vote) of labels among k nearest neighbors.
     P(y = c | x*) = (1/k) * Σ_{i ∈ N_k(x*)} 𝟙(yᵢ = c)
   - Regression: Mean of target values among k nearest neighbors.
     ŷ* = (1/k) * Σ_{i ∈ N_k(x*)} yᵢ
   - Optional distance weighting: wᵢ = 1 / (d(x*, xᵢ) + ε)

5. Computational Complexity:
   - Training Time: O(1)
   - Training Memory: O(N * D)
   - Inference Time: O(M * N * D + M * N * log(k))
   - Curse of Dimensionality: In high dimensions (large D), all pairwise distances
     concentrate around a similar value, degrading discriminative power.
"""

from __future__ import annotations

from typing import Literal, Optional
import numpy as np
from numpy.typing import ArrayLike

from src.evaluation.metrics import accuracy, r2_score


def pairwise_distances(
    X1: np.ndarray,
    X2: np.ndarray,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
) -> np.ndarray:
    """
    Vectorized calculation of pairwise distances between X1 (M, D) and X2 (N, D).
    
    Returns distance matrix of shape (M, N).
    """
    if metric == "euclidean":
        # Using expanded square formula for vectorized speed: ||a - b||² = ||a||² + ||b||² - 2 a·b
        # Clipping at 0 prevents small negative values due to floating-point imprecision.
        x1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)  # (M, 1)
        x2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T  # (1, N)
        cross = X1 @ X2.T  # (M, N)
        dist_sq = np.maximum(x1_sq + x2_sq - 2.0 * cross, 0.0)
        return np.sqrt(dist_sq)
    elif metric == "manhattan":
        # Using broadcasting: |X1[:, None, :] - X2[None, :, :]|
        # For memory efficiency with large N, process in chunks if needed
        return np.sum(np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose 'euclidean' or 'manhattan'.")


class KNNClassifierScratch:
    """
    K-Nearest Neighbors Classifier from scratch in NumPy.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric.
    weights : {"uniform", "distance"}, default="uniform"
        Weight function used in prediction.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNNClassifierScratch:
        """Store training vectors and class labels."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y).ravel()
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        self.classes_ = np.unique(self.y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return probability estimates for the test data X."""
        if not self.is_fitted or self.X_train is None or self.y_train is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_test = np.asarray(X, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        distances = pairwise_distances(X_test, self.X_train, metric=self.metric)
        k = min(self.n_neighbors, self.X_train.shape[0])

        # Indices of k nearest neighbors for each query sample
        k_indices = np.argsort(distances, axis=1)[:, :k]

        n_queries = len(X_test)
        n_classes = len(self.classes_)
        probs = np.zeros((n_queries, n_classes), dtype=np.float64)

        for i in range(n_queries):
            neighbor_indices = k_indices[i]
            neighbor_labels = self.y_train[neighbor_indices]

            if self.weights == "uniform":
                for c_idx, cls in enumerate(self.classes_):
                    probs[i, c_idx] = np.mean(neighbor_labels == cls)
            else:
                neighbor_dists = distances[i, neighbor_indices]
                inv_dists = 1.0 / (neighbor_dists + 1e-8)
                total_w = np.sum(inv_dists)
                for c_idx, cls in enumerate(self.classes_):
                    mask = (neighbor_labels == cls)
                    probs[i, c_idx] = np.sum(inv_dists[mask]) / total_w

        return probs

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class labels for the provided data."""
        probs = self.predict_proba(X)
        assert self.classes_ is not None
        best_class_indices = np.argmax(probs, axis=1)
        return self.classes_[best_class_indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return accuracy score on test data."""
        y_pred = self.predict(X)
        return accuracy(y, y_pred)


class KNNRegressorScratch:
    """
    K-Nearest Neighbors Regressor from scratch in NumPy.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric.
    weights : {"uniform", "distance"}, default="uniform"
        Weight function used in prediction.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNNRegressorScratch:
        """Store training data."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64).ravel()
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        self.is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous target values for test data."""
        if not self.is_fitted or self.X_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_test = np.asarray(X, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        distances = pairwise_distances(X_test, self.X_train, metric=self.metric)
        k = min(self.n_neighbors, self.X_train.shape[0])
        k_indices = np.argsort(distances, axis=1)[:, :k]

        n_queries = len(X_test)
        predictions = np.zeros(n_queries, dtype=np.float64)

        for i in range(n_queries):
            neighbor_indices = k_indices[i]
            neighbor_values = self.y_train[neighbor_indices]

            if self.weights == "uniform":
                predictions[i] = np.mean(neighbor_values)
            else:
                neighbor_dists = distances[i, neighbor_indices]
                inv_dists = 1.0 / (neighbor_dists + 1e-8)
                predictions[i] = np.sum(inv_dists * neighbor_values) / np.sum(inv_dists)

        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R² score on test data."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
