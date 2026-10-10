"""
EPM Decision Tree from Scratch
==============================

A pure NumPy implementation of CART (Classification and Regression Trees).

MATHEMATICAL FOUNDATION
-----------------------
1. Recursive Binary Splitting:
   A decision tree recursively partitions the feature space into axis-aligned
   hyper-rectangles R_m. Within each region R_m, the model predicts a constant value:
     - Classification: Mode class or class distribution
     - Regression: Mean of target values in R_m

2. Impurity Measures for Classification:
   Given a node with class proportions p = [p₁, p₂, ..., p_C]:
     - Gini Impurity:
       G(p) = 1 - Σ_{c=1}^C p_c²
       Measures probability that a randomly chosen element is incorrectly labeled.
     - Cross-Entropy:
       H(p) = - Σ_{c=1}^C p_c log₂(p_c + ε)

3. Impurity Measure for Regression:
   Variance (Mean Squared Error from node mean):
     MSE(y) = (1/N) * Σ_{i=1}^N (yᵢ - ȳ)²

4. Information Gain (Splitting Criterion):
   Given parent node S partitioned into left child S_L and right child S_R:
     Gain(S, j, t) = I(S) - [ (|S_L| / |S|) * I(S_L) + (|S_R| / |S|) * I(S_R) ]
   
   The algorithm greedily searches over all features j and candidate thresholds t
   to maximize Gain.

5. Complexity:
   - Training Time: O(D * N * log(N) * depth)
   - Inference Time: O(depth) - typically O(log N) for balanced trees
   - Memory: O(nodes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Union, Any
import numpy as np
from numpy.typing import ArrayLike

from src.evaluation.metrics import accuracy, r2_score


@dataclass
class Node:
    """A single node in a decision tree."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[Node] = None
    right: Optional[Node] = None
    value: Any = None
    is_leaf: bool = False
    impurity: float = 0.0
    n_samples: int = 0


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity for classification labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs ** 2))


def entropy_impurity(y: np.ndarray) -> float:
    """Compute Shannon entropy for classification labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / len(y)
    eps = 1e-15
    return float(max(0.0, -np.sum(probs * np.log2(probs + eps))))


def variance_impurity(y: np.ndarray) -> float:
    """Compute variance (MSE) for regression targets."""
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier implemented from scratch in NumPy.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {"gini", "entropy"}, default="gini"
        Function to measure split quality.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        criterion: Literal["gini", "entropy"] = "gini",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

        self.root: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            return gini_impurity(y)
        return entropy_impurity(y)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[Optional[int], Optional[float], float]:
        """Find the feature and threshold that maximize information gain."""
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feat in range(n_features):
            values = X[:, feat]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            # Candidate thresholds as midpoints between consecutive sorted unique values
            # To maintain efficiency on larger datasets, subsample if too many unique values
            if len(unique_vals) > 50:
                percentiles = np.linspace(0, 100, 50)
                thresholds = np.percentile(unique_vals, percentiles)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                if n_l == 0 or n_r == 0:
                    continue

                impurity_l = self._impurity(y[left_mask])
                impurity_r = self._impurity(y[right_mask])

                child_impurity = (n_l / n_samples) * impurity_l + (n_r / n_samples) * impurity_r
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = float(thresh)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        node_impurity = self._impurity(y)

        # Stopping conditions: pure node, max depth, or min samples
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(classes) == 1
        ):
            return Node(
                value=majority_class,
                is_leaf=True,
                impurity=node_impurity,
                n_samples=n_samples,
            )

        best_feat, best_thresh, best_gain = self._best_split(X, y)

        if best_feat is None or best_gain <= 0.0:
            return Node(
                value=majority_class,
                is_leaf=True,
                impurity=node_impurity,
                n_samples=n_samples,
            )

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=majority_class,
            is_leaf=False,
            impurity=node_impurity,
            n_samples=n_samples,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> DecisionTreeClassifierScratch:
        """Fit decision tree classifier."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.classes_ = np.unique(y_arr)
        self.root = self._build_tree(X_arr, y_arr, depth=0)
        self.is_fitted = True
        return self

    def _predict_sample(self, node: Node, x: np.ndarray) -> Any:
        if node.is_leaf or node.feature_idx is None or node.threshold is None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            assert node.left is not None
            return self._predict_sample(node.left, x)
        else:
            assert node.right is not None
            return self._predict_sample(node.right, x)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels for samples in X."""
        if not self.is_fitted or self.root is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        preds = [self._predict_sample(self.root, row) for row in X_arr]
        return np.array(preds)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return mean accuracy on test set."""
        y_pred = self.predict(X)
        return accuracy(y, y_pred)


class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor implemented from scratch in NumPy.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of tree.
    min_samples_split : int, default=2
        Minimum samples to split node.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.root: Optional[Node] = None
        self.is_fitted: bool = False

    def _impurity(self, y: np.ndarray) -> float:
        return variance_impurity(y)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feat in range(n_features):
            values = X[:, feat]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) > 50:
                percentiles = np.linspace(0, 100, 50)
                thresholds = np.percentile(unique_vals, percentiles)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                if n_l == 0 or n_r == 0:
                    continue

                impurity_l = self._impurity(y[left_mask])
                impurity_r = self._impurity(y[right_mask])

                child_impurity = (n_l / n_samples) * impurity_l + (n_r / n_samples) * impurity_r
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = float(thresh)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples = len(y)
        mean_val = float(np.mean(y))
        node_impurity = self._impurity(y)

        if depth >= self.max_depth or n_samples < self.min_samples_split or node_impurity == 0.0:
            return Node(
                value=mean_val,
                is_leaf=True,
                impurity=node_impurity,
                n_samples=n_samples,
            )

        best_feat, best_thresh, best_gain = self._best_split(X, y)

        if best_feat is None or best_gain <= 0.0:
            return Node(
                value=mean_val,
                is_leaf=True,
                impurity=node_impurity,
                n_samples=n_samples,
            )

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=mean_val,
            is_leaf=False,
            impurity=node_impurity,
            n_samples=n_samples,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> DecisionTreeRegressorScratch:
        """Fit decision tree regressor."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.root = self._build_tree(X_arr, y_arr, depth=0)
        self.is_fitted = True
        return self

    def _predict_sample(self, node: Node, x: np.ndarray) -> float:
        if node.is_leaf or node.feature_idx is None or node.threshold is None:
            return float(node.value)
        if x[node.feature_idx] <= node.threshold:
            assert node.left is not None
            return self._predict_sample(node.left, x)
        else:
            assert node.right is not None
            return self._predict_sample(node.right, x)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous targets for samples in X."""
        if not self.is_fitted or self.root is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        preds = [self._predict_sample(self.root, row) for row in X_arr]
        return np.array(preds, dtype=np.float64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R² score on test set."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
