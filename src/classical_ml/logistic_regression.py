"""
EPM Logistic Regression from Scratch
====================================

A pure NumPy implementation of Logistic Regression for binary and multiclass
classification (via One-vs-Rest), with full gradient derivation and numerical stability.

MATHEMATICAL FOUNDATION
-----------------------
1. Linear Score to Probability:
   z = Xw + b
   p = σ(z) = 1 / (1 + e^(-z))
   
   Properties of Sigmoid σ(z):
     - As z → +∞, σ(z) → 1
     - As z → -∞, σ(z) → 0
     - At z = 0,   σ(z) = 0.5 (Decision boundary)
     - Derivative: dσ/dz = σ(z) * (1 - σ(z))

2. Binary Cross-Entropy Loss (Log Loss):
   J(w, b) = - (1/N) * Σ_{i=1}^N [ yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ) ] + (λ / 2) * ||w||₂²

   Why Log Loss instead of MSE?
     - Maximum Likelihood: Minimizing log loss = Maximizing data likelihood under Bernoulli assumption
     - Convexity: With log loss and sigmoid, J(w, b) is strictly convex (guarantees global minimum)
     - MSE with sigmoid produces a non-convex surface plagued by plateau regions where gradients vanish.

3. Gradient Derivation via Chain Rule:
   ∂J/∂p = -(y/p - (1-y)/(1-p)) = (p - y) / (p(1-p))
   ∂p/∂z = p(1-p)
   ∂z/∂w = X
   
   By Chain Rule:
     ∂J/∂w = (∂J/∂p) * (∂p/∂z) * (∂z/∂w)
           = [ (p - y) / (p(1-p)) ] * [ p(1-p) ] * X
           = (1/N) * X^T (p - y) + λ w
     
     ∂J/∂b = (1/N) * Σ (p - y)
   
   Notice the beautiful symmetry: The gradient looks identical to Linear Regression,
   except p = σ(Xw + b) instead of ŷ = Xw + b.

4. Numerical Stability:
   - Sigmoid: For z < 0, e^(-z) can overflow. We compute:
       σ(z) = e^z / (1 + e^z) for z < 0
       σ(z) = 1 / (1 + e^(-z)) for z >= 0
   - Log Loss: Clip probabilities to [1e-15, 1 - 1e-15] to prevent log(0).
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
from numpy.typing import ArrayLike

from src.evaluation.metrics import accuracy, roc_auc


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    
    Prevents floating-point overflow for large negative z.
    """
    z = np.asarray(z, dtype=np.float64)
    # Clip extreme values to prevent underflow/overflow
    z_clipped = np.clip(z, -500.0, 500.0)
    return np.where(
        z_clipped >= 0,
        1.0 / (1.0 + np.exp(-z_clipped)),
        np.exp(z_clipped) / (1.0 + np.exp(z_clipped))
    )


class LogisticRegressionScratch:
    """
    Logistic Regression classifier implemented from scratch using NumPy.
    Supports binary classification and multiclass classification via One-vs-Rest (OvR).

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for gradient descent.
    n_iterations : int, default=1000
        Maximum number of optimization iterations.
    l2_reg : float, default=0.0
        L2 regularization strength (λ).
    fit_intercept : bool, default=True
        Whether to learn bias term b.
    tolerance : float, default=1e-6
        Tolerance for early stopping when loss improvement is negligible.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        l2_reg: float = 0.0,
        fit_intercept: bool = True,
        tolerance: float = 1e-6,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance

        self.weights: Optional[np.ndarray] = None
        self.bias: Union[float, np.ndarray] = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.loss_history: list[float] = []
        self.is_multiclass: bool = False
        self._models: list[LogisticRegressionScratch] = []
        self.is_fitted: bool = False

    def _compute_loss(self, y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
        """Compute binary cross-entropy loss with L2 penalty."""
        eps = 1e-15
        p_clipped = np.clip(p, eps, 1.0 - eps)
        n = len(y)
        bce = - (1.0 / n) * np.sum(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
        reg = 0.5 * self.l2_reg * np.sum(w ** 2) if self.l2_reg > 0 else 0.0
        return float(bce + reg)

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a single binary logistic regression model."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        prev_loss = float("inf")

        for iteration in range(self.n_iterations):
            # 1. Forward linear score: z = Xw + b
            z = X @ self.weights + self.bias
            # 2. Predicted probability: p = σ(z)
            p = sigmoid(z)

            # 3. Compute loss
            loss = self._compute_loss(y, p, self.weights)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

            # 4. Error: e = p - y
            error = p - y

            # 5. Gradients:
            dw = (1.0 / n_samples) * (X.T @ error) + self.l2_reg * self.weights
            db = (1.0 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            # 6. Parameter updates
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegressionScratch:
        """
        Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (N, D)
            Training vectors.
        y : array-like of shape (N,)
            Target class labels.

        Returns
        -------
        self : LogisticRegressionScratch
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).ravel()

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Training data must contain at least 2 distinct classes.")

        if n_classes == 2:
            self.is_multiclass = False
            # Map classes to 0 and 1
            y_binary = (y_arr == self.classes_[1]).astype(np.float64)
            self._fit_binary(X_arr, y_binary)
        else:
            # One-vs-Rest (OvR) multiclass strategy
            self.is_multiclass = True
            self._models = []
            for cls in self.classes_:
                y_binary = (y_arr == cls).astype(np.float64)
                sub_model = LogisticRegressionScratch(
                    learning_rate=self.learning_rate,
                    n_iterations=self.n_iterations,
                    l2_reg=self.l2_reg,
                    fit_intercept=self.fit_intercept,
                    tolerance=self.tolerance,
                )
                sub_model.fit(X_arr, y_binary)
                self._models.append(sub_model)

        self.is_fitted = True
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Probability estimates of classes.

        Parameters
        ----------
        X : array-like of shape (N, D)

        Returns
        -------
        p : np.ndarray of shape (N, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if not self.is_multiclass:
            assert self.weights is not None
            z = X_arr @ self.weights + self.bias
            p1 = sigmoid(z)
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        else:
            # Gather OvR probabilities and normalize
            probs = np.column_stack([
                sub_model.predict_proba(X_arr)[:, 1] for sub_model in self._models
            ])
            # Normalize so probabilities sum to 1 across classes
            row_sums = np.sum(probs, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return probs / row_sums

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (N, D)
        threshold : float, default=0.5
            Probability threshold for binary classification.

        Returns
        -------
        y_pred : np.ndarray of shape (N,)
        """
        probs = self.predict_proba(X)
        assert self.classes_ is not None

        if not self.is_multiclass:
            # Binary classification using threshold
            indices = (probs[:, 1] >= threshold).astype(int)
            return self.classes_[indices]
        else:
            indices = np.argmax(probs, axis=1)
            return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return accuracy(y, y_pred)

    def check_gradient(self, X: ArrayLike, y: ArrayLike, epsilon: float = 1e-5) -> dict[str, float]:
        """
        Perform numerical gradient checking to verify analytical gradients for binary classification.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape

        w = np.random.randn(n_features)
        b = np.random.randn()

        # Analytical gradient
        z = X_arr @ w + b
        p = sigmoid(z)
        error = p - y_arr
        analytical_dw = (1.0 / n_samples) * (X_arr.T @ error) + self.l2_reg * w
        analytical_db = (1.0 / n_samples) * np.sum(error)

        # Numerical gradient for w
        numerical_dw = np.zeros_like(w)
        for i in range(n_features):
            w_pos = w.copy()
            w_pos[i] += epsilon
            p_pos = sigmoid(X_arr @ w_pos + b)
            loss_pos = self._compute_loss(y_arr, p_pos, w_pos)

            w_neg = w.copy()
            w_neg[i] -= epsilon
            p_neg = sigmoid(X_arr @ w_neg + b)
            loss_neg = self._compute_loss(y_arr, p_neg, w_neg)

            numerical_dw[i] = (loss_pos - loss_neg) / (2.0 * epsilon)

        # Numerical gradient for b
        loss_pos_b = self._compute_loss(y_arr, sigmoid(X_arr @ w + (b + epsilon)), w)
        loss_neg_b = self._compute_loss(y_arr, sigmoid(X_arr @ w + (b - epsilon)), w)
        numerical_db = (loss_pos_b - loss_neg_b) / (2.0 * epsilon)

        err_w = float(np.linalg.norm(analytical_dw - numerical_dw) / (
            np.linalg.norm(analytical_dw) + np.linalg.norm(numerical_dw) + 1e-12
        ))
        err_b = float(abs(analytical_db - numerical_db) / (
            abs(analytical_db) + abs(numerical_db) + 1e-12
        ))

        return {
            "relative_error_w": err_w,
            "relative_error_b": err_b,
            "passed": err_w < 1e-5 and err_b < 1e-5,
        }
