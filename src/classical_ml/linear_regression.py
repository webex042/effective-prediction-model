"""
EPM Linear Regression from Scratch
==================================

A pure NumPy implementation of Ordinary Least Squares (OLS) Linear Regression,
supporting both the analytical Normal Equation and iterative Gradient Descent,
with optional L2 (Ridge) regularization.

MATHEMATICAL FOUNDATION
-----------------------
1. Hypothesis:
   ŷ = Xw + b
   Where:
     - X is the design matrix of shape (N, D)
     - w is the weight vector of shape (D,)
     - b is the bias (scalar)
     - ŷ is the prediction vector of shape (N,)

2. Objective (Mean Squared Error Loss):
   J(w, b) = (1 / (2N)) * Σ_{i=1}^N (ŷᵢ - yᵢ)² + (λ / 2) * ||w||₂²
   
   Note: The factor 1/2 cancels out when taking derivatives, yielding clean gradients.

3. Analytical Solution (Normal Equation):
   Augment X with a column of 1s: X_aug = [X, 1], θ = [w, b]^T.
   Set ∇_θ J(θ) = 0:
     θ* = (X_aug^T X_aug + λ I_aug)⁻¹ X_aug^T y
   
   Complexity:
     - Time: O(D³ + N D²) for matrix multiplication and inversion
     - Memory: O(N D + D²)
     - Best when D is relatively small (< 10,000 features)

4. Iterative Solution (Gradient Descent):
   Gradients:
     ∂J/∂w = (1/N) * X^T (ŷ - y) + λ w
     ∂J/∂b = (1/N) * Σ (ŷ - y)
   
   Update Rule:
     w ← w - η * (∂J/∂w)
     b ← b - η * (∂J/∂b)
   
   Complexity per step:
     - Time: O(N D)
     - Memory: O(N D)
     - Scalable to massive datasets and large D
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np
from numpy.typing import ArrayLike

from src.evaluation.metrics import r2_score, mse, mae


class LinearRegressionScratch:
    """
    Ordinary Least Squares (OLS) Linear Regression implemented from scratch in NumPy.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate (step size η) for gradient descent.
    n_iterations : int, default=1000
        Number of gradient descent iterations.
    fit_intercept : bool, default=True
        Whether to calculate the intercept/bias term b.
    l2_reg : float, default=0.0
        L2 regularization parameter (Ridge penalty λ).
    method : {"gradient_descent", "normal_equation"}, default="gradient_descent"
        Solver method to compute model parameters.
    tolerance : float, default=1e-6
        Stopping criterion for gradient descent when loss change is below tolerance.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        l2_reg: float = 0.0,
        method: Literal["gradient_descent", "normal_equation"] = "gradient_descent",
        tolerance: float = 1e-6,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.l2_reg = l2_reg
        self.method = method
        self.tolerance = tolerance

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self.is_fitted: bool = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add a column of ones to X for the bias term."""
        intercept = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack((X, intercept))

    def _compute_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute MSE loss with optional L2 regularization."""
        n_samples = len(y_true)
        data_loss = (1.0 / (2.0 * n_samples)) * np.sum((y_pred - y_true) ** 2)
        w = weights if weights is not None else self.weights
        reg_loss = 0.0
        if self.l2_reg > 0.0 and w is not None:
            reg_loss = 0.5 * self.l2_reg * np.sum(w ** 2)
        return float(data_loss + reg_loss)

    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute parameters analytically via the Normal Equation."""
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_aug = self._add_intercept(X)
            # Regularization matrix (do not regularize bias)
            reg_matrix = np.eye(n_features + 1) * self.l2_reg
            reg_matrix[-1, -1] = 0.0

            # Solve (X^T X + λI) θ = X^T y
            XtX = X_aug.T @ X_aug + reg_matrix
            Xty = X_aug.T @ y
            theta = np.linalg.pinv(XtX) @ Xty

            self.weights = theta[:-1]
            self.bias = float(theta[-1])
        else:
            reg_matrix = np.eye(n_features) * self.l2_reg
            XtX = X.T @ X + reg_matrix
            Xty = X.T @ y
            self.weights = np.linalg.pinv(XtX) @ Xty
            self.bias = 0.0

        self.is_fitted = True
        y_pred = self.predict(X)
        self.loss_history = [self._compute_loss(y, y_pred)]

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize parameters iteratively via batch gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0 if not self.fit_intercept else float(np.mean(y))
        self.loss_history = []

        prev_loss = float("inf")

        for iteration in range(self.n_iterations):
            # Forward pass: ŷ = Xw + b
            y_pred = X @ self.weights + self.bias
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

            # Error residual
            error = y_pred - y  # shape (N,)

            # Gradients:
            # ∂J/∂w = (1/N) * X^T error + λw
            dw = (1.0 / n_samples) * (X.T @ error) + self.l2_reg * self.weights

            # ∂J/∂b = (1/N) * Σ error
            db = (1.0 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            # Gradient update
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegressionScratch:
        """
        Fit linear model to training data.

        Parameters
        ----------
        X : array-like of shape (N, D)
            Training feature matrix.
        y : array-like of shape (N,)
            Target values.

        Returns
        -------
        self : LinearRegressionScratch
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Sample count mismatch: X has {len(X_arr)} rows, y has {len(y_arr)} elements."
            )

        if self.method == "normal_equation":
            self._fit_normal_equation(X_arr, y_arr)
        elif self.method == "gradient_descent":
            self._fit_gradient_descent(X_arr, y_arr)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose 'gradient_descent' or 'normal_equation'.")

        self.is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (N, D)
            Samples.

        Returns
        -------
        y_pred : np.ndarray of shape (N,)
            Predicted values.
        """
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        return (X_arr @ self.weights + self.bias).ravel()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R² of the prediction."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def check_gradient(self, X: ArrayLike, y: ArrayLike, epsilon: float = 1e-5) -> dict[str, float]:
        """
        Perform numerical gradient checking to verify analytical gradients.

        Returns relative error between analytical and central-difference numerical gradients.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape

        w = np.random.randn(n_features)
        b = np.random.randn()

        # Analytical gradient
        y_pred = X_arr @ w + b
        error = y_pred - y_arr
        analytical_dw = (1.0 / n_samples) * (X_arr.T @ error) + self.l2_reg * w
        analytical_db = (1.0 / n_samples) * np.sum(error)

        # Numerical gradient for w
        numerical_dw = np.zeros_like(w)
        for i in range(n_features):
            w_pos = w.copy()
            w_pos[i] += epsilon
            loss_pos = self._compute_loss(y_arr, X_arr @ w_pos + b, weights=w_pos)

            w_neg = w.copy()
            w_neg[i] -= epsilon
            loss_neg = self._compute_loss(y_arr, X_arr @ w_neg + b, weights=w_neg)

            numerical_dw[i] = (loss_pos - loss_neg) / (2.0 * epsilon)

        # Numerical gradient for b
        loss_pos_b = self._compute_loss(y_arr, X_arr @ w + (b + epsilon), weights=w)
        loss_neg_b = self._compute_loss(y_arr, X_arr @ w + (b - epsilon), weights=w)
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
