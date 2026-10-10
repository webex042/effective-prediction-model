"""
EPM Neural Network Model from Scratch
=====================================

A modular Multilayer Perceptron (MLP) implemented in pure NumPy.
Supports arbitrary layer stacks, numerical gradient checking, mini-batch training,
and Adam/Momentum-SGD optimizers.

ARCHITECTURE IMPLEMENTED
-----------------------
Input (D) -> Linear (H) -> ReLU -> Linear (C) -> Softmax & Cross-Entropy
"""

from __future__ import annotations

from typing import Optional, Union, Literal
import numpy as np
from numpy.typing import ArrayLike

from src.neural_networks.layers import Layer, Linear, ReLU, SoftmaxCrossEntropyLoss
from src.evaluation.metrics import accuracy


class NeuralNetworkScratch:
    """
    Multi-Layer Perceptron (MLP) built from scratch in NumPy.

    Parameters
    ----------
    layers : list of Layer, optional
        Pre-configured list of layers. If None, default 2-layer MLP (Input->Linear->ReLU->Linear) is built.
    in_features : int, default=5
        Number of input dimensions.
    hidden_dim : int, default=16
        Number of hidden units in the intermediate layer.
    num_classes : int, default=2
        Number of output classes.
    """

    def __init__(
        self,
        layers: Optional[list[Layer]] = None,
        in_features: int = 5,
        hidden_dim: int = 16,
        num_classes: int = 2,
    ) -> None:
        if layers is not None:
            self.layers = layers
        else:
            # Default architecture: Input -> Linear -> ReLU -> Linear
            self.layers = [
                Linear(in_features, hidden_dim, weight_init="kaiming"),
                ReLU(),
                Linear(hidden_dim, num_classes, weight_init="xavier"),
            ]

        self.loss_fn = SoftmaxCrossEntropyLoss()

        # Training history
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.train_acc_history: list[float] = []
        self.val_acc_history: list[float] = []
        self.is_fitted: bool = False

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Propagate inputs forward through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self) -> None:
        """Backpropagate loss gradient through all layers in reverse order."""
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_all_params(self) -> list[np.ndarray]:
        """Collect all trainable parameters across layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def get_all_grads(self) -> list[np.ndarray]:
        """Collect all parameter gradients across layers."""
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_grads())
        return grads

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer: Literal["adam", "sgd", "momentum"] = "adam",
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        verbose: bool = False,
    ) -> NeuralNetworkScratch:
        """
        Train the neural network using mini-batch gradient descent.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int).ravel()
        n_samples = len(X_arr)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_arr = np.asarray(X_val, dtype=np.float64)
            y_val_arr = np.asarray(y_val, dtype=int).ravel()

        params = self.get_all_params()

        # Optimizer state buffers
        velocities = [np.zeros_like(p) for p in params]  # For momentum
        m_buffers = [np.zeros_like(p) for p in params]   # For Adam 1st moment
        v_buffers = [np.zeros_like(p) for p in params]   # For Adam 2nd moment
        t_step = 0

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        for epoch in range(1, epochs + 1):
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            epoch_losses = []

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                X_batch = X_arr[batch_idx]
                y_batch = y_arr[batch_idx]

                # 1. Forward pass
                logits = self.forward(X_batch, training=True)

                # 2. Compute loss
                loss = self.loss_fn.forward(logits, y_batch)
                epoch_losses.append(loss)

                # 3. Backward pass (compute gradients)
                self.backward()

                # 4. Parameter update
                grads = self.get_all_grads()
                t_step += 1

                for i, (p, g) in enumerate(zip(params, grads)):
                    # L2 Weight decay
                    if weight_decay > 0.0:
                        g = g + weight_decay * p

                    if optimizer == "sgd":
                        p -= learning_rate * g

                    elif optimizer == "momentum":
                        velocities[i] = momentum * velocities[i] + learning_rate * g
                        p -= velocities[i]

                    elif optimizer == "adam":
                        # Biased moment estimates
                        m_buffers[i] = beta1 * m_buffers[i] + (1.0 - beta1) * g
                        v_buffers[i] = beta2 * v_buffers[i] + (1.0 - beta2) * (g ** 2)

                        # Bias-corrected moments
                        m_hat = m_buffers[i] / (1.0 - beta1 ** t_step)
                        v_hat = v_buffers[i] / (1.0 - beta2 ** t_step)

                        # Adam update
                        p -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # End of epoch tracking
            avg_train_loss = float(np.mean(epoch_losses))
            self.train_loss_history.append(avg_train_loss)

            train_preds = self.predict(X_arr)
            train_acc = accuracy(y_arr, train_preds)
            self.train_acc_history.append(train_acc)

            if has_val:
                val_logits = self.forward(X_val_arr, training=False)
                val_loss = self.loss_fn.forward(val_logits, y_val_arr)
                val_preds = self.predict(X_val_arr)
                val_acc = accuracy(y_val_arr, val_preds)

                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

                if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs):
                    print(
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%"
                    )
            else:
                if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs):
                    print(
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.1f}%"
                    )

        self.is_fitted = True
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return class probability distributions."""
        X_arr = np.asarray(X, dtype=np.float64)
        logits = self.forward(X_arr, training=False)
        # Compute stable softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict integer class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        preds = self.predict(X)
        return accuracy(y, preds)

    def gradient_check(
        self, X: ArrayLike, y: ArrayLike, epsilon: float = 1e-5
    ) -> dict[str, Union[float, bool]]:
        """
        Verify analytical backpropagation against numerical two-sided central differences:
          dJ/dθ ≈ (J(θ + ε) - J(θ - ε)) / (2ε)
        
        Returns max relative error across all parameters.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int).ravel()

        # 1. Forward and analytical backward
        logits = self.forward(X_arr, training=True)
        _ = self.loss_fn.forward(logits, y_arr)
        self.backward()

        all_params = self.get_all_params()
        all_grads = self.get_all_grads()

        max_rel_error = 0.0

        for param, analytical_grad in zip(all_params, all_grads):
            numerical_grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]

                # θ + ε
                param[idx] = orig_val + epsilon
                loss_pos = self.loss_fn.forward(self.forward(X_arr, training=False), y_arr)

                # θ - ε
                param[idx] = orig_val - epsilon
                loss_neg = self.loss_fn.forward(self.forward(X_arr, training=False), y_arr)

                # Restore
                param[idx] = orig_val

                # Central difference
                numerical_grad[idx] = (loss_pos - loss_neg) / (2.0 * epsilon)
                it.iternext()

            numerator = np.linalg.norm(analytical_grad - numerical_grad)
            denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad) + 1e-12
            rel_error = float(numerator / denominator)

            if rel_error > max_rel_error:
                max_rel_error = rel_error

        return {
            "max_relative_error": max_rel_error,
            "passed": max_rel_error < 1e-5,
        }
