"""
EPM Neural Network in PyTorch & NumPy Parity Bridge
===================================================

PyTorch equivalent of the from-scratch Neural Network architecture, with
utilities to copy weights directly between NumPy and PyTorch for mathematical
verification down to machine precision.

ARCHITECTURE
------------
Input (D) -> Linear (H) -> ReLU -> Linear (C) -> Softmax / CrossEntropy
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Optional, Union, Literal, Any
import numpy as np
from numpy.typing import ArrayLike

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object  # type: ignore

from src.neural_networks.models_scratch import NeuralNetworkScratch
from src.neural_networks.layers import Linear, ReLU
from src.evaluation.metrics import accuracy


if HAS_TORCH:
    class PyTorchMLP(nn.Module):
        """
        PyTorch equivalent of the 2-layer NeuralNetworkScratch architecture.

        Parameters
        ----------
        in_features : int, default=5
        hidden_dim : int, default=16
        num_classes : int, default=2
        """

        def __init__(
            self,
            in_features: int = 5,
            hidden_dim: int = 16,
            num_classes: int = 2,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

            self.fc1 = nn.Linear(in_features, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass returning raw unnormalized logits."""
            h = self.relu(self.fc1(x))
            out = self.fc2(h)
            return out

        def copy_weights_from_scratch(self, scratch_model: NeuralNetworkScratch) -> None:
            """
            Synchronize PyTorch model weights with NumPy scratch model.
            
            Note: PyTorch stores linear layer weights as (out_features, in_features),
            whereas our NumPy Linear layer stores weights as (in_features, out_features).
            We transpose accordingly to ensure 100% mathematical parity.
            """
            l1: Linear = scratch_model.layers[0]  # type: ignore
            l2: Linear = scratch_model.layers[2]  # type: ignore

            with torch.no_grad():
                self.fc1.weight.copy_(torch.from_numpy(l1.W.T.astype(np.float32)))
                self.fc1.bias.copy_(torch.from_numpy(l1.b.astype(np.float32)))
                self.fc2.weight.copy_(torch.from_numpy(l2.W.T.astype(np.float32)))
                self.fc2.bias.copy_(torch.from_numpy(l2.b.astype(np.float32)))

        def copy_weights_to_scratch(self, scratch_model: NeuralNetworkScratch) -> None:
            """Synchronize NumPy scratch model weights from PyTorch model."""
            l1: Linear = scratch_model.layers[0]  # type: ignore
            l2: Linear = scratch_model.layers[2]  # type: ignore

            with torch.no_grad():
                l1.W = self.fc1.weight.detach().cpu().numpy().T.astype(np.float64)
                l1.b = self.fc1.bias.detach().cpu().numpy().astype(np.float64)
                l2.W = self.fc2.weight.detach().cpu().numpy().T.astype(np.float64)
                l2.b = self.fc2.bias.detach().cpu().numpy().astype(np.float64)

        def predict_proba(self, X: ArrayLike) -> np.ndarray:
            """Predict class probabilities."""
            self.eval()
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
            with torch.no_grad():
                logits = self(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        def predict(self, X: ArrayLike) -> np.ndarray:
            """Predict class labels."""
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)

        def score(self, X: ArrayLike, y: ArrayLike) -> float:
            """Return accuracy."""
            preds = self.predict(X)
            return accuracy(y, preds)

else:
    class PyTorchMLP:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is not installed in the current environment.")


def train_pytorch_model(
    model: Any,
    X_train: ArrayLike,
    y_train: ArrayLike,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    X_val: Optional[ArrayLike] = None,
    y_val: Optional[ArrayLike] = None,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a PyTorchMLP model and return metrics history."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for train_pytorch_model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
        y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(avg_loss)

        # Train accuracy
        model.eval()
        with torch.no_grad():
            train_logits = model(X_t.to(device))
            train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()
            train_acc = accuracy(y_train, train_preds)
            history["train_acc"].append(train_acc)

            if has_val:
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                val_acc = accuracy(y_val, val_preds)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

    return history


def benchmark_scratch_vs_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 16,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """
    Rigorously benchmark NumPy from-scratch MLP against PyTorch MLP.
    Compares:
      1. Correctness (Logit parity with identical weights)
      2. Training time & throughput
      3. Inference latency (us/sample)
      4. Memory consumption
      5. Implementation complexity (lines of code)
    """
    in_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # --- 1. Correctness Test (Exact Weight Parity) ---
    scratch_parity = NeuralNetworkScratch(
        in_features=in_features, hidden_dim=hidden_dim, num_classes=num_classes
    )
    pytorch_parity = PyTorchMLP(
        in_features=in_features, hidden_dim=hidden_dim, num_classes=num_classes
    )
    pytorch_parity.copy_weights_from_scratch(scratch_parity)

    logits_scratch = scratch_parity.forward(X_test, training=False)
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
    with torch.no_grad():
        logits_pytorch = pytorch_parity(X_test_tensor).numpy()

    max_logit_diff = float(np.max(np.abs(logits_scratch - logits_pytorch)))

    # --- 2. Train Scratch Model ---
    scratch_model = NeuralNetworkScratch(
        in_features=in_features, hidden_dim=hidden_dim, num_classes=num_classes
    )
    tracemalloc.start()
    start_t_scratch = time.perf_counter()
    scratch_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, optimizer="adam"
    )
    train_time_scratch = time.perf_counter() - start_t_scratch
    _, peak_mem_scratch = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start_inf = time.perf_counter()
    preds_scratch = scratch_model.predict(X_test)
    infer_time_scratch = time.perf_counter() - start_inf
    acc_scratch = accuracy(y_test, preds_scratch)

    # --- 3. Train PyTorch Model ---
    pytorch_model = PyTorchMLP(
        in_features=in_features, hidden_dim=hidden_dim, num_classes=num_classes
    )
    tracemalloc.start()
    start_t_pt = time.perf_counter()
    _ = train_pytorch_model(
        pytorch_model, X_train, y_train, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate
    )
    train_time_pt = time.perf_counter() - start_t_pt
    _, peak_mem_pt = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start_inf_pt = time.perf_counter()
    preds_pt = pytorch_model.predict(X_test)
    infer_time_pt = time.perf_counter() - start_inf_pt
    acc_pt = accuracy(y_test, preds_pt)

    n_test = len(X_test)

    return {
        "parity": {
            "max_absolute_logit_difference": max_logit_diff,
            "parity_verified": max_logit_diff < 1e-4,
        },
        "scratch": {
            "framework": "NumPy (Scratch)",
            "test_accuracy": round(acc_scratch, 4),
            "training_time_s": round(train_time_scratch, 4),
            "inference_time_s": round(infer_time_scratch, 5),
            "latency_us_per_sample": round((infer_time_scratch / n_test) * 1e6, 2),
            "peak_memory_mb": round(peak_mem_scratch / (1024 * 1024), 3),
            "lines_of_code": 380,  # layers + model + backprop
        },
        "pytorch": {
            "framework": "PyTorch",
            "test_accuracy": round(acc_pt, 4),
            "training_time_s": round(train_time_pt, 4),
            "inference_time_s": round(infer_time_pt, 5),
            "latency_us_per_sample": round((infer_time_pt / n_test) * 1e6, 2),
            "peak_memory_mb": round(peak_mem_pt / (1024 * 1024), 3),
            "lines_of_code": 35,   # nn.Module definition
        },
    }
