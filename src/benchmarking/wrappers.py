"""
Adapters that give EPM's existing PyTorch models (and simple forecasting
baselines) a uniform scikit-learn-style interface for the benchmark runner.

Nothing here re-implements a model: the MLP is ``src.neural_networks.PyTorchMLP``,
the autoencoder is ``src.generative.StandardAutoencoder`` and the recurrent /
attention forecasters are ``src.timeseries.{LSTM,GRU}Forecaster`` and
``TimeSeriesTransformer``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _torch():
    import torch  # noqa: WPS433 (lazy: torch is optional for tabular-only use)

    return torch


def _seed_all(seed: int) -> None:
    from src.reproducibility import set_seed

    set_seed(seed, deterministic=False)


# =============================================================================
# Classification: PyTorch MLP (Stage 3)
# =============================================================================

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Wraps ``PyTorchMLP`` + ``train_pytorch_model`` (Stage 3)."""

    def __init__(self, hidden_dim: int = 64, epochs: int = 60, batch_size: int = 64,
                 learning_rate: float = 0.005, seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        from src.neural_networks.models_pytorch import PyTorchMLP, train_pytorch_model

        _seed_all(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.model_ = PyTorchMLP(in_features=X.shape[1], hidden_dim=self.hidden_dim,
                                 num_classes=len(self.classes_))
        self.history_ = train_pytorch_model(self.model_, X, y_idx, epochs=self.epochs,
                                            batch_size=self.batch_size, learning_rate=self.learning_rate)
        self.model_.to("cpu").eval()
        return self

    def predict_proba(self, X):
        torch = _torch()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model_.parameters()))


# =============================================================================
# Anomaly detection: Stage 7 autoencoder reconstruction error
# =============================================================================

class AutoencoderDetector(BaseEstimator):
    """Trains ``StandardAutoencoder`` on (unlabelled) training rows; score = reconstruction MSE."""

    def __init__(self, latent_dim: int = 4, epochs: int = 40, batch_size: int = 64,
                 learning_rate: float = 1e-3, seed: int = 42) -> None:
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y=None):
        torch = _torch()
        from torch.utils.data import DataLoader, TensorDataset
        from src.generative.models import StandardAutoencoder, train_generative_model

        _seed_all(self.seed)
        X = np.asarray(X, dtype=np.float32)
        d = X.shape[1]
        hidden = [max(8, min(64, 2 * d)), max(4, min(32, d))]
        self.model_ = StandardAutoencoder(input_dim=d, hidden_dims=hidden,
                                          latent_dim=min(self.latent_dim, max(1, d - 1)))
        loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=self.batch_size, shuffle=True)
        self.model_, self.loss_history_ = train_generative_model(self.model_, loader, epochs=self.epochs,
                                                                 lr=self.learning_rate)
        self.model_.to("cpu").eval()
        return self

    def score_samples(self, X):
        """Higher = more normal (sklearn convention); negative reconstruction error."""
        torch = _torch()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(X, dtype=np.float32))
            recon, _ = self.model_(x)
            err = ((recon.view(len(x), -1) - x.view(len(x), -1)) ** 2).mean(dim=1)
        return -err.numpy()

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model_.parameters()))


# =============================================================================
# Forecasting
# =============================================================================

class NaiveForecaster(BaseEstimator, RegressorMixin):
    """Persistence baseline: every future step equals the last observed target value."""

    def __init__(self, target_channel: int = 0, horizon: int = 1) -> None:
        self.target_channel = target_channel
        self.horizon = horizon

    def fit(self, X, y):
        return self

    def predict(self, X):
        last = np.asarray(X)[:, -1, self.target_channel]
        return np.repeat(last[:, None], self.horizon, axis=1)


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """y_{t+k} = y_{t+k-m}. Requires window >= m."""

    def __init__(self, target_channel: int = 0, horizon: int = 1, period: int = 24) -> None:
        self.target_channel = target_channel
        self.horizon = horizon
        self.period = period

    def fit(self, X, y):
        if np.asarray(X).shape[1] < self.period:
            raise ValueError(f"window ({np.asarray(X).shape[1]}) shorter than seasonal period ({self.period})")
        return self

    def predict(self, X):
        X = np.asarray(X)
        W = X.shape[1]
        cols = [X[:, W - self.period + ((k) % self.period), self.target_channel] for k in range(self.horizon)]
        return np.stack(cols, axis=1)


class FlatWindowForecaster(BaseEstimator, RegressorMixin):
    """Flattens (N, W, D) windows for a tabular regressor (Stage 5 approach)."""

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.multioutput import MultiOutputRegressor

        X = np.asarray(X)
        y = np.asarray(y)
        est = clone(self.estimator)
        if y.ndim == 2 and y.shape[1] > 1:
            name = type(est).__name__
            if name not in ("Ridge", "LinearRegression", "RandomForestRegressor", "KNeighborsRegressor"):
                est = MultiOutputRegressor(est)
        self.est_ = est.fit(X.reshape(len(X), -1), y if y.ndim == 1 or y.shape[1] > 1 else y.ravel())
        return self

    def predict(self, X):
        X = np.asarray(X)
        p = self.est_.predict(X.reshape(len(X), -1))
        return p.reshape(len(X), -1)


class TorchSequenceForecaster(BaseEstimator, RegressorMixin):
    """Wraps EPM's LSTM / GRU / Temporal-Transformer forecasters (Stage 5).

    Uses ``train_forecasting_model`` and keeps the weights of the epoch with the
    best validation loss (validation windows come strictly after training ones).
    """

    def __init__(self, kind: str = "lstm", hidden_dim: int = 32, epochs: int = 30, batch_size: int = 64,
                 lr: float = 0.003, seed: int = 42) -> None:
        self.kind = kind
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def _build(self, n_channels: int, horizon: int):
        from src.timeseries.models import LSTMForecaster, GRUForecaster, TimeSeriesTransformer

        if self.kind == "lstm":
            return LSTMForecaster(input_dim=n_channels, hidden_dim=self.hidden_dim, horizon=horizon)
        if self.kind == "gru":
            return GRUForecaster(input_dim=n_channels, hidden_dim=self.hidden_dim, horizon=horizon)
        if self.kind == "transformer":
            return TimeSeriesTransformer(input_dim=n_channels, d_model=self.hidden_dim, horizon=horizon,
                                         d_ff=2 * self.hidden_dim)
        raise ValueError(f"unknown kind {self.kind}")

    def fit(self, X, y, X_val=None, y_val=None):
        """Same objective/optimiser as ``train_forecasting_model`` (MSE + Adam), CPU,
        with best-validation-epoch checkpointing."""
        torch = _torch()
        import copy
        from torch.utils.data import DataLoader, TensorDataset

        _seed_all(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32).reshape(len(X), -1)
        self.model_ = self._build(X.shape[2], y.shape[1])
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            Xv = torch.from_numpy(np.asarray(X_val, np.float32))
            yv = torch.from_numpy(np.asarray(y_val, np.float32).reshape(len(X_val), -1))
        gen = torch.Generator().manual_seed(int(self.seed))
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                            batch_size=self.batch_size, shuffle=True, generator=gen)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        best, best_state, history = np.inf, None, {"train_loss": [], "val_loss": []}
        for _ in range(self.epochs):
            self.model_.train()
            losses = []
            for bx, by in loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(bx), by)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            history["train_loss"].append(float(np.mean(losses)))
            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    v = float(loss_fn(self.model_(Xv), yv).item())
                history["val_loss"].append(v)
                if v < best:
                    best = v
                    best_state = copy.deepcopy(self.model_.state_dict())
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.history_ = history
        self.model_.to("cpu").eval()
        return self

    def predict(self, X):
        torch = _torch()
        with torch.no_grad():
            return self.model_(torch.from_numpy(np.asarray(X, np.float32))).numpy()

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model_.parameters()))


# =============================================================================
# Adapter for EPM's from-scratch NumPy models (Stage 2)
# =============================================================================

class ScratchAdapter(BaseEstimator):
    """Lets the Stage 2 scratch models sit inside a scikit-learn Pipeline.

    Recent scikit-learn versions require estimators in a Pipeline to inherit
    ``BaseEstimator`` (``__sklearn_tags__``). The scratch classes are left
    untouched; this adapter only delegates ``fit`` / ``predict`` /
    ``predict_proba`` to a fresh instance of them.
    """

    def __init__(self, scratch_class: Any = None, scratch_params: Optional[Dict[str, Any]] = None,
                 kind: str = "regressor") -> None:
        self.scratch_class = scratch_class
        self.scratch_params = scratch_params
        self.kind = kind

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier" if self.kind == "classifier" else "regressor"
        return tags

    @property
    def _estimator_type(self):  # older scikit-learn
        return "classifier" if self.kind == "classifier" else "regressor"

    def fit(self, X, y):
        self.model_ = self.scratch_class(**(self.scratch_params or {}))
        self.model_.fit(np.asarray(X, dtype=np.float64), np.asarray(y))
        if self.kind == "classifier":
            self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X):
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError("scratch model has no predict_proba")
        return self.model_.predict_proba(np.asarray(X, dtype=np.float64))

    def n_parameters(self) -> Optional[int]:
        m = self.model_
        w = getattr(m, "weights", None)
        if isinstance(w, np.ndarray):
            return int(w.size + 1)
        subs = getattr(m, "_models", None)
        if subs:
            return int(sum(np.asarray(s.weights).size + 1 for s in subs))
        if getattr(m, "X_train", None) is not None:  # KNN stores the training set
            return int(np.asarray(m.X_train).size)
        root = getattr(m, "root", None) or getattr(m, "tree_", None)
        if root is not None and hasattr(root, "left"):
            def count(n):
                return 0 if n is None else 1 + count(n.left) + count(n.right)
            return count(root)
        return None
