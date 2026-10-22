"""
EPM Algorithm-Aware, Leakage-Free Preprocessing
===============================================

Every learned transformation (imputer, scaler, encoder, clipper) lives inside a
scikit-learn ``Pipeline`` together with the estimator, so ``fit`` only ever sees
training rows and ``predict`` re-uses the *fitted* transforms unchanged. That
gives identical preprocessing at training and inference time and keeps
cross-validation honest (each fold refits on its own training part).

Time series are handled separately: segments are cut chronologically *before*
anything is fitted, gaps are filled causally (past values only), scaling
statistics come from the training segment only, and windows never cross a
segment boundary on the target side.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# =============================================================================
# Configuration (recorded with every experiment for reproducibility)
# =============================================================================

@dataclass
class PreprocessingConfig:
    impute_numeric: str = "median"          # median | mean
    impute_categorical: str = "most_frequent"
    scale: str = "auto"                     # auto (only scale-sensitive models) | always | never
    encode_categorical: str = "onehot"
    max_categories: int = 50                # rare levels beyond this are grouped
    clip_outliers: bool = False             # winsorise numeric columns to train quantiles
    clip_quantiles: Tuple[float, float] = (0.005, 0.995)
    log1p_columns: Tuple[str, ...] = ()     # e.g. skewed spending columns
    drop_duplicates: bool = True
    class_weight: str = "none"              # none | balanced (only for estimators that support it)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["clip_quantiles"] = list(self.clip_quantiles)
        d["log1p_columns"] = list(self.log1p_columns)
        return d


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column to quantiles learned on the training data only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


def _log1p_nonneg(X):
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


def build_tabular_preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str],
    needs_scaling: bool,
    cfg: Optional[PreprocessingConfig] = None,
) -> ColumnTransformer:
    """ColumnTransformer for mixed tabular data (unfitted).

    Scaling is applied only when ``needs_scaling`` (``cfg.scale='auto'``) so tree
    ensembles are not given an unnecessary step.
    """
    cfg = cfg or PreprocessingConfig()
    scale = cfg.scale == "always" or (cfg.scale == "auto" and needs_scaling)

    num_steps: List[Tuple[str, Any]] = [("impute", SimpleImputer(strategy=cfg.impute_numeric))]
    if cfg.log1p_columns:
        # log1p is a fixed (non-learned) transform; applied to all numeric cols in this block
        num_steps.append(("log1p", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")))
    if cfg.clip_outliers:
        num_steps.append(("clip", QuantileClipper(*cfg.clip_quantiles)))
    if scale:
        num_steps.append(("scale", StandardScaler()))

    transformers: List[Tuple[str, Any, List[str]]] = []
    log_cols = [c for c in numeric if c in cfg.log1p_columns]
    plain_cols = [c for c in numeric if c not in cfg.log1p_columns]
    if plain_cols:
        steps = [s for s in num_steps if s[0] != "log1p"]
        transformers.append(("num", Pipeline(steps), plain_cols))
    if log_cols:
        transformers.append(("num_log", Pipeline(num_steps), log_cols))
    if categorical:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")
             if cfg.impute_categorical == "constant" else SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False,
                                     max_categories=cfg.max_categories, min_frequency=5)),
        ])
        transformers.append(("cat", cat_pipe, list(categorical)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_model_pipeline(estimator: Any, numeric: Sequence[str], categorical: Sequence[str],
                        needs_scaling: bool, cfg: Optional[PreprocessingConfig] = None) -> Pipeline:
    """Preprocessor + estimator as one leakage-safe Pipeline."""
    return Pipeline([
        ("prep", build_tabular_preprocessor(numeric, categorical, needs_scaling, cfg)),
        ("model", estimator),
    ])


def describe_preprocessing(numeric: Sequence[str], categorical: Sequence[str], needs_scaling: bool,
                           cfg: PreprocessingConfig) -> List[str]:
    scale = cfg.scale == "always" or (cfg.scale == "auto" and needs_scaling)
    steps = [f"{cfg.impute_numeric} imputation of {len(numeric)} numeric column(s)"]
    if cfg.log1p_columns:
        steps.append(f"log1p on {list(cfg.log1p_columns)}")
    if cfg.clip_outliers:
        steps.append(f"winsorise to train quantiles {list(cfg.clip_quantiles)}")
    steps.append("standard scaling" if scale else "no scaling (model is scale-invariant)")
    if categorical:
        steps.append(f"impute + one-hot encode {len(categorical)} categorical column(s) (unseen -> ignored)")
    return steps


# =============================================================================
# Splitting (tabular)
# =============================================================================

@dataclass
class TabularSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    n_duplicates_dropped: int = 0
    n_train_subsampled_from: Optional[int] = None

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.train_idx, self.val_idx, self.test_idx):
            h.update(np.asarray(a, dtype=np.int64).tobytes())
        return h.hexdigest()[:12]

    def summary(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy, "n_train": int(len(self.train_idx)), "n_val": int(len(self.val_idx)),
            "n_test": int(len(self.test_idx)), "duplicates_dropped": self.n_duplicates_dropped,
            "train_subsampled_from": self.n_train_subsampled_from, "fingerprint": self.fingerprint(),
        }


def deduplicate(X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series], int]:
    """Drop exact duplicate rows (features + target) before splitting.

    This uses no target statistics, so it cannot leak; it prevents identical rows
    from landing in both train and test.
    """
    full = X if y is None else X.assign(__y__=y.values)
    keep = ~full.duplicated(keep="first").to_numpy()
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return X, y, 0
    X2 = X.loc[keep].reset_index(drop=True)
    y2 = None if y is None else y.loc[keep].reset_index(drop=True)
    return X2, y2, n_drop


def split_tabular(
    n: int,
    y: Optional[pd.Series],
    strategy: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    max_train_rows: Optional[int] = None,
) -> TabularSplit:
    """Reproducible train/val/test indices. ``strategy``: stratified | random | chronological."""
    if not 0 < test_size < 0.9 or not 0 <= val_size < 0.5 or test_size + val_size >= 0.95:
        raise ValueError("invalid test_size/val_size")
    idx = np.arange(n)
    if strategy == "chronological":
        n_test = max(1, int(round(n * test_size)))
        n_val = int(round(n * val_size))
        tr, va, te = idx[: n - n_test - n_val], idx[n - n_test - n_val: n - n_test], idx[n - n_test:]
    else:
        strat = None
        if strategy == "stratified" and y is not None:
            counts = y.value_counts()
            strat = y.to_numpy() if counts.min() >= 3 else None
        tr_va, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
        if val_size > 0:
            rel = val_size / (1 - test_size)
            strat2 = None if strat is None else y.to_numpy()[tr_va]
            tr, va = train_test_split(tr_va, test_size=rel, random_state=seed, stratify=strat2)
        else:
            tr, va = tr_va, np.array([], dtype=int)
        tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    sub_from = None
    if max_train_rows and len(tr) > max_train_rows:
        sub_from = int(len(tr))
        if strategy == "chronological":
            tr = tr[-max_train_rows:]  # keep the most recent history
        else:
            rng = np.random.default_rng(seed)
            tr = np.sort(rng.choice(tr, size=max_train_rows, replace=False))
    return TabularSplit(tr, va, te, strategy, n_train_subsampled_from=sub_from)


# =============================================================================
# Time-series preparation
# =============================================================================

@dataclass
class SeriesData:
    """Leakage-free windows for forecasting (scaled), plus what is needed to undo scaling."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    channel_mean: np.ndarray
    channel_std: np.ndarray
    target_channel: int
    window: int
    horizon: int
    channels: List[str]
    train_target_raw: np.ndarray  # unscaled training target (for MASE scaling)
    seasonal_period: int
    boundaries: Dict[str, Any] = field(default_factory=dict)

    @property
    def target_mean(self) -> float:
        return float(self.channel_mean[self.target_channel])

    @property
    def target_std(self) -> float:
        return float(self.channel_std[self.target_channel])

    def unscale_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y) * self.target_std + self.target_mean


def causal_fill(df: pd.DataFrame, fallback: Optional[pd.Series] = None) -> pd.DataFrame:
    """Forward-fill (past values only). Leading gaps use ``fallback`` (train medians)."""
    out = df.ffill()
    if fallback is not None:
        out = out.fillna(fallback)
    return out


def _windows(arr: np.ndarray, window: int, horizon: int, target: int, start_target: int, end_target: int,
             stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Windows whose *targets* lie in [start_target, end_target).

    Inputs may reach back before ``start_target`` (past observations are known at
    forecast time) but targets never cross into the next segment.
    """
    first = max(start_target, window)
    last = end_target - horizon  # last valid first-target index
    if last < first:
        return np.zeros((0, window, arr.shape[1]), np.float32), np.zeros((0, horizon), np.float32)
    starts = np.arange(first, last + 1, stride)
    X = np.stack([arr[s - window: s] for s in starts]).astype(np.float32)
    y = np.stack([arr[s: s + horizon, target] for s in starts]).astype(np.float32)
    return X, y


def prepare_series(
    frame: pd.DataFrame,
    target: str,
    window: int,
    horizon: int,
    seasonal_period: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_train_windows: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
) -> SeriesData:
    """Chronological split -> causal fill -> train-only scaling -> windows."""
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be >= 1")
    cols = list(channels) if channels else list(frame.columns)
    if target not in cols:
        cols = [target] + cols
    df = frame[cols].astype(float)
    n = len(df)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))
    if tr_end < window + horizon + 10 or n - va_end < horizon + 1:
        raise ValueError(f"Series too short ({n}) for window={window}, horizon={horizon}")

    train_part = df.iloc[:tr_end]
    medians = train_part.median()
    filled = causal_fill(df, fallback=medians)  # ffill only looks backwards in time

    mean = filled.iloc[:tr_end].mean().to_numpy()
    std = filled.iloc[:tr_end].std(ddof=0).to_numpy()
    std = np.where(std < 1e-8, 1.0, std)
    arr = (filled.to_numpy() - mean) / std
    t_idx = cols.index(target)

    Xtr, ytr = _windows(arr, window, horizon, t_idx, 0, tr_end)
    Xva, yva = _windows(arr, window, horizon, t_idx, tr_end, va_end)
    Xte, yte = _windows(arr, window, horizon, t_idx, va_end, n)
    if max_train_windows and len(Xtr) > max_train_windows:
        Xtr, ytr = Xtr[-max_train_windows:], ytr[-max_train_windows:]
    if len(Xte) == 0 or len(Xtr) == 0:
        raise ValueError("No windows produced; reduce window/horizon")
    return SeriesData(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte, y_test=yte,
        channel_mean=mean, channel_std=std, target_channel=t_idx, window=window, horizon=horizon,
        channels=cols, train_target_raw=filled.iloc[:tr_end][target].to_numpy(),
        seasonal_period=seasonal_period,
        boundaries={"n": n, "train_end": tr_end, "val_end": va_end,
                    "n_train_windows": int(len(Xtr)), "n_val_windows": int(len(Xva)), "n_test_windows": int(len(Xte))},
    )


def rolling_origin_splits(n: int, n_folds: int = 3, min_train_frac: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window (rolling-origin) folds over ``n`` chronologically ordered samples."""
    if n_folds < 1:
        return []
    start = int(n * min_train_frac)
    fold = (n - start) // n_folds
    if fold < 1:
        return []
    out = []
    for k in range(n_folds):
        tr_end = start + k * fold
        va_end = n if k == n_folds - 1 else tr_end + fold
        out.append((np.arange(0, tr_end), np.arange(tr_end, va_end)))
    return out


def lag_window_features(values: np.ndarray, window: int) -> np.ndarray:
    """Rows i -> [x_{i-window+1}, ..., x_i] using past values only (first rows padded by edge value)."""
    v = np.asarray(values, dtype=float).ravel()
    padded = np.r_[np.full(window - 1, v[0]), v]
    return np.lib.stride_tricks.sliding_window_view(padded, window).copy()
