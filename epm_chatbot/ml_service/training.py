"""
Explicit model training through EPM's own pipeline.

Training is a separate, user-requested step. It mirrors
``src.benchmarking.runner`` exactly for the parts that matter for validity:

* duplicates dropped before splitting, EPM's ``split_tabular`` (stratified /
  random / chronological) with a fixed seed;
* ``make_model_pipeline`` so imputation, scaling and encoding are fitted on the
  training split only and travel with the model to inference;
* metrics from ``src.evaluation.task_metrics`` on the held-out test split.

In addition it records what the chatbot needs later: the feature schema with
training ranges, permutation importance on the *test* split, native
importances, and for regression a split-conformal interval half-width from
*validation* residuals.
"""

from __future__ import annotations

import io
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.benchmarking.catalog import AlgorithmSpec, check_compatibility, get_algorithm
from src.data.loaders import LoadedDataset
from src.evaluation import task_metrics as TM
from src.preprocessing.pipelines import (
    PreprocessingConfig, deduplicate, describe_preprocessing, make_model_pipeline, prepare_series, split_tabular,
)

ProgressFn = Callable[[float, str], None]
SEED = 42
IMPORTANCE_MAX_ROWS = 2000
CONFORMAL_LEVEL = 0.9


class TrainingError(ValueError):
    """User-facing training problem."""


def _noop(p: float, msg: str) -> None:  # pragma: no cover - default progress sink
    pass


# =============================================================================
# Algorithm checks
# =============================================================================

def check_trainable(algo_id: str, ds: LoadedDataset, task: str, max_train_rows: int, allow_neural: bool) -> AlgorithmSpec:
    algo = get_algorithm(algo_id)
    if algo.task != task:
        raise TrainingError(f"{algo.name} is a {algo.task.replace('_', ' ')} algorithm, not {task}.")
    if algo.framework == "baseline" and task != "forecasting":
        raise TrainingError(f"{algo.name} is a reference baseline; it is used for comparisons, not predictions.")
    if algo.framework == "pytorch" and not allow_neural:
        raise TrainingError(f"{algo.name} is disabled on this server (EPM_CHAT_ALLOW_NEURAL=false).")
    n_train = min(int(ds.n_samples * 0.7), max_train_rows)
    ok, why = check_compatibility(algo, ds.spec, task, n_train=n_train, window=ds.spec.default_window, allow_large=False)
    if not ok:
        raise TrainingError(f"{algo.name} can't be trained on {ds.spec.name}: " + "; ".join(why) + ".")
    return algo


# =============================================================================
# Tabular (classification / regression)
# =============================================================================

def _feature_schema(X: pd.DataFrame, num: List[str], cat: List[str]) -> List[Dict[str, Any]]:
    out = []
    for c in X.columns:
        s = X[c]
        if c in num:
            v = pd.to_numeric(s, errors="coerce")
            out.append({"name": c, "type": "numeric", "min": _f(v.min()), "max": _f(v.max()),
                        "median": _f(v.median()), "mean": _f(v.mean()), "std": _f(v.std()),
                        "missing_pct": round(float(v.isna().mean() * 100), 2)})
        else:
            vc = s.astype("string").value_counts()
            out.append({"name": c, "type": "categorical", "categories": [str(x) for x in vc.index[:40]],
                        "mode": str(vc.index[0]) if len(vc) else None,
                        "missing_pct": round(float(s.isna().mean() * 100), 2)})
    return out


def _f(v: Any, digits: int = 6) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def _native_importance(pipe, feature_names_in: List[str]) -> Optional[Dict[str, Any]]:
    """Impurity importances or |coefficients| on the *transformed* features, when the model has them."""
    model = pipe.named_steps.get("model")
    est = getattr(model, "est_", None) or getattr(model, "model_", None) or model
    try:
        names = list(pipe.named_steps["prep"].get_feature_names_out())
    except Exception:
        names = None
    vals, kind = None, None
    if hasattr(est, "feature_importances_"):
        vals, kind = np.asarray(est.feature_importances_, float), "impurity / gain importance (training data)"
    elif hasattr(est, "coef_"):
        c = np.asarray(est.coef_, float)
        vals = np.abs(c).mean(axis=0) if c.ndim == 2 else np.abs(c)
        kind = "mean |coefficient| on preprocessed features"
    if vals is None or names is None or len(vals) != len(names):
        return None
    order = np.argsort(-vals)[:15]
    return {"method": kind, "items": [{"feature": str(names[i]), "value": _f(vals[i])} for i in order]}


def _permutation_importance(pipe, X: pd.DataFrame, y: np.ndarray, task: str, seed: int) -> Dict[str, Any]:
    from sklearn.inspection import permutation_importance

    if len(X) > IMPORTANCE_MAX_ROWS:
        idx = np.random.default_rng(seed).choice(len(X), IMPORTANCE_MAX_ROWS, replace=False)
        X, y = X.iloc[np.sort(idx)], y[np.sort(idx)]
    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    r = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=5, random_state=seed, n_jobs=1)
    order = np.argsort(-r.importances_mean)
    metric = "macro F1" if task == "classification" else "RMSE"
    return {
        "method": f"permutation importance on the held-out test split ({len(X)} rows, 5 repeats)",
        "metric": metric,
        "meaning": f"drop in {metric} quality when the feature's values are shuffled; ≈0 means the model does not rely on it",
        "items": [{"feature": str(X.columns[i]), "mean": _f(r.importances_mean[i]), "std": _f(r.importances_std[i])}
                  for i in order],
    }


def train_tabular(ds: LoadedDataset, task: str, algo: AlgorithmSpec, max_train_rows: int, epochs: int,
                  progress: ProgressFn = _noop) -> Tuple[Any, Dict[str, Any]]:
    from sklearn.preprocessing import LabelEncoder

    spec = ds.spec
    prep = PreprocessingConfig()
    X, y = ds.features, ds.target
    if y is None:
        raise TrainingError(f"{spec.name} has no target column for {task}.")
    X, y, n_dup = deduplicate(X, y)
    strategy = spec.split_strategy if task == "classification" else (
        "random" if spec.split_strategy == "stratified" else spec.split_strategy)
    split = split_tabular(len(X), y if task == "classification" else None, strategy, 0.2, 0.1, SEED, max_train_rows)
    split.n_duplicates_dropped = n_dup
    progress(0.15, "Split data (train / validation / test)")

    classes: Optional[List[str]] = None
    if task == "classification":
        le = LabelEncoder().fit(y)
        y_arr = le.transform(y)
        classes = [str(c) for c in le.classes_]
    else:
        y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y_arr).any():
            raise TrainingError("The regression target has missing or non-numeric values.")

    num, cat = list(ds.numeric_features), list(ds.categorical_features)
    tr, va, te = split.train_idx, split.val_idx, split.test_idx
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y_arr[tr], y_arr[va], y_arr[te]

    ctx = {"seed": SEED, "class_weight": prep.class_weight, "epochs": epochs, "params": {}}
    pipe = make_model_pipeline(algo.factory(ctx), num, cat, algo.needs_scaling, prep)
    progress(0.3, f"Training {algo.name} on {len(Xtr):,} rows")
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0
    progress(0.6, "Evaluating on the held-out test split")

    pred = pipe.predict(Xte)
    extra: Dict[str, Any] = {}
    if task == "classification":
        proba = None
        if algo.supports_proba and hasattr(pipe, "predict_proba"):
            try:
                proba = np.asarray(pipe.predict_proba(Xte), float)
            except Exception:
                proba = None
        metrics, notes = TM.classification_metrics(yte, np.asarray(pred).astype(int), proba,
                                                   classes=list(range(len(classes))), positive_label=1)
        metrics["confusion_matrix"]["labels"] = classes
        metrics.pop("reliability", None)
        if len(Xva):
            vm, _ = TM.classification_metrics(yva, np.asarray(pipe.predict(Xva)).astype(int),
                                              classes=list(range(len(classes))))
            extra["val_f1"] = vm["f1"]
        extra["supports_proba"] = proba is not None
        extra["class_balance_train"] = {classes[int(k)]: round(float(v), 4)
                                        for k, v in pd.Series(ytr).value_counts(normalize=True).sort_index().items()}
    else:
        metrics, notes = TM.regression_metrics(yte, pred)
        if len(Xva) >= 20:
            res = np.abs(np.asarray(pipe.predict(Xva), float) - yva)
            n = len(res)
            q = min(1.0, np.ceil((n + 1) * CONFORMAL_LEVEL) / n)
            extra["conformal"] = {"level": CONFORMAL_LEVEL, "halfwidth": _f(np.quantile(res, q, method="higher")),
                                  "n_calibration": n,
                                  "method": "split conformal: quantile of |residual| on the validation split"}
        extra["target_summary"] = {"min": _f(np.min(ytr)), "max": _f(np.max(ytr)), "mean": _f(np.mean(ytr))}

    progress(0.75, "Computing feature importance on held-out data")
    importance = _permutation_importance(pipe, Xte, yte, task, SEED)
    native = _native_importance(pipe, list(X.columns))

    meta = {
        "task": task,
        "features": _feature_schema(Xtr, num, cat),
        "feature_order": list(X.columns),
        "target": {"name": str(spec.target or ds.target.name), "classes": classes},
        "split": split.summary(),
        "metrics": metrics,
        "notes": notes,
        "preprocessing": describe_preprocessing(num, cat, algo.needs_scaling, prep),
        "preprocessing_config": prep.to_dict(),
        "importance": importance,
        "native_importance": native,
        "train_time_s": round(train_s, 4),
        "train_rows": int(len(Xtr)),
        **extra,
    }
    return pipe, meta


# =============================================================================
# Forecasting
# =============================================================================

def train_forecaster(ds: LoadedDataset, algo: AlgorithmSpec, max_train_rows: int, epochs: int,
                     horizon: Optional[int] = None, window: Optional[int] = None,
                     progress: ProgressFn = _noop) -> Tuple[Any, Dict[str, Any]]:
    from src.benchmarking.wrappers import NaiveForecaster

    spec = ds.spec
    window = int(window or spec.default_window or 24)
    horizon = int(horizon or spec.default_horizon or 1)
    m_season = int(spec.seasonal_period or 1)
    if algo.id == "seasonal_naive" and window < m_season:
        window = m_season
    frame = ds.features
    sd = prepare_series(frame, spec.target, window, horizon, m_season, 0.7, 0.15, max_train_windows=max_train_rows)
    progress(0.15, "Chronological split; scaling fitted on the training segment only")
    ctx = {"seed": SEED, "target_channel": sd.target_channel, "horizon": horizon, "seasonal_period": m_season,
           "epochs": epochs, "params": {}}
    model = algo.factory(ctx)
    progress(0.3, f"Training {algo.name} on {len(sd.X_train):,} windows")
    t0 = time.perf_counter()
    if algo.framework == "pytorch":
        model.fit(sd.X_train, sd.y_train, sd.X_val, sd.y_val)
    else:
        model.fit(sd.X_train, sd.y_train)
    train_s = time.perf_counter() - t0
    progress(0.7, "Evaluating on the held-out (latest) test segment")

    yte = sd.unscale_target(sd.y_test)
    pred = sd.unscale_target(np.asarray(model.predict(sd.X_test)).reshape(len(sd.X_test), -1))
    naive = sd.unscale_target(NaiveForecaster(sd.target_channel, horizon).predict(sd.X_test))
    naive_rmse = float(np.sqrt(np.mean((naive - yte) ** 2)))
    scale = TM.mase_scale(sd.train_target_raw, m_season)
    metrics, notes = TM.forecasting_metrics(yte, pred, scale, naive_rmse)

    tr_end = sd.boundaries["train_end"]
    medians = frame.iloc[:tr_end][sd.channels].astype(float).median()
    # residual spread on validation windows → empirical ±q band per horizon step
    band = None
    if len(sd.X_val) >= 20:
        pv = sd.unscale_target(np.asarray(model.predict(sd.X_val)).reshape(len(sd.X_val), -1))
        res = np.abs(pv - sd.unscale_target(sd.y_val))
        n = len(res)
        q = min(1.0, np.ceil((n + 1) * CONFORMAL_LEVEL) / n)
        band = {"level": CONFORMAL_LEVEL, "halfwidth_per_step": [_f(v) for v in np.quantile(res, q, axis=0, method="higher")],
                "n_calibration": n, "method": "split conformal on validation windows (per horizon step)"}
    ti = ds.time_index
    last_ts = str(ti[-1]) if ti is not None and len(ti) else None
    freq = None
    if ti is not None and len(ti) > 3 and isinstance(ti, pd.DatetimeIndex):
        freq = pd.infer_freq(ti[-50:]) if len(ti) >= 3 else None
    meta = {
        "task": "forecasting",
        "forecast": {
            "window": window, "horizon": horizon, "channels": list(sd.channels), "target": spec.target,
            "target_channel": int(sd.target_channel), "channel_mean": [float(v) for v in sd.channel_mean],
            "channel_std": [float(v) for v in sd.channel_std], "fill_medians": {k: float(v) for k, v in medians.items()},
            "seasonal_period": m_season, "last_timestamp": last_ts, "freq": freq, "n_rows": int(len(frame)),
            "naive_test_rmse": _f(naive_rmse), "mase_scale": scale, "interval": band,
        },
        "features": _feature_schema(frame.iloc[:tr_end][sd.channels], list(sd.channels), []),
        "feature_order": list(sd.channels),
        "target": {"name": spec.target, "classes": None},
        "split": {"strategy": "chronological", **sd.boundaries},
        "metrics": metrics,
        "notes": notes,
        "preprocessing": ["chronological split (70% train / 15% validation / 15% test)", "causal forward-fill",
                          "per-channel z-score fitted on the training segment", f"windows: {window} past steps → {horizon} ahead"],
        "train_time_s": round(train_s, 4),
        "train_rows": int(len(sd.X_train)),
    }
    return model, meta


# =============================================================================
# Entry point (also used inside the sandboxed training subprocess)
# =============================================================================

def train(ds: LoadedDataset, task: str, algo_id: str, *, max_train_rows: int, allow_neural: bool, epochs: int,
          horizon: Optional[int] = None, window: Optional[int] = None,
          progress: ProgressFn = _noop) -> Tuple[Any, Dict[str, Any]]:
    algo = check_trainable(algo_id, ds, task, max_train_rows, allow_neural)
    if task == "forecasting":
        model, meta = train_forecaster(ds, algo, max_train_rows, epochs, horizon, window, progress)
    elif task in ("classification", "regression"):
        model, meta = train_tabular(ds, task, algo, max_train_rows, epochs, progress)
    else:
        raise TrainingError(f"Predictions for {task} are not supported by the chatbot.")
    meta.update({"algorithm_id": algo.id, "algorithm_name": algo.name, "framework": algo.framework,
                 "family": algo.family, "interpretability": algo.interpretability,
                 "dataset_id": ds.spec.id, "dataset_name": ds.spec.name})
    progress(0.95, "Saving model")
    return model, meta


def serialize(model: Any) -> bytes:
    import joblib

    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.getvalue()
