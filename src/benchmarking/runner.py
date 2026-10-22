"""
EPM Fair Benchmark Runner
=========================

``run_benchmark(BenchmarkConfig)`` trains and evaluates several algorithms on
ONE dataset under identical conditions:

* one split (same indices, same seed) shared by every model;
* preprocessing fitted on training rows only, inside each model's pipeline;
* optional cross-validation / rolling-origin validation on the training part;
* optional small grid search on the training part only (never the test set);
* training time, batch inference latency per sample, serialized model size and
  parameter count (when the model exposes it);
* task-correct metrics with explicit notes for anything undefined;
* a multi-criteria summary (best metric, fastest, smallest, Pareto front) instead
  of a single "winner".

Per-model failures are captured in the result (status='failed') and never abort
the whole run. Results are persisted to ``experiments/benchmarks/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.benchmarking.catalog import AlgorithmSpec, check_compatibility, get_algorithm, list_algorithms, recommend_algorithms
from src.data.loaders import LoadedDataset, load_dataset
from src.data.registry import get_dataset_spec
from src.evaluation import task_metrics as TM
from src.preprocessing.pipelines import (
    PreprocessingConfig, build_tabular_preprocessor, deduplicate, describe_preprocessing,
    lag_window_features, make_model_pipeline, prepare_series, rolling_origin_splits, split_tabular,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ProgressFn = Callable[[float, str], None]


class BenchmarkError(ValueError):
    """Invalid or incompatible benchmark configuration."""


@dataclass
class BenchmarkConfig:
    dataset_id: str
    task: Optional[str] = None
    algorithms: Optional[List[str]] = None     # None -> recommended set
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 0
    tune: bool = False
    max_train_rows: Optional[int] = None
    allow_large: bool = False
    allow_download: bool = True
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    window: Optional[int] = None
    horizon: Optional[int] = None
    n_clusters: Optional[int] = None
    contamination: Optional[float] = None
    epochs: Optional[int] = None
    max_algorithms: int = 16
    algorithm_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {algorithm_id: {param: value}}
    save: bool = True
    save_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        d = dict(d)
        prep = d.pop("preprocessing", None) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise BenchmarkError(f"Unknown config fields: {sorted(unknown)}")
        pc = PreprocessingConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in prep.items()})
        return cls(preprocessing=pc, **d)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["preprocessing"] = self.preprocessing.to_dict()
        return d


# =============================================================================
# Helpers
# =============================================================================

def _model_size_kb(model: Any) -> Optional[float]:
    try:
        inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
        if hasattr(inner, "model_") and type(inner.model_).__module__.startswith("src.") and hasattr(inner.model_, "state_dict"):
            import io
            import torch

            buf = io.BytesIO()
            torch.save(inner.model_.state_dict(), buf)
            return round(len(buf.getvalue()) / 1024, 2)
        return round(len(pickle.dumps(model)) / 1024, 2)
    except Exception:
        return None


def _n_params(model: Any) -> Optional[int]:
    inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(inner, "est_"):
        inner = inner.est_
    try:
        if hasattr(inner, "n_parameters"):
            return int(inner.n_parameters())
        if hasattr(inner, "coef_"):
            c = np.asarray(inner.coef_)
            b = np.asarray(getattr(inner, "intercept_", 0.0))
            return int(c.size + b.size)
        if hasattr(inner, "tree_"):
            return int(inner.tree_.node_count)
        if hasattr(inner, "estimators_") and hasattr(inner.estimators_, "__iter__"):
            ests = np.ravel(np.asarray(inner.estimators_, dtype=object))
            if all(hasattr(e, "tree_") for e in ests):
                return int(sum(e.tree_.node_count for e in ests))
        if hasattr(inner, "cluster_centers_"):
            return int(np.asarray(inner.cluster_centers_).size)
    except Exception:
        return None
    return None


def _timed(fn: Callable[[], Any]) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _latency(predict: Callable[[Any], Any], X: Any) -> Tuple[Any, float]:
    """Warm-up on a few rows, then time the full batch. Returns (pred, µs/sample)."""
    n = len(X)
    try:
        predict(X[: min(8, n)])
    except Exception:
        pass
    pred, dt = _timed(lambda: predict(X))
    return pred, round(dt / max(n, 1) * 1e6, 3)


def _env() -> Dict[str, Any]:
    try:
        from src.research.registry import _software_env

        return _software_env()
    except Exception:
        import platform
        import sys

        return {"python": sys.version.split()[0], "platform": platform.platform()}


def _rows(X, idx):
    return X.iloc[idx] if isinstance(X, (pd.DataFrame, pd.Series)) else X[idx]


# =============================================================================
# Algorithm selection
# =============================================================================

def _select_algorithms(cfg: BenchmarkConfig, spec, task: str, n_train: int, window: Optional[int]
                       ) -> Tuple[List[AlgorithmSpec], Dict[str, List[str]]]:
    if cfg.algorithms:
        chosen = []
        for aid in cfg.algorithms:
            a = get_algorithm(aid)
            chosen.append(a)
    else:
        rec = recommend_algorithms(spec.id, task, n_train=n_train, window=window, allow_large=True)
        chosen = [get_algorithm(a) for a in rec["recommended"]]
    if len(chosen) > cfg.max_algorithms:
        raise BenchmarkError(f"At most {cfg.max_algorithms} algorithms per run (got {len(chosen)}).")
    ok, skipped = [], {}
    for a in chosen:
        good, why = check_compatibility(a, spec, task, n_train=n_train, window=window, allow_large=cfg.allow_large)
        if good:
            ok.append(a)
        else:
            skipped[a.id] = why
    if not ok:
        raise BenchmarkError(
            "None of the selected algorithms is compatible with this dataset: "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in skipped.items())
        )
    return ok, skipped


# =============================================================================
# Supervised tabular (regression / classification)
# =============================================================================

def _cv_score(task, factory_ctx, algo, X, y, num, cat, prep, folds, seed, classes=None):
    from sklearn.model_selection import KFold, StratifiedKFold

    splitter = (StratifiedKFold(folds, shuffle=True, random_state=seed) if task == "classification"
                else KFold(folds, shuffle=True, random_state=seed))
    scores = []
    for tr, va in splitter.split(X, y):
        pipe = make_model_pipeline(algo.factory(factory_ctx), num, cat, algo.needs_scaling, prep)
        pipe.fit(_rows(X, tr), y[tr])
        pred = pipe.predict(_rows(X, va))
        if task == "classification":
            m, _ = TM.classification_metrics(y[va], pred, classes=classes)
            scores.append(m["f1"])
        else:
            m, _ = TM.regression_metrics(y[va], pred)
            scores.append(m["rmse"])
    scores = [s for s in scores if s is not None]
    return {"metric": "f1" if task == "classification" else "rmse", "folds": folds,
            "mean": round(float(np.mean(scores)), 6) if scores else None,
            "std": round(float(np.std(scores)), 6) if scores else None}


def _tune(task, ctx, algo, X, y, num, cat, prep, seed, classes=None):
    """Small grid search with 3-fold CV on the TRAINING split only."""
    import itertools

    keys = sorted(algo.tune_grid)
    best, best_params, tried = None, {}, []
    for combo in itertools.product(*[algo.tune_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        c = dict(ctx, params={**ctx.get("params", {}), **params})
        s = _cv_score(task, c, algo, X, y, num, cat, prep, 3, seed, classes)["mean"]
        tried.append({"params": params, "cv_score": s})
        if s is None:
            continue
        better = best is None or (s > best if task == "classification" else s < best)
        if better:
            best, best_params = s, params
    return best_params, tried


def _run_supervised(ds: LoadedDataset, task: str, cfg: BenchmarkConfig, algos, progress: ProgressFn,
                    result: Dict[str, Any]) -> None:
    from sklearn.preprocessing import LabelEncoder

    spec = ds.spec
    X, y = ds.features, ds.target
    n_dup = 0
    if cfg.preprocessing.drop_duplicates:
        X, y, n_dup = deduplicate(X, y)
    strategy = spec.split_strategy if task == "classification" else ("random" if spec.split_strategy == "stratified" else spec.split_strategy)
    split = split_tabular(len(X), y if task == "classification" else None, strategy,
                          cfg.test_size, cfg.val_size, cfg.seed, cfg.max_train_rows)
    split.n_duplicates_dropped = n_dup
    result["split"] = split.summary()

    classes = None
    if task == "classification":
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = list(range(len(le.classes_)))
        result["class_labels"] = [str(c) for c in le.classes_]
        counts = pd.Series(y_enc[split.train_idx]).value_counts(normalize=True)
        result["class_balance_train"] = {str(le.classes_[i]): round(float(p), 4) for i, p in counts.items()}
        imbalanced = counts.min() < min(0.2, 0.5 / len(counts))
        if imbalanced and cfg.preprocessing.class_weight == "none":
            result["warnings"].append(
                f"Minority class is {counts.min():.1%} of training rows: prefer PR-AUC / F1 / balanced accuracy, "
                "and consider class_weight='balanced'."
            )
        y_arr = y_enc
    else:
        y_arr = y.to_numpy(dtype=float)

    num, cat = ds.numeric_features, ds.categorical_features
    Xtr, ytr = _rows(X, split.train_idx), y_arr[split.train_idx]
    Xva, yva = _rows(X, split.val_idx), y_arr[split.val_idx]
    Xte, yte = _rows(X, split.test_idx), y_arr[split.test_idx]

    for i, algo in enumerate(algos):
        progress(0.1 + 0.85 * i / len(algos), f"Training {algo.name}")
        row: Dict[str, Any] = {"algorithm": algo.id, "name": algo.name, "framework": algo.framework,
                               "family": algo.family, "cost": algo.cost, "interpretability": algo.interpretability}
        try:
            ctx = {"seed": cfg.seed, "class_weight": cfg.preprocessing.class_weight,
                   "epochs": cfg.epochs or 60, "params": dict(cfg.algorithm_params.get(algo.id, {}))}
            if cfg.tune and algo.tune_grid:
                best_params, tried = _tune(task, ctx, algo, Xtr, ytr, num, cat, cfg.preprocessing, cfg.seed, classes)
                ctx["params"] = best_params
                row["tuning"] = {"best_params": best_params, "grid": tried, "cv": "3-fold on training split"}
            if task == "classification" and cfg.preprocessing.class_weight == "balanced" and not algo.supports_class_weight:
                row.setdefault("notes", []).append("class_weight not supported by this model; trained unweighted")
            pipe = make_model_pipeline(algo.factory(ctx), num, cat, algo.needs_scaling, cfg.preprocessing)
            _, t_train = _timed(lambda: pipe.fit(Xtr, ytr))
            pred, lat = _latency(pipe.predict, Xte)
            row.update({"train_time_s": round(t_train, 4), "latency_us_per_sample": lat,
                        "model_size_kb": _model_size_kb(pipe), "n_parameters": _n_params(pipe),
                        "preprocessing": describe_preprocessing(num, cat, algo.needs_scaling, cfg.preprocessing)})
            if task == "classification":
                proba = None
                if algo.supports_proba and hasattr(pipe, "predict_proba"):
                    try:
                        proba = np.asarray(pipe.predict_proba(Xte), dtype=float)
                        if proba.ndim == 1:
                            proba = np.column_stack([1 - proba, proba])
                    except Exception as exc:
                        row.setdefault("notes", []).append(f"predict_proba failed: {exc}")
                m, notes = TM.classification_metrics(yte, np.asarray(pred).astype(int), proba, classes=classes,
                                                     positive_label=1)
                m["confusion_matrix"]["labels"] = result["class_labels"]
                if len(Xva):
                    vm, _ = TM.classification_metrics(yva, pipe.predict(Xva), classes=classes)
                    row["val_f1"] = vm["f1"]
                sens_cols = [c for c in spec.sensitive_features if c in X.columns]
                if sens_cols:
                    sens_te = X[sens_cols].astype("string").fillna("missing").iloc[split.test_idx]
                    row["fairness"] = {c: TM.group_fairness(yte, np.asarray(pred).astype(int), sens_te[c].to_numpy())
                                       for c in sens_te.columns}
            else:
                m, notes = TM.regression_metrics(yte, pred)
                if len(Xva):
                    vm, _ = TM.regression_metrics(yva, pipe.predict(Xva))
                    row["val_rmse"] = vm["rmse"]
            row["metrics"] = m
            row.setdefault("notes", []).extend(notes)
            if cfg.cv_folds and cfg.cv_folds >= 2:
                progress(0.1 + 0.85 * (i + 0.5) / len(algos), f"Cross-validating {algo.name}")
                row["cv"] = _cv_score(task, ctx, algo, Xtr, ytr, num, cat, cfg.preprocessing, cfg.cv_folds, cfg.seed, classes)
            row["status"] = "ok"
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3)})
        result["models"].append(row)


# =============================================================================
# Clustering
# =============================================================================

def _dbscan_eps(Xs: np.ndarray, k: int = 5, q: float = 0.9) -> float:
    from sklearn.neighbors import NearestNeighbors

    n = len(Xs)
    sample = Xs if n <= 5000 else Xs[np.random.default_rng(0).choice(n, 5000, replace=False)]
    d, _ = NearestNeighbors(n_neighbors=min(k + 1, len(sample))).fit(sample).kneighbors(sample)
    return float(np.quantile(d[:, -1], q))


def _run_clustering(ds: LoadedDataset, cfg: BenchmarkConfig, algos, progress, result) -> None:
    spec = ds.spec
    X = ds.features
    labels_true = ds.target  # external evaluation only - never passed to fit
    n_dup = 0
    if cfg.preprocessing.drop_duplicates:
        X, labels_true, n_dup = deduplicate(X, labels_true)
    idx = np.arange(len(X))
    sub_from = None
    if cfg.max_train_rows and len(idx) > cfg.max_train_rows:
        sub_from = len(idx)
        idx = np.sort(np.random.default_rng(cfg.seed).choice(idx, cfg.max_train_rows, replace=False))
    X = X.iloc[idx].reset_index(drop=True)
    labels_true = None if labels_true is None else labels_true.iloc[idx].reset_index(drop=True)
    prep_cfg = cfg.preprocessing
    if spec.id == "wholesale_customers" and not prep_cfg.log1p_columns:
        prep_cfg = PreprocessingConfig(**{**prep_cfg.to_dict(), "log1p_columns": tuple(spec.numeric_features),
                                          "clip_quantiles": tuple(prep_cfg.clip_quantiles)})
    pre = build_tabular_preprocessor(ds.numeric_features, ds.categorical_features, True,
                                     PreprocessingConfig(**{**prep_cfg.to_dict(), "scale": "always",
                                                            "clip_quantiles": tuple(prep_cfg.clip_quantiles),
                                                            "log1p_columns": tuple(prep_cfg.log1p_columns)}))
    Xs = pre.fit_transform(X)
    k = cfg.n_clusters or spec.n_clusters_hint or 3
    eps = _dbscan_eps(Xs)
    result["split"] = {"strategy": "transductive (all rows clustered; labels held out)", "n_rows": int(len(Xs)),
                       "duplicates_dropped": n_dup, "subsampled_from": sub_from}
    result["clustering_setup"] = {"n_clusters": k, "dbscan_eps": round(eps, 4), "dbscan_min_samples": 5,
                                  "external_labels_available": labels_true is not None,
                                  "preprocessing": describe_preprocessing(ds.numeric_features, ds.categorical_features, True, prep_cfg)}
    if any(a.id == "dbscan" for a in algos):
        result["warnings"].append(
            "DBSCAN's internal indices are computed without its noise points, which can flatter it; "
            "compare noise_fraction and the external ARI/NMI before preferring it."
        )
    for i, algo in enumerate(algos):
        progress(0.1 + 0.85 * i / len(algos), f"Clustering with {algo.name}")
        row = {"algorithm": algo.id, "name": algo.name, "framework": algo.framework, "family": algo.family,
               "cost": algo.cost, "interpretability": algo.interpretability}
        try:
            est = algo.factory({"seed": cfg.seed, "n_clusters": k, "dbscan_eps": eps,
                                "params": dict(cfg.algorithm_params.get(algo.id, {}))})
            if hasattr(est, "fit_predict"):
                labels, t = _timed(lambda: est.fit_predict(Xs))
            else:
                _, t = _timed(lambda: est.fit(Xs))
                labels = est.predict(Xs)
            lat = None
            if hasattr(est, "predict"):
                _, lat = _latency(est.predict, Xs)
            m, notes = TM.clustering_metrics(Xs, labels, None if labels_true is None else labels_true.to_numpy(),
                                             seed=cfg.seed)
            row.update({"status": "ok", "train_time_s": round(t, 4), "latency_us_per_sample": lat,
                        "model_size_kb": _model_size_kb(est), "n_parameters": _n_params(est),
                        "metrics": m, "notes": notes})
            if lat is None:
                row["notes"].append("no out-of-sample predict(): latency not applicable")
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3)})
        result["models"].append(row)


# =============================================================================
# Anomaly detection
# =============================================================================

def _run_anomaly(ds: LoadedDataset, cfg: BenchmarkConfig, algos, progress, result) -> None:
    from sklearn.preprocessing import StandardScaler

    spec = ds.spec
    y = None if ds.target is None else ds.target.to_numpy().astype(int)
    contamination = cfg.contamination or spec.contamination_prior or 0.05
    sequential = spec.split_strategy == "chronological"
    if sequential:
        w = cfg.window or spec.default_window or 12
        feats = lag_window_features(ds.features[spec.numeric_features[0]].to_numpy(), w)
        n = len(feats)
        cut = int(n * (1 - max(cfg.test_size, 0.5)))  # earliest part = training history
        tr, te = np.arange(cut), np.arange(cut, n)
        if cfg.max_train_rows and len(tr) > cfg.max_train_rows:
            tr = tr[-cfg.max_train_rows:]
        result["split"] = {"strategy": "chronological", "n_train": int(len(tr)), "n_test": int(len(te)),
                           "window": w, "test_start": str(ds.time_index[cut]) if ds.time_index is not None else cut}
        X_all = feats
    else:
        X_df = ds.features
        n_dup = 0
        if cfg.preprocessing.drop_duplicates:
            X_df, yser, n_dup = deduplicate(X_df, ds.target)
            y = None if yser is None else yser.to_numpy().astype(int)
        split = split_tabular(len(X_df), None if y is None else pd.Series(y), "stratified",
                              cfg.test_size, 0.0, cfg.seed, cfg.max_train_rows)
        split.n_duplicates_dropped = n_dup
        tr, te = split.train_idx, split.test_idx
        result["split"] = split.summary()
        pre = build_tabular_preprocessor(ds.numeric_features, ds.categorical_features, True, cfg.preprocessing)
        pre.fit(X_df.iloc[tr])
        X_all = pre.transform(X_df)
        X_all = np.asarray(X_all, dtype=float)
    scaler = StandardScaler().fit(X_all[tr])  # unsupervised fit on training rows only
    Xtr, Xte = scaler.transform(X_all[tr]), scaler.transform(X_all[te])
    ytr = None if y is None else y[tr]
    yte = None if y is None else y[te]
    result["anomaly_setup"] = {
        "contamination_prior": contamination,
        "threshold_rule": "flag test points whose score exceeds the (1 - contamination) quantile of TRAINING scores",
        "labels_used_for_training": False,
        "train_anomaly_fraction": None if ytr is None else round(float(ytr.mean()), 5),
    }
    for i, algo in enumerate(algos):
        progress(0.1 + 0.85 * i / len(algos), f"Fitting {algo.name}")
        row = {"algorithm": algo.id, "name": algo.name, "framework": algo.framework, "family": algo.family,
               "cost": algo.cost, "interpretability": algo.interpretability}
        try:
            est = algo.factory({"seed": cfg.seed, "contamination": contamination, "epochs": cfg.epochs or 40,
                                "params": dict(cfg.algorithm_params.get(algo.id, {}))})
            _, t = _timed(lambda: est.fit(Xtr))
            s_tr = -np.asarray(est.score_samples(Xtr))
            s_te, lat = _latency(lambda Z: -np.asarray(est.score_samples(Z)), Xte)
            thr = float(np.quantile(s_tr, 1 - contamination))
            flagged = (s_te > thr).astype(int)
            m, notes = TM.anomaly_metrics(yte, s_te, flagged, sequential=sequential)
            row.update({"status": "ok", "train_time_s": round(t, 4), "latency_us_per_sample": lat,
                        "model_size_kb": _model_size_kb(est), "n_parameters": _n_params(est),
                        "metrics": m, "notes": notes, "threshold": round(thr, 6)})
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3)})
        result["models"].append(row)


# =============================================================================
# Forecasting
# =============================================================================

def _run_forecasting(ds: LoadedDataset, cfg: BenchmarkConfig, algos, progress, result) -> None:
    from src.benchmarking.wrappers import NaiveForecaster

    spec = ds.spec
    window = cfg.window or spec.default_window or 24
    horizon = cfg.horizon or spec.default_horizon or 1
    m_season = spec.seasonal_period or 1
    train_ratio = 1 - cfg.test_size - cfg.val_size
    sd = prepare_series(ds.features, spec.target, window, horizon, m_season, train_ratio, cfg.val_size,
                        max_train_windows=cfg.max_train_rows)
    result["split"] = {"strategy": "chronological", **sd.boundaries, "window": window, "horizon": horizon,
                       "seasonal_period": m_season}
    if ds.time_index is not None and len(ds.time_index):
        ti = ds.time_index
        result["split"].update({"train_period": f"{ti[0]} -> {ti[sd.boundaries['train_end'] - 1]}",
                                "test_period": f"{ti[sd.boundaries['val_end']]} -> {ti[-1]}"})
    yte_raw = sd.unscale_target(sd.y_test)
    naive_pred = sd.unscale_target(NaiveForecaster(sd.target_channel, horizon).predict(sd.X_test))
    naive_rmse = float(np.sqrt(np.mean((naive_pred - yte_raw) ** 2)))
    scale = TM.mase_scale(sd.train_target_raw, m_season)
    result["forecast_setup"] = {"naive_test_rmse": round(naive_rmse, 6), "mase_scale": scale,
                                "preprocessing": ["chronological split", "causal forward-fill",
                                                  "per-channel z-score fit on training segment",
                                                  f"windows W={window} -> H={horizon}"],
                                "channels": sd.channels, "target": spec.target}
    ctx_base = {"seed": cfg.seed, "target_channel": sd.target_channel, "horizon": horizon,
                "seasonal_period": m_season, "epochs": cfg.epochs or 30, "params": {}}
    for i, algo in enumerate(algos):
        progress(0.1 + 0.85 * i / len(algos), f"Training {algo.name}")
        row = {"algorithm": algo.id, "name": algo.name, "framework": algo.framework, "family": algo.family,
               "cost": algo.cost, "interpretability": algo.interpretability}
        try:
            model = algo.factory({**ctx_base, "params": dict(cfg.algorithm_params.get(algo.id, {}))})
            if algo.framework == "pytorch":
                _, t = _timed(lambda: model.fit(sd.X_train, sd.y_train, sd.X_val, sd.y_val))
            else:
                _, t = _timed(lambda: model.fit(sd.X_train, sd.y_train))
            pred, lat = _latency(model.predict, sd.X_test)
            pred = sd.unscale_target(np.asarray(pred).reshape(len(sd.X_test), -1))
            m, notes = TM.forecasting_metrics(yte_raw, pred, scale, naive_rmse)
            row.update({"status": "ok", "train_time_s": round(t, 4), "latency_us_per_sample": lat,
                        "model_size_kb": _model_size_kb(model), "n_parameters": _n_params(model),
                        "metrics": m, "notes": notes})
            if hasattr(model, "history_") and model.history_.get("val_loss"):
                row["best_val_loss"] = round(float(np.min(model.history_["val_loss"])), 6)
            if cfg.cv_folds and cfg.cv_folds >= 2 and algo.framework != "pytorch":
                scores = []
                for tr, va in rolling_origin_splits(len(sd.X_train), cfg.cv_folds):
                    mm = algo.factory({**ctx_base, "params": dict(cfg.algorithm_params.get(algo.id, {}))}).fit(
                        sd.X_train[tr], sd.y_train[tr])
                    p = sd.unscale_target(np.asarray(mm.predict(sd.X_train[va])).reshape(len(va), -1))
                    scores.append(float(np.sqrt(np.mean((p - sd.unscale_target(sd.y_train[va])) ** 2))))
                row["cv"] = {"metric": "rmse", "scheme": f"rolling-origin, {cfg.cv_folds} expanding folds",
                             "mean": round(float(np.mean(scores)), 6) if scores else None,
                             "std": round(float(np.std(scores)), 6) if scores else None}
            elif cfg.cv_folds and algo.framework == "pytorch":
                row["notes"].append("rolling-origin CV skipped for neural models (cost); see val loss")
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3)})
        result["models"].append(row)


# =============================================================================
# Summary
# =============================================================================

def _summarise(result: Dict[str, Any], task: str) -> None:
    key, mode = TM.PRIMARY_METRIC[task]
    ok = [r for r in result["models"] if r.get("status") == "ok" and r["metrics"].get(key) is not None]
    result["primary_metric"] = {"name": key, "goal": "minimise" if mode == "min" else "maximise"}
    if not ok:
        result["summary"] = {"note": "no model produced the primary metric"}
        return
    sign = 1 if mode == "min" else -1
    ok.sort(key=lambda r: sign * r["metrics"][key])
    for rank, r in enumerate(ok, 1):
        r["rank"] = rank
    # Pareto front on (primary metric, latency)
    timed = [r for r in ok if r.get("latency_us_per_sample") is not None]
    pts = [(sign * r["metrics"][key], r["latency_us_per_sample"], r) for r in timed]
    for a, la, r in pts:
        r["pareto_optimal"] = not any((b <= a and lb <= la) and (b < a or lb < la) for b, lb, _ in pts)
    best = ok[0]
    summary = {
        "best_by_metric": best["algorithm"],
        "best_value": best["metrics"][key],
        "fastest_inference": min(timed, key=lambda r: r["latency_us_per_sample"])["algorithm"] if timed else None,
        "fastest_training": min(ok, key=lambda r: r.get("train_time_s") or np.inf)["algorithm"],
        "smallest_model": min(ok, key=lambda r: r.get("model_size_kb") or np.inf)["algorithm"],
        "pareto_front": [r["algorithm"] for r in ok if r.get("pareto_optimal")],
        "latency_note": "latency = batch prediction time / n_test, including the preprocessing pipeline",
        "note": ("Rankings use one test split on one dataset. Weigh accuracy against latency, size, "
                 "interpretability and CV variance; differences inside the CV std are not meaningful."),
    }
    base_id = {"regression": "dummy_reg", "classification": "dummy_clf", "forecasting": "naive"}.get(task)
    base = next((r for r in ok if r["algorithm"] == base_id), None)
    if base:
        bv = base["metrics"][key]
        for r in ok:
            r["beats_baseline"] = (r["metrics"][key] < bv) if mode == "min" else (r["metrics"][key] > bv)
        summary["baseline"] = base_id
    result["summary"] = summary


# =============================================================================
# Entry point
# =============================================================================

def _save(result: Dict[str, Any], save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{result['run_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    idx_path = save_dir / "index.json"
    try:
        index = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        index = []
    index = [e for e in index if e.get("run_id") != result["run_id"]]
    index.append({k: result.get(k) for k in ("run_id", "created_at", "dataset_id", "dataset_name", "task",
                                              "primary_metric", "status", "duration_s")}
                 | {"best": (result.get("summary") or {}).get("best_by_metric"),
                    "best_value": (result.get("summary") or {}).get("best_value"),
                    "n_models": len(result.get("models", []))})
    index = index[-200:]
    tmp = idx_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, idx_path)
    return path


def default_save_dir() -> Path:
    return PROJECT_ROOT / "experiments" / "benchmarks"


def run_benchmark(cfg: BenchmarkConfig, progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    """Run one benchmark. Raises BenchmarkError / DatasetLoadError for invalid setups."""
    from src.reproducibility import set_seed

    progress = progress or (lambda f, m: None)
    spec = get_dataset_spec(cfg.dataset_id)
    task = cfg.task or spec.tasks[0]
    if not spec.supports(task):
        raise BenchmarkError(f"Dataset '{spec.id}' does not support task '{task}' (supports {list(spec.tasks)}).")
    if spec.size_class == "large" and not cfg.allow_large:
        raise BenchmarkError(f"'{spec.id}' is a large dataset (~{spec.approx_memory_mb} MB in memory, "
                             f"{spec.approx_download_mb} MB download). Set allow_large to run it.")
    t0 = time.perf_counter()
    set_seed(cfg.seed, deterministic=False)
    progress(0.02, f"Loading {spec.name}")
    ds = load_dataset(spec.id, allow_download=cfg.allow_download, allow_large=cfg.allow_large, seed=cfg.seed)

    # rough training size for compatibility checks
    if task == "forecasting":
        n_train = int(ds.n_samples * (1 - cfg.test_size - cfg.val_size))
    elif task == "clustering":
        n_train = ds.n_samples
    elif task == "anomaly_detection" and spec.split_strategy == "chronological":
        n_train = int(ds.n_samples * (1 - max(cfg.test_size, 0.5)))
    else:
        n_train = int(ds.n_samples * (1 - cfg.test_size - (cfg.val_size if task != "anomaly_detection" else 0)))
    if cfg.max_train_rows:
        n_train = min(n_train, cfg.max_train_rows)
    window = cfg.window or spec.default_window
    algos, skipped = _select_algorithms(cfg, spec, task, n_train, window)

    created = datetime.now(timezone.utc)
    cfg_hash = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:8]
    result: Dict[str, Any] = {
        "run_id": f"{created:%Y%m%d_%H%M%S}_{spec.id}_{task}_{cfg_hash}",
        "created_at": created.isoformat(timespec="seconds"),
        "dataset_id": spec.id, "dataset_name": spec.name, "task": task,
        "dataset": {"source": spec.source, "source_url": spec.source_url, "license": spec.license,
                    "n_samples": ds.n_samples, "n_features": int(ds.features.shape[1]),
                    "numeric_features": len(ds.numeric_features),
                    "categorical_features": len(ds.categorical_features), "caveats": list(spec.caveats),
                    **{k: v for k, v in ds.info.items() if k in ("start", "end", "load_seconds", "memory_mb")}},
        "config": cfg.to_dict(), "config_hash": cfg_hash,
        "skipped_algorithms": skipped, "warnings": [], "models": [],
        "environment": _env(),
    }
    if skipped:
        result["warnings"].append(f"{len(skipped)} selected algorithm(s) skipped as incompatible (see skipped_algorithms).")

    runner = {"regression": _run_supervised, "classification": _run_supervised}.get(task)
    if runner:
        runner(ds, task, cfg, algos, progress, result)
    elif task == "clustering":
        _run_clustering(ds, cfg, algos, progress, result)
    elif task == "anomaly_detection":
        _run_anomaly(ds, cfg, algos, progress, result)
    elif task == "forecasting":
        _run_forecasting(ds, cfg, algos, progress, result)

    _summarise(result, task)
    n_fail = sum(1 for r in result["models"] if r.get("status") != "ok")
    result["status"] = "completed" if n_fail == 0 else ("partial" if n_fail < len(result["models"]) else "failed")
    result["duration_s"] = round(time.perf_counter() - t0, 3)
    progress(0.98, "Saving results")
    if cfg.save:
        path = _save(result, Path(cfg.save_dir) if cfg.save_dir else default_save_dir())
        try:
            result["saved_to"] = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            result["saved_to"] = path.name
    progress(1.0, "Done")
    return result


def load_history(save_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = (save_dir or default_save_dir()) / "index.json"
    try:
        return list(reversed(json.loads(p.read_text(encoding="utf-8"))))
    except Exception:
        return []


def load_run(run_id: str, save_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    safe = "".join(c for c in run_id if c.isalnum() or c in "_-")
    p = (save_dir or default_save_dir()) / f"{safe}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
