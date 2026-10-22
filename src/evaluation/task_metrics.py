"""
EPM Task-Specific Evaluation
============================

Metrics for every task the benchmarking pipeline supports. Each function
returns ``(metrics, notes)``: undefined values are reported as ``None`` with a
note explaining why (e.g. ROC-AUC with a single class in the test set), rather
than a misleading 0 or 0.5.

The original from-scratch metrics in :mod:`src.evaluation.metrics` are kept
unchanged for the stage demos; this module uses scikit-learn's implementations
(already a project dependency) because they handle the multiclass averaging
that the binary-only scratch versions do not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics as skm

Metrics = Dict[str, Any]


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


# =============================================================================
# Regression
# =============================================================================

def regression_metrics(y_true, y_pred) -> Tuple[Metrics, List[str]]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    notes: List[str] = []
    if not np.all(np.isfinite(y_pred)):
        return {"mae": None, "mse": None, "rmse": None, "r2": None}, ["predictions contain NaN/inf"]
    mse = skm.mean_squared_error(y_true, y_pred)
    r2 = None
    if len(y_true) >= 2 and np.var(y_true) > 0:
        r2 = _f(skm.r2_score(y_true, y_pred))
    else:
        notes.append("R² undefined: constant target in the evaluation set")
    return {
        "mae": _f(skm.mean_absolute_error(y_true, y_pred)),
        "mse": _f(mse),
        "rmse": _f(np.sqrt(mse)),
        "r2": r2,
        "median_ae": _f(skm.median_absolute_error(y_true, y_pred)),
    }, notes


# =============================================================================
# Classification
# =============================================================================

def classification_metrics(
    y_true, y_pred, y_proba: Optional[np.ndarray] = None, classes: Optional[Sequence[Any]] = None,
    positive_label: Any = 1,
) -> Tuple[Metrics, List[str]]:
    """Binary: metrics for ``positive_label``. Multiclass: macro averages (+ weighted F1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    notes: List[str] = []
    labels = list(classes) if classes is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    binary = len(labels) == 2
    m: Metrics = {
        "accuracy": _f(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _f(skm.balanced_accuracy_score(y_true, y_pred)),
    }
    if binary:
        pos = positive_label if positive_label in labels else labels[1]
        m["precision"] = _f(skm.precision_score(y_true, y_pred, pos_label=pos, zero_division=0))
        m["recall"] = _f(skm.recall_score(y_true, y_pred, pos_label=pos, zero_division=0))
        m["f1"] = _f(skm.f1_score(y_true, y_pred, pos_label=pos, zero_division=0))
        m["averaging"] = f"binary (positive class = {pos})"
        if not np.any(y_pred == pos):
            notes.append("model never predicted the positive class; precision set to 0 by convention")
    else:
        m["precision"] = _f(skm.precision_score(y_true, y_pred, average="macro", zero_division=0))
        m["recall"] = _f(skm.recall_score(y_true, y_pred, average="macro", zero_division=0))
        m["f1"] = _f(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
        m["f1_weighted"] = _f(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0))
        m["averaging"] = "macro over classes"

    m["roc_auc"] = None
    m["pr_auc"] = None
    present = set(y_true.tolist())
    if y_proba is None:
        notes.append("ROC-AUC/PR-AUC not computed: model provides no probability or decision scores")
    elif len(present) < 2:
        notes.append("ROC-AUC/PR-AUC undefined: only one class present in the evaluation set")
    else:
        proba = np.asarray(y_proba, dtype=float)
        try:
            if binary:
                pos = positive_label if positive_label in labels else labels[1]
                score = proba[:, labels.index(pos)] if proba.ndim == 2 else proba
                yb = (y_true == pos).astype(int)
                m["roc_auc"] = _f(skm.roc_auc_score(yb, score))
                m["pr_auc"] = _f(skm.average_precision_score(yb, score))
            else:
                if len(present) < len(labels):
                    notes.append("some classes absent from the evaluation set; multiclass AUC computed on present classes")
                keep = [i for i, c in enumerate(labels) if c in present]
                p = proba[:, keep]
                p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
                lab_keep = [labels[i] for i in keep]
                if len(lab_keep) > 2:
                    m["roc_auc"] = _f(skm.roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=lab_keep))
                    Y = np.stack([(y_true == c).astype(int) for c in lab_keep], axis=1)
                    m["pr_auc"] = _f(skm.average_precision_score(Y, p, average="macro"))
                m["auc_averaging"] = "one-vs-rest, macro"
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"AUC computation failed: {exc}")
    if y_proba is not None:
        try:
            cal = calibration_metrics(y_true, np.asarray(y_proba, dtype=float), labels)
            m.update(cal)
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"calibration metrics failed: {exc}")
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    m["confusion_matrix"] = {"labels": [str(c) for c in labels], "matrix": cm.astype(int).tolist()}
    return m, notes


def calibration_metrics(y_true, proba: np.ndarray, labels: Sequence[Any], n_bins: int = 10) -> Metrics:
    """Expected calibration error (top-label, equal-width bins) and multiclass Brier score.

    Lower is better for both. ``reliability`` lists (mean confidence, accuracy, count)
    per non-empty bin so a reliability diagram can be drawn.
    """
    y_true = np.asarray(y_true)
    P = np.asarray(proba, dtype=float)
    if P.ndim == 1:
        P = np.column_stack([1 - P, P])
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    idx = {c: i for i, c in enumerate(labels)}
    onehot = np.zeros_like(P)
    for r, y in enumerate(y_true):
        if y in idx:
            onehot[r, idx[y]] = 1.0
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    conf = P.max(axis=1)
    pred = np.asarray(labels, dtype=object)[P.argmax(axis=1)]
    correct = (pred == y_true.astype(object)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            gap = abs(conf[mask].mean() - correct[mask].mean())
            ece += mask.mean() * gap
            rel.append([round(float(conf[mask].mean()), 4), round(float(correct[mask].mean()), 4), int(mask.sum())])
    return {"ece": _f(ece), "brier": _f(brier), "mean_confidence": _f(conf.mean()), "reliability": rel}


def group_fairness(y_true, y_pred, groups, positive_label: Any = 1, min_group: int = 30) -> Dict[str, Any]:
    """Selection rate & true-positive rate per group, plus the max gaps.

    Groups smaller than ``min_group`` are reported but excluded from the gaps.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups).astype(str)
    out: Dict[str, Any] = {"groups": {}}
    sel, tpr = [], []
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        n = int(mask.sum())
        pos = y_true[mask] == positive_label
        sr = float(np.mean(y_pred[mask] == positive_label)) if n else None
        t = float(np.mean(y_pred[mask][pos] == positive_label)) if pos.sum() else None
        out["groups"][g] = {"n": n, "selection_rate": _f(sr), "tpr": _f(t)}
        if n >= min_group:
            if sr is not None:
                sel.append(sr)
            if t is not None:
                tpr.append(t)
    out["demographic_parity_diff"] = _f(max(sel) - min(sel)) if len(sel) >= 2 else None
    out["equal_opportunity_diff"] = _f(max(tpr) - min(tpr)) if len(tpr) >= 2 else None
    out["min_group_size_for_gaps"] = min_group
    return out


# =============================================================================
# Clustering
# =============================================================================

def clustering_metrics(X, labels, true_labels=None, sample_size: int = 5000, seed: int = 0) -> Tuple[Metrics, List[str]]:
    """Internal indices on ``X`` (+ optional external ARI/NMI; labels never used for fitting)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    notes: List[str] = []
    noise = labels == -1
    m: Metrics = {
        "n_clusters": int(len(set(labels[~noise].tolist()))),
        "noise_fraction": _f(noise.mean()) if noise.any() else 0.0,
        "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
    }
    Xc, lc = X[~noise], labels[~noise]
    if noise.any():
        notes.append(f"{int(noise.sum())} noise points (label -1) excluded from internal indices")
    k = len(set(lc.tolist()))
    if k < 2 or k >= len(lc):
        notes.append(f"internal indices undefined for {k} cluster(s) on {len(lc)} points")
    else:
        if len(lc) > sample_size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(lc), sample_size, replace=False)
            m["silhouette"] = _f(skm.silhouette_score(Xc[idx], lc[idx]))
            notes.append(f"silhouette estimated on a fixed random sample of {sample_size} points")
        else:
            m["silhouette"] = _f(skm.silhouette_score(Xc, lc))
        m["davies_bouldin"] = _f(skm.davies_bouldin_score(Xc, lc))
        m["calinski_harabasz"] = _f(skm.calinski_harabasz_score(Xc, lc))
    if true_labels is not None:
        t = np.asarray(true_labels)
        m["ari_external"] = _f(skm.adjusted_rand_score(t, labels))
        m["nmi_external"] = _f(skm.normalized_mutual_info_score(t, labels))
        notes.append("ARI/NMI use held-out reference labels for evaluation only (never for fitting)")
    return m, notes


# =============================================================================
# Forecasting
# =============================================================================

def smape(y_true, y_pred) -> Optional[float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-12
    if not mask.any():
        return None
    return float(np.mean(2.0 * np.abs(y_pred - y_true)[mask] / denom[mask]) * 100.0)


def mase_scale(train_series: np.ndarray, m: int) -> Optional[float]:
    """In-sample MAE of the seasonal-naive forecast on the training series."""
    s = np.asarray(train_series, float)
    s = s[np.isfinite(s)]
    m = m if len(s) > m + 1 else 1
    if len(s) <= m:
        return None
    d = np.mean(np.abs(s[m:] - s[:-m]))
    return float(d) if d > 1e-12 else None


def forecasting_metrics(y_true, y_pred, scale: Optional[float], naive_rmse: Optional[float] = None) -> Tuple[Metrics, List[str]]:
    """``y_true``/``y_pred`` shape (N, H) in original units."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    if yt.ndim == 1:
        yt, yp = yt[:, None], yp.reshape(len(yp), -1)
    notes: List[str] = []
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    m: Metrics = {
        "mae": _f(mae), "rmse": _f(rmse), "smape": _f(smape(yt, yp)),
        "mase": _f(mae / scale) if scale else None,
        "rmse_per_horizon": [_f(v) for v in np.sqrt(np.mean(err ** 2, axis=0))],
        "horizon": int(yt.shape[1]),
    }
    if not scale:
        notes.append("MASE undefined: training series has no seasonal variation")
    if naive_rmse:
        m["skill_vs_naive"] = _f(1.0 - rmse / naive_rmse)
    return m, notes


# =============================================================================
# Anomaly detection
# =============================================================================

def detection_delays(y_true: np.ndarray, flagged: np.ndarray) -> Dict[str, Any]:
    """For each contiguous true-anomaly window: steps from window start to first flag inside it."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(bool)
    delays, missed = [], 0
    i, n = 0, len(y_true)
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            hits = np.flatnonzero(flagged[i:j])
            if hits.size:
                delays.append(int(hits[0]))
            else:
                missed += 1
            i = j
        else:
            i += 1
    return {
        "windows": len(delays) + missed, "detected": len(delays), "missed": missed,
        "mean_delay_steps": _f(np.mean(delays)) if delays else None,
        "delays_steps": delays,
    }


def anomaly_metrics(y_true, scores, flagged, sequential: bool = False) -> Tuple[Metrics, List[str]]:
    """``scores``: higher = more anomalous. ``flagged``: thresholded decisions."""
    scores = np.asarray(scores, float)
    flagged = np.asarray(flagged).astype(int)
    notes: List[str] = []
    m: Metrics = {"flagged_fraction": _f(flagged.mean())}
    if y_true is None:
        notes.append("no ground-truth anomaly labels: only unsupervised diagnostics are reported")
        m.update({"score_mean": _f(scores.mean()), "score_p99": _f(np.quantile(scores, 0.99))})
        return m, notes
    y = np.asarray(y_true).astype(int)
    m["precision"] = _f(skm.precision_score(y, flagged, zero_division=0))
    m["recall"] = _f(skm.recall_score(y, flagged, zero_division=0))
    m["f1"] = _f(skm.f1_score(y, flagged, zero_division=0))
    m["true_anomaly_fraction"] = _f(y.mean())
    if 0 < y.sum() < len(y):
        m["pr_auc"] = _f(skm.average_precision_score(y, scores))
        m["roc_auc"] = _f(skm.roc_auc_score(y, scores))
        m["pr_auc_random_baseline"] = _f(y.mean())
    else:
        m["pr_auc"] = m["roc_auc"] = None
        notes.append("PR-AUC/ROC-AUC undefined: evaluation segment contains only one class")
    if sequential:
        m["detection"] = detection_delays(y, flagged)
    return m, notes


PRIMARY_METRIC = {
    "regression": ("rmse", "min"),
    "classification": ("f1", "max"),
    "clustering": ("silhouette", "max"),
    "forecasting": ("rmse", "min"),
    "anomaly_detection": ("pr_auc", "max"),
}
