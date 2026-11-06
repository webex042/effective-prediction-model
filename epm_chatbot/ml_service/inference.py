"""
Inference on saved EPM models: input validation, predictions, uncertainty and
local explanations. Every number returned here is computed from the loaded
model and its training card; nothing is estimated or invented.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MAX_BATCH_ROWS = 10000


class InputError(ValueError):
    """Invalid user input; ``details`` lists every problem for the reply."""

    def __init__(self, message: str, details: Optional[List[str]] = None, missing: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.details = details or []
        self.missing = missing or []


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


# =============================================================================
# Parsing feature values out of a chat message
# =============================================================================

_NUM = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


def parse_feature_values(text: str, card: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Extract ``feature -> value`` pairs from free text or a JSON object.

    Recognised forms: ``petal_length=1.4``, ``petal length: 1.4``, ``petal length is 1.4``,
    a JSON object, or a plain list of numbers whose length equals the number of features
    (mapped in the model's feature order). Returns (values, notes).
    """
    feats = card["features"]
    names = [f["name"] for f in feats]
    notes: List[str] = []
    t = text.strip()
    # JSON object
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        import json

        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}, notes
        except Exception:
            pass
    values: Dict[str, Any] = {}
    raw_low = t.lower()
    # longest names first so "mean radius" does not steal "mean radius error"
    for f in sorted(feats, key=lambda f: -len(_norm(f["name"]))):
        n = _norm(f["name"])
        if not n:
            continue
        if f["type"] == "numeric":
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(n).replace(r"\ ", r"[\s_\-/.()]+")
                             + r"(?![a-z0-9])\s*(?:=|:|is|of|at|equals|to)?\s*(" + _NUM + ")", re.I)
            mm = pat.search(raw_low.replace("_", " "))
            if mm:
                values[f["name"]] = mm.group(1)
                raw_low = raw_low[:mm.start()] + " " * (mm.end() - mm.start()) + raw_low[mm.end():]
        else:
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(n).replace(r"\ ", r"[\s_\-/.()]+")
                             + r"(?![a-z0-9])\s*(?:=|:|is)\s*\"?([A-Za-z0-9_\-\.\s/]{1,40}?)\"?(?=,|;|\band\b|$)", re.I)
            mm = pat.search(t.replace("_", " "))
            if mm:
                values[f["name"]] = mm.group(1).strip()
    # generic "key = value" pairs: catches misspelt names and non-numeric values so they can be reported
    by_norm = {_norm(n): n for n in names}
    for mm in re.finditer(r"([A-Za-z_][\w./()\-]*)\s*[=:]\s*([^,;=]+?)\s*(?=,|;|$|\band\b)", t):
        key, val = mm.group(1), mm.group(2).strip().strip("\"'")
        name = by_norm.get(_norm(key))
        if name:
            values.setdefault(name, val)
        elif _norm(key) not in ("predict", "with", "using", "model") and not any(
                _norm(key) in _norm(n).split() for n in names):   # tail of a multi-word name, e.g. "width"
            values.setdefault(key, val)
    if not values:
        nums = re.findall(r"(?<![a-z_])" + _NUM, t.lower())
        if len(nums) == len(names) and all(f["type"] == "numeric" for f in feats):
            values = dict(zip(names, nums))
            notes.append("Values were mapped in the model's feature order: " + ", ".join(names) + ".")
    return values, notes


def unknown_keys(values: Dict[str, Any], card: Dict[str, Any]) -> List[str]:
    known = {f["name"] for f in card["features"]}
    return [k for k in values if k not in known]


def suggest_feature(name: str, card: Dict[str, Any]) -> Optional[str]:
    names = [f["name"] for f in card["features"]]
    hit = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    return hit[0] if hit else None


# =============================================================================
# Validation
# =============================================================================

def validate_row(values: Dict[str, Any], card: Dict[str, Any], fill_missing: bool = False
                 ) -> Tuple[pd.DataFrame, List[str]]:
    """Validate one input row against the model's training schema.

    Errors (raise ``InputError``): unknown feature names, non-numeric numbers,
    missing features unless ``fill_missing`` (then the model's own trained
    imputer fills them and a warning says so).
    Warnings: values outside the training range, unseen categories.
    """
    errors, warnings, missing = [], [], []
    unknown = unknown_keys(values, card)
    for k in unknown:
        s = suggest_feature(k, card)
        errors.append(f"'{k}' is not a feature of this model" + (f" — did you mean '{s}'?" if s else "."))
    row: Dict[str, Any] = {}
    for f in card["features"]:
        name = f["name"]
        v = values.get(name)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(name)
            row[name] = np.nan if f["type"] == "numeric" else None
            continue
        if f["type"] == "numeric":
            try:
                x = float(str(v).replace(",", "").strip())
            except ValueError:
                errors.append(f"'{name}' must be a number (got '{v}').")
                continue
            if not math.isfinite(x):
                errors.append(f"'{name}' must be a finite number.")
                continue
            lo, hi = f.get("min"), f.get("max")
            if lo is not None and hi is not None and (x < lo or x > hi):
                warnings.append(f"{name} = {x:g} is outside the training range [{lo:g}, {hi:g}]; "
                                "the model is extrapolating, so treat this prediction with extra caution.")
            row[name] = x
        else:
            sv = str(v).strip()
            cats = f.get("categories") or []
            if cats and sv not in cats:
                close = difflib.get_close_matches(sv, cats, n=1, cutoff=0.75)
                warnings.append(f"{name} = '{sv}' was not seen in training"
                                + (f" (closest known value: '{close[0]}')" if close else "")
                                + "; the encoder ignores unseen categories.")
            row[name] = sv
    if errors:
        raise InputError("Some inputs are invalid.", errors, missing)
    if missing and not fill_missing:
        raise InputError("Some features are missing.", [], missing)
    if missing:
        warnings.append("Missing features (" + ", ".join(missing) + ") were filled by the model's trained imputer "
                        "(training median / most frequent value).")
    order = card.get("feature_order") or [f["name"] for f in card["features"]]
    return pd.DataFrame([row])[order], warnings


def validate_batch(df: pd.DataFrame, card: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    order = card.get("feature_order") or [f["name"] for f in card["features"]]
    cols = {_norm(c): c for c in df.columns}
    missing = [n for n in order if n not in df.columns and _norm(n) not in cols]
    if missing:
        raise InputError("The file does not contain all the model's features.",
                         [f"Missing column(s): {', '.join(missing[:15])}" + (" …" if len(missing) > 15 else "")])
    if len(df) > MAX_BATCH_ROWS:
        raise InputError(f"At most {MAX_BATCH_ROWS:,} rows can be scored at once.")
    out = pd.DataFrame({n: df[n] if n in df.columns else df[cols[_norm(n)]] for n in order})
    warnings = []
    for f in card["features"]:
        if f["type"] == "numeric":
            v = pd.to_numeric(out[f["name"]], errors="coerce")
            bad = int(out[f["name"]].notna().sum() - v.notna().sum())
            if bad:
                warnings.append(f"{bad} non-numeric value(s) in '{f['name']}' were treated as missing.")
            out[f["name"]] = v
            lo, hi = f.get("min"), f.get("max")
            if lo is not None and hi is not None:
                n_out = int(((v < lo) | (v > hi)).sum())
                if n_out:
                    warnings.append(f"{n_out} row(s) have '{f['name']}' outside the training range.")
    n_missing = int(out.isna().sum().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing cell(s) were filled by the model's trained imputer.")
    extra = [c for c in df.columns if c not in order and _norm(c) not in {_norm(o) for o in order}]
    if extra:
        warnings.append("Ignored column(s) not used by the model: " + ", ".join(map(str, extra[:10])))
    return out, warnings


# =============================================================================
# Prediction
# =============================================================================

def predict_rows(model: Any, card: Dict[str, Any], X: pd.DataFrame) -> Dict[str, Any]:
    task = card["task"]
    pred = np.asarray(model.predict(X))
    out: Dict[str, Any] = {"n": int(len(X))}
    if task == "classification":
        classes = card["target"]["classes"]
        labels = [classes[int(p)] for p in pred]
        out["labels"] = labels
        if card.get("supports_proba") and hasattr(model, "predict_proba"):
            P = np.asarray(model.predict_proba(X), float)
            out["probabilities"] = [{classes[j]: round(float(P[i, j]), 6) for j in range(P.shape[1])}
                                    for i in range(len(X))]
    else:
        vals = pred.astype(float).ravel()
        out["values"] = [round(float(v), 6) for v in vals]
        conf = card.get("conformal")
        if conf and conf.get("halfwidth") is not None:
            h = conf["halfwidth"]
            out["intervals"] = [[round(float(v - h), 6), round(float(v + h), 6)] for v in vals]
            out["interval_level"] = conf["level"]
    return out


def local_explanation(model: Any, card: Dict[str, Any], X: pd.DataFrame) -> Dict[str, Any]:
    """One-feature-at-a-time occlusion against the training baseline.

    For each feature, its value is replaced by the training median (numeric) or
    most frequent value (categorical) and the change in the model output is
    measured. Positive = this input pushed the output up. This is a model-agnostic
    sensitivity analysis, not SHAP: effects are not additive when features interact.
    """
    task = card["task"]
    x = X.iloc[[0]].copy()

    def score(frame: pd.DataFrame) -> Tuple[np.ndarray, str]:
        if task == "classification":
            if card.get("supports_proba") and hasattr(model, "predict_proba"):
                return np.asarray(model.predict_proba(frame), float), "probability"
            p = np.asarray(model.predict(frame)).astype(int)
            out = np.zeros((len(frame), len(card["target"]["classes"])))
            out[np.arange(len(frame)), p] = 1.0
            return out, "predicted class (0/1)"
        return np.asarray(model.predict(frame), float).reshape(-1, 1), "prediction"

    base, unit = score(x)
    if task == "classification":
        k = int(np.argmax(base[0]))
        focus = card["target"]["classes"][k]
    else:
        k, focus = 0, card["target"]["name"]
    variants, names = [], []
    for f in card["features"]:
        b = f.get("median") if f["type"] == "numeric" else f.get("mode")
        if b is None:
            continue
        xv = x.copy()
        xv[f["name"]] = b
        variants.append(xv)
        names.append((f["name"], b))
    if not variants:
        return {"items": []}
    S, _ = score(pd.concat(variants, ignore_index=True))
    items = []
    for (name, b), s in zip(names, S):
        delta = float(base[0, k] - s[k])
        items.append({"feature": name, "value": _fmt_val(x.iloc[0][name]), "baseline": _fmt_val(b),
                      "effect": round(delta, 6)})
    items.sort(key=lambda d: -abs(d["effect"]))
    return {
        "method": "occlusion: each feature reset to its training median / most frequent value, one at a time",
        "output": f"{unit} of '{focus}'" if task == "classification" else f"predicted {focus}",
        "current": round(float(base[0, k]), 6),
        "items": items,
        "caveat": "Sensitivity analysis, not SHAP; effects can overlap when features interact.",
    }


def _fmt_val(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v if v is None or isinstance(v, (int, str)) else str(v)


# =============================================================================
# Forecasting
# =============================================================================

def forecast_next(model: Any, card: Dict[str, Any], history: pd.DataFrame) -> Dict[str, Any]:
    """Forecast the steps after the end of ``history`` with the training scaler."""
    fc = card["forecast"]
    chans, W, H = fc["channels"], fc["window"], fc["horizon"]
    missing = [c for c in chans if c not in history.columns]
    if missing:
        raise InputError("The series is missing channels the model was trained on.", [", ".join(missing)])
    if len(history) < W:
        raise InputError(f"At least {W} past rows are needed (the model's lookback window); got {len(history)}.")
    df = history[chans].astype(float).ffill().fillna(pd.Series(fc["fill_medians"]))
    arr = (df.to_numpy()[-W:] - np.asarray(fc["channel_mean"])) / np.asarray(fc["channel_std"])
    pred = np.asarray(model.predict(arr[None, :, :].astype(np.float32)), float).reshape(-1)[:H]
    t = fc["target_channel"]
    values = pred * fc["channel_std"][t] + fc["channel_mean"][t]
    out: Dict[str, Any] = {"values": [round(float(v), 6) for v in values], "horizon": H, "window": W,
                           "last_observed": round(float(df.iloc[-1][fc["target"]]), 6)}
    band = fc.get("interval")
    if band:
        hw = band["halfwidth_per_step"][:H]
        out["intervals"] = [[round(float(v - h), 6), round(float(v + h), 6)] for v, h in zip(values, hw)]
        out["interval_level"] = band["level"]
    steps = _future_index(history, fc, H)
    out["steps"] = steps
    return out


def _future_index(history: pd.DataFrame, fc: Dict[str, Any], H: int) -> List[str]:
    idx = history.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 2:
        freq = fc.get("freq") or pd.infer_freq(idx[-50:])
        if freq:
            return [str(ts) for ts in pd.date_range(idx[-1], periods=H + 1, freq=freq)[1:]]
    return [f"t+{k}" for k in range(1, H + 1)]
