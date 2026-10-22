"""
EPM Dataset Schema & Upload Validation
======================================

* :func:`validate_loaded_dataset` checks a loaded registry dataset against the
  schema its spec declares (columns present, target present/typed, enough rows,
  numeric columns really numeric, no all-empty columns, no infinities).
* :func:`validate_uploaded_csv` safely parses a user-supplied CSV: size, row and
  column caps, sanitised column names, no binary content, target checks.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MIN_ROWS = 20

UPLOAD_MAX_BYTES = 20 * 1024 * 1024
UPLOAD_MAX_ROWS = 200_000
UPLOAD_MAX_COLS = 500


class UploadValidationError(ValueError):
    """Raised when an uploaded file is rejected."""


def validate_loaded_dataset(ds: Any) -> List[str]:
    """Return a list of schema problems (empty list = valid)."""
    spec = ds.spec
    df: pd.DataFrame = ds.features
    problems: List[str] = []

    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (minimum {MIN_ROWS})")
    if df.shape[1] == 0:
        problems.append("no feature columns")

    expected = list(spec.numeric_features) + list(spec.categorical_features)
    expected = [c for c in expected if c not in spec.dropped_columns]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        problems.append(f"missing expected columns {missing}")

    for c in spec.numeric_features:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"column '{c}' should be numeric but is {df[c].dtype}")

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        problems.append(f"all-missing columns {empty}")

    num = df.select_dtypes(include=[np.number])
    if num.size and np.isinf(num.to_numpy(dtype=float, na_value=np.nan)).any():
        problems.append("infinite values present")

    if spec.target is not None:
        if ds.target is None:
            problems.append(f"target '{spec.target}' missing")
        else:
            if len(ds.target) != len(df):
                problems.append("target length differs from features")
            if ds.target.isna().any() and spec.target_type not in ("series",):
                problems.append("target contains missing values")
            if spec.target_type in ("binary", "anomaly_flag"):
                vals = set(pd.unique(ds.target.dropna()))
                if not vals <= {0, 1}:
                    problems.append(f"binary target must be 0/1, found {sorted(map(str, vals))[:5]}")
                if len(vals) < 2:
                    problems.append("binary target has a single class")
            if spec.target_type == "continuous" and not pd.api.types.is_numeric_dtype(ds.target):
                problems.append("continuous target is not numeric")

    if "forecasting" in spec.tasks and ds.time_index is not None:
        idx = pd.Index(ds.time_index)
        if not idx.is_monotonic_increasing:
            problems.append("time index is not sorted ascending")
        if idx.has_duplicates:
            problems.append("time index has duplicate timestamps")
    return problems


_SAFE_COL = re.compile(r"[^0-9A-Za-z_\-\. ()%/]")


def _sanitize_columns(cols: List[str]) -> List[str]:
    out, seen = [], set()
    for i, c in enumerate(cols):
        name = _SAFE_COL.sub("_", str(c)).strip()[:64] or f"col_{i}"
        if name[0] in "=+-@":  # spreadsheet-formula injection when results are exported
            name = "_" + name
        base, k = name, 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen.add(name)
        out.append(name)
    return out


def validate_uploaded_csv(
    raw: bytes,
    task: str,
    target: Optional[str],
    time_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse and validate an uploaded CSV. Returns (dataframe, report)."""
    from src.data.registry import TASKS

    if task not in TASKS:
        raise UploadValidationError(f"Unknown task '{task}'.")
    if not raw:
        raise UploadValidationError("Empty file.")
    if len(raw) > UPLOAD_MAX_BYTES:
        raise UploadValidationError(f"File is {len(raw) / 1e6:.1f} MB; limit is {UPLOAD_MAX_BYTES // (1024*1024)} MB.")
    if b"\x00" in raw[:8192]:
        raise UploadValidationError("File looks binary; only plain-text CSV is accepted.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadValidationError("File must be UTF-8 encoded CSV.") from None
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", nrows=UPLOAD_MAX_ROWS + 1)
    except Exception as exc:
        raise UploadValidationError(f"Could not parse CSV: {exc}") from None
    if len(df) > UPLOAD_MAX_ROWS:
        raise UploadValidationError(f"More than {UPLOAD_MAX_ROWS} rows.")
    if df.shape[1] > UPLOAD_MAX_COLS:
        raise UploadValidationError(f"More than {UPLOAD_MAX_COLS} columns.")
    if len(df) < MIN_ROWS:
        raise UploadValidationError(f"Need at least {MIN_ROWS} rows, got {len(df)}.")

    original = list(df.columns)
    df.columns = _sanitize_columns(original)
    rename = dict(zip(original, df.columns))
    if target is not None:
        target = rename.get(target, target)
    if time_column is not None:
        time_column = rename.get(time_column, time_column)

    needs_target = task in ("regression", "classification", "forecasting")
    if needs_target and not target:
        raise UploadValidationError(f"Task '{task}' needs a target column.")
    if target and target not in df.columns:
        raise UploadValidationError(f"Target column '{target}' not found. Columns: {list(df.columns)[:20]}")
    if target and df[target].isna().all():
        raise UploadValidationError("Target column is empty.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        raise UploadValidationError("Regression target must be numeric.")
    if task == "classification":
        n_cls = df[target].nunique(dropna=True)
        if n_cls < 2:
            raise UploadValidationError("Classification target needs at least 2 classes.")
        if n_cls > 100:
            raise UploadValidationError(f"Classification target has {n_cls} classes (limit 100); is it continuous?")
    if task == "anomaly_detection" and target:
        vals = set(pd.unique(df[target].dropna()))
        if not vals <= {0, 1}:
            raise UploadValidationError("Anomaly label column must be 0/1.")
    if task == "forecasting":
        if not time_column or time_column not in df.columns:
            raise UploadValidationError("Forecasting uploads need a time column.")
        ts = pd.to_datetime(df[time_column], errors="coerce")
        if ts.isna().mean() > 0.01:
            raise UploadValidationError("Time column could not be parsed as dates.")
        if not ts.is_monotonic_increasing:
            raise UploadValidationError("Rows must be sorted by time (ascending).")
        if not pd.api.types.is_numeric_dtype(df[target]):
            raise UploadValidationError("Forecast target must be numeric.")

    report = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "renamed_columns": {k: v for k, v in rename.items() if k != v},
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target": target, "time_column": time_column,
    }
    return df, report
