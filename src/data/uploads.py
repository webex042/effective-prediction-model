"""
User-uploaded datasets
======================

Validated CSV uploads are stored under ``data/raw/uploads/<id>.csv`` with a
JSON sidecar describing the task/target the user declared. They are exposed
through the same :class:`DatasetSpec` / :class:`LoadedDataset` interfaces as
registry datasets, with ids prefixed ``upload_``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.registry import DatasetSpec, REG_METRICS, CLF_METRICS, CLU_METRICS, FC_METRICS, AD_METRICS, TAB_PREP, CAT_PREP, TS_PREP
from src.data.validation import validate_uploaded_csv

_TASK_METRICS = {
    "regression": REG_METRICS, "classification": CLF_METRICS + ("pr_auc",), "clustering": CLU_METRICS,
    "forecasting": FC_METRICS, "anomaly_detection": AD_METRICS,
}


def uploads_dir() -> Path:
    from src.data.loaders import data_home

    return data_home() / "raw" / "uploads"


def _meta_path(upload_id: str) -> Path:
    return uploads_dir() / f"{upload_id}.json"


def _safe_id(upload_id: str) -> str:
    if not upload_id.startswith("upload_") or not all(c.isalnum() or c == "_" for c in upload_id):
        raise KeyError(f"Invalid upload id '{upload_id}'")
    return upload_id


def save_upload(raw: bytes, filename: str, task: str, target: Optional[str],
                time_column: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    df, report = validate_uploaded_csv(raw, task, target, time_column)
    digest = hashlib.sha256(raw).hexdigest()
    upload_id = f"upload_{digest[:12]}"
    d = uploads_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{upload_id}.csv.part"
    df.to_csv(tmp, index=False)
    os.replace(tmp, d / f"{upload_id}.csv")
    target = report["target"]
    time_column = report["time_column"]
    feat_cols = [c for c in df.columns if c not in (target, time_column)]
    num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feat_cols if c not in num]
    if task == "classification":
        ttype = "binary" if df[target].nunique() == 2 else "multiclass"
    elif task == "anomaly_detection":
        ttype = "anomaly_flag" if target else None
    elif task == "forecasting":
        ttype = "series"
        num = [c for c in df.columns if c != time_column and pd.api.types.is_numeric_dtype(df[c])]
        cat = []
    else:
        ttype = "continuous" if task == "regression" else ("multiclass" if target else None)
    meta = {
        "id": upload_id, "name": (name or Path(str(filename)).name)[:80], "task": task,
        "target": target, "target_type": ttype, "time_column": time_column,
        "numeric_features": num, "categorical_features": cat,
        "sha256": digest, "report": report,
    }
    _meta_path(upload_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def list_uploads() -> List[Dict[str, Any]]:
    d = uploads_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("upload_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def upload_spec(upload_id: str) -> DatasetSpec:
    upload_id = _safe_id(upload_id)
    try:
        m = json.loads(_meta_path(upload_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KeyError(f"Unknown upload '{upload_id}'") from None
    task = m["task"]
    tasks = (task,)
    label_role = "target"
    if task == "clustering":
        label_role = "external_eval_only"
    elif task == "anomaly_detection":
        label_role = "anomaly_label"
    n = m["report"]["rows"]
    return DatasetSpec(
        id=upload_id, name=f"[upload] {m['name']}",
        description=f"User-uploaded CSV ({n} rows) declared as a {task} dataset.",
        tasks=tasks, source="User upload", source_url="local file", download_url=None,
        download_method="upload", license="Provided by the user - check your own rights to the data",
        usage_notes="Validated on upload; column names were sanitised.",
        numeric_features=tuple(m["numeric_features"]), categorical_features=tuple(m["categorical_features"]),
        target=m["target"], target_type=m["target_type"], label_role=label_role,
        n_samples=n, n_features=len(m["numeric_features"]) + len(m["categorical_features"]),
        missing_values=f"{m['report']['missing_cells']} missing cells",
        approx_memory_mb=round(n * max(1, len(m["numeric_features"]) + len(m["categorical_features"])) * 8 / 1e6, 2),
        size_class="small" if n <= 5000 else ("medium" if n <= 60000 else "large"),
        preprocessing=TS_PREP if task == "forecasting" else (CAT_PREP if m["categorical_features"] else TAB_PREP),
        metrics={task: _TASK_METRICS[task]},
        split_strategy="chronological" if task == "forecasting" else ("stratified" if task == "classification" else "random"),
        time_column=m.get("time_column"), seasonal_period=24 if task == "forecasting" else None,
        default_window=24 if task == "forecasting" else None, default_horizon=1 if task == "forecasting" else None,
        contamination_prior=0.05 if task == "anomaly_detection" else None,
        n_clusters_hint=3 if task == "clustering" else None,
    )


def load_upload(upload_id: str):
    from src.data.loaders import LoadedDataset, _from_frame, _series_dataset

    spec = upload_spec(upload_id)
    df = pd.read_csv(uploads_dir() / f"{upload_id}.csv")
    if spec.tasks[0] == "forecasting":
        df[spec.time_column] = pd.to_datetime(df[spec.time_column])
        df = df.set_index(spec.time_column)[list(spec.numeric_features)]
        return _series_dataset(spec, df)
    return _from_frame(spec, df, spec.target)
