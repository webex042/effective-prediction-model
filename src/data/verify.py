"""
Download + schema-verify every registered dataset.

    python run.py --verify-datasets            # all except 'large' datasets
    python run.py --verify-datasets --allow-large

Writes ``data/verification_report.json`` with one entry per dataset:
status (verified / failed / skipped), rows, columns, target summary, SHA-256 of
the cached download, and any mismatch against the registry's expected size.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.loaders import DatasetLoadError, clear_memory_cache, data_home, dataset_status, load_dataset
from src.data.registry import list_datasets


def verify_all(allow_large: bool = False, only: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, Any]:
    report: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": []}
    for spec in list_datasets():
        if only and spec.id not in only:
            continue
        entry: Dict[str, Any] = {"id": spec.id, "name": spec.name, "source": spec.source,
                                 "download_method": spec.download_method}
        if spec.size_class == "large" and not allow_large:
            entry["status"] = "skipped (large; pass --allow-large)"
            report["datasets"].append(entry)
            if verbose:
                print(f"  [SKIP] {spec.id:26s} large dataset")
            continue
        t0 = time.perf_counter()
        try:
            ds = load_dataset(spec.id, allow_download=True, allow_large=allow_large, use_memory_cache=False)
            entry.update({
                "status": "verified", "rows": ds.n_samples, "columns": int(ds.features.shape[1]),
                "seconds": round(time.perf_counter() - t0, 2), "memory_mb": round(ds.memory_mb, 2),
                "sha256": dataset_status(spec.id).get("sha256"),
            })
            if ds.target is not None:
                t = ds.target
                entry["target"] = (t.value_counts(normalize=True).round(4).astype(float).to_dict()
                                   if spec.target_type not in ("continuous", "series")
                                   else {"mean": float(t.mean()), "std": float(t.std())})
                entry["target"] = {str(k): v for k, v in entry["target"].items()}
            if spec.n_samples and abs(ds.n_samples - spec.n_samples) > max(5, 0.02 * spec.n_samples):
                entry["warning"] = f"expected ~{spec.n_samples} rows, got {ds.n_samples}"
        except (DatasetLoadError, KeyError) as exc:
            entry.update({"status": "failed", "error": str(exc), "seconds": round(time.perf_counter() - t0, 2)})
        if verbose:
            mark = {"verified": "OK  ", "failed": "FAIL"}.get(entry["status"], "SKIP")
            detail = f"{entry.get('rows')} rows x {entry.get('columns')} cols" if entry["status"] == "verified" else entry.get("error", "")
            print(f"  [{mark}] {spec.id:26s} {detail} {entry.get('warning', '')}")
        report["datasets"].append(entry)
        clear_memory_cache()
    out = data_home() / "verification_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(out)
    return report
