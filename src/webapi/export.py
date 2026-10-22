"""
Export the dataset/algorithm catalogue and benchmark history as static JSON
under ``web/catalog/`` so the dashboard can show them without a Python
backend (e.g. on the Vercel deployment). Training still requires the local API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from src.benchmarking.catalog import check_compatibility, list_algorithms
from src.benchmarking.runner import load_history, load_run
from src.data.registry import list_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def export_static_catalog(out_dir: Path | None = None, max_runs: int = 25) -> Dict[str, Any]:
    out = out_dir or PROJECT_ROOT / "web" / "catalog"
    datasets = []
    for spec in list_datasets():
        d = spec.to_dict()
        d["status"] = {"state": "static", "note": "Download state is only known to the local server."}
        d["algorithms_by_task"] = {
            t: [a.id for a in list_algorithms(t)
                if not [r for r in check_compatibility(a, spec, t, allow_large=True)[1] if "optional dependency" not in r]]
            for t in spec.tasks
        }
        datasets.append(d)
    algos = [a.to_dict() for a in list_algorithms()]
    for a in algos:
        a["available"] = None  # depends on the machine running the local server
    history = load_history()[:max_runs]
    _write(out / "datasets.json", {"datasets": datasets})
    _write(out / "algorithms.json", {"algorithms": algos})
    _write(out / "history.json", {"runs": history})
    runs_dir = out / "runs"
    kept = set()
    for h in history:
        r = load_run(h["run_id"])
        if r:
            r.pop("environment", None)
            for m in r.get("models", []):
                m.pop("traceback", None)
            _write(runs_dir / f"{h['run_id']}.json", r)
            kept.add(f"{h['run_id']}.json")
    if runs_dir.exists():
        for p in runs_dir.glob("*.json"):
            if p.name not in kept:
                p.unlink()
    return {"datasets": len(datasets), "algorithms": len(algos), "runs": len(kept), "out_dir": str(out)}
