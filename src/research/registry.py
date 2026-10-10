"""
EPM Experiment Registry (shared by Stage 12 compression & Stage 13 research)
============================================================================

A unified, structured record of a single evaluated model/experiment. Every
field the roadmap asks us to track (architecture, dataset split, seed,
parameter count, sparsity, serialized size, peak memory, latency, predictive
metrics, environment, checkpoint, status, limitations) has an explicit slot.

Persistence
-----------
* One JSON file per record:  ``experiments/registry/<experiment_id>.json``
* A rolling index:           ``experiments/registry/index.json`` and
                             ``experiments/registry/index.csv``

The registry never overwrites a baseline model artifact; it only records
*where* checkpoints live via ``checkpoint_location``.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _software_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in ("numpy", "torch", "sklearn"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[pkg] = "not installed"
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        env["cuda_available"] = False
        env["device_name"] = "cpu"
    return env


@dataclass
class ExperimentRecord:
    """A single row in the experiment registry."""

    experiment_id: str
    name: str
    stage: int
    status: str = "completed"           # completed | failed | running
    hypothesis: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Model / method
    model_architecture: str = ""
    method: str = ""                    # e.g. "baseline", "ddim_10", "dynamic_int8"
    baseline_checkpoint: Optional[str] = None
    checkpoint_location: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Data
    dataset: str = ""
    dataset_version: str = ""
    split: Dict[str, int] = field(default_factory=dict)   # {train, val, test}
    seed: Optional[int] = None
    seeds: List[int] = field(default_factory=list)

    # Model statistics
    parameter_count: Optional[int] = None
    sparsity: Optional[float] = None
    model_size_kb: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    # Efficiency
    inference_latency_ms: Optional[float] = None   # median, per sample/trajectory
    latency_iqr_ms: Optional[float] = None
    throughput_per_sec: Optional[float] = None
    sampling_steps: Optional[int] = None           # NFE for generative models

    # Quality
    predictive_metrics: Dict[str, Any] = field(default_factory=dict)
    probabilistic_metrics: Dict[str, Any] = field(default_factory=dict)

    # Provenance
    environment: Dict[str, Any] = field(default_factory=_software_env)
    notes: List[str] = field(default_factory=list)
    errors: Optional[str] = None
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Flat columns surfaced in the CSV index (nested dicts are summarized/omitted).
_INDEX_COLUMNS = [
    "experiment_id", "timestamp", "stage", "hypothesis", "name", "status",
    "model_architecture", "method", "dataset", "dataset_version", "seed",
    "parameter_count", "sparsity", "model_size_kb", "peak_memory_mb",
    "inference_latency_ms", "throughput_per_sec", "sampling_steps",
]


class ExperimentRegistry:
    """Append-only store of :class:`ExperimentRecord` objects."""

    def __init__(self, root: str | Path = "experiments/registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_json = self.root / "index.json"
        self.index_csv = self.root / "index.csv"

    # ---- write -------------------------------------------------------- #
    def save(self, record: ExperimentRecord) -> Path:
        path = self.root / f"{record.experiment_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
        self._refresh_index()
        return path

    def new_id(self, prefix: str, fingerprint: str = "") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{fingerprint}" if fingerprint else ""
        return f"{ts}_{prefix}{suffix}"

    # ---- read --------------------------------------------------------- #
    def list_records(self) -> List[Dict[str, Any]]:
        records = []
        for p in sorted(self.root.glob("*.json")):
            if p.name in ("index.json",):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                continue
        return records

    def load(self, experiment_id: str) -> Dict[str, Any]:
        with open(self.root / f"{experiment_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def query(self, **filters: Any) -> List[Dict[str, Any]]:
        out = []
        for rec in self.list_records():
            if all(rec.get(k) == v for k, v in filters.items()):
                out.append(rec)
        return out

    # ---- index -------------------------------------------------------- #
    def _refresh_index(self) -> None:
        records = self.list_records()
        records.sort(key=lambda r: r.get("timestamp", ""))
        with open(self.index_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
        with open(self.index_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_INDEX_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                writer.writerow({c: rec.get(c, "") for c in _INDEX_COLUMNS})
