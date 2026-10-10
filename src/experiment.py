"""
EPM Experiment Tracker
======================

A lightweight experiment tracking system that logs results to JSON files.

WHY NOT USE MLFLOW/WANDB?
    Those are great tools, but at Stage 0 we want:
    1. Zero external dependencies (beyond what we already have)
    2. Full understanding of what experiment tracking DOES
    3. Simple, version-controllable JSON files
    4. Easy to inspect and compare without a server

    Later stages can integrate with MLflow/W&B if desired.

HOW IT WORKS:
    1. Start an experiment with a name and config
    2. Log metrics as the experiment runs
    3. End the experiment to save everything to a JSON file
    4. Compare experiments by loading multiple JSON files

USAGE:
    from src.experiment import Experiment

    exp = Experiment(
        name="linear_regression_baseline",
        config=config,
        tags=["stage2", "baseline", "regression"]
    )

    exp.log_params({"learning_rate": 0.01, "epochs": 100})
    exp.log_metric("train_loss", 0.542, step=1)
    exp.log_metric("train_loss", 0.321, step=2)
    exp.log_metrics({"val_loss": 0.45, "val_accuracy": 0.87})

    exp.end()  # Saves to experiments/<timestamp>_<name>.json
"""

from __future__ import annotations

import json
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.logger import get_logger
from src.reproducibility import get_environment_info

logger = get_logger(__name__)


class Experiment:
    """
    A single experiment run with metric tracking.

    Captures:
    - Configuration / hyperparameters
    - Metrics over time (with optional step tracking)
    - Timing information
    - Memory usage
    - Environment details
    - Custom notes / tags

    Parameters
    ----------
    name : str
        A descriptive name for this experiment (e.g., "lr_baseline_v1").
    config : dict or Config, optional
        The configuration used for this experiment.
    save_dir : str or Path
        Directory to save experiment results.
    tags : list of str, optional
        Tags for categorizing experiments (e.g., ["baseline", "stage2"]).

    Examples
    --------
    >>> exp = Experiment("test_run", tags=["debug"])
    >>> exp.log_metric("accuracy", 0.95)
    >>> exp.log_metric("loss", 0.05)
    >>> exp.end()
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        save_dir: str | Path = "experiments",
        tags: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self.save_dir = Path(save_dir)
        self.tags = tags or []

        # Timing
        self.start_time = time.time()
        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[float] = None

        # Data storage
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, list[dict[str, Any]]] = {}
        self.notes: list[str] = []

        # Config handling
        if config is not None:
            if hasattr(config, "to_dict"):
                self.config_dict = config.to_dict()
            elif isinstance(config, dict):
                self.config_dict = config
            else:
                self.config_dict = {}
        else:
            self.config_dict = {}

        # Environment
        self.environment = get_environment_info()

        # Memory tracking
        self._track_memory = False
        self.peak_memory_mb: Optional[float] = None

        # Status
        self._ended = False

        logger.info(f"Experiment '{name}' started")

    # ----- Logging Methods -----

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self.params[key] = value
        logger.debug(f"[{self.name}] param: {key}={value}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple hyperparameters at once."""
        self.params.update(params)
        logger.debug(f"[{self.name}] params: {params}")

    def log_metric(
        self, name: str, value: float, step: Optional[int] = None
    ) -> None:
        """
        Log a single metric value.

        Parameters
        ----------
        name : str
            Metric name (e.g., "train_loss", "val_accuracy").
        value : float
            The metric value.
        step : int, optional
            Training step or epoch number. If None, only the value is recorded.
        """
        if name not in self.metrics:
            self.metrics[name] = []

        entry: dict[str, Any] = {
            "value": value,
            "timestamp": time.time() - self.start_time,
        }
        if step is not None:
            entry["step"] = step

        self.metrics[name].append(entry)

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_note(self, note: str) -> None:
        """Add a text note to the experiment."""
        self.notes.append(note)
        logger.info(f"[{self.name}] note: {note}")

    # ----- Memory Tracking -----

    def start_memory_tracking(self) -> None:
        """Start tracking peak memory usage."""
        tracemalloc.start()
        self._track_memory = True
        logger.debug(f"[{self.name}] Memory tracking started")

    def stop_memory_tracking(self) -> None:
        """Stop tracking and record peak memory."""
        if self._track_memory:
            _, peak = tracemalloc.get_traced_memory()
            self.peak_memory_mb = peak / 1024 / 1024  # Convert to MB
            tracemalloc.stop()
            self._track_memory = False
            logger.debug(
                f"[{self.name}] Peak memory: {self.peak_memory_mb:.2f} MB"
            )

    # ----- Summary -----

    def get_final_metrics(self) -> dict[str, float]:
        """Get the last recorded value for each metric."""
        final = {}
        for name, values in self.metrics.items():
            if values:
                final[name] = values[-1]["value"]
        return final

    def get_best_metric(self, name: str, mode: str = "min") -> Optional[float]:
        """
        Get the best (min or max) value for a metric.

        Parameters
        ----------
        name : str
            Metric name.
        mode : str
            "min" for loss-like metrics, "max" for accuracy-like metrics.

        Returns
        -------
        float or None
            The best metric value, or None if no values logged.
        """
        if name not in self.metrics or not self.metrics[name]:
            return None
        values = [entry["value"] for entry in self.metrics[name]]
        return min(values) if mode == "min" else max(values)

    # ----- Save / Load -----

    def end(self) -> Path:
        """
        End the experiment and save results to a JSON file.

        Returns
        -------
        Path
            Path to the saved experiment file.
        """
        if self._ended:
            logger.warning(f"[{self.name}] Experiment already ended")
            return self._save_path

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if self._track_memory:
            self.stop_memory_tracking()

        # Build the experiment record
        record = {
            "name": self.name,
            "tags": self.tags,
            "start_time": self.start_timestamp,
            "duration_seconds": round(duration, 3),
            "config": self.config_dict,
            "params": self.params,
            "metrics": self.metrics,
            "final_metrics": self.get_final_metrics(),
            "peak_memory_mb": self.peak_memory_mb,
            "notes": self.notes,
            "environment": self.environment,
        }

        # Save
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.name}.json"
        self._save_path = self.save_dir / filename

        with open(self._save_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

        self._ended = True
        logger.info(
            f"Experiment '{self.name}' ended "
            f"(duration={duration:.1f}s, saved to {self._save_path})"
        )

        return self._save_path

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """
        Load a saved experiment from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the experiment JSON file.

        Returns
        -------
        dict
            The full experiment record.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# =============================================================================
# Experiment Comparison
# =============================================================================

def compare_experiments(
    paths: list[str | Path],
    metric_names: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Compare multiple experiments side-by-side.

    Parameters
    ----------
    paths : list of str or Path
        Paths to experiment JSON files.
    metric_names : list of str, optional
        Specific metrics to compare. If None, uses all metrics.

    Returns
    -------
    list of dict
        Summary for each experiment with name, params, and final metrics.

    Examples
    --------
    >>> results = compare_experiments([
    ...     "experiments/lr_baseline.json",
    ...     "experiments/rf_baseline.json",
    ... ])
    >>> for r in results:
    ...     print(f"{r['name']}: accuracy={r['metrics'].get('accuracy')}")
    """
    summaries = []
    for path in paths:
        exp = Experiment.load(path)
        summary: dict[str, Any] = {
            "name": exp["name"],
            "tags": exp.get("tags", []),
            "duration_seconds": exp.get("duration_seconds"),
            "params": exp.get("params", {}),
        }

        final = exp.get("final_metrics", {})
        if metric_names:
            summary["metrics"] = {k: final.get(k) for k in metric_names}
        else:
            summary["metrics"] = final

        summaries.append(summary)

    return summaries
