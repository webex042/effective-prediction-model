"""
Tests for the EPM Experiment Tracker
=====================================

These tests verify that:
1. Experiment records metrics correctly
2. Metric history is preserved with timestamps
3. Experiments save to JSON files
4. Saved experiments can be reloaded
5. Experiment comparison works
"""

import json
from pathlib import Path

import pytest

from src.experiment import Experiment, compare_experiments


class TestExperiment:
    """Tests for the Experiment class."""

    def test_creation(self):
        """Experiment should initialize with correct name and empty metrics."""
        exp = Experiment("test_exp", save_dir="tests/tmp_experiments")
        assert exp.name == "test_exp"
        assert exp.metrics == {}
        assert exp.params == {}

    def test_log_metric(self):
        """Single metrics should be recorded with timestamps."""
        exp = Experiment("test_exp", save_dir="tests/tmp_experiments")
        exp.log_metric("loss", 0.5, step=1)
        exp.log_metric("loss", 0.3, step=2)

        assert "loss" in exp.metrics
        assert len(exp.metrics["loss"]) == 2
        assert exp.metrics["loss"][0]["value"] == 0.5
        assert exp.metrics["loss"][1]["value"] == 0.3
        assert exp.metrics["loss"][0]["step"] == 1

    def test_log_metrics_batch(self):
        """Multiple metrics logged at once should all be recorded."""
        exp = Experiment("test_exp", save_dir="tests/tmp_experiments")
        exp.log_metrics({"acc": 0.9, "loss": 0.1}, step=5)

        assert "acc" in exp.metrics
        assert "loss" in exp.metrics
        assert exp.metrics["acc"][0]["value"] == 0.9

    def test_log_params(self):
        """Parameters should be stored in the params dict."""
        exp = Experiment("test_exp", save_dir="tests/tmp_experiments")
        exp.log_params({"lr": 0.01, "epochs": 100})
        assert exp.params["lr"] == 0.01
        assert exp.params["epochs"] == 100

    def test_get_final_metrics(self):
        """Final metrics should return the last value for each metric."""
        exp = Experiment("test_exp", save_dir="tests/tmp_experiments")
        exp.log_metric("loss", 0.5, step=1)
        exp.log_metric("loss", 0.3, step=2)
        exp.log_metric("loss", 0.1, step=3)
        exp.log_metric("acc", 0.95)

        final = exp.get_final_metrics()
        assert final["loss"] == 0.1
        assert final["acc"] == 0.95

    def test_get_best_metric(self):
        """Best metric should respect min/max mode."""
        exp = Experiment("test_exp", save_dir="tests/tmp_experiments")
        exp.log_metric("loss", 0.5)
        exp.log_metric("loss", 0.1)
        exp.log_metric("loss", 0.3)
        exp.log_metric("acc", 0.7)
        exp.log_metric("acc", 0.95)
        exp.log_metric("acc", 0.9)

        assert exp.get_best_metric("loss", mode="min") == 0.1
        assert exp.get_best_metric("acc", mode="max") == 0.95

    def test_save_and_load(self, tmp_path):
        """Experiment should save to JSON and be reloadable."""
        exp = Experiment("save_test", save_dir=tmp_path)
        exp.log_params({"lr": 0.01})
        exp.log_metric("loss", 0.5, step=1)
        exp.log_metric("acc", 0.9)
        exp.log_note("This is a test experiment")

        save_path = exp.end()

        # Verify file exists
        assert save_path.exists()
        assert save_path.suffix == ".json"

        # Reload and verify contents
        loaded = Experiment.load(save_path)
        assert loaded["name"] == "save_test"
        assert loaded["params"]["lr"] == 0.01
        assert loaded["final_metrics"]["loss"] == 0.5
        assert loaded["final_metrics"]["acc"] == 0.9
        assert "duration_seconds" in loaded
        assert loaded["duration_seconds"] >= 0

    def test_end_idempotent(self, tmp_path):
        """Calling end() twice should not crash."""
        exp = Experiment("idempotent_test", save_dir=tmp_path)
        exp.log_metric("x", 1.0)
        path1 = exp.end()
        path2 = exp.end()  # Should warn but not crash
        assert path1 == path2

    def test_config_from_dict(self, tmp_path):
        """Config passed as dict should be stored."""
        cfg = {"lr": 0.01, "batch_size": 32}
        exp = Experiment("cfg_test", config=cfg, save_dir=tmp_path)
        path = exp.end()
        loaded = Experiment.load(path)
        assert loaded["config"]["lr"] == 0.01

    def test_tags(self, tmp_path):
        """Tags should be preserved in saved experiments."""
        exp = Experiment(
            "tagged", tags=["baseline", "stage2"], save_dir=tmp_path
        )
        path = exp.end()
        loaded = Experiment.load(path)
        assert "baseline" in loaded["tags"]
        assert "stage2" in loaded["tags"]


class TestCompareExperiments:
    """Tests for comparing experiments."""

    def test_compare_two_experiments(self, tmp_path):
        """Should produce summaries for each experiment."""
        # Create two experiments
        exp1 = Experiment("model_a", save_dir=tmp_path)
        exp1.log_metric("acc", 0.85)
        path1 = exp1.end()

        exp2 = Experiment("model_b", save_dir=tmp_path)
        exp2.log_metric("acc", 0.92)
        path2 = exp2.end()

        results = compare_experiments([path1, path2])
        assert len(results) == 2
        assert results[0]["name"] == "model_a"
        assert results[1]["name"] == "model_b"
        assert results[0]["metrics"]["acc"] == 0.85
        assert results[1]["metrics"]["acc"] == 0.92

    def test_compare_with_metric_filter(self, tmp_path):
        """Should filter to only requested metrics."""
        exp = Experiment("filtered", save_dir=tmp_path)
        exp.log_metric("acc", 0.9)
        exp.log_metric("loss", 0.1)
        exp.log_metric("f1", 0.88)
        path = exp.end()

        results = compare_experiments([path], metric_names=["acc", "f1"])
        assert "acc" in results[0]["metrics"]
        assert "f1" in results[0]["metrics"]
        assert "loss" not in results[0]["metrics"]
