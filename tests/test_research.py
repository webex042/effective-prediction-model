"""
Unit Tests for Stage 13 — Research Experiment Pipeline
======================================================
Covers config validation, leakage-free splits, reproducibility, metric math,
the shared experiment registry, report generation, and an end-to-end run.
"""

import math
import numpy as np
import pytest

from src.research.config import ResearchConfig
from src.research import metrics as M
from src.research.registry import ExperimentRegistry, ExperimentRecord
from src.research.pipeline import prepare_data, train_baseline
from src.research.report import generate_report


class TestConfig:
    def test_valid_default(self):
        cfg = ResearchConfig().validate()
        assert cfg.hypothesis == "H1"
        assert 0 < cfg.train_ratio + cfg.val_ratio < 1

    def test_split_ratio_overflow_rejected(self):
        with pytest.raises(ValueError):
            ResearchConfig(train_ratio=0.9, val_ratio=0.2).validate()

    def test_sampling_steps_bounded_by_timesteps(self):
        with pytest.raises(ValueError):
            ResearchConfig(diffusion_timesteps=50, sampling_steps=[80]).validate()

    def test_mc_samples_minimum(self):
        with pytest.raises(ValueError):
            ResearchConfig(n_mc_samples=1).validate()

    def test_fingerprint_is_deterministic_and_sensitive(self):
        a = ResearchConfig.quick()
        b = ResearchConfig.quick()
        c = ResearchConfig.quick(epochs=99)
        assert a.fingerprint() == b.fingerprint()
        assert a.fingerprint() != c.fingerprint()

    def test_roundtrip_dict(self):
        cfg = ResearchConfig.quick()
        assert ResearchConfig.from_dict(cfg.to_dict()).fingerprint() == cfg.fingerprint()


class TestMetrics:
    def test_mae_rmse_known(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = np.array([[1.0, 2.0], [3.0, 6.0]])
        assert M.mae(y, p) == pytest.approx(0.5)
        assert M.rmse(y, p) == pytest.approx(math.sqrt(1.0))

    def test_crps_zero_for_perfect_degenerate_ensemble(self):
        y = np.zeros((4, 3))
        samples = np.zeros((4, 5, 3))  # all mass on the truth
        assert M.crps_ensemble(y, samples) == pytest.approx(0.0, abs=1e-9)

    def test_crps_nonnegative(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(6, 2))
        samples = rng.normal(size=(6, 8, 2))
        assert M.crps_ensemble(y, samples) >= 0.0

    def test_interval_coverage(self):
        y = np.array([[0.5, 0.5]])
        lo = np.array([[0.0, 0.0]]); hi = np.array([[1.0, 1.0]])
        picp, mpiw = M.interval_coverage(y, lo, hi)
        assert picp == pytest.approx(1.0)
        assert mpiw == pytest.approx(1.0)

    def test_bootstrap_ci_constant(self):
        lo, hi = M.bootstrap_ci([2.0, 2.0, 2.0], resamples=50)
        assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)

    def test_aggregate_seeds_structure(self):
        agg = M.aggregate_seeds([{"rmse": 1.0}, {"rmse": 3.0}], resamples=100)
        assert agg["rmse"]["mean"] == pytest.approx(2.0)
        assert agg["rmse"]["n"] == 2
        assert set(agg["rmse"]) == {"mean", "std", "ci_low", "ci_high", "n"}

    def test_measure_latency_keys(self):
        out = M.measure_latency(lambda: sum(range(100)), warmup=1, reps=5, n_items=10)
        assert out["latency_ms_median"] >= 0.0
        assert out["throughput_per_sec"] >= 0.0

class TestDataIntegrity:
    def test_split_is_chronological_and_complete(self):
        cfg = ResearchConfig.quick()
        data = prepare_data(cfg, seed=0)
        total = len(data.X_train) + len(data.X_val) + len(data.X_test)
        assert total > 0
        # train fraction is close to the configured ratio (chronological, no shuffle)
        assert abs(len(data.X_train) / total - cfg.train_ratio) < 0.05
        assert data.X_train.shape[1:] == (cfg.window_size, data.in_channels)
        assert data.y_test.shape[1] == cfg.horizon

    def test_scaler_fit_on_train_only(self):
        cfg = ResearchConfig.quick()
        data = prepare_data(cfg, seed=0)
        # Train inputs are standardized → ~0 mean / ~1 std per channel.
        flat = data.X_train.reshape(-1, data.in_channels)
        assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-3)
        assert np.allclose(flat.std(axis=0), 1.0, atol=1e-2)

    def test_reproducible_same_seed(self):
        cfg = ResearchConfig.quick()
        d1 = prepare_data(cfg, seed=0)
        d2 = prepare_data(cfg, seed=0)
        assert np.array_equal(d1.X_train, d2.X_train)
        assert np.array_equal(d1.y_test, d2.y_test)

    def test_unknown_baseline_rejected(self):
        cfg = ResearchConfig.quick()
        data = prepare_data(cfg, seed=0)
        with pytest.raises(ValueError):
            train_baseline("does_not_exist", cfg, data, seed=0)


class TestRegistry:
    def test_save_load_list_query_and_csv(self, tmp_path):
        reg = ExperimentRegistry(root=tmp_path / "reg")
        rec = ExperimentRecord(
            experiment_id=reg.new_id("s13_test", "abcd1234"),
            name="unit_test", stage=13, hypothesis="H1",
            model_architecture="ts_diffusion", method="ddim_10steps",
            sampling_steps=10, parameter_count=1234,
            predictive_metrics={"rmse": {"mean": 1.0}},
        )
        path = reg.save(rec)
        assert path.exists()
        loaded = reg.load(rec.experiment_id)
        assert loaded["name"] == "unit_test"
        assert loaded["sampling_steps"] == 10
        assert len(reg.list_records()) == 1
        assert reg.query(stage=13) and not reg.query(stage=99)
        assert (tmp_path / "reg" / "index.csv").exists()
        assert (tmp_path / "reg" / "index.json").exists()

    def test_record_captures_environment(self, tmp_path):
        reg = ExperimentRegistry(root=tmp_path / "reg")
        rec = ExperimentRecord(experiment_id="x", name="n", stage=13)
        assert "python" in rec.environment
        assert "cuda_available" in rec.environment  # hardware availability recorded


class TestReport:
    def test_report_sections_present(self):
        summary = {
            "config": ResearchConfig.quick().to_dict(),
            "fingerprint": "deadbeef",
            "split": {"train": 10, "val": 2, "test": 3},
            "baselines": {"linear": {"metrics": {"mae": {"mean": 1.0, "std": 0.1,
                           "ci_low": 0.9, "ci_high": 1.1}, "rmse": {"mean": 1.2, "std": 0.1,
                           "ci_low": 1.1, "ci_high": 1.3}, "latency_ms": {"mean": 0.01,
                           "std": 0.0, "ci_low": 0.01, "ci_high": 0.01}},
                           "parameter_count": 100, "model_size_kb": 0.4}},
            "diffusion": {10: {"metrics": {"rmse": {"mean": 1.5, "std": 0.1, "ci_low": 1.4,
                           "ci_high": 1.6}, "crps": {"mean": 0.8, "std": 0.1, "ci_low": 0.7,
                           "ci_high": 0.9}, "picp_90": {"mean": 0.9, "std": 0.0, "ci_low": 0.9,
                           "ci_high": 0.9}, "mpiw_90": {"mean": 2.0, "std": 0.1, "ci_low": 1.9,
                           "ci_high": 2.1}, "latency_ms": {"mean": 5.0, "std": 0.2,
                           "ci_low": 4.8, "ci_high": 5.2}}, "parameter_count": 5000,
                           "model_size_kb": 20.0, "sampling_steps": 10}},
            "h1_analysis": {"full_steps": 10, "full_rmse_mean": 1.5,
                            "full_latency_ms_mean": 5.0, "comparisons": []},
            "record_ids": ["id1"], "environment": {"python": "3.13", "torch": "x",
            "device_name": "cpu"}, "seeds": [0],
        }
        report = generate_report(summary)
        for section in ("# Stage 13 Research Report", "## 1. Research Question",
                        "## 3. Baselines", "## 4. Diffusion NFE Sweep",
                        "## 6. Interpretation", "## 8. Reproduction"):
            assert section in report


@pytest.mark.slow
class TestEndToEnd:
    def test_quick_experiment_runs(self, tmp_path):
        from src.research.pipeline import run_experiment
        cfg = ResearchConfig.quick()
        reg = ExperimentRegistry(root=tmp_path / "reg")
        summary = run_experiment(cfg, registry=reg, verbose=False)
        # one record per baseline + one per sampling-step setting
        assert len(summary["record_ids"]) == len(cfg.baselines) + len(cfg.sampling_steps)
        assert "rmse" in next(iter(summary["baselines"].values()))["metrics"]
        for s in cfg.sampling_steps:
            assert "crps" in summary["diffusion"][s]["metrics"]
        assert len(reg.list_records()) == len(summary["record_ids"])
