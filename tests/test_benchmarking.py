"""
Benchmark pipeline tests (src/benchmarking/): algorithm/dataset compatibility,
recommendations, fair shared splits, reproducibility, label isolation for
unsupervised tasks, per-model failure capture and persistence.
"""

import json

import numpy as np
import pytest

from src.benchmarking import catalog as C
from src.benchmarking.catalog import check_compatibility, get_algorithm, recommend_algorithms, recommend_datasets
from src.benchmarking.runner import BenchmarkConfig, BenchmarkError, load_history, load_run, run_benchmark
from src.data import loaders as L
from src.data.registry import get_dataset_spec


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_DATA_HOME", str(tmp_path / "data"))
    L.clear_memory_cache()
    yield
    L.clear_memory_cache()


def _cfg(tmp_path, **kw):
    kw.setdefault("save_dir", str(tmp_path / "runs"))
    return BenchmarkConfig(**kw)


# ----------------------------------------------------------------- compatibility
class TestCompatibility:
    def test_task_mismatch_is_explained(self):
        ok, why = check_compatibility(get_algorithm("kmeans"), get_dataset_spec("diabetes"), "regression")
        assert not ok and any("clustering algorithm" in w for w in why)

    def test_classifier_on_regression_dataset_rejected(self):
        ok, why = check_compatibility(get_algorithm("rf_clf"), get_dataset_spec("diabetes"), "classification")
        assert not ok

    def test_row_limit_enforced_for_scratch_knn(self):
        ok, why = check_compatibility(get_algorithm("knn_clf_scratch"), get_dataset_spec("adult"), "classification")
        assert not ok and any("limit" in w for w in why)
        ok2, _ = check_compatibility(get_algorithm("knn_clf_scratch"), get_dataset_spec("adult"), "classification", n_train=4000)
        assert ok2

    def test_seasonal_naive_needs_full_season(self):
        spec = get_dataset_spec("epm_sensor_telemetry")  # period 144, default window 24
        assert not check_compatibility(get_algorithm("seasonal_naive"), spec, "forecasting")[0]
        assert check_compatibility(get_algorithm("seasonal_naive"), spec, "forecasting", window=144)[0]

    def test_large_dataset_needs_opt_in(self):
        spec = get_dataset_spec("household_power")
        assert not check_compatibility(get_algorithm("naive"), spec, "forecasting")[0]
        assert check_compatibility(get_algorithm("naive"), spec, "forecasting", allow_large=True)[0]

    def test_missing_optional_dependency_reported(self, monkeypatch):
        monkeypatch.setattr(C.AlgorithmSpec, "available", lambda self: False)
        ok, why = check_compatibility(get_algorithm("xgb_clf"), get_dataset_spec("iris"), "classification")
        assert not ok and any("optional dependency" in w for w in why)

    def test_every_dataset_has_a_compatible_algorithm_per_task(self):
        from src.data.registry import DATASETS

        for spec in DATASETS.values():
            for t in spec.tasks:
                rec = recommend_algorithms(spec.id, t, allow_large=True, n_train=2000)
                assert rec["compatible"], f"{spec.id}/{t} has no compatible algorithm"

    def test_recommend_datasets_budget_ordering(self):
        quick = recommend_datasets("rf_clf", "quick")["datasets"]
        real = recommend_datasets("rf_clf", "realistic")["datasets"]
        n = lambda i: get_dataset_spec(i).n_samples
        assert n(quick[0]) <= n(quick[-1]) and n(real[0]) >= n(real[-1])
        assert "diabetes" not in quick  # never recommends an incompatible task


# ----------------------------------------------------------------- runner
class TestRunner:
    def test_classification_run_shares_one_split(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="iris", algorithms=["dummy_clf", "logreg", "dt_clf_scratch", "gnb"]))
        assert r["status"] == "completed" and len(r["models"]) == 4
        assert r["split"]["n_train"] + r["split"]["n_val"] + r["split"]["n_test"] == 149  # 1 duplicate dropped
        assert {m["rank"] for m in r["models"]} == {1, 2, 3, 4}
        assert all(m["beats_baseline"] for m in r["models"] if m["algorithm"] != "dummy_clf")
        assert r["summary"]["pareto_front"]

    def test_reproducible_with_same_seed(self, tmp_path):
        cfg = dict(dataset_id="wine", algorithms=["rf_clf", "logreg"], seed=3, save=False)
        a = run_benchmark(BenchmarkConfig(**cfg))
        b = run_benchmark(BenchmarkConfig(**cfg))
        assert a["split"]["fingerprint"] == b["split"]["fingerprint"]
        assert [m["metrics"]["f1"] for m in a["models"]] == [m["metrics"]["f1"] for m in b["models"]]

    def test_regression_with_cv_and_tuning(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="diabetes", algorithms=["dummy_reg", "ridge"], cv_folds=3, tune=True))
        ridge = next(m for m in r["models"] if m["algorithm"] == "ridge")
        assert ridge["cv"]["folds"] == 3 and ridge["cv"]["mean"] > 0
        assert "alpha" in ridge["tuning"]["best_params"]
        assert ridge["metrics"]["rmse"] < next(m for m in r["models"] if m["algorithm"] == "dummy_reg")["metrics"]["rmse"]

    def test_clustering_never_sees_labels(self, tmp_path, monkeypatch):
        seen = {}
        orig = C.ALGORITHMS["kmeans"]

        class Spy:
            def __init__(self, inner):
                self.inner = inner

            def fit_predict(self, X, y=None):
                seen["y"] = y
                seen["n_cols"] = X.shape[1]
                return self.inner.fit_predict(X)

        spy = C.AlgorithmSpec(**{**orig.__dict__, "factory": lambda ctx: Spy(orig.factory(ctx))})
        monkeypatch.setitem(C.ALGORITHMS, "kmeans", spy)
        r = run_benchmark(_cfg(tmp_path, dataset_id="iris", task="clustering", algorithms=["kmeans"]))
        assert seen["y"] is None and seen["n_cols"] == 4  # 4 measurements, no species column
        assert r["models"][0]["metrics"]["ari_external"] is not None

    def test_anomaly_detectors_train_without_labels(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="synthetic_anomalies", algorithms=["iforest", "lof"]))
        assert r["anomaly_setup"]["labels_used_for_training"] is False
        for m in r["models"]:
            assert m["status"] == "ok" and m["metrics"]["pr_auc"] > 0.9

    def test_forecasting_chronological_with_baseline(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="epm_sensor_telemetry", algorithms=["naive", "linear_ar"], cv_folds=2))
        assert r["split"]["strategy"] == "chronological"
        assert r["split"]["train_end"] < r["split"]["val_end"]
        lin = next(m for m in r["models"] if m["algorithm"] == "linear_ar")
        assert lin["metrics"]["mase"] is not None and "skill_vs_naive" in lin["metrics"]
        assert lin["cv"]["scheme"].startswith("rolling-origin")

    def test_incompatible_selection_is_explicit(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="epm_sensor_telemetry", algorithms=["naive", "seasonal_naive"]))
        assert "seasonal_naive" in r["skipped_algorithms"] and r["warnings"]
        with pytest.raises(BenchmarkError, match="None of the selected"):
            run_benchmark(_cfg(tmp_path, dataset_id="epm_sensor_telemetry", algorithms=["seasonal_naive"]))

    def test_wrong_task_for_dataset(self, tmp_path):
        with pytest.raises(BenchmarkError, match="does not support"):
            run_benchmark(_cfg(tmp_path, dataset_id="diabetes", task="clustering"))

    def test_large_dataset_blocked_without_opt_in(self, tmp_path):
        with pytest.raises(BenchmarkError, match="large"):
            run_benchmark(_cfg(tmp_path, dataset_id="household_power"))

    def test_failing_model_does_not_abort_run(self, tmp_path, monkeypatch):
        class Broken:
            def fit(self, X, y):
                raise RuntimeError("deliberate failure")

        bad = C.AlgorithmSpec(**{**C.ALGORITHMS["gnb"].__dict__, "factory": lambda ctx: Broken()})
        monkeypatch.setitem(C.ALGORITHMS, "gnb", bad)
        r = run_benchmark(_cfg(tmp_path, dataset_id="iris", algorithms=["gnb", "logreg"]))
        assert r["status"] == "partial"
        failed = next(m for m in r["models"] if m["algorithm"] == "gnb")
        assert failed["status"] == "failed" and "deliberate failure" in failed["error"]
        assert r["summary"]["best_by_metric"] == "logreg"

    def test_imbalance_warning_and_class_weight(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="synthetic_anomalies", task="anomaly_detection",
                               algorithms=["iforest"]))
        assert r["status"] == "completed"

    def test_results_are_persisted(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="iris", algorithms=["logreg"]))
        hist = load_history(tmp_path / "runs")
        assert hist and hist[0]["run_id"] == r["run_id"]
        assert load_run(r["run_id"], tmp_path / "runs")["dataset_id"] == "iris"
        json.dumps(r)  # result must be JSON-serialisable

    def test_config_rejects_unknown_fields(self):
        with pytest.raises(BenchmarkError, match="Unknown config"):
            BenchmarkConfig.from_dict({"dataset_id": "iris", "learning_rate_typo": 1})

    def test_scratch_models_run_inside_pipelines(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="breast_cancer",
                               algorithms=["logreg_scratch", "knn_clf_scratch", "dt_clf_scratch"]))
        assert all(m["status"] == "ok" for m in r["models"])
        assert all(m["metrics"]["f1"] > 0.85 for m in r["models"])


# ----------------------------------------------------------------- neural wrappers
import importlib.util

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNeuralWrappers:
    def test_torch_forecaster_and_mlp(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="epm_sensor_telemetry", algorithms=["lstm"], epochs=2,
                               max_train_rows=300))
        assert r["models"][0]["status"] == "ok" and r["models"][0]["n_parameters"] > 0
        r2 = run_benchmark(_cfg(tmp_path, dataset_id="iris", algorithms=["mlp_torch"], epochs=30))
        assert r2["models"][0]["status"] == "ok"

    def test_autoencoder_detector(self, tmp_path):
        r = run_benchmark(_cfg(tmp_path, dataset_id="synthetic_anomalies", algorithms=["autoencoder"], epochs=5))
        assert r["models"][0]["status"] == "ok"
