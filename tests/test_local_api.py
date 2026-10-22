"""
Local experiment API (src/webapi/local_api.py) and the unchanged read-only
artifact API (api/index.py), exercised through their router functions.
"""

import base64
import importlib.util
import time
from pathlib import Path

import pandas as pd
import pytest

from src.data import loaders as L
from src.webapi import local_api as A

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_DATA_HOME", str(tmp_path / "data"))
    import src.benchmarking.runner as R

    monkeypatch.setattr(R, "default_save_dir", lambda: tmp_path / "runs")
    L.clear_memory_cache()
    yield


def _wait(job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, j = A.route("GET", f"/api/jobs/{job_id}", {}, {})
        assert status == 200
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_capabilities():
    status, body = A.route("GET", "/api/capabilities", {}, {})
    assert status == 200 and body["training"] is True


def test_dataset_catalogue_filtered_by_task():
    status, body = A.route("GET", "/api/datasets", {"task": "forecasting"}, {})
    assert status == 200
    ids = {d["id"] for d in body["datasets"]}
    assert {"air_quality", "epm_sensor_telemetry"} <= ids and "iris" not in ids
    entry = next(d for d in body["datasets"] if d["id"] == "air_quality")
    assert entry["status"]["state"] == "not_downloaded" and entry["license"]


def test_algorithms_and_compatibility():
    s, body = A.route("GET", "/api/algorithms", {"task": "clustering"}, {})
    assert s == 200 and {a["task"] for a in body["algorithms"]} == {"clustering"}
    s, rec = A.route("GET", "/api/compatibility", {"dataset": "iris", "task": "classification"}, {})
    assert s == 200 and "logreg" in rec["recommended"]


def test_recommend_endpoint():
    s, body = A.route("GET", "/api/recommend", {"algorithm": "kmeans", "budget": "quick"}, {})
    assert s == 200 and body["datasets"]


def test_preview_bundled_dataset():
    s, body = A.route("GET", "/api/datasets/iris/preview", {"rows": "5"}, {})
    assert s == 200 and len(body["rows"]) == 5 and body["n_rows"] == 150
    assert any(c["name"] == "sepal_length" for c in body["column_stats"])


def test_preview_refuses_undownloaded_dataset():
    s, body = A.route("GET", "/api/datasets/seeds/preview", {}, {})
    assert s == 409 and "not downloaded" in body["error"]


def test_unknown_dataset_is_404():
    s, body = A.route("GET", "/api/datasets/nope", {}, {})
    assert s == 404


def test_run_benchmark_job_completes():
    s, job = A.route("POST", "/api/benchmarks/run", {}, {"config": {"dataset_id": "iris", "algorithms": ["logreg", "gnb"]}})
    assert s == 202
    j = _wait(job["id"])
    assert j["status"] == "done" and j["result"]["status"] == "completed"
    s, hist = A.route("GET", "/api/benchmarks/history", {}, {})
    assert hist["runs"][0]["run_id"] == j["result"]["run_id"]


def test_invalid_config_is_400():
    s, body = A.route("POST", "/api/benchmarks/run", {}, {"config": {"dataset_id": "iris", "cv_folds": 99}})
    assert s == 400
    s, body = A.route("POST", "/api/benchmarks/run", {}, {"config": {"dataset_id": "iris", "bogus": 1}})
    assert s == 400
    s, body = A.route("POST", "/api/benchmarks/run", {}, {"config": {"dataset_id": "diabetes", "task": "clustering"}})
    assert s == 400


def test_incompatible_run_reports_error_in_job():
    s, job = A.route("POST", "/api/benchmarks/run", {}, {"config": {"dataset_id": "iris", "algorithms": ["ridge"]}})
    j = _wait(job["id"])
    assert j["status"] == "error" and "compatible" in j["error"]


def test_upload_roundtrip():
    df = pd.DataFrame({"a": range(40), "b": [i % 3 for i in range(40)], "label": ["p", "q"] * 20})
    payload = {"filename": "mine.csv", "task": "classification", "target": "label",
               "content_b64": base64.b64encode(df.to_csv(index=False).encode()).decode()}
    s, meta = A.route("POST", "/api/datasets/upload", {}, payload)
    assert s == 201 and meta["id"].startswith("upload_")
    s, body = A.route("GET", "/api/datasets", {"task": "classification"}, {})
    assert meta["id"] in {d["id"] for d in body["datasets"]}


def test_upload_rejects_bad_payload():
    s, body = A.route("POST", "/api/datasets/upload", {}, {"task": "regression", "content_b64": "!!notbase64"})
    assert s == 400


def test_parse_body_requires_json():
    with pytest.raises(ValueError):
        A.parse_body(b"a=b", "application/x-www-form-urlencoded")
    assert A.parse_body(b'{"x": 1}', "application/json") == {"x": 1}


def test_unmatched_routes_fall_through():
    assert A.route("GET", "/api/health", {}, {}) is None
    assert A.route("GET", "/index.html", {}, {}) is None


def test_artifact_api_still_serves_existing_routes():
    spec = importlib.util.spec_from_file_location("epm_artifact_api", ROOT / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    status, info = mod.route_request("GET", "/api/health", {}, {})
    assert status == 200 and info["stages_total"] == 14
    status, _ = mod.route_request("GET", "/api/datasets", {}, {})
    assert status == 200
    status, body = mod.route_request("POST", "/api/benchmarks/run", {}, {})
    assert status == 501 and "locally" in body["how_to_run"]
