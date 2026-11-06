"""Shared fixtures: isolated settings/dirs and a few models trained once per test session."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="EPM's synthetic telemetry generator imports torch")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epm_chatbot.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Shared models trained in-process (fast) through the same ``training.train`` code the server uses."""
    from epm_chatbot.ml_service.model_store import ModelStore
    from epm_chatbot.ml_service.datasets import load_registry
    from epm_chatbot.ml_service.training import serialize, train

    d = tmp_path_factory.mktemp("models")
    store = ModelStore(d)
    tmp = d / "_tmp"
    tmp.mkdir()
    specs = [("iris", "classification", "rf_clf", None), ("diabetes", "regression", "ridge", None)]
    if HAS_TORCH:
        specs.append(("epm_sensor_telemetry", "forecasting", "linear_ar", 3))
    for ds_id, task, algo, horizon in specs:
        ds = load_registry(ds_id, False)
        model, meta = train(ds, task, algo, max_train_rows=20000, allow_neural=False, epochs=5, horizon=horizon)
        art = tmp / f"{ds_id}.bin"
        art.write_bytes(serialize(model))
        store.save(art, meta, owner=None)
    return d


@pytest.fixture()
def settings(tmp_path, model_dir):
    import shutil

    md = tmp_path / "models"
    shutil.copytree(model_dir, md)
    s = Settings(host="127.0.0.1", port=0, allowed_origins=("https://partner.example",),
                 secret_key="x" * 40, model_dir=md, upload_dir=tmp_path / "uploads", train_timeout_s=120,
                 rate_per_min=1000, rate_ip_per_min=5000, train_per_hour=50)
    s.validate()
    return s


@pytest.fixture()
def app(settings):
    from epm_chatbot.backend.server import App

    return App(settings)


@pytest.fixture()
def controller(app):
    return app.controller


def model_card(store, dataset_id):
    return next(c for c in store.list(owner=None) if c["dataset_id"] == dataset_id)
