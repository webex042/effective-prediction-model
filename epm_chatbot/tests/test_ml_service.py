"""Model adapters: training validity, leakage, persistence integrity, inference, explanations."""

import numpy as np
import pandas as pd
import pytest

from epm_chatbot.ml_service import inference as I
from epm_chatbot.ml_service.datasets import DataAccessError, UploadStore, load_registry
from epm_chatbot.ml_service.model_store import ModelStore, ModelStoreError
from epm_chatbot.ml_service.training import TrainingError, check_trainable, train
from epm_chatbot.tests.conftest import model_card, needs_torch


@pytest.fixture()
def store(settings):
    return ModelStore(settings.model_dir)


def test_cards_hold_heldout_metrics_and_schema(store):
    c = model_card(store, "iris")
    assert c["task"] == "classification" and c["target"]["classes"] == ["setosa", "versicolor", "virginica"]
    assert 0.8 < c["metrics"]["accuracy"] <= 1
    assert c["split"]["n_test"] > 0 and c["split"]["n_train"] > c["split"]["n_test"]
    assert [f["name"] for f in c["features"]] == c["feature_order"]
    assert c["importance"]["items"] and "held-out" in c["importance"]["method"]
    r = model_card(store, "diabetes")
    assert r["conformal"]["level"] == 0.9 and r["conformal"]["halfwidth"] > 0


def test_preprocessing_fitted_on_training_rows_only(store):
    """The saved imputer's medians equal the TRAIN-split medians recorded in the card, not the full-data ones."""
    c = model_card(store, "diabetes")
    pipe = store.load(c["id"], None)
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    card_medians = np.array([f["median"] for f in c["features"]])
    assert np.allclose(imputer.statistics_, card_medians, atol=1e-6)
    full = load_registry("diabetes", False).features.median().to_numpy()
    assert not np.allclose(imputer.statistics_, full, atol=1e-9)   # proves the test split was excluded


@needs_torch
def test_forecaster_scaler_from_training_segment(store):
    c = model_card(store, "epm_sensor_telemetry")
    fc = c["forecast"]
    ds = load_registry("epm_sensor_telemetry", False)
    tr_end = c["split"]["train_end"]
    expected = ds.features[fc["channels"]].iloc[:tr_end].mean().to_numpy()
    assert np.allclose(fc["channel_mean"], expected, rtol=1e-6)
    assert c["split"]["strategy"] == "chronological"


def test_artifact_integrity_check(store):
    c = model_card(store, "iris")
    p = store.root / f"{c['id']}.joblib"
    data = bytearray(p.read_bytes())
    data[-10] ^= 0xFF
    p.write_bytes(bytes(data))
    store._cache.clear()
    with pytest.raises(ModelStoreError, match="integrity"):
        store.load(c["id"], None)


def test_private_models_are_owner_scoped(store, tmp_path):
    c = model_card(store, "iris")
    art = tmp_path / "copy.bin"
    art.write_bytes((store.root / f"{c['id']}.joblib").read_bytes())
    meta = {k: v for k, v in c.items() if k not in ("id", "owner", "created_at", "artifact_sha256", "artifact_kb")}
    mine = store.save(art, meta, owner="alice")
    assert mine["id"] in {x["id"] for x in store.list(owner="alice")}
    assert mine["id"] not in {x["id"] for x in store.list(owner="bob")}
    with pytest.raises(ModelStoreError):
        store.card(mine["id"], "bob")
    assert store.delete_owner("alice") == 1


def test_parse_feature_values_forms(store):
    c = model_card(store, "iris")
    v, _ = I.parse_feature_values("predict sepal length 5.1, sepal_width=3.5, petal length is 1.4 and petal width: 0.2", c)
    assert v == {"sepal_length": "5.1", "sepal_width": "3.5", "petal_length": "1.4", "petal_width": "0.2"}
    v, notes = I.parse_feature_values("5.1 3.5 1.4 0.2", c)
    assert len(v) == 4 and notes
    v, _ = I.parse_feature_values('{"sepal_length": 5, "petal_width": 1}', c)
    assert v == {"sepal_length": 5, "petal_width": 1}
    v, _ = I.parse_feature_values("sepal_length=abc, petal_lenght=2", c)
    assert v == {"sepal_length": "abc", "petal_lenght": "2"}


def test_validate_row_errors_and_warnings(store):
    c = model_card(store, "iris")
    with pytest.raises(I.InputError) as e:
        I.validate_row({"sepal_length": "abc", "petal_lenght": 2}, c)
    msg = " ".join(e.value.details)
    assert "must be a number" in msg and "did you mean 'petal_length'" in msg
    with pytest.raises(I.InputError) as e:
        I.validate_row({"sepal_length": 5}, c)
    assert set(e.value.missing) == {"sepal_width", "petal_length", "petal_width"}
    X, warns = I.validate_row({"sepal_length": 50, "sepal_width": 3, "petal_length": 1, "petal_width": 0.2}, c)
    assert any("outside the training range" in w for w in warns)
    X, warns = I.validate_row({"sepal_length": 5}, c, fill_missing=True)
    assert any("imputer" in w for w in warns) and X.isna().sum().sum() == 3


def test_prediction_matches_pipeline_and_is_deterministic(store):
    c = model_card(store, "iris")
    m = store.load(c["id"], None)
    X, _ = I.validate_row({"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}, c)
    r1, r2 = I.predict_rows(m, c, X), I.predict_rows(m, c, X)
    assert r1 == r2
    assert r1["labels"][0] == c["target"]["classes"][int(m.predict(X)[0])]
    assert abs(sum(r1["probabilities"][0].values()) - 1) < 1e-6


def test_regression_interval_and_explanation(store):
    c = model_card(store, "diabetes")
    m = store.load(c["id"], None)
    X, _ = I.validate_row({f["name"]: f["median"] for f in c["features"]}, c)
    r = I.predict_rows(m, c, X)
    lo, hi = r["intervals"][0]
    assert lo < r["values"][0] < hi and hi - lo == pytest.approx(2 * c["conformal"]["halfwidth"], rel=1e-4)
    X2 = X.copy()
    X2["bmi"] = c["features"][[f["name"] for f in c["features"]].index("bmi")]["max"]
    e = I.local_explanation(m, c, X2)
    top = e["items"][0]
    assert top["feature"] == "bmi" and top["effect"] != 0
    # effect equals the real change in prediction when bmi is reset to its median
    base = float(m.predict(X2)[0]); reset = X2.copy(); reset["bmi"] = top["baseline"]
    assert top["effect"] == pytest.approx(base - float(m.predict(reset)[0]), rel=1e-5)


@needs_torch
def test_forecast_next(store):
    c = model_card(store, "epm_sensor_telemetry")
    m = store.load(c["id"], None)
    ds = load_registry("epm_sensor_telemetry", False)
    out = I.forecast_next(m, c, ds.features)
    assert len(out["values"]) == 3 and len(out["intervals"]) == 3
    with pytest.raises(I.InputError):
        I.forecast_next(m, c, ds.features.head(5))


def test_batch_validation(store):
    c = model_card(store, "iris")
    df = load_registry("iris", False).features.head(10).copy()
    X, warns = I.validate_batch(df.assign(extra=1), c)
    assert list(X.columns) == c["feature_order"] and any("Ignored" in w for w in warns)
    with pytest.raises(I.InputError):
        I.validate_batch(df.drop(columns=["petal_width"]), c)


def test_training_refuses_incompatible_or_baseline():
    ds = load_registry("iris", False)
    with pytest.raises(TrainingError, match="regression"):
        check_trainable("ridge", ds, "classification", 1000, True)
    with pytest.raises(TrainingError, match="baseline"):
        check_trainable("dummy_clf", ds, "classification", 1000, True)
    with pytest.raises(TrainingError, match="disabled"):
        check_trainable("mlp_torch", ds, "classification", 1000, False)


def test_large_and_uncached_datasets_refused():
    with pytest.raises(DataAccessError, match="large"):
        load_registry("credit_card_fraud", False)


def test_upload_store_validation_and_cleanup(tmp_path):
    st = UploadStore(tmp_path, ttl_s=60, max_bytes=10_000)
    with pytest.raises(DataAccessError, match="binary"):
        st.save("s1", b"\x00\x01abc", "x.csv")
    with pytest.raises(DataAccessError, match=".csv"):
        st.save("s1", b"a,b\n1,2\n", "x.exe")
    with pytest.raises(DataAccessError, match="limit"):
        st.save("s1", b"a" * 20_000, "x.csv")
    rows = "\n".join(f"{i},{i * 2},{'a' if i % 2 else 'b'}" for i in range(40))
    meta = st.save("s1", ("x,y,=cmd\n" + rows).encode(), "data.csv")
    names = [c["name"] for c in meta["columns"]]
    assert not any(n.startswith(("=", "+", "-", "@")) for n in names)   # formula-injection neutralised
    with pytest.raises(DataAccessError):
        st.meta("s2", meta["id"])                # other sessions cannot see it
    ds = st.as_dataset("s1", meta["id"], "regression", "y")
    assert ds.target.name == "y" and "y" not in ds.features.columns
    assert st.sweep(now=meta["created"] + 61) >= 1
    with pytest.raises(DataAccessError, match="expired"):
        st.meta("s1", meta["id"])


def test_train_on_upload_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "c": rng.choice(["u", "v"], 200)})
    df["y"] = 3 * df.a - df.b + (df.c == "u") * 2 + rng.normal(0, 0.1, 200)
    st = UploadStore(tmp_path, 3600, 10 ** 7)
    meta = st.save("s", df.to_csv(index=False).encode(), "f.csv")
    ds = st.as_dataset("s", meta["id"], "regression", "y")
    model, card = train(ds, "regression", "ridge", max_train_rows=20000, allow_neural=False, epochs=5)
    assert card["metrics"]["r2"] > 0.95
    assert {f["name"] for f in card["features"]} == {"a", "b", "c"}
