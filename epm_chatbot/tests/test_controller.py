"""Conversation controller: intents, slot filling, validation, no silent training, grounded replies."""

import re

import pytest

from epm_chatbot.backend.controller import WELCOME_CHIPS
from epm_chatbot.tests.conftest import model_card, needs_torch


def blocks(msg, kind):
    return [b for b in msg["blocks"] if b and b["type"] == kind]


def alltext(msg):
    out = []
    for b in msg["blocks"]:
        if not b:
            continue
        if b.get("text"):
            out.append(b["text"])
        for inner in b.get("blocks", []):
            if inner.get("text"):
                out.append(inner["text"])
    return " ".join(out)


@pytest.mark.parametrize("message,intent", [
    ("Explain how this prediction works.", "how_it_works"),
    ("Run a prediction using my data.", "predict"),
    ("Compare available ML models.", "compare"),
    ("Explain the prediction results.", "explain_result"),
    ("Show me important features.", "importance"),
    ("predict iris species", "predict"),
    ("train a random forest on wine", "train"),
    ("what is gradient boosting?", "concept"),
    ("list datasets", "list_datasets"),
    ("which models are available", "list_models"),
    ("list algorithms", "list_algorithms"),
    ("describe diabetes", "describe_dataset"),
    ("what is EPM", "about"),
    ("hello", "help"),
    ("zxcv qwerty", "unknown"),
])
def test_intents(controller, message, intent):
    s = controller.session("intent-test")
    assert controller.detect_intent(s, message) == intent


def test_every_welcome_chip_gets_a_real_answer(controller):
    for chip in WELCOME_CHIPS:
        m = controller.handle("chips-" + chip[:6], chip)
        assert m["state"] != "error", chip
        assert len(alltext(m)) > 40, chip
        assert m["suggestions"], chip


def test_prediction_with_slot_filling(controller, app):
    sid = "slot"
    m = controller.handle(sid, "predict iris species")
    assert m["state"] == "slot_filling" and blocks(m, "table")
    m = controller.handle(sid, "sepal length 5.1, sepal width 3.5")
    assert m["state"] == "slot_filling" and "petal_length" in alltext(m)
    m = controller.handle(sid, "petal_length=1.4, petal_width=0.2")
    pred = blocks(m, "prediction")[0]
    card = model_card(app.store, "iris")
    model = app.store.load(card["id"], None)
    import pandas as pd

    X = pd.DataFrame([{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}])
    assert pred["label"] == card["target"]["classes"][int(model.predict(X)[0])]          # real model output
    assert pred["confidence"] == pytest.approx(float(model.predict_proba(X).max()), abs=1e-6)


def test_invalid_input_is_explained(controller):
    m = controller.handle("bad", "predict iris with sepal_length=abc, petal_lenght=2")
    t = alltext(m)
    assert "must be a number" in t and "did you mean 'petal_length'" in t
    assert not blocks(m, "prediction")


def test_never_trains_silently_on_predict(controller, app):
    before = len(app.store.list(owner=None))
    m = controller.handle("nosilent", "predict wine with alcohol=13")
    assert m["state"] == "confirming" and m["job"] is None
    assert len(app.store.list(owner=controller.session("nosilent").owner)) == before
    m = controller.handle("nosilent", "no")
    assert "won't" in alltext(m)
    assert app.jobs.active_for("nosilent") is None


def test_explain_without_prediction(controller):
    m = controller.handle("noexp", "Explain the prediction results.")
    assert "no prediction" in alltext(m).lower()


def test_numbers_come_from_the_model_card(controller, app):
    card = model_card(app.store, "diabetes")
    m = controller.handle("nums", "predict diabetes with bmi=0.05")
    m = controller.handle("nums", "use typical values for the rest")
    model_block = blocks(m, "model")[0]
    rmse = dict(model_block["metrics"])["RMSE"]
    assert float(rmse.replace(",", "")) == pytest.approx(card["metrics"]["rmse"], rel=1e-3)
    assert "imputer" in alltext(m)


def test_importance_uses_heldout_permutation(controller, app):
    controller.handle("imp", "predict iris species")
    m = controller.handle("imp", "show important features")
    b = blocks(m, "bars")[0]
    card = model_card(app.store, "iris")
    assert [i["label"] for i in b["items"]] == [i["feature"] for i in card["importance"]["items"]]
    assert "SHAP is not installed" in alltext(m)


@needs_torch
def test_forecast_reply_and_honest_baseline(controller):
    m = controller.handle("fc", "forecast sensor telemetry 2 steps ahead")
    f = blocks(m, "forecast")[0]
    assert len(f["values"]) == 2
    m = controller.handle("fc", "forecast sensor telemetry 12 steps ahead")
    assert m["state"] == "confirming"            # longer horizon than any trained model → ask, don't train


def test_unsupported_algorithm_for_task(controller):
    m = controller.handle("algo", "train an lstm on iris")
    assert "forecasting, not classification" in alltext(m)


def test_upload_flow_and_clear(controller, app, tmp_path):
    from epm_chatbot.ml_service.datasets import load_registry

    df = load_registry("iris", False).features.head(25)
    m = controller.handle_upload("up", "new.csv", df.to_csv(index=False).encode())
    assert "Received" in alltext(m) and m["state"] == "slot_filling"
    m = controller.handle("up", "predict with my file")
    assert "Scored" in alltext(m) and blocks(m, "download")
    fid = blocks(m, "download")[0]["file_id"]
    name, data = controller.download("up", fid)
    assert data.decode().splitlines()[0].endswith("predicted_species,probability")
    assert controller.download("someone-else", fid) is None
    m = controller.handle_upload("up", "bad.csv", b"\x00\x00")
    assert m["state"] == "error"
    controller.clear("up")
    assert not any((app.uploads.root).glob("*/*.csv"))


def test_message_length_limit(controller):
    m = controller.handle("long", "x" * 5000)
    assert m["state"] == "error"


def test_training_job_end_to_end(controller, app):
    m = controller.handle("job", "train a decision tree on iris")
    assert m["job"]["state"] in ("queued", "running")
    job = app.jobs.get(m["job"]["id"], "job")
    assert job.done_event.wait(180)
    assert job.state == "done", job.error
    reply = job.reply
    card = app.store.card(job.result["id"], controller.session("job").owner)
    assert card["owner"] == controller.session("job").owner          # chat-trained models are session-private
    assert re.search(r"held-out accuracy", alltext(reply))
    m = controller.handle("job", "predict iris with decision tree, sepal_length=5, sepal_width=3, petal_length=1.5, petal_width=0.3")
    assert blocks(m, "prediction") and "Decision Tree" in blocks(m, "model")[0]["title"]
