"""
EPM chatbot tests (src/chatbot/, POST /api/chat).

* Intent parsing and slot extraction are deterministic (rules + a TF-IDF fallback
  trained at start-up on fixed phrases).
* Conversation-flow tests use the synchronous engine.
* End-to-end tests train real (small) models through the existing runner /
  workflows; results are written to temporary directories.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

from src.chatbot import ChatEngine, classify_intent, extract_slots, parse_intent
from src.chatbot.slots import resolve_algorithms
from src.data import loaders as L

HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_DATA_HOME", str(tmp_path / "data"))
    import src.benchmarking.runner as R
    import src.chatbot.workflows as W

    monkeypatch.setattr(R, "default_save_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(W, "workflows_dir", lambda: tmp_path / "workflows")
    L.clear_memory_cache()
    yield


@pytest.fixture
def engine():
    return ChatEngine()


# =============================================================================
# Phase 1 — intents
# =============================================================================
class TestIntentParsing:
    @pytest.mark.parametrize("text,intent", [
        ("classify iris", "classify"),
        ("classify iris with random forest", "classify"),
        ("predict house prices", "regress"),
        ("I want to do regression", "regress"),
        ("group the wine data", "cluster"),
        ("cluster the wine dataset", "cluster"),
        ("show me all datasets", "list_datasets"),
        ("list classification algorithms", "list_algorithms"),
        ("what can you do?", "help"),
        ("what's the best regression model for diabetes?", "recommend"),
        ("which datasets can I use with random forest", "recommend"),
        ("describe the wine dataset", "describe_dataset"),
        ("preview iris", "preview_data"),
        ("show results", "show_results"),
        ("compare random forest and svm on iris", "compare"),
        ("benchmark all algorithms on wine", "benchmark"),
        ("status", "status"),
        ("cancel", "cancel"),
        ("yes", "confirm"),
        ("no", "deny"),
        ("download seeds", "download"),
        ("classify credit card fraud", "classify"),
    ])
    def test_canonical_phrases(self, text, intent):
        assert parse_intent(text) == intent

    def test_fallback_classifier_is_used_and_reported(self):
        intent, conf, source = classify_intent("make fake rows like the iris data")
        assert source == "model" and intent == "generate" and 0 < conf <= 1

    def test_gibberish_is_unknown(self):
        assert classify_intent("banana")[0] == "unknown"

    def test_non_training_intents_are_fast(self, engine):
        engine.handle("warm", "help")  # build the fallback model once
        t0 = time.perf_counter()
        for msg in ("list datasets", "describe iris", "recommend a model for wine", "explain attention", "status"):
            engine.handle("lat", msg)
        assert (time.perf_counter() - t0) / 5 < 0.2


# =============================================================================
# Phase 1 — slots
# =============================================================================
class TestSlotExtraction:
    def test_dataset_exact(self):
        assert extract_slots("classify iris").dataset_id == "iris"

    def test_dataset_fuzzy(self):
        assert extract_slots("classify the brest cancer dataset").dataset_id == "breast_cancer"

    def test_dataset_by_human_name(self):
        assert extract_slots("predict house prices").dataset_id == "california_housing"

    def test_algorithm_family_resolves_per_task(self):
        s = extract_slots("use random forest on iris")
        assert resolve_algorithms(s, "classification")[0] == ["rf_clf"]
        assert resolve_algorithms(s, "regression")[0] == ["rf_reg"]

    def test_algorithm_fuzzy(self):
        assert extract_slots("use random forrest on iris").algorithm_families == ["random forest"]

    def test_longest_algorithm_name_wins(self):
        s = extract_slots("classify wine with knn and naive bayes")
        assert s.algorithm_families == ["knn", "naive bayes"] and not s.algorithm_ids

    def test_algorithm_name_does_not_set_task(self):
        s = extract_slots("logistic regression on wine")
        assert s.task is None and resolve_algorithms(s, "classification")[0] == ["logreg"]

    def test_cv_folds(self):
        assert extract_slots("classify iris with 5-fold cv").cv_folds == 5
        assert extract_slots("regression with cross-validation").cv_folds == 5
        assert extract_slots("cv=3 please").cv_folds == 3

    def test_seed_rows_epochs(self):
        s = extract_slots("seed 7, max train rows 5000, 20 epochs")
        assert (s.seed, s.max_train_rows, s.epochs) == (7, 5000, 20)

    def test_hyperparameters(self):
        assert extract_slots("ridge alpha=0.5 on diabetes").hyperparams == {"alpha": 0.5}

    def test_no_phantom_algorithm_from_task_word(self):
        s = extract_slots("I want to do regression")
        assert s.task == "regression" and not s.algorithm_families and not s.algorithm_ids

    def test_mnist_is_not_silently_mapped_to_digits(self):
        assert extract_slots("classify mnist").dataset_id is None

    def test_wrong_task_algorithm_reports_alternative(self):
        _, problems = resolve_algorithms(extract_slots("logistic regression"), "regression")
        assert problems and "Did you mean" in problems[0]

    def test_merge_keeps_existing_values(self):
        a = extract_slots("classify iris")
        a.merge(extract_slots("with 3-fold cv"))
        assert a.dataset_id == "iris" and a.cv_folds == 3


# =============================================================================
# Phase 1 — conversation flow
# =============================================================================
class TestConversationFlow:
    def test_slot_filling_asks_for_dataset(self, engine):
        r = engine.handle("s1", "I want to classify")
        assert "which dataset" in r.text.lower() and r.state == "slot_filling" and r.kind == "question"

    def test_slot_filling_then_algorithms_then_run(self, engine):
        engine.handle("s1", "I want to do regression")
        r = engine.handle("s1", "diabetes")
        assert r.state == "slot_filling" and "which algorithms" in r.text.lower()
        r = engine.handle("s1", "ridge and lasso")
        assert r.state == "showing_results" and r.tables and len(r.tables[0]["rows"]) == 2

    def test_recommended_answer_runs_default_set(self, engine):
        engine.handle("s2", "I want to classify")
        engine.handle("s2", "iris")
        r = engine.handle("s2", "recommended")
        assert r.state == "showing_results" and len(r.tables[0]["rows"]) >= 4

    def test_dataset_task_mismatch_is_explained(self, engine):
        r = engine.handle("s3", "classify diabetes")
        assert r.kind == "error" and "regression dataset" in r.text and r.intent == "classify"

    def test_wrong_algorithm_suggests_alternative(self, engine):
        r = engine.handle("s3", "logistic regression on diabetes")
        assert r.kind == "error" and "Did you mean" in r.text

    def test_large_dataset_requires_confirmation_and_no_cancels(self, engine):
        r = engine.handle("s4", "detect anomalies in the credit card dataset")
        assert r.state == "confirming"
        r = engine.handle("s4", "no")
        assert "nothing was run" in r.text and engine.session("s4").pending is None

    def test_changing_topic_drops_pending_question(self, engine):
        engine.handle("s5", "I want to classify")
        r = engine.handle("s5", "list regression datasets")
        assert r.intent == "list_datasets" and engine.session("s5").pending is None

    def test_follow_up_rerun_with_cv(self, engine):
        engine.handle("s6", "classify iris with logistic regression")
        r = engine.handle("s6", "run again with 3-fold cv")
        assert r.state == "showing_results"
        assert engine.session("s6").last_result["config"]["cv_folds"] == 3

    def test_compare_uses_last_run(self, engine):
        engine.handle("s7", "classify wine with knn and naive bayes")
        n_runs = len(engine.session("s7").history)
        r = engine.handle("s7", "compare knn and naive bayes")
        assert r.intent == "compare" and "beats" in r.text
        assert engine.session("s7").last_result["run_id"] in r.text  # no new run
        assert len(engine.session("s7").history) == n_runs + 2

    def test_explicit_short_request_is_not_taken_as_an_answer(self, engine):
        engine.handle("s12", "I want to classify")
        r = engine.handle("s12", "describe iris")
        assert r.intent == "describe_dataset" and engine.session("s12").pending is None

    def test_cancel_pending(self, engine):
        engine.handle("s8", "I want to cluster")
        r = engine.handle("s8", "cancel")
        assert engine.session("s8").pending is None and r.intent == "cancel"

    def test_unknown_message_offers_help(self, engine):
        r = engine.handle("s9", "banana")
        assert r.kind == "error" and "help" in r.suggestions

    def test_history_is_recorded(self, engine):
        engine.handle("s10", "help")
        info = engine.session_info("s10")
        assert [m["role"] for m in info["history"]] == ["user", "bot"]

    def test_message_length_is_capped(self, engine):
        engine.handle("s11", "x" * 5000)
        assert len(engine.session("s11").history[0]["text"]) == 1000


# =============================================================================
# Phase 1 — exploratory intents
# =============================================================================
class TestExploration:
    def test_list_datasets_for_task(self, engine):
        r = engine.handle("e1", "list clustering datasets")
        ids = [row[0] for row in r.tables[0]["rows"]]
        assert "wholesale_customers" in ids and "diabetes" not in ids

    def test_algorithms_for_dataset(self, engine):
        r = engine.handle("e2", "which algorithms work on seeds?")
        assert r.intent == "list_algorithms" and r.tables[0]["rows"]

    def test_describe_includes_source_and_licence(self, engine):
        r = engine.handle("e3", "describe adult")
        assert "openml.org/d/1590" in r.text and "Licence" in r.text

    def test_what_is_dataset_routes_to_describe(self, engine):
        assert engine.handle("e4", "what is iris").intent == "describe_dataset"

    def test_preview_bundled_and_refuse_remote(self, engine):
        r = engine.handle("e5", "preview iris")
        assert len(r.tables[0]["rows"]) == 5
        r = engine.handle("e5", "preview seeds")
        assert r.kind == "error" and "download seeds" in r.suggestions

    def test_recommend_states_it_is_not_a_prediction(self, engine):
        r = engine.handle("e6", "what's the best regression model for diabetes?")
        assert "not predicted accuracy" in r.text

    def test_datasets_for_algorithm(self, engine):
        r = engine.handle("e7", "which datasets can I use with random forest")
        assert "Quick experiments" in r.text and "iris" in r.text

    def test_status(self, engine):
        assert "Datasets ready locally" in engine.handle("e8", "status").text


# =============================================================================
# Phase 1 — end to end
# =============================================================================
class TestEndToEnd:
    def test_classify_iris(self, engine):
        r = engine.handle("x1", "classify iris with logistic regression")
        assert r.slots["task"] == "classification" and r.kind == "results"
        assert r.data["run_id"] and r.tables[0]["rows"][0][1].startswith("Logistic Regression")

    def test_cluster_wine_uses_labels_only_for_evaluation(self, engine):
        r = engine.handle("x2", "cluster the wine dataset with kmeans")
        res = engine.session("x2").last_result
        assert res["task"] == "clustering" and res["models"][0]["metrics"]["ari_external"] is not None

    def test_hyperparameters_reach_the_model(self, engine):
        engine.handle("x3", "decision tree max_depth=2 on iris")
        res = engine.session("x3").last_result
        assert res["config"]["algorithm_params"] == {"dt_clf": {"max_depth": 2}}
        assert res["models"][0]["n_parameters"] <= 7  # depth-2 tree has at most 7 nodes


# =============================================================================
# Phase 2 — forecasting & anomaly detection
# =============================================================================
class TestPhase2:
    @pytest.mark.parametrize("text,intent", [
        ("forecast air quality 24 steps ahead", "forecast"),
        ("predict the next 12 hours of power", "forecast"),
        ("detect anomalies in the NAB data", "detect_anomalies"),
        ("find outliers in the synthetic anomalies", "detect_anomalies"),
    ])
    def test_intents(self, text, intent):
        assert parse_intent(text) == intent

    def test_slots(self):
        s = extract_slots("forecast air quality 12 steps ahead with lstm window 48")
        assert (s.dataset_id, s.horizon, s.window, s.algorithm_families) == ("air_quality", 12, 48, ["lstm"])
        assert extract_slots("contamination=0.02").contamination == 0.02
        assert extract_slots("contamination of 5%").contamination == pytest.approx(0.05)

    def test_window_must_cover_twice_the_horizon(self, engine):
        r = engine.handle("p1", "forecast the sensor telemetry 12 steps ahead with window 10")
        assert r.kind == "error" and "≥ 24" in r.text

    def test_default_window_is_raised_and_reported(self, engine):
        r = engine.handle("p2", "forecast the sensor telemetry 20 steps ahead with naive")
        assert "Lookback window set to 40" in r.text and r.state == "showing_results"

    def test_invalid_contamination(self, engine):
        r = engine.handle("p3", "detect anomalies in synthetic anomalies with contamination=0.7")
        assert r.kind == "error" and "between 0 and 0.5" in r.text

    def test_forecast_e2e(self, engine):
        r = engine.handle("p4", "forecast the sensor telemetry 3 steps ahead with naive and linear")
        res = engine.session("p4").last_result
        assert res["split"]["strategy"] == "chronological" and res["split"]["horizon"] == 3
        assert {m["algorithm"] for m in res["models"]} == {"naive", "linear_ar"}

    def test_anomaly_e2e(self, engine):
        r = engine.handle("p5", "detect anomalies in the synthetic anomalies data with isolation forest")
        res = engine.session("p5").last_result
        assert res["anomaly_setup"]["labels_used_for_training"] is False
        assert res["models"][0]["metrics"]["pr_auc"] > 0.9


# =============================================================================
# Phase 3 — generative, compression, uncertainty, explain
# =============================================================================
class TestPhase3:
    @pytest.mark.parametrize("text,intent", [
        ("generate 100 samples from diabetes distribution using VAE", "generate"),
        ("compare diffusion vs flow matching on sensor data", "compare_generative"),
        ("compress a model trained on breast cancer", "compress"),
        ("quantize and prune the wine classifier", "compress"),
        ("how calibrated is the classifier on wine?", "uncertainty"),
        ("explain what flow matching is", "explain"),
    ])
    def test_intents(self, text, intent):
        assert parse_intent(text) == intent

    def test_generative_slots(self):
        s = extract_slots("generate 250 samples from wine with a vae, latent 3, 10 epochs")
        assert (s.generative_model, s.num_samples, s.latent_dim, s.epochs, s.dataset_id) == ("vae", 250, 3, 10, "wine")
        assert not s.algorithm_families  # "vae" is the generator, not the anomaly autoencoder

    def test_explain_quotes_docs_with_source(self, engine):
        r = engine.handle("g1", "explain what flow matching is")
        assert r.data["source"] == "docs/12_flow_matching.md" and "Flow Matching" in r.text

    def test_explain_unknown_topic(self, engine):
        r = engine.handle("g2", "explain quantum chromodynamics")
        assert "couldn't find" in r.text

    def test_uncertainty_reports_calibration(self, engine):
        r = engine.handle("g3", "how calibrated is the classifier on wine?")
        assert r.tables[0]["title"].startswith("Calibration")
        res = engine.session("g3").last_result
        ece = [m["metrics"]["ece"] for m in res["models"] if m["status"] == "ok"]
        assert ece and all(0 <= e <= 1 for e in ece)

    def test_uncertainty_on_forecasting_dataset_is_honest(self, engine):
        r = engine.handle("g4", "uncertainty on air quality")
        assert "measured for classifiers" in r.text

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_generate_e2e(self, engine):
        r = engine.handle("g5", "generate 50 samples from wine with a vae, 5 epochs")
        res = engine.session("g5").last_result
        assert res["num_samples"] == 50 and Path(res["samples_csv"]).name.endswith("_samples.csv")
        assert {d["model"] for d in res["distribution"]} >= {"VAE (Stage 7)", "Reference: real training rows"}
        assert r.data["samples_csv"]

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_compress_e2e(self, engine):
        engine.handle("g6", "compress a model on iris with 5 epochs")
        res = engine.session("g6").last_result
        names = [v["variant"] for v in res["variants"]]
        assert names[0].startswith("Teacher") and any("INT8" in n for n in names)
        assert all(0 <= v["accuracy"] <= 1 for v in res["variants"])

    def test_compress_rejects_regression_dataset(self, engine):
        r = engine.handle("g7", "compress a model on diabetes")
        assert r.kind == "error"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_generative_comparison_e2e(self, engine):
        engine.handle("g8", "compare diffusion vs flow matching on wine with 3 epochs")
        res = engine.session("g8").last_result
        fams = {m["family"] for m in res["models"]}
        assert fams == {"flow matching", "diffusion"} and len(res["references"]) == 2
        assert all(np.isfinite(m["sliced_w"]) for m in res["models"])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_diffusion_clip_is_optional_and_bounds_samples():
    import torch

    from src.diffusion.gaussian_diffusion import GaussianDiffusion
    from src.diffusion.models import DenoisingMLP
    from src.diffusion.schedules import DiffusionSchedule

    torch.manual_seed(0)
    diff = GaussianDiffusion(DiffusionSchedule(50, "cosine"))
    net = DenoisingMLP(input_dim=3, hidden_dim=16, num_blocks=1)  # untrained: worst case
    torch.manual_seed(1)
    a = diff.sample_ddim(net, (64, 3), num_steps=10)
    torch.manual_seed(1)
    b = diff.sample_ddim(net, (64, 3), num_steps=10, clip_x0=None)
    assert torch.equal(a, b)  # default behaviour unchanged
    torch.manual_seed(1)
    c = diff.sample_ddim(net, (64, 3), num_steps=10, clip_x0=3.0)
    assert c.abs().max() <= 3.0 + 1e-4


def test_calibration_metrics_known_values():
    from src.evaluation.task_metrics import calibration_metrics

    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    m = calibration_metrics(np.array([0, 1, 1, 1]), p, [0, 1])
    assert m["ece"] == pytest.approx(0.3) and m["brier"] == pytest.approx(0.25)


# =============================================================================
# API
# =============================================================================
class TestChatAPI:
    def _wait(self, A, sid, jid):
        for _ in range(300):
            s, r = A.route("GET", "/api/chat/job", {"session_id": sid, "job_id": jid}, {})
            if r["state"] != "executing":
                return r
            time.sleep(0.1)
        raise AssertionError("chat job did not finish")

    def test_post_chat_runs_job_and_formats_result(self):
        from src.webapi import local_api as A

        s, r = A.route("POST", "/api/chat", {}, {"message": "classify iris with naive bayes"})
        assert s == 200 and r["job_id"] and r["state"] == "executing" and r["reply"]
        done = self._wait(A, r["session_id"], r["job_id"])
        assert done["kind"] == "results" and done["tables"]
        s, again = A.route("GET", "/api/chat/job", {"session_id": r["session_id"], "job_id": r["job_id"]}, {})
        assert again["reply"] == done["reply"]  # cached
        s, info = A.route("GET", "/api/chat/sessions", {"session_id": r["session_id"]}, {})
        assert s == 200 and [m["role"] for m in info["history"]] == ["user", "bot", "bot"]  # ask, start, result

    def test_chat_cancel_stops_job(self):
        from src.webapi import local_api as A

        s, r = A.route("POST", "/api/chat", {}, {"message": "benchmark digits"})
        s, c = A.route("POST", "/api/chat", {}, {"session_id": r["session_id"], "message": "cancel"})
        assert "Cancellation requested" in c["reply"] or "Nothing is running" in c["reply"]
        done = self._wait(A, r["session_id"], r["job_id"])
        assert done["intent"] in ("cancel", "benchmark")

    def test_validation_errors(self):
        from src.webapi import local_api as A

        assert A.route("POST", "/api/chat", {}, {"message": ""})[0] == 400
        assert A.route("POST", "/api/chat", {}, {"message": "x" * 1001})[0] == 400
        assert A.route("GET", "/api/chat/sessions", {}, {})[0] == 400
        assert A.route("GET", "/api/chat/job", {"session_id": "a", "job_id": "nope"}, {})[0] == 404
        assert A.route("GET", "/api/chat/samples", {"run_id": "../../etc"}, {})[0] == 404

    def test_vercel_api_reports_local_only(self):
        spec = importlib.util.spec_from_file_location("epm_artifact_api_chat", ROOT / "api" / "index.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        status, body = mod.route_request("POST", "/api/chat", {}, {"message": "hi"})
        assert status == 501 and "run.py --serve" in body["reply"]
