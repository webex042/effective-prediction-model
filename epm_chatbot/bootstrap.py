"""
``python -m epm_chatbot.bootstrap`` — explicitly train the shared starter models.

This is an operator command (run once at install / image build). It trains a
small set of models on EPM's bundled datasets through the same sandboxed
training path the chat uses, evaluates them on held-out data and saves them to
``EPM_CHAT_MODEL_DIR``. The chat never trains anything on its own.
"""

from __future__ import annotations

import argparse
import sys
import time

STARTER = [
    ("iris", "classification", "rf_clf", None),
    ("iris", "classification", "logreg", None),
    ("breast_cancer", "classification", "logreg", None),
    ("breast_cancer", "classification", "rf_clf", None),
    ("wine", "classification", "rf_clf", None),
    ("diabetes", "regression", "ridge", None),
    ("diabetes", "regression", "gbr", None),
    ("epm_sensor_failure", "classification", "rf_clf", None),
    ("epm_sensor_regression", "regression", "ridge", None),
    ("epm_sensor_telemetry", "forecasting", "linear_ar", 6),
    ("epm_sensor_telemetry", "forecasting", "gru", 6),
]


def main(argv=None) -> int:
    from epm_chatbot import require_runtime

    require_runtime()
    from epm_chatbot.config import PROJECT_ROOT, Settings
    from epm_chatbot.ml_service.model_store import ModelStore
    from epm_chatbot.ml_service.runner import JobFailed, JobTimeout, run_sandboxed

    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--quick", action="store_true", help="skip neural models (faster)")
    p.add_argument("--force", action="store_true", help="retrain even if a shared model already exists")
    args = p.parse_args(argv)
    s = Settings.from_env()
    store = ModelStore(s.model_dir)
    existing = {(c["dataset_id"], c["algorithm_id"]) for c in store.list(owner=None) if c.get("owner") is None}
    tmp = s.model_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    failures = 0
    for ds, task, algo, horizon in STARTER:
        if args.quick and algo in ("gru", "lstm", "ts_transformer", "mlp_torch"):
            continue
        from src.benchmarking.catalog import get_algorithm

        if not get_algorithm(algo).available():
            print(f"  - {ds:24s} {algo:12s} skipped: optional dependency '{get_algorithm(algo).requires}' not installed")
            continue
        if (ds, algo) in existing and not args.force:
            print(f"  = {ds:24s} {algo:12s} already trained (use --force to retrain)")
            continue
        art = tmp / f"boot_{ds}_{algo}.bin"
        job = {"kind": "train", "project_root": str(PROJECT_ROOT), "dataset": {"kind": "registry", "dataset_id": ds,
                                                                              "allow_downloads": s.allow_downloads},
               "task": task, "algorithm_id": algo, "max_train_rows": s.max_train_rows, "allow_neural": True,
               "epochs": s.neural_epochs, "horizon": horizon, "artifact_path": str(art)}
        t0 = time.time()
        try:
            meta = run_sandboxed(job, s.train_timeout_s * 2, lambda p, m: None, lambda: False)
        except (JobFailed, JobTimeout) as e:
            if "optional package" in str(e):
                print(f"  - {ds:24s} {algo:12s} skipped: {e}")
                continue
            failures += 1
            print(f"  ! {ds:24s} {algo:12s} FAILED: {e}")
            continue
        card = store.save(art, meta, owner=None)
        m = card["metrics"]
        score = (f"F1={m.get('f1'):.4f}" if task == "classification" else f"RMSE={m.get('rmse'):.4f}")
        print(f"  + {ds:24s} {algo:12s} {score}  ({time.time() - t0:.1f}s)  id={card['id']}")
    print(f"Done. {len(store.list(owner=None))} shared model(s) in {s.model_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
