# 16 — EPM Chatbot: Experiments in Plain English

The chatbot is a natural-language layer over EPM. You describe an ML problem, and it
works out the task, dataset and algorithms. It then runs them through the existing
benchmark pipeline and reports measured results.

The chatbot uses no LLM, makes no network calls and never runs arbitrary code. It only calls
`run_benchmark()`, `load_dataset()`, `recommend_algorithms()`/`recommend_datasets()` and
the Phase 3 workflows. Those workflows are themselves composed from EPM's own generative and
compression modules.

```bash
python run.py --chat        # terminal chat (or [C] in the interactive menu)
python run.py --serve       # dashboard → "EPM>_ CHAT" button, bottom-right (or the "Chat" nav item)
```

```python
from src.chatbot import ChatEngine
engine = ChatEngine()                                   # synchronous (terminal / scripts / tests)
print(engine.handle("s1", "classify iris with random forest").to_text())
```

## 1. Architecture

```
web/chatbot.js ──POST /api/chat──▶ src/webapi/local_api.py ──▶ ChatEngine.handle(session, message)
                 GET  /api/chat/job (poll)                        │
                                                                  ├─ intents.py   rules → TF-IDF + LogReg fallback
                                                                  ├─ slots.py     exact → regex → fuzzy (difflib)
                                                                  ├─ engine.py    state machine, validation, routing
                                                                  ├─ responses.py text + tables (web & terminal)
                                                                  ├─ docs_index.py  section search over docs/*.md
                                                                  └─ workflows.py   VAE generation, diffusion vs flow, compression
                                                                         │
            run_benchmark · load_dataset · recommend_* · JobManager (shared queue, cancellation)
```

| File | Purpose |
|---|---|
| `src/chatbot/intents.py` | 25 intents. Ordered regex rules come first; a TF-IDF (word + char n-grams) + logistic-regression fallback, trained on built-in example phrases, is used only above 0.35 confidence. |
| `src/chatbot/slots.py` | `ChatSlots` plus extraction of dataset, algorithms (as task-independent *families*), task, cv, seed, horizon, window, contamination, k, epochs, rows, hyper-parameters (`alpha=0.1`), and generative settings. |
| `src/chatbot/engine.py` | `ChatEngine`: sessions, the state machine, slot filling, confirmation, validation, routing, async jobs and cancellation. |
| `src/chatbot/responses.py` | `ChatResponse` (text, tables, suggestions, state, job id) and the formatters. |
| `src/chatbot/docs_index.py` | Answers the `explain` intent with verbatim excerpts from `docs/*.md`, including their source. |
| `src/chatbot/workflows.py` | Phase 3 workflows, evaluated on held-out data with reference baselines. |
| `web/chatbot.js` | Floating terminal-style panel: suggestions as pills, a progress bar, result tables and a CSV download. |

## 2. Intents

| Group | Intents |
|---|---|
| Run (Phase 1) | `classify`, `regress`, `cluster`, `benchmark` (all compatible), `train` (task inferred from the dataset) |
| Run (Phase 2) | `forecast`, `detect_anomalies` |
| Run (Phase 3) | `generate`, `compare_generative`, `compress`, `uncertainty` |
| Explore | `list_datasets`, `list_algorithms`, `describe_dataset`, `preview_data`, `download`, `recommend`, `explain` |
| Results | `show_results`, `compare` (reuses the last run when possible) |
| Meta | `help`, `status`, `cancel`, `confirm`, `deny` |

## 3. Conversation state machine

```
idle → slot_filling (ask dataset → ask algorithms) → confirming (large datasets only) → executing → showing_results → idle
                                                                                  └→ error → idle
```

- **Missing dataset:** the chatbot asks for one and lists the compatible datasets, small first.
- **Interactive dataset choice:** if you picked the dataset in reply to that question, it then asks which algorithms. You can name them, or answer **all** or **recommended**.
- **Large datasets:** these (credit-card fraud, household power) need a yes/no confirmation.
- **Follow-ups use context:** for example "run again with 5-fold cv", "compare knn and naive bayes", "show results", or "now with seed 7".
- **Topic changes:** a new explicit request (e.g. "describe iris") drops the pending question.

## 4. Validation & guardrails

- **Task fit:** the dataset must support the task. Algorithm families are resolved per task, and a mismatch says which task the algorithm belongs to and suggests the equivalent. Example: *"Logistic Regression is a classification algorithm, not regression. Did you mean Linear Regression (OLS)?"*
- **Value ranges:**
  - CV folds: 2–10.
  - k: 2–50.
  - Epochs: 1–300.
  - Max train rows: at least 50.
  - Contamination: between 0 and 0.5.
  - Horizon: 1–168.
  - Window: at least 2 × horizon. If the default window is too short, it is raised and the reply says so.
- **Hyper-parameters:** applied only when exactly one algorithm is named. Only simple scalar values are accepted.
- **Limits:**
  - Message length: 1,000 characters.
  - Sessions: 500, kept in memory.
  - History: 200 messages per session.
  - Jobs: the existing `MAX_QUEUED_JOBS` (5) applies.
- **Shared job queue:** chat jobs share the Experiment Lab's queue. `cancel` stops a job at the next model or phase boundary.
- **Data exposure:** previews show at most 20 rows. Unknown datasets are never silently substituted; for example, "mnist" is *not* mapped to the 8×8 digits set.

## 5. Phase 3 workflows (measured, held-out)

| Workflow | Models (existing EPM modules) | Reported |
|---|---|---|
| `generate` | `VariationalAutoencoder` (+ `StandardAutoencoder` for reconstruction only) | Held-out reconstruction MSE and KL; sliced-Wasserstein and MMD vs held-out rows; per-feature real/synthetic mean and std with KS p-values; correlation error; CSV of samples |
| `compare_generative` | OT-CFM flow matching (Euler 20/10) vs DDPM-100 and DDIM-20/10 | Sliced-W, MMD, path straightness, latency, NFE, params |
| `compress` | Teacher → student, knowledge distillation, INT8 dynamic quantisation, 50% unstructured pruning, structured pruning | Accuracy, F1, size, × smaller, single-sample latency, params |
| `uncertainty` | Classifiers via `run_benchmark` | ECE (10 bins) and Brier score. For regression/forecasting datasets it explains the limitation and quotes the recorded Stage 9 artifact. |

Every distance table includes two references: real training rows (the sampling-noise floor) and an independent Gaussian (a baseline that ignores correlations).

### Findings while building Phase 3

1. **VAE posterior collapse.** The Stage 7 VAE averages reconstruction MSE over features but sums the KL over latent dims. With β = 1 this collapses on low-dimensional tables: KL → 0 and the samples are near-constant (synthetic σ ≈ 0.05 vs 1.0). The chatbot therefore uses β = 0.5/d. That is the standard Gaussian-decoder ELBO with σ = 0.5 in standardised units, and it can be overridden with `beta=…`. Even so, on these small tables the VAE does not always beat the independent-Gaussian baseline, and the reply says so.
2. **Diffusion sampling blow-up.** With the cosine schedule (T = 100), the last ᾱ is about 2.4×10⁻⁷. The unclipped x₀ estimate therefore multiplies noise-prediction error by about 2,000, which explains the very large W₂ values in the Stage 10 leaderboard (DDIM 208, DDPM 84). `GaussianDiffusion.sample_ddpm/sample_ddim` now accept an **optional** `clip_x0` (the reference DDPM "clip_denoised"). The default `None` keeps all earlier behaviour and results unchanged. With `clip_x0=5`, DDPM on sensor telemetry goes from sliced-W 146 to 0.149.
3. **Stage 10 NLL values are not measurements.** `run_flow_matching_benchmark` reports NLL for the 5-step OT-CFM and Gaussian-CFM variants as `nll_ot + 0.12` and `nll_ot + 0.18`. It also evaluates on rows it trained on. The chatbot does not reuse it and reports no NLL.
4. **Stage 9 coverage mismatch.** `benchmarks/stage9_timeseries_diffusion.json` records 15.4% empirical coverage for the 80% interval. The hard-coded Stage 9 leaderboard in `web/app.js` shows 80.4%.

## 6. API

```
POST /api/chat                 {"session_id"?: str, "message": str}
  → {session_id, reply, text, state, intent, kind, job_id, suggestions, slots, tables, data}
GET  /api/chat/job?session_id=…&job_id=…   → same shape; progress while running (data.progress), final reply when done
GET  /api/chat/sessions?session_id=…        → {history, current_state, last_job_id, context, pending}
GET  /api/chat/samples?run_id=…             → {filename, csv} for a generate run
POST /api/jobs/<id>/cancel                  → cancel any job (also used by the "cancel" intent)
```

On Vercel, `POST /api/chat` returns `501` with instructions to run locally, because the
serverless API is stdlib-only and read-only.

## 7. Tests

`tests/test_chatbot.py` (95 tests) covers:
- intent parsing (canonical and fallback);
- slot extraction (exact, fuzzy, regex, hyper-parameters, longest match);
- conversation flow (slot filling, confirmation, topic change, follow-ups, compare from the last run, cancel);
- Phase 2 validation and end-to-end runs;
- Phase 3 workflows end to end;
- calibration maths;
- the optional diffusion clip;
- API routes, including async jobs, caching and cancellation;
- a latency check: non-training replies average under 200 ms.
