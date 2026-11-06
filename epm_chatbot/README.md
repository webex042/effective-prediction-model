# EPM Chat — standalone chatbot for EPM

EPM Chat is a conversational front end to EPM's real models. It runs as its **own service** with its own
frontend, backend and API. It does not touch the EPM website: `web/`, `api/`, `run.py` and the Vercel config are
unchanged. You can serve it at its own URL, put it in an iframe, or add it as a floating widget, but only on the
websites you list.

```
epm_chatbot/
├── __main__.py          python -m epm_chatbot            → start the server
├── bootstrap.py         python -m epm_chatbot.bootstrap  → explicitly train the shared starter models
├── config/settings.py   every setting comes from environment variables (.env supported)
├── backend/
│   ├── server.py        HTTP server (standard library): pages, JSON + SSE API, uploads, downloads
│   ├── controller.py    conversation controller: intents, follow-up questions, validation, replies
│   ├── security.py      origin allowlist, signed session tokens, rate limits, CSP
│   ├── jobs.py          bounded job queue (training / comparisons) with cancel + timeout
│   ├── llm.py           OPTIONAL LLM: rewording only, every number checked against model output
│   ├── knowledge.py     number-free explanations of algorithms and of EPM
│   └── logging_setup.py JSON logs without message text, uploads or secrets
├── ml_service/          explicit adapters to EPM (src/)
│   ├── training.py      train via EPM's leakage-free pipelines → held-out metrics, importance, intervals
│   ├── inference.py     input parsing/validation, predictions, forecasts, local explanations
│   ├── model_store.py   saved models + metadata cards, SHA-256 integrity check, per-session scoping
│   ├── datasets.py      EPM registry datasets + per-session CSV uploads (validated, auto-deleted)
│   └── runner.py        sandboxed subprocess for training (hard timeout, crash isolation)
├── frontend/            index.html · app.css · app.js · embed.js (no build step, no dependencies)
├── tests/               75 tests: security, ML adapters, controller, HTTP API, UI, browser end-to-end
├── Dockerfile · .env.example · requirements.txt
```

## 1. What it can do (and where each answer comes from)

| You ask | What happens | Source of every number |
|---|---|---|
| "predict iris species" | Asks for the 4 inputs, showing training ranges and typical values; validates them; runs the saved model | `model.predict` / `predict_proba` on the saved EPM pipeline |
| "petal length 1.4, …" or a CSV upload | Checks names, types and ranges, warns on extrapolation or unseen categories, then predicts one row or the whole file (CSV download) | Same saved pipeline (identical preprocessing to training) |
| "forecast sensor telemetry 6 steps ahead" | Forecasts the steps after the last observation with a 90% interval and a chart | Saved forecaster plus the training-segment scaler |
| "Explain the prediction results." | Shows how much each input moved the result compared with its typical value | Re-runs the model with each feature reset to its training median |
| "Show me important features." | Permutation importance on the **held-out test split**, plus model-internal importances | Computed at training time and stored in the model card |
| "Compare available ML models." | Compares the trained models' held-out scores, or (after you confirm) runs EPM's benchmark on one shared split | Model cards / `src.benchmarking.run_benchmark` |
| "train a random forest on wine" / "train on my file, target price" | Trains **only because you asked**, in a sandboxed process, then reports held-out metrics | `src.preprocessing` pipelines + `src.evaluation.task_metrics` |
| "what is gradient boosting?", "explain data leakage" | Plain-language explanation; ML concepts are quoted from EPM's `docs/` | Static text, or excerpts with the source file named |

**Supported tasks for predictions:** classification, regression and forecasting. These use the algorithms in
EPM's catalogue:
- Linear/Ridge/Lasso/Elastic Net, Logistic Regression, KNN, Decision Tree, Random Forest, SVM/SVR, Naive Bayes,
  Gradient Boosting, XGBoost and a PyTorch MLP;
- Linear AR, RF/XGBoost lag-window models, LSTM, GRU and Temporal Transformer for forecasting.

Clustering and anomaly-detection models exist in EPM, but the chatbot does not expose them for per-row
predictions. Baseline models are used only for comparisons.

**Never fabricated:**
- **No silent training.** If no trained model exists, the bot asks before training anything.
- **Honest probabilities.** Probabilities are shown only for models that produce them, together with the measured
  calibration error (ECE).
- **Measured intervals.** Regression and forecast intervals are split-conformal intervals from validation errors.
- **Weak models are flagged.** If a forecaster is worse than the naive baseline on held-out data, the reply says so.
- **SHAP is not installed,** so the bot says that and uses measured permutation importance instead.

## 2. Quick start (local)

From the repository root (the folder containing `src/`). No new packages are needed beyond `requirements.txt`.

```bash
pip install -r requirements.txt               # if not installed yet
python -m epm_chatbot.bootstrap               # one-off: train 11 starter models (~1 min; --quick skips GRU)
python -m epm_chatbot                         # http://127.0.0.1:8765
```

**On Windows**, run the commands with the project's virtual environment, or the packages won't be found
(`ModuleNotFoundError: No module named 'pandas'`):

```powershell
.\venv\Scripts\python.exe -m epm_chatbot.bootstrap
.\venv\Scripts\python.exe -m epm_chatbot
# or simply double-click / run:  epm_chatbot\start_chat.bat   (uses the venv, trains on first run, starts the server)
``` To let a website embed it, set the allowlist
first (step 4):

```powershell
$env:EPM_CHAT_ALLOWED_ORIGINS = "http://localhost:5500"
python -m epm_chatbot
```

The starter models are trained on EPM's bundled datasets: Iris, Breast Cancer, Wine, Diabetes, the EPM equipment
failure set, sensor regression and sensor telemetry. Datasets that need downloading (UCI/OpenML) are only fetched
when `EPM_CHAT_ALLOW_DOWNLOADS=true`. Large datasets are always disabled in the chat.

## 3. Configuration

Copy `.env.example` to `epm_chatbot/.env`, or set real environment variables (these take precedence).

| Variable | Default | Purpose |
|---|---|---|
| `EPM_CHAT_ALLOWED_ORIGINS` | *(empty)* | Exact origins allowed to embed or call the API, e.g. `https://www.mysite.com`. No wildcards. |
| `EPM_CHAT_SECRET_KEY` | random per process | HMAC key for session tokens (≥32 chars). **Set it in production.** |
| `EPM_CHAT_PUBLIC_ORIGIN` | — | The chat's public URL when it runs behind a proxy |
| `EPM_CHAT_HOST` / `EPM_CHAT_PORT` | `127.0.0.1` / `8765` | Bind address (`0.0.0.0` in containers) |
| `EPM_CHAT_ACCESS_CODE` | — | Optional code users must enter before chatting (for private deployments) |
| `EPM_CHAT_API_KEYS` | — | Keys for server-to-server use without a browser Origin |
| `EPM_CHAT_RATE_PER_MIN` / `_RATE_IP_PER_MIN` / `_TRAIN_PER_HOUR` | 40 / 120 / 12 | Abuse limits |
| `EPM_CHAT_MAX_UPLOAD_BYTES` / `_MAX_BODY_BYTES` / `_MAX_MESSAGE_CHARS` | 5 MB / 64 KB / 1000 | Size limits |
| `EPM_CHAT_TRAIN_TIMEOUT_S` / `_PREDICT_TIMEOUT_S` | 300 / 20 | Hard time limits |
| `EPM_CHAT_UPLOAD_TTL_S` | 3600 | Uploads and chat-trained models are deleted after this |
| `EPM_CHAT_ALLOW_DOWNLOADS` / `_ALLOW_NEURAL` | false / true | Dataset downloads; LSTM/GRU/MLP training |
| `EPM_CHAT_LLM_PROVIDER` / `_LLM_API_KEY` / `_LLM_MODEL` | off | Optional LLM (`anthropic`); the key stays on the server |

**Without an LLM**, every feature works: intents are recognised by rules plus EPM's slot extractor, and
explanations are generated from the model output.

**With an LLM configured**, two things change:
- ambiguous messages can be routed to one of the fixed intents;
- explanations are reworded and streamed token by token.

Any LLM draft containing a number that is not in the model's actual output is discarded, and the deterministic
explanation is shown instead (see `backend/llm.py::ungrounded_numbers`).

## 4. Putting it on a website you choose

Nothing is installed anywhere automatically. Each option works only for origins in `EPM_CHAT_ALLOWED_ORIGINS`.

**Option A — standalone page.** Deploy the service and link to it: `https://chat.example.com/`.

**Option B — iframe.** Your page controls the size:

```html
<iframe src="https://chat.example.com/?embed=1&parent=https://www.mysite.com"
        title="EPM Chat" style="width:100%;height:640px;border:0" allow="clipboard-write"></iframe>
```

**Option C — floating widget.** Add one opt-in script tag:

```html
<script src="https://chat.example.com/embed.js" data-epm-chat defer></script>
<!-- options: data-position="left" data-label="Ask EPM" data-theme="dark"
     inline mode: data-mode="inline" data-target="#chat-box" -->
```

How the embedding is protected:
- **Who may frame it.** The chat page sends `Content-Security-Policy: frame-ancestors 'self' <allowlist>`, so
  browsers refuse to show it inside any other site. This was verified in tests: a non-allowlisted host gets
  "Refused to frame".
- **Widget pre-check.** The widget first calls `/embed/status`. That endpoint returns CORS headers only for
  allowlisted origins, so on any other site the loader stops and logs a console warning.
- **Style isolation.** The launcher lives in a closed shadow root, and the chat itself is a cross-origin iframe.
  Host CSS cannot restyle it, and its CSS cannot leak into the host page (verified against a host page with
  `button{background:red!important}`).
- **Messages between page and chat.** `postMessage` uses an exact target origin, never `*`. Both sides check
  `event.origin` and `event.source`.
  - From the chat to the host page: `epm-chat:ready` and `epm-chat:close`.
  - From the host page to the chat: `epm-chat:focus`, `epm-chat:theme {theme}` and `epm-chat:prefill {text}`.
    Prefill only fills the box; it never sends anything.

## 5. Security model (what is and isn't protected)

- **Sessions.** `POST /api/session` issues an HMAC-signed token (default 30 minutes). It is issued only to the
  chat's own page, to an allowlisted origin, or to a holder of a server API key, and optionally only with the
  access code. Each token is bound to its origin: a token minted for site A is rejected when it is used from
  site B.
- **CORS is not authentication.** Every data route also requires the token. Browser requests from origins that
  aren't allowlisted are refused server-side (403), not just hidden by CORS.
- **Public chatbots are still open to visitors.** Anyone who can load an allowlisted page can chat, by design.
  Abuse is limited by:
  - rate limits per IP and per session;
  - a training quota per session;
  - a bounded job queue;
  - size limits and hard timeouts;
  - training in a separate process that is killed when it exceeds its time limit.

  Use `EPM_CHAT_ACCESS_CODE` or a reverse proxy with authentication for private use.
- **Files.**
  - **What is accepted:** UTF-8 CSV only, checked by size, by rejecting binary content, and by row and column
    caps. Column names are sanitised against spreadsheet-formula injection.
  - **Where uploads go:** each session gets its own folder, separate from EPM's `data/`.
  - **Cleanup:** uploads are deleted on "New chat" or after the TTL. Uploaded files are never unpickled.
  - **Saved models:** only the server's own model files are loaded, and each is checked against its SHA-256
    first.
- **Browser hardening.** The page uses a strict CSP: no inline script, `connect-src 'self'`. All text from the
  server is rendered with `textContent`, never as HTML.
- **Logs.** JSON lines with route, status, latency, intent and job state. Messages, uploads, tokens and keys are
  never logged (this is tested).
- **Existing site data.** EPM Chat reads EPM's dataset registry and caches and its documentation. It does not read
  or write the website's experiment history (`experiments/benchmarks`, `web/catalog`): comparisons run with
  `save=False`.

## 6. API

All routes except `/`, `/static/*`, `/embed.js`, `/embed/status`, `/api/health`, `/api/config` and
`POST /api/session` require `Authorization: Bearer <token>`.

| Method & path | Body / query | Returns |
|---|---|---|
| `GET /api/health` | — | `{status, version, models, llm, uptime_s}` |
| `GET /api/config` | — | limits, `access_code_required`, `embed_origins` |
| `POST /api/session` | `{access_code?}`; send an existing token to resume | `{token, expires_at, resumed, messages[]}` |
| `POST /api/chat` | `{message}` | `{message}`: an assistant message (see below) |
| `POST /api/chat?stream=1` | `{message}` + `Accept: text/event-stream` | SSE: `delta` (LLM tokens), `message`, `job` (progress), `reply` (job result), `done` |
| `GET /api/jobs/<id>` · `GET /api/jobs/<id>/events` · `POST /api/jobs/<id>/cancel` | — | job status / SSE progress / cancel |
| `POST /api/upload` | raw CSV body, `Content-Type: text/csv`, `X-Filename` | `{message}` with the column analysis |
| `GET /api/download/<id>` | — | CSV of batch predictions (session-scoped) |
| `GET /api/history` · `DELETE /api/session` | — | conversation / clear everything for this session |

**Assistant message format:**
- **Fields:** `{id, role, blocks[], suggestions[], intent, state, job}`.
- **`state`:** `idle`, `slot_filling`, `confirming`, `executing` or `error`.
- **`blocks` types:**
  - `text` (with `**bold**`, `` `code` `` and `- ` bullets);
  - `notice` (`info`, `warning` or `error`);
  - `table`, `bars`, `prediction`, `forecast`;
  - `model` (a collapsible card);
  - `details`, `download`, `source`.

Example of a server-to-server call:

```bash
TOKEN=$(curl -s -X POST localhost:8765/api/session -H 'X-API-Key: <key>' | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -s localhost:8765/api/chat -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"message":"predict iris with sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2"}'
```

## 7. Deployment

The chat needs a long-running Python process: it trains in subprocesses and keeps sessions in memory. Use a VM,
Docker host, Render, Railway or Fly.io. Vercel's serverless functions are not suitable, and the existing EPM
Vercel deployment is left as it is.

```bash
docker build -f epm_chatbot/Dockerfile -t epm-chat .          # from the repo root; trains starter models
docker run -d -p 8765:8765 \
  -e EPM_CHAT_SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))") \
  -e EPM_CHAT_ALLOWED_ORIGINS=https://www.mysite.com \
  -e EPM_CHAT_PUBLIC_ORIGIN=https://chat.mysite.com -e EPM_CHAT_TRUST_PROXY=true \
  -v epm_chat_var:/app/epm_chatbot/var epm-chat
```

Put it behind HTTPS with a reverse proxy (Caddy, nginx). The proxy must not buffer `text/event-stream`; the
server already sends `X-Accel-Buffering: no`.

Run **one instance**, or use sticky sessions, because sessions are in memory. The model store is a folder, so
mount it as a volume.

## 8. Tests

```bash
python -m pytest epm_chatbot/tests                 # 74 pass + 1 browser test skipped by default
EPM_CHAT_E2E=1 python -m pytest epm_chatbot/tests  # also drives Chromium via Playwright
python -m pytest tests                             # EPM's existing suite (unchanged: 502 pass)
```

The tests cover:
- tokens (tampering, expiry, origin binding), the allowlist, CORS, CSP, rate limits and access codes;
- leakage, by checking that saved imputer statistics equal the **training-split** medians rather than the
  full-data medians;
- the forecast scaler comes from the training segment;
- artifact integrity and per-session model scoping;
- input parsing and validation, determinism, and that predictions equal the pipeline's own output;
- conformal intervals, and that explanation effects equal real changes in the prediction;
- every welcome chip, slot filling, and no silent training;
- the HTTP API (sizes, content types, SSE progress order), uploads and downloads;
- logs without message text;
- LLM grounding;
- a real browser run, including the embed being allowed on one origin and refused on another.

## 9. Known limitations

- **No SHAP.** Explanations use permutation importance and one-at-a-time sensitivity, which do not split effects
  between interacting features the way SHAP does.
- **Sessions and job progress live in memory**, so a restart clears conversations. Models and uploads are on disk.
- **No feature importance for forecasters,** only accuracy against the naive baseline.
- **Uncached UCI/OpenML datasets** (Adult, Bank Marketing, Air Quality, …) need `EPM_CHAT_ALLOW_DOWNLOADS=true`
  and internet access on the server.
- **Clustering and anomaly-detection models** are not exposed for per-row predictions.
- **No LLM provider is configured** in this repository, so the LLM path is tested only with a stub client.
