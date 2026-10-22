# 15 — Datasets, Algorithm Compatibility & Fair Benchmarking

This stage moves EPM from algorithms trained on synthetic data to an
experimentation platform. It adds a **dataset registry** with documented public sources,
**algorithm-aware, leakage-free preprocessing**, a **fair benchmark runner** and an
**Experiment Lab** section in the dashboard.

Nothing in Stages 0–13 was changed; everything here is additive.

## 1. Architecture

```
src/data/registry.py             DatasetSpec metadata for every dataset (source, licence, schema, metrics, …)
src/data/loaders.py              safe download → cache (data/raw/<id>/) → parse → LoadedDataset (+ in-memory LRU)
src/data/validation.py           schema checks for registry datasets; validation of uploaded CSVs
src/data/uploads.py              user CSV uploads exposed as datasets with id "upload_<sha>"
src/data/verify.py               `python run.py --verify-datasets`
src/preprocessing/pipelines.py   ColumnTransformer pipelines, splits, time-series windows
src/evaluation/task_metrics.py   task-specific metrics with explicit "undefined" handling
src/benchmarking/catalog.py      AlgorithmSpec catalogue, compatibility & recommendations
src/benchmarking/wrappers.py     adapters for EPM's scratch (Stage 2) and PyTorch (Stages 3/5/7) models
src/benchmarking/runner.py       run_benchmark(BenchmarkConfig) → result JSON in experiments/benchmarks/
src/webapi/local_api.py          training-capable API used by `python run.py --serve`
src/webapi/export.py             static catalogue for Vercel (web/catalog/*.json)
web/datalab.js                   Experiment Lab UI section (#datalab)
```

## 2. Datasets

| id | Dataset | Tasks | Rows × features | Size | Source | Licence | Verification status |
|---|---|---|---|---|---|---|---|
| `iris` | Iris | classification, clustering | 150 × 4 | small | [scikit-learn (bundled; originally UCI id 53)](https://archive.ics.uci.edu/dataset/53/iris) | CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause | ✅ loaded, schema-checked, benchmarked |
| `wine` | Wine Recognition | classification, clustering | 178 × 13 | small | [scikit-learn (bundled; originally UCI id 109)](https://archive.ics.uci.edu/dataset/109/wine) | CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause | ✅ loaded, schema-checked, benchmarked |
| `seeds` | UCI Seeds | clustering, classification | 210 × 7 | small | [UCI Machine Learning Repository (id 236)](https://archive.ics.uci.edu/dataset/236/seeds) | CC BY 4.0 (UCI Machine Learning Repository) | ⚠️ parser fixture-tested; real download not verified in this session |
| `wholesale_customers` | UCI Wholesale Customers | clustering | 440 × 6 | small | [UCI Machine Learning Repository (id 292)](https://archive.ics.uci.edu/dataset/292/wholesale+customers) | CC BY 4.0 (UCI Machine Learning Repository) | ⚠️ parser fixture-tested; real download not verified in this session |
| `diabetes` | Diabetes (Efron et al.) | regression | 442 × 10 | small | [scikit-learn (bundled)](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) | BSD-3-Clause (bundled with scikit-learn) | ✅ loaded, schema-checked, benchmarked |
| `breast_cancer` | Breast Cancer Wisconsin (Diagnostic) | classification | 569 × 30 | small | [scikit-learn (bundled; originally UCI id 17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause | ✅ loaded, schema-checked, benchmarked |
| `energy_efficiency` | UCI Energy Efficiency | regression | 768 × 8 | small | [UCI Machine Learning Repository (id 242)](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | CC BY 4.0 (UCI Machine Learning Repository) | ⚠️ parser fixture-tested; real download not verified in this session |
| `epm_sensor_regression` | EPM Synthetic Sensor Regression | regression | 1,200 × 5 | small | Synthetic (src/classical_ml/baselines.py) | Project-generated (MIT, same as EPM) | ✅ loaded, schema-checked, benchmarked |
| `epm_sensor_failure` | EPM Synthetic Equipment Failure | classification | 1,200 × 5 | small | Synthetic (src/classical_ml/baselines.py) | Project-generated (MIT, same as EPM) | ✅ loaded, schema-checked, benchmarked |
| `synthetic_blobs` | Synthetic Gaussian Blobs | clustering | 1,500 × 6 | small | [Synthetic (sklearn.datasets.make_blobs)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) | Project-generated | ✅ loaded, schema-checked, benchmarked |
| `digits` | Optical Handwritten Digits (8x8) | classification | 1,797 × 64 | small | [scikit-learn (bundled; test split of UCI id 80)](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause | ✅ loaded, schema-checked, benchmarked |
| `epm_sensor_telemetry` | EPM Synthetic Sensor Telemetry | forecasting | 2,000 × 5 | small | Synthetic (src/timeseries/dataset.py) | Project-generated (MIT, same as EPM) | ✅ loaded, schema-checked, benchmarked |
| `synthetic_anomalies` | Synthetic Gaussian Anomalies | anomaly_detection | 2,000 × 8 | small | Synthetic (NumPy, fixed seed) | Project-generated | ✅ loaded, schema-checked, benchmarked |
| `air_quality` | UCI Air Quality | forecasting | 9,357 × 12 | medium | [UCI Machine Learning Repository (id 360)](https://archive.ics.uci.edu/dataset/360/air+quality) | CC BY 4.0 (UCI Machine Learning Repository) - the source page states the data may be used exclusively for research purposes (no commercial use) | ⚠️ parser fixture-tested; real download not verified in this session |
| `california_housing` | California Housing | regression | 20,640 × 8 | medium | [scikit-learn (StatLib repository)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | Public (StatLib); distributed via scikit-learn's official fetcher | ⚠️ parser fixture-tested; real download not verified in this session |
| `nab_machine_temperature` | NAB Machine Temperature System Failure | anomaly_detection | 22,695 × 1 | medium | [Numenta Anomaly Benchmark (GitHub numenta/NAB)](https://github.com/numenta/NAB) | NAB repository licence (MIT-style, Copyright Numenta Inc.) | ✅ real download (GitHub) + benchmarked |
| `bank_marketing` | UCI Bank Marketing | classification | 45,211 × 15 | medium | [OpenML (data_id=1461; originally UCI id 222, bank-full)](https://www.openml.org/d/1461) | OpenML: Public; UCI original: CC BY 4.0 | ⚠️ parser fixture-tested; real download not verified in this session |
| `adult` | Adult Census Income | classification | 48,842 × 13 | medium | [OpenML (data_id=1590, version 2; originally UCI id 2)](https://www.openml.org/d/1590) | OpenML: Public; UCI original: CC BY 4.0 | ⚠️ parser fixture-tested; real download not verified in this session |
| `household_power` | Individual Household Electric Power Consumption | forecasting | 34,589 × 7 | large | [UCI Machine Learning Repository (id 235)](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | CC BY 4.0 (UCI Machine Learning Repository) | ⚠️ parser fixture-tested; real download not verified in this session |
| `credit_card_fraud` | Credit Card Fraud Detection (ULB) | anomaly_detection, classification | 284,807 × 30 | large | [OpenML (data_id=1597; originally ULB Machine Learning Group)](https://www.openml.org/d/1597) | OpenML: Public (original release by the ULB Machine Learning Group; check the original terms before redistribution) | ⚠️ parser fixture-tested; real download not verified in this session |

**Verification:** the development environment could not reach `archive.ics.uci.edu`,
`openml.org` or figshare. The UCI/OpenML/California loaders use the official URLs/IDs.
They were tested against fixture files in the documented formats, but not against the real
downloads. Run `python run.py --verify-datasets` once on a machine with internet access.
The command downloads, parses and schema-checks every dataset and writes `data/verification_report.json`.

**Not integrated (and why)**
- *Mall Customers*: only available from Kaggle mirrors without documented provenance.
- *NASA C-MAPSS / bearing data*: the NASA portal has no stable direct-download URL. Add it once a stable source is chosen.
- *Electricity Load Diagrams (UCI 321)*: ~250 MB uncompressed. Deferred until a streaming loader exists.
- *Fashion-MNIST / CIFAR-10*: EPM has no CNN, so no image model could use them.
  Digits (8×8, flattened) is provided for the MLP instead.
- *ARIMA / SARIMA / Prophet*: not implemented in EPM (statsmodels/prophet are not dependencies).

## 3. Compatibility matrix

Legend:
- `✓`: compatible.
- `✓ (≤ row limit)`: compatible only when *max train rows* is lowered to the implementation's limit.
- `✓ (window ≥ season)`: needs a lookback window at least one seasonal period long.
- `—`: not applicable.

Large datasets also need *allow large*. The XGBoost and PyTorch rows need those
optional packages installed. The matrix is generated from `check_compatibility`, so it
matches the code.

#### Regression

| Algorithm | diabetes | energy_efficiency | epm_sensor_regression | california_housing |
|---|---|---|---|---|
| Mean Baseline (`dummy_reg`) | ✓ | ✓ | ✓ | ✓ |
| Linear Regression (scratch, normal eq.) (`linreg_scratch_ne`) | ✓ | ✓ | ✓ | ✓ |
| Linear Regression (scratch, gradient descent) (`linreg_scratch_gd`) | ✓ | ✓ | ✓ | ✓ |
| Linear Regression (OLS) (`ols`) | ✓ | ✓ | ✓ | ✓ |
| Ridge Regression (L2) (`ridge`) | ✓ | ✓ | ✓ | ✓ |
| Lasso Regression (L1) (`lasso`) | ✓ | ✓ | ✓ | ✓ |
| Elastic Net (`elasticnet`) | ✓ | ✓ | ✓ | ✓ |
| KNN Regressor (scratch, k=5) (`knn_reg_scratch`) | ✓ | ✓ | ✓ | ✓ (≤ row limit) |
| KNN Regressor (k=5) (`knn_reg`) | ✓ | ✓ | ✓ | ✓ |
| Decision Tree Regressor (scratch, depth 5) (`dt_reg_scratch`) | ✓ | ✓ | ✓ | ✓ (≤ row limit) |
| Decision Tree Regressor (depth 8) (`dt_reg`) | ✓ | ✓ | ✓ | ✓ |
| Random Forest Regressor (`rf_reg`) | ✓ | ✓ | ✓ | ✓ |
| Support Vector Regression (RBF) (`svr`) | ✓ | ✓ | ✓ | ✓ |
| Gradient Boosting Regressor (`gbr`) | ✓ | ✓ | ✓ | ✓ |
| XGBoost Regressor (`xgb_reg`) | ✓ | ✓ | ✓ | ✓ |

#### Classification

| Algorithm | iris | wine | seeds | breast_cancer | epm_sensor_failure | digits | bank_marketing | adult | credit_card_fraud |
|---|---|---|---|---|---|---|---|---|---|
| Majority-Class Baseline (`dummy_clf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Logistic Regression (scratch) (`logreg_scratch`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (≤ row limit) |
| Logistic Regression (`logreg`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| KNN Classifier (scratch, k=5) (`knn_clf_scratch`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (≤ row limit) | ✓ (≤ row limit) | ✓ (≤ row limit) |
| KNN Classifier (k=5) (`knn_clf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Decision Tree (scratch, depth 5) (`dt_clf_scratch`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (≤ row limit) | ✓ (≤ row limit) | ✓ (≤ row limit) |
| Decision Tree (depth 8) (`dt_clf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Random Forest (`rf_clf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Support Vector Machine (RBF) (`svc`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (≤ row limit) | ✓ (≤ row limit) | ✓ (≤ row limit) |
| Gaussian Naive Bayes (`gnb`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Gradient Boosting (`gb_clf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| XGBoost Classifier (`xgb_clf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MLP (PyTorch, Stage 3) (`mlp_torch`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (≤ row limit) |

#### Clustering

| Algorithm | iris | wine | seeds | wholesale_customers | synthetic_blobs |
|---|---|---|---|---|---|
| K-Means (`kmeans`) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Agglomerative (Ward) (`agglomerative`) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Gaussian Mixture Model (`gmm`) | ✓ | ✓ | ✓ | ✓ | ✓ |
| DBSCAN (`dbscan`) | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Forecasting

| Algorithm | epm_sensor_telemetry | air_quality | household_power |
|---|---|---|---|
| Naive (persistence) (`naive`) | ✓ | ✓ | ✓ |
| Seasonal Naive (`seasonal_naive`) | ✓ (window ≥ season) | ✓ | ✓ |
| Linear AR (Ridge on lag window) (`linear_ar`) | ✓ | ✓ | ✓ |
| Random Forest (lag window) (`rf_forecast`) | ✓ | ✓ | ✓ |
| XGBoost (lag window) (`xgb_forecast`) | ✓ | ✓ | ✓ |
| LSTM (Stage 5) (`lstm`) | ✓ | ✓ | ✓ |
| GRU (Stage 5) (`gru`) | ✓ | ✓ | ✓ |
| Temporal Transformer (Stage 5) (`ts_transformer`) | ✓ | ✓ | ✓ |

#### Anomaly Detection

| Algorithm | synthetic_anomalies | nab_machine_temperature | credit_card_fraud |
|---|---|---|---|
| Isolation Forest (`iforest`) | ✓ | ✓ | ✓ |
| Local Outlier Factor (`lof`) | ✓ | ✓ | ✓ (≤ row limit) |
| One-Class SVM (`ocsvm`) | ✓ | ✓ (≤ row limit) | ✓ (≤ row limit) |
| Autoencoder (Stage 7) (`autoencoder`) | ✓ | ✓ | ✓ |


## 4. Preprocessing (fitted on training data only)

| Step | Where | Notes |
|---|---|---|
| Exact-duplicate removal | before splitting | uses no target statistics |
| Median / most-frequent imputation | inside the Pipeline | statistics from training rows |
| Standard scaling | inside the Pipeline | **only** for scale-sensitive models (`scale=auto`); trees are not scaled |
| One-hot encoding | inside the Pipeline | `handle_unknown='infrequent_if_exist'`, rare levels grouped |
| Winsorisation (optional) | inside the Pipeline | quantiles learned on training rows |
| log1p (Wholesale) | inside the Pipeline | fixed transform for skewed spend |
| Class weighting (optional) | estimator | `class_weight='balanced'` where supported |
| Chronological split | time series | train → val → test, never shuffled |
| Causal forward-fill | time series | gaps filled from the past only; leading gaps from training medians |
| Per-channel z-score | time series | mean/std of the training segment only |
| Sliding windows | time series | targets never cross a segment boundary |
| Lag-window features | NAB anomaly detection | past values only |

Every result records the exact preprocessing steps used for each model.

## 5. Benchmarking & metrics

* All models in a run share one split (fingerprinted) and one seed.
* Cross-validation is optional: k-fold (stratified for classification), or rolling-origin for forecasting. It runs on the training part only.
* Tuning is optional: a small grid search with 3-fold CV on the training split only.
* Each model records:
  * training time;
  * batch inference latency per sample, including the preprocessing pipeline;
  * serialized size;
  * parameter count, where available.
* The summary uses several criteria instead of one winner:
  * best model by the primary metric;
  * fastest model;
  * smallest model;
  * the **Pareto front** (metric vs latency);
  * whether each model beats the baseline (mean / majority / naive).

| Task | Metrics | Primary |
|---|---|---|
| Regression | MAE, MSE, RMSE, R², median AE | RMSE |
| Classification | accuracy, balanced accuracy, precision/recall/F1 (binary, or macro for multiclass), ROC-AUC, PR-AUC, confusion matrix; group fairness gaps for Adult | F1 |
| Clustering | silhouette, Davies–Bouldin, Calinski–Harabasz, noise fraction; ARI/NMI against held-out labels | silhouette |
| Forecasting | MAE, RMSE, sMAPE, MASE, skill vs naive, per-horizon RMSE | RMSE |
| Anomaly detection | precision, recall, F1, PR-AUC, ROC-AUC, detection delay (series) | PR-AUC |

Undefined values are reported as `null` with a note, never as a misleading number.
Examples: AUC when only one class is present, silhouette with one cluster, R² with a constant target, MASE on a flat series.

## 6. Running it

```bash
python run.py --serve                       # dashboard + local API → http://localhost:3000/#datalab
python run.py --benchmark iris --cv 5       # terminal benchmark (recommended algorithms)
python run.py --benchmark adult --algorithms logreg rf_clf xgb_clf --max-train-rows 20000
python run.py --benchmark household_power --allow-large
python run.py --verify-datasets             # download + schema-check every dataset
python run.py --export-catalog              # refresh web/catalog/*.json for static hosting
pytest tests/ -v
```

Environment variables:
- `EPM_DATA_HOME`: cache root (default `data/`).
- `EPM_OFFLINE=1`: forbid downloads.
- `EPM_MAX_JOBS`: parallel training jobs in the local server (default 1).

On Vercel the API is still stdlib-only and read-only. `/api/datasets`, `/api/algorithms` and
`/api/benchmarks/history` serve the exported catalogue. Training, download and upload
endpoints return `501` with instructions to run locally.

## 7. Known limitations

* Latency on very small test sets is dominated by per-call pipeline overhead.
* `model_size_kb` is the pickle size of the fitted pipeline (or the PyTorch `state_dict`), not RAM use.
* NAB labels whole windows, so point-wise precision is pessimistic; detection delay is the more faithful measure.
* DBSCAN's internal indices exclude its noise points, which can flatter it. The runner flags this.
* Training jobs cannot be cancelled mid-model. Resource use is bounded by row limits, epochs and `allow_large`.
