"""
EPM Algorithm Catalogue & Dataset Compatibility
===============================================

One :class:`AlgorithmSpec` per benchmarkable algorithm. Specs reuse EPM's own
implementations where they exist (``framework='scratch'`` = Stage 2 NumPy
models; PyTorch models from Stages 3/5/7) and scikit-learn / XGBoost otherwise.

Compatibility is decided explicitly by :func:`check_compatibility`; the runner
never silently drops or substitutes a model - every exclusion carries a reason.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.data.registry import DatasetSpec, TASKS, list_datasets, get_dataset_spec


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    task: str
    family: str
    framework: str               # scratch | scikit-learn | xgboost | pytorch | baseline
    input_format: str            # tabular | sequence
    factory: Callable[[Dict[str, Any]], Any]
    needs_scaling: bool = True
    max_train_rows: Optional[int] = None
    cost: str = "low"            # low | medium | high
    interpretability: str = "medium"
    supports_proba: bool = False
    supports_class_weight: bool = False
    requires: Optional[str] = None   # optional python package
    tune_grid: Dict[str, List[Any]] = field(default_factory=dict)
    recommended: bool = True     # part of the default selection
    notes: str = ""

    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "task": self.task, "family": self.family,
            "framework": self.framework, "input_format": self.input_format,
            "needs_scaling": self.needs_scaling, "max_train_rows": self.max_train_rows,
            "cost": self.cost, "interpretability": self.interpretability,
            "supports_proba": self.supports_proba, "supports_class_weight": self.supports_class_weight,
            "requires": self.requires, "available": self.available(),
            "tunable": sorted(self.tune_grid), "recommended": self.recommended, "notes": self.notes,
        }


# -----------------------------------------------------------------------------
# Factories (lazy imports keep the catalogue importable without torch/xgboost)
# -----------------------------------------------------------------------------

def _cw(ctx):  # class_weight helper
    return "balanced" if ctx.get("class_weight") == "balanced" else None


def _seed(ctx):
    return int(ctx.get("seed", 42))


def _sk(path: str, **fixed):
    mod, cls = path.rsplit(".", 1)

    def make(ctx: Dict[str, Any]):
        klass = getattr(importlib.import_module(mod), cls)
        params = dict(fixed)
        params.update(ctx.get("params", {}))
        return klass(**params)

    return make





def _scratch(path: str, kind: str = "regressor", **fixed):
    """Stage 2 NumPy model wrapped so it can live inside a scikit-learn Pipeline."""
    mod, cls = path.rsplit(".", 1)

    def make(ctx):
        from src.benchmarking.wrappers import ScratchAdapter

        klass = getattr(importlib.import_module(mod), cls)
        params = dict(fixed)
        params.update(ctx.get("params", {}))
        return ScratchAdapter(klass, params, kind)

    return make


def _with_seed(path: str, **fixed):
    mod, cls = path.rsplit(".", 1)

    def make(ctx):
        klass = getattr(importlib.import_module(mod), cls)
        params = dict(fixed)
        params["random_state"] = _seed(ctx)
        params.update(ctx.get("params", {}))
        return klass(**params)

    return make


def _with_seed_cw(path: str, **fixed):
    base = _with_seed(path, **fixed)

    def make(ctx):
        est = base(ctx)
        if _cw(ctx):
            est.set_params(class_weight="balanced")
        return est

    return make


def _xgb_clf(ctx):
    import xgboost as xgb

    p = dict(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=_seed(ctx),
             eval_metric="logloss", verbosity=0, n_jobs=1)
    p.update(ctx.get("params", {}))
    return xgb.XGBClassifier(**p)


def _xgb_reg(ctx):
    import xgboost as xgb

    p = dict(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=_seed(ctx), verbosity=0, n_jobs=1)
    p.update(ctx.get("params", {}))
    return xgb.XGBRegressor(**p)


def _dbscan(ctx):
    from sklearn.cluster import DBSCAN

    p = {"eps": ctx.get("dbscan_eps", 0.5), "min_samples": ctx.get("dbscan_min_samples", 5)}
    p.update(ctx.get("params", {}))
    return DBSCAN(**p)


def _flat(est_factory):
    def make(ctx):
        from src.benchmarking.wrappers import FlatWindowForecaster

        return FlatWindowForecaster(est_factory(ctx))

    return make


def _torch_fc(kind):
    def make(ctx):
        from src.benchmarking.wrappers import TorchSequenceForecaster

        p = {"kind": kind, "epochs": ctx.get("epochs", 30), "seed": _seed(ctx)}
        p.update(ctx.get("params", {}))
        return TorchSequenceForecaster(**p)

    return make


def _naive(ctx):
    from src.benchmarking.wrappers import NaiveForecaster

    return NaiveForecaster(target_channel=ctx.get("target_channel", 0), horizon=ctx.get("horizon", 1))


def _snaive(ctx):
    from src.benchmarking.wrappers import SeasonalNaiveForecaster

    return SeasonalNaiveForecaster(target_channel=ctx.get("target_channel", 0), horizon=ctx.get("horizon", 1),
                                   period=ctx.get("seasonal_period", 24))


def _mlp(ctx):
    from src.benchmarking.wrappers import TorchMLPClassifier

    p = {"epochs": ctx.get("epochs", 60), "seed": _seed(ctx)}
    p.update(ctx.get("params", {}))
    return TorchMLPClassifier(**p)


def _ae(ctx):
    from src.benchmarking.wrappers import AutoencoderDetector

    p = {"epochs": ctx.get("epochs", 40), "seed": _seed(ctx)}
    p.update(ctx.get("params", {}))
    return AutoencoderDetector(**p)


def _kmeans(ctx):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=ctx.get("n_clusters", 3), n_init=10, random_state=_seed(ctx), **ctx.get("params", {}))


def _agglo(ctx):
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=ctx.get("n_clusters", 3), linkage="ward", **ctx.get("params", {}))


def _gmm(ctx):
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(n_components=ctx.get("n_clusters", 3), covariance_type="full", n_init=3,
                           random_state=_seed(ctx), **ctx.get("params", {}))


def _iforest(ctx):
    from sklearn.ensemble import IsolationForest

    return IsolationForest(n_estimators=200, random_state=_seed(ctx), **ctx.get("params", {}))


def _lof(ctx):
    from sklearn.neighbors import LocalOutlierFactor

    return LocalOutlierFactor(n_neighbors=20, novelty=True, **ctx.get("params", {}))


def _ocsvm(ctx):
    from sklearn.svm import OneClassSVM

    p = {"kernel": "rbf", "gamma": "scale", "nu": min(0.5, max(0.01, ctx.get("contamination", 0.05)))}
    p.update(ctx.get("params", {}))
    return OneClassSVM(**p)


A = AlgorithmSpec
_ALGOS: List[AlgorithmSpec] = [
    # ---------------------------- Regression ---------------------------------
    A("dummy_reg", "Mean Baseline", "regression", "baseline", "baseline", "tabular",
      _sk("sklearn.dummy.DummyRegressor"), needs_scaling=False, interpretability="high",
      notes="Predicts the training mean; any useful model must beat it."),
    A("linreg_scratch_ne", "Linear Regression (scratch, normal eq.)", "regression", "linear", "scratch", "tabular",
      _scratch("src.classical_ml.linear_regression.LinearRegressionScratch", method="normal_equation"),
      interpretability="high", max_train_rows=200_000, notes="EPM Stage 2 NumPy implementation."),
    A("linreg_scratch_gd", "Linear Regression (scratch, gradient descent)", "regression", "linear", "scratch", "tabular",
      _scratch("src.classical_ml.linear_regression.LinearRegressionScratch", method="gradient_descent",
               learning_rate=0.01, n_iterations=1000),
      interpretability="high", max_train_rows=100_000, recommended=False, notes="EPM Stage 2 NumPy implementation."),
    A("ols", "Linear Regression (OLS)", "regression", "linear", "scikit-learn", "tabular",
      _sk("sklearn.linear_model.LinearRegression"), interpretability="high"),
    A("ridge", "Ridge Regression (L2)", "regression", "linear", "scikit-learn", "tabular",
      _sk("sklearn.linear_model.Ridge", alpha=1.0), interpretability="high",
      tune_grid={"alpha": [0.1, 1.0, 10.0]}),
    A("lasso", "Lasso Regression (L1)", "regression", "linear", "scikit-learn", "tabular",
      _with_seed("sklearn.linear_model.Lasso", alpha=0.01, max_iter=5000), interpretability="high",
      tune_grid={"alpha": [0.001, 0.01, 0.1]}, notes="Added: sparse linear baseline."),
    A("elasticnet", "Elastic Net", "regression", "linear", "scikit-learn", "tabular",
      _with_seed("sklearn.linear_model.ElasticNet", alpha=0.01, l1_ratio=0.5, max_iter=5000), interpretability="high",
      tune_grid={"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]}, recommended=False),
    A("knn_reg_scratch", "KNN Regressor (scratch, k=5)", "regression", "neighbors", "scratch", "tabular",
      _scratch("src.classical_ml.knn.KNNRegressorScratch", n_neighbors=5), max_train_rows=5_000, cost="medium",
      recommended=False, notes="O(n_train x n_test) memory; limited to 5k training rows."),
    A("knn_reg", "KNN Regressor (k=5)", "regression", "neighbors", "scikit-learn", "tabular",
      _sk("sklearn.neighbors.KNeighborsRegressor", n_neighbors=5), tune_grid={"n_neighbors": [3, 5, 11]}),
    A("dt_reg_scratch", "Decision Tree Regressor (scratch, depth 5)", "regression", "tree", "scratch", "tabular",
      _scratch("src.classical_ml.decision_tree.DecisionTreeRegressorScratch", max_depth=5), needs_scaling=False,
      max_train_rows=5_000, cost="medium", interpretability="high", recommended=False),
    A("dt_reg", "Decision Tree Regressor (depth 8)", "regression", "tree", "scikit-learn", "tabular",
      _with_seed("sklearn.tree.DecisionTreeRegressor", max_depth=8), needs_scaling=False, interpretability="high",
      tune_grid={"max_depth": [4, 8, 12]}),
    A("rf_reg", "Random Forest Regressor", "regression", "ensemble", "scikit-learn", "tabular",
      _with_seed("sklearn.ensemble.RandomForestRegressor", n_estimators=200, n_jobs=1), needs_scaling=False,
      cost="medium", interpretability="low", tune_grid={"max_depth": [None, 12], "min_samples_leaf": [1, 5]}),
    A("svr", "Support Vector Regression (RBF)", "regression", "kernel", "scikit-learn", "tabular",
      _sk("sklearn.svm.SVR", kernel="rbf", C=10.0), max_train_rows=15_000, cost="high", interpretability="low",
      tune_grid={"C": [1.0, 10.0, 100.0]}, notes="Kernel method: training cost grows ~quadratically."),
    A("gbr", "Gradient Boosting Regressor", "regression", "boosting", "scikit-learn", "tabular",
      _with_seed("sklearn.ensemble.GradientBoostingRegressor", n_estimators=200, max_depth=3),
      needs_scaling=False, cost="medium", interpretability="low",
      tune_grid={"learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4]}),
    A("xgb_reg", "XGBoost Regressor", "regression", "boosting", "xgboost", "tabular", _xgb_reg,
      needs_scaling=False, cost="medium", interpretability="low", requires="xgboost",
      tune_grid={"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}),
    # ---------------------------- Classification -----------------------------
    A("dummy_clf", "Majority-Class Baseline", "classification", "baseline", "baseline", "tabular",
      _sk("sklearn.dummy.DummyClassifier", strategy="prior"), needs_scaling=False, interpretability="high",
      supports_proba=True, notes="Always predicts the most frequent class."),
    A("logreg_scratch", "Logistic Regression (scratch)", "classification", "linear", "scratch", "tabular",
      _scratch("src.classical_ml.logistic_regression.LogisticRegressionScratch", kind="classifier", learning_rate=0.1, n_iterations=1000),
      max_train_rows=50_000, interpretability="high", supports_proba=True,
      notes="EPM Stage 2 NumPy implementation (one-vs-rest for multiclass)."),
    A("logreg", "Logistic Regression", "classification", "linear", "scikit-learn", "tabular",
      _with_seed_cw("sklearn.linear_model.LogisticRegression", max_iter=2000), interpretability="high",
      supports_proba=True, supports_class_weight=True, tune_grid={"C": [0.1, 1.0, 10.0]}),
    A("knn_clf_scratch", "KNN Classifier (scratch, k=5)", "classification", "neighbors", "scratch", "tabular",
      _scratch("src.classical_ml.knn.KNNClassifierScratch", kind="classifier", n_neighbors=5), max_train_rows=5_000, cost="medium",
      supports_proba=True, recommended=False, notes="O(n_train x n_test) memory; limited to 5k training rows."),
    A("knn_clf", "KNN Classifier (k=5)", "classification", "neighbors", "scikit-learn", "tabular",
      _sk("sklearn.neighbors.KNeighborsClassifier", n_neighbors=5), supports_proba=True,
      tune_grid={"n_neighbors": [3, 5, 11]}),
    A("dt_clf_scratch", "Decision Tree (scratch, depth 5)", "classification", "tree", "scratch", "tabular",
      _scratch("src.classical_ml.decision_tree.DecisionTreeClassifierScratch", kind="classifier", max_depth=5), needs_scaling=False,
      max_train_rows=5_000, cost="medium", interpretability="high", recommended=False),
    A("dt_clf", "Decision Tree (depth 8)", "classification", "tree", "scikit-learn", "tabular",
      _with_seed_cw("sklearn.tree.DecisionTreeClassifier", max_depth=8), needs_scaling=False, interpretability="high",
      supports_proba=True, supports_class_weight=True, tune_grid={"max_depth": [4, 8, 12]}),
    A("rf_clf", "Random Forest", "classification", "ensemble", "scikit-learn", "tabular",
      _with_seed_cw("sklearn.ensemble.RandomForestClassifier", n_estimators=200, n_jobs=1), needs_scaling=False,
      cost="medium", interpretability="low", supports_proba=True, supports_class_weight=True,
      tune_grid={"max_depth": [None, 12], "min_samples_leaf": [1, 5]}),
    A("svc", "Support Vector Machine (RBF)", "classification", "kernel", "scikit-learn", "tabular",
      _with_seed_cw("sklearn.svm.SVC", kernel="rbf", C=1.0, probability=True), max_train_rows=15_000, cost="high",
      interpretability="low", supports_proba=True, supports_class_weight=True, tune_grid={"C": [0.1, 1.0, 10.0]},
      notes="probability=True uses internal Platt scaling (5-fold) on the training data only."),
    A("gnb", "Gaussian Naive Bayes", "classification", "probabilistic", "scikit-learn", "tabular",
      _sk("sklearn.naive_bayes.GaussianNB"), needs_scaling=False, interpretability="medium", supports_proba=True,
      notes="Added: probabilistic baseline. Assumes conditionally independent Gaussian features."),
    A("gb_clf", "Gradient Boosting", "classification", "boosting", "scikit-learn", "tabular",
      _with_seed("sklearn.ensemble.GradientBoostingClassifier", n_estimators=150, max_depth=3), needs_scaling=False,
      cost="medium", interpretability="low", supports_proba=True,
      tune_grid={"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
    A("xgb_clf", "XGBoost Classifier", "classification", "boosting", "xgboost", "tabular", _xgb_clf,
      needs_scaling=False, cost="medium", interpretability="low", supports_proba=True, requires="xgboost",
      tune_grid={"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}),
    A("mlp_torch", "MLP (PyTorch, Stage 3)", "classification", "neural", "pytorch", "tabular", _mlp,
      cost="medium", interpretability="low", supports_proba=True, requires="torch", recommended=False,
      max_train_rows=100_000, notes="EPM's PyTorchMLP (1 hidden layer) trained with Adam."),
    # ---------------------------- Clustering ---------------------------------
    A("kmeans", "K-Means", "clustering", "centroid", "scikit-learn", "tabular", _kmeans, interpretability="high"),
    A("agglomerative", "Agglomerative (Ward)", "clustering", "hierarchical", "scikit-learn", "tabular", _agglo,
      max_train_rows=15_000, cost="medium", interpretability="medium", notes="O(n^2) memory."),
    A("gmm", "Gaussian Mixture Model", "clustering", "probabilistic", "scikit-learn", "tabular", _gmm,
      cost="medium", interpretability="medium"),
    A("dbscan", "DBSCAN", "clustering", "density", "scikit-learn", "tabular", _dbscan, cost="medium",
      interpretability="medium", max_train_rows=50_000,
      notes="eps is set from the k-distance distribution of the (scaled) data unless given; finds its own cluster count and marks noise."),
    # ---------------------------- Anomaly detection --------------------------
    A("iforest", "Isolation Forest", "anomaly_detection", "ensemble", "scikit-learn", "tabular", _iforest,
      needs_scaling=False, cost="medium", interpretability="low"),
    A("lof", "Local Outlier Factor", "anomaly_detection", "neighbors", "scikit-learn", "tabular", _lof,
      max_train_rows=50_000, cost="medium", interpretability="medium"),
    A("ocsvm", "One-Class SVM", "anomaly_detection", "kernel", "scikit-learn", "tabular", _ocsvm,
      max_train_rows=10_000, cost="high", interpretability="low"),
    A("autoencoder", "Autoencoder (Stage 7)", "anomaly_detection", "neural", "pytorch", "tabular", _ae,
      cost="medium", interpretability="low", requires="torch", max_train_rows=200_000,
      notes="EPM StandardAutoencoder; anomaly score = reconstruction MSE."),
    # ---------------------------- Forecasting --------------------------------
    A("naive", "Naive (persistence)", "forecasting", "baseline", "baseline", "sequence", _naive,
      needs_scaling=False, interpretability="high", notes="Reference for skill scores."),
    A("seasonal_naive", "Seasonal Naive", "forecasting", "baseline", "baseline", "sequence", _snaive,
      needs_scaling=False, interpretability="high", notes="Needs lookback window >= seasonal period."),
    A("linear_ar", "Linear AR (Ridge on lag window)", "forecasting", "linear", "scikit-learn", "sequence",
      _flat(_sk("sklearn.linear_model.Ridge", alpha=1.0)), interpretability="high"),
    A("rf_forecast", "Random Forest (lag window)", "forecasting", "ensemble", "scikit-learn", "sequence",
      _flat(_with_seed("sklearn.ensemble.RandomForestRegressor", n_estimators=100, max_depth=12, n_jobs=1)),
      cost="medium", interpretability="low"),
    A("xgb_forecast", "XGBoost (lag window)", "forecasting", "boosting", "xgboost", "sequence",
      _flat(_xgb_reg), cost="medium", interpretability="low", requires="xgboost"),
    A("lstm", "LSTM (Stage 5)", "forecasting", "recurrent", "pytorch", "sequence", _torch_fc("lstm"),
      cost="high", interpretability="low", requires="torch", max_train_rows=60_000),
    A("gru", "GRU (Stage 5)", "forecasting", "recurrent", "pytorch", "sequence", _torch_fc("gru"),
      cost="high", interpretability="low", requires="torch", max_train_rows=60_000, recommended=False),
    A("ts_transformer", "Temporal Transformer (Stage 5)", "forecasting", "attention", "pytorch", "sequence",
      _torch_fc("transformer"), cost="high", interpretability="low", requires="torch", max_train_rows=60_000,
      recommended=False),
]

ALGORITHMS: Dict[str, AlgorithmSpec] = {a.id: a for a in _ALGOS}
assert all(a.task in TASKS for a in _ALGOS)


def get_algorithm(algorithm_id: str) -> AlgorithmSpec:
    try:
        return ALGORITHMS[algorithm_id]
    except KeyError:
        raise KeyError(f"Unknown algorithm '{algorithm_id}'. Known: {sorted(ALGORITHMS)}") from None


def list_algorithms(task: Optional[str] = None) -> List[AlgorithmSpec]:
    return [a for a in _ALGOS if task is None or a.task == task]


# =============================================================================
# Compatibility & recommendation
# =============================================================================

def check_compatibility(
    algo: AlgorithmSpec,
    spec: DatasetSpec,
    task: Optional[str] = None,
    n_train: Optional[int] = None,
    window: Optional[int] = None,
    allow_large: bool = False,
) -> Tuple[bool, List[str]]:
    """Return (compatible, reasons). Reasons explain every incompatibility."""
    reasons: List[str] = []
    task = task or algo.task
    if algo.task != task:
        reasons.append(f"{algo.name} is a {algo.task} algorithm, not {task}")
    if not spec.supports(task):
        reasons.append(f"dataset '{spec.id}' is not defined for {task} (supports: {', '.join(spec.tasks)})")
    if task in ("regression",) and spec.target_type not in ("continuous",):
        reasons.append("regression needs a continuous target")
    if task == "classification" and spec.target_type not in ("binary", "multiclass"):
        reasons.append("classification needs a categorical target")
    if not algo.available():
        reasons.append(f"optional dependency '{algo.requires}' is not installed")
    if spec.size_class == "large" and not allow_large:
        reasons.append("large dataset: enable 'allow large datasets' to use it")
    rows = n_train if n_train is not None else (int((spec.n_samples or 0) * 0.7) or None)
    if algo.max_train_rows and rows and rows > algo.max_train_rows:
        reasons.append(
            f"~{rows:,} training rows exceed this implementation's limit of {algo.max_train_rows:,}; "
            f"set 'max train rows' <= {algo.max_train_rows:,} to include it"
        )
    if algo.id == "seasonal_naive" and spec.seasonal_period:
        w = window or spec.default_window or 0
        if w < spec.seasonal_period:
            reasons.append(f"lookback window {w} < seasonal period {spec.seasonal_period}")
    return (not reasons), reasons


def compatible_algorithms(spec: DatasetSpec, task: str, **kw: Any) -> Tuple[List[AlgorithmSpec], Dict[str, List[str]]]:
    ok, excluded = [], {}
    for a in list_algorithms(task):
        good, why = check_compatibility(a, spec, task, **kw)
        if good:
            ok.append(a)
        else:
            excluded[a.id] = why
    return ok, excluded


def recommend_algorithms(dataset_id: str, task: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
    spec = get_dataset_spec(dataset_id)
    task = task or spec.tasks[0]
    ok, excluded = compatible_algorithms(spec, task, **kw)
    recommended = [a.id for a in ok if a.recommended and (a.cost != "high" or spec.size_class == "small")]
    return {"dataset_id": dataset_id, "task": task, "compatible": [a.id for a in ok],
            "recommended": recommended, "excluded": excluded}


def recommend_datasets(algorithm_id: str, budget: str = "quick", allow_large: bool = False,
                       max_memory_mb: Optional[float] = None) -> Dict[str, Any]:
    """Datasets compatible with ``algorithm_id``.

    ``budget='quick'`` ranks small datasets first (fast experiments);
    ``budget='realistic'`` ranks the largest compatible datasets first.
    """
    algo = get_algorithm(algorithm_id)
    rows, rejected = [], {}
    for spec in list_datasets(algo.task):
        good, why = check_compatibility(algo, spec, algo.task, allow_large=allow_large)
        if max_memory_mb is not None and spec.approx_memory_mb > max_memory_mb:
            good, why = False, why + [f"needs ~{spec.approx_memory_mb} MB > limit {max_memory_mb} MB"]
        if good:
            rows.append(spec)
        else:
            rejected[spec.id] = why
    rows.sort(key=lambda s: s.n_samples or 0, reverse=(budget == "realistic"))
    return {"algorithm_id": algorithm_id, "task": algo.task, "budget": budget,
            "datasets": [s.id for s in rows], "rejected": rejected}
