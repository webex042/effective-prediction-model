"""
Slot (entity) extraction for the EPM chatbot.

Three zero-dependency stages, in order:
  1. exact lookup of dataset / algorithm ids, names and curated synonyms;
  2. regular expressions for numeric settings (cv, seed, horizon, window, ...);
  3. fuzzy fallback (difflib) for misspelt dataset / algorithm names.

Algorithms are first captured as *families* ("random forest") and only resolved
to concrete catalogue ids ("rf_clf" / "rf_reg" / "rf_forecast") once the task is
known, so "random forest on diabetes" and "random forest on iris" both work.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.benchmarking.catalog import ALGORITHMS, list_algorithms
from src.data.registry import DATASETS

# -----------------------------------------------------------------------------
# Slot container
# -----------------------------------------------------------------------------


@dataclass
class ChatSlots:
    task: Optional[str] = None
    dataset_id: Optional[str] = None
    algorithm_ids: List[str] = field(default_factory=list)
    algorithm_families: List[str] = field(default_factory=list)
    all_algorithms: bool = False
    cv_folds: Optional[int] = None
    seed: Optional[int] = None
    max_train_rows: Optional[int] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    # Phase 2
    horizon: Optional[int] = None
    window: Optional[int] = None
    contamination: Optional[float] = None
    n_clusters: Optional[int] = None
    epochs: Optional[int] = None
    allow_large: bool = False
    # Phase 3
    generative_model: Optional[str] = None   # vae | ae
    num_samples: Optional[int] = None
    latent_dim: Optional[int] = None
    n_rows: Optional[int] = None             # preview rows
    concept: Optional[str] = None            # explain topic text

    def merge(self, other: "ChatSlots") -> None:
        """Copy every value that ``other`` actually set into ``self``."""
        default = ChatSlots()
        for k, v in other.__dict__.items():
            if v != getattr(default, k):
                if k == "hyperparams":
                    self.hyperparams.update(v)
                else:
                    setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {}, False)}


# -----------------------------------------------------------------------------
# Vocabularies
# -----------------------------------------------------------------------------

TASK_WORDS = {
    "classification": "classification", "classify": "classification", "classifier": "classification",
    "regression": "regression", "regress": "regression", "regressor": "regression",
    "clustering": "clustering", "cluster": "clustering", "clusters": "clustering",
    "forecasting": "forecasting", "forecast": "forecasting", "time series": "forecasting",
    "time-series": "forecasting", "anomaly": "anomaly_detection", "anomalies": "anomaly_detection",
    "anomaly detection": "anomaly_detection", "outlier": "anomaly_detection", "outliers": "anomaly_detection",
}

# Extra human names for datasets (ids and registry names are added automatically).
DATASET_SYNONYMS: Dict[str, str] = {
    "iris": "iris", "iris flowers": "iris", "wine": "wine", "wines": "wine",
    "breast cancer": "breast_cancer", "cancer": "breast_cancer", "wdbc": "breast_cancer",
    "digits": "digits", "handwritten digits": "digits", "digit images": "digits",
    "diabetes": "diabetes", "california": "california_housing", "california housing": "california_housing",
    "house prices": "california_housing", "housing": "california_housing",
    "energy": "energy_efficiency", "energy efficiency": "energy_efficiency", "heating load": "energy_efficiency",
    "adult": "adult", "census": "adult", "census income": "adult", "income": "adult",
    "bank": "bank_marketing", "bank marketing": "bank_marketing",
    "wholesale": "wholesale_customers", "wholesale customers": "wholesale_customers",
    "seeds": "seeds", "wheat seeds": "seeds", "blobs": "synthetic_blobs", "gaussian blobs": "synthetic_blobs",
    "household power": "household_power", "household electricity": "household_power",
    "electricity": "household_power", "power consumption": "household_power",
    "air quality": "air_quality", "air": "air_quality", "pollution": "air_quality",
    "sensor telemetry": "epm_sensor_telemetry", "telemetry": "epm_sensor_telemetry",
    "sensor data": "epm_sensor_telemetry", "sensor": "epm_sensor_telemetry",
    "sensor regression": "epm_sensor_regression", "equipment failure": "epm_sensor_failure",
    "sensor failure": "epm_sensor_failure", "nab": "nab_machine_temperature",
    "machine temperature": "nab_machine_temperature", "numenta": "nab_machine_temperature",
    "credit card": "credit_card_fraud", "credit card fraud": "credit_card_fraud", "fraud": "credit_card_fraud",
    "synthetic anomalies": "synthetic_anomalies", "synthetic anomaly": "synthetic_anomalies",
}

# family -> {task: algorithm id}
ALGORITHM_FAMILIES: Dict[str, Dict[str, str]] = {
    "baseline": {"regression": "dummy_reg", "classification": "dummy_clf", "forecasting": "naive"},
    "linear": {"regression": "ols", "classification": "logreg", "forecasting": "linear_ar"},
    "linear regression": {"regression": "ols"},
    "scratch linear regression": {"regression": "linreg_scratch_ne"},
    "logistic regression": {"classification": "logreg"},
    "scratch logistic regression": {"classification": "logreg_scratch"},
    "ridge": {"regression": "ridge"},
    "lasso": {"regression": "lasso"},
    "elastic net": {"regression": "elasticnet"},
    "knn": {"regression": "knn_reg", "classification": "knn_clf"},
    "scratch knn": {"regression": "knn_reg_scratch", "classification": "knn_clf_scratch"},
    "decision tree": {"regression": "dt_reg", "classification": "dt_clf"},
    "scratch decision tree": {"regression": "dt_reg_scratch", "classification": "dt_clf_scratch"},
    "random forest": {"regression": "rf_reg", "classification": "rf_clf", "forecasting": "rf_forecast"},
    "svm": {"regression": "svr", "classification": "svc", "anomaly_detection": "ocsvm"},
    "naive bayes": {"classification": "gnb"},
    "gradient boosting": {"regression": "gbr", "classification": "gb_clf"},
    "xgboost": {"regression": "xgb_reg", "classification": "xgb_clf", "forecasting": "xgb_forecast"},
    "mlp": {"classification": "mlp_torch"},
    "kmeans": {"clustering": "kmeans"},
    "agglomerative": {"clustering": "agglomerative"},
    "gmm": {"clustering": "gmm"},
    "dbscan": {"clustering": "dbscan"},
    "isolation forest": {"anomaly_detection": "iforest"},
    "lof": {"anomaly_detection": "lof"},
    "one class svm": {"anomaly_detection": "ocsvm"},
    "autoencoder": {"anomaly_detection": "autoencoder"},
    "naive": {"forecasting": "naive"},
    "seasonal naive": {"forecasting": "seasonal_naive"},
    "lstm": {"forecasting": "lstm"},
    "gru": {"forecasting": "gru"},
    "transformer": {"forecasting": "ts_transformer"},
}

FAMILY_SYNONYMS: Dict[str, str] = {
    "random forests": "random forest", "rf": "random forest", "random forest classifier": "random forest",
    "random forest regressor": "random forest",
    "logreg": "logistic regression", "logistic": "logistic regression",
    "ols": "linear regression", "least squares": "linear regression",
    "ridge regression": "ridge", "l2": "ridge", "lasso regression": "lasso", "l1": "lasso",
    "elasticnet": "elastic net", "k nearest neighbors": "knn", "k-nearest neighbors": "knn",
    "nearest neighbors": "knn", "nearest neighbours": "knn", "k-nn": "knn",
    "decision trees": "decision tree", "tree": "decision tree", "cart": "decision tree",
    "support vector machine": "svm", "support vector": "svm", "svc": "svm", "svr": "svm",
    "gaussian naive bayes": "naive bayes", "nb": "naive bayes", "gnb": "naive bayes",
    "gbm": "gradient boosting", "boosting": "gradient boosting", "xgb": "xgboost",
    "neural network": "mlp", "neural net": "mlp", "multilayer perceptron": "mlp",
    "k-means": "kmeans", "k means": "kmeans", "hierarchical": "agglomerative", "ward": "agglomerative",
    "gaussian mixture": "gmm", "gaussian mixture model": "gmm", "mixture model": "gmm",
    "iforest": "isolation forest", "local outlier factor": "lof", "one-class svm": "one class svm",
    "oneclass svm": "one class svm", "ocsvm": "one class svm", "auto-encoder": "autoencoder", "ae": "autoencoder",
    "persistence": "naive", "temporal transformer": "transformer", "time series transformer": "transformer",
    "dummy": "baseline", "majority": "baseline", "mean baseline": "baseline",
}

GENERATIVE_MODELS = {"vae": "vae", "variational autoencoder": "vae", "variational auto-encoder": "vae",
                     "plain autoencoder": "ae", "standard autoencoder": "ae"}

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\w\s\-\.=,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_dataset_lexicon() -> Dict[str, str]:
    lex = dict(DATASET_SYNONYMS)
    for ds_id, spec in DATASETS.items():
        lex[ds_id.replace("_", " ")] = ds_id
        lex[_norm(spec.name)] = ds_id
        lex[_norm(re.sub(r"\(.*?\)", "", spec.name))] = ds_id
        lex[_norm(spec.name.replace("UCI ", "").replace("EPM ", ""))] = ds_id
    return {k: v for k, v in lex.items() if k}


def _build_family_lexicon() -> Dict[str, str]:
    lex = {f: f for f in ALGORITHM_FAMILIES}
    lex.update(FAMILY_SYNONYMS)
    return lex


DATASET_LEXICON = _build_dataset_lexicon()
FAMILY_LEXICON = _build_family_lexicon()
ALGORITHM_ID_LEXICON = {aid.replace("_", " "): aid for aid in ALGORITHMS}
ALGORITHM_ID_LEXICON.update({_norm(a.name): a.id for a in ALGORITHMS.values()})

# words that should never be fuzzy-matched to a dataset/algorithm name
STOPWORDS = set("""a an the and or with on using use for to of in by from my me i want would like please run train
fit do make can you show list all what which is are best model models algorithm algorithms data dataset datasets
predict prediction it this that compare vs versus try then also some any per new set k step steps ahead next
results result cv fold folds seed rows samples sample generate explain help status how why tell about give
regression classification clustering forecasting forecast classify cluster regress anomaly anomalies detection
detect outliers time series benchmark want""".split())


def _find_phrases(text: str, lexicon: Dict[str, str]) -> List[Tuple[int, int, str]]:
    """Longest-first, non-overlapping whole-phrase matches → [(start, end, value)]."""
    hits: List[Tuple[int, int, str]] = []
    taken = [False] * (len(text) + 1)
    for phrase in sorted(lexicon, key=len, reverse=True):
        for m in re.finditer(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", text):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            hits.append((m.start(), m.end(), lexicon[phrase]))
    return sorted(hits)


def _fuzzy(tokens: List[str], lexicon: Dict[str, str], cutoff: float = 0.84) -> Optional[str]:
    hit = _fuzzy_gram(tokens, lexicon, cutoff)
    return hit[0] if hit else None


def _fuzzy_gram(tokens: List[str], lexicon: Dict[str, str], cutoff: float = 0.84) -> Optional[Tuple[str, str]]:
    """Fuzzy match; returns (value, matched text gram)."""
    keys = [k for k in lexicon if len(k) >= 4]
    cands = [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]
    # bigrams first ("brest cancer"), then single tokens
    grams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1) if tokens[i] not in STOPWORDS] + cands
    for g in grams:
        if g in STOPWORDS:
            continue
        m = difflib.get_close_matches(g, keys, n=1, cutoff=cutoff)
        if m:
            return lexicon[m[0]], g
    return None


# -----------------------------------------------------------------------------
# Regex settings
# -----------------------------------------------------------------------------

_INT = r"(\d{1,7})"
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("cv_folds", re.compile(r"(\d{1,2})\s*-?\s*folds?\b|\bcv\s*=?\s*(\d{1,2})\b|cross[\s-]?val\w*\s*(?:with\s*)?(\d{1,2})?|\bcv\b")),
    ("seed", re.compile(r"\bseed\s*(?:=|of|is)?\s*" + _INT)),
    ("horizon", re.compile(r"(\d{1,3})\s*(?:steps?|hours?|points?)?\s*ahead\b|\bhorizon\s*(?:=|of)?\s*(\d{1,3})\b|\bnext\s+(\d{1,3})\s*(?:steps?|hours?|points?)\b")),
    ("window", re.compile(r"\b(?:window|lookback|look-back|context)\s*(?:=|of|size)?\s*(\d{1,4})\b")),
    ("contamination", re.compile(r"\bcontamination\s*(?:=|of)?\s*(0?\.\d+|\d{1,2}\s*%)")),
    ("max_train_rows", re.compile(r"\b(?:max(?:imum)?\s*(?:train(?:ing)?\s*)?rows|limit(?:ed)?\s*(?:to)?|subsample(?:d)?\s*(?:to)?)\s*(?:=|of)?\s*(\d[\d,_]{0,9})(?:\s*rows)?")),
    ("n_clusters", re.compile(r"\b(?:k\s*=\s*(\d{1,2})|(\d{1,2})\s*clusters?\b|n_clusters\s*=\s*(\d{1,2}))")),
    ("epochs", re.compile(r"\b(\d{1,3})\s*epochs?\b|\bepochs?\s*=?\s*(\d{1,3})\b")),
    ("num_samples", re.compile(r"\b(\d{1,5})\s*(?:synthetic\s*)?(?:samples|records|points)\b")),
    ("latent_dim", re.compile(r"\blatent(?:\s*dim(?:ension)?)?\s*(?:=|of)?\s*(\d{1,3})\b|(\d{1,3})\s*-?\s*dim(?:ensional)?\s*latent")),
    ("n_rows", re.compile(r"\b(?:first|top|head)\s*(\d{1,3})\s*rows?\b|\b(\d{1,3})\s*rows\b")),
]
_HYPER = re.compile(r"\b([a-z][a-z_]{1,24})\s*=\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?|true|false|none|[a-z_]{1,16})\b")
_RESERVED_HYPER = {"cv", "seed", "horizon", "window", "contamination", "k", "n_clusters", "epochs", "latent",
                   "rows", "max_train_rows"}


def _strip(text: str, lexicon: Dict[str, str], min_len: int = 1) -> str:
    for phrase in sorted(lexicon, key=len, reverse=True):
        if len(phrase) >= min_len:
            text = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", " ", text)
    return text


def _first_group(m: re.Match) -> Optional[str]:
    return next((g for g in m.groups() if g), None)


def _num(v: str) -> Any:
    v = v.lower()
    if v in ("true", "false"):
        return v == "true"
    if v == "none":
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def extract_slots(message: str) -> ChatSlots:
    """Extract every slot present in ``message`` (no conversation context)."""
    s = ChatSlots()
    text = _norm(message)
    raw = message.lower()

    # --- numeric settings (regex) ------------------------------------------
    for name, pat in _PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        g = _first_group(m)
        if name == "cv_folds":
            s.cv_folds = int(g) if g else 5
        elif name == "contamination":
            g = g.replace(" ", "")
            s.contamination = float(g[:-1]) / 100 if g.endswith("%") else float(g)
        elif g is not None:
            setattr(s, name, int(g.replace(",", "").replace("_", "")))
    if re.search(r"\b(allow[\s-]large|large datasets? ok|include large|big dataset ok)\b", raw):
        s.allow_large = True

    # hyper-parameters "alpha=0.1, max_depth=4"
    for k, v in _HYPER.findall(raw):
        if k not in _RESERVED_HYPER:
            s.hyperparams[k] = _num(v)

    # --- generative model type ---------------------------------------------------
    for phrase, kind in GENERATIVE_MODELS.items():
        if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text):
            s.generative_model = kind
            break

    # --- algorithms: ids/names and families matched together, longest phrase first
    #     ("naive bayes" must win over the forecasting baseline id "naive")
    alg_text = text
    if s.generative_model:  # "vae"/"autoencoder" in a generation request is the generator, not the detector
        alg_text = re.sub(r"\b(variational )?auto-?encoder\b|\bvae\b", " ", alg_text)
    combined = {k: ("id", v) for k, v in ALGORITHM_ID_LEXICON.items()}
    combined.update({k: ("fam", v) for k, v in FAMILY_LEXICON.items()})
    for _, _, (kind, val) in _find_phrases(alg_text, combined):
        bucket = s.algorithm_ids if kind == "id" else s.algorithm_families
        if val not in bucket:
            bucket.append(val)
    rest = _strip(text, ALGORITHM_ID_LEXICON)
    rest = _strip(rest, FAMILY_LEXICON, min_len=2)
    # dataset phrases are removed before fuzzy-matching algorithm names ("random forrest on iris")
    no_ds = _strip(rest, DATASET_LEXICON, min_len=3)
    if not s.algorithm_families and not s.algorithm_ids:
        multi = {k: v for k, v in FAMILY_LEXICON.items() if " " in k}
        hit = _fuzzy_gram(_WORD.findall(no_ds), multi, 0.86)
        if hit:
            s.algorithm_families.append(hit[0])
            rest = re.sub(r"(?<!\w)" + re.escape(hit[1]) + r"(?!\w)", " ", rest)
    if re.search(r"\b(all|every|each)\s+(compatible\s+)?(algorithms?|models?)\b|^all$|\ball of them\b", text):
        s.all_algorithms = True

    # --- task words (after algorithm names are removed: "logistic regression" ≠ regression task)
    task_text = rest if (s.algorithm_families or s.algorithm_ids) else text
    for phrase, task in sorted(TASK_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", task_text):
            s.task = task
            break

    # --- dataset (exact, then fuzzy) ---------------------------------------
    ds_text = _strip(text, {**FAMILY_LEXICON, **ALGORITHM_ID_LEXICON}, min_len=4)
    hits = _find_phrases(ds_text, DATASET_LEXICON)
    if hits:
        s.dataset_id = max(hits, key=lambda h: h[1] - h[0])[2]
    else:
        s.dataset_id = _fuzzy(_WORD.findall(_strip(ds_text, FAMILY_LEXICON, 2)), DATASET_LEXICON)
    return s


def resolve_algorithms(slots: ChatSlots, task: str) -> Tuple[List[str], List[str]]:
    """Turn ids + families into catalogue ids for ``task``. Returns (ids, problems)."""
    ids: List[str] = []
    problems: List[str] = []
    for aid in slots.algorithm_ids:
        a = ALGORITHMS[aid]
        if a.task == task:
            ids.append(aid)
        else:
            alt = suggest_alternative(aid, task)
            problems.append(f"{a.name} is a {a.task.replace('_', ' ')} algorithm, not {task.replace('_', ' ')}"
                            + (f". Did you mean {ALGORITHMS[alt].name}?" if alt else "."))
    for fam in slots.algorithm_families:
        mapping = ALGORITHM_FAMILIES.get(fam, {})
        if task in mapping:
            if mapping[task] not in ids:
                ids.append(mapping[task])
        else:
            avail = ", ".join(t.replace("_", " ") for t in mapping) or "no task"
            alt = next((suggest_alternative(a, task) for a in mapping.values() if suggest_alternative(a, task)), None)
            problems.append(f"{fam.capitalize()} is available for {avail}, not {task.replace('_', ' ')}"
                            + (f". Did you mean {ALGORITHMS[alt].name}?" if alt else "."))
    return ids, problems


def suggest_alternative(algorithm_id: str, task: str) -> Optional[str]:
    """Equivalent algorithm in another task (logistic regression → ridge for regression, …)."""
    for fam, mapping in ALGORITHM_FAMILIES.items():
        if algorithm_id in mapping.values() and task in mapping:
            return mapping[task]
    equivalents = {"logreg": "ols", "logreg_scratch": "linreg_scratch_ne", "gnb": "ridge", "mlp_torch": "rf_reg"}
    if task == "regression" and algorithm_id in equivalents:
        return equivalents[algorithm_id]
    if task == "classification":
        rev = {"ridge": "logreg", "lasso": "logreg", "elasticnet": "logreg", "ols": "logreg"}
        return rev.get(algorithm_id)
    return None


TASK_PRIORITY = ("classification", "regression", "clustering", "forecasting", "anomaly_detection")


def family_default(fam: str, task: Optional[str] = None) -> Optional[str]:
    """Catalogue id for a family: for ``task`` if given, else by TASK_PRIORITY."""
    m = ALGORITHM_FAMILIES.get(fam, {})
    if task:
        return m.get(task)
    return next((m[t] for t in TASK_PRIORITY if t in m), None)


def algorithm_name(aid: str) -> str:
    return ALGORITHMS[aid].name if aid in ALGORITHMS else aid


def task_algorithms(task: str) -> List[str]:
    return [a.id for a in list_algorithms(task)]
