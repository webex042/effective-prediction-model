"""
Intent recognition for the EPM chatbot (no LLM, fully offline).

1. **Ordered regex rules** catch canonical phrasings precisely (first match wins;
   more specific intents are listed first).
2. **Fallback text classifier**: TF-IDF (word + char n-grams) + logistic
   regression trained at start-up on the example phrases below. It is only
   trusted above a confidence threshold.
3. Otherwise ``unknown``. The engine may still infer a task from a bare
   dataset name ("train on iris").
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

INTENTS: Dict[str, str] = {
    # task-oriented (Phase 1)
    "classify": "Run a classification experiment",
    "regress": "Run a regression experiment",
    "cluster": "Run a clustering experiment",
    "benchmark": "Run a full benchmark (all compatible algorithms)",
    "train": "Train on a dataset (task inferred from the dataset)",
    # exploratory
    "list_datasets": "Show available datasets for a task",
    "list_algorithms": "Show available algorithms for a task",
    "describe_dataset": "Show dataset details (source, licence, shape, target)",
    "preview_data": "Show the first rows and column statistics of a dataset",
    "download": "Download and verify a remote dataset",
    "recommend": "Suggest algorithms for a dataset (or datasets for an algorithm)",
    # results
    "show_results": "Display results of the last (or a specific) run",
    "compare": "Compare two or more algorithms",
    # meta
    "help": "Show what the chatbot can do",
    "status": "Show EPM system status",
    "cancel": "Cancel a running job or the pending request",
    "confirm": "Confirm the pending action",
    "deny": "Decline the pending action",
    # Phase 2
    "forecast": "Run a time-series forecasting experiment",
    "detect_anomalies": "Run anomaly detection",
    # Phase 3
    "generate": "Train a VAE/AE and generate synthetic samples",
    "compare_generative": "Compare diffusion (DDPM/DDIM) vs flow matching on held-out data",
    "uncertainty": "Show calibration / predictive uncertainty",
    "compress": "Distil, quantise and prune a model trained on a dataset",
    "explain": "Explain a concept from the EPM docs",
}

TASK_INTENTS = {"classify": "classification", "regress": "regression", "cluster": "clustering",
                "forecast": "forecasting", "detect_anomalies": "anomaly_detection"}

_R = lambda p: re.compile(p, re.I)  # noqa: E731

# (intent, pattern) — evaluated in order.
RULES: List[Tuple[str, re.Pattern]] = [
    ("confirm", _R(r"^\s*(y|yes|yeah|yep|sure|ok|okay|go|go ahead|run it|do it|confirm|proceed|yes please|start)\s*[.!]*\s*$")),
    ("deny", _R(r"^\s*(n|no|nope|nah|never ?mind|don'?t|no thanks)\s*[.!]*\s*$")),
    ("cancel", _R(r"\b(cancel|abort|kill|stop)\b.*\b(job|run|training|it|that|this|benchmark)?\b|^\s*(cancel|stop|abort)\s*$")),
    ("help", _R(r"^\s*(hi|hello|hey|help|\?|commands|menu|start over)\s*[!.?]*\s*$|\bwhat can you do\b|\bhow (do|can) i use (this|you|epm)\b|\bwhat do you do\b")),
    ("status", _R(r"\b(system )?status\b|\bhealth\b|\bwhat'?s running\b|\bactive jobs\b|\bis (it|the job) (done|finished|running)\b")),
    ("compare_generative", _R(r"\b(diffusion|ddpm|ddim|flow[\s-]?matching|cfm|generative models?)\b.*\b(vs\.?|versus|against|compare|comparison)\b"
                              r"|\b(compare|comparison|benchmark)\b.*\b(diffusion|ddpm|ddim|flow[\s-]?matching|generative models)\b")),
    ("generate", _R(r"\b(generate|synthesi[sz]e|synthetic (samples|data|rows|records)|sample from|samples from|new samples)\b")),
    ("compress", _R(r"\b(compress\w*|quantis\w*|quantiz\w*|prun\w*|distil\w*|int8|shrink (the )?model|smaller model)\b")),
    ("uncertainty", _R(r"\b(uncertaint\w*|calibrat\w*|confiden\w*|ece|brier|how sure|reliability diagram)\b")),
    ("explain", _R(r"^\s*(explain|define|what (is|are)|what'?s (a|an|the)? ?(meaning|idea)|tell me about|how does|how do|why does|teach me)\b(?!.*\b(best|recommend|should i)\b)")),
    ("recommend", _R(r"\b(recommend\w*|suggest\w*|best (\w+ )?(model|algorithm|approach|method|classifier|regressor)s?|which (model|algorithm)s? (should|would|to|is best|are best|works? best|do you recommend)|what (model|algorithm)s? should|what should i use)\b"
                     r"|\bwhat'?s the best\b|\bwhich datasets? (can|should|for|work)\b")),
    ("download", _R(r"\b(download|fetch)\b")),
    ("preview_data", _R(r"\b(preview|head of|first \d* ?rows|show (me )?(the )?(rows|first rows|raw data|data of|sample of)|peek|look at the data)\b")),
    ("list_algorithms", _R(r"\b(list|show|available|supported|which|what)\b[^?.]*\b(algorithms|models|methods)\b(?![^?.]*\bbest\b)|^\s*(algorithms|models)\s*\??$")),
    ("list_datasets", _R(r"\b(list|show|available|supported|which|what|all)\b[^?.]*\bdatasets\b|^\s*datasets\s*\??$"
                         r"|\b(another|other|different) dataset\b")),
    ("describe_dataset", _R(r"\b(describe|details? (of|about|on)|info(rmation)? (about|on)|schema|columns|features of|statistics|stats|tell me about|licen[cs]e|source of)\b")),
    ("show_results", _R(r"\b(show|display|see|view|print|open)\b[^.]*\b(results?|leaderboard|last run|outcome|scores?)\b|^\s*(results?|leaderboard)\s*$")),
    ("compare", _R(r"\bcompare\b|\bvs\.?\b|\bversus\b|\bhead[\s-]to[\s-]head\b")),
    ("benchmark", _R(r"\b(benchmark\w*|all (compatible )?(algorithms|models)|every (algorithm|model)|full comparison|leaderboard for)\b")),
    ("classify", _R(r"\b(classif\w*|categori[sz]\w*|predict (the )?(class|label|species|diagnosis|category|cultivar|digit|income|subscription))\b")),
    ("regress", _R(r"\b(regress\w*|predict (the )?(price|prices|value|amount|house|progression|load)|estimate (the )?\w+)\b")),
    ("detect_anomalies", _R(r"\b(anomal\w*|outliers?|abnormal|unusual (points|readings|values|rows)?|novelty detection)\b")),
    ("forecast", _R(r"\b(forecast\w*|predict (the )?(next|future)|time[\s-]?series|(steps?|hours?) ahead|horizon)\b")),
    ("cluster", _R(r"\b(cluster\w*|segment\w*|group (the|my|these|this)|grouping|unsupervised)\b")),
    ("detect_anomalies", _R(r"\bfraud\b")),
    ("train", _R(r"\b(train|run|fit|model|experiment)\b")),
]

# Example phrases for the fallback classifier.
TRAINING_PHRASES: Dict[str, List[str]] = {
    "classify": ["classify iris", "classification on wine", "predict the species", "which class is it",
                 "build a classifier for breast cancer", "label prediction on digits", "categorise the samples",
                 "detect malignant tumours", "is the client going to subscribe"],
    "regress": ["predict house prices", "regression on diabetes", "estimate the heating load",
                "predict a continuous value", "fit a regressor", "how much will it cost", "predict disease progression"],
    "cluster": ["group the wine data", "cluster the customers", "find segments", "unsupervised grouping",
                "kmeans on iris", "segment the clients", "find natural groups"],
    "forecast": ["forecast the next 24 hours", "predict future values", "time series prediction",
                 "forecast air quality", "what will the power consumption be tomorrow", "predict next steps"],
    "detect_anomalies": ["find outliers", "detect anomalies", "spot unusual readings", "fraud detection",
                         "which points are abnormal", "find the weird ones"],
    "benchmark": ["benchmark everything", "run all models", "full leaderboard", "try every algorithm",
                  "compare all compatible algorithms"],
    "list_datasets": ["show me all datasets", "what data do you have", "list datasets", "which datasets are there",
                      "available data"],
    "list_algorithms": ["what algorithms are available", "list models", "which models do you support",
                        "show the algorithms"],
    "describe_dataset": ["tell me about the adult dataset", "describe iris", "what's in the wine data",
                         "dataset details", "where does the data come from"],
    "preview_data": ["show me the first rows", "preview the data", "let me see the data", "head of the dataset"],
    "recommend": ["what is the best model for diabetes", "which algorithm should I use", "recommend a model",
                  "suggest something for iris", "what works best here"],
    "show_results": ["show results", "what were the results", "show the leaderboard", "how did it go"],
    "compare": ["compare random forest and logistic regression", "rf vs svm", "which is better, knn or trees"],
    "help": ["help", "what can you do", "how does this work", "hello", "hi there"],
    "status": ["status", "is the job done", "system health", "what is running"],
    "cancel": ["cancel", "stop the training", "abort the job", "never mind stop it"],
    "generate": ["generate synthetic data", "make fake samples", "create new rows like these",
                 "sample from a vae", "synthesise records"],
    "compare_generative": ["diffusion vs flow matching", "compare generative models", "ddpm versus cfm"],
    "uncertainty": ["how confident is the model", "calibration of the classifier", "show uncertainty"],
    "compress": ["make the model smaller", "quantize the network", "prune and distill", "compress it"],
    "explain": ["explain flow matching", "what is diffusion", "how does attention work", "teach me backprop",
                "what is a vae"],
}

FALLBACK_THRESHOLD = 0.35


class _Fallback:
    """Tiny TF-IDF + logistic-regression intent model (trained once, deterministic)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, make_pipeline

        X, y = [], []
        for intent, phrases in TRAINING_PHRASES.items():
            X.extend(phrases)
            y.extend([intent] * len(phrases))
        feats = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)),
        ])
        self.model = make_pipeline(feats, LogisticRegression(C=8.0, max_iter=2000, random_state=0))
        self.model.fit(X, y)

    def predict(self, text: str) -> Tuple[str, float]:
        proba = self.model.predict_proba([text])[0]
        i = int(proba.argmax())
        return str(self.model.classes_[i]), float(proba[i])


_FALLBACK: Optional[_Fallback] = None


def _fallback() -> _Fallback:
    global _FALLBACK
    if _FALLBACK is None:
        _FALLBACK = _Fallback()
    return _FALLBACK


def parse_intent(text: str) -> str:
    """Convenience wrapper returning only the intent name."""
    return classify_intent(text)[0]


def classify_intent(text: str) -> Tuple[str, float, str]:
    """Return (intent, confidence, source) where source is 'rule', 'model' or 'none'."""
    t = (text or "").strip()
    if not t:
        return "help", 1.0, "rule"
    for intent, pat in RULES:
        if pat.search(t):
            return intent, 1.0, "rule"
    intent, p = _fallback().predict(t.lower())
    if p >= FALLBACK_THRESHOLD:
        return intent, p, "model"
    return "unknown", p, "none"
