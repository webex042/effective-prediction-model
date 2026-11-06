"""
Static, number-free explanations used by the controller (algorithm families,
what EPM is, how a prediction is produced). Measured numbers always come from
model cards, never from here.
"""

from __future__ import annotations

ABOUT_EPM = (
    "**EPM (Effective Prediction Model)** is a machine-learning experimentation platform. It keeps a registry of "
    "documented datasets and a catalogue of classical, gradient-boosted and neural algorithms, and evaluates them "
    "with leakage-free pipelines: preprocessing is fitted on training rows only, every model gets the same split and "
    "seed, and results are reported on held-out data.\n"
    "In this chat you can:\n"
    "- **predict** with a trained model by typing feature values or uploading a CSV\n"
    "- **train** a model on a registry dataset or on your own CSV (only when you ask)\n"
    "- **compare** models on the same split\n"
    "- see **important features** and an explanation of an individual prediction\n"
    "- ask what an algorithm or ML concept means"
)

HOW_PREDICTIONS_WORK = (
    "Every prediction here goes through the same steps:\n"
    "- **Training (only when you ask):** the dataset is split into train / validation / test with a fixed seed "
    "(chronologically for time series). Imputation, scaling and encoding are *fitted on the training rows only* and "
    "saved together with the model.\n"
    "- **Evaluation:** the model is scored on the held-out test rows it never saw. Those are the metrics shown in "
    "each model card.\n"
    "- **Prediction:** your input is checked against the training schema (names, types, ranges), passed through the "
    "*same* fitted preprocessing, and then through the model.\n"
    "- **Uncertainty:** classifiers report their class probabilities (with the measured calibration error); "
    "regressors and forecasters report a 90% interval from validation residuals (split conformal).\n"
    "- **Explanations:** permutation importance on held-out data, plus a one-feature-at-a-time sensitivity check "
    "for a single prediction."
)

FAMILY_TEXT = {
    "linear": "fits a weighted sum of the (scaled) features. Easy to interpret: each coefficient says how much the "
              "prediction moves per unit of a feature. Ridge and Lasso add a penalty that keeps weights small.",
    "logistic": "computes a weighted sum of the features and passes it through a sigmoid/softmax to get class "
                "probabilities. Fast and interpretable; works best when classes are roughly linearly separable.",
    "knn": "predicts from the k most similar training rows (nearest neighbours in scaled feature space). No real "
           "training step; prediction cost grows with the training set.",
    "tree": "learns a sequence of if/else splits on single features. Very interpretable, but a single deep tree "
            "tends to overfit.",
    "forest": "averages many decision trees, each trained on a bootstrap sample with random feature subsets. Robust, "
              "needs no scaling and handles non-linear interactions well.",
    "svm": "finds the boundary with the largest margin between classes (or a tube around a regression line), using "
           "an RBF kernel to model non-linear shapes. Needs scaled features.",
    "naive_bayes": "assumes features are independent given the class and models each with a Gaussian. Very fast; "
                   "probabilities are often over-confident.",
    "boosting": "builds trees one after another, each correcting the errors of the previous ones. Usually among the "
                "most accurate methods on tabular data.",
    "mlp": "a small neural network (multi-layer perceptron) trained with gradient descent in PyTorch.",
    "baseline": "a deliberately simple reference model. A useful model must beat it.",
    "autoregressive": "predicts future values from a window of past values with a linear (ridge) model.",
    "lag_trees": "uses a window of past values as tabular features for a tree ensemble.",
    "recurrent": "a recurrent neural network (LSTM/GRU) that reads the past window step by step and outputs the "
                 "next values.",
    "transformer": "a small attention-based network that relates every past step in the window to every other step.",
}

ALGO_FAMILY = {
    "ols": "linear", "ridge": "linear", "lasso": "linear", "elasticnet": "linear", "linreg_scratch_ne": "linear",
    "linreg_scratch_gd": "linear", "logreg": "logistic", "logreg_scratch": "logistic", "knn_reg": "knn",
    "knn_clf": "knn", "knn_reg_scratch": "knn", "knn_clf_scratch": "knn", "dt_reg": "tree", "dt_clf": "tree",
    "dt_reg_scratch": "tree", "dt_clf_scratch": "tree", "rf_reg": "forest", "rf_clf": "forest", "svr": "svm",
    "svc": "svm", "gnb": "naive_bayes", "gbr": "boosting", "gb_clf": "boosting", "xgb_reg": "boosting",
    "xgb_clf": "boosting", "mlp_torch": "mlp", "dummy_reg": "baseline", "dummy_clf": "baseline",
    "naive": "baseline", "seasonal_naive": "baseline", "linear_ar": "autoregressive", "rf_forecast": "lag_trees",
    "xgb_forecast": "lag_trees", "lstm": "recurrent", "gru": "recurrent", "ts_transformer": "transformer",
}

# Default model per task when the user does not name one, with the reason we say out loud.
DEFAULT_ALGO = {
    "classification": ("rf_clf", "Random Forest is a robust default for tabular data: no scaling needed, it captures "
                                 "non-linear effects and it returns class probabilities"),
    "regression": ("rf_reg", "Random Forest is a robust default for tabular data: no scaling needed and it captures "
                             "non-linear effects"),
    "forecasting": ("linear_ar", "Linear AR is fast, stable and a strong first forecaster; neural models (LSTM/GRU) "
                                 "take longer to train"),
}


def algorithm_text(algo_id: str, name: str) -> str:
    fam = ALGO_FAMILY.get(algo_id)
    return f"**{name}** {FAMILY_TEXT[fam]}" if fam else f"**{name}** is one of EPM's catalogue algorithms."
