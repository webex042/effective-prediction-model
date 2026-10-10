"""
EPM Classical ML Module
=======================

Contains from-scratch NumPy implementations of core algorithms:
  - Linear Regression (Normal Equation & Gradient Descent)
  - Logistic Regression (Binary & Multiclass OvR)
  - K-Nearest Neighbors (Classifier & Regressor)
  - Decision Trees (Classifier & Regressor)

And benchmarking suite with scikit-learn and XGBoost:
  - baselines & leaderboard generator
"""

from src.classical_ml.linear_regression import LinearRegressionScratch
from src.classical_ml.logistic_regression import LogisticRegressionScratch, sigmoid
from src.classical_ml.knn import KNNClassifierScratch, KNNRegressorScratch
from src.classical_ml.decision_tree import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
    Node,
)
from src.classical_ml.baselines import (
    evaluate_model,
    run_classical_ml_benchmarks,
    generate_sensor_regression_dataset,
    generate_sensor_failure_dataset,
    get_classical_regression_models,
    get_classical_classification_models,
)

__all__ = [
    "LinearRegressionScratch",
    "LogisticRegressionScratch",
    "sigmoid",
    "KNNClassifierScratch",
    "KNNRegressorScratch",
    "DecisionTreeClassifierScratch",
    "DecisionTreeRegressorScratch",
    "Node",
    "evaluate_model",
    "run_classical_ml_benchmarks",
    "generate_sensor_regression_dataset",
    "generate_sensor_failure_dataset",
    "get_classical_regression_models",
    "get_classical_classification_models",
]
