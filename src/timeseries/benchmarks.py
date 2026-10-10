"""
EPM Time-Series Forecasting Benchmark Suite
===========================================

Comprehensive benchmarking suite comparing 6 model paradigms:
  1. Linear Regression (Baseline)
  2. Random Forest (Ensemble Trees)
  3. XGBoost (Gradient Boosting)
  4. LSTM (Recurrent Neural Network with Gated Memory Highway)
  5. GRU (Streamlined Gated Recurrent Unit)
  6. Time-Series Transformer (Multi-Head Self-Attention)
"""

from __future__ import annotations

import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.timeseries.models import (
    LSTMForecaster,
    GRUForecaster,
    TimeSeriesTransformer,
    train_forecasting_model,
)
from src.evaluation.metrics import rmse, mae, r2_score


def run_timeseries_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 35,
    batch_size: int = 32,
    hidden_dim: int = 32,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Train and evaluate all 6 time-series models on identical temporal test data.

    Parameters
    ----------
    X_train, X_test : np.ndarray of shape (N, W, D)
    y_train, y_test : np.ndarray of shape (N, H)

    Returns
    -------
    results_df : pd.DataFrame
        Leaderboard sorted by lowest RMSE.
    predictions : dict
        Predicted trajectory arrays for each model.
    """
    n_train, window_size, n_features = X_train.shape
    horizon = y_train.shape[1]
    n_test = len(X_test)

    # Flattened representations for tabular models (Linear, RF, XGBoost)
    X_train_flat = X_train.reshape(n_train, window_size * n_features)
    X_test_flat = X_test.reshape(n_test, window_size * n_features)

    # Flatten 1D target if horizon == 1
    y_tr = y_train if horizon > 1 else y_train.ravel()
    y_te = y_test if horizon > 1 else y_test.ravel()

    results = []
    predictions: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------
    # 1. Linear Regression Baseline
    # -------------------------------------------------------------
    lr_model = LinearRegression()
    start_t = time.perf_counter()
    lr_model.fit(X_train_flat, y_tr)
    t_train_lr = time.perf_counter() - start_t

    start_inf = time.perf_counter()
    pred_lr = lr_model.predict(X_test_flat)
    t_inf_lr = (time.perf_counter() - start_inf) / n_test * 1e6
    pred_lr_arr = pred_lr.reshape(-1, horizon) if horizon > 1 else pred_lr.reshape(-1, 1)
    predictions["Linear Regression"] = pred_lr_arr

    results.append({
        "model": "Linear Regression",
        "family": "Classical Baseline",
        "rmse": round(rmse(y_test, pred_lr_arr), 4),
        "mae": round(mae(y_test, pred_lr_arr), 4),
        "r2": round(r2_score(y_test, pred_lr_arr), 4),
        "latency_us_per_sample": round(t_inf_lr, 2),
        "training_time_s": round(t_train_lr, 4),
        "parameters": window_size * n_features * horizon + horizon,
    })

    # -------------------------------------------------------------
    # 2. Random Forest Regressor
    # -------------------------------------------------------------
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    start_t = time.perf_counter()
    rf_model.fit(X_train_flat, y_tr)
    t_train_rf = time.perf_counter() - start_t

    start_inf = time.perf_counter()
    pred_rf = rf_model.predict(X_test_flat)
    t_inf_rf = (time.perf_counter() - start_inf) / n_test * 1e6
    pred_rf_arr = pred_rf.reshape(-1, horizon) if horizon > 1 else pred_rf.reshape(-1, 1)
    predictions["Random Forest"] = pred_rf_arr

    results.append({
        "model": "Random Forest",
        "family": "Ensemble Trees",
        "rmse": round(rmse(y_test, pred_rf_arr), 4),
        "mae": round(mae(y_test, pred_rf_arr), 4),
        "r2": round(r2_score(y_test, pred_rf_arr), 4),
        "latency_us_per_sample": round(t_inf_rf, 2),
        "training_time_s": round(t_train_rf, 4),
        "parameters": 50 * 500,  # approximate tree nodes
    })

    # -------------------------------------------------------------
    # 3. XGBoost Regressor
    # -------------------------------------------------------------
    xgb_model = xgb.XGBRegressor(n_estimators=60, max_depth=5, learning_rate=0.08, random_state=42, verbosity=0)
    start_t = time.perf_counter()
    xgb_model.fit(X_train_flat, y_tr)
    t_train_xgb = time.perf_counter() - start_t

    start_inf = time.perf_counter()
    pred_xgb = xgb_model.predict(X_test_flat)
    t_inf_xgb = (time.perf_counter() - start_inf) / n_test * 1e6
    pred_xgb_arr = pred_xgb.reshape(-1, horizon) if horizon > 1 else pred_xgb.reshape(-1, 1)
    predictions["XGBoost"] = pred_xgb_arr

    results.append({
        "model": "XGBoost",
        "family": "Gradient Boosting",
        "rmse": round(rmse(y_test, pred_xgb_arr), 4),
        "mae": round(mae(y_test, pred_xgb_arr), 4),
        "r2": round(r2_score(y_test, pred_xgb_arr), 4),
        "latency_us_per_sample": round(t_inf_xgb, 2),
        "training_time_s": round(t_train_xgb, 4),
        "parameters": 60 * 300,
    })

    # -------------------------------------------------------------
    # 4. LSTM Forecaster
    # -------------------------------------------------------------
    lstm_model = LSTMForecaster(input_dim=n_features, hidden_dim=hidden_dim, horizon=horizon, num_layers=2)
    start_t = time.perf_counter()
    train_forecasting_model(lstm_model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    t_train_lstm = time.perf_counter() - start_t

    lstm_model.eval()
    start_inf = time.perf_counter()
    with torch.no_grad():
        pred_lstm = lstm_model(torch.from_numpy(X_test).float()).numpy()
    t_inf_lstm = (time.perf_counter() - start_inf) / n_test * 1e6
    predictions["LSTM"] = pred_lstm

    results.append({
        "model": "LSTM",
        "family": "Recurrent (Gated Highway)",
        "rmse": round(rmse(y_test, pred_lstm), 4),
        "mae": round(mae(y_test, pred_lstm), 4),
        "r2": round(r2_score(y_test, pred_lstm), 4),
        "latency_us_per_sample": round(t_inf_lstm, 2),
        "training_time_s": round(t_train_lstm, 4),
        "parameters": sum(p.numel() for p in lstm_model.parameters()),
    })

    # -------------------------------------------------------------
    # 5. GRU Forecaster
    # -------------------------------------------------------------
    gru_model = GRUForecaster(input_dim=n_features, hidden_dim=hidden_dim, horizon=horizon, num_layers=2)
    start_t = time.perf_counter()
    train_forecasting_model(gru_model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    t_train_gru = time.perf_counter() - start_t

    gru_model.eval()
    start_inf = time.perf_counter()
    with torch.no_grad():
        pred_gru = gru_model(torch.from_numpy(X_test).float()).numpy()
    t_inf_gru = (time.perf_counter() - start_inf) / n_test * 1e6
    predictions["GRU"] = pred_gru

    results.append({
        "model": "GRU",
        "family": "Recurrent (Streamlined Gate)",
        "rmse": round(rmse(y_test, pred_gru), 4),
        "mae": round(mae(y_test, pred_gru), 4),
        "r2": round(r2_score(y_test, pred_gru), 4),
        "latency_us_per_sample": round(t_inf_gru, 2),
        "training_time_s": round(t_train_gru, 4),
        "parameters": sum(p.numel() for p in gru_model.parameters()),
    })

    # -------------------------------------------------------------
    # 6. Time-Series Transformer
    # -------------------------------------------------------------
    transformer_model = TimeSeriesTransformer(
        input_dim=n_features, d_model=hidden_dim, num_heads=4, horizon=horizon, num_layers=2
    )
    start_t = time.perf_counter()
    train_forecasting_model(transformer_model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    t_train_trans = time.perf_counter() - start_t

    transformer_model.eval()
    start_inf = time.perf_counter()
    with torch.no_grad():
        pred_trans = transformer_model(torch.from_numpy(X_test).float()).numpy()
    t_inf_trans = (time.perf_counter() - start_inf) / n_test * 1e6
    predictions["Transformer"] = pred_trans

    results.append({
        "model": "Transformer",
        "family": "Self-Attention",
        "rmse": round(rmse(y_test, pred_trans), 4),
        "mae": round(mae(y_test, pred_trans), 4),
        "r2": round(r2_score(y_test, pred_trans), 4),
        "latency_us_per_sample": round(t_inf_trans, 2),
        "training_time_s": round(t_train_trans, 4),
        "parameters": sum(p.numel() for p in transformer_model.parameters()),
    })

    df = pd.DataFrame(results)
    df = df.sort_values(by="rmse", ascending=True).reset_index(drop=True)
    return df, predictions
