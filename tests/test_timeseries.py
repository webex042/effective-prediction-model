"""
Unit Tests for Stage 5: Time-Series Prediction & Forecasting
============================================================

Verifies:
  1. Multivariate telemetry simulation and physical channel ranges
  2. Sliding window extraction with strict chronological alignment
  3. Temporal train/val/test splitting without lookahead leakage
  4. Vanilla RNN from scratch forward pass
  5. LSTM, GRU, and Time-Series Transformer forward passes and multi-step outputs
  6. Multi-model forecasting benchmark suite
"""

import numpy as np
import pytest
import torch

from src.timeseries.dataset import (
    generate_multivariate_sensor_telemetry,
    create_sliding_windows,
    temporal_train_test_split,
)
from src.timeseries.rnn_scratch import VanillaRNNScratch
from src.timeseries.models import (
    LSTMForecaster,
    GRUForecaster,
    TimeSeriesTransformer,
)
from src.timeseries.benchmarks import run_timeseries_benchmark


# =============================================================================
# 1. Dataset & Windowing Tests
# =============================================================================

class TestTimeSeriesData:
    def test_telemetry_generation(self):
        df, names = generate_multivariate_sensor_telemetry(n_timesteps=300, random_seed=42)
        assert len(df) == 300
        assert names == ["temperature", "vibration", "pressure", "rpm", "load"]
        for col in names:
            assert col in df.columns
            # Non-trivial standard deviation
            assert df[col].std() > 0.1

    def test_sliding_windows_no_leakage(self):
        # Controlled 1D series: 0, 1, 2, ..., 9
        data = np.arange(10).reshape(-1, 1).astype(float)
        W, H = 3, 2
        X, y = create_sliding_windows(data, window_size=W, horizon=H, stride=1, target_channel=0)

        # Total windows = 10 - 3 - 2 + 1 = 6
        assert X.shape == (6, 3, 1)
        assert y.shape == (6, 2)

        # Window 0: input [0, 1, 2], target [3, 4]
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
        np.testing.assert_array_equal(y[0], [3, 4])

        # Window 5: input [5, 6, 7], target [8, 9]
        np.testing.assert_array_equal(X[-1, :, 0], [5, 6, 7])
        np.testing.assert_array_equal(y[-1], [8, 9])

    def test_temporal_split_order(self):
        X = np.arange(100).reshape(100, 1, 1)
        y = np.arange(100).reshape(100, 1)

        X_tr, y_tr, X_val, y_val, X_te, y_te = temporal_train_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15
        )

        assert len(X_tr) == 70
        assert len(X_val) == 15
        assert len(X_te) == 15

        # Strict chronological ordering: max(train) < min(val) < min(test)
        assert np.max(X_tr) < np.min(X_val)
        assert np.max(X_val) < np.min(X_te)


# =============================================================================
# 2. Model Forward Pass Tests
# =============================================================================

class TestTimeSeriesModels:
    def test_vanilla_rnn_scratch(self):
        B, L, D_in, D_h, D_out = 4, 12, 5, 8, 2
        X = np.random.randn(B, L, D_in)

        rnn = VanillaRNNScratch(input_dim=D_in, hidden_dim=D_h, output_dim=D_out)
        out, h_states = rnn.forward(X)

        assert out.shape == (B, D_out)
        assert len(h_states) == L + 1
        assert h_states[-1].shape == (B, D_h)

    def test_lstm_forecaster_multi_step(self):
        B, L, D_in, H = 4, 16, 5, 4
        model = LSTMForecaster(input_dim=D_in, hidden_dim=16, horizon=H, num_layers=2)
        x = torch.randn(B, L, D_in)

        out = model(x)
        assert out.shape == (B, H)

    def test_gru_forecaster_multi_step(self):
        B, L, D_in, H = 4, 16, 5, 3
        model = GRUForecaster(input_dim=D_in, hidden_dim=16, horizon=H, num_layers=1)
        x = torch.randn(B, L, D_in)

        out = model(x)
        assert out.shape == (B, H)

    def test_timeseries_transformer_multi_step(self):
        B, L, D_in, H = 4, 16, 5, 3
        model = TimeSeriesTransformer(input_dim=D_in, d_model=16, num_heads=2, horizon=H, num_layers=2)
        x = torch.randn(B, L, D_in)

        out = model(x)
        assert out.shape == (B, H)


# =============================================================================
# 3. Benchmark Runner Test
# =============================================================================

class TestBenchmark:
    def test_run_timeseries_benchmark(self):
        # Quick synthetic data for fast test
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, size=(40, 10, 3)).astype(np.float32)
        y_train = rng.normal(0, 1, size=(40, 2)).astype(np.float32)
        X_test = rng.normal(0, 1, size=(15, 10, 3)).astype(np.float32)
        y_test = rng.normal(0, 1, size=(15, 2)).astype(np.float32)

        df, preds = run_timeseries_benchmark(
            X_train, y_train, X_test, y_test, epochs=2, batch_size=16, hidden_dim=8
        )

        assert len(df) == 6
        expected_models = {"Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU", "Transformer"}
        assert set(df["model"].values) == expected_models
        assert "rmse" in df.columns
        assert "latency_us_per_sample" in df.columns
