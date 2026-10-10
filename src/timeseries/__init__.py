"""
EPM Time Series Module
======================

Telemetry generation, sliding window extraction, and deep forecasting models:
  - generate_multivariate_sensor_telemetry, create_sliding_windows, temporal_train_test_split
  - VanillaRNNScratch (NumPy with BPTT)
  - LSTMForecaster, GRUForecaster, TimeSeriesTransformer, train_forecasting_model
  - run_timeseries_benchmark
"""

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
    train_forecasting_model,
)
from src.timeseries.benchmarks import run_timeseries_benchmark

__all__ = [
    "generate_multivariate_sensor_telemetry",
    "create_sliding_windows",
    "temporal_train_test_split",
    "VanillaRNNScratch",
    "LSTMForecaster",
    "GRUForecaster",
    "TimeSeriesTransformer",
    "train_forecasting_model",
    "run_timeseries_benchmark",
]
