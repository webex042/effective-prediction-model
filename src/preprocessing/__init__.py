"""EPM Preprocessing Module — leakage-free, algorithm-aware preprocessing pipelines."""

from src.preprocessing.pipelines import (
    PreprocessingConfig,
    QuantileClipper,
    build_tabular_preprocessor,
    make_model_pipeline,
    describe_preprocessing,
    TabularSplit,
    deduplicate,
    split_tabular,
    SeriesData,
    prepare_series,
    causal_fill,
    rolling_origin_splits,
    lag_window_features,
)

__all__ = [
    "PreprocessingConfig", "QuantileClipper", "build_tabular_preprocessor", "make_model_pipeline",
    "describe_preprocessing", "TabularSplit", "deduplicate", "split_tabular", "SeriesData",
    "prepare_series", "causal_fill", "rolling_origin_splits", "lag_window_features",
]
