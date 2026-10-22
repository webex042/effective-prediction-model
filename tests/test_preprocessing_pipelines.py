"""
Leakage-prevention and preprocessing tests (src/preprocessing/pipelines.py).

Each test pins down one guarantee: transforms are fitted on training rows only,
splits are disjoint/reproducible, time order is preserved, and the same fitted
pipeline is used at inference.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.preprocessing.pipelines import (
    PreprocessingConfig, QuantileClipper, build_tabular_preprocessor, causal_fill, deduplicate,
    lag_window_features, make_model_pipeline, prepare_series, rolling_origin_splits, split_tabular,
)


def _blocks(pre):
    return {name: trans for name, trans, _cols in pre.transformers}


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "num1": rng.normal(10, 2, n),
        "num2": rng.normal(-5, 7, n),
        "cat": rng.choice(["a", "b", "c"], n),
    })
    df.loc[::17, "num1"] = np.nan
    df.loc[::23, "cat"] = np.nan
    y = pd.Series(2 * df["num2"].fillna(0) + rng.normal(0, 0.1, n))
    return df, y


class TestTabularPreprocessor:
    def test_scaler_statistics_come_from_training_rows_only(self, mixed_df):
        df, _ = mixed_df
        train, test = df.iloc[:200], df.iloc[200:].copy()
        test["num2"] += 1000  # wildly shifted test data must not affect the fitted scaler
        pre = build_tabular_preprocessor(["num1", "num2"], ["cat"], needs_scaling=True)
        pre.fit(train)
        scaler = pre.named_transformers_["num"].named_steps["scale"]
        assert scaler.mean_[1] == pytest.approx(train["num2"].mean())
        Z = pre.transform(test)
        assert Z[:, 1].mean() > 100  # test shift is visible, not normalised away

    def test_imputer_uses_training_median(self, mixed_df):
        df, _ = mixed_df
        train = df.iloc[:200]
        pre = build_tabular_preprocessor(["num1", "num2"], [], needs_scaling=False)
        pre.fit(train)
        imp = pre.named_transformers_["num"].named_steps["impute"]
        assert imp.statistics_[0] == pytest.approx(train["num1"].median())

    def test_unseen_category_does_not_crash(self, mixed_df):
        df, _ = mixed_df
        pre = build_tabular_preprocessor(["num1"], ["cat"], needs_scaling=True).fit(df)
        new = pd.DataFrame({"num1": [1.0], "cat": ["never_seen"]})
        out = pre.transform(new)
        assert out.shape[0] == 1 and np.isfinite(out).all()

    def test_tree_models_are_not_scaled(self):
        pre = build_tabular_preprocessor(["a"], [], needs_scaling=False)
        assert "scale" not in _blocks(pre)["num"].named_steps

    def test_scale_always_and_never_override(self):
        always = build_tabular_preprocessor(["a"], [], False, PreprocessingConfig(scale="always"))
        never = build_tabular_preprocessor(["a"], [], True, PreprocessingConfig(scale="never"))
        assert "scale" in _blocks(always)["num"].named_steps
        assert "scale" not in _blocks(never)["num"].named_steps

    def test_quantile_clipper_learns_bounds_on_train(self):
        X = np.arange(100, dtype=float).reshape(-1, 1)
        c = QuantileClipper(0.1, 0.9).fit(X)
        assert c.transform([[1e6]])[0, 0] == pytest.approx(np.quantile(X, 0.9))

    def test_pipeline_inference_is_identical_after_serialisation(self, mixed_df):
        df, y = mixed_df
        pipe = make_model_pipeline(LinearRegression(), ["num1", "num2"], ["cat"], True).fit(df.iloc[:200], y[:200])
        clone = pickle.loads(pickle.dumps(pipe))
        np.testing.assert_allclose(pipe.predict(df.iloc[200:]), clone.predict(df.iloc[200:]))

    def test_pipeline_refit_does_not_see_test_rows(self, mixed_df):
        df, y = mixed_df
        a = make_model_pipeline(DecisionTreeRegressor(random_state=0), ["num1", "num2"], [], False).fit(df.iloc[:200], y[:200])
        df2 = df.copy()
        df2.iloc[200:, 0] = 1e9  # corrupt only test rows
        b = make_model_pipeline(DecisionTreeRegressor(random_state=0), ["num1", "num2"], [], False).fit(df2.iloc[:200], y[:200])
        np.testing.assert_allclose(a.predict(df.iloc[200:210]), b.predict(df.iloc[200:210]))


class TestSplits:
    def test_disjoint_complete_and_reproducible(self):
        y = pd.Series(np.r_[np.zeros(80), np.ones(20)].astype(int))
        s1 = split_tabular(100, y, "stratified", 0.2, 0.1, seed=7)
        s2 = split_tabular(100, y, "stratified", 0.2, 0.1, seed=7)
        assert s1.fingerprint() == s2.fingerprint()
        all_idx = np.concatenate([s1.train_idx, s1.val_idx, s1.test_idx])
        assert len(set(all_idx)) == 100 == len(all_idx)
        assert y[s1.test_idx].mean() == pytest.approx(0.2)  # stratified

    def test_different_seed_changes_split(self):
        a = split_tabular(200, None, "random", seed=1)
        b = split_tabular(200, None, "random", seed=2)
        assert a.fingerprint() != b.fingerprint()

    def test_chronological_split_keeps_time_order(self):
        s = split_tabular(100, None, "chronological", 0.2, 0.1)
        assert s.train_idx.max() < s.val_idx.min() < s.test_idx.min()

    def test_max_train_rows_subsamples_train_only(self):
        s = split_tabular(1000, None, "random", 0.2, 0.1, seed=0, max_train_rows=100)
        assert len(s.train_idx) == 100 and s.n_train_subsampled_from == 700 and len(s.test_idx) == 200

    def test_invalid_fractions(self):
        with pytest.raises(ValueError):
            split_tabular(100, None, "random", 0.6, 0.4)

    def test_deduplicate_drops_exact_duplicates(self):
        X = pd.DataFrame({"a": [1, 1, 2, 2], "b": [0, 0, 1, 1]})
        y = pd.Series([0, 0, 1, 0])
        X2, y2, n = deduplicate(X, y)
        assert n == 1 and len(X2) == 3  # rows with differing targets are kept


class TestTimeSeries:
    def _frame(self, n=400):
        t = np.arange(n, dtype=float)
        return pd.DataFrame({"y": np.sin(t / 10) + t / 100, "x": np.cos(t / 7)})

    def test_scaling_uses_training_segment_only(self):
        df = self._frame()
        sd = prepare_series(df, "y", window=12, horizon=1, seasonal_period=24)
        tr_end = sd.boundaries["train_end"]
        assert sd.channel_mean[sd.target_channel] == pytest.approx(df["y"].iloc[:tr_end].mean())
        df2 = df.copy()
        df2.iloc[tr_end:, 0] += 500  # future values change; train statistics must not
        sd2 = prepare_series(df2, "y", window=12, horizon=1, seasonal_period=24)
        np.testing.assert_allclose(sd.channel_mean, sd2.channel_mean)
        np.testing.assert_allclose(sd.X_train, sd2.X_train)

    def test_training_targets_never_cross_into_validation(self):
        n = 300
        df = pd.DataFrame({"y": np.arange(n, dtype=float)})
        sd = prepare_series(df, "y", window=10, horizon=3, seasonal_period=1, train_ratio=0.6, val_ratio=0.2)
        tr_end, va_end = sd.boundaries["train_end"], sd.boundaries["val_end"]
        y_train_raw = sd.unscale_target(sd.y_train)
        y_test_raw = sd.unscale_target(sd.y_test)
        assert y_train_raw.max() < tr_end  # value == time index here
        assert y_test_raw.min() >= va_end
        # input windows end strictly before their first target
        x_last = sd.X_train[:, -1, 0] * sd.channel_std[0] + sd.channel_mean[0]
        np.testing.assert_allclose(x_last + 1, y_train_raw[:, 0], atol=1e-4)

    def test_causal_fill_never_uses_future_values(self):
        s = pd.DataFrame({"v": [1.0, np.nan, np.nan, 9.0]})
        out = causal_fill(s)
        assert out["v"].tolist() == [1.0, 1.0, 1.0, 9.0]

    def test_leading_gap_filled_with_train_median(self):
        s = pd.DataFrame({"v": [np.nan, 2.0, 4.0]})
        out = causal_fill(s, fallback=pd.Series({"v": 3.0}))
        assert out["v"].iloc[0] == 3.0

    def test_rolling_origin_folds_expand_and_stay_ordered(self):
        folds = rolling_origin_splits(100, 3)
        assert len(folds) == 3
        for tr, va in folds:
            assert tr.max() < va.min()
        assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])

    def test_lag_window_features_use_past_only(self):
        v = np.arange(10.0)
        F = lag_window_features(v, 3)
        assert F.shape == (10, 3)
        np.testing.assert_allclose(F[5], [3, 4, 5])
        np.testing.assert_allclose(F[0], [0, 0, 0])

    def test_too_short_series_rejected(self):
        with pytest.raises(ValueError):
            prepare_series(pd.DataFrame({"y": np.arange(20.0)}), "y", window=24, horizon=1, seasonal_period=1)
