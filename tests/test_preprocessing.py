"""
Unit tests for src/preprocessing.py

Covers the artifact-free building blocks: feature/target split, preprocessor
construction, fit/transform shape behaviour and feature-name cleaning.
No dataset on disk is required — small synthetic frames are used.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    split_features_target,
    build_preprocessor,
    get_feature_names,
    create_one_hot_encoder,
)
from config import TARGET, ID_COLUMNS


def _toy_frame(n=20):
    """Small mixed numeric/categorical frame with a target column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "PolicyNumber": range(n),                      # id column
            "Age": rng.integers(18, 70, n),                # numeric
            "Deductible": rng.choice([300, 400, 500], n),  # numeric
            "Sex": rng.choice(["Male", "Female"], n),      # categorical
            "Fault": rng.choice(["Policy Holder", "Third Party"], n),
            TARGET: rng.integers(0, 2, n),                 # target
        }
    )


class TestSplitFeaturesTarget:
    def test_target_removed_from_features(self):
        df = _toy_frame()
        X, y = split_features_target(df)
        assert TARGET not in X.columns

    def test_id_columns_removed(self):
        df = _toy_frame()
        X, _ = split_features_target(df)
        for col in ID_COLUMNS:
            assert col not in X.columns

    def test_target_values_returned_as_int(self):
        df = _toy_frame()
        _, y = split_features_target(df)
        assert y.dtype.kind in ("i", "u")
        assert set(y.unique()).issubset({0, 1})

    def test_handles_missing_id_column_gracefully(self):
        df = _toy_frame().drop(columns=["PolicyNumber"])
        X, y = split_features_target(df)  # must not raise
        assert TARGET not in X.columns


class TestOneHotEncoder:
    def test_encoder_is_dense(self):
        enc = create_one_hot_encoder()
        out = enc.fit_transform(pd.DataFrame({"c": ["a", "b", "a"]}))
        # Dense output (not a scipy sparse matrix).
        assert isinstance(out, np.ndarray)

    def test_handle_unknown_ignore(self):
        enc = create_one_hot_encoder()
        enc.fit(pd.DataFrame({"c": ["a", "b"]}))
        # Unseen category at transform time must not raise.
        out = enc.transform(pd.DataFrame({"c": ["zzz"]}))
        assert out.shape[0] == 1


class TestBuildPreprocessor:
    def test_fit_transform_shapes_match(self):
        df = _toy_frame()
        X, _ = split_features_target(df)
        pre = build_preprocessor(X)
        Xt = pre.fit_transform(X)
        assert Xt.shape[0] == X.shape[0]
        # One-hot expands columns, so processed width >= raw numeric count.
        assert Xt.shape[1] >= 2

    def test_transform_consistent_width_train_test(self):
        df = _toy_frame(40)
        X, _ = split_features_target(df)
        X_train, X_test = X.iloc[:30], X.iloc[30:]
        pre = build_preprocessor(X_train)
        Xt_train = pre.fit_transform(X_train)
        Xt_test = pre.transform(X_test)
        assert Xt_train.shape[1] == Xt_test.shape[1]

    def test_feature_names_cleaned_and_match_width(self):
        df = _toy_frame()
        X, _ = split_features_target(df)
        pre = build_preprocessor(X)
        Xt = pre.fit_transform(X)
        names = get_feature_names(pre)
        assert len(names) == Xt.shape[1]
        # Prefixes added by ColumnTransformer must be stripped.
        assert all(not n.startswith("numeric__") for n in names)
        assert all(not n.startswith("categorical__") for n in names)

    def test_median_imputation_fills_numeric_nan(self):
        df = _toy_frame()
        df.loc[0, "Age"] = np.nan
        X, _ = split_features_target(df)
        pre = build_preprocessor(X)
        Xt = pre.fit_transform(X)
        # No NaNs should remain after imputation.
        assert not np.isnan(Xt).any()
