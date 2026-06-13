"""
Lightweight tests for the model-improvement builder functions.

These construct estimators and search spaces (no fitting / no CV), so they run
fast and don't need the dataset. They guard against import errors and obviously
broken configuration in hyperparameter_tuning.py, model_comparison.py,
calibration.py and stability.py.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# hyperparameter_tuning.py
# ---------------------------------------------------------------------------

class TestHyperparameterTuning:
    def test_search_space_has_expected_keys(self):
        from hyperparameter_tuning import build_search_space

        space = build_search_space()
        for key in [
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_weight",
            "gamma", "reg_alpha", "reg_lambda",
        ]:
            assert key in space

    def test_search_space_samples_are_in_range(self):
        from hyperparameter_tuning import build_search_space

        space = build_search_space()
        rng = np.random.default_rng(0)
        # Each distribution should sample without error and stay sane.
        depth = space["max_depth"].rvs(random_state=rng)
        lr = space["learning_rate"].rvs(random_state=rng)
        assert 3 <= depth < 8
        assert 0.0 < lr <= 0.20

    def test_base_estimator_sets_scale_pos_weight(self):
        from hyperparameter_tuning import build_base_estimator

        est = build_base_estimator(scale_pos_weight=15.0)
        assert est.get_params()["scale_pos_weight"] == 15.0

    def test_to_native_converts_numpy(self):
        from hyperparameter_tuning import _to_native

        assert isinstance(_to_native(np.int64(3)), int)
        assert isinstance(_to_native(np.float64(0.5)), float)
        assert _to_native("xgboost") == "xgboost"


# ---------------------------------------------------------------------------
# model_comparison.py
# ---------------------------------------------------------------------------

class TestModelComparisonBuilders:
    def test_builds_three_models(self):
        from model_comparison import build_models

        models = build_models(scale_pos_weight=15.0)
        assert set(models.keys()) == {"XGBoost", "LightGBM", "CatBoost"}

    def test_imbalance_handling_applied(self):
        from model_comparison import build_models

        models = build_models(scale_pos_weight=15.0)
        assert models["XGBoost"].get_params()["scale_pos_weight"] == 15.0
        assert models["LightGBM"].get_params()["scale_pos_weight"] == 15.0
        # CatBoost uses auto_class_weights instead of scale_pos_weight.
        assert models["CatBoost"].get_params()["auto_class_weights"] == "Balanced"

    def test_catboost_estimator_is_cloneable(self):
        # Regression guard: a dict class_weights broke sklearn clone(); the
        # auto_class_weights form must round-trip.
        from sklearn.base import clone
        from model_comparison import build_models

        cat = build_models(scale_pos_weight=15.0)["CatBoost"]
        clone(cat)  # must not raise


# ---------------------------------------------------------------------------
# calibration.py / stability.py model builders
# ---------------------------------------------------------------------------

class TestOtherBuilders:
    def test_calibration_base_model(self):
        from calibration import build_base_model

        model = build_base_model(scale_pos_weight=10.0)
        assert model.get_params()["scale_pos_weight"] == 10.0

    def test_calibration_metrics_keys(self):
        from calibration import calibration_metrics

        out = calibration_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        assert "brier_score" in out and "log_loss" in out
        assert out["brier_score"] >= 0

    def test_stability_build_model(self):
        from stability import build_model

        model = build_model(scale_pos_weight=10.0)
        assert model.get_params()["scale_pos_weight"] == 10.0
