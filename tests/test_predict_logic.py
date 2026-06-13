"""
Unit tests for the pure (artifact-free) logic in src/predict.py:
risk-level assignment, business recommendation, input shaping/alignment and
the threshold loader fallback.
"""

import json

import numpy as np
import pandas as pd
import pytest

import predict
from predict import (
    assign_risk_level,
    create_recommendation,
    convert_input_to_dataframe,
    prepare_input_for_prediction,
    load_selected_threshold,
)
from config import RISK_LEVEL_THRESHOLDS, DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Risk level
# ---------------------------------------------------------------------------

class TestAssignRiskLevel:
    def test_low(self):
        assert assign_risk_level(0.0) == "Low"
        assert assign_risk_level(RISK_LEVEL_THRESHOLDS["low"] - 0.01) == "Low"

    def test_medium(self):
        assert assign_risk_level(RISK_LEVEL_THRESHOLDS["low"]) == "Medium"
        assert assign_risk_level(RISK_LEVEL_THRESHOLDS["medium"] - 0.01) == "Medium"

    def test_high(self):
        assert assign_risk_level(RISK_LEVEL_THRESHOLDS["medium"]) == "High"
        assert assign_risk_level(1.0) == "High"

    @pytest.mark.parametrize(
        "prob,expected",
        [(0.0, "Low"), (0.29, "Low"), (0.30, "Medium"), (0.69, "Medium"),
         (0.70, "High"), (0.99, "High")],
    )
    def test_boundaries_parametrized(self, prob, expected):
        assert assign_risk_level(prob) == expected


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class TestCreateRecommendation:
    def test_high_risk_always_investigate(self):
        assert create_recommendation(0.9, 1, "High") == "Manual investigation required"
        assert create_recommendation(0.8, 0, "High") == "Manual investigation required"

    def test_medium_risk_predicted_fraud(self):
        assert (
            create_recommendation(0.5, 1, "Medium")
            == "Additional document check recommended"
        )

    def test_medium_risk_not_fraud(self):
        assert (
            create_recommendation(0.5, 0, "Medium")
            == "Monitor claim and continue standard process"
        )

    def test_low_risk_standard(self):
        assert create_recommendation(0.1, 0, "Low") == "Standard claim process"


# ---------------------------------------------------------------------------
# Input conversion
# ---------------------------------------------------------------------------

class TestConvertInputToDataframe:
    def test_dict_becomes_single_row(self):
        df = convert_input_to_dataframe({"a": 1, "b": 2})
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 2)

    def test_dataframe_is_copied_not_aliased(self):
        original = pd.DataFrame({"a": [1]})
        out = convert_input_to_dataframe(original)
        out["a"] = 999
        assert original["a"].iloc[0] == 1  # original untouched

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            convert_input_to_dataframe([1, 2, 3])


# ---------------------------------------------------------------------------
# Prepare input for prediction (column alignment)
# ---------------------------------------------------------------------------

class TestPrepareInputForPrediction:
    def test_aligns_to_raw_feature_columns(self, sample_claim):
        raw_cols = ["Age", "Fault", "is_policy_holder_fault", "NonexistentCol"]
        X = prepare_input_for_prediction(sample_claim, raw_cols)
        # Output columns must exactly match (and be in the order of) raw_cols.
        assert list(X.columns) == raw_cols

    def test_missing_columns_filled_with_nan(self, sample_claim):
        raw_cols = ["Age", "ColumnThatDoesNotExist"]
        X = prepare_input_for_prediction(sample_claim, raw_cols)
        assert X["ColumnThatDoesNotExist"].isna().all()

    def test_drops_target_and_id_columns(self, sample_claim):
        claim = dict(sample_claim)
        claim["FraudFound_P"] = 1  # target should never survive into features
        # Ask for a feature set that does NOT include target/id.
        raw_cols = ["Age", "Fault"]
        X = prepare_input_for_prediction(claim, raw_cols)
        assert "FraudFound_P" not in X.columns
        assert "PolicyNumber" not in X.columns

    def test_engineered_feature_present_after_prepare(self, sample_claim):
        raw_cols = ["is_policy_holder_fault"]
        X = prepare_input_for_prediction(sample_claim, raw_cols)
        assert X["is_policy_holder_fault"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Threshold loader
# ---------------------------------------------------------------------------

class TestLoadSelectedThreshold:
    def test_falls_back_to_default_when_missing(self, monkeypatch, tmp_path):
        missing = tmp_path / "nope.json"
        monkeypatch.setattr(predict, "THRESHOLD_PATH", missing)
        assert load_selected_threshold() == DEFAULT_THRESHOLD

    def test_reads_selected_threshold(self, monkeypatch, tmp_path):
        path = tmp_path / "threshold.json"
        path.write_text(json.dumps({"selected_threshold": 0.42}), encoding="utf-8")
        monkeypatch.setattr(predict, "THRESHOLD_PATH", path)
        assert load_selected_threshold() == 0.42

    def test_falls_back_when_key_absent(self, monkeypatch, tmp_path):
        path = tmp_path / "threshold.json"
        path.write_text(json.dumps({"other": 1}), encoding="utf-8")
        monkeypatch.setattr(predict, "THRESHOLD_PATH", path)
        assert load_selected_threshold() == DEFAULT_THRESHOLD
