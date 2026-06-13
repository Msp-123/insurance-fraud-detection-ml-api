"""
Unit tests for src/feature_engineering.py

These are pure tests: no model artifacts and no dataset required.
They verify that engineered binary flags are produced correctly and that the
`safe_*` helpers behave defensively when columns are missing.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    apply_feature_engineering,
    get_created_features,
    get_feature_engineering_steps,
    safe_equals,
    safe_isin,
    safe_not_equals,
    safe_numeric_greater_equal,
    safe_numeric_less_than,
    column_exists,
    clean_special_values,
    add_demographic_features,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestSafeHelpers:
    def test_column_exists(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert column_exists(df, "A") is True
        assert column_exists(df, "B") is False

    def test_safe_equals_basic(self):
        df = pd.DataFrame({"Fault": ["Policy Holder", "Third Party", "Policy Holder"]})
        result = safe_equals(df, "Fault", "Policy Holder")
        assert result.tolist() == [1, 0, 1]

    def test_safe_equals_missing_column_returns_zeros(self):
        df = pd.DataFrame({"X": [1, 2, 3]})
        result = safe_equals(df, "Fault", "Policy Holder")
        assert result.tolist() == [0, 0, 0]
        # Index must be preserved so the column can be assigned back to df.
        assert result.index.tolist() == df.index.tolist()

    def test_safe_not_equals(self):
        df = pd.DataFrame({"AddressChange_Claim": ["no change", "1 year"]})
        result = safe_not_equals(df, "AddressChange_Claim", "no change")
        assert result.tolist() == [0, 1]

    def test_safe_not_equals_missing_column_returns_zeros(self):
        df = pd.DataFrame({"X": [1]})
        assert safe_not_equals(df, "Missing", "x").tolist() == [0]

    def test_safe_isin(self):
        df = pd.DataFrame({"PastNumberOfClaims": ["none", "2 to 4", "more than 4"]})
        result = safe_isin(df, "PastNumberOfClaims", ["2 to 4", "more than 4"])
        assert result.tolist() == [0, 1, 1]

    def test_safe_isin_missing_column_returns_zeros(self):
        df = pd.DataFrame({"X": [1, 2]})
        assert safe_isin(df, "Missing", ["a"]).tolist() == [0, 0]

    def test_safe_numeric_greater_equal(self):
        df = pd.DataFrame({"Deductible": [300, 500, 700]})
        result = safe_numeric_greater_equal(df, "Deductible", 500)
        assert result.tolist() == [0, 1, 1]

    def test_safe_numeric_greater_equal_coerces_strings(self):
        df = pd.DataFrame({"Deductible": ["300", "500", "bad"]})
        result = safe_numeric_greater_equal(df, "Deductible", 500)
        # Non-numeric becomes NaN, comparison yields False -> 0.
        assert result.tolist() == [0, 1, 0]

    def test_safe_numeric_less_than(self):
        df = pd.DataFrame({"Age": [18, 25, 40]})
        result = safe_numeric_less_than(df, "Age", 25)
        assert result.tolist() == [1, 0, 0]


# ---------------------------------------------------------------------------
# Individual feature steps
# ---------------------------------------------------------------------------

class TestFeatureSteps:
    def test_clean_special_values_age_zero(self):
        df = pd.DataFrame({"Age": [0, 30, 0]})
        out = clean_special_values(df)
        assert out["Age_Zero_Flag"].tolist() == [1, 0, 1]
        # Age 0 should be turned into NaN (suspicious value).
        assert np.isnan(out["Age"].iloc[0])
        assert out["Age"].iloc[1] == 30

    def test_clean_special_values_unknown_claimed_dates(self):
        df = pd.DataFrame(
            {"MonthClaimed": ["0", "Jan"], "DayOfWeekClaimed": ["0", "Monday"]}
        )
        out = clean_special_values(df)
        assert out["MonthClaimed"].tolist() == ["Unknown", "Jan"]
        assert out["DayOfWeekClaimed"].tolist() == ["Unknown", "Monday"]

    def test_clean_special_values_does_not_mutate_input(self):
        df = pd.DataFrame({"Age": [0, 30]})
        _ = clean_special_values(df)
        # Original frame must remain untouched (functions copy internally).
        assert df["Age"].tolist() == [0, 30]

    def test_demographic_features(self):
        df = pd.DataFrame({"Age": [20, 70, 40], "Sex": ["Male", "Female", "Male"]})
        out = add_demographic_features(df)
        assert out["is_young_driver"].tolist() == [1, 0, 0]
        assert out["is_senior_driver"].tolist() == [0, 1, 0]
        assert out["is_male"].tolist() == [1, 0, 1]
        assert out["is_female"].tolist() == [0, 1, 0]

    def test_demographic_features_without_age_column(self):
        df = pd.DataFrame({"Sex": ["Male"]})
        out = add_demographic_features(df)
        # Missing Age must not crash; senior flag defaults to 0.
        assert out["is_senior_driver"].tolist() == [0]
        assert out["is_young_driver"].tolist() == [0]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestApplyFeatureEngineering:
    def test_pipeline_adds_expected_features(self, raw_claims_df):
        out = apply_feature_engineering(raw_claims_df)

        expected_new = {
            "is_policy_holder_fault",
            "is_third_party_fault",
            "is_no_police_report",
            "is_witness_present",
            "is_external_agent",
            "has_address_change",
            "is_young_driver",
            "is_male",
            "is_high_vehicle_price",
            "is_all_perils_policy",
        }
        assert expected_new.issubset(set(out.columns))

    def test_pipeline_preserves_row_count(self, raw_claims_df):
        out = apply_feature_engineering(raw_claims_df)
        assert len(out) == len(raw_claims_df)

    def test_pipeline_does_not_mutate_input(self, raw_claims_df):
        before = list(raw_claims_df.columns)
        _ = apply_feature_engineering(raw_claims_df)
        assert list(raw_claims_df.columns) == before

    def test_pipeline_is_deterministic(self, raw_claims_df):
        out1 = apply_feature_engineering(raw_claims_df)
        out2 = apply_feature_engineering(raw_claims_df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_engineered_flags_correct_for_known_row(self, sample_claim):
        df = pd.DataFrame([sample_claim])
        out = apply_feature_engineering(df)
        # sample_claim: Policy Holder fault, no police report, external agent, young driver.
        assert out["is_policy_holder_fault"].iloc[0] == 1
        assert out["is_third_party_fault"].iloc[0] == 0
        assert out["is_no_police_report"].iloc[0] == 1
        assert out["is_external_agent"].iloc[0] == 1
        assert out["is_young_driver"].iloc[0] == 1  # Age 21 < 25

    def test_works_on_empty_columns_subset(self):
        # Even a frame missing most columns must not raise: helpers are defensive.
        df = pd.DataFrame({"SomeUnrelatedColumn": [1, 2]})
        out = apply_feature_engineering(df)
        assert len(out) == 2
        # All engineered flags should exist and be 0 for missing source columns.
        assert out["is_policy_holder_fault"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Registry / introspection helpers
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_feature_engineering_steps(self):
        steps = get_feature_engineering_steps()
        assert not steps.empty
        assert {"order", "name", "description", "function"}.issubset(steps.columns)
        # Orders should be 1..N with no gaps.
        assert steps["order"].tolist() == list(range(1, len(steps) + 1))

    def test_get_created_features(self, raw_claims_df):
        created = get_created_features(raw_claims_df)
        assert isinstance(created, list)
        assert "is_policy_holder_fault" in created
        # None of the original columns should be reported as "created".
        assert "Fault" not in created
