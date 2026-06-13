"""
Unit tests for the artifact-free SHAP helpers in src/explain.py:
shap-value extraction across formats, feature-name alignment, mapping a
one-hot processed name back to its raw column, and reason-text generation.
"""

import numpy as np
import pytest

from explain import (
    extract_shap_values,
    align_feature_names,
    infer_raw_feature_name,
    build_reason_text,
)


class TestExtractShapValues:
    def test_list_of_two_returns_positive_class(self):
        neg = np.zeros((3, 4))
        pos = np.ones((3, 4))
        out = extract_shap_values([neg, pos])
        assert np.array_equal(out, pos)

    def test_list_of_one_returns_first(self):
        arr = np.arange(12).reshape(3, 4)
        out = extract_shap_values([arr])
        assert np.array_equal(out, arr)

    def test_2d_array_passthrough(self):
        arr = np.random.default_rng(0).normal(size=(5, 6))
        out = extract_shap_values(arr)
        assert out.shape == (5, 6)

    def test_3d_array_selects_positive_class_slice(self):
        arr = np.zeros((5, 6, 2))
        arr[:, :, 1] = 7.0  # positive-class slice
        out = extract_shap_values(arr)
        assert out.shape == (5, 6)
        assert np.all(out == 7.0)


class TestAlignFeatureNames:
    def test_returns_names_when_count_matches(self):
        names = ["a", "b", "c"]
        assert align_feature_names(names, 3) == names

    def test_generates_placeholder_names_on_mismatch(self):
        out = align_feature_names(["a", "b"], 4)
        assert out == ["feature_0", "feature_1", "feature_2", "feature_3"]


class TestInferRawFeatureName:
    def test_exact_match_numeric_feature(self):
        result = infer_raw_feature_name("Age", ["Age", "BasePolicy"])
        assert result == {"raw_column": "Age", "category_value": None}

    def test_one_hot_split(self):
        result = infer_raw_feature_name("BasePolicy_All Perils", ["BasePolicy", "Age"])
        assert result["raw_column"] == "BasePolicy"
        assert result["category_value"] == "All Perils"

    def test_longest_prefix_wins(self):
        # "Days_Policy_Claim" must win over the shorter "Days_Policy_Accident"
        # style prefixes; longest column name is tried first.
        cols = ["Days", "Days_Policy_Claim"]
        result = infer_raw_feature_name("Days_Policy_Claim_none", cols)
        assert result["raw_column"] == "Days_Policy_Claim"
        assert result["category_value"] == "none"

    def test_unmappable_feature_returns_nones(self):
        result = infer_raw_feature_name("totally_unknown", ["Age", "Sex"])
        assert result == {"raw_column": None, "category_value": None}


class TestBuildReasonText:
    def test_categorical_increase(self):
        text = build_reason_text("BasePolicy_All Perils", "BasePolicy", "All Perils", 0.5)
        assert "BasePolicy" in text and "All Perils" in text and "increased" in text

    def test_numeric_decrease(self):
        text = build_reason_text("Age", "Age", None, -0.3)
        assert "Age" in text and "decreased" in text

    def test_fallback_uses_processed_name(self):
        text = build_reason_text("feature_7", None, None, 0.1)
        assert "feature_7" in text and "increased" in text
