"""
End-to-end integration tests that exercise the real trained artifacts
(model.pkl, preprocessor.pkl, threshold.json, ...).

These are skipped automatically when the artifacts are not present (e.g. a
fresh clone), so the suite still passes on CI before training has run.
"""

import pandas as pd
import pytest

from conftest import requires_artifacts


@requires_artifacts
class TestPredictSingleClaim:
    def test_returns_expected_schema(self, real_sample_claim):
        from predict import predict_single_claim

        result = predict_single_claim(real_sample_claim)

        for key in [
            "fraud_probability", "threshold", "prediction",
            "prediction_label", "risk_level", "recommendation",
        ]:
            assert key in result

    def test_probability_in_unit_interval(self, real_sample_claim):
        from predict import predict_single_claim

        result = predict_single_claim(real_sample_claim)
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_prediction_consistent_with_threshold(self, real_sample_claim):
        from predict import predict_single_claim

        result = predict_single_claim(real_sample_claim)
        expected = int(result["fraud_probability"] >= result["threshold"])
        assert result["prediction"] == expected
        assert result["prediction_label"] in {"Fraud", "Not Fraud"}

    def test_risk_level_valid(self, real_sample_claim):
        from predict import predict_single_claim

        result = predict_single_claim(real_sample_claim)
        assert result["risk_level"] in {"Low", "Medium", "High"}


@requires_artifacts
class TestPredictBatch:
    def test_batch_matches_single(self, real_sample_claim):
        from predict import predict_single_claim, predict_batch_claims

        single = predict_single_claim(real_sample_claim)
        batch = predict_batch_claims(pd.DataFrame([real_sample_claim]))

        assert len(batch) == 1
        # Same input -> same probability via either code path.
        assert batch["fraud_probability"].iloc[0] == pytest.approx(
            single["fraud_probability"], abs=1e-4
        )


@requires_artifacts
class TestExplain:
    def test_explanation_has_top_reasons(self, real_sample_claim):
        from explain import explain_single_claim

        result = explain_single_claim(real_sample_claim, top_n=5)
        assert "top_reasons" in result
        assert len(result["top_reasons"]) <= 5
        if result["top_reasons"]:
            reason = result["top_reasons"][0]
            assert "shap_value" in reason
            assert "reason" in reason
            assert reason["direction"] in {"increases_fraud_risk", "decreases_fraud_risk"}

    def test_top_reasons_sorted_by_abs_shap(self, real_sample_claim):
        from explain import explain_single_claim

        result = explain_single_claim(real_sample_claim, top_n=10)
        abs_values = [r["absolute_shap_value"] for r in result["top_reasons"]]
        assert abs_values == sorted(abs_values, reverse=True)
