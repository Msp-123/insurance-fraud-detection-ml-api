"""
Unit tests for the artifact-free statistics in src/stability.py:
coefficient of variation, the per-fold summary aggregation and the
human-readable stability verdict.
"""

import numpy as np
import pytest

from stability import (
    coefficient_of_variation,
    stability_summary,
    interpret_stability,
)


class TestCoefficientOfVariation:
    def test_zero_for_constant(self):
        assert coefficient_of_variation([0.5, 0.5, 0.5]) == 0.0

    def test_known_value(self):
        scores = np.array([0.2, 0.4])  # mean 0.3, std 0.1
        assert coefficient_of_variation(scores) == pytest.approx(0.1 / 0.3)

    def test_zero_mean_guard(self):
        # All zeros -> mean 0 -> guarded to 0.0 (no division error).
        assert coefficient_of_variation([0.0, 0.0]) == 0.0

    def test_higher_spread_higher_cv(self):
        tight = coefficient_of_variation([0.50, 0.51, 0.49])
        wide = coefficient_of_variation([0.20, 0.80, 0.50])
        assert wide > tight


class TestStabilitySummary:
    def test_fields_and_values(self):
        out = stability_summary([0.2, 0.3, 0.4, 0.5])
        assert out["n_scores"] == 4
        assert out["mean"] == pytest.approx(0.35)
        assert out["min"] == pytest.approx(0.2)
        assert out["max"] == pytest.approx(0.5)
        assert out["range"] == pytest.approx(0.3)
        assert "coefficient_of_variation" in out

    def test_range_is_max_minus_min(self):
        out = stability_summary([0.1, 0.9])
        assert out["range"] == pytest.approx(out["max"] - out["min"])


class TestInterpretStability:
    @pytest.mark.parametrize(
        "cv,expected_keyword",
        [
            (0.01, "Very stable"),
            (0.07, "Stable"),
            (0.15, "Moderately stable"),
            (0.30, "Unstable"),
        ],
    )
    def test_verdict_bands(self, cv, expected_keyword):
        assert expected_keyword in interpret_stability(cv)

    def test_boundary_at_five_percent(self):
        # 0.05 is not < 0.05, so it falls into the next band ("Stable").
        assert "Stable" in interpret_stability(0.05)
        assert "Very" not in interpret_stability(0.05)
