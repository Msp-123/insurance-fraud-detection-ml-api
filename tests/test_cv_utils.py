"""
Unit tests for src/cv_utils.py — the artifact-free helpers shared by the
model-improvement modules (imbalance weights, cost model, CV splitter, score
summaries).
"""

import numpy as np
import pytest

from cv_utils import (
    compute_scale_pos_weight,
    compute_class_weights,
    expected_cost,
    make_cv,
    summarize_scores,
)
from config import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST


class TestScalePosWeight:
    def test_basic_ratio(self):
        y = np.array([0, 0, 0, 0, 1])  # 4 neg / 1 pos
        assert compute_scale_pos_weight(y) == pytest.approx(4.0)

    def test_balanced_is_one(self):
        y = np.array([0, 1, 0, 1])
        assert compute_scale_pos_weight(y) == pytest.approx(1.0)

    def test_raises_without_positives(self):
        with pytest.raises(ValueError):
            compute_scale_pos_weight(np.zeros(5))

    def test_accepts_list(self):
        assert compute_scale_pos_weight([0, 0, 1]) == pytest.approx(2.0)


class TestClassWeights:
    def test_weights_sum_relationship(self):
        y = np.array([0, 0, 0, 0, 1])  # n=5, neg=4, pos=1
        w = compute_class_weights(y)
        assert w[0] == pytest.approx(5 / (2 * 4))
        assert w[1] == pytest.approx(5 / (2 * 1))

    def test_minority_class_weighted_higher(self):
        y = np.array([0] * 90 + [1] * 10)
        w = compute_class_weights(y)
        assert w[1] > w[0]

    def test_raises_when_one_class_missing(self):
        with pytest.raises(ValueError):
            compute_class_weights(np.ones(5))


class TestExpectedCost:
    def test_uses_default_costs(self):
        # 2 missed frauds + 3 false alarms
        cost = expected_cost(false_negatives=2, false_positives=3)
        assert cost == pytest.approx(2 * FALSE_NEGATIVE_COST + 3 * FALSE_POSITIVE_COST)

    def test_missed_fraud_more_expensive(self):
        fn_cost = expected_cost(false_negatives=1, false_positives=0)
        fp_cost = expected_cost(false_negatives=0, false_positives=1)
        assert fn_cost > fp_cost

    def test_custom_costs(self):
        assert expected_cost(1, 1, fn_cost=5, fp_cost=2) == pytest.approx(7.0)

    def test_zero_errors_zero_cost(self):
        assert expected_cost(0, 0) == 0.0


class TestMakeCv:
    def test_returns_requested_splits(self):
        cv = make_cv(n_splits=4)
        assert cv.get_n_splits() == 4

    def test_is_stratified_and_shuffled(self):
        cv = make_cv(n_splits=3, random_state=1)
        assert cv.shuffle is True
        assert cv.random_state == 1

    def test_preserves_class_ratio_across_folds(self):
        X = np.zeros((100, 2))
        y = np.array([0] * 90 + [1] * 10)
        cv = make_cv(n_splits=5)
        for _, test_idx in cv.split(X, y):
            frac = y[test_idx].mean()
            # Each fold should hold ~10% positives (stratified).
            assert frac == pytest.approx(0.10, abs=0.05)


class TestSummarizeScores:
    def test_basic_stats(self):
        out = summarize_scores([0.2, 0.4, 0.6])
        assert out["mean"] == pytest.approx(0.4)
        assert out["min"] == pytest.approx(0.2)
        assert out["max"] == pytest.approx(0.6)
        assert out["std"] >= 0

    def test_constant_scores_zero_std(self):
        out = summarize_scores([0.5, 0.5, 0.5])
        assert out["std"] == 0.0
        assert out["mean"] == pytest.approx(0.5)
