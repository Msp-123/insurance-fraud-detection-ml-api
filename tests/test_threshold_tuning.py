"""
Unit tests for src/threshold_tuning.py

Tests the threshold sweep and the business selection strategy with fully
synthetic probabilities — no model or dataset required.
"""

import numpy as np
import pandas as pd
import pytest

from threshold_tuning import (
    evaluate_thresholds,
    select_best_threshold,
    MIN_ACCEPTABLE_RECALL,
)


@pytest.fixture
def synthetic_scores():
    """
    20 samples: positives generally score higher than negatives, with a little
    overlap so different thresholds produce genuinely different metrics.
    """
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55,
                        0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    return y_true, y_proba


class TestEvaluateThresholds:
    def test_returns_one_row_per_threshold(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        thresholds = [0.3, 0.5, 0.7]
        df = evaluate_thresholds(y_true, y_proba, thresholds)
        assert len(df) == len(thresholds)

    def test_expected_columns_present(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        df = evaluate_thresholds(y_true, y_proba, [0.5])
        for col in [
            "threshold", "precision", "recall", "f1_score",
            "true_negative", "false_positive", "false_negative", "true_positive",
            "predicted_fraud_count", "predicted_not_fraud_count",
        ]:
            assert col in df.columns

    def test_confusion_counts_sum_to_total(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        df = evaluate_thresholds(y_true, y_proba, [0.5])
        row = df.iloc[0]
        total = row.true_negative + row.false_positive + row.false_negative + row.true_positive
        assert total == len(y_true)

    def test_recall_decreases_as_threshold_rises(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        df = evaluate_thresholds(y_true, y_proba, [0.2, 0.5, 0.8])
        recalls = df.sort_values("threshold")["recall"].tolist()
        # Recall is monotonically non-increasing in the threshold.
        assert recalls == sorted(recalls, reverse=True)

    def test_metrics_within_unit_interval(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        df = evaluate_thresholds(y_true, y_proba, [0.1, 0.5, 0.9])
        for col in ["precision", "recall", "f1_score"]:
            assert df[col].between(0.0, 1.0).all()


class TestSelectBestThreshold:
    def test_best_f1_row_is_actual_max(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        df = evaluate_thresholds(y_true, y_proba, np.round(np.arange(0.1, 0.91, 0.05), 2))
        info = select_best_threshold(df)
        assert info["best_f1_score"] == pytest.approx(df["f1_score"].max())

    def test_business_threshold_respects_min_recall(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        df = evaluate_thresholds(y_true, y_proba, np.round(np.arange(0.1, 0.91, 0.05), 2))
        info = select_best_threshold(df)
        # When an eligible threshold exists, selected recall must meet the floor.
        eligible = df[df["recall"] >= MIN_ACCEPTABLE_RECALL]
        if not eligible.empty:
            assert info["selected_recall"] >= MIN_ACCEPTABLE_RECALL
            assert "maximizes precision" in info["selection_reason"]

    def test_fallback_when_no_threshold_meets_recall(self):
        # All recalls are low -> strategy must fall back to best-F1.
        df = pd.DataFrame(
            {
                "threshold": [0.5, 0.6],
                "precision": [0.9, 0.95],
                "recall": [0.2, 0.1],            # both below the 0.7 floor
                "f1_score": [0.33, 0.18],
                "false_positive": [1, 0],
                "false_negative": [8, 9],
                "true_positive": [2, 1],
                "true_negative": [9, 10],
            }
        )
        info = select_best_threshold(df)
        assert "Fallback" in info["selection_reason"]
        assert info["selected_threshold"] == info["best_f1_threshold"]

    def test_output_contains_all_expected_keys(self, synthetic_scores):
        y_true, y_proba = synthetic_scores
        df = evaluate_thresholds(y_true, y_proba, [0.3, 0.5, 0.7])
        info = select_best_threshold(df)
        for key in [
            "selected_threshold", "selection_strategy", "minimum_acceptable_recall",
            "selected_precision", "selected_recall", "selected_f1_score",
            "best_f1_threshold", "best_f1_score",
        ]:
            assert key in info
