"""
Shared utilities for the model-improvement modules
(hyperparameter tuning, model comparison, calibration and stability testing).

Everything here is artifact-free and deterministic so it can be unit tested
without a trained model.
"""

from typing import Dict

import numpy as np

from sklearn.model_selection import StratifiedKFold

from config import (
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
    CV_FOLDS,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
)


# =========================================================
# Data loading
# =========================================================

def load_preprocessed_data() -> dict:
    """
    Load the preprocessed train/test split saved by preprocessing.py.
    """

    import joblib

    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {PREPROCESSED_DATA_PATH}. "
            "Run preprocessing.py first."
        )

    data = joblib.load(PREPROCESSED_DATA_PATH)

    required = ["X_train_processed", "X_test_processed", "y_train", "y_test"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key in preprocessed data: {key}")

    return data


# =========================================================
# Class imbalance / cost-sensitive learning
# =========================================================

def compute_scale_pos_weight(y) -> float:
    """
    scale_pos_weight = n_negative / n_positive.

    Used by XGBoost/LightGBM to up-weight the rare positive (fraud) class.
    """

    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0:
        raise ValueError("No positive (fraud) samples found.")

    return n_neg / n_pos


def compute_class_weights(y) -> Dict[int, float]:
    """
    Balanced class weights: n_samples / (n_classes * n_samples_in_class).

    Used by models that accept a class_weight mapping (e.g. CatBoost).
    """

    y = np.asarray(y)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must be present to compute class weights.")

    return {
        0: n / (2.0 * n_neg),
        1: n / (2.0 * n_pos),
    }


def expected_cost(
    false_negatives: int,
    false_positives: int,
    fn_cost: float = FALSE_NEGATIVE_COST,
    fp_cost: float = FALSE_POSITIVE_COST,
) -> float:
    """
    Total business cost of a confusion outcome under an asymmetric cost model.

    A missed fraud (false negative) is far more expensive than an unnecessary
    investigation (false positive). Lower is better.
    """

    return float(false_negatives) * fn_cost + float(false_positives) * fp_cost


# =========================================================
# Cross-validation splitter
# =========================================================

def make_cv(n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """
    Stratified K-fold splitter.

    Stratification keeps the ~6% fraud ratio stable across folds, which matters
    a lot for an imbalanced problem.
    """

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def summarize_scores(scores) -> Dict[str, float]:
    """
    Reduce an array of per-fold scores to mean / std / min / max.
    """

    scores = np.asarray(scores, dtype=float)

    return {
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
    }
