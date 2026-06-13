"""
Probability calibration for the fraud model.

A model that ranks well (good PR-AUC) can still output poorly-calibrated
probabilities — e.g. claims it scores "0.8" may not actually be fraud 80% of
the time. For insurance decisioning the probability itself is used (risk bands,
expected-cost thresholds), so calibration matters.

This module:
- fits the base XGBoost model on a calibration-train split,
- wraps it with CalibratedClassifierCV using both sigmoid (Platt) and isotonic,
- measures Brier score and log loss before/after on a held-out split,
- saves a reliability-curve plot and the best calibrated model.

Outputs:
- artifacts/calibrated_model.pkl
- artifacts/calibration_results.json
- reports/model/calibration_curve.png
"""

import json

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from xgboost import XGBClassifier

from config import (
    RANDOM_STATE,
    CALIBRATED_MODEL_PATH,
    CALIBRATION_REPORT_PATH,
    MODEL_REPORT_DIR,
    ARTIFACTS_DIR,
)
from cv_utils import load_preprocessed_data, compute_scale_pos_weight


# =========================================================
# Base model
# =========================================================

def build_base_model(scale_pos_weight: float) -> XGBClassifier:
    """Same shape as the production XGBoost model."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =========================================================
# Metrics
# =========================================================

def calibration_metrics(y_true, y_proba) -> dict:
    """Brier score (lower is better) and log loss for a probability vector."""
    return {
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 6),
        "log_loss": round(float(log_loss(y_true, y_proba, labels=[0, 1])), 6),
    }


# =========================================================
# Calibration
# =========================================================

def calibrate(X_train, y_train, X_eval, y_eval) -> dict:
    """
    Fit base + sigmoid + isotonic calibrated models, score each on eval split.

    Returns a dict with metrics for every variant and the fitted estimators
    for the best one (by Brier score).
    """

    scale_pos_weight = compute_scale_pos_weight(y_train)

    # Uncalibrated baseline.
    base = build_base_model(scale_pos_weight)
    base.fit(X_train, y_train)
    base_proba = base.predict_proba(X_eval)[:, 1]

    variants = {"uncalibrated": {"model": base, "metrics": calibration_metrics(y_eval, base_proba),
                                 "proba": base_proba}}

    # Calibrated variants. cv="prefit" calibrates the already-fitted base model
    # on the eval split is not ideal; instead we let CalibratedClassifierCV do
    # its own internal CV on the training split (no leakage from eval).
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(
            estimator=build_base_model(scale_pos_weight),
            method=method,
            cv=3,
        )
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_eval)[:, 1]
        variants[method] = {
            "model": calibrated,
            "metrics": calibration_metrics(y_eval, proba),
            "proba": proba,
        }

    return variants


def pick_best(variants: dict) -> str:
    """Best variant = lowest Brier score on the eval split."""
    return min(variants, key=lambda k: variants[k]["metrics"]["brier_score"])


# =========================================================
# Reporting
# =========================================================

def save_calibration_plot(y_eval, variants: dict):
    """Reliability diagram for every variant + the perfect-calibration line."""
    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for name, info in variants.items():
        frac_pos, mean_pred = calibration_curve(
            y_eval, info["proba"], n_bins=10, strategy="quantile"
        )
        plt.plot(mean_pred, frac_pos, marker="o",
                 label=f"{name} (Brier={info['metrics']['brier_score']:.4f})")

    plt.title("Calibration (Reliability) Curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "calibration_curve.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return output_path


def save_results(variants: dict, best_name: str):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the best calibrated estimator (skip if the raw model is best).
    joblib.dump(variants[best_name]["model"], CALIBRATED_MODEL_PATH)

    report = {
        "best_method": best_name,
        "metrics_by_method": {
            name: info["metrics"] for name, info in variants.items()
        },
        "note": (
            "Brier score and log loss measured on a held-out calibration "
            "evaluation split. Lower is better for both."
        ),
    }
    with open(CALIBRATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report


# =========================================================
# Main
# =========================================================

def main():
    from sklearn.model_selection import train_test_split

    data = load_preprocessed_data()
    X_train_full = data["X_train_processed"]
    y_train_full = np.asarray(data["y_train"])

    # Split the training data into calibration-train / calibration-eval so the
    # test set stays untouched for final reporting elsewhere.
    X_tr, X_eval, y_tr, y_eval = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print("=" * 70)
    print("PROBABILITY CALIBRATION (XGBoost)")
    print("=" * 70)

    variants = calibrate(X_tr, y_tr, X_eval, y_eval)
    best_name = pick_best(variants)

    save_calibration_plot(y_eval, variants)
    report = save_results(variants, best_name)

    print("\nBrier score by method (lower is better):")
    for name, info in variants.items():
        marker = "  <-- best" if name == best_name else ""
        print(f"  {name:13s}: brier={info['metrics']['brier_score']:.5f}  "
              f"log_loss={info['metrics']['log_loss']:.5f}{marker}")

    print("\nArtifacts saved:")
    print(f"- {CALIBRATED_MODEL_PATH}")
    print(f"- {CALIBRATION_REPORT_PATH}")
    print(f"- {MODEL_REPORT_DIR / 'calibration_curve.png'}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
