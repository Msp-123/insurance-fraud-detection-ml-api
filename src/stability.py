"""
Model stability testing.

A single train/test number can be lucky. To judge whether the XGBoost fraud
model is *stable*, we repeat stratified K-fold cross-validation across several
random seeds and look at the spread of the scores. Small spread (low std,
small max-min range, low coefficient of variation) => the model generalises
consistently rather than depending on a particular split.

Outputs:
- artifacts/stability_results.json
- reports/model/stability_pr_auc.png   (per-seed PR-AUC distribution)
"""

import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    CV_SCORING,
    STABILITY_SEEDS,
    STABILITY_REPORT_PATH,
    MODEL_REPORT_DIR,
    ARTIFACTS_DIR,
)
from cv_utils import load_preprocessed_data, compute_scale_pos_weight


# =========================================================
# Model
# =========================================================

def build_model(scale_pos_weight: float) -> XGBClassifier:
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
# Stability statistics
# =========================================================

def coefficient_of_variation(scores) -> float:
    """
    std / mean — a scale-free measure of spread. Lower == more stable.
    """

    scores = np.asarray(scores, dtype=float)
    mean = scores.mean()
    if mean == 0:
        return 0.0
    return float(scores.std() / mean)


def stability_summary(all_scores) -> dict:
    """
    Aggregate every per-fold score (across all seeds) into stability stats.
    """

    all_scores = np.asarray(all_scores, dtype=float)

    return {
        "n_scores": int(all_scores.size),
        "mean": round(float(all_scores.mean()), 4),
        "std": round(float(all_scores.std()), 4),
        "min": round(float(all_scores.min()), 4),
        "max": round(float(all_scores.max()), 4),
        "range": round(float(all_scores.max() - all_scores.min()), 4),
        "coefficient_of_variation": round(coefficient_of_variation(all_scores), 4),
    }


def interpret_stability(cv: float) -> str:
    """Human-readable verdict from the coefficient of variation."""
    if cv < 0.05:
        return "Very stable (CV < 5%)."
    if cv < 0.10:
        return "Stable (CV < 10%)."
    if cv < 0.20:
        return "Moderately stable (CV < 20%)."
    return "Unstable (CV >= 20%) — scores depend heavily on the split."


# =========================================================
# Run
# =========================================================

def run_stability(X, y, seeds=STABILITY_SEEDS) -> dict:
    """
    Repeat stratified K-fold CV for each seed and collect PR-AUC scores.
    """

    scale_pos_weight = compute_scale_pos_weight(y)
    model = build_model(scale_pos_weight)

    per_seed = {}
    all_scores = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, scoring=CV_SCORING, cv=cv, n_jobs=-1)
        per_seed[str(seed)] = {
            "scores": [round(float(s), 4) for s in scores],
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }
        all_scores.extend(scores.tolist())

    summary = stability_summary(all_scores)

    return {
        "scoring": CV_SCORING,
        "cv_folds": CV_FOLDS,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "overall": summary,
        "verdict": interpret_stability(summary["coefficient_of_variation"]),
    }


# =========================================================
# Reporting
# =========================================================

def save_stability_plot(results: dict):
    """Box/scatter of per-seed PR-AUC scores around the overall mean."""
    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = results["seeds"]
    data = [results["per_seed"][str(s)]["scores"] for s in seeds]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=[str(s) for s in seeds], showmeans=True)

    overall_mean = results["overall"]["mean"]
    plt.axhline(overall_mean, linestyle="--", color="#ef4444",
                label=f"Overall mean = {overall_mean:.4f}")

    plt.title("Model Stability — PR-AUC across CV folds by seed")
    plt.xlabel("Random seed")
    plt.ylabel("PR-AUC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = MODEL_REPORT_DIR / "stability_pr_auc.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return output_path


def save_results(results: dict):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STABILITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# =========================================================
# Main
# =========================================================

def main():
    data = load_preprocessed_data()
    X_train = data["X_train_processed"]
    y_train = np.asarray(data["y_train"])

    print("=" * 70)
    print("MODEL STABILITY TESTING (XGBoost, repeated CV)")
    print("=" * 70)
    print(f"Seeds: {STABILITY_SEEDS}  |  {CV_FOLDS}-fold each  |  scoring: {CV_SCORING}")

    results = run_stability(X_train, y_train)

    save_results(results)
    save_stability_plot(results)

    o = results["overall"]
    print(f"\nOverall PR-AUC: mean={o['mean']}  std={o['std']}  "
          f"range={o['range']}  CV={o['coefficient_of_variation']}")
    print(f"Verdict: {results['verdict']}")

    print("\nArtifacts saved:")
    print(f"- {STABILITY_REPORT_PATH}")
    print(f"- {MODEL_REPORT_DIR / 'stability_pr_auc.png'}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
