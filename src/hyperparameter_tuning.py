"""
Hyperparameter tuning for the XGBoost fraud model.

Uses RandomizedSearchCV with stratified K-fold cross-validation, optimising
PR-AUC (average precision) — the right metric for a ~6% fraud rate. Class
imbalance is handled with scale_pos_weight (cost-sensitive learning), so the
search focuses purely on the tree/regularisation hyperparameters.

Outputs:
- artifacts/best_params.json                  (best hyperparameters + CV score)
- artifacts/hyperparameter_search_results.json (full ranked search table)

The selected parameters can then be fed into train.py to retrain the final model.
"""

import json
from typing import Any, Dict

import numpy as np
from scipy.stats import randint, uniform

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    CV_SCORING,
    BEST_PARAMS_PATH,
    HYPERPARAMETER_SEARCH_PATH,
    ARTIFACTS_DIR,
)
from cv_utils import (
    load_preprocessed_data,
    compute_scale_pos_weight,
    make_cv,
)


# =========================================================
# Search space
# =========================================================

def build_search_space() -> Dict[str, Any]:
    """
    Randomised search space over XGBoost hyperparameters.

    Distributions (not fixed grids) let RandomizedSearchCV sample a wider,
    cheaper-to-explore space than an exhaustive GridSearchCV would.
    """

    return {
        "n_estimators": randint(200, 600),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.19),      # 0.01 - 0.20
        "subsample": uniform(0.6, 0.4),            # 0.6 - 1.0
        "colsample_bytree": uniform(0.6, 0.4),     # 0.6 - 1.0
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0.0, 0.5),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.5, 2.0),
    }


def build_base_estimator(scale_pos_weight: float) -> XGBClassifier:
    """
    Base XGBoost estimator with fixed, non-searched settings.
    """

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =========================================================
# Search
# =========================================================

def run_search(
    X_train,
    y_train,
    n_iter: int = 40,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """
    Run RandomizedSearchCV with stratified K-fold CV.
    """

    scale_pos_weight = compute_scale_pos_weight(y_train)

    search = RandomizedSearchCV(
        estimator=build_base_estimator(scale_pos_weight),
        param_distributions=build_search_space(),
        n_iter=n_iter,
        scoring=CV_SCORING,
        cv=make_cv(n_splits=CV_FOLDS),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    return search


# =========================================================
# Result extraction / persistence
# =========================================================

def extract_top_results(search: RandomizedSearchCV, top_n: int = 10) -> list:
    """
    Build a ranked, JSON-serialisable table of the top candidates.
    """

    results = search.cv_results_

    rows = []
    for rank, (mean, std, params) in enumerate(
        zip(
            results["mean_test_score"],
            results["std_test_score"],
            results["params"],
        )
    ):
        rows.append({
            "mean_cv_score": round(float(mean), 4),
            "std_cv_score": round(float(std), 4),
            "params": {k: _to_native(v) for k, v in params.items()},
        })

    rows.sort(key=lambda r: r["mean_cv_score"], reverse=True)
    return rows[:top_n]


def _to_native(value):
    """Convert numpy scalars to plain python types for JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    return value


def save_results(search: RandomizedSearchCV, scoring: str = CV_SCORING) -> dict:
    """
    Persist the best parameters and the ranked search table.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    best = {
        "scoring": scoring,
        "best_cv_score": round(float(search.best_score_), 4),
        "best_params": {k: _to_native(v) for k, v in search.best_params_.items()},
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=4)

    top_results = extract_top_results(search, top_n=10)
    with open(HYPERPARAMETER_SEARCH_PATH, "w", encoding="utf-8") as f:
        json.dump(top_results, f, indent=4)

    return best


# =========================================================
# Main
# =========================================================

def main(n_iter: int = 40):
    data = load_preprocessed_data()

    X_train = data["X_train_processed"]
    y_train = data["y_train"]

    print("=" * 70)
    print("HYPERPARAMETER TUNING (XGBoost, RandomizedSearchCV)")
    print("=" * 70)
    print(f"Scoring          : {CV_SCORING} (PR-AUC)")
    print(f"CV folds         : {CV_FOLDS}")
    print(f"Search iterations: {n_iter}")
    print(f"Train shape      : {np.asarray(X_train).shape}")

    search = run_search(X_train, y_train, n_iter=n_iter)

    best = save_results(search)

    print("\nBest CV score (PR-AUC):", best["best_cv_score"])
    print("\nBest params:")
    for k, v in best["best_params"].items():
        print(f"  {k}: {v}")

    print("\nArtifacts saved:")
    print(f"- {BEST_PARAMS_PATH}")
    print(f"- {HYPERPARAMETER_SEARCH_PATH}")
    print("=" * 70)

    return best


if __name__ == "__main__":
    main()
