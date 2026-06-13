"""
Compare gradient-boosting models for fraud detection:
XGBoost vs LightGBM vs CatBoost.

Each model is evaluated with the SAME stratified K-fold cross-validation and
the SAME imbalance handling (cost-sensitive learning via scale_pos_weight /
class weights), so the comparison is apples-to-apples. We report PR-AUC
(primary), ROC-AUC and F1, plus a held-out test-set score for the best model.

Outputs:
- artifacts/model_comparison.json          (per-model CV + test metrics)
- reports/model/model_comparison.png       (PR-AUC bar chart with error bars)
- reports/model/model_comparison_report.html
"""

import json
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    CV_SCORING,
    MODEL_COMPARISON_PATH,
    MODEL_REPORT_DIR,
)
from cv_utils import (
    load_preprocessed_data,
    compute_scale_pos_weight,
    make_cv,
    summarize_scores,
)


# =========================================================
# Model factory
# =========================================================

def build_models(scale_pos_weight: float) -> dict:
    """
    Build the three candidate models, each with imbalance handling.

    All use modest, comparable settings so the comparison reflects the
    algorithm rather than wildly different capacities.
    """

    xgb = XGBClassifier(
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

    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    # auto_class_weights="Balanced" applies the same n/(2*n_class) weighting as
    # compute_class_weights, but round-trips cleanly through sklearn's clone()
    # (a dict class_weights does not).
    catboost = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )

    return {"XGBoost": xgb, "LightGBM": lgbm, "CatBoost": catboost}


# =========================================================
# Evaluation
# =========================================================

def cross_validate_model(model, X, y) -> dict:
    """
    Cross-validate a single model on PR-AUC, ROC-AUC and F1.
    """

    cv = make_cv(n_splits=CV_FOLDS)

    pr_auc = cross_val_score(model, X, y, scoring=CV_SCORING, cv=cv, n_jobs=-1)
    roc_auc = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    f1 = cross_val_score(model, X, y, scoring="f1", cv=cv, n_jobs=-1)

    return {
        "pr_auc": summarize_scores(pr_auc),
        "roc_auc": summarize_scores(roc_auc),
        "f1": summarize_scores(f1),
    }


def evaluate_on_test(model, X_train, y_train, X_test, y_test) -> dict:
    """
    Fit on the full training split and score on the held-out test split.
    """

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_f1_at_0.5": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }


def run_comparison() -> dict:
    """
    Cross-validate and test-evaluate every candidate model.
    """

    data = load_preprocessed_data()
    X_train = data["X_train_processed"]
    y_train = np.asarray(data["y_train"])
    X_test = data["X_test_processed"]
    y_test = np.asarray(data["y_test"])

    scale_pos_weight = compute_scale_pos_weight(y_train)

    models = build_models(scale_pos_weight)

    results = {}
    for name, model in models.items():
        print(f"  Evaluating {name} ...")
        cv_metrics = cross_validate_model(model, X_train, y_train)
        test_metrics = evaluate_on_test(model, X_train, y_train, X_test, y_test)
        results[name] = {"cv": cv_metrics, "test": test_metrics}

    best_model = max(results, key=lambda n: results[n]["cv"]["pr_auc"]["mean"])

    return {
        "scoring": CV_SCORING,
        "cv_folds": CV_FOLDS,
        "best_model": best_model,
        "best_cv_pr_auc": results[best_model]["cv"]["pr_auc"]["mean"],
        "models": results,
    }


# =========================================================
# Reporting
# =========================================================

def save_comparison_plot(results: dict):
    """
    Bar chart of mean CV PR-AUC per model with std error bars.
    """

    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    names = list(results["models"].keys())
    means = [results["models"][n]["cv"]["pr_auc"]["mean"] for n in names]
    stds = [results["models"][n]["cv"]["pr_auc"]["std"] for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, means, yerr=stds, capsize=6, color="#2563eb", alpha=0.85)
    plt.title("Cross-Validated PR-AUC by Model")
    plt.ylabel("PR-AUC (mean ± std)")
    plt.ylim(0, max(means) * 1.3 if means else 1)

    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=11,
        )

    plt.tight_layout()
    output_path = MODEL_REPORT_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return output_path


def save_comparison_html(results: dict):
    """
    Simple HTML report summarising the comparison.
    """

    rows = ""
    for name, r in results["models"].items():
        highlight = ' style="background:#ecfdf5;font-weight:bold;"' if name == results["best_model"] else ""
        rows += f"""
            <tr{highlight}>
                <td>{name}</td>
                <td>{r['cv']['pr_auc']['mean']:.4f} ± {r['cv']['pr_auc']['std']:.4f}</td>
                <td>{r['cv']['roc_auc']['mean']:.4f} ± {r['cv']['roc_auc']['std']:.4f}</td>
                <td>{r['cv']['f1']['mean']:.4f} ± {r['cv']['f1']['std']:.4f}</td>
                <td>{r['test']['test_pr_auc']:.4f}</td>
                <td>{r['test']['test_roc_auc']:.4f}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Model Comparison Report</title>
<style>
body {{ font-family: Arial, sans-serif; background:#f5f6f8; color:#222; margin:0; }}
.container {{ max-width:1100px; margin:auto; padding:36px; }}
.section {{ background:#fff; padding:24px; border-radius:14px; margin-bottom:24px;
           box-shadow:0 2px 10px rgba(0,0,0,.08); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ border:1px solid #ddd; padding:10px; text-align:left; }}
th {{ background:#f3f4f6; }}
img {{ max-width:100%; }}
.good {{ background:#ecfdf5; border-left:5px solid #10b981; padding:14px; border-radius:8px; }}
</style></head>
<body><div class="container">
<h1>Model Comparison — XGBoost vs LightGBM vs CatBoost</h1>
<div class="section">
  <div class="good"><strong>Best model (CV PR-AUC):</strong> {results['best_model']}
  ({results['best_cv_pr_auc']:.4f})</div>
  <p>All models use stratified {results['cv_folds']}-fold cross-validation and
  cost-sensitive imbalance handling. PR-AUC is the primary metric for this
  imbalanced (~6% fraud) problem.</p>
</div>
<div class="section">
  <h2>Metrics</h2>
  <table>
    <tr><th>Model</th><th>CV PR-AUC</th><th>CV ROC-AUC</th><th>CV F1</th>
        <th>Test PR-AUC</th><th>Test ROC-AUC</th></tr>
    {rows}
  </table>
</div>
<div class="section">
  <h2>Cross-Validated PR-AUC</h2>
  <img src="model_comparison.png" />
</div>
</div></body></html>"""

    output_path = MODEL_REPORT_DIR / "model_comparison_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def save_results(results: dict):
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 70)
    print("MODEL COMPARISON (XGBoost vs LightGBM vs CatBoost)")
    print("=" * 70)

    results = run_comparison()

    save_results(results)
    save_comparison_plot(results)
    save_comparison_html(results)

    print("\nCross-validated PR-AUC (mean ± std):")
    for name, r in results["models"].items():
        marker = "  <-- best" if name == results["best_model"] else ""
        print(f"  {name:10s}: {r['cv']['pr_auc']['mean']:.4f} "
              f"± {r['cv']['pr_auc']['std']:.4f}{marker}")

    print("\nArtifacts saved:")
    print(f"- {MODEL_COMPARISON_PATH}")
    print(f"- {MODEL_REPORT_DIR / 'model_comparison.png'}")
    print(f"- {MODEL_REPORT_DIR / 'model_comparison_report.html'}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
