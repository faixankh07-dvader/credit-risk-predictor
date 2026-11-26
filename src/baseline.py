"""Phase 1: baseline models, evaluated on the training data only.

    python src/baseline.py

Everything uses 5-fold stratified cross-validation on the TRAIN split.
The test set is not touched here (it is used once, at the very end).

Outputs:
  results/baseline_cv.csv               majority baseline vs logistic regression
  results/threshold_table_logistic.csv  precision, recall and cost across thresholds
  images/threshold_logistic.png         the same, as plots
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_data, model_features, split_data  # noqa: E402
from src.metrics import evaluate, threshold_table  # noqa: E402
from src.preprocess import build_preprocessor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RATIOS = (1, 2, 5, 10)


def cv_proba(clf, cols, X, y, cv):
    pipe = make_pipeline(build_preprocessor(cols), clf)
    return cross_val_predict(pipe, X[cols], y, cv=cv, method="predict_proba")[:, 1]


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "images").mkdir(exist_ok=True)

    df = load_data()
    X_train, _X_test, y_train, _y_test = split_data(df)
    cols = model_features(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Logistic (class_weight=balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
    probas, rows = {}, []
    for name, clf in models.items():
        probas[name] = cv_proba(clf, cols, X_train, y_train, cv)
        rows.append({"model": name, **evaluate(y_train, probas[name], threshold=0.5)})
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results" / "baseline_cv.csv", index=False)
    print(summary[["model", "roc_auc", "pr_auc", "precision", "recall", "f1",
                   "tn", "fp", "fn", "tp"]].to_string(index=False))

    # Threshold analysis for plain logistic regression
    thresholds = np.round(np.arange(0.05, 0.65, 0.05), 2)
    table = threshold_table(y_train, probas["Logistic regression"], thresholds, RATIOS)
    table.to_csv(ROOT / "results" / "threshold_table_logistic.csv", index=False)
    print("\n", table.to_string(index=False))
    print("\nBest threshold on these CV predictions, per cost ratio "
          "(chosen on the same data it is measured on, so slightly optimistic):")
    for r in RATIOS:
        best = table.loc[table[f"cost_r{r}"].idxmin()]
        print(f"  r={r:<3} threshold={best['threshold']:.2f}  cost={int(best[f'cost_r{r}'])}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for col in ("precision", "recall", "f1"):
        ax[0].plot(table["threshold"], table[col], marker="o", label=col)
    ax[0].set_xlabel("Threshold")
    ax[0].set_title("Precision / recall trade-off")
    ax[0].legend()
    ref = table.loc[(table["threshold"] - 0.5).abs().idxmin()]
    for r in RATIOS:
        ax[1].plot(table["threshold"], table[f"cost_r{r}"] / ref[f"cost_r{r}"],
                   marker="o", label=f"r={r}")
    ax[1].axhline(1, color="grey", lw=0.8)
    ax[1].set_xlabel("Threshold")
    ax[1].set_ylabel("Cost relative to threshold 0.5")
    ax[1].set_title("Best threshold depends on the cost ratio")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(ROOT / "images" / "threshold_logistic.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
