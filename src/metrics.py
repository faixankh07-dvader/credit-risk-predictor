"""Imbalance-aware evaluation (spec section 7). Accuracy is NOT the main metric."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Ranking metrics from probabilities, plus error counts at one threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
    }


def threshold_table(y_true, proba, thresholds, ratios=(1, 2, 5, 10)):
    """Metrics and expected cost at several thresholds.

    cost_r{r} = r * FN + FP, i.e. approving a defaulter costs r times as much as
    declining a customer who would have repaid. The ratio is an assumption.
    """
    import pandas as pd

    rows = []
    for t in thresholds:
        m = evaluate(y_true, proba, threshold=float(t))
        row = {k: m[k] for k in ("threshold", "precision", "recall", "f1", "fp", "fn")}
        for r in ratios:
            row[f"cost_r{r}"] = r * m["fn"] + m["fp"]
        rows.append(row)
    return pd.DataFrame(rows)
