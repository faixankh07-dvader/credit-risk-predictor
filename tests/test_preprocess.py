import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import evaluate  # noqa: E402
from src.preprocess import build_preprocessor, clean_codes  # noqa: E402


def frame():
    return pd.DataFrame({
        "LIMIT_BAL": [20000, 120000, 90000, 50000],
        "EDUCATION": [0, 1, 5, 6],
        "MARRIAGE": [0, 1, 2, 3],
        "PAY_0": [2, -1, 0, -2],
    })


def test_undocumented_codes_are_merged():
    out = clean_codes(frame())
    assert list(out["EDUCATION"]) == [4, 1, 4, 4]
    assert list(out["MARRIAGE"]) == [3, 1, 2, 3]


def test_clean_codes_does_not_modify_input():
    df = frame()
    clean_codes(df)
    assert df["EDUCATION"].tolist() == [0, 1, 5, 6]


def test_pipeline_handles_unseen_category():
    df = frame()
    cols = ["LIMIT_BAL", "EDUCATION", "PAY_0"]
    pipe = make_pipeline(build_preprocessor(cols), LogisticRegression())
    pipe.fit(df[cols], [0, 1, 0, 1])
    new = pd.DataFrame({"LIMIT_BAL": [1000], "EDUCATION": [9], "PAY_0": [1]})
    assert pipe.predict_proba(new).shape == (1, 2)


def test_evaluate_reports_counts_and_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = evaluate(y, p, threshold=0.5)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 0, 0, 2)


def test_majority_baseline_has_zero_recall_but_high_accuracy():
    y = np.array([0] * 78 + [1] * 22)
    m = evaluate(y, np.zeros(100), threshold=0.5)
    assert m["recall"] == 0.0
    assert (m["tn"] + m["tp"]) / 100 == 0.78


def test_threshold_table_costs():
    from src.metrics import threshold_table
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.5, 0.9])
    t = threshold_table(y, p, [0.5], ratios=(1, 5))
    row = t.iloc[0]
    # at 0.5: preds [0,0,1,1,1] -> fp=1, fn=0
    assert (row["fp"], row["fn"]) == (1, 0)
    assert row["cost_r1"] == 1 and row["cost_r5"] == 1
    t2 = threshold_table(y, p, [0.95], ratios=(1, 5))
    assert t2.iloc[0]["fn"] == 2 and t2.iloc[0]["cost_r5"] == 10
