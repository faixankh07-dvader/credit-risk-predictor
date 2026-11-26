import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import SENSITIVE, TARGET, model_features, split_data  # noqa: E402


def fake_df(n=1000, rate=0.22, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "LIMIT_BAL": rng.integers(10000, 500000, n),
        "SEX": rng.integers(1, 3, n),
        "AGE": rng.integers(21, 70, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "PAY_0": rng.integers(-2, 9, n),
    })
    df[TARGET] = (rng.random(n) < rate).astype(int)
    return df


def test_split_is_stratified_and_disjoint():
    df = fake_df()
    X_tr, X_te, y_tr, y_te = split_data(df)
    assert len(X_te) == 200 and len(X_tr) == 800
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_split_is_reproducible():
    df = fake_df()
    a = split_data(df)[1].index
    b = split_data(df)[1].index
    assert list(a) == list(b)


def test_sensitive_attributes_are_not_model_inputs():
    cols = model_features(fake_df())
    assert TARGET not in cols
    assert not set(SENSITIVE) & set(cols)
    assert "LIMIT_BAL" in cols and "PAY_0" in cols
