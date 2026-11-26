"""Download the UCI "Default of Credit Card Clients" dataset and save a clean CSV.

Source:  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Licence: CC BY 4.0 (credit the authors)
Cite as: Yeh, I. (2009). Default of Credit Card Clients [Dataset].
         UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

Usage
-----
    python data/download.py                      # fetch with the ucimlrepo package
    python data/download.py --xls path/to/file   # fallback: the .xls downloaded by hand

Output: data/raw/credit_default.csv  (30,000 rows, 23 features + target "default").
Raw data is not committed to git (see .gitignore).
"""
import argparse
from pathlib import Path

import pandas as pd

OUT_PATH = Path(__file__).resolve().parent / "raw" / "credit_default.csv"
DATASET_ID = 350

# UCI names the features X1..X23 and the target Y. We use the descriptive names.
FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
TARGET = "default"
RENAME = {f"X{i + 1}": name for i, name in enumerate(FEATURES)}
RENAME["Y"] = TARGET


def from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=DATASET_ID)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return df.rename(columns=RENAME)


def from_xls(path: str) -> pd.DataFrame:
    # The UCI .xls has a row of X1..X23 labels first, then the real header row.
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"default payment next month": TARGET,
                            "default.payment.next.month": TARGET})
    return df.drop(columns=["ID"], errors="ignore")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    expected = FEATURES + [TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Columns found: {list(df.columns)}")
    df = df[expected].copy()
    if len(df) != 30000:
        print(f"WARNING: expected 30,000 rows, found {len(df)}")
    if df.isna().any().any():
        print("WARNING: dataset contains missing values")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError(f"Target should be 0/1, found {df[TARGET].unique()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xls", help="path to a manually downloaded .xls file")
    args = parser.parse_args()

    df = from_xls(args.xls) if args.xls else from_ucimlrepo()
    df = validate(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}  shape={df.shape}")
    print(f"Default rate: {df[TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
