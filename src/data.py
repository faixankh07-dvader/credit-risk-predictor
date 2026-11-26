"""Loading and splitting the credit default data."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_default.csv"
TARGET = "default"
RANDOM_STATE = 42

# Decision (spec section 12): these are NOT model inputs. They are kept in the data
# only so the fairness audit can compare error rates across groups.
SENSITIVE = ["SEX", "AGE", "MARRIAGE"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python data/download.py")
    return pd.read_csv(path)


def model_features(df: pd.DataFrame) -> list:
    """Columns a model may use: everything except the target and sensitive attributes."""
    return [c for c in df.columns if c != TARGET and c not in SENSITIVE]


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Stratified train/test split. The test set is touched once, at the very end.

    Returns X_train, X_test, y_train, y_test with ALL feature columns kept
    (including sensitive ones); select model_features(df) before training.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
