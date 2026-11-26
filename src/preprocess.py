"""Preprocessing shared by every model (spec FR-3).

Everything here lives inside an sklearn Pipeline so each cross-validation fold
fits it on its own training part only (no leakage), and so training and serving
apply exactly the same steps.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = ["EDUCATION"]

# Codes not in the UCI documentation (found in EDA): EDUCATION 0, 5, 6 and MARRIAGE 0.
# Documented value 4 / 3 means "others", so undocumented codes are merged into it.
EDUCATION_MAP = {0: 4, 5: 4, 6: 4}
MARRIAGE_MAP = {0: 3}


def clean_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic recoding. It learns nothing from data, so it cannot leak."""
    df = df.copy()
    if "EDUCATION" in df:
        df["EDUCATION"] = df["EDUCATION"].replace(EDUCATION_MAP)
    if "MARRIAGE" in df:
        df["MARRIAGE"] = df["MARRIAGE"].replace(MARRIAGE_MAP)
    return df


def build_preprocessor(columns, scale: bool = True) -> Pipeline:
    """Clean codes, one-hot encode EDUCATION, scale the numeric columns.

    PAY_0 ... PAY_6 stay numeric for now. Whether to treat them as categories
    is a planned experiment (see the notebooks), not an assumption.
    """
    cat = [c for c in CATEGORICAL if c in columns]
    num = [c for c in columns if c not in cat]
    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler() if scale else "passthrough", num),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(clean_codes, validate=False)),
        ("prep", transformer),
    ])
