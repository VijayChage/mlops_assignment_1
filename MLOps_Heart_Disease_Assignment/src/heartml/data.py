from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.heartml.schema import ALIASES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def normalize_category(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return format(number, ".12g")
    except ValueError:
        return text


def normalize_column_names(columns):
    normalized = [str(c).strip().lower().replace(" ", "_") for c in columns]
    return [ALIASES.get(c, c) for c in normalized]


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = normalize_column_names(df.columns)
    if "target" not in df.columns and "num" in df.columns:
        df = df.rename(columns={"num": "target"})
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}. Found: {list(df.columns)}")
    df = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Keep categorical codes/text as categories; do not coerce valid strings to missing values.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].map(normalize_category)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = (df[TARGET] > 0).astype(int)
    if df.empty:
        raise ValueError("No usable rows remain after target cleaning.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both target classes after binarization.")
    return df


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def validate_features(payload: dict) -> dict:
    missing = set(FEATURES) - set(payload)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    result = {}
    for name in FEATURES:
        value = payload[name]
        if name in NUMERIC_FEATURES:
            try:
                result[name] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Feature '{name}' must be numeric") from None
        else:
            if value is None:
                result[name] = np.nan
            else:
                result[name] = normalize_category(value)
    return result
