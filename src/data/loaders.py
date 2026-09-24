"""
Dataset loaders for financial classification benchmarks.

Each loader returns a standardized dict with the following keys:
    - X: feature matrix (np.ndarray)
    - y: labels (np.ndarray, integer-encoded)
    - sensitive: sensitive attribute values (np.ndarray)
    - feature_names: list of feature name strings
    - sensitive_name: name of the sensitive attribute
    - label_names: list of class names
    - description: short dataset description

We use publicly available datasets that are standard in the
fairness literature. No private or proprietary data.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any


# Where to cache downloaded data
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def load_german_credit(sensitive="age") -> Dict[str, Any]:
    """
    Load the Statlog German Credit dataset from UCI.

    Classic credit risk dataset with 1000 samples. Each person is
    classified as good or bad credit risk based on 20 attributes.
    Small but widely used in fairness research.

    Parameters
    ----------
    sensitive : str
        Which attribute to treat as sensitive.
        'age' -> binary: age >= 25 vs age < 25
        'gender' -> extracted from personal status field

    Returns
    -------
    dict with standardized keys (see module docstring)
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=144)
        X_df = dataset.data.features.copy()
        y_series = dataset.data.targets.iloc[:, 0].copy()
    except Exception:
        # Fallback: generate a synthetic version for testing
        warnings.warn(
            "Could not fetch German Credit from UCI. "
            "Using sklearn's make_classification as a stand-in. "
            "Install ucimlrepo and check your internet connection.",
            UserWarning,
        )
        return _synthetic_fallback("german_credit", sensitive)

    # The target: 1 = good, 2 = bad credit in the original encoding
    # We remap to 0 = bad, 1 = good (standard convention)
    y = np.where(y_series.values.ravel() == 1, 1, 0)

    # Extract sensitive attribute
    if sensitive == "age":
        # Age is typically column 'Attribute13' in the UCI version
        # Binary: young (< 25) vs old (>= 25) following Kamiran & Calders
        age_col = _find_column(X_df, ["age", "Attribute13", "Age"])
        if age_col is not None:
            age_vals = pd.to_numeric(X_df[age_col], errors="coerce").fillna(30)
            sens = np.where(age_vals >= 25, "age>=25", "age<25")
        else:
            sens = np.random.choice(["age>=25", "age<25"], size=len(y), p=[0.7, 0.3])
    elif sensitive == "gender":
        gender_col = _find_column(X_df, ["personal_status", "Attribute9"])
        if gender_col is not None:
            # Personal status encodes gender + marital status
            # UCI codes: A91=male, A92=female, A93=male, A94=male, A95=female
            vals = X_df[gender_col].astype(str).str.strip()
            is_female_code = vals.isin(["A92", "A95"])
            is_female_text = vals.str.contains("female", case=False, na=False)
            sens = np.where(is_female_code | is_female_text, "female", "male")
        else:
            sens = np.random.choice(["male", "female"], size=len(y), p=[0.69, 0.31])
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive}")

    # Encode categorical features as numeric
    X_encoded = _encode_features(X_df)

    return {
        "X": X_encoded,
        "y": y,
        "sensitive": sens,
        "feature_names": list(X_df.columns),
        "sensitive_name": sensitive,
        "label_names": ["bad_credit", "good_credit"],
        "description": "German Credit — credit risk classification (UCI)",
    }


def load_taiwan_credit(sensitive="gender") -> Dict[str, Any]:
    """
    Load the Taiwan Credit Card Default dataset from UCI.

    30,000 samples of credit card clients in Taiwan. Target is whether
    the client defaults on payment next month. Larger than German Credit
    and commonly used alongside it.

    Parameters
    ----------
    sensitive : str
        'gender' -> SEX column (1=male, 2=female)
        'education' -> EDUCATION column (bucketed)
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=350)
        X_df = dataset.data.features.copy()
        y_series = dataset.data.targets.iloc[:, 0].copy()
    except Exception:
        warnings.warn(
            "Could not fetch Taiwan Credit from UCI. Using fallback.",
            UserWarning,
        )
        return _synthetic_fallback("taiwan_credit", sensitive)

    y = y_series.values.ravel().astype(int)

    if sensitive == "gender":
        sex_col = _find_column(X_df, ["SEX", "sex", "X2"])
        if sex_col is not None:
            sex_vals = pd.to_numeric(X_df[sex_col], errors="coerce").fillna(1)
            sens = np.where(sex_vals == 1, "male", "female")
        else:
            sens = np.random.choice(["male", "female"], size=len(y))
    elif sensitive == "education":
        edu_col = _find_column(X_df, ["EDUCATION", "education", "X3"])
        if edu_col is not None:
            edu_vals = pd.to_numeric(X_df[edu_col], errors="coerce").fillna(4)
            # Bucket: 1=grad school, 2=university, 3=high school, 4+=other
            sens = np.where(
                edu_vals <= 2, "higher_ed", "other_ed"
            )
        else:
            sens = np.random.choice(["higher_ed", "other_ed"], size=len(y))
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive}")

    X_encoded = _encode_features(X_df)

    return {
        "X": X_encoded,
        "y": y,
        "sensitive": sens,
        "feature_names": list(X_df.columns),
        "sensitive_name": sensitive,
        "label_names": ["no_default", "default"],
        "description": "Taiwan Credit — credit card default prediction (UCI)",
    }


def load_adult_income(sensitive="gender") -> Dict[str, Any]:
    """
    Load the Adult Income (Census) dataset.

    Classic fairness benchmark: predict whether income exceeds 50K/year.
    ~49K samples with well-known demographic disparities.

    Parameters
    ----------
    sensitive : str
        'gender' -> sex column
        'race' -> race column (white vs non-white)
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=2)
        X_df = dataset.data.features.copy()
        y_series = dataset.data.targets.iloc[:, 0].copy()
    except Exception:
        warnings.warn(
            "Could not fetch Adult Income from UCI. Using fallback.",
            UserWarning,
        )
        return _synthetic_fallback("adult_income", sensitive)

    # Binary label: <=50K -> 0, >50K -> 1
    y_str = y_series.astype(str).str.strip()
    y = np.where(y_str.str.contains(">50K"), 1, 0)

    if sensitive == "gender":
        sex_col = _find_column(X_df, ["sex", "Sex", "gender"])
        if sex_col is not None:
            sens = np.where(
                X_df[sex_col].astype(str).str.strip().str.lower() == "male",
                "male", "female"
            )
        else:
            sens = np.random.choice(["male", "female"], size=len(y))
    elif sensitive == "race":
        race_col = _find_column(X_df, ["race", "Race"])
        if race_col is not None:
            sens = np.where(
                X_df[race_col].astype(str).str.strip().str.lower() == "white",
                "white", "non-white"
            )
        else:
            sens = np.random.choice(["white", "non-white"], size=len(y))
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive}")

    X_encoded = _encode_features(X_df)

    return {
        "X": X_encoded,
        "y": y,
        "sensitive": sens,
        "feature_names": list(X_df.columns),
        "sensitive_name": sensitive,
        "label_names": ["<=50K", ">50K"],
        "description": "Adult Income — income prediction (UCI Census)",
    }


# ── Helper functions ──────────────────────────────────────────────


def _find_column(df, candidates):
    """Try to find a column by checking a list of possible names."""
    for name in candidates:
        if name in df.columns:
            return name
    # Fuzzy match: check if any candidate is a substring
    for name in candidates:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


def _encode_features(df):
    """
    Encode a mixed-type DataFrame into a numeric numpy array.

    For categorical columns we use label encoding (ordinal). This is
    fine for tree-based models which handle ordinal features natively.
    For a more principled approach one could use target encoding,
    but label encoding keeps things simple and reproducible.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object or df_copy[col].dtype.name == "category":
            le = LabelEncoder()
            # Handle NaN by converting to string first
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        else:
            # Fill numeric NaN with median
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())

    return df_copy.values.astype(np.float64)


def _synthetic_fallback(dataset_name, sensitive):
    """
    Generate synthetic data when real datasets can't be downloaded.
    Only for development/testing — real experiments use actual data.
    """
    from sklearn.datasets import make_classification

    rng = np.random.RandomState(42)

    sizes = {"german_credit": 1000, "taiwan_credit": 5000, "adult_income": 10000}
    n = sizes.get(dataset_name, 2000)

    X, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    # Fabricate a sensitive attribute that's correlated with y
    # (this mimics real-world disparities)
    noise = rng.normal(0, 0.3, size=n)
    sens_score = 0.3 * y + noise
    sens = np.where(sens_score > 0, "group_A", "group_B")

    return {
        "X": X,
        "y": y,
        "sensitive": sens,
        "feature_names": [f"feat_{i}" for i in range(20)],
        "sensitive_name": sensitive,
        "label_names": ["class_0", "class_1"],
        "description": f"Synthetic fallback for {dataset_name}",
    }


# ── Registry for easy access ────────────────────────────────────

DATASET_LOADERS = {
    "german_credit": load_german_credit,
    "taiwan_credit": load_taiwan_credit,
    "adult_income": load_adult_income,
}


def load_dataset(name, **kwargs) -> Dict[str, Any]:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        One of: 'german_credit', 'taiwan_credit', 'adult_income'
    **kwargs
        Passed to the specific loader (e.g. sensitive='gender')
    """
    if name not in DATASET_LOADERS:
        available = ", ".join(DATASET_LOADERS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    return DATASET_LOADERS[name](**kwargs)
