"""
Preprocessing and validation utilities for the diabetes prediction project.

Responsibilities:
- Create the base train/test split:
    * Drop leakage variables (LEAKAGE_COLS)
    * Perform stratified split on TARGET_COL
- Standardize data types (numeric, categorical, binary)
- Validate binary domains
- Validate categorical levels against either:
    - REF_* (documentation/Kaggle reference; audit/reporting), or
    - operational schema (what we use in this project)
- Validate numeric ranges against either:
    - REF_* (documentation/Kaggle reference; audit/reporting), or
    - operational sanity bounds (guardrails)

Model-specific transformations (one-hot encoding, scaling, etc.) should be
implemented in sklearn pipelines during model training.
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split
from .schema import TARGET_COL

from .schema import (
    LEAKAGE_COLS,
    CATEGORICAL_COLS,
    BINARY_COLS,
    NUMERIC_COLS,
    REF_ALLOWED_CATEGORIES,
    ALLOWED_CATEGORIES,
    REF_RANGE_RULES,
    RANGE_RULES,
)


def make_base_split(
    df_raw: pd.DataFrame,
    *,
    seed: int,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the base train/test split for the project.
    - Drops leakage columns (LEAKAGE_COLS)
    - Stratified train/test split on TARGET_COL

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe loaded from CSV.
    seed : int
        Random seed for reproducibility.
    test_size : float
        Fraction of data assigned to the test set.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
        Train and test dataframes after leakage-drop, stratified split.
    """
    df = df_raw.drop(columns=[c for c in LEAKAGE_COLS if c in df_raw.columns]).copy()

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe.")

    y = df[TARGET_COL]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Return a copy of df with the specified columns dropped (ignores missing cols).
    """
    out = df.copy()
    to_drop = [c for c in cols if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)
    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with standardized dtypes for downstream use."""
    out = df.copy()

    # Categoricals
    for c in CATEGORICAL_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category")

    # Binary (keep as 0/1 integers)
    for c in BINARY_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="raise").astype("int64")

    # Numeric
    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="raise")

    return out


def check_binary_domains(df: pd.DataFrame) -> dict:
    """Return {col: bad_values} for any binary column containing values not in {0,1}."""
    violations = {}
    for c in BINARY_COLS:
        if c in df.columns:
            vals = set(pd.unique(df[c]))
            bad = sorted([v for v in vals if v not in (0, 1)])
            if bad:
                violations[c] = bad
    return violations


def check_allowed_categories(df: pd.DataFrame, *, reference: bool = False) -> dict:
    """
    Return {col: unexpected_values} for categorical columns with values outside allowed lists.

    If reference=True, uses REF_ALLOWED_CATEGORIES (documentation audit).
    Otherwise, uses ALLOWED_CATEGORIES (operational project schema).
    """
    allowed_map = REF_ALLOWED_CATEGORIES if reference else ALLOWED_CATEGORIES
    unexpected = {}

    for c, allowed in allowed_map.items():
        if c in df.columns:
            vals = set(pd.unique(df[c].dropna()))
            bad = sorted([v for v in vals if v not in set(allowed)])
            if bad:
                unexpected[c] = bad

    return unexpected


def check_numeric_ranges(df: pd.DataFrame, *, reference: bool = False) -> pd.DataFrame:
    """
    Return a diagnostic table for numeric range checks.

    If reference=True:  Checks against REF_RANGE_RULES (documentation audit).
    Else:               Checks against RANGE_RULES (operational sanity bounds).

    Returns a DataFrame with:
        feature, min_allowed, max_allowed, data_min, 
        data_max, below_min, above_max, pct_outside
    """
    rules = REF_RANGE_RULES if reference else RANGE_RULES
    rows = []
    n = len(df)

    for col, (lo, hi) in rules.items():
        if col in df.columns:
            x = df[col].dropna()

            if x.empty:
                continue

            below_mask = x < lo
            above_mask = x > hi

            below = int(below_mask.sum())
            above = int(above_mask.sum())

            if below + above > 0:
                rows.append(
                    {
                        "feature": col,
                        "min_allowed": float(lo),
                        "max_allowed": float(hi),
                        "data_min": float(x.min()),
                        "data_max": float(x.max()),
                        "below_min": below,
                        "above_max": above,
                        "pct_outside": (below + above) / n,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["feature", "min_allowed", "max_allowed", "data_min", 
                     "data_max", "below_min", "above_max", "pct_outside",]
                     )

    return pd.DataFrame(rows).sort_values(
        by=["pct_outside", "below_min", "above_max"], ascending=False
        ).reset_index(drop=True)


def print_categorical_audit(df: pd.DataFrame) -> None:
    """
    Print only categorical columns with differences vs Kaggle reference levels:
    missing expected levels and/or extra unexpected levels.
    """
    any_diff = False

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        observed = set(df[col].dropna().unique())
        expected = REF_ALLOWED_CATEGORIES.get(col)

        if expected is None:
            continue

        expected_set = set(expected)
        missing = sorted(list(expected_set - observed))
        extra = sorted(list(observed - expected_set))

        if missing or extra:
            any_diff = True
            print("\n" + "-" * 60)
            print(f"{col}")
            print("-" * 60)
            print("Observed:", sorted(observed))
            print("Expected (REF):", expected)
            print("Missing expected:", missing if missing else "None")
            print("Extra/unexpected:", extra if extra else "None")

    if not any_diff:
        print("All categorical columns match REF levels (dataset card).")