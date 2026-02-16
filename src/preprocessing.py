"""
Preprocessing and validation utilities for the diabetes prediction project.

Responsibilities:
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

from .schema import (
    DROP_COLS,
    CATEGORICAL_COLS,
    BINARY_COLS,
    NUMERIC_COLS,
    REF_ALLOWED_CATEGORIES,
    ALLOWED_CATEGORIES,
    REF_RANGE_RULES,
    RANGE_RULES,
)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with standardized dtypes for downstream use."""
    out = df.copy()

    # Drop any known non-feature columns
    for c in DROP_COLS:
        if c in out.columns:
            out = out.drop(columns=[c])

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
    Return a table of out-of-range counts for numeric features.

    If reference=True, checks against REF_RANGE_RULES (documentation audit).
    Otherwise, checks against RANGE_RULES (operational sanity bounds).
    """
    rules = REF_RANGE_RULES if reference else RANGE_RULES
    rows = []

    for col, (lo, hi) in rules.items():
        if col in df.columns:
            x = df[col]
            below = int((x < lo).sum())
            above = int((x > hi).sum())
            if below + above > 0:
                rows.append(
                    {
                        "feature": col,
                        "min_allowed": float(lo),
                        "max_allowed": float(hi),
                        "below_min": below,
                        "above_max": above,
                    }
                )
    
    if not rows:
        return pd.DataFrame(
            columns=["feature", "min_allowed", "max_allowed", "below_min", "above_max"]
        )

    return pd.DataFrame(rows).sort_values(
        by=["below_min", "above_max"], ascending=False
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