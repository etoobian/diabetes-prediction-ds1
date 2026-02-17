"""
Modeling utilities for the Diabetes Prediction project.

This module contains reusable routines to:
- Fit supervised learning models (via sklearn Pipelines when appropriate)
- Return fitted models and standardized result artifacts (e.g., importance tables)

Notebook sections should call functions here to keep analysis cells clean and reproducible.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def fit_rf_feature_screening(
    df: pd.DataFrame,
    *,
    target_col: str,
    categorical_cols: List[str],
    numeric_cols: List[str],
    random_state: int = 587,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float | int | None = "sqrt",
    n_jobs: int = -1,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Fit a RandomForestClassifier on the full dataset for feature screening.

    Notes:
    - This is intended for ranking feature importances (not unbiased performance).
    - Categorical features are one-hot encoded within the pipeline.

    Returns
    -------
    pipe:
        Fitted sklearn Pipeline (preprocessor + RandomForestClassifier).
    importance_df:
        DataFrame with columns ['feature', 'importance'], sorted descending.
    """
    X = df[categorical_cols + numeric_cols].copy()
    y = df[target_col].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("rf", rf)])
    pipe.fit(X, y)

    # Feature names after preprocessing
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    feature_names = cat_feature_names + numeric_cols

    importances = pipe.named_steps["rf"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return pipe, importance_df