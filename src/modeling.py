"""
Modeling utilities for the Diabetes Prediction project.

This module centralizes all supervised learning routines used in the project,
including model fitting, probability prediction, and likelihood-based model
comparison.

Current scope
-------------
Implemented:
- Random Forest (feature screening only; MDI importance ranking)
- Logistic Regression via MLE (statsmodels Logit)
  - Nested model comparison using Likelihood Ratio Test (LRT)

Planned:
- Random Forest (final predictive classifier)
- Gradient Boosting (e.g., XGBoost / HistGradientBoosting)
- Multi-layer Perceptron (MLP)

Design principles
-----------------
1. Reproducibility:
   All models are fit using explicit preprocessing pipelines or stored
   preprocessing objects to ensure consistent design matrices.

2. Valid statistical inference:
   Logistic regression inference uses maximum likelihood estimation (MLE).
   For nested model comparisons, reduced models are constructed by:
       (a) reusing the SAME fitted preprocessing object from the full model, and
       (b) masking encoded columns to preserve strict nesting.

3. Separation of concerns:
   - Model fitting lives here.
   - Evaluation metrics live in `metrics.py`.
   - Visualization lives in `viz.py`.

Notes
-----
- Random Forest feature screening is used strictly as a heuristic ranking
  mechanism (Mean Decrease in Impurity) and NOT as an unbiased estimator
  of predictive performance.
- Likelihood-based comparisons (LRT/"ANOVA-style") apply only to MLE
  logistic regression models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm
from scipy.stats import chi2


# =============================================================================
# 0) Shared preprocessing helpers (private)
# =============================================================================

def _build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that one-hot encodes categoricals and passes numeric through.

    Uses sparse_output=False when available (sklearn>=1.2), otherwise falls back to sparse=False.
    """
    # sklearn compatibility: sparse_output replaced sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )


def _get_feature_names(pre: ColumnTransformer, categorical_cols: list[str], numeric_cols: list[str]) -> list[str]:
    """
    Return the expanded feature names from a fitted preprocessor.

    This assumes the categorical transformer is named "cat".
    """
    ohe = pre.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    return cat_feature_names + list(numeric_cols)


def _apply_mask(X: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Apply an encoded-column mask if provided."""
    if mask is None:
        return X
    return X[:, mask]


def _encoded_mask_for_original_predictors(
    encoded_feature_names: Sequence[str],
    *,
    keep_predictors: Sequence[str],
    categorical_cols: Sequence[str],
) -> np.ndarray:
    """
    Build a boolean mask selecting encoded columns that correspond to keep_predictors.

    Parameters
    ----------
    encoded_feature_names:
        Feature names AFTER preprocessing (e.g., 'gender_Male', 'age', ...).
    keep_predictors:
        Original predictor names to keep (e.g., 'age', 'gender', 'bmi').
        For categoricals, specifying 'gender' keeps ALL one-hot levels for gender.
    categorical_cols:
        Original categorical columns (used to detect '{cat}_' prefixes).

    Returns
    -------
    mask : np.ndarray[bool]
        Boolean mask aligned with encoded_feature_names.
    """
    keep_set = set(keep_predictors)
    cat_prefixes = {cat: f"{cat}_" for cat in categorical_cols}

    mask_list: list[bool] = []
    for fn in encoded_feature_names:
        kept = False

        # If encoded name belongs to a categorical (prefix match), keep if that categorical is kept
        for cat, prefix in cat_prefixes.items():
            if fn.startswith(prefix):
                kept = (cat in keep_set)
                break

        # Otherwise it is numeric/binary passthrough: keep if exact predictor name is kept
        if not any(fn.startswith(prefix) for prefix in cat_prefixes.values()):
            kept = (fn in keep_set)

        mask_list.append(kept)

    return np.array(mask_list, dtype=bool)



# =============================================================================
# A) Random Forest feature screening (sklearn)
# =============================================================================

def fit_rf_feature_screening(
    df: pd.DataFrame,
    *,
    target_col: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
    random_state: int = 587,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float | int | None = "sqrt",
    n_jobs: int = -1,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Fit a RandomForestClassifier for feature screening (MDI importances).

    Notes
    -----
    - Intended for feature ranking / screening only (not unbiased performance).
    - Impurity-based importance (MDI) can be biased (e.g., toward continuous or
      high-cardinality features). Treat rankings as heuristic.

    Returns
    -------
    pipe:
        Fitted sklearn Pipeline (preprocessor + RandomForestClassifier).
    importance_df:
        DataFrame with columns ['feature', 'importance'], sorted descending.
        Feature names correspond to one-hot encoded columns + numeric columns.
    """
    X_df = df[categorical_cols + numeric_cols].copy()
    y = df[target_col].astype(int).to_numpy()

    preprocessor = _build_preprocessor(categorical_cols, numeric_cols)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("rf", rf)])
    pipe.fit(X_df, y)

    feature_names = _get_feature_names(preprocessor, categorical_cols, numeric_cols)
    importances = pipe.named_steps["rf"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return pipe, importance_df


def aggregate_importance_by_feature(
    importance_df: pd.DataFrame,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate one-hot encoded categorical levels back to their original feature.

    Parameters
    ----------
    importance_df:
        DataFrame with columns ['feature', 'importance'] where 'feature' includes
        one-hot encoded names (e.g., 'gender_Male').
    categorical_cols:
        Original categorical column names. Any encoded name beginning with
        '{col}_' is summed into that original '{col}'.

    Returns
    -------
    DataFrame with columns ['feature', 'importance'], aggregated and sorted descending.
    """
    agg: dict[str, float] = {}

    for _, row in importance_df.iterrows():
        feat = str(row["feature"])
        imp = float(row["importance"])

        matched_cat = False
        for cat in categorical_cols:
            prefix = f"{cat}_"
            if feat.startswith(prefix):
                agg[cat] = agg.get(cat, 0.0) + imp
                matched_cat = True
                break

        if not matched_cat:
            agg[feat] = agg.get(feat, 0.0) + imp

    out = (
        pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return out



# =============================================================================
# B) Logistic regression (statsmodels MLE) + nested comparison (LRT)
# =============================================================================

@dataclass
class LogitMLEFit:
    """
    Container for a statsmodels Logit MLE fit plus the preprocessing design.

    Attributes
    ----------
    result:
        statsmodels LogitResults (MLE fit object).
    preprocessor:
        Fitted sklearn ColumnTransformer used to build the design matrix.
    feature_names:
        Encoded feature names produced by the preprocessor (OHE + numeric cols).
    categorical_cols, numeric_cols:
        Original predictor columns used to build X_df prior to preprocessing.
    col_mask:
        Optional boolean mask selecting a subset of encoded columns (for nested reduced model).
        If None, all columns are used.
    add_intercept:
        Whether an intercept column was added using statsmodels add_constant().
    """
    result: Any
    preprocessor: ColumnTransformer
    feature_names: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    col_mask: Optional[np.ndarray]
    add_intercept: bool = True


def fit_logit_mle(
    df: pd.DataFrame,
    *,
    target_col: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
    add_intercept: bool = True,
    maxiter: int = 200,
    disp: bool = False,
) -> LogitMLEFit:
    """
    Fit an unpenalized logistic regression using MLE (statsmodels Logit),
    with sklearn one-hot encoding for categorical predictors.

    This fit type supports likelihood-based inference (LRT/"ANOVA"-style tests).
    """
    X_df = df[categorical_cols + numeric_cols].copy()
    y = df[target_col].astype(int).to_numpy()

    pre = _build_preprocessor(categorical_cols, numeric_cols)
    X = pre.fit_transform(X_df)
    feature_names = _get_feature_names(pre, categorical_cols, numeric_cols)

    X_use = X
    if add_intercept:
        X_use = sm.add_constant(X_use, has_constant="add")

    model = sm.Logit(y, X_use)
    result = model.fit(disp=disp, maxiter=maxiter)

    return LogitMLEFit(
        result=result,
        preprocessor=pre,
        feature_names=feature_names,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        col_mask=None,
        add_intercept=add_intercept,
    )


def fit_logit_mle_reduced_from_full(
    full_fit: LogitMLEFit,
    df_train: pd.DataFrame,
    *,
    target_col: str,
    keep_predictors: Sequence[str],
    maxiter: int = 200,
    disp: bool = False,
) -> LogitMLEFit:
    """
    Fit a nested reduced MLE logistic regression using the SAME preprocessing
    design as the full model, by masking encoded columns.

    IMPORTANT
    ---------
    For a valid Likelihood Ratio Test (LRT), the reduced model must be nested within
    the full model. We guarantee nesting by:
      1) reusing the SAME fitted preprocessor from full_fit, and
      2) selecting a subset of encoded columns via a boolean mask.
    """
    X_df = df_train[full_fit.categorical_cols + full_fit.numeric_cols].copy()
    y = df_train[target_col].astype(int).to_numpy()

    X_full = full_fit.preprocessor.transform(X_df)
    mask = _encoded_mask_for_original_predictors(
        full_fit.feature_names,
        keep_predictors=keep_predictors,
        categorical_cols=full_fit.categorical_cols,
    )
    X_red = _apply_mask(X_full, mask)

    X_use = X_red
    if full_fit.add_intercept:
        X_use = sm.add_constant(X_use, has_constant="add")

    model = sm.Logit(y, X_use)
    result = model.fit(disp=disp, maxiter=maxiter)

    return LogitMLEFit(
        result=result,
        preprocessor=full_fit.preprocessor,  # reuse same fitted encoder
        feature_names=full_fit.feature_names,
        categorical_cols=full_fit.categorical_cols,
        numeric_cols=full_fit.numeric_cols,
        col_mask=mask,
        add_intercept=full_fit.add_intercept,
    )


def predict_proba_logit_mle(fit: LogitMLEFit, df: pd.DataFrame) -> np.ndarray:
    """
    Predict probabilities P(y=1) using a fitted LogitMLEFit on new data.
    """
    X_df = df[fit.categorical_cols + fit.numeric_cols].copy()
    X = fit.preprocessor.transform(X_df)
    X = _apply_mask(X, fit.col_mask)

    if fit.add_intercept:
        X = sm.add_constant(X, has_constant="add")

    return np.asarray(fit.result.predict(X)).reshape(-1)


def lrt_compare_nested(full_fit: LogitMLEFit, reduced_fit: LogitMLEFit) -> dict[str, float]:
    """
    Likelihood Ratio Test (LRT) for nested logistic regression models.

    Test statistic
    --------------
    LR = 2 * (LL_full - LL_reduced),  where LL is the log-likelihood.

    Degrees of freedom
    ------------------
    df = k_full - k_reduced, where k is the number of estimated parameters.

    Returns
    -------
    dict with:
      - lr_stat, df_diff, p_value
      - llf_full, llf_reduced
      - aic_full, aic_reduced
      - bic_full, bic_reduced

    Notes
    -----
    Assumes both models were fit on the same response vector and reduced is nested in full.
    """
    llf_full = float(full_fit.result.llf)
    llf_red = float(reduced_fit.result.llf)

    # statsmodels: df_model counts parameters excluding intercept (but consistent for diff)
    df_full = int(full_fit.result.df_model)
    df_red = int(reduced_fit.result.df_model)
    df_diff = df_full - df_red

    if df_diff <= 0:
        raise ValueError(
            f"Expected full model to have more parameters than reduced. Got df_diff={df_diff}."
        )

    lr_stat = 2.0 * (llf_full - llf_red)
    p_value = float(chi2.sf(lr_stat, df_diff))

    return {
        "lr_stat": float(lr_stat),
        "df_diff": float(df_diff),
        "p_value": p_value,
        "llf_full": llf_full,
        "llf_reduced": llf_red,
        "aic_full": float(full_fit.result.aic),
        "aic_reduced": float(reduced_fit.result.aic),
        "bic_full": float(full_fit.result.bic),
        "bic_reduced": float(reduced_fit.result.bic),
    }



# =============================================================================
# C) Random Forest — Final Predictive Classifier (Planned)
# =============================================================================
# Planned functions:
# - fit_rf_classifier(...)
# - predict_proba_rf(...)
# - optional: extract_feature_importance_rf(...)
#
# Notes:
# - This is distinct from RF feature screening above.
# - Will be used for predictive performance comparison across models.


# =============================================================================
# D) Gradient Boosting (e.g., XGBoost / HistGradientBoosting) — Planned
# =============================================================================
# Planned functions:
# - fit_boosting_classifier(...)
# - predict_proba_boosting(...)
# - optional: feature importance or SHAP utilities
#
# Notes:
# - Sequential tree boosting (bias reduction).
# - Will optimize log-loss unless otherwise specified.


# =============================================================================
# E) Multi-layer Perceptron (MLP) — Planned
# =============================================================================
# Planned functions:
# - fit_mlp_classifier(...)
# - predict_proba_mlp(...)
# - optional: training history extraction (loss curves)
#
# Notes:
# - Fully connected neural network baseline.
# - May include regularization and early stopping.