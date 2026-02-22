"""
Evaluation metrics and result summarization utilities.

This module provides helpers for:
- Computing classification metrics from predicted probabilities
- Building standardized result dictionaries / tables for model comparison
- Producing calibration artifacts (Brier score + calibration curve data)
- Selecting probability thresholds on training data

Notes
-----
- No plotting logic should live here.
- Functions should accept y_true and y_prob (predicted probability for the positive class).
- Threshold tuning should be performed on training data only (or CV predictions).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _to_1d_array(x) -> np.ndarray:
    """Coerce input to a 1D numpy array."""
    return np.asarray(x).reshape(-1)


def threshold_predictions(y_prob, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities to hard labels using a threshold."""
    y_prob = _to_1d_array(y_prob).astype(float)
    return (y_prob >= threshold).astype(int)


def _binary_guardrails(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Validate binary labels and probability range."""
    uniq = set(np.unique(y_true))
    if uniq - {0, 1}:
        raise ValueError(f"y_true must be binary {{0,1}}. Got labels={sorted(uniq)}")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must be probabilities in [0,1].")


def compute_classification_metrics(
    y_true,
    y_prob,
    *,
    threshold: float = 0.5,
    pos_label: int = 1,
    sample_weight=None,
) -> Dict[str, Any]:
    """
    Compute a standard set of metrics for binary classification.

    Parameters
    ----------
    y_true:
        True labels (0/1).
    y_prob:
        Predicted probabilities for the positive class (P(Y=1|X)).
    threshold:
        Classification threshold for label metrics (default 0.5).
    pos_label:
        Positive class label (default 1).
    sample_weight:
        Optional per-sample weights.

    Returns
    -------
    dict
        Dictionary containing threshold-free and threshold-based metrics plus confusion matrix.
    """
    y_true = _to_1d_array(y_true).astype(int)
    y_prob = _to_1d_array(y_prob).astype(float)
    _binary_guardrails(y_true, y_prob)

    if pos_label != 1:
        raise ValueError(
            "This project assumes positive class label is 1."
        )
    
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    y_pred = threshold_predictions(y_prob, threshold=float(threshold))

    # Confusion matrix in standard order: [[TN, FP],[FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()

    # Account for undefined metrics if y_true has only one class
    uniq = np.unique(y_true)
    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(uniq) == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
        pr_auc = float(average_precision_score(y_true, y_prob, sample_weight=sample_weight))

    out: Dict[str, Any] = {
        "threshold": float(threshold),

        # Threshold-free ranking metrics
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        # Probabilistic metrics
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1], sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight)),

        # Thresholded metrics
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0, sample_weight=sample_weight)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0, sample_weight=sample_weight)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0, sample_weight=sample_weight)),

        # Confusion matrix entries
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    return out


def calibration_artifacts(
    y_true,
    y_prob,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """
    Return calibration curve data (no plotting).

    Returns a DataFrame with columns:
        bin_mean_pred, bin_frac_pos
    """
    y_true = _to_1d_array(y_true).astype(int)
    y_prob = _to_1d_array(y_prob).astype(float)
    _binary_guardrails(y_true, y_prob)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return pd.DataFrame({"bin_mean_pred": mean_pred, "bin_frac_pos": frac_pos})


def metrics_to_frame(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert {model_name: metrics_dict} into a tidy DataFrame.
    """
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "model"
    return df


def pick_threshold_by_f1(
    y_true,
    y_prob,
    *,
    grid: Optional[np.ndarray] = None,
    sample_weight=None,
) -> Dict[str, Any]:
    """
    Choose a threshold that maximizes F1 on provided data.

    IMPORTANT:
    - Use on training data only (or CV predictions), never on the test set.

    Returns a dict with best_threshold, best f1, and a small diagnostic table.
    """
    y_true = _to_1d_array(y_true).astype(int)
    y_prob = _to_1d_array(y_prob).astype(float)
    _binary_guardrails(y_true, y_prob)

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    rows = []
    best = {"threshold": None, "f1": -np.inf}

    for t in grid:
        y_pred = threshold_predictions(y_prob, threshold=float(t))
        f1 = f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        rows.append({"threshold": float(t), "f1": float(f1)})
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}

    diag = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return {"best_threshold": best["threshold"], "best_f1": best["f1"], "diagnostic": diag}


def pick_threshold_by_youden_j(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """
    Pick a probability threshold using Youden's J statistic on the ROC curve.

    Youden's J = TPR - FPR = sensitivity + specificity - 1.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels in {0,1}.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    dict with:
        - best_threshold
        - best_j
        - best_tpr
        - best_fpr
        - best_specificity
    """
    from sklearn.metrics import roc_curve

    y_true = _to_1d_array(y_true)
    y_prob = _to_1d_array(y_prob)
    _binary_guardrails(y_true, y_prob)

    fpr, tpr, thr = roc_curve(y_true, y_prob)

    # roc_curve may include an initial threshold of inf; exclude it.
    mask = np.isfinite(thr)
    fpr = fpr[mask]
    tpr = tpr[mask]
    thr = thr[mask]

    j = tpr - fpr
    idx = int(np.argmax(j))

    best_thr = float(thr[idx])
    best_tpr = float(tpr[idx])
    best_fpr = float(fpr[idx])
    best_j = float(j[idx])

    return {
        "best_threshold": best_thr,
        "best_j": best_j,
        "best_tpr": best_tpr,
        "best_fpr": best_fpr,
        "best_specificity": float(1.0 - best_fpr),
    }


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    method: str = "youden",
) -> dict[str, float]:
    """
    Convenience wrapper to pick a threshold.

    method:
      - "youden" -> pick_threshold_by_youden_j
      - "f1"     -> pick_threshold_by_f1
    """
    method = method.lower().strip()
    if method == "youden":
        return pick_threshold_by_youden_j(y_true, y_prob)
    if method == "f1":
        return pick_threshold_by_f1(y_true, y_prob)
    raise ValueError(f"Unknown method={method!r}. Use 'youden' or 'f1'.")


def compare_models_table(
    results: dict[str, dict[str, float]],
    *,
    sort_by: str = "f1",
    descending: bool = True,
) -> pd.DataFrame:
    """
    Build a comparison table from model-name -> metrics dict.

    Intended for reuse across Logistic Regression / RF / Boosting / MLP sections.

    Parameters
    ----------
    results : dict
        Mapping model_name -> metrics dict (e.g., output of compute_classification_metrics).
    sort_by : str
        Column to sort by if present.
    """
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending).reset_index(drop=True)
    return df