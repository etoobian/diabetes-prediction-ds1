"""
Visualization utilities for EDA and model comparison.

This module contains:
- EDA plotting helpers
- Model comparison plots
- Figure formatting utilities (e.g., global defaults)
- Figure finalization helpers (save/show/close) using caller-provided paths

Plotting functions should not hard-code file paths; paths are supplied by the notebook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def set_plot_defaults(
    *,
    figsize: tuple[float, float] = (7.0, 4.5),
    grid: bool = True,
    title_size: int = 14,
    label_size: int = 12,
    tick_size: int = 10,
    legend_size: int = 10,
    title_weight: str = "bold",
    label_pad: int = 10,
):
    """
    Set global matplotlib defaults used across the project.

    Notes
    -----
    Individual plots may override defaults locally if needed.
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["axes.grid"] = grid

    plt.rcParams["axes.titlesize"] = title_size
    plt.rcParams["axes.titleweight"] = title_weight
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size
    plt.rcParams["legend.fontsize"] = legend_size
    plt.rcParams["axes.labelpad"] = label_pad


def finalize_fig(fig, *, save_path: Path | None = None, show: bool = True, dpi: int = 300):
    """
    Save and/or show a matplotlib figure, then close it.

    Parameters
    ----------
    fig:        Matplotlib figure.
    save_path:  If provided, save figure to this path.
    show:       If True, display the figure.
    dpi:        Save resolution.
    """
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 20,
    title: str = "Random Forest Feature Importance (MDI)",
):
    """Return a bar plot (fig, ax) of top-N feature importances."""
    d = importance_df.head(top_n).copy()
    d = d.iloc[::-1]

    fig, ax = plt.subplots()
    ax.barh(d["feature"], d["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    
    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    title: str = "ROC Curve",
):
    """
    Plot ROC curve and return (fig, ax).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_proba = np.asarray(y_proba).reshape(-1)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_multiple_roc_curves(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    *,
    title: str = "ROC Curves",
):
    """
    Plot multiple ROC curves on the same axes.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    model_probas : dict
        Mapping {model_name: predicted_probabilities}.
    title : str
        Plot title.

    Returns
    -------
    (fig, ax)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    y_true = np.asarray(y_true).reshape(-1)

    fig, ax = plt.subplots()

    for name, y_proba in model_probas.items():
        y_proba = np.asarray(y_proba).reshape(-1)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    title: str = "Precision-Recall Curve",
):
    """
    Plot Precision-Recall curve and return (fig, ax).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_proba = np.asarray(y_proba).reshape(-1)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_multiple_pr_curves(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    *,
    title: str = "Precision–Recall Curves",
):
    """
    Plot multiple Precision-Recall curves on the same axes.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    model_probas : dict
        Mapping {model_name: predicted_probabilities}.
    title : str
        Plot title.

    Returns
    -------
    (fig, ax)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_true = np.asarray(y_true).reshape(-1)

    fig, ax = plt.subplots()

    for name, y_proba in model_probas.items():
        y_proba = np.asarray(y_proba).reshape(-1)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_calibration_curve_from_bins(
    calib_df: pd.DataFrame,
    *,
    prob_col: str = "bin_mean_pred",
    frac_col: str = "bin_frac_pos",
    title: str = "Calibration Curve",
):
    """
    Plot calibration curve from calibration_artifacts() output.

    Expected columns (by default):
      - bin_mean_pred
      - bin_frac_pos
    """
    fig, ax = plt.subplots()
    ax.plot(calib_df[prob_col], calib_df[frac_col], marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_multiple_calibration_curves(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Calibration Curves",
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true).reshape(-1)

    fig, ax = plt.subplots()

    for name, y_proba in model_probas.items():
        y_proba = np.asarray(y_proba).reshape(-1)
        frac_pos, mean_pred = calibration_curve(
            y_true, y_proba,
            n_bins=n_bins,
            strategy=strategy,
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_confusion_matrix_binary(
    tn: int, fp: int, fn: int, tp: int,
    *,
    title: str | None = None,
    normalize: bool = False,
    cmap: str = "Blues",
    vmax_scale: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    annot_fontsize: int = 11,
):
    import numpy as np
    import matplotlib.pyplot as plt

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm / row_sums
    else:
        cm_plot = cm

    # Defaults for color scaling
    if vmin is None:
        vmin = 0.0

    if vmax is None:
        vmax = float(cm_plot.max()) * float(vmax_scale)

        if normalize:
            vmax = 1.0 * float(vmax_scale)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_plot, cmap=cmap, vmin=vmin, vmax=vmax)

    # Annotate
    for (i, j), val in np.ndenumerate(cm_plot):
        if normalize:
            text = f"{val:.3f}\n({int(cm[i, j])})"
        else:
            text = f"{int(val)}"
        ax.text(j, i, text, ha="center", va="center", fontsize=annot_fontsize)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if title:
        ax.set_title(title)

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, ax