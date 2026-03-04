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
    ax.legend(loc="lower right")
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
    cmap: str = "viridis",
    vmax_scale: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    annot_fontsize: int = 14,
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

    # Annotate with contrast logic
    threshold = (vmin + vmax) / 2.0 if vmin is not None and vmax is not None else cm_plot.max() / 2.0

    # Annotate
    for (i, j), val in np.ndenumerate(cm_plot):
        # Choose text color based on background lightness
        text_color = "white" if val > threshold else "black"

        if normalize:
            text = f"{val:.3f}\n({int(cm[i, j])})"
        else:
            text = f"{int(val)}"

        ax.text(j, i, text, 
                ha="center", va="center", 
                fontsize=annot_fontsize,
                fontweight="bold",
                color=text_color
                )

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


def plot_single_metric_compare_bars(
    df_cmp: "pd.DataFrame",
    *,
    metric: str,
    variant_order: list[str],
    model_order: list[str],
    title: str | None = None,
):
    """
    Plot a single-metric grouped bar chart comparing model classes across dataset variants,
    using a broken y-axis to zoom into the performance range while still showing the low range.

    Assumptions (opinionated by design)
    -----------------------------------
    - `df_cmp` has columns: ['variant', 'model_class', metric]
    - Variants and model classes are exactly those passed via variant_order/model_order
    - Uses a fixed, nice palette and fixed styling
    - Draws a horizontal reference line at the mean metric value for the 'HbA1c-only' variant
    - Renders bar value labels on the top axis only

    Parameters
    ----------
    df_cmp        :  Long-form comparison table.
    metric        :  Column name of metric to plot (e.g., 'f1', 'roc_auc', 'accuracy').
    variant_order :  Display order for x-axis variant groups.
    model_order   :  Display order for bars/legend (model classes).
    title         :  Optional plot title. If None, a default is used.

    Returns
    -------
    fig, (ax_top, ax_bot)
    """
    # Pivot to wide for grouped bars
    wide = df_cmp.pivot(index="variant", columns="model_class", values=metric).loc[
        variant_order, model_order
    ]

    # Fixed palette
    palette = {
        "Logistic": "#4C72B0",
        "XGBoost": "#DD8452",
        "MLP": "#55A868",
    }

    # Geometry
    x = np.arange(len(wide.index))
    n_groups = len(wide.columns)
    bar_w = 0.2

    # Reference line at HbA1c-only mean
    hba_ref = float(wide.loc["HbA1c-only", :].mean())

    # Figure + broken axis layout
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.3, 0.9], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # Draw bars on both axes
    for j, col in enumerate(wide.columns):
        vals = wide[col].to_numpy(dtype=float)
        xpos = x + (j - (n_groups - 1) / 2) * bar_w
        color = palette.get(str(col), None)

        bars_top = ax_top.bar(xpos, vals, width=bar_w, label=str(col), color=color)
        ax_bot.bar(xpos, vals, width=bar_w, color=color)

        # Value labels on TOP axis only
        for b in bars_top:
            h = float(b.get_height())
            ax_top.text(
                b.get_x() + b.get_width() / 2,
                h + 0.002,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Zoom range (top) + low range (bottom)
    y_min_zoom = max(0.0, float(np.nanmin(wide.to_numpy())) - 0.05)
    y_max_zoom = min(1.0, float(np.nanmax(wide.to_numpy())) + 0.05)

    ax_top.set_ylim(y_min_zoom, y_max_zoom)
    ax_bot.set_ylim(0.0, y_min_zoom)

    # Broken-axis styling
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)

    top_ticks = [0.60, 0.70, 0.80, 0.90, 1.00]
    top_ticks = [t for t in top_ticks if y_min_zoom <= t <= y_max_zoom]
    ax_top.set_yticks(top_ticks)
    ax_bot.set_yticks([0.00, 0.20, 0.40])

    # Diagonal marks
    d = 0.008
    kwargs = dict(color="k", clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)

    # X-axis
    ax_top.tick_params(labelbottom=False)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(wide.index, fontsize=12)
    # Remove Numeric x-ticks
    ax_bot.set_xlim(-0.5, len(x) - 0.5)
    ax_bot.xaxis.set_major_locator(plt.FixedLocator(x))
    ax_bot.xaxis.set_minor_locator(plt.NullLocator())
    ax_bot.tick_params(axis="x", which="minor", bottom=False)

    # Titles / labels
    if title is None:
        title = (
            f"Test-Set {metric.upper()} Comparison (Train-Selected Youden Thresholds)"
        )

    ax_top.set_title(title, fontsize=16, fontweight="bold", pad=10)

    if metric == "f1":
        y_lab = "F1 Score"
    else:
        y_lab = metric.upper()
    ax_top.set_ylabel(y_lab, fontsize=13, fontweight="bold")
    ax_bot.set_xlabel("Dataset Variant", fontsize=13, fontweight="bold")

    # Grid: y-only, no vertical lines
    for a, alpha in [(ax_top, 0.25), (ax_bot, 0.15)]:
        a.grid(False)
        a.grid(axis="y", alpha=alpha)

    # Clean spines
    for ax in (ax_top, ax_bot):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Reference line + label
    ax_top.axhline(hba_ref, linestyle="--", linewidth=1.6, alpha=0.9)
    ax_top.text(
        0.98,
        hba_ref + 0.001,
        rf"HbA1c-only reference ($\approx$ {hba_ref:.3f})",
        ha="right",
        va="bottom",
        transform=ax_top.get_yaxis_transform(),
        fontsize=9,
        alpha=0.6,
    )

    # Legend + note
    leg = ax_top.legend(
        title="Model Class",
        loc="upper left",
        bbox_to_anchor=(0.77, 0.85),
        frameon=True,
    )
    leg.get_frame().set_alpha(0.95)

    fig.text(
        0.70,
        0.57,
        "NOTE: y-axis break zooms\n" "           performance range.",
        ha="left",
        va="top",
        fontsize=9,
        alpha=0.9,
    )

    return fig