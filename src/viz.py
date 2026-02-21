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


def set_plot_defaults():
    plt.rcParams["figure.figsize"] = (7, 4.5)
    plt.rcParams["axes.grid"] = True


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