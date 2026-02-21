"""
Input/output utilities for the Diabetes Prediction project.

This module handles:
- Loading raw data from data/raw/
- Saving and loading processed datasets from data/processed/
- Saving Tables to results/tables (CSV and/or LaTex)
"""

from __future__ import annotations

from pathlib import Path
from .paths import DATA_RAW_DIR

import pandas as pd

import json
from typing import Any


# ----- DATA (CSV) -----
def load_raw_diabetes(filename: str = "diabetes_dataset.csv") -> pd.DataFrame:
    """
    Load the raw diabetes dataset from data/raw/.
    Returns raw dataframe as loaded from CSV.
    """
    path = DATA_RAW_DIR / filename
    df = pd.read_csv(path)
    return df


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Load a processed dataset from data/processed/.
    """
    from .paths import DATA_PROC_DIR

    path = DATA_PROC_DIR / filename
    return pd.read_csv(path)


def save_processed_data(
    df: pd.DataFrame,
    filename: str,
    *,
    index: bool = False,
) -> None:
    """
    Save a processed dataset to data/processed/.

    Parameters
    ----------
    df :        DataFrame to save.
    filename :  Name of output file (e.g., 'diabetes_clean.csv').
    index :     Whether to save the index (default False).
    """
    from .paths import DATA_PROC_DIR

    path = DATA_PROC_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


# ----- METADATA (JSON) -----
def save_metadata_json(
    obj: dict[str, Any],
    filename: str,
) -> None:
    """
    Save a metadata dictionary to data/processed/ as JSON.

    Parameters
    ----------
    obj : dict
        Metadata to save.
    filename : str
        Output file name (e.g., 'split.json').
    """
    from .paths import DATA_PROC_DIR

    path = DATA_PROC_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_metadata_json(filename: str) -> dict[str, Any]:
    """
    Load a metadata dictionary from data/processed/ JSON.
    """
    from .paths import DATA_PROC_DIR

    path = DATA_PROC_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----- TABLES -----

# Tables as CSV
def save_table_csv(df: pd.DataFrame, path: Path, *, index: bool = False, save: bool = True) -> Path | None:
    """
    Save a DataFrame to disk as CSV.

    Parameters
    ----------
    df :    DataFrame  -  Table to save.
    path :  Path  -  Output file path.
    index : bool
    save :  bool  -  If False, do nothing and return None.

    Returns
    -------
    Path | None  -  The output path if saved, else None.
    """
    if not save:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


# LaTex TABLES
def latex_escape_underscores(s) -> str:
    """Escape underscores for LaTeX."""
    return str(s).replace("_", r"\_")


def escape_df_underscores(df: pd.DataFrame, cols: tuple[str, ...] = ("feature",)) -> pd.DataFrame:
    """Escape underscores in specified string columns (returns a copy)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).apply(latex_escape_underscores)
    return out


def df_to_tabular_tex(df: pd.DataFrame, *, float_fmt: str = "%.4f", index: bool = False) -> str:
    """Return LaTeX tabular with booktabs formatting."""
    return df.to_latex(
        index=index,
        escape=False,
        float_format=(lambda x: float_fmt % x) if float_fmt else None,
        bold_rows=False,
        longtable=False,
    )


def wrap_table(tabular_tex: str, *, caption: str, label: str) -> str:
    """Wrap a tabular in a standalone LaTeX table environment."""
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\begin{center}",
            tabular_tex.strip(),
            r"\end{center}",
            r"\vspace{-5pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


def save_table_tex(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    float_fmt: str = "%.4f",
    index: bool = False,
    escape_underscore_cols: tuple[str, ...] | None = ("feature",),
    save: bool = True,
) -> Path | None:
    """
    Save a DataFrame as a LaTeX table file.

    Parameters
    ----------
    df :       DataFrame
    path :     Path  -  Output file path (e.g., TAB_DIR / "my_table.tex")
    caption :  str
    label :    str
    save :     bool  -  If False, do nothing and return None.
    """
    if not save:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if escape_underscore_cols:
        df = escape_df_underscores(df, cols=escape_underscore_cols)

    tex = wrap_table(
        df_to_tabular_tex(df, float_fmt=float_fmt, index=index),
        caption=caption,
        label=label,
    )
    path.write_text(tex, encoding="utf-8")
    return path