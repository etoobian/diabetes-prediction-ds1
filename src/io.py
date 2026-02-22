"""
Input/output utilities for the Diabetes Prediction project.

This module handles:
- Loading raw data from data/raw/
- Saving and loading processed datasets from data/processed/
- Saving Tables to results/tables (CSV and/or LaTex)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .paths import DATA_RAW_DIR

import json
from typing import Any

import pandas as pd


# ----- DATA (CSV) -----
def load_raw_diabetes(filename: str = "diabetes_dataset.csv") -> pd.DataFrame:
    """
    Load the raw diabetes dataset from data/raw/.
    Returns raw dataframe as loaded from CSV.
    """
    path = DATA_RAW_DIR / filename
    df = pd.read_csv(path)
    return df


def load_processed_data(path: Path) -> pd.DataFrame:
    """
    Load a processed dataset from disk.
    """
    path = Path(path)
    return pd.read_csv(path)


def save_processed_data(
    df: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> None:
    """
    Save a processed dataset to disk.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Full output path (e.g., PROC_SPLITS_DIR / "train_base.csv").
    index : bool, default False
        Whether to save the DataFrame index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

# ----- MODELING DATASET HELPERS -----

@dataclass(frozen=True)
class ThreeVariantSplits:
    """
    Container for 3 predictor-set variants built from the same base dataset (train/test):

    - all: all predictors present in the input artifact
    - hba1c: HbA1c-only baseline
    - non_hba1c: all predictors except HbA1c

    Each variant always includes the target column.
    """
    train_all: pd.DataFrame
    test_all: pd.DataFrame
    train_hba1c: pd.DataFrame
    test_hba1c: pd.DataFrame
    train_non_hba1c: pd.DataFrame
    test_non_hba1c: pd.DataFrame
    predictors_all: list[str]
    predictors_hba1c: list[str]
    predictors_non_hba1c: list[str]


def load_and_build_three_variant_splits(
    *,
    proc_dir: Path,
    target_col: str,
    hba1c_col: str = "hba1c",
    train_filename: str,
    test_filename: str,
) -> ThreeVariantSplits:
    """
    Load a processed train/test artifact pair and construct 3 predictor-set variants:

    1) all predictors (as provided by the artifact)
    2) HbA1c-only
    3) all predictors except HbA1c

    Parameters
    ----------
    proc_dir : Path
        Directory containing the processed artifacts.
    target_col : str
        Target column name.
    hba1c_col : str, default "hba1c"
        HbA1c column name.
    train_filename, test_filename : str
        CSV filenames inside proc_dir.

    Returns
    -------
    ThreeVariantSplits
        Dataclass containing six datasets and predictor lists.
    """
    proc_dir = Path(proc_dir)
    train_path = proc_dir / train_filename
    test_path = proc_dir / test_filename

    train_df = load_processed_data(train_path)
    test_df = load_processed_data(test_path)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in artifacts.")

    predictors_all = [c for c in train_df.columns if c != target_col]

    if hba1c_col not in predictors_all:
        raise KeyError(f"Expected '{hba1c_col}' to be present in the artifact predictors.")

    predictors_hba1c = [hba1c_col]
    predictors_non_hba1c = [c for c in predictors_all if c != hba1c_col]

    def _subset(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
        return df[preds + [target_col]].copy()

    return ThreeVariantSplits(
        train_all=_subset(train_df, predictors_all),
        test_all=_subset(test_df, predictors_all),
        train_hba1c=_subset(train_df, predictors_hba1c),
        test_hba1c=_subset(test_df, predictors_hba1c),
        train_non_hba1c=_subset(train_df, predictors_non_hba1c),
        test_non_hba1c=_subset(test_df, predictors_non_hba1c),
        predictors_all=predictors_all,
        predictors_hba1c=predictors_hba1c,
        predictors_non_hba1c=predictors_non_hba1c,
    )



# ----- METADATA (JSON) -----
def save_metadata_json(
    obj: dict[str, Any],
    path: Path,
) -> None:
    """
    Save a metadata dictionary to disk as JSON.

    Parameters
    ----------
    obj : dict[str, Any]
        Metadata to save.
    path : Path
        Full output path (e.g., PROC_SPLITS_DIR / "split.json").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_metadata_json(path: Path) -> dict[str, Any]:
    """
    Load a metadata dictionary from a JSON file on disk.

    Parameters
    ----------
    path : Path
        Full path to the JSON file.

    Returns
    -------
    dict[str, Any]
        Loaded metadata dictionary.
    """
    path = Path(path)
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