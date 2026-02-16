"""
Input/output utilities for the Diabetes Prediction project.

This module handles:
- Loading raw data from data/raw/
- Saving and loading processed datasets from data/processed/
"""

from __future__ import annotations
import pandas as pd
from .paths import DATA_RAW_DIR


def load_raw_diabetes(filename: str = "diabetes_dataset.csv") -> pd.DataFrame:
    """
    Load the raw diabetes dataset from data/raw/.
    Returns raw dataframe as loaded from CSV.
    """
    path = DATA_RAW_DIR / filename
    df = pd.read_csv(path)
    return df

def append_data_log(row: dict, filename: str = "data_log.csv") -> None:
    """Append one row to results/tables/data_log.csv (create if missing)."""
    from .paths import TAB_DIR
    import pandas as pd

    path = TAB_DIR / filename
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)
