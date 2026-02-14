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