"""
Centralized project paths.

Defines repository-root-relative paths for:
- Raw and Processed data
- Saved figures
- Saved tables

Also provides ensure_dirs() to create output directories as needed.
"""

from pathlib import Path

# Project root: diabetes-predicton-ds1/
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"

def ensure_dirs() -> None:
    """Create required project directories if they do not exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
