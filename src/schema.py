"""
Dataset schema and validation references for the diabetes prediction project.

This module is the single source of truth for:
- Target column name
- Feature groupings (categorical / numeric / binary)
- Reference expectations from the dataset card (REF_*)
- Operational expectations used in this project (non-REF)

Design:
- REF_* represent documentation-provided expectations (e.g., Kaggle card).
  These are used for audit/reporting and may not perfectly match the CSV.
- Operational expectations represent how we treat the dataset used in this repo.
  These are used for consistent preprocessing/EDA/modeling and guardrail validation.

Other modules (preprocessing, viz, modeling, metrics) should import schema
information from here rather than redefining it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# -------- CORE COLUMNS --------
TARGET_COL: str = "diagnosed_diabetes"

# Remove Alternative Target Variables (avoid data leakage)
LEAKAGE_COLS: List[str] = [
    "diabetes_stage", 
    "diabetes_risk_score",
]

# -------- Feature groupings --------
CATEGORICAL_COLS: List[str] = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
    #"diabetes_stage",
]

BINARY_COLS: List[str] = [
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    TARGET_COL,
]

NUMERIC_COLS: List[str] = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    #"diabetes_risk_score",
]


# --------REFERENCE EXPECTATIONS --------
# From dataset card / Kaggle
REF_ALLOWED_CATEGORIES: Dict[str, List[str]] = {
    "gender": ["Male", "Female", "Other"],
    "ethnicity": ["White", "Hispanic", "Black", "Asian", "Other"],
    "education_level": ["No formal", "Highschool", "Graduate", "Postgraduate"],
    "income_level": ["Low", "Medium", "High"],
    "employment_status": ["Employed", "Unemployed", "Retired", "Student"],
    "smoking_status": ["Never", "Former", "Current"],
    #"diabetes_stage": ["No Diabetes", "Pre-Diabetes", "Type 1", "Type 2", "Gestational"],
}

REF_RANGE_RULES: Dict[str, Tuple[float, float]] = {
    "age": (18, 90),
    "alcohol_consumption_per_week": (0, 30),
    "physical_activity_minutes_per_week": (0, 600),
    "diet_score": (0, 10),
    "sleep_hours_per_day": (3, 12),
    "screen_time_hours_per_day": (0, 12),
    "bmi": (15, 45),
    "waist_to_hip_ratio": (0.7, 1.2),
    "systolic_bp": (90, 180),
    "diastolic_bp": (60, 120),
    "heart_rate": (50, 120),
    "cholesterol_total": (120, 300),
    "hdl_cholesterol": (20, 100),
    "ldl_cholesterol": (50, 200),
    "triglycerides": (50, 500),
    "glucose_fasting": (70, 250),
    "glucose_postprandial": (90, 350),
    "insulin_level": (2, 50),
    "hba1c": (4, 14),
    #"diabetes_risk_score": (0, 100),
}


# --------OPERATIONAL EXPECTATIONS --------
# Operational categories reflect the CSV used in this repo.

ALLOWED_CATEGORIES = dict(REF_ALLOWED_CATEGORIES)
# income_level differs from the dataset card.
ALLOWED_CATEGORIES["income_level"] = ["Low","Lower-Middle","Middle","Upper-Middle","High"]


RANGE_RULES: Dict[str, Tuple[float, float]] = dict(REF_RANGE_RULES)
# Ranges updated kept within reasonable values
RANGE_RULES.update(
    {
        # Observed max ~833
        "physical_activity_minutes_per_week": (0, 900),
        # Observed max ~16.8
        "screen_time_hours_per_day": (0, 18),
        # Observed min ~0.67
        "waist_to_hip_ratio": (0.65, 1.2),
        # observed min ~50
        "diastolic_bp": (50, 120),
        # observed min ~40
        "heart_rate": (40, 120),
        # observed min ~100, max ~318
        "cholesterol_total": (100, 320),
        # observed max ~263
        "ldl_cholesterol": (50, 270),
        # observed min ~30
        "triglycerides": (30, 500),
        # observed min ~60
        "glucose_fasting": (60, 250),
        # observed min ~70
        "glucose_postprandial": (70, 350),
    }
)