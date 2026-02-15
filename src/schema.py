"""
Project schema for the diabetes prediction dataset.

This module is the single source of truth for:
- Target column name
- Feature groupings (categorical / numeric / binary)
- Allowed categorical levels (when known)
- Numeric range rules (when known)

Other modules (preprocessing, viz, modeling, metrics) should import schema
information from here rather than redefining it.
"""

from __future__ import annotations
from typing import Dict, List, Tuple


# -------- CORE COLUMNS --------
TARGET_COL: str = "diagnosed_diabetes"

# Empty to Start
DROP_COLS: List[str] = []

# -------- Feature groupings --------
CATEGORICAL_COLS: List[str] = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
    "diabetes_stage",
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
    "diabetes_risk_score",
]

# -------- ALLOWED CATEGORY LEVELS --------
# (from Kaggle Data Card)
ALLOWED_CATEGORIES: Dict[str, List[str]] = {
    "gender": ["Male", "Female", "Other"],
    "ethnicity": ["White", "Hispanic", "Black", "Asian", "Other"],
    "education_level": ["No formal", "Highschool", "Graduate", "Postgraduate"],
    "income_level": ["Low", "Medium", "High"],
    "employment_status": ["Employed", "Unemployed", "Retired", "Student"],
    "smoking_status": ["Never", "Former", "Current"],
    "diabetes_stage": ["No Diabetes", "Pre-Diabetes", "Type 1", "Type 2", "Gestational"],
}

# -------- NUMERIC RANGE RULES --------
# (from Kaggle Data Card)
RANGE_RULES: Dict[str, Tuple[float, float]] = {
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
    "diabetes_risk_score": (0, 100),
}