"""
Source package for the Diabetes Prediction DS1 project.

Modules:
- io: data loading/saving and paths
- preprocessing: cleaning/feature engineering/splits
- modeling: fit/predict for each model
- metrics: scoring and summary tables
- viz: EDA and comparison plots
"""

from . import schema
from . import io
from . import preprocessing
from . import modeling
from . import metrics
from . import viz

__all__ = ["schema", "io", "preprocessing", "modeling", "metrics", "viz"]