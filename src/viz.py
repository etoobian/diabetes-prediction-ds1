"""
Visualization utilities for EDA and model comparison.

This module contains functions for:
- exploratory data analysis plots
- model performance visualizations
- comparison plots used in the final report

All plotting should be side-effect free
(i.e., no hard-coded file paths).
"""

def set_plot_defaults():
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (7, 4.5)
    plt.rcParams["axes.grid"] = True