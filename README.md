# **Leveraging Machine Learning to Predict Diabetes Using Patient-Level Health and Risk Factors:** Insights from the *Diabetes Helath Indicators* Dataset

**Course:** STAT 587 &mdash; Data Science I  
**Term:** Winter 2026  
**Team:** Esther Toobian, Edwin Mutimba  

---

## Project Overview
This project investigates the prediction of diabetes diagnosis using supervised learning models applied to patient-level health and risk factor data. The analysis compares multiple modeling approaches drawn from the *Elements of Statistical Learning (ESL)* framework, with emphasis on both **predictive performance** and **model interpretability**.

The final deliverable is a **single, fully reproducible Jupyter notebook** that runs end-to-end and produces all results, figures, and tables used in the final report and presentation.

---

## Project Goals
- Compare supervised learning methods for diabetes prediction
- Evaluate performance using appropriate classification metrics
- Identify influential demographic, lifestyle, and clinical risk factors
- Assess trade-offs between interpretability and model flexibility

---

## Repository Structure

```
README.md                       # Project overview and instructions

requirements.txt                # Python package dependencies

data/
├── README.md                   # Data documentation
├── raw/                        # Raw input dataset
│   └── diabetes_dataset.csv
├── processed/                  # Directory for generated, reproducible processed data
|   ├── splits/                 # Stratified 80/20 train and test splits from raw data
|   ├── curated/                # Train / test splits post-removal of specified variables
│   └── reduced/                # Reduced datasets used for nested comparison

notebooks/
└── final_project.ipynb         # Single submission notebook (run top-to-bottom)

src/
├── __init__.py                 # Source package initializer
├── paths.py                    # Centralized project paths and directory initializations
├── io.py                       # Data loading/saving and path utilities
├── schema.py                   # Dataset schema and validation reference
├── preprocessing.py            # Cleaning, encoding, scaling, splitting
├── modeling.py                 # Model training/evaluation routines
├── metrics.py                  # Evaluation metrics and result summaries
└── viz.py                      # EDA and model comparison visualizations

results/
├── figures/                    # Saved plots, organized by notebook section
└── tables/                     # Saved tables (CSV and LaTex), organized by notebook section

reports/
├── final_presentations/        # Final written deliverables (report + slideshow)
└── milestones/                 # Milestone delivarables (written reports)

team_docs/
├── GitWorkflow.md              # Git workflow and branching conventions
└── PR_Checklist.md             # Pull request checklist and expectations

.github/
└── CODEOWNERS                  # Repository ownership and review rules
```

---

## Setup Instructions

### Clone the Repository

```
git clone https://github.com/etoobian/diabetes-prediction-ds1.git
cd diabetes-pediction-ds1
```

---

## Environment

This project uses Python and standard data science libraries.

Required packages are listed in `requirements.txt`.

It is recommended to work in a virtual environment (conda or venv).

To install dependencies:

```
pip install -r requirements.txt
```

Exact package versions are not pinned to avoid platform-specific issues.  
All results are reproducible due to fixed random seeds and a deterministic workflow.

```

---

## Data Setup

### Raw Data
- Place the provided dataset in:
  ```
  data/raw/diabetes_dataset.csv
  ```
- Raw data **is tracked** in this repository for reproducibility.
- Raw files should never be modified directly.

### Processed Data
- Processed datasets are generated automatically by the notebook.
- Outputs are written to:
  ```
  data/processed/
  ```
- Processed data files are **not tracked** in version control.

See `data/README.md` for full dataset documentation.

---

## Running the Analysis

Open and run:

```
notebooks/final_project.ipynb
```

**Important:**
- Run the notebook **top to bottom**
- All randomness is controlled via a fixed seed
- All outputs are generated automatically

Generated artifacts:
- Figures $\to$ `results/figures/`
- Tables $\to$ `results/tables/`
- Processed data $\to$ `data/processed/`

---

## Modeling Approaches
The following supervised learning methods are evaluated:

1. **Logistic Regression**
   - Baseline model

2. **Gradient Boosting (XGBoost)**
   - High-capacity ensemble method

3. **Multilayer Perceptron (MLP)**
   - Shallow neural network explored

---

## Evaluation Framework
Models are compared using consistent metrics appropriate for a moderately imbalanced binary outcome (~60/40 split):

- ROC-AUC
- PR-AUC
- Log-Loss
- Brier's
- Accuracy
- Confusion matrix and derived metrics (precision, recall, F1)

All models use the same train/test split with a fixed random seed to ensure reproducibility.

---

## Reproducibility
- A fixed random seed is set at the top of the notebook
- All file paths are relative and platform-independent
- The notebook is self-contained and produces all outputs from raw data

---

## Notes
- The dataset is synthetically generated but medically realistic.
- Minor inconsistencies exist between dataset documentation and the provided CSV; the CSV is treated as the authoritative source.
- All conclusions should be interpreted with appropriate caution regarding real-world generalization.