# xG Analyzer

An Expected Goals (xG) predictive model built using Python, Scikit-Learn, and XGBoost to quantify shot quality and scoring probability from match event data with a ROC-AUC score of 0.812

## Overview
This project processes historical shot event data (8,118 records) to train a machine learning model that predicts the probability of a shot resulting in a goal. It includes data cleaning, feature engineering, and model evaluation using standard classification metrics.

## Features
- **Data Preprocessing & Cleaning:** Handles missing values, filters non-shot events, and casts columns to correct data types.
- **Feature Engineering:** Computes Euclidean shot distance and spatial variables from coordinate data.
- **Model Training:** Implements machine learning algorithms using `XGBoost` and `Scikit-Learn`.
- **Model Evaluation:** Evaluates predictive performance using ROC-AUC, Log Loss, and Brier Score Loss.
- **Data Visualization:** Plots shot maps and expected goal gradients using `matplotlib` and `mplsoccer`.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn, XGBoost
- Matplotlib, MPL Soccer

## Installation & Usage
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib mplsoccer
