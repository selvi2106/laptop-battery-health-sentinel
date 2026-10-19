# Machine Learning Models Directory

This directory stores trained scikit-learn models and training metadata saved via `joblib`.

## Saved Model Artifacts
- `linear_regression_baseline.joblib`: Baseline Linear Regression model trained on cycle-aging laboratory data.
- `random_forest_regressor.joblib`: Random Forest Regressor capturing non-linear capacity fade kinetics.
- `model_metadata.json`: Records training timestamp, training cells, test cells, MAE, RMSE, and feature columns to guarantee transparency and reproducibility.

## Engineering Principles
1. **Zero Data Leakage**:
   Train/Test splits are conducted **by battery cell identity** (e.g., Cells B0005 and B0007 for training, Cell B0006 for out-of-sample evaluation). Cycles from the same battery cell are never randomly mixed across train and test sets.
2. **Honest Metrics**:
   Evaluation metrics (MAE and RMSE) reflect real out-of-sample predictions on unseen cells.
3. **Monitoring-Only Fallback**:
   The web dashboard operates in fully functional "Monitoring-Only Mode" even if this directory contains no trained model files.
