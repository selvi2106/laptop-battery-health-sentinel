"""
train_model.py
==============
Train, evaluate, and persist machine learning models for battery degradation estimation.

Models Compared:
1. Linear Regression (interpretable linear baseline)
2. Random Forest Regressor (non-linear ensemble capturing complex fade dynamics)

Evaluation Philosophy:
- Cell-based train/test splitting (no cycle mixing leakage).
- Honest evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Persists trained models to models/ directory using joblib.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_dataset
from src.preprocessing import clean_lab_dataset, split_by_battery_cell


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate honest regression metrics: MAE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def train_and_evaluate_models(
    csv_path: str,
    train_cells: list = None,
    test_cells: list = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Train Linear Regression baseline and Random Forest on laboratory battery data.

    Parameters
    ----------
    csv_path : str
        Path to NASA lab experimental CSV.
    train_cells : list, optional
        Battery IDs for training (e.g., ['B0005', 'B0007']).
    test_cells : list, optional
        Battery IDs for testing (e.g., ['B0006']).
    save_artifacts : bool
        Whether to serialize trained models to disk via joblib.

    Returns
    -------
    dict
        Evaluation results, model instances, test set predictions, and metadata.
    """
    # 1. Load data
    load_res = load_dataset(csv_path)
    if not load_res["is_valid"]:
        raise ValueError(f"Failed to load dataset from {csv_path}: {load_res['message']}")

    raw_df = load_res["data"]

    # 2. Preprocess
    cleaned_df, report = clean_lab_dataset(raw_df)
    if not report["is_valid"]:
        raise ValueError(f"Preprocessing failed: {report['message']}")

    # 3. Cell-level split to avoid target/cycle leakage
    if train_cells is None or test_cells is None:
        cells = list(cleaned_df["battery_id"].unique())
        if "B0006" in cells and len(cells) >= 3:
            test_cells = ["B0006"]
            train_cells = [c for c in cells if c != "B0006"]
        else:
            test_cells = [cells[-1]]
            train_cells = cells[:-1]

    X_train, y_train, X_test, y_test, feature_cols = split_by_battery_cell(
        cleaned_df,
        train_cells=train_cells,
        test_cells=test_cells,
        target_col="soh_percent"
    )

    # 4. Model 1: Baseline Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics_lr = evaluate_model(y_test, y_pred_lr)

    # 5. Model 2: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_rf = evaluate_model(y_test, y_pred_rf)

    # Feature importance from Random Forest
    feat_importance = dict(zip(feature_cols, [round(float(imp), 4) for imp in rf.feature_importances_]))

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_cells": train_cells,
        "test_cells": test_cells,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": feature_cols,
        "target": "soh_percent",
        "linear_regression": {
            "metrics": metrics_lr,
            "coefficients": dict(zip(feature_cols, [round(float(c), 4) for c in lr.coef_])),
            "intercept": round(float(lr.intercept_), 4),
        },
        "random_forest": {
            "metrics": metrics_rf,
            "feature_importance": feat_importance,
        },
        "test_evaluation_df": pd.DataFrame({
            "cycle": X_test["cycle"].values,
            "actual_soh": y_test.values,
            "pred_lr": np.round(y_pred_lr, 2),
            "pred_rf": np.round(y_pred_rf, 2),
        }),
    }

    # 6. Save artifacts if requested
    if save_artifacts:
        os.makedirs(MODELS_DIR, exist_ok=True)
        lr_path = os.path.join(MODELS_DIR, "linear_regression_baseline.joblib")
        rf_path = os.path.join(MODELS_DIR, "random_forest_regressor.joblib")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

        joblib.dump(lr, lr_path)
        joblib.dump(rf, rf_path)

        # Write clean metadata JSON (exclude non-serializable DataFrame)
        meta_to_save = {
            "timestamp": results["timestamp"],
            "dataset_source": csv_path,
            "train_cells": train_cells,
            "test_cells": test_cells,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_columns": feature_cols,
            "target": "soh_percent",
            "linear_regression_metrics": metrics_lr,
            "random_forest_metrics": metrics_rf,
            "feature_importance": feat_importance,
            "disclaimer": (
                "Trained strictly on laboratory cell cycling experiments. "
                "Not validated for real laptop battery packs without transfer calibration."
            )
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=4)

        results["artifacts_saved"] = {
            "lr_model": lr_path,
            "rf_model": rf_path,
            "metadata": meta_path,
        }

    return results


if __name__ == "__main__":
    default_csv = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "sample_nasa_battery.csv"
    )
    print("=" * 60)
    print("Laptop Battery Health Sentinel - Machine Learning Training")
    print("=" * 60)
    print(f"Loading experimental dataset from: {default_csv}")

    res = train_and_evaluate_models(default_csv)

    print("\n--- Training Configuration ---")
    print(f"Train Cells: {res['train_cells']} ({res['train_samples']} samples)")
    print(f"Test Cells : {res['test_cells']} ({res['test_samples']} samples)")
    print(f"Features   : {res['feature_columns']}")

    print("\n--- Model 1: Baseline Linear Regression ---")
    print(f"  MAE : {res['linear_regression']['metrics']['MAE']:.4f}% SOH")
    print(f"  RMSE: {res['linear_regression']['metrics']['RMSE']:.4f}% SOH")
    print(f"  R^2 : {res['linear_regression']['metrics']['R2']:.4f}")

    print("\n--- Model 2: Random Forest Regressor ---")
    print(f"  MAE : {res['random_forest']['metrics']['MAE']:.4f}% SOH")
    print(f"  RMSE: {res['random_forest']['metrics']['RMSE']:.4f}% SOH")
    print(f"  R^2 : {res['random_forest']['metrics']['R2']:.4f}")

    print(f"\nArtifacts successfully saved to: {MODELS_DIR}")
    print("=" * 60)
