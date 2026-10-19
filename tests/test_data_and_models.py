"""
test_data_and_models.py
=======================
Unit tests for data loading, preprocessing, model training, and forecasting.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.data_loader import detect_data_source, load_dataset
from src.preprocessing import clean_lab_dataset, split_by_battery_cell, prepare_laptop_history_series
from src.train_model import train_and_evaluate_models, evaluate_model
from src.prediction import forecast_laptop_degradation, load_model_artifact


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SAMPLE_NASA = os.path.join(DATA_DIR, "sample_nasa_battery.csv")
SAMPLE_LAPTOP = os.path.join(DATA_DIR, "sample_laptop_history.csv")


def test_detect_data_source():
    """Verify schema detection distinguishes lab data from laptop data."""
    lab_df = pd.DataFrame({"battery_id": ["B0005"], "cycle": [1], "capacity_ah": [1.85]})
    assert detect_data_source(lab_df) == "LAB_EXPERIMENTAL"

    laptop_df = pd.DataFrame({"date": ["2024-01-01"], "full_charge_capacity_mwh": [50000], "design_capacity_mwh": [52000]})
    assert detect_data_source(laptop_df) == "LAPTOP_MONITORING"

    unknown_df = pd.DataFrame({"col_a": [1], "col_b": [2]})
    assert detect_data_source(unknown_df) == "UNKNOWN"


def test_load_dataset_sample_files():
    """Verify loading real sample CSV files."""
    res_nasa = load_dataset(SAMPLE_NASA)
    assert res_nasa["is_valid"] is True
    assert res_nasa["source_type"] == "LAB_EXPERIMENTAL"
    assert res_nasa["rows"] > 50

    res_laptop = load_dataset(SAMPLE_LAPTOP)
    assert res_laptop["is_valid"] is True
    assert res_laptop["source_type"] == "LAPTOP_MONITORING"
    assert res_laptop["rows"] == 16


def test_clean_lab_dataset_and_cell_split():
    """Verify clean_lab_dataset and zero-leakage cell-based splitting."""
    df = pd.read_csv(SAMPLE_NASA)
    cleaned, report = clean_lab_dataset(df)
    assert report["is_valid"] is True
    assert "soh_percent" in cleaned.columns

    X_train, y_train, X_test, y_test, feats = split_by_battery_cell(
        cleaned,
        train_cells=["B0005", "B0007"],
        test_cells=["B0006"]
    )
    # Check that test cells have ZERO overlap with train cells
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert "capacity_ah" not in feats  # Leakage prevention!
    assert "soh_percent" not in feats  # Leakage prevention!


def test_prepare_laptop_history_insufficient_data():
    """Fewer than 3 records must trigger insufficient historical data warning."""
    short_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-08"],
        "full_charge_capacity_mwh": [50000, 49500],
        "design_capacity_mwh": [52000, 52000]
    })
    is_valid, _, msg = prepare_laptop_history_series(short_df)
    assert is_valid is False
    assert "Insufficient historical data" in msg


def test_forecast_laptop_degradation():
    """Test degradation forecasting on sample laptop history."""
    df = pd.read_csv(SAMPLE_LAPTOP)
    res = forecast_laptop_degradation(df, future_steps=5, eol_threshold=80.0)
    assert res["is_forecastable"] is True
    assert res["current_soh"] is not None
    assert not res["forecast_df"].empty
    assert len(res["forecast_df"]) == 5
    assert "disclaimer" in res["disclaimer"].lower()
    assert res["degradation_rate_per_step"] < 0  # Battery capacity degrades


def test_train_and_evaluate_models():
    """Test training Linear Regression and Random Forest models."""
    res = train_and_evaluate_models(SAMPLE_NASA, save_artifacts=False)
    assert "linear_regression" in res
    assert "random_forest" in res
    assert res["linear_regression"]["metrics"]["MAE"] > 0
    assert res["random_forest"]["metrics"]["MAE"] > 0
