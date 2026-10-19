"""
preprocessing.py
================
Data preprocessing, feature engineering, and leakage-free dataset splitting.

Machine Learning & Integrity Rules:
-----------------------------------
1. TARGET LEAKAGE PREVENTION:
   - Target is 'soh_percent'.
   - Capacity (e.g. 'capacity_ah' or 'full_charge_capacity') is NEVER used as an input
     feature when predicting SOH because SOH is directly calculated from capacity.
   - Future cycles or future timestamps are never leaked into historical features.
2. CELL-LEVEL SPLIT:
   - Data is split by physical battery cell identity (e.g., train on B0005 & B0007,
     test on B0006). Random cycle-level mixing is strictly avoided.
3. CANONICAL HISTORY NORMALIZATION:
   - Ensures all historical tables (Windows reports, SQLite, CSVs) expose consistent,
     interchangeable column names for both programmatic and display usage.
"""

from typing import Dict, Any, List, Tuple, Optional
import re
import pandas as pd
import numpy as np


CANONICAL_HISTORY_COLUMNS = [
    "Period",
    "timestamp",
    "date",
    "Full Charge Capacity (mWh)",
    "full_charge_capacity_mwh",
    "Design Capacity (mWh)",
    "design_capacity_mwh",
    "SOH (%)",
    "soh_percent",
    "cycle_count",
    "step_index"
]


def normalize_history_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Standardize historical battery observations into a canonical, consistent schema.

    Handles:
    - Empty DataFrames or None inputs (returns empty DataFrame with canonical columns).
    - Windows battery reports ('Period', 'Full Charge Capacity (mWh)', 'Design Capacity (mWh)', 'SOH (%)').
    - Database observations ('Timestamp', 'Design Cap (mWh)', 'Full Charge Cap (mWh)', 'SOH (%)', 'Cycle Count').
    - CSV logs ('date', 'full_charge_capacity_mwh', 'design_capacity_mwh', 'cycle_count').
    - Missing SOH: automatically calculated as (FCC / DC) * 100 when FCC and DC are positive.
    - Legacy schemas and inconsistent casings.

    Guarantees the returned DataFrame contains:
    - 'soh_percent' and 'SOH (%)'
    - 'full_charge_capacity_mwh' and 'Full Charge Capacity (mWh)'
    - 'design_capacity_mwh' and 'Design Capacity (mWh)'
    - 'Period', 'timestamp', and 'date'
    - 'cycle_count' (None if unavailable)
    - 'step_index' (0, 1, ..., N-1)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=CANONICAL_HISTORY_COLUMNS)

    src = df.copy()

    # Identify date / period / timestamp column
    col_map = {str(c).lower().strip(): c for c in src.columns}
    
    date_col_name = None
    for cand in ["period", "timestamp", "date", "time", "datetime"]:
        match = next((col_map[k] for k in col_map if cand in k), None)
        if match:
            date_col_name = match
            break

    # Identify Full Charge Capacity column
    fcc_col_name = None
    for cand in ["full charge capacity", "full_charge_capacity", "full charge cap", "fcc"]:
        match = next((col_map[k] for k in col_map if cand in k), None)
        if match:
            fcc_col_name = match
            break
    if not fcc_col_name:
        match = next((col_map[k] for k in col_map if "capacity" in k and "design" not in k and "nominal" not in k), None)
        if match:
            fcc_col_name = match

    # Identify Design Capacity column
    dc_col_name = None
    for cand in ["design capacity", "design_capacity", "design cap", "dc"]:
        match = next((col_map[k] for k in col_map if cand in k), None)
        if match:
            dc_col_name = match
            break

    # Identify SOH column
    soh_col_name = None
    for cand in ["soh (%)", "soh_percent", "soh", "state of health"]:
        match = next((col_map[k] for k in col_map if cand in k), None)
        if match:
            soh_col_name = match
            break

    # Identify Cycle Count column
    cyc_col_name = None
    for cand in ["cycle_count", "cycle count", "cycles", "cycle"]:
        match = next((col_map[k] for k in col_map if cand in k and "battery" not in k), None)
        if match:
            cyc_col_name = match
            break

    # Extract series safely
    if date_col_name:
        raw_dates = src[date_col_name].astype(str)
        # Clean ISO date if it's a period range like '2024-01-01 - 2024-01-08'
        period_series = raw_dates.apply(lambda p: re.search(r"\b\d{4}-\d{2}-\d{2}\b", str(p)).group(0) if re.search(r"\b\d{4}-\d{2}-\d{2}\b", str(p)) else str(p))
    else:
        period_series = pd.Series([f"Entry {i+1}" for i in range(len(src))], index=src.index)

    fcc_series = pd.to_numeric(src[fcc_col_name], errors="coerce") if fcc_col_name else pd.Series(np.nan, index=src.index)
    dc_series = pd.to_numeric(src[dc_col_name], errors="coerce") if dc_col_name else pd.Series(np.nan, index=src.index)
    soh_series = pd.to_numeric(src[soh_col_name], errors="coerce") if soh_col_name else pd.Series(np.nan, index=src.index)
    cyc_series = pd.to_numeric(src[cyc_col_name], errors="coerce") if cyc_col_name else pd.Series(np.nan, index=src.index)

    # Compute SOH if missing or null but FCC and DC are valid positive numbers
    calc_mask = (soh_series.isna() | (soh_series <= 0)) & (fcc_series > 0) & (dc_series > 0)
    soh_series.loc[calc_mask] = np.round((fcc_series.loc[calc_mask] / dc_series.loc[calc_mask]) * 100.0, 2)

    # If FCC or DC are missing but SOH is present, keep if possible, otherwise drop invalid rows
    valid_mask = (soh_series > 0) & (soh_series <= 150)
    if (fcc_series > 0).any():
        valid_mask = valid_mask & (fcc_series > 0)
    if (dc_series > 0).any():
        valid_mask = valid_mask & (dc_series > 0)

    clean_period = period_series[valid_mask].reset_index(drop=True)
    clean_fcc = fcc_series[valid_mask].reset_index(drop=True)
    clean_dc = dc_series[valid_mask].reset_index(drop=True)
    clean_soh = soh_series[valid_mask].round(2).reset_index(drop=True)
    clean_cyc = cyc_series[valid_mask].reset_index(drop=True)

    if len(clean_soh) == 0:
        return pd.DataFrame(columns=CANONICAL_HISTORY_COLUMNS)

    norm_df = pd.DataFrame()
    norm_df["Period"] = clean_period
    norm_df["timestamp"] = clean_period
    norm_df["date"] = clean_period
    norm_df["Full Charge Capacity (mWh)"] = clean_fcc
    norm_df["full_charge_capacity_mwh"] = clean_fcc
    norm_df["Design Capacity (mWh)"] = clean_dc
    norm_df["design_capacity_mwh"] = clean_dc
    norm_df["SOH (%)"] = clean_soh
    norm_df["soh_percent"] = clean_soh
    norm_df["cycle_count"] = clean_cyc
    norm_df["step_index"] = np.arange(len(norm_df))

    # Sort chronologically if possible
    try:
        norm_df["_dt_sort"] = pd.to_datetime(norm_df["timestamp"])
        norm_df = norm_df.sort_values(by="_dt_sort").drop(columns=["_dt_sort"]).reset_index(drop=True)
        norm_df["step_index"] = np.arange(len(norm_df))
    except Exception:
        pass

    return norm_df


def clean_lab_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and validate NASA/CALCE laboratory experimental battery data.

    Returns:
    --------
    (cleaned_df, report_dict)
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "dropped_rows": 0,
        "missing_values_handled": 0,
        "is_valid": True,
        "message": ""
    }

    cleaned = df.copy()

    # Standardize column names to lowercase
    cleaned.columns = [str(c).strip().lower() for c in cleaned.columns]

    # Map required columns
    required = ["battery_id", "cycle"]
    missing_req = [c for c in required if c not in cleaned.columns]
    if missing_req:
        report["is_valid"] = False
        report["message"] = f"Missing mandatory columns: {missing_req}"
        return cleaned, report

    # Find capacity column
    cap_col = next((c for c in cleaned.columns if "capacity" in c and "nominal" not in c), None)
    if not cap_col:
        report["is_valid"] = False
        report["message"] = "No discharge capacity column found (e.g., 'capacity_ah')."
        return cleaned, report

    # Clean capacity and cycle values
    cleaned["cycle"] = pd.to_numeric(cleaned["cycle"], errors="coerce")
    cleaned[cap_col] = pd.to_numeric(cleaned[cap_col], errors="coerce")

    # Drop NaNs in crucial columns
    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=["battery_id", "cycle", cap_col])
    # Keep strictly positive cycles and capacities
    cleaned = cleaned[(cleaned["cycle"] > 0) & (cleaned[cap_col] > 0)]
    report["dropped_rows"] = before_drop - len(cleaned)

    # Calculate SOH if not already present
    if "soh_percent" not in cleaned.columns:
        nom_col = next((c for c in cleaned.columns if "nominal" in c), None)
        if nom_col:
            cleaned["nominal_capacity_ah"] = pd.to_numeric(cleaned[nom_col], errors="coerce").fillna(2.0)
        else:
            # Assume 2.0 Ah for NASA 18650 cells
            cleaned["nominal_capacity_ah"] = 2.0

        cleaned["soh_percent"] = (cleaned[cap_col] / cleaned["nominal_capacity_ah"]) * 100.0

    # Sort by battery and cycle
    cleaned = cleaned.sort_values(by=["battery_id", "cycle"]).reset_index(drop=True)
    report["final_rows"] = len(cleaned)
    report["message"] = f"Cleaned {len(cleaned)} records across cells: {list(cleaned['battery_id'].unique())}."

    return cleaned, report


def split_by_battery_cell(
    df: pd.DataFrame,
    train_cells: Optional[List[str]] = None,
    test_cells: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "soh_percent"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Split experimental data by battery cell identity to prevent data leakage.
    """
    unique_cells = list(df["battery_id"].unique())

    if len(unique_cells) < 2:
        raise ValueError(
            f"Need at least 2 distinct battery cells for cell-based split. Found: {unique_cells}. "
            "Cannot evaluate out-of-cell generalization."
        )

    if train_cells is None or test_cells is None:
        test_cells = [unique_cells[1]]
        train_cells = [c for c in unique_cells if c not in test_cells]

    if feature_cols is None:
        candidate_features = ["cycle", "ambient_temperature"]
        feature_cols = [c for c in candidate_features if c in df.columns]
        if not feature_cols:
            feature_cols = ["cycle"]

    train_mask = df["battery_id"].isin(train_cells)
    test_mask = df["battery_id"].isin(test_cells)

    train_df = df[train_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    return X_train, y_train, X_test, y_test, feature_cols


def prepare_laptop_history_series(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, str]:
    """
    Clean, normalize, and structure dated laptop capacity history for forecasting.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, pd.DataFrame(columns=CANONICAL_HISTORY_COLUMNS), "Laptop history dataset is empty."

    norm_df = normalize_history_dataframe(df)

    if len(norm_df) == 0:
        return False, norm_df, "No valid capacity or SOH observations found in history."

    if len(norm_df) < 3:
        return False, norm_df, "Insufficient historical data (minimum 3 chronological records needed for degradation trend analysis)."

    return True, norm_df, f"Prepared {len(norm_df)} historical observations."
