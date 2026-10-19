"""
data_loader.py
==============
Data loader and schema validator for battery datasets.

Supports:
1. NASA/CALCE Experimental Lab Battery Data (e.g., cell cycling datasets)
2. Windows Laptop Battery Monitoring History Logs (dated capacity records)
"""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd


# Column name aliases for flexibility
LAB_REQUIRED_COLUMNS = ["battery_id", "cycle", "capacity_ah"]
LAPTOP_REQUIRED_COLUMNS = ["date", "full_charge_capacity_mwh", "design_capacity_mwh"]


def detect_data_source(df: pd.DataFrame) -> str:
    """
    Detect whether a DataFrame represents experimental lab cell data or laptop monitoring logs.

    Returns:
    - 'LAB_EXPERIMENTAL'
    - 'LAPTOP_MONITORING'
    - 'UNKNOWN'
    """
    lower_cols = [c.lower().strip() for c in df.columns]

    has_cell = any("battery_id" in c or "cell" in c for c in lower_cols)
    has_cycle = any("cycle" in c for c in lower_cols)
    has_cap_ah = any("ah" in c or "capacity_ah" in c for c in lower_cols)

    if has_cell and has_cycle and has_cap_ah:
        return "LAB_EXPERIMENTAL"

    has_date = any("date" in c or "period" in c or "time" in c for c in lower_cols)
    has_fcc = any("full" in c and "charge" in c for c in lower_cols) or any("fcc" in c for c in lower_cols)
    has_dc = any("design" in c for c in lower_cols) or any("dc" in c for c in lower_cols)

    if (has_date and has_fcc) or (has_fcc and has_dc):
        return "LAPTOP_MONITORING"

    return "UNKNOWN"


def load_dataset(file_path_or_buffer: Any) -> Dict[str, Any]:
    """
    Load and validate a CSV dataset.

    Parameters
    ----------
    file_path_or_buffer : str or file-like object
        Path to CSV file or uploaded file stream (e.g., from Streamlit).

    Returns
    -------
    dict
        - 'is_valid': bool
        - 'source_type': 'LAB_EXPERIMENTAL' | 'LAPTOP_MONITORING' | 'UNKNOWN'
        - 'data': pd.DataFrame or None
        - 'message': str
        - 'rows': int
        - 'columns': list
    """
    try:
        if isinstance(file_path_or_buffer, str) and not os.path.exists(file_path_or_buffer):
            return {
                "is_valid": False,
                "source_type": "UNKNOWN",
                "data": None,
                "message": f"File does not exist: {file_path_or_buffer}",
                "rows": 0,
                "columns": [],
            }

        df = pd.read_csv(file_path_or_buffer)

        if df.empty:
            return {
                "is_valid": False,
                "source_type": "UNKNOWN",
                "data": None,
                "message": "The uploaded CSV file is empty.",
                "rows": 0,
                "columns": [],
            }

        # Normalize column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]
        source_type = detect_data_source(df)

        if source_type == "UNKNOWN":
            return {
                "is_valid": False,
                "source_type": "UNKNOWN",
                "data": df,
                "message": (
                    "Unrecognized column schema. For NASA lab data, expect columns: "
                    f"{LAB_REQUIRED_COLUMNS}. For laptop logs, expect: {LAPTOP_REQUIRED_COLUMNS}."
                ),
                "rows": len(df),
                "columns": list(df.columns),
            }

        return {
            "is_valid": True,
            "source_type": source_type,
            "data": df,
            "message": f"Successfully loaded {source_type} dataset with {len(df)} records.",
            "rows": len(df),
            "columns": list(df.columns),
        }

    except pd.errors.EmptyDataError:
        return {
            "is_valid": False,
            "source_type": "UNKNOWN",
            "data": None,
            "message": "The CSV file contains no data or is corrupted.",
            "rows": 0,
            "columns": [],
        }
    except Exception as e:
        return {
            "is_valid": False,
            "source_type": "UNKNOWN",
            "data": None,
            "message": f"Failed to load CSV: {str(e)}",
            "rows": 0,
            "columns": [],
        }
