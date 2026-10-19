"""
prediction.py
=============
Empirical degradation forecasting, statistical uncertainty modeling, and ML model inference.

Forecasting Principles:
-----------------------
1. Standard Li-ion End-of-Life (EOL) threshold is defined explicitly (default: 80% SOH).
2. RUL estimations are statistical linear extrapolations based on observed historical capacity logs.
   They are NOT guaranteed hardware replacement dates.
3. If historical observations are insufficient (< 3 data points), forecasting is halted
   and an 'Insufficient historical data for a reliable forecast.' notice is returned.
4. Confidence envelopes are computed directly from sample residual standard errors (SE).
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_EOL_THRESHOLD = 80.0  # Industry standard End-of-Life threshold (80% SOH)


def load_model_artifact(model_filename: str = "random_forest_regressor.joblib") -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Safely load a trained model and its metadata from disk.
    """
    model_path = os.path.join(MODELS_DIR, model_filename)
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    metadata = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    if not os.path.exists(model_path):
        return None, {
            "loaded": False,
            "message": f"Model artifact not found at {model_path}. System operating in Monitoring-Only mode.",
            **metadata
        }

    try:
        model = joblib.load(model_path)
        return model, {
            "loaded": True,
            "model_path": model_path,
            "message": f"Successfully loaded model from {os.path.basename(model_path)}.",
            **metadata
        }
    except Exception as e:
        return None, {
            "loaded": False,
            "message": f"Error loading model: {str(e)}. Operating in Monitoring-Only mode.",
            **metadata
        }


def forecast_laptop_degradation(
    history_df: pd.DataFrame,
    future_steps: int = 10,
    eol_threshold: float = DEFAULT_EOL_THRESHOLD
) -> Dict[str, Any]:
    """
    Forecast future battery health trends and estimate RUL with statistical uncertainty envelopes.

    Parameters
    ----------
    history_df : pd.DataFrame
        DataFrame containing chronological columns: 'soh_percent' and either 'cycle_count' or sequential steps.
    future_steps : int
        Number of future observation intervals/cycles to project forward.
    eol_threshold : float
        End-of-life SOH percentage threshold (default: 80.0%).

    Returns
    -------
    dict
        - 'is_forecastable': bool
        - 'message': str
        - 'current_soh': float
        - 'eol_threshold': float
        - 'forecast_df': pd.DataFrame (containing step/cycle, predicted SOH, upper and lower bounds)
        - 'estimated_steps_to_eol': int or None
        - 'degradation_rate_per_step': float (% SOH drop per interval)
        - 'residual_std_error': float
        - 'disclaimer': str
    """
    disclaimer = (
        "Statistical Forecast Disclaimer: Degradation projections and Remaining Useful Life (RUL) estimates "
        "are empirical linear extrapolations based on observed historical capacity logs. "
        "They do NOT represent guaranteed hardware replacement timelines or manufacturer warranties."
    )

    insufficient_msg = "Insufficient historical data for a reliable forecast (minimum 3 chronological records needed)."

    from src.preprocessing import normalize_history_dataframe
    df = normalize_history_dataframe(history_df)

    if df.empty or len(df) < 3:
        return {
            "is_forecastable": False,
            "message": insufficient_msg,
            "current_soh": None,
            "eol_threshold": eol_threshold,
            "forecast_df": pd.DataFrame(),
            "estimated_steps_to_eol": None,
            "degradation_rate_per_step": None,
            "residual_std_error": None,
            "disclaimer": disclaimer,
        }

    # X axis: cycle_count if present and strictly increasing, otherwise step index
    use_cycle = False
    if "cycle_count" in df.columns and df["cycle_count"].notna().all() and (df["cycle_count"] > 0).all():
        x = df["cycle_count"].values.astype(float)
        x_label = "Cycle Count"
        use_cycle = True
    else:
        x = np.arange(len(df), dtype=float)
        x_label = "Observation Step"

    y = df["soh_percent"].values.astype(float)
    n = len(x)
    current_soh = round(float(y[-1]), 2)

    # Fit linear regression model
    X_mat = x.reshape(-1, 1)
    trend_model = LinearRegression()
    trend_model.fit(X_mat, y)

    slope = float(trend_model.coef_[0])
    intercept = float(trend_model.intercept_)

    # Calculate residual standard error (s_e)
    y_fitted = trend_model.predict(X_mat)
    residuals = y - y_fitted
    rss = np.sum(residuals ** 2)
    s_e = float(np.sqrt(rss / max(1, (n - 2)))) if n > 2 else 0.5
    x_mean = float(np.mean(x))
    ss_x = float(np.sum((x - x_mean) ** 2))
    if ss_x == 0:
        ss_x = 1.0

    # Generate future projections
    last_x = float(x[-1])
    step_diffs = np.diff(x)
    step_size = float(np.mean(step_diffs)) if len(step_diffs) > 0 and np.mean(step_diffs) > 0 else 1.0

    future_x_vals = [last_x + (i + 1) * step_size for i in range(future_steps)]
    future_x_arr = np.array(future_x_vals).reshape(-1, 1)
    future_preds = trend_model.predict(future_x_arr)

    forecast_rows = []
    for fx, fpred in zip(future_x_vals, future_preds):
        # Genuine standard error of forecast at point fx
        se_forecast = s_e * np.sqrt(1.0 + (1.0 / n) + ((fx - x_mean) ** 2) / ss_x)
        # 95% approximate confidence envelope (+/- 1.96 * se)
        upper_bound = min(105.0, float(fpred + 1.96 * se_forecast))
        lower_bound = max(0.0, float(fpred - 1.96 * se_forecast))
        clipped_pred = max(0.0, min(105.0, float(fpred)))

        forecast_rows.append({
            x_label: round(fx, 1) if not use_cycle else int(round(fx)),
            "Projected SOH (%)": round(clipped_pred, 2),
            "Lower Bound (95%)": round(lower_bound, 2),
            "Upper Bound (95%)": round(upper_bound, 2),
            "Status": "Normal" if clipped_pred >= 90.0 else ("Caution" if clipped_pred >= eol_threshold else "Critical (EOL)")
        })

    forecast_df = pd.DataFrame(forecast_rows)

    # Calculate remaining steps/cycles to EOL
    steps_to_eol = None
    if current_soh <= eol_threshold:
        steps_to_eol = 0
    elif slope < -1e-4:
        delta_x = (eol_threshold - current_soh) / slope
        steps_to_eol = max(0, int(round(delta_x / step_size)))

    return {
        "is_forecastable": True,
        "message": "Degradation trend successfully computed with residual uncertainty bounds.",
        "current_soh": current_soh,
        "eol_threshold": eol_threshold,
        "x_label": x_label,
        "forecast_df": forecast_df,
        "estimated_steps_to_eol": steps_to_eol,
        "degradation_rate_per_step": round(slope, 4),
        "residual_std_error": round(s_e, 4),
        "disclaimer": disclaimer,
    }
