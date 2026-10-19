"""
soh_calculator.py
=================
Core module for calculating battery State of Health (SOH).

Formula:
    SOH (%) = (Full Charge Capacity / Design Capacity) * 100

Engineering Context:
--------------------
- Design Capacity: The original nominal energy or charge capacity specified
  by the battery manufacturer (typically in mWh or mAh).
- Full Charge Capacity: The current maximum capacity the battery can hold
  when 100% charged.
- Capacity-based SOH measures capacity degradation over cycles. It does NOT
  measure internal impedance, cell imbalance, or physical swelling.
"""

from typing import Dict, Any, Optional, Tuple
import math


def calculate_soh(
    full_charge_capacity: float,
    design_capacity: float,
    unit: str = "mWh"
) -> Dict[str, Any]:
    """
    Calculate capacity-based State of Health (SOH) percentage.

    Parameters
    ----------
    full_charge_capacity : float
        Maximum charge the battery can currently hold (positive float).
    design_capacity : float
        Factory-rated design capacity (positive float).
    unit : str, optional
        Unit of measurement ('mWh' or 'mAh'), default 'mWh'.

    Returns
    -------
    dict
        A dictionary containing:
        - 'soh_percent': float or None (rounded to 2 decimal places)
        - 'is_valid': bool (True if inputs are physically valid)
        - 'status': str ('Success' or error description)
        - 'message': str (Human-readable explanation for engineering students)
        - 'full_charge_capacity': float
        - 'design_capacity': float
        - 'unit': str

    Assumptions & Validations:
    -------------------------
    1. Both capacities must be positive real numbers.
    2. Capacities must share identical units (both mWh or both mAh).
    3. If SOH exceeds 100%, it is noted: brand-new batteries sometimes
       ship with slight factory over-capacity (e.g., 101% - 105%).
    4. If either capacity is missing, zero, or negative, an informative
       failure result is returned without inventing fake data.
    """
    # 1. Type and Null checks
    if full_charge_capacity is None or design_capacity is None:
        return {
            "soh_percent": None,
            "is_valid": False,
            "status": "MissingData",
            "message": "Both Full Charge Capacity and Design Capacity must be provided.",
            "full_charge_capacity": full_charge_capacity,
            "design_capacity": design_capacity,
            "unit": unit,
        }

    # 2. Convert to float and check for NaN or infinite values
    try:
        fcc = float(full_charge_capacity)
        dc = float(design_capacity)
    except (ValueError, TypeError):
        return {
            "soh_percent": None,
            "is_valid": False,
            "status": "InvalidType",
            "message": "Capacities must be valid numerical values.",
            "full_charge_capacity": full_charge_capacity,
            "design_capacity": design_capacity,
            "unit": unit,
        }

    if math.isnan(fcc) or math.isnan(dc) or math.isinf(fcc) or math.isinf(dc):
        return {
            "soh_percent": None,
            "is_valid": False,
            "status": "NonFiniteValue",
            "message": "Capacities cannot be NaN or Infinite.",
            "full_charge_capacity": fcc,
            "design_capacity": dc,
            "unit": unit,
        }

    # 3. Check for positive physical values
    if dc <= 0:
        return {
            "soh_percent": None,
            "is_valid": False,
            "status": "InvalidDesignCapacity",
            "message": f"Design Capacity must be strictly positive (received: {dc} {unit}).",
            "full_charge_capacity": fcc,
            "design_capacity": dc,
            "unit": unit,
        }

    if fcc < 0:
        return {
            "soh_percent": None,
            "is_valid": False,
            "status": "InvalidFullChargeCapacity",
            "message": f"Full Charge Capacity cannot be negative (received: {fcc} {unit}).",
            "full_charge_capacity": fcc,
            "design_capacity": dc,
            "unit": unit,
        }

    # 4. Compute SOH (%)
    soh = (fcc / dc) * 100.0
    soh_rounded = round(soh, 2)

    # 5. Engineering context notes
    if soh_rounded > 105.0:
        message = (
            f"Calculated SOH is {soh_rounded}%. Unusually high (>105%); "
            f"verify whether units match or if replacement battery specs were modified."
        )
    elif soh_rounded > 100.0:
        message = (
            f"Calculated SOH is {soh_rounded}%. Slight factory over-capacity is normal "
            f"for brand new battery cells."
        )
    else:
        message = f"Capacity SOH successfully estimated at {soh_rounded}%."

    return {
        "soh_percent": soh_rounded,
        "is_valid": True,
        "status": "Success",
        "message": message,
        "full_charge_capacity": fcc,
        "design_capacity": dc,
        "unit": unit,
    }


def validate_capacity_inputs(fcc: Any, dc: Any) -> Tuple[bool, str]:
    """
    Lightweight validation helper for UI forms and fast validation.
    """
    res = calculate_soh(fcc, dc)
    return res["is_valid"], res["message"]
