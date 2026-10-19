"""
test_soh_calculator.py
======================
Unit tests for the capacity-based SOH calculator.
"""

import math
import pytest
from src.soh_calculator import calculate_soh, validate_capacity_inputs


def test_calculate_soh_nominal():
    """Test standard nominal calculation (e.g. 45000 mWh FCC, 50000 mWh DC = 90%)."""
    result = calculate_soh(45000, 50000)
    assert result["is_valid"] is True
    assert result["status"] == "Success"
    assert result["soh_percent"] == 90.0
    assert result["unit"] == "mWh"


def test_calculate_soh_100_percent():
    """Test 100% healthy brand new battery."""
    result = calculate_soh(52000, 52000)
    assert result["is_valid"] is True
    assert result["soh_percent"] == 100.0


def test_calculate_soh_overcapacity_flag():
    """Test slight factory over-capacity (e.g. 102%)."""
    result = calculate_soh(51000, 50000)
    assert result["is_valid"] is True
    assert result["soh_percent"] == 102.0
    assert "factory over-capacity" in result["message"]


def test_calculate_soh_zero_fcc():
    """Test zero full charge capacity (completely dead capacity)."""
    result = calculate_soh(0, 50000)
    assert result["is_valid"] is True
    assert result["soh_percent"] == 0.0


def test_calculate_soh_zero_design_capacity_invalid():
    """Test zero design capacity (division by zero physical impossibility)."""
    result = calculate_soh(45000, 0)
    assert result["is_valid"] is False
    assert result["status"] == "InvalidDesignCapacity"
    assert result["soh_percent"] is None


def test_calculate_soh_negative_capacities():
    """Test negative numbers which are physically non-existent."""
    res1 = calculate_soh(-100, 50000)
    assert res1["is_valid"] is False
    assert res1["status"] == "InvalidFullChargeCapacity"

    res2 = calculate_soh(45000, -50000)
    assert res2["is_valid"] is False
    assert res2["status"] == "InvalidDesignCapacity"


def test_calculate_soh_missing_and_none():
    """Test None / missing values return clear errors without crashing."""
    result = calculate_soh(None, 50000)
    assert result["is_valid"] is False
    assert result["status"] == "MissingData"
    assert result["soh_percent"] is None


def test_calculate_soh_nan_inf():
    """Test math.nan and float('inf') are rejected cleanly."""
    res_nan = calculate_soh(float("nan"), 50000)
    assert res_nan["is_valid"] is False
    assert res_nan["status"] == "NonFiniteValue"

    res_inf = calculate_soh(45000, float("inf"))
    assert res_inf["is_valid"] is False
    assert res_inf["status"] == "NonFiniteValue"


def test_calculate_soh_non_numeric_strings():
    """Test arbitrary non-numeric text inputs are caught."""
    result = calculate_soh("unknown", 50000)
    assert result["is_valid"] is False
    assert result["status"] == "InvalidType"


def test_validate_capacity_inputs_helper():
    """Test the UI validation helper."""
    is_valid, msg = validate_capacity_inputs(40000, 50000)
    assert is_valid is True
    assert "80.0%" in msg

    is_valid_bad, msg_bad = validate_capacity_inputs(-10, 50000)
    assert is_valid_bad is False
