"""
test_alerts.py
==============
Unit tests for the alerts module, threshold classifications, and multi-factor diagnostics.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.alerts import (
    evaluate_soh_alert,
    build_alert_history_entry,
    check_data_freshness,
    detect_rapid_degradation,
    generate_system_alerts
)


def test_evaluate_soh_alert_normal():
    """SOH >= 90% should yield NORMAL severity."""
    alert = evaluate_soh_alert(95.5)
    assert alert["severity"] == "NORMAL"
    assert "Good" in alert["title"]
    assert "disclaimer" in alert["safety_disclaimer"].lower()


def test_evaluate_soh_alert_caution():
    """80% <= SOH < 90% should yield CAUTION severity."""
    alert = evaluate_soh_alert(85.0)
    assert alert["severity"] == "CAUTION"
    assert "Degrading" in alert["title"]


def test_evaluate_soh_alert_critical():
    """SOH < 80% should yield CRITICAL severity (Li-ion EOL)."""
    alert = evaluate_soh_alert(76.2)
    assert alert["severity"] == "CRITICAL"
    assert "End of Life" in alert["title"]


def test_evaluate_soh_alert_custom_thresholds():
    """Custom thresholds should take precedence."""
    alert1 = evaluate_soh_alert(86.0, caution_threshold=85.0, critical_threshold=70.0)
    assert alert1["severity"] == "NORMAL"

    alert2 = evaluate_soh_alert(75.0, caution_threshold=85.0, critical_threshold=70.0)
    assert alert2["severity"] == "CAUTION"

    alert3 = evaluate_soh_alert(68.0, caution_threshold=85.0, critical_threshold=70.0)
    assert alert3["severity"] == "CRITICAL"


def test_evaluate_soh_alert_none_input():
    """None input returns UNKNOWN severity."""
    alert = evaluate_soh_alert(None)
    assert alert["severity"] == "UNKNOWN"
    assert alert["soh_percent"] is None


def test_build_alert_history_entry():
    """Test generating a structured row for history table."""
    row = build_alert_history_entry("2024-05-10", 88.5, 44250, 50000)
    assert row["Date"] == "2024-05-10"
    assert row["SOH (%)"] == 88.5
    assert row["Severity"] == "CAUTION"
    assert row["Full Charge (mWh)"] == 44250.0


def test_check_data_freshness():
    """Test stale data detection (>14 days)."""
    fresh_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    assert check_data_freshness(fresh_date, max_stale_days=14) is None

    stale_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")
    alert = check_data_freshness(stale_date, max_stale_days=14)
    assert alert is not None
    assert alert["severity"] == "ADVISORY"
    assert "stale" in alert["title"].lower()


def test_detect_rapid_degradation():
    """Test rapid capacity drop across recent observations."""
    # Gradual fade
    df_normal = pd.DataFrame({"soh_percent": [95.0, 94.5, 94.2, 93.9]})
    assert detect_rapid_degradation(df_normal, drop_threshold_pct=3.0) is None

    # Steep drop (-4% in 4 steps)
    df_steep = pd.DataFrame({"soh_percent": [94.0, 92.5, 91.0, 89.8]})
    alert = detect_rapid_degradation(df_steep, drop_threshold_pct=3.0)
    assert alert is not None
    assert alert["severity"] == "WARNING"
    assert "accelerated" in alert["title"].lower()


def test_generate_system_alerts_integration():
    """Test multi-factor alert generator."""
    alerts = generate_system_alerts(
        current_soh=84.5,
        current_fcc=42250,
        current_dc=50000,
        last_scan_date="2020-01-01"  # Old date triggers freshness
    )
    # Expect at least Caution SOH and Stale data alerts
    severities = [a["severity"] for a in alerts]
    assert "CAUTION" in severities
    assert "ADVISORY" in severities


def test_alerts_low_charge_and_critical():
    """Verify low charge warnings and critical charge alarms while discharging."""
    alerts_low = generate_system_alerts(
        current_soh=95.0,
        current_fcc=50000,
        current_dc=50000,
        charge_percent=12.0,
        power_status="Discharging"
    )
    categories = [a["category"] for a in alerts_low]
    assert "LOW_CHARGE" in categories

    alerts_crit = generate_system_alerts(
        current_soh=95.0,
        current_fcc=50000,
        current_dc=50000,
        charge_percent=4.0,
        power_status="Discharging"
    )
    categories_crit = [a["category"] for a in alerts_crit]
    assert "CRITICAL_CHARGE" in categories_crit


def test_alerts_high_discharge_rate():
    """Verify elevated power drain alert (>35W)."""
    alerts = generate_system_alerts(
        current_soh=95.0,
        current_fcc=50000,
        current_dc=50000,
        discharge_rate_w=42.5
    )
    categories = [a["category"] for a in alerts]
    assert "HIGH_POWER_DRAW" in categories


def test_alerts_temperature_thresholds():
    """Verify warning (>45C) and critical (>55C) temperature alerts."""
    alerts_warn = generate_system_alerts(
        current_soh=95.0,
        current_fcc=50000,
        current_dc=50000,
        temperature_c=48.0
    )
    assert any(a["category"] == "HIGH_TEMPERATURE" for a in alerts_warn)

    alerts_crit = generate_system_alerts(
        current_soh=95.0,
        current_fcc=50000,
        current_dc=50000,
        temperature_c=58.0
    )
    assert any(a["category"] == "CRITICAL_TEMPERATURE" for a in alerts_crit)


def test_alerts_voltage_sag():
    """Verify voltage sag warning when live voltage is >15% below nominal."""
    alerts = generate_system_alerts(
        current_soh=95.0,
        current_fcc=50000,
        current_dc=50000,
        charge_percent=60.0,
        live_voltage_v=9.2,
        design_voltage_v=11.4
    )
    assert any(a["category"] == "VOLTAGE_SAG" for a in alerts)

