"""
test_windows_battery_report.py
==============================
Unit tests for parsing Windows powercfg battery report HTML and live WMI querying.
"""

from src.windows_battery_report import parse_battery_report_html, _clean_number, get_live_battery_status

SAMPLE_REPORT_HTML = """
<!DOCTYPE html>
<html>
<head><title>Battery Report</title></head>
<body>
<h1>Battery report</h1>
<table>
    <tr><td>COMPUTER NAME</td><td>DESKTOP-ABC1234</td></tr>
    <tr><td>SYSTEM PRODUCT NAME</td><td>Dell Inc. Latitude 7420</td></tr>
    <tr><td>REPORT TIME</td><td>2026-05-10 14:30:00</td></tr>
</table>
<h2>Installed batteries</h2>
<table>
    <tr><td>NAME</td><td>L19C4PC1</td></tr>
    <tr><td>MANUFACTURER</td><td>Celxpert</td></tr>
    <tr><td>CHEMISTRY</td><td>LiP</td></tr>
    <tr><td>DESIGN CAPACITY</td><td>52,500 mWh</td></tr>
    <tr><td>FULL CHARGE CAPACITY</td><td>47,250 mWh</td></tr>
    <tr><td>CYCLE COUNT</td><td>115</td></tr>
</table>
<h2>Battery capacity history</h2>
<table>
    <tr><th>PERIOD</th><th>FULL CHARGE CAPACITY</th><th>DESIGN CAPACITY</th></tr>
    <tr><td>2024-01-01 - 2024-01-08</td><td>52,000 mWh</td><td>52,500 mWh</td></tr>
    <tr><td>2024-01-08 - 2024-01-15</td><td>50,500 mWh</td><td>52,500 mWh</td></tr>
    <tr><td>2024-01-15 - 2024-01-22</td><td>48,000 mWh</td><td>52,500 mWh</td></tr>
    <tr><td>2024-01-22 - 2024-01-29</td><td>47,250 mWh</td><td>52,500 mWh</td></tr>
</table>
</body>
</html>
"""

EMPTY_REPORT_HTML = """
<!DOCTYPE html>
<html>
<body>
<h1>Battery report</h1>
<p>No batteries currently installed.</p>
</body>
</html>
"""


def test_clean_number():
    """Verify number cleanup from Windows report formats."""
    assert _clean_number("52,500 mWh") == 52500.0
    assert _clean_number("48 120 mWh") == 48120.0
    assert _clean_number("115") == 115.0
    assert _clean_number("-") is None
    assert _clean_number("N/A") is None


def test_parse_valid_battery_report():
    """Test parsing a valid Windows battery report HTML with installed battery and history."""
    result = parse_battery_report_html(SAMPLE_REPORT_HTML)
    assert result["success"] is True
    info = result["battery_info"]
    assert info["design_capacity"] == 52500.0
    assert info["full_charge_capacity"] == 47250.0
    assert info["cycle_count"] == 115
    # Battery name should NOT be confused with computer name
    assert info["name"] == "L19C4PC1"
    assert info["manufacturer"] == "Celxpert"

    sys_info = result["system_info"]
    assert sys_info["computer_name"] == "DESKTOP-ABC1234"
    assert "Latitude" in sys_info["product_name"]

    history_df = result["history_records"]
    assert not history_df.empty
    assert len(history_df) == 4
    assert "SOH (%)" in history_df.columns
    assert round(history_df.iloc[0]["SOH (%)"], 1) == 99.0


def test_parse_empty_battery_report():
    """Test desktop PC without physical battery handles gracefully."""
    result = parse_battery_report_html(EMPTY_REPORT_HTML)
    assert result["success"] is False
    assert "No battery information found" in result["error"]


def test_get_live_battery_status_structure():
    """Test live status returns valid dictionary schema."""
    res = get_live_battery_status()
    assert isinstance(res, dict)
    assert "is_available" in res
    assert "charge_percent" in res
    assert "status_text" in res
    assert "timestamp" in res
