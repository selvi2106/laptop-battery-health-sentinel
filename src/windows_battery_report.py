"""
windows_battery_report.py
=========================
Hardware diagnostics interface for Microsoft Windows laptops.

Capabilities:
-------------
1. Live Battery Telemetry:
   Retrieves real-time charge percentage, discharging/charging status,
   estimated remaining runtime, and battery device ID via Windows WMI/CIM.
2. Battery Lifecycle Report (powercfg /batteryreport):
   Executes non-elevated report generation and robustly parses:
   - System product name and OS details
   - Installed battery specifications (Manufacturer, Chemistry, Design Capacity, Full Charge Capacity, Cycles)
   - Longitudinal battery capacity history table (periodic capacity degradation)
3. Safety & Privacy:
   Strictly read-only access. Never transmits telemetry outside the local host.
   Gracefully handles desktops or virtual machines lacking physical battery hardware.
"""

import os
import re
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from html.parser import HTMLParser
import pandas as pd


# WMI Win32_Battery BatteryStatus mapping
BATTERY_STATUS_MAP = {
    1: "Discharging",
    2: "AC Connected (Unknown State)",
    3: "Fully Charged",
    4: "Low Battery",
    5: "Critical Battery",
    6: "Charging",
    7: "Charging (High)",
    8: "Charging (Low)",
    9: "Charging (Critical)",
    10: "Undefined",
    11: "Partially Charged"
}


def get_live_battery_status() -> Dict[str, Any]:
    """
    Retrieve real-time live battery status from the operating system.

    Queries Windows CIM/WMI (Win32_Battery, Win32_PortableBattery, and root/wmi:BatteryStatus)
    via safe, non-elevated read-only queries.

    Returns
    -------
    dict
        - 'is_available': bool (True if a physical battery was queried)
        - 'charge_percent': int or None (e.g., 72%)
        - 'status_code': int or None
        - 'status_text': str ('Discharging', 'Charging', 'AC Connected', etc.)
        - 'estimated_runtime_minutes': int or None (e.g., 111 min)
        - 'device_id': str or None
        - 'device_name': str or None
        - 'manufacturer': str or None (e.g., 'BYD')
        - 'chemistry': str or None (e.g., 'Li-Ion')
        - 'design_voltage_v': float or None (e.g., 11.35 V)
        - 'live_voltage_v': float or None (e.g., 11.20 V)
        - 'discharge_rate_w': float or None (e.g., 14.8 W)
        - 'charge_rate_w': float or None
        - 'power_online': bool or None (True if plugged into AC)
        - 'temperature_c': float or None (None if not exposed by hardware ACPI)
        - 'temperature_status': str
        - 'cycle_count': int or None
        - 'timestamp': str (ISO format)
        - 'error_message': str or None
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.name != "nt":
        return {
            "is_available": False,
            "charge_percent": None,
            "status_code": None,
            "status_text": "Unsupported OS (Windows required for live telemetry)",
            "estimated_runtime_minutes": None,
            "device_id": None,
            "device_name": None,
            "manufacturer": None,
            "chemistry": None,
            "design_voltage_v": None,
            "live_voltage_v": None,
            "discharge_rate_w": None,
            "charge_rate_w": None,
            "power_online": None,
            "temperature_c": None,
            "temperature_status": "Unavailable on non-Windows OS",
            "cycle_count": None,
            "timestamp": ts,
            "error_message": "Operating system is not Microsoft Windows."
        }

    # Safe PowerShell script querying Win32_Battery, Win32_PortableBattery, and root/wmi:BatteryStatus
    ps_script = (
        "$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1; "
        "$w = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1; "
        "$pb = Get-CimInstance Win32_PortableBattery -ErrorAction SilentlyContinue | Select-Object -First 1; "
        "if ($b) { "
        "  [PSCustomObject]@{ "
        "    Present = 1; "
        "    Charge = $b.EstimatedChargeRemaining; "
        "    Status = $b.BatteryStatus; "
        "    Runtime = $b.EstimatedRunTime; "
        "    DeviceID = $b.DeviceID; "
        "    Name = $b.Name; "
        "    DesignVoltage_mV = $b.DesignVoltage; "
        "    LiveVoltage_mV = if ($w) { $w.Voltage } else { $null }; "
        "    DischargeRate_mW = if ($w) { $w.DischargeRate } else { $null }; "
        "    ChargeRate_mW = if ($w) { $w.ChargeRate } else { $null }; "
        "    PowerOnline = if ($w) { [int]$w.PowerOnline } else { $null }; "
        "    Manufacturer = if ($pb) { $pb.Manufacturer } else { $null }; "
        "    Chemistry = if ($b.Chemistry) { $b.Chemistry } else { if ($pb) { $pb.Chemistry } else { $null } } "
        "  } | ConvertTo-Json -Compress "
        "} else { "
        "  [PSCustomObject]@{ Present = 0 } | ConvertTo-Json -Compress "
        "}"
    )

    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=8,
            check=False
        )

        output = proc.stdout.strip()
        if not output:
            return {
                "is_available": False,
                "charge_percent": None,
                "status_code": None,
                "status_text": "No Battery Detected (Desktop or VM)",
                "estimated_runtime_minutes": None,
                "device_id": None,
                "device_name": None,
                "manufacturer": None,
                "chemistry": None,
                "design_voltage_v": None,
                "live_voltage_v": None,
                "discharge_rate_w": None,
                "charge_rate_w": None,
                "power_online": None,
                "temperature_c": None,
                "temperature_status": "Unavailable",
                "cycle_count": None,
                "timestamp": ts,
                "error_message": "System reported no physical Win32_Battery instance."
            }

        import json
        data = json.loads(output)

        if not data.get("Present", 0):
            return {
                "is_available": False,
                "charge_percent": None,
                "status_code": None,
                "status_text": "No Physical Battery Detected",
                "estimated_runtime_minutes": None,
                "device_id": None,
                "device_name": None,
                "manufacturer": None,
                "chemistry": None,
                "design_voltage_v": None,
                "live_voltage_v": None,
                "discharge_rate_w": None,
                "charge_rate_w": None,
                "power_online": None,
                "temperature_c": None,
                "temperature_status": "Unavailable",
                "cycle_count": None,
                "timestamp": ts,
                "error_message": "Desktop computer or AC-only power detected."
            }

        status_code = data.get("Status")
        status_text = BATTERY_STATUS_MAP.get(status_code, "Online")
        charge = data.get("Charge")
        runtime = data.get("Runtime")
        # 71582788 minutes is a known Windows 32-bit sentinel value for 'Unknown/Charging'
        if runtime is not None and (runtime <= 0 or runtime > 100000):
            runtime = None

        # Format voltages (mV -> V)
        design_v_raw = data.get("DesignVoltage_mV")
        design_v = round(design_v_raw / 1000.0, 3) if design_v_raw and design_v_raw > 0 else None

        live_v_raw = data.get("LiveVoltage_mV")
        live_v = round(live_v_raw / 1000.0, 3) if live_v_raw and live_v_raw > 0 else design_v

        # Format rates (mW -> W)
        dis_rate_raw = data.get("DischargeRate_mW")
        dis_rate_w = round(dis_rate_raw / 1000.0, 2) if dis_rate_raw and dis_rate_raw > 0 else None

        chg_rate_raw = data.get("ChargeRate_mW")
        chg_rate_w = round(chg_rate_raw / 1000.0, 2) if chg_rate_raw and chg_rate_raw > 0 else None

        p_online_raw = data.get("PowerOnline")
        power_online = bool(p_online_raw) if p_online_raw is not None else (status_code in (2, 3, 6, 7, 8, 9))

        # Chemistry mapping: 1=Other, 2=Unknown/Li-Ion, 3=Lead Acid, 4=NiCd, 5=NiMH, 6=Li-Ion, 7=Zinc, 8=LiP
        chem_code = data.get("Chemistry")
        chem_map = {1: "Other", 2: "Li-Ion", 3: "Lead Acid", 4: "NiCd", 5: "NiMH", 6: "Li-Ion", 7: "Zinc Air", 8: "Li-Polymer"}
        chem_str = chem_map.get(chem_code, "Li-Ion") if chem_code else "Li-Ion"

        mfg = data.get("Manufacturer") or "OEM"

        return {
            "is_available": True,
            "charge_percent": int(charge) if charge is not None else None,
            "status_code": status_code,
            "status_text": status_text,
            "estimated_runtime_minutes": int(runtime) if runtime is not None else None,
            "device_id": data.get("DeviceID"),
            "device_name": data.get("Name"),
            "manufacturer": mfg,
            "chemistry": chem_str,
            "design_voltage_v": design_v,
            "live_voltage_v": live_v,
            "discharge_rate_w": dis_rate_w,
            "charge_rate_w": chg_rate_w,
            "power_online": power_online,
            "temperature_c": None,  # Handled honestly: not reported by standard consumer ACPI WMI
            "temperature_status": "Not reported by hardware/ACPI",
            "cycle_count": None,
            "timestamp": ts,
            "error_message": None
        }

    except subprocess.TimeoutExpired:
        return {
            "is_available": False,
            "charge_percent": None,
            "status_code": None,
            "status_text": "Query Timeout",
            "estimated_runtime_minutes": None,
            "device_id": None,
            "device_name": None,
            "manufacturer": None,
            "chemistry": None,
            "design_voltage_v": None,
            "live_voltage_v": None,
            "discharge_rate_w": None,
            "charge_rate_w": None,
            "power_online": None,
            "temperature_c": None,
            "temperature_status": "Unavailable",
            "cycle_count": None,
            "timestamp": ts,
            "error_message": "PowerShell battery query timed out."
        }
    except Exception as e:
        return {
            "is_available": False,
            "charge_percent": None,
            "status_code": None,
            "status_text": "Query Failed",
            "estimated_runtime_minutes": None,
            "device_id": None,
            "device_name": None,
            "manufacturer": None,
            "chemistry": None,
            "design_voltage_v": None,
            "live_voltage_v": None,
            "discharge_rate_w": None,
            "charge_rate_w": None,
            "power_online": None,
            "temperature_c": None,
            "temperature_status": "Unavailable",
            "cycle_count": None,
            "timestamp": ts,
            "error_message": str(e)
        }


def generate_battery_report(output_file_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Execute 'powercfg /batteryreport' to generate the official Windows battery report.
    """
    if os.name != "nt":
        return (
            False,
            "The 'powercfg' utility is only available on Microsoft Windows operating systems."
        )

    if output_file_path is None:
        temp_dir = tempfile.gettempdir()
        output_file_path = os.path.join(temp_dir, "sentinel_battery_report.html")

    try:
        cmd = ["powercfg", "/batteryreport", "/output", output_file_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15
        )

        if result.returncode == 0 and os.path.exists(output_file_path):
            return True, output_file_path
        else:
            err_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error generating report."
            return False, f"Failed to generate battery report: {err_msg}"

    except FileNotFoundError:
        return False, "'powercfg' utility was not found in system PATH."
    except Exception as e:
        return False, f"Unexpected error while generating battery report: {str(e)}"


class BatteryReportHTMLParser(HTMLParser):
    """HTML parser extracting table contents from powercfg reports."""
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row: List[str] = []
        self.current_cell: List[str] = []
        self.tables: List[List[List[str]]] = []
        self.current_table: List[List[str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.current_table = []
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            cell_text = " ".join("".join(self.current_cell).split()).strip()
            self.current_row.append(cell_text)
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_row:
                self.current_table.append(self.current_row)
        elif tag == "table" and self.in_table:
            self.in_table = False
            if self.current_table:
                self.tables.append(self.current_table)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def _clean_number(val_str: str) -> Optional[float]:
    """Extract float from strings like '52,500 mWh', '48 120 mWh', '142'."""
    if not val_str or val_str.strip() in ("-", "--", "N/A", "n/a"):
        return None
    cleaned = re.sub(r"[^\d.]", "", val_str.replace(",", "").replace(" ", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def parse_battery_report_html(html_text: str) -> Dict[str, Any]:
    """
    Parse Windows battery report HTML content.

    Extracts:
    - System metadata (Computer Name, System Product, Report Generation Time)
    - Installed battery specifications (Battery Name, Manufacturer, Chemistry, Design Capacity, Full Charge Capacity, Cycles)
    - Historical capacity log table (Period, Full Charge Capacity, Design Capacity, SOH)
    """
    parser = BatteryReportHTMLParser()
    try:
        parser.feed(html_text)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse HTML format: {str(e)}",
            "system_info": {},
            "battery_info": {},
            "history_records": pd.DataFrame(),
        }

    system_info: Dict[str, Any] = {
        "computer_name": None,
        "product_name": None,
        "bios_version": None,
        "report_time": None
    }

    battery_info: Dict[str, Any] = {
        "name": None,
        "manufacturer": None,
        "chemistry": None,
        "serial_number": None,
        "design_capacity": None,
        "full_charge_capacity": None,
        "cycle_count": None,
        "unit": "mWh",
    }

    history_rows: List[Dict[str, Any]] = []

    for table in parser.tables:
        # Check if table contains System Info
        for row in table:
            if len(row) >= 2:
                label = row[0].upper().strip()
                val = row[1].strip()

                # System details
                if label == "COMPUTER NAME":
                    system_info["computer_name"] = val
                elif label in ("SYSTEM PRODUCT NAME", "PRODUCT NAME"):
                    system_info["product_name"] = val
                elif label == "BIOS":
                    system_info["bios_version"] = val
                elif label == "REPORT TIME":
                    system_info["report_time"] = val

                # Battery details (Ensure not confused with Computer Name)
                if "DESIGN CAPACITY" in label and battery_info["design_capacity"] is None:
                    battery_info["design_capacity"] = _clean_number(val)
                elif "FULL CHARGE CAPACITY" in label and battery_info["full_charge_capacity"] is None:
                    battery_info["full_charge_capacity"] = _clean_number(val)
                elif "CYCLE COUNT" in label and battery_info["cycle_count"] is None:
                    num = _clean_number(val)
                    battery_info["cycle_count"] = int(num) if num is not None else None
                elif (label == "NAME" or label == "BATTERY NAME") and battery_info["name"] is None:
                    # Ignore if it is COMPUTER NAME
                    if "COMPUTER" not in label:
                        battery_info["name"] = val
                elif "MANUFACTURER" in label and battery_info["manufacturer"] is None:
                    battery_info["manufacturer"] = val
                elif "CHEMISTRY" in label and battery_info["chemistry"] is None:
                    battery_info["chemistry"] = val
                elif "SERIAL NUMBER" in label and battery_info["serial_number"] is None:
                    battery_info["serial_number"] = val

        # Check if this table is the "Battery capacity history" table
        if len(table) > 1:
            header = [c.upper() for c in table[0]]
            is_history = any("PERIOD" in c or "START" in c for c in header) and any("FULL CHARGE" in c for c in header)
            if is_history:
                col_period = next((i for i, c in enumerate(header) if "PERIOD" in c or "START" in c), 0)
                col_fcc = next((i for i, c in enumerate(header) if "FULL CHARGE" in c), 1)
                col_dc = next((i for i, c in enumerate(header) if "DESIGN" in c), 2)

                for data_row in table[1:]:
                    if len(data_row) > max(col_period, col_fcc, col_dc):
                        p_str = data_row[col_period]
                        fcc_val = _clean_number(data_row[col_fcc])
                        dc_val = _clean_number(data_row[col_dc])

                        if fcc_val and dc_val and dc_val > 0:
                            soh = round((fcc_val / dc_val) * 100.0, 2)
                            date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", p_str)
                            clean_date = date_match.group(0) if date_match else p_str

                            history_rows.append({
                                "Period": clean_date,
                                "Full Charge Capacity (mWh)": fcc_val,
                                "Design Capacity (mWh)": dc_val,
                                "SOH (%)": soh,
                            })

    dc = battery_info.get("design_capacity")
    fcc = battery_info.get("full_charge_capacity")

    if dc is None and fcc is None and not history_rows:
        return {
            "success": False,
            "error": (
                "No battery information found in this report. This occurs on desktop computers "
                "without a physical battery, or if the report is incomplete."
            ),
            "system_info": system_info,
            "battery_info": battery_info,
            "history_records": pd.DataFrame(),
        }

    from src.preprocessing import normalize_history_dataframe
    history_df = pd.DataFrame(history_rows)
    history_df = normalize_history_dataframe(history_df)

    return {
        "success": True,
        "error": None,
        "system_info": system_info,
        "battery_info": battery_info,
        "history_records": history_df,
    }

