"""
alerts.py
=========
Intelligent, multi-factor battery alert engine and diagnostics classifier.

Engineering & Safety Principles:
--------------------------------
1. Severity Levels:
   - INFO: Informational status notifications.
   - ADVISORY: Non-urgent guidance (e.g. stale reports or 80% charge thresholds).
   - WARNING: Accelerated capacity fading or notable capacity degradation.
   - CRITICAL: Standard lithium-ion End-of-Life (EOL) reached (<80% capacity retention).
2. Distinction Between Capacity Health & Hardware Hazards:
   A capacity-based alert reflects electrochemical cycle wear and lost unplugged runtime.
   It is NOT an electrical short-circuit or thermal runaway alarm.
3. State & Deduplication:
   Uses state-checking to avoid spamming repeated duplicate alerts for unchanged conditions.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd


DEFAULT_CAUTION_THRESHOLD = 90.0   # SOH below this triggers Advisory / Caution
DEFAULT_CRITICAL_THRESHOLD = 80.0  # SOH below this triggers Warning / Critical EOL


def evaluate_soh_alert(
    soh_percent: Optional[float],
    caution_threshold: float = DEFAULT_CAUTION_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate SOH percentage against configured engineering thresholds.
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    disclaimer = (
        "Safety Disclaimer: This is a software capacity estimate reflecting cycle aging and reduced runtime. "
        "It is NOT an electrical, thermal, or hardware safety detector."
    )

    if soh_percent is None:
        return {
            "severity": "UNKNOWN",
            "color": "#64748b",
            "badge_class": "badge-secondary",
            "title": "Health Status Unknown",
            "description": "Insufficient capacity data to evaluate battery State of Health.",
            "recommendation": "Import a valid Windows battery report or input design and full charge capacity.",
            "safety_disclaimer": disclaimer,
            "timestamp": ts,
            "soh_percent": None,
        }

    if critical_threshold >= caution_threshold:
        critical_threshold = caution_threshold - 5.0

    if soh_percent >= caution_threshold:
        return {
            "severity": "NORMAL",
            "color": "#16a34a",
            "badge_class": "badge-success",
            "title": "Battery Health Good (Normal)",
            "description": f"Battery capacity is at {soh_percent:.1f}%, exceeding the {caution_threshold:.1f}% normal threshold.",
            "recommendation": "Standard battery retention. To extend cell longevity, avoid keeping the pack at 100% in high-heat environments.",
            "safety_disclaimer": disclaimer,
            "timestamp": ts,
            "soh_percent": soh_percent,
        }
    elif soh_percent >= critical_threshold:
        return {
            "severity": "CAUTION",
            "color": "#d97706",
            "badge_class": "badge-warning",
            "title": "Battery Capacity Degrading (Caution)",
            "description": (
                f"Battery capacity has degraded to {soh_percent:.1f}%, which is below {caution_threshold:.1f}% "
                f"but above the critical retirement line ({critical_threshold:.1f}%)."
            ),
            "recommendation": (
                "Expect moderately reduced unplugged runtime. Enable manufacturer battery conservation "
                "or 80% charge thresholds (e.g. Dell Power Manager, Lenovo Vantage, ASUS Battery Health Charging)."
            ),
            "safety_disclaimer": disclaimer,
            "timestamp": ts,
            "soh_percent": soh_percent,
        }
    else:
        return {
            "severity": "CRITICAL",
            "color": "#dc2626",
            "badge_class": "badge-danger",
            "title": "Battery End of Life Reached (Critical)",
            "description": (
                f"Battery capacity is at {soh_percent:.1f}%, falling below the {critical_threshold:.1f}% "
                f"industry End-of-Life (EOL) threshold."
            ),
            "recommendation": (
                "Unplugged laptop runtime will be substantially curtailed. Plan for a genuine "
                "OEM replacement battery pack when convenient."
            ),
            "safety_disclaimer": disclaimer,
            "timestamp": ts,
            "soh_percent": soh_percent,
        }


def check_data_freshness(latest_timestamp_str: Optional[str], max_stale_days: int = 14) -> Optional[Dict[str, Any]]:
    """
    Check if the latest battery report or snapshot is older than the configured freshness window.
    """
    if not latest_timestamp_str:
        return None

    try:
        # Match YYYY-MM-DD
        import re
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", latest_timestamp_str)
        if not match:
            return None
        obs_date = datetime.strptime(match.group(1), "%Y-%m-%d")
        age_days = (datetime.now() - obs_date).days

        if age_days > max_stale_days:
            return {
                "severity": "ADVISORY",
                "category": "DATA_FRESHNESS",
                "title": f"Telemetry Data Stale ({age_days} days old)",
                "reason": f"Last recorded battery telemetry is from {match.group(1)} (exceeds {max_stale_days} day freshness threshold).",
                "supporting_measurement": f"Age: {age_days} days",
                "recommended_action": "Execute a new Windows scan or upload a fresh battery report to obtain up-to-date metrics.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except Exception:
        pass
    return None


def detect_rapid_degradation(history_df: pd.DataFrame, drop_threshold_pct: float = 3.0) -> Optional[Dict[str, Any]]:
    """
    Detect abnormally rapid capacity drop across recent sequential historical observations.
    """
    if history_df is None or len(history_df) < 3:
        return None

    df = history_df.copy()
    soh_col = next((c for c in df.columns if "soh" in c.lower()), None)
    if not soh_col:
        return None

    recent_sohs = df[soh_col].dropna().tail(4).values
    if len(recent_sohs) >= 3:
        drop = recent_sohs[0] - recent_sohs[-1]
        if drop >= drop_threshold_pct:
            return {
                "severity": "WARNING",
                "category": "DEGRADATION_RATE",
                "title": f"Accelerated Capacity Fade Detected (-{drop:.1f}% drop)",
                "reason": f"Capacity SOH dropped by {drop:.1f}% across recent observation intervals.",
                "supporting_measurement": f"Delta: -{drop:.1f}% SOH over {len(recent_sohs)} checkpoints",
                "recommended_action": "Inspect background workload thermals. Operating at sustained elevated temperatures accelerated cell wear.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    return None


def generate_system_alerts(
    current_soh: Optional[float],
    current_fcc: Optional[float],
    current_dc: Optional[float],
    history_df: Optional[pd.DataFrame] = None,
    last_scan_date: Optional[str] = None,
    caution_th: float = DEFAULT_CAUTION_THRESHOLD,
    critical_th: float = DEFAULT_CRITICAL_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Run multi-factor alert evaluation and return active diagnostic alerts.
    """
    alerts: List[Dict[str, Any]] = []

    # 1. Primary SOH Threshold Alert
    base_eval = evaluate_soh_alert(current_soh, caution_th, critical_th)
    if base_eval["severity"] != "NORMAL" and base_eval["severity"] != "UNKNOWN":
        alerts.append({
            "severity": base_eval["severity"],
            "category": "HEALTH_THRESHOLD",
            "title": base_eval["title"],
            "reason": base_eval["description"],
            "supporting_measurement": f"SOH: {current_soh:.1f}% (Caution: {caution_th}%, Critical: {critical_th}%)",
            "recommended_action": base_eval["recommendation"],
            "timestamp": base_eval["timestamp"]
        })

    # 2. Capacity Specification Anomalies
    if current_fcc is not None and current_dc is not None and current_dc > 0:
        if current_fcc > current_dc * 1.08:
            alerts.append({
                "severity": "ADVISORY",
                "category": "SPEC_ANOMALY",
                "title": "Full Charge Capacity Exceeds Design Specification (>108%)",
                "reason": "Current full charge capacity is substantially higher than factory design rating.",
                "supporting_measurement": f"FCC: {current_fcc:,.0f} mWh vs DC: {current_dc:,.0f} mWh",
                "recommended_action": "Verify whether a third-party high-capacity replacement battery was installed.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # 3. Rapid Degradation Rate Check
    if history_df is not None and not history_df.empty:
        rapid_alert = detect_rapid_degradation(history_df)
        if rapid_alert:
            alerts.append(rapid_alert)

    # 4. Data Freshness Check
    fresh_alert = check_data_freshness(last_scan_date)
    if fresh_alert:
        alerts.append(fresh_alert)

    return alerts


def build_alert_history_entry(
    date_str: str,
    soh_percent: float,
    fcc: float,
    dc: float,
    caution_th: float = DEFAULT_CAUTION_THRESHOLD,
    critical_th: float = DEFAULT_CRITICAL_THRESHOLD,
    context_note: str = ""
) -> Dict[str, Any]:
    """
    Format a single history record for tabular display in the dashboard.
    """
    alert = evaluate_soh_alert(soh_percent, caution_th, critical_th, timestamp=date_str)
    return {
        "Date": date_str,
        "SOH (%)": round(soh_percent, 2),
        "Full Charge (mWh)": round(fcc, 1),
        "Design Cap (mWh)": round(dc, 1),
        "Severity": alert["severity"],
        "Reason": alert["description"],
        "Recommendation": alert["recommendation"],
        "Context Note": context_note or "Regular monitoring log",
    }
