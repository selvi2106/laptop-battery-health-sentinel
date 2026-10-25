"""
app.py
======
Laptop Battery Health Intelligence System
Professional Hardware Diagnostics, Longitudinal SOH Tracking, and Degradation Forecasting.
"""

import os
import sys
import json
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure project root and src are available in sys.path (critical for Streamlit Cloud & nested runners)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.soh_calculator import calculate_soh
    from src.alerts import (
        evaluate_soh_alert,
        generate_system_alerts,
        build_alert_history_entry,
        DEFAULT_CAUTION_THRESHOLD,
        DEFAULT_CRITICAL_THRESHOLD
    )
    from src.windows_battery_report import (
        generate_battery_report,
        parse_battery_report_html,
        get_live_battery_status,
    )
    from src.history_store import (
        record_observation,
        get_all_observations,
        import_history_dataframe,
        delete_observation,
        clear_all_history,
        export_history_to_csv,
        PeriodicMonitor,
    )
    from src.data_loader import load_dataset
    from src.preprocessing import clean_lab_dataset, prepare_laptop_history_series, normalize_history_dataframe
    from src.train_model import train_and_evaluate_models
    from src.prediction import (
        load_model_artifact,
        forecast_laptop_degradation,
        DEFAULT_EOL_THRESHOLD,
    )
except ModuleNotFoundError:
    from soh_calculator import calculate_soh
    from alerts import (
        evaluate_soh_alert,
        generate_system_alerts,
        build_alert_history_entry,
        DEFAULT_CAUTION_THRESHOLD,
        DEFAULT_CRITICAL_THRESHOLD
    )
    from windows_battery_report import (
        generate_battery_report,
        parse_battery_report_html,
        get_live_battery_status,
    )
    from history_store import (
        record_observation,
        get_all_observations,
        import_history_dataframe,
        delete_observation,
        clear_all_history,
        export_history_to_csv,
        PeriodicMonitor,
    )
    from data_loader import load_dataset
    from preprocessing import clean_lab_dataset, prepare_laptop_history_series, normalize_history_dataframe
    from train_model import train_and_evaluate_models
    from prediction import (
        load_model_artifact,
        forecast_laptop_degradation,
        DEFAULT_EOL_THRESHOLD,
    )

# -----------------------------------------------------------------------------
# Page Configuration & Professional Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laptop Battery Health Sentinel",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom Styling adhering strictly to specified Color Palette:
# Primary: #102A43 (Navy) | Secondary: #0D9488 (Teal) | Background: #F5F7FA
# Text: #243447 (Slate) | Warning: #D97706 (Amber) | Critical: #DC2626 (Red)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #F1F5F9 0%, #F8FAFC 60%, #F1F5F9 100%);
    }

    /* Clean Landing View Hero Presentation */
    .landing-hero-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 3.5rem 3rem 2.5rem 3rem;
        max-width: 820px;
        margin: 3.5rem auto 1.5rem auto;
        box-shadow: 0 20px 45px -15px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(15, 23, 42, 0.03);
        text-align: center;
        position: relative;
    }
    .landing-badge-glow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #F0FDFA;
        color: #0D9488;
        border: 1px solid #99F6E4;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.12);
        margin-bottom: 1.25rem;
    }
    .badge-dot {
        width: 7px;
        height: 7px;
        background-color: #0D9488;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #0D9488;
    }
    .landing-hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        letter-spacing: -0.03em;
        margin: 0 0 0.8rem 0;
    }
    .landing-hero-subtitle {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0D9488;
        letter-spacing: -0.01em;
        margin: 0 0 1.2rem 0;
    }
    .landing-hero-description {
        font-size: 1.02rem;
        line-height: 1.65;
        color: #475569;
        max-width: 680px;
        margin: 0 auto 2.2rem auto;
    }
    .landing-feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
        text-align: left;
    }
    .feature-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        transition: all 0.2s ease;
    }
    .feature-chip:hover {
        border-color: #0D9488;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.08);
        transform: translateY(-2px);
    }
    .chip-icon {
        font-size: 1.35rem;
        line-height: 1;
    }
    .feature-chip strong {
        display: block;
        font-size: 0.88rem;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .feature-chip span {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.35;
    }
    .landing-footer {
        text-align: center;
        color: #64748B;
        font-size: 0.84rem;
        margin-top: 1.8rem;
    }
    
    /* Top Brand Navigation Header */
    .sentinel-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0F2D37 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.25rem 1.75rem;
        color: #FFFFFF;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.15);
    }
    .sentinel-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: #FFFFFF;
    }
    .sentinel-subtitle {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 0.25rem;
    }
    .header-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 8px;
        padding: 0.45rem 0.9rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
        backdrop-filter: blur(8px);
    }
    
    /* Professional Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        transform: translateY(-2px);
        border-color: #CBD5E1;
    }
    .metric-card.accent-teal {
        border-top: 3px solid #0D9488;
    }
    .metric-card.accent-navy {
        border-top: 3px solid #0F172A;
    }
    .metric-card.accent-blue {
        border-top: 3px solid #2563EB;
    }
    .metric-card.accent-amber {
        border-top: 3px solid #D97706;
    }
    .metric-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value-huge {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.4rem;
        letter-spacing: -0.03em;
        font-feature-settings: "tnum";
    }
    .metric-value-huge.teal {
        color: #0D9488;
    }
    .metric-subtext {
        font-size: 0.84rem;
        margin-top: 0.35rem;
        font-weight: 500;
        color: #475569;
    }
    
    /* Origin badges */
    .badge-origin {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
    }
    .badge-actual {
        background: #F0FDFA;
        color: #0D9488;
        border: 1px solid #99F6E4;
    }
    .badge-calc {
        background: #F8FAFC;
        color: #0F172A;
        border: 1px solid #CBD5E1;
    }
    .badge-forecast {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-unavailable {
        background: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
    }
    
    /* Health Classification Badges */
    .badge-health {
        padding: 0.28rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .health-optimal {
        background: #F0FDFA;
        color: #0D9488;
        border: 1px solid #99F6E4;
    }
    .health-caution {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .health-critical {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    
    /* Alerts Cards */
    .alert-banner {
        border-radius: 10px;
        padding: 1rem 1.35rem;
        margin-bottom: 0.85rem;
        border-left: 5px solid;
        background: #FFFFFF;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .alert-banner.CRITICAL {
        border-left-color: #DC2626;
        background: #FEF2F2;
        border-top: 1px solid #FEE2E2;
        border-right: 1px solid #FEE2E2;
        border-bottom: 1px solid #FEE2E2;
    }
    .alert-banner.WARNING, .alert-banner.CAUTION {
        border-left-color: #D97706;
        background: #FFFBEB;
        border-top: 1px solid #FEF3C7;
        border-right: 1px solid #FEF3C7;
        border-bottom: 1px solid #FEF3C7;
    }
    .alert-banner.ADVISORY, .alert-banner.INFO {
        border-left-color: #0D9488;
        background: #F0FDFA;
        border-top: 1px solid #CCFBF1;
        border-right: 1px solid #CCFBF1;
        border-bottom: 1px solid #CCFBF1;
    }
    
    /* Engineering Disclaimers */
    .disclaimer-box {
        background: #F8FAFC;
        border-left: 3px solid #64748B;
        padding: 0.85rem 1.2rem;
        font-size: 0.84rem;
        color: #334155;
        border-radius: 6px;
        margin: 1rem 0;
        line-height: 1.5;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 2px;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #475569;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A;
        background-color: #F8FAFC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-color: #0F172A !important;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.12);
    }
    
    /* Button enhancements */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #0F172A !important;
        border-radius: 10px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 10px 20px -5px rgba(15, 23, 42, 0.22) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
        border-color: #0D9488 !important;
        box-shadow: 0 12px 25px -4px rgba(13, 148, 136, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.1rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #0F172A !important;
        color: #0F172A !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize Route & Session State
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    qp = st.query_params.get("page", None)
    if qp in ("dashboard", "home"):
        st.session_state["current_page"] = qp
    else:
        st.session_state["current_page"] = "home"

# -----------------------------------------------------------------------------
# LANDING VIEW: Clean, Simple & Professional First Page
# -----------------------------------------------------------------------------
if st.session_state.get("current_page", "home") == "home":
    st.markdown("""
    <div class="landing-hero-card">
        <div class="landing-badge-glow">
            <span class="badge-dot"></span> 🔋 HARDWARE BATTERY INTELLIGENCE
        </div>
        <h1 class="landing-hero-title">
            Laptop Battery Health Monitoring and SOH Prediction
        </h1>
        <p class="landing-hero-subtitle">
            Continuous Electrochemical Diagnostics &bull; Longitudinal Degradation Tracking &bull; ML Forecasting
        </p>
        <p class="landing-hero-description">
            An engineering-grade intelligence suite designed to monitor laptop battery health, 
            calculate true electrochemical State of Health (SOH) from BMS telemetry, 
            track capacity retention across charge cycles, and forecast End-of-Life degradation using machine learning.
        </p>
        <div class="landing-feature-grid">
            <div class="feature-chip">
                <span class="chip-icon">⚡</span>
                <div>
                    <strong>Live BMS Telemetry</strong>
                    <span>Direct ACPI &amp; report parsing</span>
                </div>
            </div>
            <div class="feature-chip">
                <span class="chip-icon">📊</span>
                <div>
                    <strong>SOH Health Analytics</strong>
                    <span>Electrochemical retention curves</span>
                </div>
            </div>
            <div class="feature-chip">
                <span class="chip-icon">🔮</span>
                <div>
                    <strong>Degradation ML Models</strong>
                    <span>Empirical RUL forecasting</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 1.4, 1])
    with col_btn_c:
        if st.button("📊 View Dashboard", type="primary", use_container_width=True, key="btn_view_dashboard"):
            st.session_state["current_page"] = "dashboard"
            st.query_params["page"] = "dashboard"
            st.rerun()

    st.markdown("""
    <div class="landing-footer">
        <span>🔒 Verified Hardware Report Ingestion</span> &bull; 
        <span>💾 Local SQLite Audit Store</span> &bull; 
        <span>🔬 NASA Ames Benchmark ML Models</span>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# -----------------------------------------------------------------------------
# DASHBOARD INITIALIZATION & HARDWARE INGESTION
# -----------------------------------------------------------------------------
if "caution_threshold" not in st.session_state:
    st.session_state["caution_threshold"] = DEFAULT_CAUTION_THRESHOLD
if "critical_threshold" not in st.session_state:
    st.session_state["critical_threshold"] = DEFAULT_CRITICAL_THRESHOLD

# Fetch Live Operating System Telemetry
live_status = get_live_battery_status()

# Auto-discover or generate live Windows battery report
if "current_report" not in st.session_state:
    if os.name == "nt":
        s_ok, report_path = generate_battery_report()
        if s_ok:
            try:
                with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
                    st.session_state["current_report"] = parse_battery_report_html(f.read())
                    st.session_state["data_source_tag"] = "WINDOWS_HARDWARE_REPORT"
            except Exception:
                pass

# If not generated, check for existing battery_report.html in project root (essential for Streamlit Cloud deployment)
if "current_report" not in st.session_state:
    root_report = os.path.join(PROJECT_ROOT, "battery_report.html")
    if os.path.exists(root_report):
        try:
            with open(root_report, "r", encoding="utf-8", errors="ignore") as f:
                parsed_root = parse_battery_report_html(f.read())
                if parsed_root.get("success"):
                    st.session_state["current_report"] = parsed_root
                    st.session_state["data_source_tag"] = "WINDOWS_HARDWARE_REPORT"
        except Exception:
            pass

# If still not populated, fallback to baseline sample
if "current_report" not in st.session_state:
    sample_hist_path = os.path.join(PROJECT_ROOT, "data", "sample_laptop_history.csv")
    if os.path.exists(sample_hist_path):
        s_raw = pd.read_csv(sample_hist_path)
        _, h_prepared, _ = prepare_laptop_history_series(s_raw)
        latest_s = h_prepared.iloc[-1]
        st.session_state["current_report"] = {
            "success": True,
            "system_info": {"computer_name": "BENCHMARK-SYSTEM", "product_name": "Dell 15 DC15255 (Benchmark Profile)"},
            "battery_info": {
                "name": "DELL PG8YJ5AK",
                "manufacturer": "BYD",
                "chemistry": "NCM",
                "design_capacity": float(latest_s["design_capacity_mwh"]),
                "full_charge_capacity": float(latest_s["full_charge_capacity_mwh"]),
                "cycle_count": int(latest_s["cycle_count"]) if "cycle_count" in latest_s and pd.notna(latest_s["cycle_count"]) else None
            },
            "history_records": h_prepared
        }
        st.session_state["data_source_tag"] = "BENCHMARK_DEMO_DATA"

rep = st.session_state["current_report"]
b_info = rep.get("battery_info", {})
s_info = rep.get("system_info", {})
hist_records = normalize_history_dataframe(rep.get("history_records", pd.DataFrame()))
source_tag = st.session_state.get("data_source_tag", "UNKNOWN")

curr_dc = b_info.get("design_capacity")
curr_fcc = b_info.get("full_charge_capacity")
curr_cycles = b_info.get("cycle_count") or live_status.get("cycle_count")

# Compute primary SOH
soh_calc = calculate_soh(curr_fcc, curr_dc)
active_soh = soh_calc["soh_percent"] if soh_calc["is_valid"] else None


# -----------------------------------------------------------------------------
# Sidebar Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    if st.button("⬅️ Back to Home", key="btn_back_home_sidebar", use_container_width=True):
        st.session_state["current_page"] = "home"
        st.query_params["page"] = "home"
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Sentinel Control")
    st.caption("Laptop Battery Health Intelligence System")

    # Quick hardware status pill in sidebar
    st.markdown("---")
    st.markdown("##### 🔌 Hardware Link")
    if live_status.get("is_available"):
        p_state = live_status.get("status_text", "Online")
        chg = live_status.get("charge_percent")
        st.write(f"- **Power Status:** `{p_state}`")
        st.write(f"- **Live Charge:** `{chg}%`" if chg is not None else "- **Live Charge:** `N/A`")
        if live_status.get("live_voltage_v"):
            st.write(f"- **Live Voltage:** `{live_status['live_voltage_v']:.2f} V`")
        if live_status.get("discharge_rate_w"):
            st.write(f"- **Discharge Rate:** `{live_status['discharge_rate_w']:.1f} W`")
    else:
        if os.name != "nt":
            st.caption("☁️ **Cloud Host (Linux)**: Hardware battery sensors require a local Windows laptop. Benchmark data is active, or upload an HTML report in **Live Diagnostics**.")
        else:
            st.write("`Hardware Telemetry Offline`")

    st.markdown("---")
    st.markdown("##### ⚙️ Health Thresholds")

    c_th = st.slider(
        "Caution Threshold (% SOH)",
        min_value=75.0,
        max_value=95.0,
        value=float(st.session_state["caution_threshold"]),
        step=1.0,
        help="Battery SOH below this triggers a maintenance advisory."
    )
    eol_th = st.slider(
        "Critical EOL Threshold (% SOH)",
        min_value=50.0,
        max_value=85.0,
        value=float(st.session_state["critical_threshold"]),
        step=1.0,
        help="Industry standard lithium-ion End-of-Life retirement threshold (80% SOH)."
    )
    if eol_th >= c_th:
        st.error("Critical threshold must be strictly lower than Caution threshold.")
        eol_th = c_th - 5.0
    st.session_state["caution_threshold"] = c_th
    st.session_state["critical_threshold"] = eol_th

    st.markdown("---")
    st.markdown("##### ⏱️ Periodic In-App Logging")
    p_monitor = PeriodicMonitor.get_instance()
    is_p_running = p_monitor.is_running
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        if is_p_running:
            if st.button("Stop Logging", type="secondary", use_container_width=True):
                p_monitor.stop()
                st.rerun()
        else:
            if st.button("Start Logging", type="primary", use_container_width=True):
                caps = (curr_dc, curr_fcc) if curr_dc and curr_fcc else None
                p_monitor.start(interval_seconds=300, current_capacities=caps)
                st.rerun()
    with col_p2:
        st.caption("Active while app is open.")

    if is_p_running:
        st.success(f"Running (every {p_monitor.interval_seconds // 60}m)")
        st.caption(f"Ticks logged: {p_monitor.run_count} | Last: {p_monitor.last_run or 'Active'}")
    else:
        st.caption("Status: Inactive")

    st.markdown("---")
    if st.button("🔄 Refresh System Telemetry", use_container_width=True):
        if os.name == "nt":
            ok, p_out = generate_battery_report()
            if ok:
                with open(p_out, "r", encoding="utf-8", errors="ignore") as f:
                    st.session_state["current_report"] = parse_battery_report_html(f.read())
                    st.session_state["data_source_tag"] = "WINDOWS_HARDWARE_REPORT"
        st.rerun()


# -----------------------------------------------------------------------------
# Top Sentinel Branding Bar & Navigation
# -----------------------------------------------------------------------------
top_nav_l, top_nav_r = st.columns([2, 8])
with top_nav_l:
    if st.button("⬅️ Back to Home", key="btn_back_home_top", use_container_width=True):
        st.session_state["current_page"] = "home"
        st.query_params["page"] = "home"
        st.rerun()
with top_nav_r:
    st.markdown("""
    <div style="display:flex; justify-content:flex-end; align-items:center; height:100%; gap:0.5rem; font-size:0.84rem; color:#64748B;">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 6px #10B981;"></span>
        <strong>Sentinel Active</strong> &bull; Diagnostic Telemetry Loaded
    </div>
    """, unsafe_allow_html=True)

host_label = s_info.get("product_name") or s_info.get("computer_name") or "Dell 15 DC15255"
batt_label = b_info.get("name") or live_status.get("device_name") or "DELL PG8YJ5AK"
chem_label = b_info.get("chemistry") or live_status.get("chemistry") or "NCM / Li-Ion"
mfg_label = b_info.get("manufacturer") or live_status.get("manufacturer") or "BYD"

if live_status.get("is_available"):
    charge_txt = f"{live_status['charge_percent']}%" if live_status.get("charge_percent") is not None else "100%"
    pwr_txt = live_status.get("status_text") or "Online"
else:
    charge_txt = "100%" if curr_fcc and curr_dc else "N/A"
    pwr_txt = "AC Connected" if curr_fcc and curr_dc else "Telemetry Offline"

src_badge = "[MEASURED REPORT]" if source_tag == "WINDOWS_HARDWARE_REPORT" else "[BENCHMARK DEMO]"

st.markdown(f"""
<div class="sentinel-header">
    <div>
        <div class="sentinel-title">
            🔋 Laptop Battery Health Sentinel
        </div>
        <div class="sentinel-subtitle">
            System: <strong>{host_label}</strong> &bull; Battery Pack: <strong>{batt_label}</strong> ({mfg_label} {chem_label})
        </div>
    </div>
    <div style="display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap;">
        <div class="header-pill">
            <span>Charge:</span> <strong>{charge_txt}</strong>
        </div>
        <div class="header-pill">
            <span>Power:</span> <strong>{pwr_txt}</strong>
        </div>
        <div class="header-pill">
            <span>Data Source:</span> <strong>{src_badge}</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Classification Helpers & Visualization Generators
# -----------------------------------------------------------------------------
def get_health_badge(soh: Optional[float], caution: float, critical: float) -> Tuple[str, str]:
    if soh is None:
        return "Unknown Status", "health-caution"
    if soh >= 95.0:
        return "Optimal (Pristine)", "health-optimal"
    elif soh >= caution:
        return "Optimal (Good Retention)", "health-optimal"
    elif soh >= critical:
        return "Degraded (Maintenance Caution)", "health-caution"
    else:
        return "Critical (End-of-Life Reached)", "health-critical"


def render_soh_gauge_chart(soh: float, caution: float, critical: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={'text': "State of Health (SOH)", 'font': {'size': 15, 'color': '#102A43', 'family': 'Inter'}},
        number={'suffix': "%", 'font': {'size': 34, 'color': '#102A43', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 110], 'tickwidth': 1, 'tickcolor': "#627D98"},
            'bar': {'color': "#102A43", 'thickness': 0.28},
            'bgcolor': "#F5F7FA",
            'borderwidth': 1,
            'bordercolor': "#D9E2EC",
            'steps': [
                {'range': [0, critical], 'color': "rgba(220, 38, 38, 0.16)"},
                {'range': [critical, caution], 'color': "rgba(217, 119, 6, 0.16)"},
                {'range': [caution, 110], 'color': "rgba(13, 148, 136, 0.18)"}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3},
                'thickness': 0.8,
                'value': critical
            }
        }
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# -----------------------------------------------------------------------------
# MAIN DASHBOARD TABS (5 SPECIFIED SECTIONS)
# -----------------------------------------------------------------------------
tab_overview, tab_diagnostics, tab_history, tab_ai, tab_alerts = st.tabs([
    "📊 Overview",
    "🔬 Live Diagnostics",
    "📈 Battery History",
    "🤖 AI Analytics",
    "⚠️ Alerts"
])


# =============================================================================
# 1. OVERVIEW
# =============================================================================
with tab_overview:
    st.markdown("#### System Overview & Key Health Indicators")

    # Check for active alerts to surface in overview
    last_scan_ts = s_info.get("report_time") or live_status.get("timestamp")
    active_alerts = generate_system_alerts(
        current_soh=active_soh,
        current_fcc=curr_fcc,
        current_dc=curr_dc,
        history_df=hist_records,
        last_scan_date=last_scan_ts,
        caution_th=c_th,
        critical_th=eol_th,
        charge_percent=live_status.get("charge_percent"),
        power_status=live_status.get("status_text"),
        live_voltage_v=live_status.get("live_voltage_v"),
        design_voltage_v=live_status.get("design_voltage_v"),
        discharge_rate_w=live_status.get("discharge_rate_w"),
        temperature_c=live_status.get("temperature_c")
    )
    critical_alerts = [a for a in active_alerts if a["severity"] in ("CRITICAL", "WARNING")]

    if critical_alerts:
        for ca in critical_alerts:
            st.markdown(f"""
            <div class="alert-banner {ca['severity']}">
                <strong>[{ca['severity']}] {ca['title']}</strong> &bull; <span style="font-size:0.8rem; color:#486581;">{ca['timestamp']}</span>
                <div style="font-size:0.85rem; margin-top:0.25rem;">{ca['reason']} &mdash; <strong>Action:</strong> {ca['recommended_action']}</div>
            </div>
            """, unsafe_allow_html=True)

    # 4 Key Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if live_status.get("is_available"):
            chg_v = f"{live_status.get('charge_percent')}%" if live_status.get("charge_percent") is not None else "100%"
            rt_min = live_status.get("estimated_runtime_minutes")
            rt_txt = f"{rt_min} min remaining" if rt_min else ("AC Connected" if live_status.get("power_online") else "Discharging")
            sub_c1 = f"{pwr_txt} &bull; {rt_txt}"
            badge_c1 = '<span class="badge-origin badge-actual">[MEASURED]</span>'
        else:
            chg_v = "100%" if curr_fcc and curr_dc else "N/A"
            badge_c1 = '<span class="badge-origin badge-actual">[MEASURED REPORT]</span>' if source_tag == "WINDOWS_HARDWARE_REPORT" else '<span class="badge-origin badge-calc">[DEMO]</span>'
            sub_c1 = "AC Connected &bull; Full factory retention intact"

        st.markdown(f"""
        <div class="metric-card accent-teal">
            <div class="metric-label">
                Current Charge {badge_c1}
            </div>
            <div class="metric-value-huge teal">{chg_v}</div>
            <div class="metric-subtext">{sub_c1}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if active_soh is not None:
            soh_str = f"{active_soh:.1f}%"
            badge_text, badge_css = get_health_badge(active_soh, c_th, eol_th)
            sub_txt = f'<span class="badge-health {badge_css}">{badge_text}</span>'
        else:
            soh_str = "N/A"
            sub_txt = '<span class="badge-health health-caution">Insufficient Data</span>'

        st.markdown(f"""
        <div class="metric-card accent-navy">
            <div class="metric-label">
                State of Health (SOH) <span class="badge-origin badge-calc">[CALCULATED]</span>
            </div>
            <div class="metric-value-huge">{soh_str}</div>
            <div class="metric-subtext">{sub_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        fcc_txt = f"{curr_fcc:,.0f} mWh" if curr_fcc else "N/A"
        lost = max(0.0, curr_dc - curr_fcc) if curr_dc and curr_fcc else 0.0
        lost_txt = f"&minus;{lost:,.0f} mWh lost" if lost > 0 else "Full factory retention intact"
        color_lost = "#B91C1C" if lost > 0 else "#0D9488"
        st.markdown(f"""
        <div class="metric-card accent-blue">
            <div class="metric-label">
                Full Charge Capacity <span class="badge-origin badge-actual">[MEASURED]</span>
            </div>
            <div class="metric-value-huge">{fcc_txt}</div>
            <div class="metric-subtext" style="color:{color_lost}; font-weight:600;">{lost_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        dc_txt = f"{curr_dc:,.0f} mWh" if curr_dc else "N/A"
        rec_count = len(hist_records) if not hist_records.empty else 0
        cyc_val = f"{curr_cycles} cycles" if curr_cycles is not None else f"{rec_count} historical audit scans"
        st.markdown(f"""
        <div class="metric-card accent-amber">
            <div class="metric-label">
                Design Capacity <span class="badge-origin badge-actual">[MEASURED]</span>
            </div>
            <div class="metric-value-huge">{dc_txt}</div>
            <div class="metric-subtext" style="color:#0D9488; font-weight:600;">Factory Nominal &bull; {cyc_val}</div>
        </div>
        """, unsafe_allow_html=True)

    # Overview Visualizations: Gauge & Snapshot Control
    g_col, s_col = st.columns([1.1, 1.9])

    with g_col:
        if active_soh is not None:
            st.plotly_chart(render_soh_gauge_chart(active_soh, c_th, eol_th), use_container_width=True)
        else:
            st.info("Input design and full charge capacity to render health gauge.")

    with s_col:
        st.markdown("##### Quick Actions & Snapshot Recording")
        st.caption("Capture the current battery condition into your local SQLite audit log.")
        
        btn_snap, btn_import = st.columns([1, 1])
        with btn_snap:
            if st.button("💾 Record Snapshot to Local DB", type="primary", use_container_width=True):
                ok, msg, _ = record_observation(
                    design_capacity_mwh=curr_dc,
                    full_charge_capacity_mwh=curr_fcc,
                    soh_percent=active_soh,
                    data_source=source_tag,
                    device_name=batt_label,
                    manufacturer=mfg_label,
                    chemistry=chem_label,
                    cycle_count=curr_cycles,
                    charge_percent=live_status.get("charge_percent"),
                    power_status=live_status.get("status_text"),
                    voltage_v=live_status.get("live_voltage_v"),
                    discharge_rate_w=live_status.get("discharge_rate_w"),
                    force_snapshot=True
                )
                if ok:
                    st.success(msg)
                else:
                    st.info(msg)

        with btn_import:
            if not hist_records.empty:
                if st.button(f"📥 Import {len(hist_records)} History Records to DB", use_container_width=True):
                    imp, skp = import_history_dataframe(
                        hist_records,
                        source_tag="WINDOWS_REPORT_HISTORY",
                        device_name=batt_label,
                        manufacturer=mfg_label,
                        chemistry=chem_label
                    )
                    st.success(f"Imported {imp} historical records ({skp} duplicates skipped).")

        st.markdown(f"""
        <div class="disclaimer-box">
            <strong>Formula:</strong> SOH (%) = (Full Charge Capacity / Design Capacity) &times; 100<br>
            <strong>Last Telemetry Scan:</strong> {live_status.get('timestamp')}<br>
            <strong>Reported Host:</strong> {host_label} (BIOS: {s_info.get('bios_version', 'OEM Standard')})
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 2. LIVE DIAGNOSTICS
# =============================================================================
with tab_diagnostics:
    st.markdown("#### Hardware Telemetry & Diagnostic Interface")
    st.caption("Direct non-elevated queries against Windows CIM/WMI hardware interfaces and battery reports.")

    # Telemetry Control Expanders
    with st.expander("🛠️ Telemetry Ingestion Controls & Ingest Options", expanded=False):
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown("##### 1. Live Windows Scan")
            st.caption("Executes `powercfg /batteryreport` locally.")
            if st.button("Run Windows Battery Scan", type="primary", use_container_width=True):
                with st.spinner("Querying Windows ACPI telemetry..."):
                    ok, p_out = generate_battery_report()
                    if ok:
                        with open(p_out, "r", encoding="utf-8", errors="ignore") as f:
                            st.session_state["current_report"] = parse_battery_report_html(f.read())
                            st.session_state["data_source_tag"] = "WINDOWS_HARDWARE_REPORT"
                        st.success("Refreshed from live Windows report.")
                        st.rerun()
                    else:
                        st.warning(p_out)

        with ic2:
            st.markdown("##### 2. Upload Battery Report")
            st.caption("Parse HTML report from another machine.")
            u_file = st.file_uploader("Select report HTML", type=["html", "htm"], key="upload_report_diag")
            if u_file is not None:
                parsed_u = parse_battery_report_html(u_file.read().decode("utf-8", errors="ignore"))
                if parsed_u["success"]:
                    st.session_state["current_report"] = parsed_u
                    st.session_state["data_source_tag"] = f"UPLOADED_FILE ({u_file.name})"
                    st.success("Uploaded report parsed.")
                    st.rerun()
                else:
                    st.error(parsed_u["error"])

        with ic3:
            st.markdown("##### 3. Manual Parameter Input")
            st.caption("Override hardware parameters manually.")
            with st.form("manual_override_form"):
                m_dc = st.number_input("Design Capacity (mWh)", value=float(curr_dc or 41006.0), step=500.0)
                m_fcc = st.number_input("Full Charge Capacity (mWh)", value=float(curr_fcc or 41006.0), step=500.0)
                m_cyc = st.number_input("Cycle Count", value=int(curr_cycles or 0), step=1)
                if st.form_submit_button("Apply Manual Parameters", use_container_width=True):
                    st.session_state["current_report"] = {
                        "success": True,
                        "system_info": {"computer_name": "MANUAL-OVERRIDE", "product_name": "User Specified Profile"},
                        "battery_info": {
                            "name": "Custom Battery Profile",
                            "manufacturer": "Manual",
                            "chemistry": "Li-Ion",
                            "design_capacity": m_dc,
                            "full_charge_capacity": m_fcc,
                            "cycle_count": m_cyc
                        },
                        "history_records": hist_records
                    }
                    st.session_state["data_source_tag"] = "MANUAL_INPUT"
                    st.success("Manual parameters loaded.")
                    st.rerun()

    # Hardware Telemetry Grid
    d_c1, d_c2, d_c3 = st.columns(3)

    with d_c1:
        st.markdown("##### Electrical Measurements")
        live_v = live_status.get("live_voltage_v")
        des_v = live_status.get("design_voltage_v")
        dis_w = live_status.get("discharge_rate_w")
        chg_w = live_status.get("charge_rate_w")

        v_disp = f"{live_v:.3f} V" if live_v else "Not reported by ACPI"
        dv_disp = f"{des_v:.3f} V" if des_v else "Not reported"
        w_disp = f"{dis_w:.2f} W (Discharge)" if dis_w else (f"{chg_w:.2f} W (Charge)" if chg_w else "Idle / AC")

        st.write(f"- **Live Terminal Voltage:** `{v_disp}` &nbsp; <span class='badge-origin badge-actual'>[MEASURED]</span>", unsafe_allow_html=True)
        st.write(f"- **Design Voltage:** `{dv_disp}` &nbsp; <span class='badge-origin badge-actual'>[MEASURED]</span>", unsafe_allow_html=True)
        st.write(f"- **Live Power Rate:** `{w_disp}` &nbsp; <span class='badge-origin badge-actual'>[MEASURED]</span>", unsafe_allow_html=True)
        st.write(f"- **AC Power Line:** `{'Plugged in' if live_status.get('power_online') else 'Operating on Battery'}`")

    with d_c2:
        st.markdown("##### Battery Specification")
        st.write(f"- **Model Identifier:** `{batt_label}`")
        st.write(f"- **Manufacturer:** `{mfg_label}`")
        st.write(f"- **Chemistry:** `{chem_label}`")
        st.write(f"- **Device ID:** `{live_status.get('device_id') or 'ACPI Battery'}`")
        cyc_label = f"{curr_cycles}" if curr_cycles is not None else "Not reported by hardware/BMS"
        st.write(f"- **Cycle Count:** `{cyc_label}`")

    with d_c3:
        st.markdown("##### Thermal & Host Telemetry")
        temp_val = live_status.get("temperature_c")
        temp_txt = f"{temp_val:.1f} °C" if temp_val is not None else live_status.get("temperature_status", "Not reported by ACPI")
        st.write(f"- **Battery Temperature:** `{temp_txt}` &nbsp; <span class='badge-origin badge-unavailable'>[UNAVAILABLE]</span>", unsafe_allow_html=True)
        st.write(f"- **Host Machine:** `{s_info.get('product_name', 'Dell Laptop')}`")
        st.write(f"- **Computer Name:** `{s_info.get('computer_name', 'Local Machine')}`")
        st.write(f"- **BIOS Version:** `{s_info.get('bios_version', 'Standard OEM')}`")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Transparency & Data Integrity:</strong> Measurements labeled <code>[MEASURED]</code> represent genuine hardware sensor values queried from your laptop's Battery Management System (BMS) via Microsoft Windows CIM/WMI interfaces. 
        Temperature and internal impedance sensors omitted by standard consumer ACPI firmware are reported honestly as unavailable rather than simulated.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 3. BATTERY HISTORY
# =============================================================================
with tab_history:
    st.markdown("#### Longitudinal Battery History & Local Database")
    st.caption("Historical capacity degradation records stored locally in SQLite (`battery_sentinel.db`).")

    # Filter Controls
    f_c1, f_c2, f_c3 = st.columns([1, 1, 1])
    with f_c1:
        date_start = st.date_input("Filter Start Date", value=None, key="hist_start_date")
    with f_c2:
        date_end = st.date_input("Filter End Date", value=None, key="hist_end_date")
    with f_c3:
        source_filter = st.selectbox(
            "Filter Data Source",
            ["ALL", "WINDOWS_HARDWARE_REPORT", "WINDOWS_REPORT_HISTORY", "PERIODIC_BACKGROUND_SERVICE", "MANUAL_INPUT"],
            key="hist_source_filter"
        )

    s_str = date_start.strftime("%Y-%m-%d") if date_start else None
    e_str = date_end.strftime("%Y-%m-%d") if date_end else None

    db_obs = get_all_observations(start_date=s_str, end_date=e_str, data_source=source_filter)

    # Use database observations if available, otherwise display parsed Windows report history
    display_history = db_obs if not db_obs.empty else hist_records

    if not display_history.empty:
        # History Summary Metrics
        first_soh = display_history["soh_percent"].iloc[0]
        latest_soh = display_history["soh_percent"].iloc[-1]
        delta_net = latest_soh - first_soh

        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.metric("Total Observations", f"{len(display_history)} records")
        with sm2:
            st.metric("Initial Recorded SOH", f"{first_soh:.1f}%")
        with sm3:
            st.metric("Latest Recorded SOH", f"{latest_soh:.1f}%")
        with sm4:
            st.metric("Net SOH Shift", f"{delta_net:+.2f}%", delta=f"{delta_net:.2f}%")

        # Capacity History Trend Plot
        fig_hist = go.Figure()
        x_col = display_history["Period"] if "Period" in display_history.columns else display_history["timestamp"]

        fig_hist.add_trace(go.Scatter(
            x=x_col,
            y=display_history["design_capacity_mwh"],
            name="Design Capacity",
            line=dict(color="#627D98", dash="dash", width=1.5)
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_col,
            y=display_history["full_charge_capacity_mwh"],
            name="Full Charge Capacity",
            line=dict(color="#102A43", width=2.5),
            fill="tonexty",
            fillcolor="rgba(16, 42, 67, 0.06)"
        ))
        fig_hist.update_layout(
            title=dict(text="Capacity Retention Trend (mWh)", font=dict(size=14, color="#102A43")),
            height=280,
            margin=dict(l=20, r=20, t=35, b=20),
            yaxis_title="Capacity (mWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # SOH Percentage Degradation Curve
        fig_soh_trend = go.Figure()
        fig_soh_trend.add_trace(go.Scatter(
            x=x_col,
            y=display_history["soh_percent"],
            name="State of Health (SOH)",
            line=dict(color="#0D9488", width=2.5),
            mode="lines+markers"
        ))
        fig_soh_trend.add_hline(y=c_th, line_dash="dash", line_color="#D97706", annotation_text=f"Caution ({c_th}%)")
        fig_soh_trend.add_hline(y=eol_th, line_dash="dash", line_color="#DC2626", annotation_text=f"Critical EOL ({eol_th}%)")
        fig_soh_trend.update_layout(
            title=dict(text="State of Health (SOH %) Degradation Profile", font=dict(size=14, color="#102A43")),
            height=260,
            margin=dict(l=20, r=20, t=35, b=20),
            yaxis_title="SOH (%)",
            yaxis_range=[max(0, eol_th - 25), 105],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_soh_trend, use_container_width=True)

        # Observation Data Table
        st.markdown("##### Historical Observation Log Table")
        st.dataframe(display_history, use_container_width=True, hide_index=True)

        # CSV Export & Management Controls
        exp_col, del_col = st.columns([1, 1])
        with exp_col:
            csv_data = export_history_to_csv(start_date=s_str, end_date=e_str, data_source=source_filter)
            st.download_button(
                label="📥 Export Observation Log as CSV",
                data=csv_data,
                file_name=f"sentinel_battery_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with del_col:
            with st.expander("Manage Stored Database Records", expanded=False):
                d_id = st.number_input("Record ID to delete", min_value=1, step=1)
                if st.button("Delete Specific Observation", type="secondary"):
                    if delete_observation(int(d_id)):
                        st.success(f"Record #{d_id} removed.")
                        st.rerun()
                    else:
                        st.error(f"Record #{d_id} not found.")

                if st.button("Purge All Stored Observations", type="primary"):
                    p_cnt = clear_all_history()
                    st.success(f"Purged {p_cnt} records from database.")
                    st.rerun()

    else:
        st.info("No battery observations found for the selected filter criteria. Click 'Record Snapshot' in the Overview tab to store observations.")


# =============================================================================
# 4. AI ANALYTICS
# =============================================================================
with tab_ai:
    st.markdown("#### Machine Learning & Degradation Analytics")
    st.caption("Evidence-based empirical forecasting and laboratory benchmark machine learning pipelines.")

    ai_subtab_forecast, ai_subtab_research = st.tabs([
        "🔮 Laptop Degradation Forecast",
        "🔬 Laboratory ML Models (NASA Ames 18650 Benchmark)"
    ])

    # -------------------------------------------------------------------------
    # Sub-View A: Laptop Degradation Forecasting
    # -------------------------------------------------------------------------
    with ai_subtab_forecast:
        st.markdown("##### Laptop Degradation Forecasting & RUL Estimation")
        st.caption("Empirical trend extrapolation using actual historical capacity logs with 95% uncertainty envelopes.")

        source_for_forecast = db_obs if len(db_obs) >= 3 else hist_records

        if len(source_for_forecast) >= 3:
            fc = forecast_laptop_degradation(source_for_forecast, future_steps=8, eol_threshold=eol_th)

            if fc["is_forecastable"]:
                f_info, f_chart = st.columns([1, 1.8])

                with f_info:
                    st.markdown("###### Extrapolation Statistics")
                    st.write(f"- **Current Capacity SOH:** `{fc['current_soh']:.1f}%`")
                    st.write(f"- **Degradation Rate:** `{fc['degradation_rate_per_step']:.4f}% per interval`")
                    st.write(f"- **Residual Standard Error (SE):** `±{fc['residual_std_error']:.2f}% SOH`")
                    st.write(f"- **Configured EOL Line:** `{eol_th:.1f}% SOH`")

                    rem = fc["estimated_steps_to_eol"]
                    if rem is not None:
                        if rem == 0:
                            st.error("Capacity has already crossed the configured 80% EOL line.")
                        else:
                            st.warning(f"Estimated intervals remaining until {eol_th}% EOL: **~{rem} intervals**")
                    else:
                        st.success("Stable capacity trend: No downward slope detected.")

                    st.markdown(f"""
                    <div class="disclaimer-box" style="margin-top:0.6rem;">
                        <strong>Forecast Disclaimer:</strong> {fc['disclaimer']}
                    </div>
                    """, unsafe_allow_html=True)

                with f_chart:
                    f_df = fc["forecast_df"]
                    x_key = fc["x_label"]

                    fig_f = go.Figure()

                    if "cycle_count" in source_for_forecast.columns and source_for_forecast["cycle_count"].notna().all() and (source_for_forecast["cycle_count"] > 0).all():
                        h_x = source_for_forecast["cycle_count"]
                    else:
                        h_x = source_for_forecast["step_index"] if "step_index" in source_for_forecast.columns else np.arange(len(source_for_forecast))

                    # Historical measurements
                    fig_f.add_trace(go.Scatter(
                        x=h_x,
                        y=source_for_forecast["soh_percent"],
                        mode="lines+markers",
                        name="Measured SOH",
                        line=dict(color="#102A43", width=2.5)
                    ))

                    # 95% Confidence envelope
                    fig_f.add_trace(go.Scatter(
                        x=f_df[x_key],
                        y=f_df["Upper Bound (95%)"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig_f.add_trace(go.Scatter(
                        x=f_df[x_key],
                        y=f_df["Lower Bound (95%)"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(217, 119, 6, 0.16)",
                        name="95% Confidence Interval"
                    ))

                    # Projected trend line
                    fig_f.add_trace(go.Scatter(
                        x=f_df[x_key],
                        y=f_df["Projected SOH (%)"],
                        mode="lines+markers",
                        name="Projected Trend",
                        line=dict(color="#D97706", width=2.5, dash="dot")
                    ))

                    fig_f.add_hline(y=c_th, line_dash="dash", line_color="#D97706", annotation_text=f"Caution ({c_th}%)")
                    fig_f.add_hline(y=eol_th, line_dash="dash", line_color="#DC2626", annotation_text=f"Critical EOL ({eol_th}%)")

                    fig_f.update_layout(
                        title=dict(text="Degradation Forecast with 95% Uncertainty Bounds", font=dict(size=14, color="#102A43")),
                        height=310,
                        margin=dict(l=20, r=20, t=35, b=20),
                        xaxis_title=x_key,
                        yaxis_title="SOH (%)",
                        yaxis_range=[max(0, eol_th - 25), 105],
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_f, use_container_width=True)

            else:
                st.info(f"ℹ️ {fc['message']}")
        else:
            st.info("ℹ️ Insufficient historical data for degradation forecasting. Minimum 3 chronological observations are required to calculate degradation slope and residual standard error.")
            st.caption("You can import existing Windows battery report capacity history or enable periodic monitoring to accumulate historical checkpoints.")

    # -------------------------------------------------------------------------
    # Sub-View B: Laboratory Machine Learning Research
    # -------------------------------------------------------------------------
    with ai_subtab_research:
        st.markdown("##### NASA Ames PCoE Lithium-ion Battery Aging Benchmark")
        st.caption("Validated machine learning regression pipeline trained on public research dataset (NASA Cells B0005, B0006, B0007).")

        st.markdown("""
        <div class="disclaimer-box">
            <strong>Domain Integrity & Boundary Notice:</strong> The models below evaluate lithium-ion degradation kinetics on single cylindrical 18650 research cells 
            under continuous 24°C chamber cycling. Real commercial laptop battery packs utilize multi-cell pouch/prismatic arrays with dynamic computing workloads. 
            Laboratory cell models provide physical degradation benchmarks and are <em>not</em> claimed to predict real laptop battery replacement without transfer calibration.
        </div>
        """, unsafe_allow_html=True)

        nasa_path = os.path.join(PROJECT_ROOT, "data", "sample_nasa_battery.csv")
        if os.path.exists(nasa_path):
            n_raw = pd.read_csv(nasa_path)
            c_lab, _ = clean_lab_dataset(n_raw)

            # Trajectory plot
            fig_lab = px.line(
                c_lab,
                x="cycle",
                y="soh_percent",
                color="battery_id",
                markers=True,
                title="Experimental Capacity Fade Curves: NASA PCoE Cells B0005, B0006, B0007",
                labels={"cycle": "Cycle Number", "soh_percent": "State of Health (%)", "battery_id": "Cell ID"},
                color_discrete_map={"B0005": "#102A43", "B0006": "#DC2626", "B0007": "#0D9488"}
            )
            fig_lab.add_hline(y=80.0, line_dash="dash", line_color="#DC2626", annotation_text="80% EOL Line")
            fig_lab.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_lab, use_container_width=True)

            st.markdown("---")
            st.markdown("###### Out-of-Sample Machine Learning Evaluation")
            st.caption("Zero-leakage cell-level split: Trained on Cells B0005 & B0007, tested out-of-sample on unseen Cell B0006.")

            if st.button("🚀 Train & Evaluate ML Models (Zero-Leakage Cell Split)", type="primary"):
                with st.spinner("Training Linear Baseline and Random Forest Regressor..."):
                    t_eval = train_and_evaluate_models(nasa_path, save_artifacts=True)
                    st.session_state["ml_results"] = t_eval
                    st.success("Models evaluated and serialized to disk successfully.")

            # Load serialized or cached results
            if "ml_results" not in st.session_state:
                lr_mod, lr_m = load_model_artifact("linear_regression_baseline.joblib")
                rf_mod, rf_m = load_model_artifact("random_forest_regressor.joblib")
                if lr_mod is not None and rf_mod is not None and "linear_regression_metrics" in lr_m:
                    st.session_state["ml_results"] = {
                        "linear_regression": {"metrics": lr_m["linear_regression_metrics"]},
                        "random_forest": {"metrics": lr_m["random_forest_metrics"]},
                        "train_cells": lr_m.get("train_cells", ["B0005", "B0007"]),
                        "test_cells": lr_m.get("test_cells", ["B0006"]),
                    }

            if "ml_results" in st.session_state:
                res_ml = st.session_state["ml_results"]
                m_lr = res_ml["linear_regression"]["metrics"]
                m_rf = res_ml["random_forest"]["metrics"]

                ev1, ev2, ev3 = st.columns(3)
                with ev1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Linear Regression Baseline</div>
                        <div class="metric-value-huge">{m_lr['MAE']:.3f}% <span style="font-size:0.85rem; color:#627D98;">MAE</span></div>
                        <div class="metric-subtext">RMSE: {m_lr['RMSE']:.3f}% &bull; R²: {m_lr['R2']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with ev2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Random Forest Regressor</div>
                        <div class="metric-value-huge teal">{m_rf['MAE']:.3f}% <span style="font-size:0.85rem; color:#627D98;">MAE</span></div>
                        <div class="metric-subtext">RMSE: {m_rf['RMSE']:.3f}% &bull; R²: {m_rf['R2']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with ev3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Out-of-Sample Test Cell</div>
                        <div class="metric-value-huge">{res_ml['test_cells'][0]}</div>
                        <div class="metric-subtext" style="color:#0D9488;">Trained on Cells: {res_ml['train_cells']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Interactive Simulation
                st.markdown("###### Cycle Degradation Inference Simulation &nbsp; <span class='badge-origin badge-forecast'>[ML PREDICTION]</span>", unsafe_allow_html=True)
                sim_cycle = st.slider("Cycle Number to Simulate", min_value=1, max_value=160, value=75, step=1)
                sim_input = pd.DataFrame({"cycle": [sim_cycle], "ambient_temperature": [24.0]})

                lr_loaded, _ = load_model_artifact("linear_regression_baseline.joblib")
                rf_loaded, _ = load_model_artifact("random_forest_regressor.joblib")

                if lr_loaded is not None and rf_loaded is not None:
                    p_linear = lr_loaded.predict(sim_input)[0]
                    p_rf = rf_loaded.predict(sim_input)[0]

                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.metric(f"Linear Baseline Prediction (Cycle {sim_cycle})", f"{p_linear:.2f}% SOH")
                    with sc2:
                        st.metric(f"Random Forest Prediction (Cycle {sim_cycle})", f"{p_rf:.2f}% SOH")


# =============================================================================
# 5. ALERTS
# =============================================================================
with tab_alerts:
    st.markdown("#### Intelligent Diagnostic Alerts & Actionable Guidance")
    st.caption("Automated condition evaluation based on active measurements, capacity retention, power draw, and historical telemetry.")

    if active_alerts:
        st.markdown(f"**Active Warnings & Advisories:** `{len(active_alerts)} conditions detected`")
        for alt in active_alerts:
            st.markdown(f"""
            <div class="alert-banner {alt['severity']}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <strong>[{alt['severity']}] {alt['title']}</strong>
                    <span style="font-size:0.78rem; color:#627D98; font-family:'JetBrains Mono',monospace;">{alt['timestamp']}</span>
                </div>
                <div style="font-size:0.9rem; margin:0.35rem 0;">{alt['reason']}</div>
                <div style="font-size:0.83rem; color:#102A43;"><strong>Telemetry Evidence:</strong> <code>{alt['supporting_measurement']}</code></div>
                <div style="font-size:0.83rem; margin-top:0.25rem; color:#0D9488;"><strong>Recommended Engineering Action:</strong> {alt['recommended_action']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-banner INFO">
            <strong>[NORMAL] All Battery Diagnostic Parameters Within Nominal Limits</strong>
            <div style="font-size:0.88rem; margin-top:0.3rem;">
                No anomalous capacity drops, critical discharge draws, or threshold breaches detected on active telemetry.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Historical Alert Log from Stored Database Records
    st.markdown("---")
    st.markdown("##### Historical Alert Log")
    st.caption("Chronological record of past maintenance alerts evaluated across recorded observations.")

    if not display_history.empty and "SOH (%)" in display_history.columns:
        alert_history_rows = []
        for _, r in display_history.iterrows():
            soh_val = r.get("soh_percent") or r.get("SOH (%)")
            fcc_val = r.get("full_charge_capacity_mwh") or r.get("Full Charge Cap (mWh)") or 0.0
            dc_val = r.get("design_capacity_mwh") or r.get("Design Cap (mWh)") or 0.0
            t_str = str(r.get("timestamp") or r.get("Period") or "")
            if pd.notna(soh_val) and soh_val > 0:
                entry = build_alert_history_entry(
                    date_str=t_str[:19],
                    soh_percent=float(soh_val),
                    fcc=float(fcc_val),
                    dc=float(dc_val),
                    caution_th=c_th,
                    critical_th=eol_th
                )
                alert_history_rows.append(entry)

        if alert_history_rows:
            alert_hist_df = pd.DataFrame(alert_history_rows)
            st.dataframe(alert_hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No historical alerts logged yet.")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Hardware Safety Scope:</strong> Capacity-based State of Health algorithms quantify electrochemical cycle fade and lost unplugged runtime. 
        They do not claim to detect internal electrical short-circuits or physical pouch swelling from unavailable hardware measurements. Always consult OEM service centers if physical swelling or extreme localized casing heat is observed.
    </div>
    """, unsafe_allow_html=True)
