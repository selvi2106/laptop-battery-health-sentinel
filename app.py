"""
app.py
======
Laptop Battery Health Sentinel
Professional Hardware Diagnostics, Longitudinal SOH Tracking, and Degradation Forecasting.
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure project root is available in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    delete_observation,
    clear_all_history,
    export_history_to_csv,
)
from src.data_loader import load_dataset
from src.preprocessing import clean_lab_dataset, prepare_laptop_history_series, normalize_history_dataframe
from src.train_model import train_and_evaluate_models
from src.prediction import (
    load_model_artifact,
    forecast_laptop_degradation,
    DEFAULT_EOL_THRESHOLD,
)

# -----------------------------------------------------------------------------
# Page Configuration & Professional Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laptop Battery Health Sentinel",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Bar */
    .telemetry-header {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.9rem 1.4rem;
        color: #f8fafc;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
    }
    .telemetry-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .telemetry-subtitle {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.35rem 0.75rem;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        color: #e2e8f0;
    }
    
    /* Professional Card */
    .metric-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
    }
    .metric-tag {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
        font-feature-settings: "tnum";
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 0.2rem;
        font-weight: 500;
    }
    
    /* Classification badges */
    .badge-optimal {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-caution {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-critical {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    
    /* Origin badges */
    .origin-actual {
        font-size: 0.68rem;
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        border-radius: 3px;
        padding: 0.1rem 0.35rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .origin-calc {
        font-size: 0.68rem;
        background: #f5f3ff;
        color: #5b21b6;
        border: 1px solid #ddd6fe;
        border-radius: 3px;
        padding: 0.1rem 0.35rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .origin-forecast {
        font-size: 0.68rem;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
        border-radius: 3px;
        padding: 0.1rem 0.35rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .origin-demo {
        font-size: 0.68rem;
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
        border-radius: 3px;
        padding: 0.1rem 0.35rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Alerts */
    .alert-card {
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid transparent;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .alert-card.CRITICAL {
        border-left: 4px solid #ef4444;
        background: #fef2f2;
    }
    .alert-card.WARNING, .alert-card.CAUTION {
        border-left: 4px solid #f59e0b;
        background: #fffbeb;
    }
    .alert-card.ADVISORY, .alert-card.INFO {
        border-left: 4px solid #3b82f6;
        background: #eff6ff;
    }
    
    /* Engineering Disclaimers */
    .engineering-notice {
        background: #f8fafc;
        border-left: 3px solid #64748b;
        padding: 0.6rem 0.9rem;
        font-size: 0.8rem;
        color: #475569;
        border-radius: 4px;
        margin: 0.9rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration & Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ Sentinel Control")
st.sidebar.caption("Laptop Hardware Diagnostics & SOH")

nav_view = st.sidebar.radio(
    "Workstation",
    [
        "System Diagnostics & Live Monitor",
        "Historical Audit Log (Local Database)",
        "Degradation Intelligence Lab (Research ML)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Threshold Parameters")

caution_th = st.sidebar.slider(
    "Advisory / Caution Threshold (% SOH)",
    min_value=75.0,
    max_value=95.0,
    value=DEFAULT_CAUTION_THRESHOLD,
    step=1.0,
    help="Battery retention falling below this level triggers a maintenance advisory."
)

critical_th = st.sidebar.slider(
    "End-of-Life (EOL) Threshold (% SOH)",
    min_value=50.0,
    max_value=85.0,
    value=DEFAULT_CRITICAL_THRESHOLD,
    step=1.0,
    help="Industry standard lithium-ion retirement threshold (80% SOH)."
)

if critical_th >= caution_th:
    st.sidebar.error("Critical threshold must be strictly lower than Caution threshold.")
    critical_th = caution_th - 5.0

# Model persistence check
lr_model, lr_meta = load_model_artifact("linear_regression_baseline.joblib")
rf_model, rf_meta = load_model_artifact("random_forest_regressor.joblib")

st.sidebar.markdown("---")
if lr_model is not None and rf_model is not None:
    st.sidebar.success("Research Models: Serialized & Ready")
    if "test_cells" in lr_meta:
        st.sidebar.caption(f"Evaluated on NASA Cell: {lr_meta.get('test_cells')}")
else:
    st.sidebar.info("Research Models: Uncompiled (Monitoring Active)")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def classify_health(soh: float, c_th: float, eol_th: float) -> tuple[str, str]:
    if soh >= 95.0:
        return "Optimal (Pristine)", "badge-optimal"
    elif soh >= c_th:
        return "Optimal (Good)", "badge-optimal"
    elif soh >= eol_th:
        return "Degraded (Caution)", "badge-caution"
    else:
        return "Critical (EOL Reached)", "badge-critical"


def render_soh_gauge(soh: float, c_th: float, eol_th: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={'text': "Capacity State of Health (SOH)", 'font': {'size': 15, 'color': '#0f172a'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#0f172a'}},
        gauge={
            'axis': {'range': [0, 110], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#1e293b", 'thickness': 0.28},
            'bgcolor': "#f8fafc",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, eol_th], 'color': "rgba(239, 68, 68, 0.18)"},
                {'range': [eol_th, c_th], 'color': "rgba(245, 158, 11, 0.18)"},
                {'range': [c_th, 110], 'color': "rgba(34, 197, 94, 0.18)"}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 3},
                'thickness': 0.8,
                'value': eol_th
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# -----------------------------------------------------------------------------
# WORKSTATION 1: SYSTEM DIAGNOSTICS & LIVE MONITOR
# -----------------------------------------------------------------------------
if nav_view == "System Diagnostics & Live Monitor":
    # 1. Fetch live hardware telemetry
    live_status = get_live_battery_status()

    # Determine default state
    if "current_report" not in st.session_state:
        # Generate initial live Windows report if on NT
        if os.name == "nt":
            ok, report_p = generate_battery_report()
            if ok:
                try:
                    with open(report_p, "r", encoding="utf-8", errors="ignore") as f:
                        st.session_state["current_report"] = parse_battery_report_html(f.read())
                        st.session_state["data_source_tag"] = "WINDOWS_HARDWARE_REPORT"
                except Exception:
                    pass

    # If still not populated, fallback to sample benchmark profile
    if "current_report" not in st.session_state:
        sample_hist_path = os.path.join(PROJECT_ROOT, "data", "sample_laptop_history.csv")
        if os.path.exists(sample_hist_path):
            s_raw = pd.read_csv(sample_hist_path)
            _, h_prepared, _ = prepare_laptop_history_series(s_raw)
            latest_s = h_prepared.iloc[-1]
            st.session_state["current_report"] = {
                "success": True,
                "system_info": {"computer_name": "BENCHMARK-SYSTEM", "product_name": "Standard Laptop Battery Benchmark"},
                "battery_info": {
                    "name": "Benchmark Pack",
                    "manufacturer": "OEM",
                    "chemistry": "Li-Ion",
                    "design_capacity": float(latest_s["design_capacity_mwh"]),
                    "full_charge_capacity": float(latest_s["full_charge_capacity_mwh"]),
                    "cycle_count": int(latest_s["cycle_count"]) if "cycle_count" in latest_s else None
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
    curr_cycles = b_info.get("cycle_count")

    # Top Status Bar
    host_name = s_info.get("product_name") or s_info.get("computer_name") or "Local Host"
    batt_dev = b_info.get("name") or live_status.get("device_name") or "System Battery"
    chem = b_info.get("chemistry") or "Li-Ion"
    mfg = b_info.get("manufacturer") or "OEM"

    charge_display = f"{live_status['charge_percent']}%" if live_status["charge_percent"] is not None else "N/A"
    power_display = live_status["status_text"] if live_status["status_text"] else "Offline"
    runtime_display = f"{live_status['estimated_runtime_minutes']} min runtime" if live_status["estimated_runtime_minutes"] else None

    pill_source = "[ACTUAL MEASUREMENT]" if source_tag == "WINDOWS_HARDWARE_REPORT" else "[DEMO BENCHMARK]"

    st.markdown(f"""
    <div class="telemetry-header">
        <div>
            <div class="telemetry-title">
                Hardware Diagnostics: {host_name}
            </div>
            <div class="telemetry-subtitle">
                Battery Unit: <strong>{batt_dev}</strong> ({mfg} {chem}) &bull; Last Polled: {live_status['timestamp']}
            </div>
        </div>
        <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
            <div class="telemetry-pill">
                <span>Charge:</span> <strong>{charge_display}</strong>
            </div>
            <div class="telemetry-pill">
                <span>Power:</span> <strong>{power_display}</strong>
            </div>
            <div class="telemetry-pill">
                <span>Source:</span> <strong>{pill_source}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Controls
    with st.expander("Telemetry Controls & Input Sources", expanded=False):
        c_act1, c_act2, c_act3 = st.columns([1, 1, 1])
        with c_act1:
            st.markdown("##### 1. Refresh Windows Battery Scan")
            st.caption("Executes `powercfg /batteryreport` (safe non-elevated).")
            if st.button("Run Windows Scan", type="primary", use_container_width=True):
                with st.spinner("Extracting battery report from system..."):
                    s_ok, p_out = generate_battery_report()
                    if s_ok:
                        try:
                            with open(p_out, "r", encoding="utf-8", errors="ignore") as f:
                                st.session_state["current_report"] = parse_battery_report_html(f.read())
                                st.session_state["data_source_tag"] = "WINDOWS_HARDWARE_REPORT"
                            st.success("Refreshed from live Windows report.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Parse error: {str(e)}")
                    else:
                        st.warning(p_out)

        with c_act2:
            st.markdown("##### 2. Upload Battery Report HTML")
            st.caption("Import an existing report from another system.")
            u_html = st.file_uploader("Select HTML file", type=["html", "htm"], key="diag_html_up")
            if u_html is not None:
                parsed_u = parse_battery_report_html(u_html.read().decode("utf-8", errors="ignore"))
                if parsed_u["success"]:
                    st.session_state["current_report"] = parsed_u
                    st.session_state["data_source_tag"] = f"UPLOADED_FILE ({u_html.name})"
                    st.success("Uploaded report loaded.")
                    st.rerun()
                else:
                    st.error(parsed_u["error"])

        with c_act3:
            st.markdown("##### 3. Manual Parameter Entry")
            st.caption("Specify hardware parameters directly.")
            with st.form("manual_entry_form"):
                in_dc = st.number_input("Design Capacity (mWh)", min_value=1000.0, max_value=150000.0, value=float(curr_dc or 52500.0), step=500.0)
                in_fcc = st.number_input("Full Charge Capacity (mWh)", min_value=100.0, max_value=150000.0, value=float(curr_fcc or 46200.0), step=500.0)
                in_cyc = st.number_input("Cycle Count", min_value=0, max_value=3000, value=int(curr_cycles or 120), step=1)
                if st.form_submit_button("Apply Parameters", use_container_width=True):
                    st.session_state["current_report"] = {
                        "success": True,
                        "system_info": {"computer_name": "MANUAL-INPUT", "product_name": "User Defined Hardware Profile"},
                        "battery_info": {
                            "name": "Custom Hardware Input",
                            "manufacturer": "Manual",
                            "chemistry": "Li-Ion",
                            "design_capacity": in_dc,
                            "full_charge_capacity": in_fcc,
                            "cycle_count": in_cyc
                        },
                        "history_records": hist_records
                    }
                    st.session_state["data_source_tag"] = "MANUAL_INPUT"
                    st.success("Applied custom parameters.")
                    st.rerun()

    # -------------------------------------------------------------------------
    # Core Capacity Metrics Display
    # -------------------------------------------------------------------------
    soh_res = calculate_soh(curr_fcc, curr_dc)

    if soh_res["is_valid"]:
        active_soh = soh_res["soh_percent"]
        lost_mwh = max(0.0, curr_dc - curr_fcc)
        grade_str, grade_css = classify_health(active_soh, caution_th, critical_th)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-tag">
                    State of Health <span class="origin-calc">[CALCULATED]</span>
                </div>
                <div class="metric-value">{active_soh:.1f}%</div>
                <div class="metric-sub"><span class="{grade_css}">{grade_str}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-tag">
                    Full Charge Capacity <span class="origin-actual">[MEASURED]</span>
                </div>
                <div class="metric-value">{curr_fcc:,.0f} <span style="font-size:0.95rem; font-weight:500; color:#64748b;">mWh</span></div>
                <div class="metric-sub" style="color:#b91c1c;">&minus;{lost_mwh:,.0f} mWh capacity lost</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-tag">
                    Design Capacity <span class="origin-actual">[MEASURED]</span>
                </div>
                <div class="metric-value">{curr_dc:,.0f} <span style="font-size:0.95rem; font-weight:500; color:#64748b;">mWh</span></div>
                <div class="metric-sub" style="color:#15803d;">Original Factory Rating</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            cyc_val = f"{curr_cycles}" if curr_cycles is not None else "Not reported"
            cyc_pct = f"~{(curr_cycles / 500.0) * 100:.0f}% of 500-cycle life" if curr_cycles else "BMS telemetry omitted"
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-tag">
                    Cycle Count <span class="origin-actual">[MEASURED]</span>
                </div>
                <div class="metric-value">{cyc_val}</div>
                <div class="metric-sub" style="color:#64748b;">{cyc_pct}</div>
            </div>
            """, unsafe_allow_html=True)

        # Snapshot Persistence Button
        col_snap, col_empty = st.columns([1, 2])
        with col_snap:
            if st.button("💾 Record Snapshot to Local History Database", use_container_width=True):
                s_ok, s_msg, _ = record_observation(
                    design_capacity_mwh=curr_dc,
                    full_charge_capacity_mwh=curr_fcc,
                    soh_percent=active_soh,
                    data_source=source_tag,
                    device_name=batt_dev,
                    manufacturer=mfg,
                    chemistry=chem,
                    cycle_count=curr_cycles,
                    charge_percent=live_status.get("charge_percent"),
                    power_status=live_status.get("status_text")
                )
                if s_ok:
                    st.success(s_msg)
                else:
                    st.info(s_msg)

        # Primary Visualizations: Gauge & Capacity Trend
        g_col, c_col = st.columns([1, 1.6])
        with g_col:
            st.plotly_chart(render_soh_gauge(active_soh, caution_th, critical_th), use_container_width=True)

        with c_col:
            if not hist_records.empty:
                x_axis = hist_records["Period"]
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(
                    x=x_axis, y=hist_records["Design Capacity (mWh)"],
                    name="Design Capacity",
                    line=dict(color="#94a3b8", dash="dash", width=1.5)
                ))
                fig_t.add_trace(go.Scatter(
                    x=x_axis, y=hist_records["Full Charge Capacity (mWh)"],
                    name="Full Charge Capacity",
                    line=dict(color="#0f172a", width=2.5),
                    fill="tonexty",
                    fillcolor="rgba(15, 23, 42, 0.05)"
                ))
                fig_t.update_layout(
                    title=dict(text="Capacity Retention History (Measured)", font=dict(size=14)),
                    height=240,
                    margin=dict(l=20, r=20, t=35, b=20),
                    yaxis_title="Capacity (mWh)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("No historical battery observations available yet.")

        # ---------------------------------------------------------------------
        # Degradation Forecasting & Residual Uncertainty Envelope
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### Degradation Forecasting & RUL Estimation &nbsp; <span class='origin-forecast'>[STATISTICAL EXTRAPOLATION]</span>", unsafe_allow_html=True)

        if hist_records.empty:
            st.info("No historical battery observations available yet for degradation forecasting.")
        else:
            fc_result = forecast_laptop_degradation(hist_records, future_steps=8, eol_threshold=critical_th)

            if fc_result["is_forecastable"]:
                fc_info, fc_plot = st.columns([1, 1.8])
                with fc_info:
                    st.markdown("##### Extrapolation Metrics")
                    st.write(f"- **Current Capacity SOH:** `{fc_result['current_soh']}%`")
                    st.write(f"- **Degradation Slope:** `{fc_result['degradation_rate_per_step']}% per observation interval`")
                    st.write(f"- **Residual Standard Error (SE):** `±{fc_result['residual_std_error']}% SOH`")
                    st.write(f"- **End-of-Life Target Line:** `{critical_th}% SOH`")

                    rem_steps = fc_result["estimated_steps_to_eol"]
                    if rem_steps is not None:
                        if rem_steps == 0:
                            st.error("Capacity has already crossed the configured 80% EOL threshold.")
                        else:
                            st.warning(f"Estimated intervals remaining until {critical_th}% EOL: **~{rem_steps} intervals**")
                    else:
                        st.success("Stable capacity trend: No downward slope detected.")

                    st.caption(f"ℹ️ {fc_result['disclaimer']}")

                with fc_plot:
                    f_df = fc_result["forecast_df"]
                    x_lbl = fc_result["x_label"]

                    fig_fc = go.Figure()

                    if "cycle_count" in hist_records.columns and hist_records["cycle_count"].notna().all() and (hist_records["cycle_count"] > 0).all():
                        h_x = hist_records["cycle_count"]
                    else:
                        h_x = hist_records["step_index"] if "step_index" in hist_records.columns else np.arange(len(hist_records))

                    h_soh = hist_records["soh_percent"]

                    # Historical Measured
                    fig_fc.add_trace(go.Scatter(
                        x=h_x, y=h_soh,
                        mode="lines+markers",
                        name="Measured SOH",
                        line=dict(color="#0f172a", width=2.5)
                    ))

                    # 95% Confidence Envelope
                    fig_fc.add_trace(go.Scatter(
                        x=f_df[x_lbl], y=f_df["Upper Bound (95%)"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig_fc.add_trace(go.Scatter(
                        x=f_df[x_lbl], y=f_df["Lower Bound (95%)"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(245, 158, 11, 0.15)",
                        name="95% Confidence Interval"
                    ))

                    # Projected Trend Line
                    fig_fc.add_trace(go.Scatter(
                        x=f_df[x_lbl], y=f_df["Projected SOH (%)"],
                        mode="lines+markers",
                        name="Projected Trend",
                        line=dict(color="#d97706", width=2.5, dash="dot")
                    ))

                    fig_fc.add_hline(y=caution_th, line_dash="dash", line_color="#d97706", annotation_text=f"Caution ({caution_th}%)")
                    fig_fc.add_hline(y=critical_th, line_dash="dash", line_color="#dc2626", annotation_text=f"Critical EOL ({critical_th}%)")

                    fig_fc.update_layout(
                        title=dict(text="Degradation Forecast with 95% Uncertainty Envelope", font=dict(size=14)),
                        height=280,
                        margin=dict(l=20, r=20, t=35, b=20),
                        xaxis_title=x_lbl,
                        yaxis_title="SOH (%)",
                        yaxis_range=[max(0, critical_th - 25), 105],
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.info(f"ℹ️ {fc_result['message']}")

        # ---------------------------------------------------------------------
        # Active Alerts Sentinel
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### Active System Alerts & Diagnostics")

        last_scan_ts = s_info.get("report_time") or live_status.get("timestamp")
        active_alerts = generate_system_alerts(
            current_soh=active_soh,
            current_fcc=curr_fcc,
            current_dc=curr_dc,
            history_df=hist_records,
            last_scan_date=last_scan_ts,
            caution_th=caution_th,
            critical_th=critical_th
        )

        if active_alerts:
            for alt in active_alerts:
                st.markdown(f"""
                <div class="alert-card {alt['severity']}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>[{alt['severity']}] {alt['title']}</strong>
                        <span style="font-size:0.75rem; color:#64748b;">{alt['timestamp']}</span>
                    </div>
                    <div style="font-size:0.88rem; margin:0.35rem 0;">{alt['reason']}</div>
                    <div style="font-size:0.82rem; color:#334155;"><strong>Evidence:</strong> {alt['supporting_measurement']}</div>
                    <div style="font-size:0.82rem; margin-top:0.25rem;"><strong>Recommended Action:</strong> {alt['recommended_action']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No active anomalies detected. Battery operates within normal capacity degradation limits.")

        st.markdown("""
        <div class="engineering-notice">
            <strong>Hardware Diagnostics Disclaimer:</strong> Capacity-based State of Health quantifies gradual electrochemical capacity fade 
            and lost runtime. It does NOT monitor thermal runaway, internal micro-shorts, or physical pouch swelling. Always follow OEM hardware safety guidelines.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error(f"SOH Calculation Failure: {soh_res['message']}")


# -----------------------------------------------------------------------------
# WORKSTATION 2: HISTORICAL AUDIT LOG (LOCAL DATABASE)
# -----------------------------------------------------------------------------
elif nav_view == "Historical Audit Log (Local Database)":
    st.markdown("### 🗄️ Historical Audit Log & Local Persistence")
    st.caption("Inspect and manage locally recorded battery observations stored in SQLite.")

    obs_df = get_all_observations()

    if not obs_df.empty:
        st.markdown(f"**Total Stored Observations:** `{len(obs_df)} records`")

        st.dataframe(obs_df, use_container_width=True, hide_index=True)

        col_exp, col_del = st.columns([1, 1])
        with col_exp:
            csv_bytes = export_history_to_csv()
            st.download_button(
                label="📥 Export Full Audit Log as CSV",
                data=csv_bytes,
                file_name=f"battery_sentinel_audit_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_del:
            with st.expander("Record Management (Delete)", expanded=False):
                del_id = st.number_input("Enter Record ID to delete", min_value=1, step=1)
                if st.button("Delete Specific Record", type="secondary"):
                    if delete_observation(int(del_id)):
                        st.success(f"Record #{del_id} deleted.")
                        st.rerun()
                    else:
                        st.error(f"Record #{del_id} not found.")

                if st.button("Purge All Local History", type="primary"):
                    cnt = clear_all_history()
                    st.success(f"Purged {cnt} records from local database.")
                    st.rerun()
    else:
        st.info("No observations recorded in local storage yet. Click 'Record Snapshot to Local History Database' in the Diagnostics tab to capture timestamped snapshots.")


# -----------------------------------------------------------------------------
# WORKSTATION 3: DEGRADATION INTELLIGENCE LAB (RESEARCH ML)
# -----------------------------------------------------------------------------
elif nav_view == "Degradation Intelligence Lab (Research ML)":
    st.markdown("### 🔬 Degradation Intelligence Lab & ML Research")
    st.caption("Empirical machine learning models evaluated on public laboratory cycling data (NASA Ames PCoE 18650 cells).")

    st.markdown("""
    <div class="engineering-notice">
        <strong>Domain Integrity Note:</strong> The models below evaluate lithium-ion degradation kinetics on single cylindrical 18650 research cells 
        under continuous 24°C chamber cycling. Real commercial laptop battery packs utilize multi-cell arrays with dynamic computer workloads. 
        Laboratory cell models provide physical degradation benchmarks and are not validated for direct laptop battery warranty replacement without transfer adaptation.
    </div>
    """, unsafe_allow_html=True)

    nasa_p = os.path.join(PROJECT_ROOT, "data", "sample_nasa_battery.csv")
    if os.path.exists(nasa_p):
        nasa_df = pd.read_csv(nasa_p)
        cleaned_lab, _ = clean_lab_dataset(nasa_df)

        st.markdown("#### NASA 18650 Cycling Degradation Trajectories")
        fig_cells = px.line(
            cleaned_lab,
            x="cycle",
            y="soh_percent",
            color="battery_id",
            markers=True,
            title="Capacity Fade Curves: NASA PCoE Cells B0005, B0006, B0007",
            labels={"cycle": "Cycle Number", "soh_percent": "State of Health (%)", "battery_id": "Cell ID"}
        )
        fig_cells.add_hline(y=80.0, line_dash="dash", line_color="#dc2626", annotation_text="80% EOL Line")
        fig_cells.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_cells, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Out-of-Sample Machine Learning Evaluation")
        st.caption("Zero-leakage cell-level split: Trained on Cells B0005 & B0007, tested out-of-sample on unseen Cell B0006.")

        if st.button("Train & Evaluate ML Models", type="primary"):
            with st.spinner("Training models with out-of-sample cell split..."):
                t_res = train_and_evaluate_models(nasa_p, save_artifacts=True)
                st.session_state["ml_train_results"] = t_res
                st.success("Trained and serialized models successfully.")

        if "ml_train_results" in st.session_state:
            t_res = st.session_state["ml_train_results"]
            lr_m = t_res["linear_regression"]["metrics"]
            rf_m = t_res["random_forest"]["metrics"]

            c_lr, c_rf, c_split = st.columns(3)
            with c_lr:
                st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-tag">Linear Regression Baseline</div>
                    <div class="metric-value">{lr_m['MAE']:.3f}% <span style="font-size:0.9rem; font-weight:500; color:#64748b;">MAE</span></div>
                    <div class="metric-sub">RMSE: {lr_m['RMSE']:.3f}% | R²: {lr_m['R2']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c_rf:
                st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-tag">Random Forest Regressor</div>
                    <div class="metric-value">{rf_m['MAE']:.3f}% <span style="font-size:0.9rem; font-weight:500; color:#64748b;">MAE</span></div>
                    <div class="metric-sub">RMSE: {rf_m['RMSE']:.3f}% | R²: {rf_m['R2']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c_split:
                st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-tag">Out-of-Sample Test Cell</div>
                    <div class="metric-value">{t_res['test_cells'][0]}</div>
                    <div class="metric-sub" style="color:#15803d;">Train Cells: {t_res['train_cells']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Test Cell Prediction Plot
            eval_df = t_res["test_evaluation_df"]
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(
                x=eval_df["cycle"], y=eval_df["actual_soh"],
                mode="lines+markers", name=f"Ground Truth (Cell {t_res['test_cells'][0]})",
                line=dict(color="#0f172a", width=2.5)
            ))
            fig_eval.add_trace(go.Scatter(
                x=eval_df["cycle"], y=eval_df["pred_lr"],
                mode="lines", name="Linear Regression Baseline",
                line=dict(color="#2563eb", width=2, dash="dash")
            ))
            fig_eval.add_trace(go.Scatter(
                x=eval_df["cycle"], y=eval_df["pred_rf"],
                mode="lines", name="Random Forest Regressor",
                line=dict(color="#16a34a", width=2)
            ))
            fig_eval.update_layout(
                title=dict(text=f"Out-of-Sample Generalization on Test Cell {t_res['test_cells'][0]}", font=dict(size=14)),
                xaxis_title="Cycle Number",
                yaxis_title="SOH (%)",
                height=340,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_eval, use_container_width=True)

            # Interactive cycle simulation
            st.markdown("##### Cycle Degradation Simulation &nbsp; <span class='origin-forecast'>[ML PREDICTION]</span>", unsafe_allow_html=True)
            sim_c = st.slider("Cycle Number to Simulate", min_value=1, max_value=160, value=75, step=1)
            in_sim = pd.DataFrame({"cycle": [sim_c], "ambient_temperature": [24.0]})

            if lr_model is not None and rf_model is not None:
                p_lr = lr_model.predict(in_sim)[0]
                p_rf = rf_model.predict(in_sim)[0]

                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    st.metric(f"Linear Regression Prediction (Cycle {sim_c})", f"{p_lr:.2f}% SOH")
                with s_c2:
                    st.metric(f"Random Forest Prediction (Cycle {sim_c})", f"{p_rf:.2f}% SOH")
