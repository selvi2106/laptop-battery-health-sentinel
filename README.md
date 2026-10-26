# 🔋 Laptop Battery Health Sentinel
## Live Demo

[Open Laptop Battery Health Intelligence System](https://laptop-battery-health-sentinel-ruhyhyqkfa4osvcyrtvxzd.streamlit.app/)
### Intelligent Battery Telemetry, Longitudinal SOH Estimation & Degradation Forecasting

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Visualizations](https://img.shields.io/badge/Visuals-Plotly-239120.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-40%20Passed-brightgreen.svg)]()

---

## 📌 1. Overview

**Laptop Battery Health Sentinel** is a practical, production-grade battery diagnostics and degradation intelligence workstation. It bridges real-world laptop hardware monitoring with machine learning research on lithium-ion degradation kinetics.

### Key Capabilities:
- **Direct Windows Battery Integration**: Safe non-elevated queries against Windows CIM/WMI hardware interfaces and 1-click execution of `powercfg /batteryreport`.
- **Accurate Capacity-Based SOH**: Computes State of Health (SOH) strictly using physical capacities:
  $$\text{SOH} (\%) = \left( \frac{\text{Full Charge Capacity}}{\text{Design Capacity}} \right) \times 100$$
- **Real-Time Telemetry Tracking**: Live charge percentage, charging/discharging/AC state, live terminal voltage (V), design voltage (V), power discharge/charge rates (W), and estimated unplugged runtime. Gracefully distinguishes measured telemetry from hardware-omitted metrics (e.g. cycle count or temperature).
- **Persistent Local Database (SQLite)**: Stores timestamped battery observations with delta tracking, date filtering, and CSV export in `data/battery_sentinel.db`.
- **In-App Periodic Background Logging**: Thread-safe background monitoring service that logs snapshots at regular intervals while the application is running.
- **Degradation Forecasting & RUL Estimation**: Projects future capacity trends and remaining useful life until the configured $80\%$ End-of-Life (EOL) line using observed logs with $95\%$ residual uncertainty envelopes. Halts honestly with clear guidance when history is $<3$ checkpoints.
- **Laboratory ML Research Benchmark**: Evaluates scikit-learn models (Linear Regression baseline vs. Random Forest Regressor) on authentic experimental cycling datasets (NASA Ames PCoE 18650 cells) with cell-level train/test splits preventing data leakage.
- **Intelligent Diagnostics & Alerts**: Comprehensive condition evaluation (SOH thresholds, rapid fade, high power draw, low battery buffer, voltage depression) with actionable engineering advice.

---

## 🏛️ 2. System Architecture & Workstations

The dashboard is structured into 5 dedicated workstations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAPTOP BATTERY HEALTH SENTINEL                        │
├───────────────┬──────────────────┬─────────────────┬───────────────┬────────┤
│  1. OVERVIEW  │ 2. DIAGNOSTICS   │ 3. HISTORY      │ 4. AI LAB     │5.ALERTS│
│  - Charge %   │ - Live Voltage   │ - SOH Curves    │ - Empirical   │-Active │
│  - SOH Gauge  │ - Discharge Rate │ - SQLite Audit  │   Forecast    │ Warnings
│  - Capacities │ - BMS Specs      │ - Date Filter   │ - NASA Ames   │-Evidence
│  - Power State│ - Ingestion Ctrl │ - CSV Export    │   Benchmarking│-Action 
└───────────────┴──────────────────┴─────────────────┴───────────────┴────────┘
```

---

## 🎨 3. Design System & Palette

The user interface follows a curated, high-contrast engineering aesthetic:

| Role | Color Name | Hex Code | Visual Application |
| :--- | :--- | :--- | :--- |
| **Primary** | Deep Navy Blue | `#102A43` | Application identity, navigation, headers, and ground truth |
| **Secondary** | Teal | `#0D9488` | Battery health, normal state metrics, and healthy gauge zones |
| **Background** | Soft Off-White | `#F5F7FA` | High-contrast main canvas and card backgrounds |
| **Typography** | Dark Slate | `#243447` | High-readability body and metric labels |
| **Warning** | Amber | `#D97706` | Maintenance advisories, rapid degradation, caution thresholds |
| **Critical** | Red | `#DC2626` | Lithium-ion End-of-Life ($\le 80\%$ SOH), critically low charge |

---

## 🗂️ 4. Supported Data Sources & Data Integrity

The system maintains strict domain separation:

| Dimension | NASA Research Data (`data/sample_nasa_battery.csv`) | Windows Laptop Reports (`powercfg /batteryreport`) |
| :--- | :--- | :--- |
| **Domain** | Public experimental research on single cylindrical 18650 cells. | Real laptop multi-cell series/parallel battery packs. |
| **Units** | Ampere-hours ($\text{Ah}$) or Milliampere-hours ($\text{mAh}$). | Milliwatt-hours ($\text{mWh}$) or Milliampere-hours ($\text{mAh}$). |
| **Test Conditions** | Controlled $24^\circ\text{C}$ chamber, continuous fixed-current cycles. | Real-world bursty CPU/GPU loads, variable chassis thermals. |
| **Purpose** | Machine learning algorithm evaluation and kinetic benchmarking. | Laptop-specific health monitoring and capacity trend alerts. |

---

## 📐 5. Mathematical Methodology

### 1. Capacity-Based SOH
$$\text{SOH}_{\text{capacity}} (\%) = \left( \frac{\text{Full Charge Capacity}}{\text{Design Capacity}} \right) \times 100$$

### 2. Machine Learning Evaluation
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2}$$
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 📁 6. Directory Structure

```
laptop-battery-health-sentinel/
├── app.py                      # Interactive Streamlit Web Dashboard (5 Workstations)
├── requirements.txt            # Python dependencies (Python 3.10 - 3.14 compatible)
├── pytest.ini                  # Pytest configuration
├── README.md                   # Technical documentation and operational guide
├── .gitignore                  # Keeps local dumps & private database files secure
├── data/
│   ├── README.md               # Dataset source documentation
│   ├── battery_sentinel.db     # SQLite persistence database (created locally)
│   ├── sample_nasa_battery.csv # Authentic NASA PCoE cell cycling dataset (B0005, B0006, B0007)
│   └── sample_laptop_history.csv # Benchmark laptop capacity degradation history
├── models/
│   ├── README.md               # Model persistence documentation
│   ├── linear_regression_baseline.joblib  # Trained baseline regression model
│   ├── random_forest_regressor.joblib     # Trained non-linear ensemble model
│   └── model_metadata.json     # Training timestamps, cell splits, and honest metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading and schema detection
│   ├── preprocessing.py        # Cleaning, canonical normalization, cell-based split
│   ├── soh_calculator.py       # Capacity-based SOH calculation & validations
│   ├── train_model.py          # ML training pipeline and zero-leakage evaluation
│   ├── prediction.py           # Degradation trend forecasting & uncertainty envelopes
│   ├── windows_battery_report.py # Non-elevated CIM/WMI telemetry & powercfg parser
│   ├── history_store.py        # SQLite persistence, migrations, and periodic monitor
│   └── alerts.py               # Alert engine, live telemetry checks, recommendations
└── tests/
    ├── test_soh_calculator.py  # Unit tests for SOH calculations and physical validations
    ├── test_alerts.py          # Unit tests for alert thresholds, voltage sag, power draw
    ├── test_windows_battery_report.py # Unit tests for HTML parser and live telemetry schema
    ├── test_history_store.py   # Unit tests for SQLite storage, deduplication, date filtering
    └── test_data_and_models.py # Unit tests for data loading, preprocessing, ML, and forecasting
```

---

## 🚀 7. Execution Commands (Windows)

On Windows systems where `py` is the launcher for Python:

### 1. Run Automated Unit Tests (40 Tests)
```powershell
py -m pytest -v
# or using the virtual environment directly:
.\.venv\Scripts\pytest.exe -v
```

### 2. Train and Serialize Machine Learning Models
```powershell
py src/train_model.py
# or:
.\.venv\Scripts\python.exe src/train_model.py
```

### 3. Launch Sentinel Dashboard
```powershell
py -m streamlit run app.py
# or:
.\.venv\Scripts\streamlit.exe run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🌐 8. Remote Access & Cloud Deployment Considerations

### Important Hardware Reality:
A cloud-hosted server (such as Streamlit Community Cloud or Heroku) runs in a Linux container inside a cloud datacenter. It **cannot directly read your physical laptop's battery hardware** or execute Windows `powercfg` commands.

### Supported Operational Modes:
1. **Local Workstation (Recommended)**:
   - Run the Sentinel dashboard locally via `py -m streamlit run app.py`.
   - Directly polls your laptop's BMS hardware via Windows WMI/CIM and generates official battery reports.
2. **Local Network Access (LAN)**:
   - Run on your laptop with `--server.address 0.0.0.0`:
     ```powershell
     py -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
     ```
   - Connect from other devices on the same Wi-Fi network using your laptop's local IP address (`http://192.168.x.x:8501`).
3. **Cloud-Hosted Dashboard with File Ingestion**:
   - When deployed to a cloud server, live hardware scanning is disabled.
   - Users can upload their laptop's exported battery report HTML (`powercfg /batteryreport`) or CSV logs into the dashboard via the **Telemetry Controls** tab.

