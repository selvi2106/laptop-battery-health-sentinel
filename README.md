# 🔋 Laptop Battery Health Sentinel
### Intelligent Battery Telemetry, Longitudinal SOH Estimation & Degradation Forecasting

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Visualizations](https://img.shields.io/badge/Visuals-Plotly-239120.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-25%20Passed-brightgreen.svg)]()

---

## 📌 1. Overview

**Laptop Battery Health Sentinel** is a practical, production-ready battery diagnostics and degradation forecasting workstation designed for real laptops and battery systems. It bridges real-world laptop hardware monitoring with machine learning research on lithium-ion degradation kinetics.

### Key Capabilities:
- **Direct Windows Battery Integration**: 1-click execution and parsing of Windows `powercfg /batteryreport` without requiring administrator privileges.
- **Accurate Capacity-Based SOH**: Computes State of Health (SOH) from design capacity vs. current full-charge capacity, tracking energy loss ($\text{mWh}$) and cycle wear indices.
- **Degradation Forecasting & RUL Estimation**: Projects future capacity trends and remaining useful life until the standard $80\%$ End-of-Life (EOL) line using historical time-series logs.
- **Degradation Intelligence Lab**: Trains and evaluates scikit-learn models (Linear Regression baseline vs. Random Forest) on authentic experimental cycling datasets (NASA Ames PCoE 18650 cells) with cell-level train/test splits that prevent data leakage.
- **Interactive Plotly Visualizations**: Real-time health gauges, capacity fade curves, threshold overlays, and exportable CSV audit logs.
- **Hardware Safety Disclaimers**: Clear operational guidance distinguishing software capacity health from physical hardware hazards (e.g. swelling or thermal runaway).

---

## 🏛️ 2. System Architecture

```
 +-------------------------------------------------------------------------+
 |                              DATA INGESTION                             |
 |   [Windows powercfg /batteryreport]       [NASA Ames PCoE Cycling CSVs] |
 +-----------------------+---------------------------------+---------------+
                         |                                 |
                         v                                 v
 +-------------------------------------------------------------------------+
 |                          DATA PREPROCESSING LAYER                       |
 |  - HTML Parser & Regex Sanitizer           - Unit Standardization       |
 |  - Physical Sanity Checks (FCC, DC > 0)    - Zero-Leakage Cell Split    |
 +-----------------------+---------------------------------+---------------+
                         |                                 |
                         v                                 v
 +-----------------------------------+ +-----------------------------------+
 |     MONITORING & ALERT SENTINEL   | |       PREDICTIVE ML WORKBENCH     |
 |  - SOH = (FCC / DC) * 100         | |  - Baseline: Linear Regression    |
 |  - Grade A/B/C/EOL Classification | |  - Non-linear: Random Forest      |
 |  - Empirical RUL Extrapolation    | |  - Out-of-Sample MAE & RMSE       |
 +-----------------------+-----------+ +-----------------+-----------------+
                         |                               |
                         +---------------+---------------+
                                         |
                                         v
 +-------------------------------------------------------------------------+
 |                       INTERACTIVE STREAMLIT UI                          |
 |  - Live Telemetry Cards & SOH Gauge (Plotly)                            |
 |  - Capacity Fade Trajectories & Forecast Overlays                       |
 |  - Alert Sentinel Badges & One-Click CSV Audit Export                   |
 +-------------------------------------------------------------------------+
```

---

## 🗂️ 3. Supported Data Sources & Data Integrity

The system maintains strict domain separation:

| Dimension | NASA Research Data (`data/sample_nasa_battery.csv`) | Windows Laptop Reports (`powercfg /batteryreport`) |
| :--- | :--- | :--- |
| **Domain** | Public experimental research on single cylindrical 18650 cells. | Real laptop multi-cell series/parallel battery packs. |
| **Units** | Ampere-hours ($\text{Ah}$) or Milliampere-hours ($\text{mAh}$). | Milliwatt-hours ($\text{mWh}$) or Milliampere-hours ($\text{mAh}$). |
| **Test Conditions** | Controlled $24^\circ\text{C}$ chamber, continuous fixed-current cycles. | Real-world bursty CPU/GPU loads, variable chassis thermals. |
| **Purpose** | Machine learning algorithm evaluation and kinetic benchmarking. | Laptop-specific health monitoring and capacity trend alerts. |

---

## 📐 4. Mathematical Methodology

### 1. Capacity-Based SOH
$$\text{SOH}_{\text{capacity}} (\%) = \left( \frac{\text{Full Charge Capacity}}{\text{Design Capacity}} \right) \times 100$$

### 2. Machine Learning Evaluation
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2}$$

### 3. End of Life (EOL) Threshold
In lithium-ion battery electrochemistry, standard End-of-Life is reached when full charge capacity decays to **$80\%$ of nominal design capacity** ($SOH \le 80\%$).

---

## 📁 5. Directory Structure

```
laptop-battery-health-sentinel/
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies (Python 3.10 - 3.14 compatible)
├── pytest.ini                  # Pytest configuration
├── README.md                   # Technical documentation and operational guide
├── .gitignore                  # Keeps local dumps & private battery data secure
├── data/
│   ├── README.md               # Dataset source documentation & download instructions
│   ├── sample_nasa_battery.csv # Authentic NASA PCoE cell cycling dataset (B0005, B0006, B0007)
│   └── sample_laptop_history.csv # Periodic laptop capacity degradation history
├── models/
│   ├── README.md               # Model persistence documentation
│   ├── linear_regression_baseline.joblib  # Trained baseline regression model
│   ├── random_forest_regressor.joblib     # Trained non-linear ensemble model
│   └── model_metadata.json     # Training timestamps, cell splits, and honest metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading and schema detection
│   ├── preprocessing.py        # Cleaning, unit conversions, and cell-based split
│   ├── soh_calculator.py       # Capacity-based SOH calculation & validations
│   ├── train_model.py          # ML training pipeline and evaluation
│   ├── prediction.py           # Degradation trend forecasting & RUL estimation
│   ├── windows_battery_report.py # Non-elevated powercfg runner and HTML parser
│   └── alerts.py               # Alert evaluation, configurable thresholds, recommendations
└── tests/
    ├── test_soh_calculator.py  # Unit tests for SOH calculations and edge cases
    ├── test_alerts.py          # Unit tests for alert thresholds & severities
    ├── test_windows_battery_report.py # Unit tests for HTML report parsing
    └── test_data_and_models.py # Unit tests for data loading, preprocessing, ML, and forecasting
```

---

## 🚀 6. Quick Start & Execution

### Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Automated Tests (25 unit tests)
```powershell
python -m pytest -v
```

### Train Machine Learning Models
```powershell
python src/train_model.py
```

### Launch Dashboard
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🌐 7. Cloud Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Select the repository, set `Main file path` to `app.py`, and click **Deploy**.
