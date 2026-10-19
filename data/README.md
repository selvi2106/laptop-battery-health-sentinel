# Battery Datasets & Data Integrity Guide

This repository distinguishes strictly between **Laboratory Experimental Battery Datasets** and **Real-World Laptop Battery Reports**.

---

## 1. Laboratory Experimental Datasets (NASA Ames PCoE)

### Source Overview
- **Repository**: NASA Ames Prognostics Center of Excellence (PCoE) Battery Aging Data Set.
- **Official Download URL**: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (Datasets #5, #6, #7, #18).
- **Physical Cell Type**: Commercial 18650 Lithium-ion cylindrical cells (LiCoO2 cathode / graphite anode).
- **Test Protocol**: Repeated charge/discharge cycles under controlled ambient temperature ($24^\circ\text{C}$). Charged at constant current (1.5 A) until 4.2 V, then constant voltage until current dropped to 20 mA. Discharged at 2.0 A down to cut-off voltage ($2.7\text{ V}$ for B0005, $2.5\text{ V}$ for B0006, $2.2\text{ V}$ for B0007).
- **Nominal Rated Capacity**: $2.0\text{ Ah}$ ($2000\text{ mAh}$).

### Provided Sample File: `data/sample_nasa_battery.csv`
A curated sample representing the degradation cycles of cells **B0005**, **B0006**, and **B0007** is provided for evaluating machine learning algorithms and testing the interface.
- Columns:
  - `battery_id`: Identifier of the experimental cell (`B0005`, `B0006`, `B0007`).
  - `cycle`: Charge-discharge cycle index (integer $\ge 1$).
  - `ambient_temperature`: Environmental chamber temperature in $^\circ\text{C}$ ($24^\circ\text{C}$).
  - `capacity_ah`: Measured discharge capacity in Ampere-hours ($\text{Ah}$).
  - `nominal_capacity_ah`: Factory nominal rating ($2.0\text{ Ah}$).
  - `soh_percent`: Ground-truth capacity health percentage $= (\text{capacity\_ah} / 2.0) \times 100$.

---

## 2. Windows Laptop Battery Reports (`powercfg`)

### Source Overview
- Generated natively on Windows laptops using the Command Prompt or PowerShell:
  ```powershell
  powercfg /batteryreport /output "battery-report.html"
  ```
- **Privilege Requirement**: Standard user privileges (does NOT require Administrator elevation).
- **Units**: Windows reports capacity in milliwatt-hours ($\text{mWh}$) or milliampere-hours ($\text{mAh}$).
- **Key Parameters Extracted**:
  - `DESIGN CAPACITY`: Manufacturer's original battery pack capacity.
  - `FULL CHARGE CAPACITY`: Current full-charge capacity estimate maintained by the laptop's embedded Fuel Gauge / Gas Gauge IC (e.g., Texas Instruments bq40z50 or similar).
  - `CYCLE COUNT`: Total cumulative discharge cycles recorded by the Battery Management System (BMS).
  - `Battery capacity history`: Longitudinal periodic logs recorded automatically by Windows.

### Provided Sample File: `data/sample_laptop_history.csv`
A sample historical log reflecting a realistic Windows laptop pack degrading over 16 recording intervals.
- Columns:
  - `date`: Log timestamp (ISO format `YYYY-MM-DD`).
  - `full_charge_capacity_mwh`: Estimated full charge capacity in $\text{mWh}$.
  - `design_capacity_mwh`: Factory design capacity in $\text{mWh}$ (e.g., $52,500\text{ mWh}$).
  - `cycle_count`: Cumulative cycles recorded by Windows.

---

## 3. Engineering Disclaimers & Assumptions

1. **Cell vs. Pack Discrepancy**:
   NASA datasets record individual single 18650 cells cycling continuously at fixed currents and temperatures in an environmental chamber.
   Laptop batteries are multi-cell packs (e.g., 3-cell or 4-cell series-parallel configurations) subjected to highly variable, bursty CPU/GPU loads and dynamic thermal throttling.
   **Never claim a model trained on NASA single cells is validated for a specific laptop without transfer calibration.**

2. **Capacity Health vs. Safety**:
   State of Health (SOH) calculated from capacity measures gradual chemical capacity fading and lost runtime.
   It does **NOT** detect physical cell swelling, thermal runaway hazards, dendrite formation, or short circuits.
