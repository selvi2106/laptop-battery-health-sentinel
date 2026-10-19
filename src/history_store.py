"""
history_store.py
================
Local SQLite persistence layer for battery monitoring observations.

Key Principles:
---------------
1. All battery observations remain strictly local in the project data directory.
2. Deduplication: Automatically prevents storing duplicate records when capacity
   and cycle counts are unchanged, unless a force-snapshot is explicitly requested.
3. Delta Tracking: Calculates changes in SOH and full charge capacity relative to
   the immediately preceding observation.
4. Privacy: Local database file is excluded from git version control.
5. In-App Periodic Monitoring: Optional configurable thread-based background logger
   that records snapshots at regular intervals while the application is running.
"""

import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "battery_sentinel.db")


def get_db_connection() -> sqlite3.Connection:
    """Establish and return a connection to the local SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db() -> None:
    """Initialize database tables, columns, and indexes safely with migration support."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battery_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_name TEXT,
                manufacturer TEXT,
                chemistry TEXT,
                design_capacity_mwh REAL,
                full_charge_capacity_mwh REAL,
                soh_percent REAL,
                cycle_count INTEGER,
                charge_percent REAL,
                power_status TEXT,
                data_source TEXT NOT NULL,
                delta_soh REAL,
                delta_capacity_mwh REAL,
                notes TEXT,
                voltage_v REAL,
                discharge_rate_w REAL,
                temperature_c REAL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON battery_observations(timestamp)
        """)
        
        # Schema migration: check for missing columns if table existed previously
        cursor.execute("PRAGMA table_info(battery_observations)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        
        new_cols = [
            ("voltage_v", "REAL"),
            ("discharge_rate_w", "REAL"),
            ("temperature_c", "REAL")
        ]
        for col_name, col_type in new_cols:
            if col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE battery_observations ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

        conn.commit()


def record_observation(
    design_capacity_mwh: Optional[float],
    full_charge_capacity_mwh: Optional[float],
    soh_percent: Optional[float],
    data_source: str,
    device_name: Optional[str] = None,
    manufacturer: Optional[str] = None,
    chemistry: Optional[str] = None,
    cycle_count: Optional[int] = None,
    charge_percent: Optional[float] = None,
    power_status: Optional[str] = None,
    notes: Optional[str] = None,
    voltage_v: Optional[float] = None,
    discharge_rate_w: Optional[float] = None,
    temperature_c: Optional[float] = None,
    timestamp: Optional[str] = None,
    force_snapshot: bool = False
) -> Tuple[bool, str, Optional[int]]:
    """
    Record a new battery observation into local SQLite storage.

    Parameters
    ----------
    force_snapshot : bool
        If True, saves snapshot even if values are identical to previous entry.
        If False, deduplicates to avoid database bloat.

    Returns
    -------
    (success, message, record_id_or_None)
    """
    init_history_db()

    if full_charge_capacity_mwh is None or design_capacity_mwh is None or soh_percent is None:
        return False, "Cannot store observation: full charge capacity, design capacity, and SOH are required.", None

    now_iso = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Fetch latest entry to check for duplicates and compute deltas
        cursor.execute("""
            SELECT id, timestamp, design_capacity_mwh, full_charge_capacity_mwh, soh_percent, cycle_count
            FROM battery_observations
            ORDER BY id DESC
            LIMIT 1
        """)
        latest = cursor.fetchone()

        delta_soh = None
        delta_cap = None

        if latest is not None:
            prev_fcc = latest["full_charge_capacity_mwh"]
            prev_soh = latest["soh_percent"]
            prev_cycles = latest["cycle_count"]

            # Deduplication check: if FCC, DC, and cycles are identical
            if not force_snapshot:
                same_fcc = abs(prev_fcc - full_charge_capacity_mwh) < 0.1
                same_cycles = (prev_cycles == cycle_count) or (prev_cycles is None and cycle_count is None)
                if same_fcc and same_cycles:
                    return False, f"Duplicate observation skipped (identical to record #{latest['id']} from {latest['timestamp']}).", latest["id"]

            delta_soh = round(soh_percent - prev_soh, 2)
            delta_cap = round(full_charge_capacity_mwh - prev_fcc, 1)

        cursor.execute("""
            INSERT INTO battery_observations (
                timestamp, device_name, manufacturer, chemistry,
                design_capacity_mwh, full_charge_capacity_mwh, soh_percent,
                cycle_count, charge_percent, power_status, data_source,
                delta_soh, delta_capacity_mwh, notes,
                voltage_v, discharge_rate_w, temperature_c
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso, device_name, manufacturer, chemistry,
            design_capacity_mwh, full_charge_capacity_mwh, soh_percent,
            cycle_count, charge_percent, power_status, data_source,
            delta_soh, delta_cap, notes,
            voltage_v, discharge_rate_w, temperature_c
        ))
        new_id = cursor.lastrowid
        conn.commit()

        return True, f"Observation #{new_id} recorded successfully.", new_id


def get_all_observations(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_source: Optional[str] = None
) -> pd.DataFrame:
    """
    Retrieve stored observations sorted chronologically with optional date and source filtering.
    """
    init_history_db()
    
    query = """
        SELECT 
            id AS "ID",
            timestamp AS "Timestamp",
            device_name AS "Device",
            design_capacity_mwh AS "Design Cap (mWh)",
            full_charge_capacity_mwh AS "Full Charge Cap (mWh)",
            soh_percent AS "SOH (%)",
            cycle_count AS "Cycle Count",
            charge_percent AS "Charge (%)",
            power_status AS "Status",
            voltage_v AS "Voltage (V)",
            discharge_rate_w AS "Power (W)",
            data_source AS "Source",
            delta_soh AS "Delta SOH (%)",
            delta_capacity_mwh AS "Delta Cap (mWh)"
        FROM battery_observations
        WHERE 1=1
    """
    params: List[Any] = []

    if start_date:
        query += " AND timestamp >= ?"
        params.append(f"{start_date} 00:00:00")
    if end_date:
        query += " AND timestamp <= ?"
        params.append(f"{end_date} 23:59:59")
    if data_source and data_source != "ALL":
        query += " AND data_source = ?"
        params.append(data_source)

    query += " ORDER BY id ASC"

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        df["soh_percent"] = df["SOH (%)"]
        df["full_charge_capacity_mwh"] = df["Full Charge Cap (mWh)"]
        df["design_capacity_mwh"] = df["Design Cap (mWh)"]
        df["Period"] = df["Timestamp"]
        df["timestamp"] = df["Timestamp"]
        df["date"] = df["Timestamp"]
        df["cycle_count"] = df["Cycle Count"]
        df["voltage_v"] = df["Voltage (V)"]
        df["discharge_rate_w"] = df["Power (W)"]
    return df


def import_history_dataframe(
    df: pd.DataFrame,
    source_tag: str = "WINDOWS_REPORT_IMPORT",
    device_name: str = "Dell Laptop",
    manufacturer: str = "OEM",
    chemistry: str = "Li-Ion"
) -> Tuple[int, int]:
    """
    Import canonical historical DataFrame into SQLite database with automatic deduplication.
    
    Returns
    -------
    (imported_count, skipped_count)
    """
    if df is None or df.empty:
        return 0, 0

    from src.preprocessing import normalize_history_dataframe
    norm_df = normalize_history_dataframe(df)

    imported = 0
    skipped = 0

    for _, row in norm_df.iterrows():
        fcc = row.get("full_charge_capacity_mwh")
        dc = row.get("design_capacity_mwh")
        soh = row.get("soh_percent")
        ts = str(row.get("timestamp") or row.get("Period") or "")
        cyc = row.get("cycle_count")
        cyc_val = int(cyc) if pd.notna(cyc) and cyc is not None else None

        if pd.notna(fcc) and pd.notna(dc) and pd.notna(soh) and fcc > 0 and dc > 0:
            ok, _, _ = record_observation(
                design_capacity_mwh=float(dc),
                full_charge_capacity_mwh=float(fcc),
                soh_percent=float(soh),
                data_source=source_tag,
                device_name=device_name,
                manufacturer=manufacturer,
                chemistry=chemistry,
                cycle_count=cyc_val,
                timestamp=ts if len(ts) >= 10 else None,
                force_snapshot=False
            )
            if ok:
                imported += 1
            else:
                skipped += 1

    return imported, skipped


def delete_observation(record_id: int) -> bool:
    """Delete a specific record by its primary key ID."""
    init_history_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM battery_observations WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_history() -> int:
    """Purge all stored observations from local database."""
    init_history_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM battery_observations")
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count


def export_history_to_csv(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_source: Optional[str] = None
) -> bytes:
    """Export stored observations as UTF-8 CSV bytes."""
    df = get_all_observations(start_date=start_date, end_date=end_date, data_source=data_source)
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# In-App Periodic Monitoring Service (Active while application is running)
# -----------------------------------------------------------------------------
class PeriodicMonitor:
    """
    Thread-safe background battery monitoring service.
    
    Runs periodic snapshot recording while the application is active.
    """
    _instance: Optional["PeriodicMonitor"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._interval = 300  # seconds (default 5 minutes)
        self._last_run: Optional[str] = None
        self._run_count = 0
        self._last_error: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "PeriodicMonitor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    @property
    def interval_seconds(self) -> int:
        return self._interval

    @property
    def last_run(self) -> Optional[str]:
        return self._last_run

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self, interval_seconds: int = 300, current_capacities: Optional[Tuple[float, float]] = None) -> bool:
        """Start periodic background monitoring."""
        with self._lock:
            if self.is_running:
                self._interval = max(30, interval_seconds)
                return True

            self._interval = max(30, interval_seconds)
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                args=(current_capacities,),
                daemon=True,
                name="SentinelPeriodicMonitor"
            )
            self._thread.start()
            return True

    def stop(self) -> bool:
        """Stop periodic background monitoring."""
        with self._lock:
            self._running = False
            return True

    def _monitor_loop(self, fallback_capacities: Optional[Tuple[float, float]]):
        """Worker loop executed in daemon thread."""
        from src.windows_battery_report import get_live_battery_status

        while self._running:
            try:
                live = get_live_battery_status()
                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if live.get("is_available"):
                    # Use capacities from Windows report if cached, or fallback
                    dc = None
                    fcc = None
                    if fallback_capacities:
                        dc, fcc = fallback_capacities

                    soh = round((fcc / dc) * 100.0, 2) if fcc and dc and dc > 0 else None

                    if fcc and dc and soh:
                        record_observation(
                            design_capacity_mwh=dc,
                            full_charge_capacity_mwh=fcc,
                            soh_percent=soh,
                            data_source="PERIODIC_BACKGROUND_SERVICE",
                            device_name=live.get("device_name"),
                            manufacturer=live.get("manufacturer"),
                            chemistry=live.get("chemistry"),
                            cycle_count=live.get("cycle_count"),
                            charge_percent=live.get("charge_percent"),
                            power_status=live.get("status_text"),
                            voltage_v=live.get("live_voltage_v"),
                            discharge_rate_w=live.get("discharge_rate_w"),
                            force_snapshot=False
                        )

                self._last_run = ts_str
                self._run_count += 1
                self._last_error = None

            except Exception as e:
                self._last_error = str(e)

            # Sleep in 1-second chunks to respond quickly to stop requests
            for _ in range(self._interval):
                if not self._running:
                    break
                time.sleep(1)
