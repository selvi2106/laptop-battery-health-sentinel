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
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "battery_sentinel.db")


def get_db_connection() -> sqlite3.Connection:
    """Establish and return a connection to the local SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db() -> None:
    """Initialize database tables and indexes if they do not exist."""
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
                notes TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON battery_observations(timestamp)
        """)
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

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                delta_soh, delta_capacity_mwh, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso, device_name, manufacturer, chemistry,
            design_capacity_mwh, full_charge_capacity_mwh, soh_percent,
            cycle_count, charge_percent, power_status, data_source,
            delta_soh, delta_cap, notes
        ))
        new_id = cursor.lastrowid
        conn.commit()

        return True, f"Observation #{new_id} recorded successfully.", new_id


def get_all_observations() -> pd.DataFrame:
    """Retrieve all stored observations sorted chronologically."""
    init_history_db()
    with get_db_connection() as conn:
        df = pd.read_sql_query("""
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
                data_source AS "Source",
                delta_soh AS "Delta SOH (%)",
                delta_capacity_mwh AS "Delta Cap (mWh)"
            FROM battery_observations
            ORDER BY id ASC
        """, conn)
    if not df.empty:
        df["soh_percent"] = df["SOH (%)"]
        df["full_charge_capacity_mwh"] = df["Full Charge Cap (mWh)"]
        df["design_capacity_mwh"] = df["Design Cap (mWh)"]
        df["Period"] = df["Timestamp"]
        df["timestamp"] = df["Timestamp"]
        df["cycle_count"] = df["Cycle Count"]
    return df


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


def export_history_to_csv() -> bytes:
    """Export all stored observations as UTF-8 CSV bytes."""
    df = get_all_observations()
    return df.to_csv(index=False).encode("utf-8")
