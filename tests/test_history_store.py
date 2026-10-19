"""
test_history_store.py
=====================
Unit tests for the local SQLite history storage and deduplication engine.
"""

import os
import pytest
from src.history_store import (
    init_history_db,
    record_observation,
    get_all_observations,
    delete_observation,
    clear_all_history,
    export_history_to_csv,
    DB_PATH
)


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure a clean database state before each test."""
    clear_all_history()
    yield
    clear_all_history()


def test_init_and_record_observation():
    """Test saving a valid battery observation."""
    success, msg, rec_id = record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=45000.0,
        soh_percent=90.0,
        data_source="TEST_SCAN",
        device_name="TestBattery",
        cycle_count=100
    )
    assert success is True
    assert rec_id is not None
    assert "recorded successfully" in msg.lower()

    df = get_all_observations()
    assert len(df) == 1
    assert df.iloc[0]["SOH (%)"] == 90.0
    assert df.iloc[0]["Device"] == "TestBattery"


def test_deduplication_prevents_duplicate_records():
    """Test that identical sequential readings are deduplicated unless forced."""
    # First record
    ok1, _, id1 = record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=45000.0,
        soh_percent=90.0,
        data_source="TEST_SCAN",
        cycle_count=100
    )
    assert ok1 is True

    # Duplicate record (same FCC and cycles)
    ok2, msg2, id2 = record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=45000.0,
        soh_percent=90.0,
        data_source="TEST_SCAN",
        cycle_count=100,
        force_snapshot=False
    )
    assert ok2 is False
    assert "skipped" in msg2.lower()
    assert id2 == id1

    # Forced snapshot should record
    ok3, _, id3 = record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=45000.0,
        soh_percent=90.0,
        data_source="TEST_SCAN",
        cycle_count=100,
        force_snapshot=True
    )
    assert ok3 is True
    assert id3 != id1

    df = get_all_observations()
    assert len(df) == 2


def test_delta_tracking_between_observations():
    """Test calculation of delta SOH and delta capacity relative to previous entry."""
    record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=48000.0,
        soh_percent=96.0,
        data_source="TEST",
        cycle_count=50
    )

    record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=46000.0,
        soh_percent=92.0,
        data_source="TEST",
        cycle_count=70
    )

    df = get_all_observations()
    assert len(df) == 2
    # Second record delta
    second = df.iloc[1]
    assert second["Delta SOH (%)"] == -4.0
    assert second["Delta Cap (mWh)"] == -2000.0


def test_delete_and_export():
    """Test record deletion and CSV export."""
    _, _, rec_id = record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=45000.0,
        soh_percent=90.0,
        data_source="TEST"
    )

    csv_data = export_history_to_csv()
    assert len(csv_data) > 0
    assert b"Design Cap (mWh)" in csv_data

    del_ok = delete_observation(rec_id)
    assert del_ok is True
    assert len(get_all_observations()) == 0


def test_date_and_source_filtering():
    """Test date and source filters in get_all_observations."""
    record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=48000.0,
        soh_percent=96.0,
        data_source="SRC_A",
        timestamp="2024-01-01 10:00:00",
        force_snapshot=True
    )
    record_observation(
        design_capacity_mwh=50000.0,
        full_charge_capacity_mwh=46000.0,
        soh_percent=92.0,
        data_source="SRC_B",
        timestamp="2024-06-01 10:00:00",
        force_snapshot=True
    )

    # Filter by date range
    df_filtered = get_all_observations(start_date="2024-05-01", end_date="2024-12-31")
    assert len(df_filtered) == 1
    assert df_filtered.iloc[0]["Source"] == "SRC_B"

    # Filter by source
    df_src = get_all_observations(data_source="SRC_A")
    assert len(df_src) == 1
    assert df_src.iloc[0]["Source"] == "SRC_A"


def test_import_history_dataframe():
    """Test importing canonical historical DataFrame into SQLite."""
    import pandas as pd
    from src.history_store import import_history_dataframe

    df = pd.DataFrame({
        "Period": ["2024-01-01", "2024-01-08", "2024-01-15"],
        "Full Charge Capacity (mWh)": [42000, 41500, 41000],
        "Design Capacity (mWh)": [42000, 42000, 42000],
        "SOH (%)": [100.0, 98.8, 97.6]
    })

    imported, skipped = import_history_dataframe(df, source_tag="TEST_IMPORT")
    assert imported == 3
    assert skipped == 0

    obs = get_all_observations()
    assert len(obs) == 3


def test_periodic_monitor_lifecycle():
    """Test PeriodicMonitor singleton controls and safety."""
    from src.history_store import PeriodicMonitor

    monitor = PeriodicMonitor.get_instance()
    assert monitor is not None
    assert not monitor.is_running

    ok = monitor.start(interval_seconds=60)
    assert ok is True
    assert monitor.is_running
    assert monitor.interval_seconds == 60

    stop_ok = monitor.stop()
    assert stop_ok is True
    assert not monitor.is_running

