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
