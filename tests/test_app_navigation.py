"""
test_app_navigation.py
======================
Automated integration tests for app landing page, "View Dashboard" transition,
and "Back to Home" navigation using Streamlit AppTest framework.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_landing_page_initial_view():
    """Verify that initial page load shows only the clean landing view and View Dashboard button."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception, f"App threw exception: {at.exception}"

    # Initial page state
    assert at.session_state["current_page"] == "home"

    # Verify only 1 primary button on landing view ("View Dashboard")
    button_labels = [b.label for b in at.button]
    assert any("View Dashboard" in lbl for lbl in button_labels)
    assert not any("Back to Home" in lbl for lbl in button_labels)

    # Verify tabs are NOT rendered on the landing page
    assert len(at.tabs) == 0


def test_view_dashboard_and_back_to_home_navigation():
    """Verify clicking View Dashboard transitions to complete dashboard, and Back to Home returns cleanly."""
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert at.session_state["current_page"] == "home"

    # Click View Dashboard
    view_dashboard_btn = [b for b in at.button if "View Dashboard" in b.label][0]
    view_dashboard_btn.click().run(timeout=30)

    # Verify state and dashboard contents
    assert at.session_state["current_page"] == "dashboard"
    dashboard_button_labels = [b.label for b in at.button]
    assert any("Back to Home" in lbl for lbl in dashboard_button_labels)
    assert len(at.tabs) >= 5

    # Click Back to Home
    back_to_home_btn = [b for b in at.button if "Back to Home" in b.label][0]
    back_to_home_btn.click().run(timeout=30)

    # Verify returned to home view
    assert at.session_state["current_page"] == "home"
    home_button_labels = [b.label for b in at.button]
    assert any("View Dashboard" in lbl for lbl in home_button_labels)
    assert not any("Back to Home" in lbl for lbl in home_button_labels)
    assert len(at.tabs) == 0
