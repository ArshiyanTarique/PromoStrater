"""Streamlit smoke coverage for every Phase 7B page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    "path",
    [
        "dashboard/Dashboard.py",
        "dashboard/pages/1_Upload_and_Process.py",
        "dashboard/pages/2_Human_Validation.py",
        "dashboard/pages/3_Results_and_Downloads.py",
        "dashboard/pages/4_Models_and_Learning.py",
        "dashboard/pages/5_Al_Kabeer_SKUs.py",
        "dashboard/pages/6_Competitor_Review.py",
    ],
)
def test_streamlit_page_smoke(path: str) -> None:
    app = AppTest.from_file(Path(path)).run(timeout=60)
    assert not app.exception


def test_al_kabeer_sku_page_renders_empty_states_not_raw_values() -> None:
    """A SKU with no mappings must read as prose, never as an empty list.

    The page is read-only over the latest completed run, so most master SKUs
    will have no offers on any given run; that state is the common case rather
    than an edge case.
    """
    app = AppTest.from_file(
        Path("dashboard/pages/5_Al_Kabeer_SKUs.py")
    ).run(timeout=60)
    assert not app.exception

    # Selecting a row is what the dataframe's on_select="rerun" writes back.
    app.session_state["master_sku_table"] = {
        "selection": {"rows": [1], "columns": []}
    }
    app.run(timeout=60)
    assert not app.exception

    rendered = " ".join(item.value for item in app.info)
    assert "[]" not in rendered
    assert "No offers mapped to this SKU" in rendered


def test_dashboard_source_does_not_render_removed_threshold_message() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("dashboard").rglob("*.py"))
    ).casefold()

    assert "operational threshold" not in source
    assert "independently approved" not in source


def test_models_page_renders_simplified_status_charts_and_collapsed_details() -> None:
    path = Path("dashboard/pages/4_Models_and_Learning.py")
    app = AppTest.from_file(path).run(timeout=30)

    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Evaluation sample",
        "Precision",
        "Manual-review share",
    ]
    assert "Model quality overview" in [
        item.value for item in app.subheader
    ]
    assert "Runtime components" in [
        item.value for item in app.subheader
    ]
    assert "Technical details" in [item.label for item in app.status]
    assert len(app.dataframe) == 1

    source = path.read_text(encoding="utf-8")
    assert source.count("st.bar_chart(") >= 4
    assert '"Technical details"' in source
    assert "expanded=False" in source
    assert "package hashes" not in source.casefold()


def test_failure_panel_renders_status_root_cause_traceback_and_timeline() -> None:
    app = AppTest.from_string(
        """
from dashboard.components.failure_panel import render_failure_panel
from dashboard.services.processing_service import DashboardProcessingError
from dashboard.services.progress import ProcessingState, ProgressUpdate

details = {
    "type": "TypeError",
    "message": "could not convert NoneType to float",
    "source_filename": "measurement_features.py",
    "file": "/app/measurement_features.py",
    "function": "parse_measurements",
    "line": 188,
    "traceback": "root traceback",
    "exception_chain": [],
}
update = ProgressUpdate(
    state=ProcessingState.FAILED,
    stage_key="inference",
    stage_label="Mapping own-brand SKUs",
    overall_percent=55.0,
    last_completed_stage="Preparing offers",
    run_id="dashboard-example",
    pipeline_status="MODEL_ERROR_SAFE_FALLBACK",
    original_exception=details,
    completed_stage_keys=(
        "validation",
        "input_loading",
        "canonicalisation",
    ),
)
error = DashboardProcessingError(
    "Processing failed",
    run_id="dashboard-example",
    failed_stage="Mapping own-brand SKUs",
    last_completed_stage="Preparing offers",
    technical_details="complete chained traceback",
    partial_artifacts=(
        "/run/failure_report.json",
        "/run/internal_failure.log",
        "/run/unified_decisions.csv",
    ),
    pipeline_status="MODEL_ERROR_SAFE_FALLBACK",
    original_exception=details,
)
render_failure_panel(error, update)
"""
    ).run(timeout=20)

    assert not app.exception
    assert "Technical details" in [
        status.label for status in app.status
    ]
    code_values = [item.value for item in app.code]
    assert "MODEL_ERROR_SAFE_FALLBACK" in code_values
    assert "TypeError" in code_values
    assert "could not convert NoneType to float" in code_values
    assert "measurement_features.py" in code_values
    assert "parse_measurements" in code_values
    assert "188" in code_values
    assert "complete chained traceback" in code_values
    markdown = "\n".join(item.value for item in app.markdown)
    assert "File validation" in markdown
    assert "Mapping own-brand SKUs" in markdown
    assert "failed" in markdown
    assert "check_circle: Mapping own-brand SKUs" not in markdown
    assert "Competitor discovery" in markdown
    assert "skipped" in markdown
    assert "Failure report" in markdown
    assert "Partial outputs" in markdown


def test_processing_action_keeps_normal_primary_button_style() -> None:
    source = Path(
        "dashboard/pages/1_Upload_and_Process.py"
    ).read_text(encoding="utf-8")

    assert '"Start processing"' in source
    assert 'type="primary"' in source
    # The action must stay a real st.button wearing the theme's primary style,
    # never markup imitating one. This used to ban unsafe_allow_html outright,
    # which also banned the panel's injected CSS - the ban is on hand-rolled
    # controls, not on stylesheets.
    assert "<button" not in source
    assert "stButton" not in source


@pytest.mark.parametrize(
    "path",
    [
        "dashboard/pages/1_Upload_and_Process.py",
        "dashboard/components/sidebar_status.py",
    ],
)
def test_live_panels_poll_twice_per_second(path: str) -> None:
    """The elapsed reading is whole seconds, so the poll must outpace it.

    Polling on the second itself is not enough: each redraw costs time, the
    next tick lands late, and the timer skips a number. Both live surfaces
    resolve the same snapshot, so they must share the interval or one lags the
    other.
    """
    source = Path(path).read_text(encoding="utf-8")
    assert 'run_every="0.5s"' in source, f"{path} lost its 0.5s poll"
    for slower in ('run_every="1s"', 'run_every="2s"'):
        assert slower not in source, f"{path} regressed to {slower}"
