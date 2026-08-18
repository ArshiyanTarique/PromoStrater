"""Small presentation-only helpers shared by dashboard pages."""

from __future__ import annotations

from typing import Any
import streamlit as st

from dashboard.components.formatters import format_enum_label
from sku_mapping.learning.store import (
    DEVELOPER_RUN_MODE,
    PRODUCTION_RUN_MODE,
)

#: Set by the developer-mode toggle on the Upload page. Streamlit session
#: state is app-wide, so every page reads the same answer and a developer
#: sees their own runs rather than an empty list.
DEVELOPER_MODE_SESSION_KEY = "developer_mode"


def active_run_mode() -> str:
    """Return the run mode the operator is currently working in."""
    if st.session_state.get(DEVELOPER_MODE_SESSION_KEY, False):
        return DEVELOPER_RUN_MODE
    return PRODUCTION_RUN_MODE


def render_developer_mode_banner() -> None:
    """Make a developer session unmistakable.

    Developer runs execute the entire pipeline, review staging included, so
    nothing on screen would otherwise distinguish one from the week's real
    output. The banner is the distinction.
    """
    if active_run_mode() == DEVELOPER_RUN_MODE:
        st.warning(
            "**Developer mode.** Runs execute the full pipeline but write to "
            "the developer output tree, stay out of the business views, and "
            "are excluded from training data by default.",
            icon="🛠️",
        )


def safe_page_link(page: str, label: str, icon: str) -> None:
    """Link to a sub-page, degrading to a caption where linking is impossible.

    ``st.page_link`` raises ``KeyError: 'url_pathname'`` when a page runs
    outside a full multipage context, which is how ``AppTest`` executes it. The
    link is presentational, so falling back keeps the page renderable there.
    """
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:
        st.caption(f"{icon} {label}")


def run_label(run: dict[str, Any]) -> str:
    """Return a compact label without exposing filesystem paths."""
    mode = format_enum_label(run.get("deployment_mode"))
    status = format_enum_label(run.get("status"))
    label = f"{run['run_id']} · {mode} · {status}"
    if run.get("run_mode") == DEVELOPER_RUN_MODE:
        label = f"🛠️ {label}"
    return label


def select_run(
    runs: list[dict[str, Any]],
    *,
    key: str,
    preferred_run_id: str | None = None,
) -> dict[str, Any] | None:
    """Render durable run selection with optional session preference."""
    if not runs:
        st.info("No persisted runs are available yet.")
        return None
    identifiers = [str(run["run_id"]) for run in runs]
    index = (
        identifiers.index(preferred_run_id)
        if preferred_run_id in identifiers
        else 0
    )
    selected = st.selectbox(
        "Select a persisted run",
        identifiers,
        index=index,
        format_func=lambda run_id: run_label(
            next(run for run in runs if run["run_id"] == run_id)
        ),
        key=key,
    )
    return next(run for run in runs if run["run_id"] == selected)
