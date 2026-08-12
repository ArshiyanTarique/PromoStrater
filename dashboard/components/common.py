"""Small presentation-only helpers shared by dashboard pages."""

from __future__ import annotations

from typing import Any
import streamlit as st

from dashboard.components.formatters import format_enum_label


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
    return f"{run['run_id']} · {mode} · {status}"


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
