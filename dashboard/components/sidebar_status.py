"""Global sidebar status card for PromoStrater.

The card is a view of :func:`dashboard.components.run_status.current_status`
and nothing else. It used to resolve state from ``pipeline_runs`` rows on its
own, which is why it could report a run as idle while the upload page reported
the same run cancelled, and why it kept a dead worker's job on ACTIVE - only
the page's path swept orphans. Both surfaces now read one snapshot, so every
fact they share is the same string.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.common import safe_page_link
from dashboard.components.dismiss import dismiss_button
from dashboard.components.elapsed_clock import (
    CLOCK_CSS,
    caption_clock_html,
    clock_anchor_css,
)
from dashboard.components.formatters import get_status_color, render_badge_html
from dashboard.components.run_status import current_status, status_facts
from dashboard.services.pipeline_status import (
    JobReader,
    PipelinePhase,
    PipelineStatus,
)
from sku_mapping.learning.store import LearningStore

#: Shared with the pages; kept as a module-level alias so existing call sites
#: and any monkeypatching in tests continue to resolve.
_safe_page_link = safe_page_link


@st.fragment(run_every="1s")
def _polling_status_body(store: LearningStore, jobs: JobReader) -> None:
    """Redraw the card on its own 1s cycle while a run is active.

    The interval is the elapsed clock's resolution: ``format_elapsed`` renders
    whole seconds, so anything slower makes the timer visibly skip.
    """
    status = _render_status_body(store, jobs)
    if not status.is_active:
        # The run reached a terminal state. Rerunning the whole app swaps this
        # fragment for the static path and lets every other panel settle on the
        # outcome in the same pass, so no surface lags a cycle behind.
        st.rerun(scope="app")


def render_sidebar_status(store: LearningStore, jobs: JobReader) -> None:
    """Render the global sidebar status card.

    While a run is active the card polls in its own fragment. Without that it
    only redrew on a full page run, so it froze at whatever the job looked
    like when the page first loaded while the upload page's own fragment kept
    moving - the two then disagreed on stage and percentage.

    The sidebar container is entered *here*, before the fragment is called. A
    fragment's output binds to the container it was called in, so a
    ``with st.sidebar:`` inside the fragment is discarded and the card
    silently renders nothing.
    """
    with st.sidebar:
        if current_status(jobs, store).is_active:
            _polling_status_body(store, jobs)
        else:
            _render_status_body(store, jobs)


def _render_status_body(
    store: LearningStore, jobs: JobReader
) -> PipelineStatus:
    """Render the card into whatever container the caller has opened.

    Returns the snapshot it drew so the polling wrapper can act on the same
    state it just showed rather than resolving it a second time.
    """
    status = current_status(jobs, store)
    facts = status_facts(status)

    st.markdown('<div class="ps-sidebar-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-sidebar-card-title">'
        '<span>Pipeline Status</span>'
        '<span>'
        + render_badge_html(facts["Status"], get_status_color(facts["Status"]))
        + '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if status.has_run_detail:
        st.caption(f"**File:** {facts['File']}")
        st.caption(f"**Stage:** {facts['Stage']}")
        st.progress(status.progress_fraction)
        # The clock's stylesheet rides along with the line that uses it, the
        # way pipeline_flow ships its own CSS: a style-only markdown call would
        # otherwise open an empty block and push the card apart.
        st.markdown(
            CLOCK_CSS
            + clock_anchor_css(status.elapsed_seconds, live=status.is_active)
            + caption_clock_html(
                f"Progress: {facts['Progress']} · Elapsed: ",
                facts["Elapsed"],
                live=status.is_active,
            ),
            unsafe_allow_html=True,
        )
        if status.run_id:
            st.caption(f"**Run ID:** `{status.run_id[:18]}...`")
    else:
        st.caption(status.headline)

    if status.is_active:
        _safe_page_link(
            "pages/1_Upload_and_Process.py", "View live progress →", "🔄"
        )
    elif status.phase is PipelinePhase.COMPLETED:
        st.caption(f"**Rows:** {status.source_row_count:,}")
        _safe_page_link(
            "pages/3_Results_and_Downloads.py",
            "View results & downloads →",
            "📊",
        )
    elif status.phase is PipelinePhase.CANCELLED:
        _safe_page_link(
            "pages/1_Upload_and_Process.py", "Start new run →", "▶️"
        )
    elif status.phase is PipelinePhase.FAILED:
        error = status.error_summary or "Internal error"
        st.caption(
            f"**Error:** {error[:40]}..." if len(error) > 40
            else f"**Error:** {error}"
        )
        _safe_page_link(
            "pages/1_Upload_and_Process.py", "View error details →", "⚠️"
        )
        # Both identifiers are recorded so one click clears the notice here and
        # on the upload page, which knows the same failure by its job id.
        if dismiss_button(
            status.job_id, status.run_id, key="dismiss-failure-sidebar"
        ):
            st.rerun(scope="app")
    else:
        _safe_page_link(
            "pages/1_Upload_and_Process.py", "Start new run →", "⬆️"
        )

    st.markdown('</div>', unsafe_allow_html=True)
    return status
