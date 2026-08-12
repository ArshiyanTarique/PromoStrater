"""Session-aware access to the one pipeline status snapshot.

:mod:`dashboard.services.pipeline_status` resolves durable state without
touching Streamlit. This is the thin layer above it that every rendering
surface calls, so the two session-scoped concerns - which job this browser
started, and whether its failure notice was dismissed - are applied once
instead of per surface.

:func:`status_facts` exists for the same reason. The sidebar renders the facts
as captions and the upload page as metrics, but both read the identical label
and value strings from here, so no surface can round, word, or truncate a fact
its own way.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.dismiss import is_failure_dismissed
from dashboard.components.formatters import format_elapsed
from dashboard.components.pipeline_flow import Outcome
from dashboard.services.pipeline_status import (
    JobReader,
    PipelinePhase,
    PipelineStatus,
    resolve_pipeline_status,
)
from sku_mapping.learning.store import LearningStore

#: Flow shape per phase. A queued job has not entered a stage, so it renders as
#: the first stage waiting to move rather than as a stopped run.
_FLOW_OUTCOME: dict[PipelinePhase, Outcome] = {
    PipelinePhase.QUEUED: "running",
    PipelinePhase.RUNNING: "running",
    PipelinePhase.CANCELLING: "cancelling",
    PipelinePhase.COMPLETED: "succeeded",
    PipelinePhase.CANCELLED: "stopped",
    PipelinePhase.FAILED: "stopped",
}


def current_status(jobs: JobReader, store: LearningStore) -> PipelineStatus:
    """Resolve the status snapshot for this browser session."""
    status = resolve_pipeline_status(jobs, store, st.session_state)
    if status.phase is PipelinePhase.FAILED and is_failure_dismissed(
        status.job_id, status.run_id
    ):
        return status.dismissed()
    return status


def status_facts(status: PipelineStatus) -> dict[str, str]:
    """The facts every surface reports, in one wording and one order."""
    return {
        "Status": status.badge_label,
        "File": status.file_text,
        "Stage": status.stage_text,
        "Progress": status.percent_text,
        "Elapsed": format_elapsed(status.elapsed_seconds),
    }


def flow_outcome(status: PipelineStatus) -> Outcome:
    """Map a phase onto the pipeline flow's visual outcome."""
    return _FLOW_OUTCOME.get(status.phase, "running")
