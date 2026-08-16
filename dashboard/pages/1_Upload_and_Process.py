"""Upload and process a ClickFlyer dump with active run reconnection."""

from __future__ import annotations

import logging
import os

import streamlit as st

from dashboard.bootstrap import load_dashboard_context
from dashboard.components.common import safe_page_link
from dashboard.components.dismiss import dismiss_button
from dashboard.components.pipeline_flow import render_pipeline_flow
from dashboard.components.run_status import (
    current_status,
    flow_outcome,
    status_facts,
)
from dashboard.components.sidebar_status import render_sidebar_status
from dashboard.services.job_manager import ActiveJobExistsError
from dashboard.services.pipeline_status import (
    WATCHED_JOB_SESSION_KEY,
    PipelinePhase,
    PipelineStatus,
)
from dashboard.services.processing_service import DashboardProcessRequest
from dashboard.services.upload_service import UploadValidationError
from dashboard.theme import inject_theme
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.matching.routing import RoutingMode

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Upload & Process",
    page_icon="⬆️",
    layout="wide",
)

inject_theme()
config, store, processing, registry, _, run_service, jobs = (
    load_dashboard_context()
)
render_sidebar_status(store, jobs)

# SQLite owns job state. This page is a passive reader: it never holds the
# job, its progress, or its cancellation flag in session state, so a browser
# refresh or a second tab observes exactly the same run. The snapshot is the
# same one the sidebar renders, so the two can never describe the run
# differently.
status = current_status(jobs, store)


def artifact_names(paths: tuple[str, ...]) -> str:
    """Name preserved files without revealing server paths."""
    return "\n".join(sorted({os.path.basename(p) for p in paths}))

st.title("Upload & Process")
st.caption(
    "Upload a ClickFlyer dump, map own-brand offers, and discover matching "
    "competitor offers. Processing runs autonomously and persists results in SQLite."
)

# Reconnection comes from the durable job record, so the reported stage and
# percentage are the worker's real position rather than a placeholder.
is_running = status.is_active

if is_running:
    facts = status_facts(status)
    st.info(
        f"**Active run in progress:** `{status.run_id or status.job_id}` · "
        f"Source: {facts['File']} · State: {facts['Status']}",
        icon="🔄",
    )

try:
    models = registry.list_models()
except ValueError:
    LOGGER.exception("Dashboard model registry is unavailable")
    st.error(
        "The model registry is unavailable. An administrator must repair it "
        "before processing can start.",
        icon="⚠️",
    )
    st.stop()

if not models:
    st.error("No compatible registered model is available.")
    st.stop()

with st.container(border=True):
    st.subheader("1. Select Dataset & Configuration")
    
    upload = st.file_uploader(
        "ClickFlyer dump",
        type=[
            suffix.removeprefix(".")
            for suffix in config.dashboard.allowed_extensions
        ],
        help=f"CSV or Excel, maximum {config.dashboard.max_upload_size_mb} MB.",
        disabled=is_running,
        key="clickflyer_upload",
    )

    col1, col2 = st.columns(2)
    with col1:
        mode_value = st.segmented_control(
            "How should the model run?",
            options=["shadow", "assisted"],
            default="assisted",
            help=(
                "Shadow evaluates suggestions without controlling results. Assisted "
                "proposes mappings and routes uncertain cases to human review."
            ),
            disabled=is_running,
            key="deployment_mode",
        )
        
    with col2:
        model_ids = [model.model_id for model in models]
        labels = {model.model_id: model.display_label for model in models}
        # Preselect the configured model rather than whichever id sorts
        # first, so a run defaults to the champion the project is set up on.
        default_index = (
            model_ids.index(str(config.ml.model_id))
            if str(config.ml.model_id) in model_ids
            else 0
        )
        model_id = st.selectbox(
            "Registered Model",
            model_ids,
            index=default_index,
            format_func=lambda value: labels.get(value, value),
            disabled=is_running,
            key="registered_model",
        )
        selected_model = next(model for model in models if model.model_id == model_id)
        status_label = "Assisted mode" if mode_value == "assisted" else "Shadow evaluation only"
        st.caption(f"Version: **{selected_model.model_version}** · Mode: {status_label}")

    # ONE switch for both populations. Own-brand and competitor offers run the
    # same engine, so a per-population toggle would only let the two drift.
    st.markdown("**Matching Mode**")
    mode_columns = st.columns([1, 2])
    with mode_columns[0]:
        enable_llm = st.toggle(
            "Gemini Review",
            value=config.llm_review.enabled,
            disabled=is_running,
            key="enable_llm",
            help=(
                "One switch for both Al Kabeer and competitor offers. It "
                "changes the auto-accept cut-off and where below-cut-off "
                "offers go. Candidate generation and model scoring are "
                "identical either way."
            ),
        )
    active_mode = RoutingMode.from_toggle(
        enable_llm,
        llm_on_threshold=config.llm_review.on_auto_accept_threshold,
        llm_off_threshold=config.llm_review.off_auto_accept_threshold,
    )
    with mode_columns[1]:
        # Deliberately "model score", never "confidence" or "% accuracy": the
        # thresholds are operational cut-offs and nothing here has measured
        # accuracy at either point.
        if active_mode.llm_review_enabled:
            st.success(
                f"**ON** · Auto-accept model score "
                f"**{active_mode.auto_accept_threshold:.2f}** · "
                "Below threshold: **Gemini Review** (no human needed)",
                icon="🤖",
            )
        else:
            st.info(
                f"**OFF** · Auto-accept model score "
                f"**{active_mode.auto_accept_threshold:.2f}** · "
                "Below threshold: **Human Validation** · No Gemini calls",
                icon="👤",
            )

    allow_duplicate = st.checkbox(
        "Explicitly confirm reprocessing identical file bytes",
        value=False,
        disabled=is_running,
        key="allow_duplicate",
    )

if upload is not None:
    content = upload.getvalue()
    try:
        identity = processing.uploads.validate(upload.name, content)
        with st.container(border=True):
            st.markdown("**File Validation Summary**")
            c1, c2, c3 = st.columns(3)
            c1.caption(f"**Filename:** {identity.sanitized_filename}")
            c2.caption(f"**Size:** {identity.size_bytes:,} bytes")
            c3.caption(f"**SHA-256:** `{identity.source_file_hash[:16]}...`")
            
        duplicate_runs = store.active_or_completed_runs_for_source_hash(
            identity.source_file_hash
        )
        if duplicate_runs and not allow_duplicate:
            st.warning(
                "These exact file bytes already have an active or completed "
                "run. Use that run or explicitly check the confirmation box above."
            )
            st.code("\n".join(str(run["run_id"]) for run in duplicate_runs))
    except UploadValidationError as error:
        st.error(str(error))


start_clicked = st.button(
    "Start processing",
    type="primary",
    icon="▶️",
    disabled=upload is None or is_running or mode_value is None,
    width="stretch",
)

if start_clicked and upload is not None:
    try:
        identity = processing.uploads.validate(upload.name, upload.getvalue())
        submitted = jobs.submit(
            DashboardProcessRequest(
                filename=upload.name,
                content=upload.getvalue(),
                deployment_mode=MLDeploymentMode(str(mode_value)),
                model_id=model_id,
                allow_duplicate=allow_duplicate,
                enable_llm_review=enable_llm,
            ),
            source_file_hash=identity.source_file_hash,
        )
        st.session_state[WATCHED_JOB_SESSION_KEY] = submitted.job_id
        st.rerun()
    except ActiveJobExistsError as error:
        st.warning(
            "Another processing job is already active. Cancel it before "
            f"starting a new one (job {error.snapshot.job_id})."
        )
    except UploadValidationError as error:
        st.error(str(error))

def render_run_facts(status: PipelineStatus) -> None:
    """Draw the shared facts, in the same wording the sidebar uses."""
    facts = status_facts(status)
    st.caption(f"**File:** {facts['File']}")
    render_pipeline_flow(
        stage_key=status.stage_key,
        overall_percent=status.percent,
        detail=status.stage_detail,
        outcome=flow_outcome(status),
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Status", facts["Status"])
    s2.metric("Stage", facts["Stage"])
    s3.metric("Progress", facts["Progress"])
    # Elapsed is whatever the server measured for this run at this poll:
    # now - created_at while it is live, created_at -> finished_at once it is
    # not. It advances because the panel redraws, not because anything on the
    # client is counting, so it cannot run at its own speed or get ahead.
    s4.metric("Elapsed", facts["Elapsed"])


@st.fragment(run_every="0.5s")
def live_run_panel() -> None:
    """Poll the durable job record on its own refresh cycle.

    Only this fragment reruns, so the page script never blocks and the panel
    is driven entirely by SQLite - a refresh or a second tab shows the same
    live state. It resolves the snapshot the sidebar's own fragment resolves,
    so the two never drift apart between polls.

    The interval is half the elapsed reading's whole-second resolution, so a
    redraw's own cost cannot push the next tick past a second boundary and make
    the timer skip a number.
    """
    status = current_status(jobs, store)
    if not status.is_active:
        # Terminal state reached: rerun the whole page so the inputs
        # re-enable and the outcome panel replaces this one.
        st.rerun(scope="app")
        return

    with st.container(border=True, key="ps-live-run"):
        st.subheader("2. Current Run")
        if status.phase is PipelinePhase.QUEUED:
            st.info(status.headline, icon="⏳")
        elif status.is_cancelling:
            st.warning(
                f"{status.headline} The run stops at the next stage "
                "checkpoint so partial outputs and logs are preserved.",
                icon="⏳",
            )
        else:
            st.info(status.headline, icon="🔄")

        render_run_facts(status)
        if status.stage_detail:
            st.caption(status.stage_detail)

        if status.is_cancelling:
            st.button(
                "Cancelling...",
                icon="⏳",
                disabled=True,
                width="stretch",
                key="cancel_pending",
            )
        elif st.button(
            "Cancel run",
            type="secondary",
            icon="⛔",
            width="stretch",
            key="cancel_run",
            help=(
                "Requests a graceful stop. The worker is never "
                "force-killed; it finishes the current stage, saves "
                "logs, and releases resources."
            ),
        ):
            if status.job_id:
                jobs.cancel(status.job_id)
            st.rerun(scope="app")


if status.is_active:
    live_run_panel()

elif status.has_run_detail:
    with st.container(border=True):
        st.subheader("2. Last Run")
        # The outcome panel reports the same facts as the live one, so the
        # numbers a run finished on are the numbers it was last seen with.
        if status.phase is PipelinePhase.COMPLETED and status.run_id:
            st.session_state["active_run_id"] = status.run_id
            st.success(
                f"{status.headline} Saved run: {status.run_id}",
                icon="✅",
            )
            render_run_facts(status)
            summary = run_service.run_summary(status.run_id)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Input rows", f"{int(summary.get('input_rows') or 0):,}"
            )
            m2.metric(
                "Own-brand offers",
                f"{int(summary.get('own_brand_offer_count') or 0):,}",
            )
            m3.metric(
                "SKU mappings",
                f"{int(summary.get('sku_mapping_row_count') or 0):,}",
            )
            m4.metric(
                "Manual review",
                f"{int(summary.get('sku_mapping_manual_review_count') or 0):,}",
            )
            safe_page_link(
                "pages/2_Human_Validation.py",
                "Continue to Human Validation →",
                "✅",
            )
        elif status.phase is PipelinePhase.CANCELLED:
            st.warning(
                f"{status.headline} Any outputs below are partial.",
                icon="⛔",
            )
            render_run_facts(status)
            st.caption(status.error_summary or "Cancelled by user request.")
            if status.partial_artifacts:
                st.caption(
                    f"{len(status.partial_artifacts)} partial output(s) and "
                    "log(s) were preserved:"
                )
                st.code(artifact_names(status.partial_artifacts))
            st.info("You can start a new run now.", icon="▶️")
        elif status.phase is PipelinePhase.FAILED:
            st.error("The last processing job failed.", icon="⚠️")
            render_run_facts(status)
            st.caption(status.error_summary or "No error summary was recorded.")
            if status.partial_artifacts:
                st.caption("Preserved diagnostic files:")
                st.code(artifact_names(status.partial_artifacts))
            # Dismissal is resolved in current_status, so clearing the notice
            # here clears it in the sidebar in the same rerun.
            if dismiss_button(
                status.job_id, status.run_id, key="dismiss-failure-upload"
            ):
                st.rerun(scope="app")
        else:
            st.info("You can start a new run now.", icon="▶️")

else:
    with st.container(border=True):
        st.subheader("2. Last Run")
        st.info(f"{status.headline} You can start a new run now.", icon="▶️")
