"""Inspect persisted run summaries and download validated artifacts."""

from __future__ import annotations

import streamlit as st

from dashboard.bootstrap import load_dashboard_context
from dashboard.components.common import select_run
from dashboard.components.formatters import format_enum_label, get_status_color, render_badge
from dashboard.components.sidebar_status import render_sidebar_status
from dashboard.services.job_state import (
    CANCELLED_RUN_STATUS,
    ORPHANED_RUN_STATUS,
    JobState,
)
from dashboard.theme import inject_theme

st.set_page_config(page_title="Results & Downloads", page_icon="📥", layout="wide")

inject_theme()
_, store, _, _, _, run_service, jobs = load_dashboard_context()

render_sidebar_status(store, jobs)

st.title("Results & Downloads")
st.caption("Inspect pipeline metrics, decision distributions, and download validated output datasets.")

selected = select_run(
    run_service.runs(),
    key="results_run",
    preferred_run_id=st.session_state.get("active_run_id"),
)
if selected is None:
    st.info("No persisted runs are available yet. Run a pipeline from Upload & Process to see results.")
    st.stop()

summary = run_service.run_summary(str(selected["run_id"]))
run_status = str(summary.get("status") or "")
job = jobs.job_for_run(str(selected["run_id"]))
was_cancelled = run_status == CANCELLED_RUN_STATUS or (
    job is not None and job.state is JobState.CANCELLED
)
if was_cancelled:
    st.warning(
        "This run was CANCELLED and did not complete. Any metrics and "
        "downloads below reflect only the work finished before the run was "
        "stopped - do not read them as a completed result.",
        icon="⛔",
    )
    if job is not None and job.stage_label:
        st.caption(
            f"Cancelled during: {job.stage_label} - "
            f"after {job.elapsed_seconds:,.0f} seconds."
        )
elif run_status == ORPHANED_RUN_STATUS:
    st.error(
        "This run was interrupted: its processing worker stopped before it "
        "finished. Its outputs are incomplete.",
        icon="⚠️",
    )

if not was_cancelled and not summary.get("dashboard_outputs_complete"):
    st.warning(
        "Inference records exist for this run, but its final dashboard export "
        "stage did not complete. Metrics below use the durable unified "
        "inference statistics; mapping downloads may be unavailable."
    )


def metric_value(key: str) -> int | str:
    value = summary.get(key)
    return int(value) if value is not None else "Unavailable"


# Status & Configuration Header Card
with st.container(border=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Run ID:** `{summary.get('run_id')}`")
        status_val = str(summary.get('status', 'Unknown'))
        st.markdown("Status: ", unsafe_allow_html=True)
        render_badge(format_enum_label(status_val), get_status_color(status_val))
        st.caption(f"Deployment Mode: **{format_enum_label(str(summary.get('deployment_mode')))}**")
        
    with col_b:
        st.caption(f"LightGBM Model: **{summary.get('model_id') or 'N/A'}**")
        st.caption(f"Embedding Model: **{summary.get('embedding_model_id') or 'Disabled/Unavailable'}**")
        st.caption(f"LLM Model: **{summary.get('llm_model_id') or 'Disabled/Unavailable'}**")
        runtime_sec = summary.get("total_runtime_seconds")
        st.caption(f"Runtime: **{float(runtime_sec):.2f}s**" if runtime_sec is not None else "Runtime: N/A")

# Metrics Summary Cards
with st.container(border=True):
    st.subheader("Run Metrics Summary")
    
    first = st.columns(4)
    first[0].metric("Input Rows", f"{int(summary.get('input_rows') or 0):,}")
    first[1].metric("Unique Offers", f"{int(summary.get('unique_offers') or 0):,}")
    first[2].metric("Auto Accept", metric_value("auto_accept_count"))
    first[3].metric("LLM Approved", metric_value("llm_accept_count"))

    second = st.columns(4)
    second[0].metric("Manual Review", metric_value("manual_review_count"))
    second[1].metric("No Candidate", metric_value("no_candidate_count"))
    second[2].metric("Model Errors", metric_value("model_error_count"))
    second[3].metric(
        "Total Runtime",
        (
            f"{float(summary['total_runtime_seconds']):.2f}s"
            if summary.get("total_runtime_seconds") is not None
            else "N/A"
        ),
    )

if summary.get("inference_offer_count") is not None:
    st.caption(
        f"Canonical offers with one terminal state: "
        f"**{metric_value('terminal_decision_count')}** / "
        f"**{int(summary.get('unique_offers') or 0):,}**. "
        f"SKU inference offers: **{int(summary['inference_offer_count']):,}**; "
        f"competitor offers tracked separately: "
        f"**{metric_value('competitor_offer_count')}**."
    )

if summary.get("decision_coverage_complete") is False:
    st.error(
        "This run does not have one terminal decision for every canonical "
        "offer. Treat its outputs as incomplete."
    )

if summary.get("errors"):
    st.error("This run recorded a processing error during execution.")

# Validated Downloads Section
st.subheader("Validated Output Downloads")
artifacts = run_service.downloads(str(selected["run_id"]))

if not artifacts:
    st.info("No validated downloadable outputs exist for this run.")
else:
    for artifact in artifacts:
        label_text = format_enum_label(artifact.key.replace("_", " "))
        st.download_button(
            label=f"Download {label_text} ({artifact.filename})",
            data=artifact.content,
            file_name=artifact.filename,
            mime=artifact.media_type,
            key=f"download_{selected['run_id']}_{artifact.key}",
            width='stretch',
            icon="📥",
        )

st.caption(
    "Only existing artifacts with validated schemas are available for download. "
    "Server filesystem paths remain hidden."
)
