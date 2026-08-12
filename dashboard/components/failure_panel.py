"""Native Streamlit presentation for a preserved processing failure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.services.progress import PROCESSING_STAGES, ProgressUpdate


_TIMELINE_LABELS = {
    "validation": "File validation",
    "input_loading": "Input loading",
    "canonicalisation": "Canonicalisation",
    "inference": "Mapping own-brand SKUs",
    "sku_mapping": "Building SKU mapping output",
    "competitor_discovery": "Competitor discovery",
    "exports": "Exports",
    "persistence": "Persistence",
}


def _artifact_summary(paths: tuple[str, ...]) -> list[str]:
    lowered = [path.casefold() for path in paths]
    names: list[str] = []
    if any("failure_report" in path for path in lowered):
        names.append("Failure report")
    if any("internal_failure" in path for path in lowered):
        names.append("Traceback")
    if any(path.endswith(".log") for path in lowered):
        names.append("Logs")
    if any(
        "failure_report" not in path
        and "internal_failure" not in path
        for path in lowered
    ):
        names.append("Partial outputs")
    return names


def render_failure_panel(
    error: BaseException,
    update: ProgressUpdate | None,
) -> None:
    """Render workflow status first, then structured and raw diagnostics."""
    st.error("Processing failed", icon=":material/error:")
    
    failed_stage = getattr(error, "failed_stage", None) or getattr(
        update, "stage_label", None
    )
    if failed_stage:
        st.write("**Failed during**")
        st.write(failed_stage)
        
    st.write("**Reason**")
    st.write(
        "The SKU mapping engine encountered an internal processing error. "
        "The run stopped safely. Partial outputs were preserved."
    )
    
    pipeline_status = getattr(
        error, "pipeline_status", None
    ) or getattr(update, "pipeline_status", None)
    if pipeline_status:
        st.write("**Pipeline status**")
        st.code(pipeline_status, language=None)
        
    run_id = getattr(error, "run_id", None) or getattr(
        update, "run_id", None
    )
    if run_id:
        st.write("**Saved run**")
        st.code(run_id, language=None)

    st.subheader("Processing timeline")
    completed_stage_keys = set(
        getattr(update, "completed_stage_keys", ())
    )
    failed_stage_key = getattr(update, "stage_key", None)
    
    for stage in PROCESSING_STAGES:
        label = _TIMELINE_LABELS[stage.key]
        if stage.key in completed_stage_keys:
            st.markdown(f":material/check_circle: {label}")
        elif stage.key == failed_stage_key:
            st.markdown(f":material/cancel: {label} — failed")
        else:
            st.markdown(f":material/skip_next: {label} — skipped")

    technical = getattr(error, "technical_details", None) or getattr(
        update, "technical_details", None
    )
    original_exception: dict[str, Any] | None = getattr(
        error, "original_exception", None
    ) or getattr(update, "original_exception", None)
    
    if technical:
        with st.expander(
            "Technical details",
            expanded=False,
            icon=":material/code:",
        ):
            if original_exception:
                structured_values = (
                    ("Exception type", original_exception.get("type")),
                    ("Message", original_exception.get("message")),
                    (
                        "Origin",
                        original_exception.get("source_filename")
                        or original_exception.get("file"),
                    ),
                    ("Function", original_exception.get("function")),
                    ("Line", original_exception.get("line")),
                    ("Pipeline status", pipeline_status),
                )
                for label_str, value in structured_values:
                    if value is not None:
                        st.write(f"**{label_str}**")
                        st.code(str(value), language=None)
                st.write("**Complete traceback**")
            st.code(technical, language="text")

    partial = tuple(getattr(error, "partial_artifacts", ())) or tuple(
        getattr(update, "partial_artifacts", ())
    )
    if partial:
        st.write("**Saved artifacts**")
        st.markdown(
            "\n".join(
                f"- {name}" for name in _artifact_summary(partial)
            )
        )
        with st.expander(
            "Filesystem locations",
            expanded=False,
            icon=":material/folder:",
        ):
            st.code(
                "\n".join(str(Path(path)) for path in partial),
                language=None,
            )
