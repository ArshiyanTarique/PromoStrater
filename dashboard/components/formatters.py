"""Reusable enum-to-label formatters and badge renderers for PromoStrater."""

from __future__ import annotations

import re
import streamlit as st

KNOWN_ENUM_LABELS: dict[str, str] = {
    # Decision Sources & Policies
    "AGREEMENT": "Agreement Policy",
    "AGREEMENT_POLICY": "Agreement Policy",
    "AUTO_ACCEPT": "Auto Accept",
    "MANUAL_REVIEW": "Manual Review",
    "LLM_ACCEPT": "LLM Approved",
    "NO_CANDIDATE": "No Candidate Found",
    "MODEL_ERROR": "Model Error",
    "SHADOW_MODE_ONLY": "Shadow Evaluation Only",
    "NOT_APPROVED_FOR_AUTOMATIC_MATCHING": "Not Approved For Auto Match",
    
    # Embedding Statuses & Fallbacks
    "EMBEDDING_FAILURE_SAFE_FALLBACK": "Embedding Fallback",
    "EMBEDDING_UNAVAILABLE": "Embedding Unavailable",
    "EMBEDDING_DISABLED": "Embedding Disabled",
    "EMBEDDING_LOAD_FAILED": "Embedding Load Failed",
    "EMBEDDING_RUNTIME_FAILED": "Embedding Runtime Failed",
    "NOT_EXERCISED": "Not Exercised",
    "ACTIVE": "Active / Used",
    "USED": "Used in Run",
    "DISABLED": "Disabled",
    "UNAVAILABLE": "Unavailable",
    
    # Pipeline & Execution Statuses
    "COMPLETED_DASHBOARD_ASSISTED": "Completed (Assisted)",
    "COMPLETED_DASHBOARD_SHADOW": "Completed (Shadow)",
    "COMPLETED_ASSISTED": "Completed (Assisted)",
    "COMPLETED_SHADOW": "Completed (Shadow)",
    "FAILED_DASHBOARD": "Failed Run",
    "PROCESSING": "Processing In Progress",
    "VALIDATING": "Validating Upload",
    "SAFE_AGREEMENT": "Safe Agreement",
    "CANDIDATE_NOT_SELECTED": "Candidate Not Selected",
    "CONFIRM": "Confirmed Structure",
    "MERGE_ENTITIES": "Merged Entities",
    "SPLIT_FURTHER": "Needs Further Split",
    "CORRECT_ATTRIBUTES": "Attributes Corrected",
    "GENUINELY_MIXED": "Genuinely Mixed Offer",
    "AMBIGUOUS": "Ambiguous Structure",

    # Diagnostic & Conflict Flags
    "protein_conflict": "Protein Conflict",
    "mixed_protein_ambiguity": "Mixed Protein Ambiguity",
    "strong_family_conflict": "Family Conflict",
    "strong_size_weight_conflict": "Size Mismatch",
    "strong_pack_format_conflict": "Pack Mismatch",
    "feature_generation_failure": "Feature Gen Failure",
    "missing_master": "Missing Product Master Item",
}

STATUS_COLOR_MAP: dict[str, str] = {
    # Badges / Colors: primary, success, warning, error, info, neutral
    "AUTO_ACCEPT": "success",
    "LLM_ACCEPT": "success",
    "COMPLETED": "success",
    "COMPLETED_DASHBOARD_ASSISTED": "success",
    "COMPLETED_DASHBOARD_SHADOW": "success",
    "COMPLETED_ASSISTED": "success",
    "COMPLETED_SHADOW": "success",
    "ACTIVE": "success",
    "USED": "success",
    "SAFE_AGREEMENT": "success",
    
    # Pipeline phases. Every surface badges a run with the same word, so the
    # colour has to resolve from that one vocabulary.
    "QUEUED": "info",
    "RUNNING": "info",
    "CANCELLING": "warning",
    "CANCELLED": "warning",

    "MANUAL_REVIEW": "warning",
    "SHADOW_MODE_ONLY": "warning",
    "NOT_APPROVED_FOR_AUTOMATIC_MATCHING": "warning",
    "EMBEDDING_FAILURE_SAFE_FALLBACK": "warning",
    "NOT_EXERCISED": "warning",
    "PROCESSING": "info",
    "VALIDATING": "info",
    
    "NO_CANDIDATE": "error",
    "MODEL_ERROR": "error",
    "FAILED": "error",
    "FAILED_DASHBOARD": "error",
    "EMBEDDING_UNAVAILABLE": "error",
    "EMBEDDING_LOAD_FAILED": "error",
    "EMBEDDING_RUNTIME_FAILED": "error",
    "DISABLED": "neutral",
    "IDLE": "neutral",
    "UNAVAILABLE": "error",
}


def format_enum_label(value: str | None) -> str:
    """Format raw enum strings or snake_case keys into readable titles with graceful fallback."""
    if not value:
        return "N/A"
    clean = str(value).strip()
    if clean in KNOWN_ENUM_LABELS:
        return KNOWN_ENUM_LABELS[clean]
    
    # Fallback formatting: replace underscores/dashes with spaces, handle casing
    formatted = re.sub(r"[_\-]+", " ", clean).strip()
    return formatted.title()


def get_status_color(value: str | None) -> str:
    """Map status/enum value to semantic CSS color class name."""
    if not value:
        return "neutral"
    clean = str(value).strip()
    return STATUS_COLOR_MAP.get(clean, "neutral")


def format_embedding_status(status: str | None, failure_reason: str | None = None) -> tuple[str, str]:
    """Return readable label and color category for embedding status."""
    if not status:
        return "Disabled", "neutral"
    
    clean = str(status).upper()
    label = format_enum_label(clean)
    if failure_reason:
        label += f" ({failure_reason})"
        
    color = get_status_color(clean)
    return label, color


def format_elapsed(seconds: float) -> str:
    """Format a duration identically wherever elapsed time is shown.

    The sidebar and the upload page both display the elapsed time of the same
    run, so they must not disagree on wording - one reading "84s" beside the
    other reading "1m 24s" looks like a sync bug even when the values match.

    Minutes are always printed, including for the first minute. A run's live
    reading is a CSS counter that cannot change its own shape as it crosses
    60s (see :mod:`dashboard.components.elapsed_clock`), so a shape that
    switched would make the clock jump from "59s" to "1m 0s" and back to "45s"
    when the same run finished.
    """
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "N/A"
    return f"{total // 60}m {total % 60}s"


def render_badge_html(label: str, color_type: str = "neutral") -> str:
    """Generate HTML string for a styled badge."""
    valid_color = color_type if color_type in {"primary", "success", "warning", "error", "info", "neutral"} else "neutral"
    return f'<span class="ps-badge ps-badge-{valid_color}">{label}</span>'


def render_badge(label: str, color_type: str = "neutral") -> None:
    """Render a styled inline badge using st.markdown."""
    st.markdown(render_badge_html(label, color_type), unsafe_allow_html=True)


def render_reason_codes(reasons: list[object]) -> None:
    """Format and render reason codes as compact wrapped badges instead of raw text."""
    if not reasons:
        st.caption("No specific diagnostic flags recorded.")
        return
    
    html_parts = ['<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem; margin-top: 0.25rem;">']
    for reason in reasons:
        r_str = str(reason).strip()
        if not r_str:
            continue
        label = format_enum_label(r_str)
        color = get_status_color(r_str)
        html_parts.append(render_badge_html(label, color))
    html_parts.append('</div>')
    
    st.markdown("".join(html_parts), unsafe_allow_html=True)
