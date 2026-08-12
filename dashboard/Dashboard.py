"""PromoStrater dashboard home."""

from __future__ import annotations

import streamlit as st

from dashboard.bootstrap import load_dashboard_context
from dashboard.components.ambient import inject_ambient
from dashboard.components.run_status import current_status, status_facts
from dashboard.components.sidebar_status import render_sidebar_status
from dashboard.components.theme_toggle import render_palette_toggle
from dashboard.theme import inject_theme

st.set_page_config(
    page_title="PromoStrater",
    page_icon="📦",
    layout="wide",
)

inject_theme()
inject_ambient()

config, store, *_, jobs = load_dashboard_context()

render_sidebar_status(store, jobs)

title_col, theme_col = st.columns([5, 1], vertical_alignment="center")
with title_col:
    st.title("PromoStrater")
with theme_col:
    render_palette_toggle()
st.caption(
    "Production SKU mapping & competitor offer discovery engine. "
    "Upload ClickFlyer promotions, run machine learning candidate mapping, "
    "perform human validation, and download audit-ready artifacts."
)

summary = store.summary()
counts = summary.get("counts", {})
# The KPI reports the same snapshot the sidebar card beside it is drawing, so
# the two cannot describe the engine differently on the same screen.
status = current_status(jobs, store)
facts = status_facts(status)

# Modern KPI Cards Grid.
# The whole grid must go out in a single st.markdown call: Streamlit wraps
# each call in its own container, so an opening <div> emitted on its own is
# auto-closed and the cards below it end up as stacked siblings instead of
# grid children.
active_val = facts["Status"]
active_sub = facts["File"] if status.has_run_detail else "Ready for input"

st.markdown(f'''
<div class="ps-kpi-grid">
    <div class="ps-kpi-card">
        <div class="ps-kpi-label">Persisted Pipeline Runs</div>
        <div class="ps-kpi-value">{counts.get("pipeline_runs", 0):,}</div>
        <div class="ps-kpi-subtext">Saved in SQLite store</div>
    </div>
    <div class="ps-kpi-card">
        <div class="ps-kpi-label">Pending Validations</div>
        <div class="ps-kpi-value">{counts.get("unanswered_reviews", 0):,}</div>
        <div class="ps-kpi-subtext">Awaiting reviewer answers</div>
    </div>
    <div class="ps-kpi-card">
        <div class="ps-kpi-label">New GOLD Labels</div>
        <div class="ps-kpi-value">{counts.get("new_gold_labels_since_last_model", 0):,}</div>
        <div class="ps-kpi-subtext">Collected for future retraining</div>
    </div>
    <div class="ps-kpi-card">
        <div class="ps-kpi-label">Engine Status</div>
        <div class="ps-kpi-value">{active_val}</div>
        <div class="ps-kpi-subtext">{active_sub}</div>
    </div>
</div>
''', unsafe_allow_html=True)

st.subheader("Pipeline Workflow")

# Compact Workflow Stepper Navigation
col1, col2 = st.columns(2)

with col1:
    st.page_link(
        "pages/1_Upload_and_Process.py",
        label="1. Upload & Process",
        icon="⬆️",
        help="Upload ClickFlyer CSV/Excel dumps, configure mode, and execute candidate scoring.",
    )
    st.page_link(
        "pages/2_Human_Validation.py",
        label="2. Human Validation",
        icon="✅",
        help="Review uncertain SKU proposals, record corrections, and accumulate GOLD labels.",
    )

with col2:
    st.page_link(
        "pages/3_Results_and_Downloads.py",
        label="3. Results & Downloads",
        icon="📥",
        help="Inspect run summaries, candidate statistics, and download validated CSV outputs.",
    )
    st.page_link(
        "pages/4_Models_and_Learning.py",
        label="4. Models & Learning",
        icon="🧠",
        help="Examine registered LightGBM models, precision/recall metrics, and runtime components.",
    )

st.caption(
    "The SQLite learning store is the durable source of truth. Browser refreshes "
    "and page navigation preserve all completed runs, active processing states, and saved answers."
)
