"""Judge proposed competitor relationships and record durable labels.

This is the only place competitor ground truth is created. Competitor
discovery has always been rules - lately rules plus a borrowed own-brand
ranker - and neither has ever been measured, because no competitor pair has
ever been labelled. Every verdict saved here is one row of the dataset that
eventually calibrates a competitor threshold or trains a competitor model.

Deliberately separate from Human Validation. That page answers "is this offer
this SKU" (identity, one right answer); this one answers "would a shopper buy
this instead" (substitutability, a commercial judgement). Merging the two
would blur label sets that must stay distinguishable when either is trained on.

The queue is filled by ``competitors.review_staging_per_target``, which is 0
by default - so an empty queue here usually means staging is switched off
rather than that a run found nothing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.bootstrap import load_dashboard_context
from dashboard.components.common import safe_page_link
from dashboard.components.formatters import get_status_color, render_badge
from dashboard.components.sidebar_status import render_sidebar_status
from dashboard.theme import inject_theme
from sku_mapping.competitors.review import review_queue_summary
from sku_mapping.learning.store import LearningStoreError

st.set_page_config(
    page_title="Competitor Review", page_icon="🔍", layout="wide"
)

inject_theme()
try:
    _, store, _, _, _, _, jobs = load_dashboard_context()
except RuntimeError:
    st.error(
        "The dashboard data store could not be opened after a local schema "
        "upgrade. Restart the Streamlit server, then refresh this page."
    )
    st.stop()

render_sidebar_status(store, jobs)

st.title("Competitor Review")
st.caption(
    "Confirm or reject proposed competitor offers. These verdicts are the "
    "only competitor ground truth the system has."
)

try:
    rows = store.competitor_decisions()
except LearningStoreError as error:
    st.error(f"The competitor review queue could not be read: {error}")
    st.stop()

if not rows:
    st.info(
        "No competitor relationships have been staged for review.",
        icon="📭",
    )
    st.caption(
        "Staging is controlled by **competitors.review_staging_per_target** "
        "in `config/default.yaml`, which is `0` by default. Set it to a small "
        "number (for example `10`) and the next processing run will stage "
        "that many competitors per Master SKU."
    )
    safe_page_link(
        "pages/1_Upload_and_Process.py", "Start a processing run →", "▶️"
    )
    st.stop()

summary = review_queue_summary(rows)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Staged", summary.get("total", 0))
c2.metric("Pending", summary.get("PENDING", 0))
c3.metric("Confirmed", summary.get("CONFIRMED", 0))
c4.metric("Rejected", summary.get("REJECTED", 0))

# A reviewer works the pending queue; the decided rows stay visible so a
# verdict can be revisited without hunting through the database.
frame = pd.DataFrame(rows)
only_pending = st.toggle("Show only pending", value=True)
visible = frame[frame["decision"].eq("PENDING")] if only_pending else frame
if visible.empty:
    st.success("Every staged competitor has been reviewed.", icon="✅")
    st.stop()

skus = sorted(visible["master_sku"].dropna().unique().tolist())
selected_sku = st.selectbox("Master SKU", skus, key="competitor_review_sku")
for_sku = visible[visible["master_sku"].eq(selected_sku)]

st.subheader(f"{selected_sku} — {for_sku.iloc[0].get('master_name') or ''}")
st.caption(
    f"{len(for_sku)} competitor offer(s) awaiting judgement for this SKU."
)

reviewer = st.text_input(
    "Reviewer",
    value=st.session_state.get("competitor_reviewer", ""),
    placeholder="your.name@salesflo.com",
    help="Recorded with every verdict so a label can be traced to a person.",
)
st.session_state["competitor_reviewer"] = reviewer

for _, row in for_sku.iterrows():
    offer_id = str(row["competitor_offer_id"])
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{row.get('competitor_offer_name') or offer_id}**")
            st.caption(f"Brand: {row.get('competitor_brand') or 'Unknown'}")
            render_badge(
                str(row.get("decision") or "PENDING"),
                get_status_color(str(row.get("decision") or "PENDING")),
            )
        with right:
            # Ordering provenance, not a score. The model's margin is
            # explicitly NOT a competitor probability, so it is not shown as
            # one - a reviewer should judge the product, not chase a number.
            st.caption(f"**Proposed:** {row.get('proposed_status') or '-'}")
            st.caption(
                f"**Shortlist rank:** {row.get('lightgbm_rank') or '-'} "
                f"(ordered by {row.get('ranking_source') or 'rules'})"
            )
            if row.get("model_id"):
                st.caption(f"**Model:** `{row['model_id']}`")

        notes = st.text_input(
            "Notes (optional)",
            key=f"notes-{row['run_id']}-{selected_sku}-{offer_id}",
            placeholder="Why is this a rival, or why is it not?",
        )
        confirm, reject, unsure = st.columns(3)

        def _record(decision: str) -> None:
            if not reviewer.strip():
                st.warning("Enter a reviewer before saving a verdict.")
                return
            try:
                store.record_competitor_decision(
                    run_id=str(row["run_id"]),
                    master_sku=selected_sku,
                    competitor_offer_id=offer_id,
                    decision=decision,
                    reviewer=reviewer.strip(),
                    notes=notes or None,
                )
            except LearningStoreError as error:
                st.error(f"The verdict could not be saved: {error}")
                return
            st.rerun()

        if confirm.button(
            "Competitor", icon="✅", key=f"yes-{offer_id}", width="stretch"
        ):
            _record("CONFIRMED")
        if reject.button(
            "Not a competitor",
            icon="⛔",
            key=f"no-{offer_id}",
            width="stretch",
        ):
            _record("REJECTED")
        if unsure.button(
            "Unsure", icon="❔", key=f"maybe-{offer_id}", width="stretch"
        ):
            _record("UNSURE")

st.divider()
st.caption(
    "These labels measure precision at the top of each competitor list, "
    "because that is what the queue samples. They do not measure recall."
)
