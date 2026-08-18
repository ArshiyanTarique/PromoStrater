"""Complete a durable bounded human-validation session."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.bootstrap import load_dashboard_context
from dashboard.components.common import (
    active_run_mode,
    render_developer_mode_banner,
    select_run,
)
from dashboard.components.formatters import (
    format_enum_label,
    get_status_color,
    render_badge,
    render_reason_codes,
)
from dashboard.components.sidebar_status import render_sidebar_status
from dashboard.services.review_service import ReviewAnswerError
from dashboard.theme import inject_theme
from sku_mapping.learning.store import (
    DuplicateHumanReviewError,
    LearningStoreError,
)

st.set_page_config(page_title="Human Validation", page_icon="✅", layout="wide")

inject_theme()
try:
    _, store, _, _, reviews, _, jobs = load_dashboard_context()
except RuntimeError:
    st.error(
        "The dashboard data store could not be opened after a local schema "
        "upgrade. Restart the Streamlit server, then refresh this page."
    )
    st.stop()

render_sidebar_status(store, jobs)

st.title("Human Validation")
st.caption("Review proposed own-brand SKU mappings and save durable GOLD labels for future model retraining.")

render_developer_mode_banner()
runs = reviews.runs_with_review_sessions(run_mode=active_run_mode())
selected_run = select_run(
    runs,
    key="validation_run",
    preferred_run_id=st.session_state.get("active_run_id"),
)
if selected_run is None:
    st.stop()

run_id = str(selected_run["run_id"])
session_id, questions = reviews.questions_for_run(run_id)
if not questions:
    st.info("This run has no unreviewed own-brand SKU proposals.")
    st.stop()

progress = reviews.progress(session_id)
st.progress(progress.answered / progress.total if progress.total else 0)
st.caption(f"Validation progress: **{progress.answered} of {progress.total}** answered")

index_key = f"review_index_{session_id}"
if index_key not in st.session_state:
    first_unanswered = next(
        (
            position
            for position, question in enumerate(questions)
            if question["answered_at"] is None
        ),
        0,
    )
    st.session_state[index_key] = first_unanswered

index = max(0, min(int(st.session_state[index_key]), len(questions) - 1))
question = questions[index]

# Header for Question
st.subheader(f"Question {index + 1} of {len(questions)}")

# Primary Comparison Cards
col_offer, col_match = st.columns(2)

with col_offer:
    with st.container(border=True):
        st.markdown("**Source Offer Description**")
        st.subheader(question["offer_description"])
        
        if int(question.get("entity_count") or 1) > 1:
            st.caption("**Multi-Product Parent Offer:** " + str(question.get("source_offer_text") or question["offer_description"]))
            st.caption(
                f"Parsed Entity {question.get('entity_index')}/{question.get('entity_count')} "
                f"· Conjunction: {question.get('conjunction_type') or 'N/A'} "
                f"· Parse Confidence: {question.get('entity_parse_confidence', 'N/A')}"
            )
            flags = str(question.get("attribute_inheritance_flags") or "")
            if flags:
                st.caption("Inherited attributes: " + flags)

with col_match:
    with st.container(border=True):
        st.markdown("**Proposed Product Master Match**")
        st.code(question["suggested_candidate_id"], language=None)
        st.subheader(question["suggested_candidate_description"])

# Primary Metrics & Status Badges
with st.container(border=True):
    col_conf, col_status = st.columns([1, 2])
    
    with col_conf:
        probability = question.get("lightgbm_probability")
        conf_str = f"{float(probability):.1%}" if probability is not None else "Unavailable"
        st.metric("Model Confidence", conf_str)
        
    with col_status:
        st.markdown("**Match Diagnostics & Status**")
        source_raw = str(question.get("decision_source") or "Unknown")
        agreement_raw = str(question.get("agreement_status") or "Unavailable")
        
        st.markdown("Decision Source: ", unsafe_allow_html=True)
        render_badge(format_enum_label(source_raw), get_status_color(source_raw))
        
        st.markdown("Agreement Status: ", unsafe_allow_html=True)
        render_badge(format_enum_label(agreement_raw), get_status_color(agreement_raw))
        
        reasons = [question["selection_reason"]]
        reasons.extend(question.get("conflict_flags") or [])
        if question.get("fallback_reason"):
            reasons.append(str(question["fallback_reason"]))
        
        st.markdown("<small style='color: var(--ps-text-secondary); margin-top: 0.5rem; display: block;'>Reason Badges:</small>", unsafe_allow_html=True)
        render_reason_codes(reasons)

# Alternative Candidates Table
alternatives = pd.DataFrame(question["supplied_candidates"])
if not alternatives.empty:
    with st.expander("Supplied candidate alternatives", expanded=False):
        formatted_alt = alternatives.copy()
        if "lightgbm_probability" in formatted_alt:
            formatted_alt["Confidence"] = formatted_alt["lightgbm_probability"].apply(
                lambda p: f"{float(p):.1%}" if pd.notnull(p) else "N/A"
            )
        
        st.dataframe(
            formatted_alt[
                [
                    "candidate_rank",
                    "candidate_id",
                    "candidate_description",
                    "Confidence",
                ]
            ],
            hide_index=True,
            width='stretch',
        )

# Answer Section
if question["answered_at"] is not None:
    saved = "Correct (True)" if question["human_answer"] else "Incorrect (False)"
    st.success(
        f"Saved Answer: **{saved}** · Label Quality: **{question['label_quality']}**",
        icon="✅",
    )
    corrected_candidate_id = question.get("corrected_candidate_id")
    if corrected_candidate_id:
        master_options = reviews.master_options()
        corrected_description = master_options.get(
            str(corrected_candidate_id),
            next(
                (
                    item["candidate_description"]
                    for item in question["supplied_candidates"]
                    if item["candidate_id"] == corrected_candidate_id
                ),
                "",
            ),
        )
        st.write("**Reviewer-Selected Product Master SKU:**")
        st.code(f"{corrected_candidate_id} — {corrected_description}")
else:
    with st.container(border=True):
        st.subheader("Record Validation Decision")
        
        reviewer_id = st.text_input(
            "Reviewer identifier (optional)",
            key=f"reviewer_{question['review_id']}",
        )
        
        choice = st.radio(
            "Is the proposed master SKU correct?",
            ["True (Match is correct)", "False (Match is incorrect)"],
            horizontal=True,
            index=None,
            key=f"answer_{question['review_id']}",
        )
        
        answer_code = None
        corrected = None
        
        if choice == "True (Match is correct)":
            answer_code = "TRUE"
        elif choice == "False (Match is incorrect)":
            resolution = st.radio(
                "Required Correction Action",
                [
                    "Select correct Product Master SKU",
                    "None of these candidates (No match)",
                    "Cannot determine (Ambiguous)",
                ],
                index=None,
                key=f"resolution_{question['review_id']}",
            )
            if resolution == "Select correct Product Master SKU":
                supplied_options = [
                    item["candidate_id"]
                    for item in question["supplied_candidates"]
                ]
                master_options = reviews.master_options()
                options = list(
                    dict.fromkeys(
                        [*supplied_options, *sorted(master_options)]
                    )
                )
                corrected = st.selectbox(
                    "Correct candidate SKU",
                    options,
                    format_func=lambda code: next(
                        (
                            f"{item['candidate_id']} — "
                            f"{item['candidate_description']}"
                            for item in question["supplied_candidates"]
                            if item["candidate_id"] == code
                        ),
                        f"{code} — {master_options.get(code, '')}",
                    ),
                    key=f"corrected_{question['review_id']}",
                )
                answer_code = "FALSE_CANDIDATE"
            elif resolution == "None of these candidates (No match)":
                answer_code = "FALSE_NONE"
            elif resolution == "Cannot determine (Ambiguous)":
                answer_code = "FALSE_CANNOT_DETERMINE"

        with st.expander("Advanced Entity & Attribute Corrections (Optional)", expanded=False):
            decomposition_action = st.selectbox(
                "Entity Decomposition Action",
                [
                    "CONFIRM",
                    "MERGE_ENTITIES",
                    "SPLIT_FURTHER",
                    "CORRECT_ATTRIBUTES",
                    "GENUINELY_MIXED",
                    "AMBIGUOUS",
                ],
                key=f"decomposition_{question['review_id']}",
            )
            corrected_entity_text = st.text_input(
                "Corrected entity text",
                value=str(question.get("entity_text") or ""),
                key=f"entity_text_{question['review_id']}",
            )
            corrected_attributes_json = st.text_area(
                "Corrected attributes JSON",
                value="",
                key=f"entity_attributes_{question['review_id']}",
            )
        
        notes = st.text_area("Review notes (optional)", key=f"notes_{question['review_id']}")
        
        if st.button(
            "Save Validation Decision",
            type="primary",
            icon="💾",
            disabled=answer_code is None,
            key=f"save_{question['review_id']}",
            width='stretch',
        ):
            try:
                reviews.save(
                    review_id=str(question["review_id"]),
                    answer=str(answer_code),
                    corrected_candidate_id=corrected,
                    reviewer_id=reviewer_id,
                    notes=notes,
                    decomposition_action=decomposition_action,
                    corrected_entity_text=corrected_entity_text or None,
                    corrected_attributes_json=(
                        corrected_attributes_json or None
                    ),
                )
                st.success("Answer saved durably.")
                st.rerun()
            except (
                ReviewAnswerError,
                DuplicateHumanReviewError,
                LearningStoreError,
            ) as error:
                st.error(str(error))

# Navigation Controls
back, _, next_column = st.columns([1, 4, 1])
if back.button("← Previous", disabled=index == 0, width='stretch'):
    st.session_state[index_key] = index - 1
    st.rerun()
if next_column.button(
    "Next →",
    disabled=index == len(questions) - 1,
    width='stretch',
):
    st.session_state[index_key] = index + 1
    st.rerun()

# Collapsed Technical Details
with st.expander("Technical Details & Raw Diagnostics", expanded=False):
    st.caption("Engineering metadata and exact internal identifiers:")
    st.json(
        {
            "review_id": question.get("review_id"),
            "prediction_id": question.get("prediction_id"),
            "run_id": question.get("run_id"),
            "offer_id": question.get("offer_id"),
            "entity_id": question.get("entity_id"),
            "decision_source": question.get("decision_source"),
            "agreement_status": question.get("agreement_status"),
            "selection_reason": question.get("selection_reason"),
            "fallback_reason": question.get("fallback_reason"),
            "conflict_flags": question.get("conflict_flags"),
        },
        expanded=False,
    )

progress = reviews.progress(session_id)
if progress.total and progress.answered == progress.total:
    st.success(
        f"All {progress.total} validation questions are complete. "
        "Human-confirmed GOLD labels have been safely stored."
    )
