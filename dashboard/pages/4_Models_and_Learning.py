"""Read-only model status, evaluation evidence, and runtime availability."""

from __future__ import annotations

import streamlit as st

from dashboard.bootstrap import load_dashboard_context
from dashboard.components.sidebar_status import render_sidebar_status
from dashboard.services.model_insights_service import ModelInsightsService
from dashboard.theme import chart_accent, inject_theme

st.set_page_config(
    page_title="Models & Learning",
    page_icon=":material/model_training:",
    layout="wide",
)

inject_theme()
config, store, _, registry, _, _, jobs = load_dashboard_context()
render_sidebar_status(store, jobs)

st.title("Models & Learning")
st.caption(
    "Understand which model is selected, what its evaluation shows, and "
    "whether supporting components were actually available in the latest run."
)

models = registry.list_models()
if not models:
    st.error("No compatible registered model is available.")
    st.stop()

model_ids = [model.model_id for model in models]
default_model = (
    str(config.ml.model_id)
    if str(config.ml.model_id) in model_ids
    else model_ids[0]
)
model_id = st.selectbox(
    "Selected model",
    model_ids,
    index=model_ids.index(default_model),
    format_func=lambda value: next(
        model.display_label for model in models if model.model_id == value
    ),
)


@st.cache_data(ttl="2m", max_entries=10)
def load_insights(selected_model_id: str):
    return ModelInsightsService(config, store, registry).build(
        selected_model_id
    )


try:
    insights = load_insights(model_id)
except (OSError, ValueError) as error:
    st.error(
        f"The selected model is unavailable: {error}",
        icon=":material/error:",
    )
    st.stop()

with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(insights.option.model_version)
        st.badge(
            insights.status_label,
            color="blue",
            icon=":material/science:",
        )
        st.badge(
            "Available" if insights.available else "Unavailable",
            color="green" if insights.available else "red",
            icon=(
                ":material/check_circle:"
                if insights.available
                else ":material/error:"
            ),
        )
        st.badge(
            "Not approved for automatic matching",
            color="orange",
            icon=":material/person_check:",
        )
    st.write(insights.status_description)
    if insights.latest_run_id:
        st.caption(f"Latest observed run: {insights.latest_run_id}")

with st.container(horizontal=True):
    st.metric(
        "Evaluation sample",
        (
            f"{insights.sample_size:,}"
            if insights.sample_size is not None
            else "Not available"
        ),
        border=True,
        help="Rows used by the displayed evaluation evidence.",
    )
    st.metric(
        "Precision",
        (
            f"{insights.precision:.1%}"
            if insights.precision is not None
            else "Not available"
        ),
        border=True,
        help="How often the model's accepted proposals were correct.",
    )
    st.metric(
        "Manual-review share",
        (
            f"{insights.manual_review_rate:.1%}"
            if insights.manual_review_rate is not None
            else "Not available"
        ),
        border=True,
        help="Share routed to a person under the evaluated policy.",
    )

st.subheader("Model quality overview")
if insights.quality.empty:
    st.info(
        "No compatible evaluation metrics are stored for this model.",
        icon=":material/info:",
    )
else:
    st.bar_chart(
        insights.quality,
        x="Metric",
        y="Score",
        y_label="Score",
        color=chart_accent(),
    )
    st.caption(
        "Precision shows how often proposed matches were correct. Recall shows "
        "how many real matches were found. F1 balances precision and recall. "
        "These are calibration-policy results, not final production accuracy."
    )

chart_left, chart_right = st.columns(2)
with chart_left.container(border=True, height="stretch"):
    st.subheader("Decision outcomes")
    if insights.outcomes.empty or not insights.outcomes["Count"].sum():
        st.info("No decision-outcome counts are stored.")
    else:
        st.bar_chart(
            insights.outcomes, x="Outcome", y="Count", color=chart_accent()
        )
        st.caption(
            "Counts use the evaluation categories stored with this model; "
            "missing categories are not invented."
        )

with chart_right.container(border=True, height="stretch"):
    st.subheader("Confidence distribution")
    if insights.confidence.empty:
        st.info(
            "No confidence distribution is available for the latest run."
        )
    else:
        st.bar_chart(
            insights.confidence,
            x="Confidence range",
            y="Predictions",
            color=chart_accent(),
        )
        st.caption(
            "This shows candidate probabilities observed in the latest run, "
            "not reviewed correctness."
        )

st.subheader("Comparison with previous versions")
if insights.comparison.empty or insights.comparison["Version"].nunique() < 2:
    if insights.comparison_warning:
        st.warning(insights.comparison_warning)
    else:
        st.info("No compatible earlier evaluation is available.")
else:
    st.bar_chart(
        insights.comparison,
        x="Version",
        y="Score",
        color="Metric",
    )
    st.caption(
        "Only versions evaluated on compatible data with the same metric "
        "definition are compared."
    )

st.subheader("Most common diagnostic conflicts")
if insights.errors.empty:
    st.info(
        "No candidate conflict categories are available for the latest run."
    )
else:
    st.bar_chart(
        insights.errors,
        x="Error type",
        y="Offers",
        horizontal=True,
        color=chart_accent(),
    )
    st.caption(
        "Conflict counts use each offer's top candidate and may overlap when "
        "one offer has more than one issue."
    )

st.subheader("Runtime components")
st.dataframe(
    insights.runtime_components,
    hide_index=True,
    width="stretch",
)
st.caption(
    "Requested configuration, runtime availability, and actual use are "
    "reported separately. A selected option does not prove a component ran."
)

with st.container(border=True):
    st.subheader("What should I do next?")
    st.write(
        "Continue using human review for proposed matches. This model lacks "
        "the evidence required for automatic confirmation. Collect reviewed "
        "results before considering a separately governed promotion."
    )

with st.expander(
    "Deployment modes explained",
    expanded=False,
    icon=":material/help:",
):
    st.write(
        "**Assisted:** the model proposes matches and uncertain cases are "
        "reviewed by a person."
    )
    st.write(
        "**Shadow:** the model is evaluated in the background and does not "
        "control final results."
    )
    st.write(
        "Automatic matching is not enabled or approved by the current "
        "registry."
    )

with st.expander(
    "Technical details",
    expanded=False,
    icon=":material/code:",
):
    st.caption(
        "Engineering metadata is preserved here without dominating the "
        "decision view."
    )
    st.json(insights.technical_details, expanded=False)
