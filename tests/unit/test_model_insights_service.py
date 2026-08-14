"""User-facing model-insight evidence and missing-data behavior."""

from __future__ import annotations

import json

from dashboard.services.model_insights_service import (
    ModelInsightsService,
    _confidence_frame,
    _evaluation,
)
from dashboard.services.registry_service import DashboardRegistryService
from sku_mapping.config import load_config
from sku_mapping.learning.store import LearningStore
from tests.registered_models import registered_model_id


def test_registered_model_builds_plain_language_quality_and_outcomes(
    tmp_path,
) -> None:
    config = load_config("config/default.yaml")
    registry = DashboardRegistryService(config)
    selected = next(
        model
        for model in registry.list_models()
        if model.model_id == registered_model_id()
    )

    insights = ModelInsightsService(
        config,
        LearningStore(tmp_path / "learning.db"),
        registry,
    ).build(selected.model_id)

    assert insights.status_label == "Shadow evaluation"
    assert insights.available is True
    assert not insights.quality.empty
    assert set(insights.quality["Metric"]) == {
        "Precision",
        "Recall",
        "F1 score",
    }
    assert not insights.outcomes.empty
    assert insights.comparison_warning
    assert "metadata" in insights.technical_details


def test_missing_evaluation_and_confidence_data_are_safe() -> None:
    quality, outcomes, sample, manual_rate, signature = _evaluation({})
    confidence = _confidence_frame({})

    assert quality == {}
    assert outcomes == []
    assert sample is None
    assert manual_rate is None
    assert signature == "no_comparable_evaluation"
    assert confidence.empty


def test_confidence_chart_uses_stored_distribution_counts() -> None:
    confidence = _confidence_frame(
        {
            "calibrated_probability_distribution": [
                {"lower": 0.0, "upper": 0.1, "rows": 7},
                {"lower": 0.9, "upper": 1.0, "rows": 3},
            ]
        }
    )

    assert confidence["Predictions"].sum() == 10
    assert confidence.iloc[0]["Confidence range"] == "Below 50%"
    assert confidence.iloc[-1]["Confidence range"] == "Above 95%"


def test_requested_available_and_used_runtime_states_stay_distinct(
    tmp_path,
) -> None:
    config = load_config("config/default.yaml")
    registry = DashboardRegistryService(config)
    selected = next(
        model
        for model in registry.list_models()
        if model.model_id == registered_model_id()
    )
    statistics = tmp_path / "statistics.json"
    monitoring = tmp_path / "monitoring.json"
    statistics.write_text(
        json.dumps(
            {
                "llm_review_enabled": True,
                "llm_calls": 0,
            }
        ),
        encoding="utf-8",
    )
    monitoring.write_text(
        json.dumps(
            {
                "calibrated_probability_distribution": [],
            }
        ),
        encoding="utf-8",
    )
    store = LearningStore(tmp_path / "learning.db")
    store.upsert_pipeline_run(
        {
            "run_id": "runtime-run",
            "model_id": selected.model_id,
            "status": "FAILED_DASHBOARD",
            "output_paths": {
                "unified_statistics": statistics,
                "monitoring_report": monitoring,
            },
        }
    )

    runtime = ModelInsightsService(
        config, store, registry
    ).build(selected.model_id).runtime_components.set_index("Component")

    assert runtime.loc["Local LLM reviewer", "Requested"] == "Yes"
    assert runtime.loc[
        "Local LLM reviewer", "Available"
    ] == "Not exercised"
