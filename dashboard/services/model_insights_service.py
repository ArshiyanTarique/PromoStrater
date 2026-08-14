"""Plain-language, read-only model insights from persisted evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.services.registry_service import (
    DashboardModelOption,
    DashboardRegistryService,
)
from sku_mapping.config import PipelineConfig
from sku_mapping.learning.store import LearningStore

ERROR_COLUMNS = {
    "protein_conflict": "Protein conflict",
    "mixed_protein_ambiguity": "Mixed-protein ambiguity",
    "strong_family_conflict": "Family conflict",
    "strong_size_weight_conflict": "Size mismatch",
    "strong_pack_format_conflict": "Pack mismatch",
    "feature_generation_failure": "Feature-generation failure",
    "missing_master": "Missing Product Master item",
}


@dataclass(frozen=True)
class ModelInsights:
    option: DashboardModelOption
    available: bool
    status_label: str
    status_description: str
    sample_size: int | None
    precision: float | None
    manual_review_rate: float | None
    quality: pd.DataFrame
    outcomes: pd.DataFrame
    confidence: pd.DataFrame
    errors: pd.DataFrame
    comparison: pd.DataFrame
    comparison_warning: str | None
    runtime_components: pd.DataFrame
    latest_run_id: str | None
    technical_details: dict[str, Any]


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _metadata_path(
    config: PipelineConfig, option: DashboardModelOption
) -> Path:
    return (
        config.shadow_mode.registry_path.parent
        / "metadata"
        / f"{Path(option.package_filename).stem}.json"
    )


def _evaluation(
    metadata: dict[str, Any],
) -> tuple[
    dict[str, float],
    list[dict[str, object]],
    int | None,
    float | None,
    str,
]:
    metrics = metadata.get("metrics", {})
    threshold = metadata.get("threshold_evidence", {}).get(
        "selected_metrics", {}
    )
    if threshold:
        precision = _safe_float(threshold.get("auto_match_precision"))
        recall = _safe_float(threshold.get("auto_match_recall"))
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        quality = {
            key: value
            for key, value in {
                "Precision": precision,
                "Recall": recall,
                "F1 score": f1,
            }.items()
            if value is not None
        }
        outcomes = [
            {
                "Outcome": "Correct auto proposals",
                "Count": int(threshold.get("auto_match_true_positives", 0)),
            },
            {
                "Outcome": "Incorrect auto proposals",
                "Count": int(threshold.get("auto_match_false_positives", 0)),
            },
            {
                "Outcome": "Manual review",
                "Count": int(threshold.get("manual_review_rows", 0)),
            },
            {
                "Outcome": "No match",
                "Count": int(threshold.get("no_match_rows", 0)),
            },
        ]
        sample_size = int(
            metrics.get("calibration", {}).get("rows")
            or sum(item["Count"] for item in outcomes)
        )
        return (
            quality,
            outcomes,
            sample_size,
            _safe_float(threshold.get("manual_review_coverage")),
            "calibration_threshold_evidence",
        )

    test = metrics.get("test", {})
    if test:
        matrix = test.get("confusion_matrix", {})
        correct = int(matrix.get("true_positive", 0)) + int(
            matrix.get("true_negative", 0)
        )
        incorrect = int(matrix.get("false_positive", 0)) + int(
            matrix.get("false_negative", 0)
        )
        accuracy = (
            correct / (correct + incorrect)
            if correct + incorrect
            else None
        )
        quality = {
            key: value
            for key, value in {
                "Precision": _safe_float(test.get("precision")),
                "Recall": _safe_float(test.get("recall")),
                "F1 score": _safe_float(test.get("f1")),
                "Accuracy": accuracy,
            }.items()
            if value is not None
        }
        return (
            quality,
            [
                {"Outcome": "Correct predictions", "Count": correct},
                {"Outcome": "Incorrect predictions", "Count": incorrect},
            ],
            int(test.get("row_count", correct + incorrect)),
            None,
            "held_out_test",
        )
    # Fallback: ranked/experimental models store CV metrics directly.
    cv_hit1 = _safe_float(metrics.get("cv_hit1"))
    cv_pr_auc = _safe_float(metrics.get("cv_pr_auc"))
    if cv_hit1 is not None or cv_pr_auc is not None:
        quality = {k: v for k, v in {"Hit@1 (CV)": cv_hit1, "PR-AUC (CV)": cv_pr_auc}.items() if v is not None}
        trained_on = str(metrics.get("trained_on", ""))
        return quality, [], None, None, f"cross_validation ({trained_on})"

    return {}, [], None, None, "no_comparable_evaluation"


def _confidence_frame(monitoring: dict[str, Any]) -> pd.DataFrame:
    rows = []
    requested_bins = (
        (0.0, 0.5, "Below 50%"),
        (0.5, 0.65, "50–65%"),
        (0.65, 0.75, "65–75%"),
        (0.75, 0.85, "75–85%"),
        (0.85, 0.95, "85–95%"),
        (0.95, 1.01, "Above 95%"),
    )
    source = monitoring.get("calibrated_probability_distribution", [])
    for lower, upper, label in requested_bins:
        count = 0
        for item in source:
            item_lower = _safe_float(item.get("lower"))
            item_upper = _safe_float(item.get("upper"))
            if item_lower is None or item_upper is None:
                continue
            midpoint = (item_lower + item_upper) / 2
            if lower <= midpoint < upper:
                count += int(item.get("rows", 0))
        rows.append({"Confidence range": label, "Predictions": count})
    frame = pd.DataFrame(rows)
    return frame if frame["Predictions"].sum() else frame.iloc[0:0]


def _error_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame(columns=["Error type", "Offers"])
    requested = {"candidate_rank", *ERROR_COLUMNS}
    try:
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in requested,
            low_memory=False,
        )
    except (OSError, ValueError):
        return pd.DataFrame(columns=["Error type", "Offers"])
    if "candidate_rank" in frame:
        frame = frame[
            pd.to_numeric(frame["candidate_rank"], errors="coerce").eq(1)
        ]
    rows = []
    for column, label in ERROR_COLUMNS.items():
        if column not in frame:
            continue
        values = frame[column]
        if values.dtype == object:
            values = values.astype(str).str.casefold().isin(
                {"true", "1", "yes"}
            )
        count = int(values.fillna(False).astype(bool).sum())
        if count:
            rows.append({"Error type": label, "Offers": count})
    return pd.DataFrame(rows).sort_values(
        "Offers", ascending=False, ignore_index=True
    ) if rows else pd.DataFrame(columns=["Error type", "Offers"])


class ModelInsightsService:
    """Build user-facing model evidence without modifying registry data."""

    def __init__(
        self,
        config: PipelineConfig,
        store: LearningStore,
        registry: DashboardRegistryService,
    ) -> None:
        self.config = config
        self.store = store
        self.registry = registry

    def _latest_run(
        self, model_id: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
        for run in self.store.list_pipeline_runs(limit=100):
            if str(run.get("model_id") or "") != model_id:
                continue
            paths = run.get("output_paths", {})
            statistics = _read_json(
                Path(paths["unified_statistics"])
                if paths.get("unified_statistics")
                else None
            )
            monitoring = _read_json(
                Path(paths["monitoring_report"])
                if paths.get("monitoring_report")
                else None
            )
            return run, statistics, monitoring
        return None, {}, {}

    def build(self, model_id: str) -> ModelInsights:
        options = {
            option.model_id: option for option in self.registry.list_models()
        }
        option = options.get(model_id)
        if option is None:
            raise ValueError("Selected model is not in the registry")
        availability_error = None
        try:
            self.registry.validate_model_id(model_id)
            available = True
        except ValueError as error:
            available = False
            availability_error = str(error)
        metadata = _read_json(_metadata_path(self.config, option))
        quality_values, outcome_rows, sample_size, manual_rate, signature = (
            _evaluation(metadata)
        )
        quality = pd.DataFrame(
            [
                {"Metric": name, "Score": score}
                for name, score in quality_values.items()
            ]
        )
        outcomes = pd.DataFrame(outcome_rows)
        latest_run, statistics, monitoring = self._latest_run(model_id)
        paths = latest_run.get("output_paths", {}) if latest_run else {}

        comparisons = []
        incompatible = False
        dataset_hash = str(
            metadata.get("training_dataset_hash")
            or metadata.get("processed_feature_table_hash")
            or ""
        )
        for candidate in self.registry.list_models():
            candidate_metadata = _read_json(
                _metadata_path(self.config, candidate)
            )
            candidate_quality, _, _, _, candidate_signature = _evaluation(
                candidate_metadata
            )
            candidate_hash = str(
                candidate_metadata.get("training_dataset_hash")
                or candidate_metadata.get("processed_feature_table_hash")
                or ""
            )
            compatible = (
                signature == candidate_signature
                and bool(dataset_hash)
                and dataset_hash == candidate_hash
            )
            if candidate.model_id != model_id and not compatible:
                incompatible = True
            if compatible:
                created = str(
                    candidate_metadata.get("training_timestamp")
                    or candidate_metadata.get("creation_timestamp")
                    or ""
                )[:10]
                for metric, score in candidate_quality.items():
                    comparisons.append(
                        {
                            "Version": (
                                f"{candidate.model_version} {created}".strip()
                            ),
                            "Metric": metric,
                            "Score": score,
                        }
                    )

        llm_calls = int(statistics.get("llm_calls", 0) or 0)
        runtime_components = pd.DataFrame(
            [
                {
                    "Component": "Local LLM reviewer",
                    "Requested": (
                        "Yes"
                        if statistics.get("llm_review_enabled")
                        else "No"
                    ),
                    "Available": (
                        "Yes" if llm_calls else "Not exercised"
                    ),
                    "Used in latest run": (
                        f"Yes — {llm_calls:,} calls" if llm_calls else "No"
                    ),
                },
            ]
        )

        return ModelInsights(
            option=option,
            available=available,
            status_label=(
                "Shadow evaluation" if available else "Unavailable"
            ),
            status_description=(
                (
                    "This model can suggest SKU matches, but every result "
                    "still requires the configured review policy. It is not "
                    "approved for automatic matching."
                )
                if available
                else (
                    "This registered package cannot currently be loaded by "
                    "the runtime. Select an available model before processing."
                )
            ),
            sample_size=sample_size,
            precision=quality_values.get("Precision"),
            manual_review_rate=manual_rate,
            quality=quality,
            outcomes=outcomes,
            confidence=_confidence_frame(monitoring),
            errors=_error_frame(
                Path(paths["shadow_predictions_csv"])
                if paths.get("shadow_predictions_csv")
                else None
            ),
            comparison=pd.DataFrame(comparisons),
            comparison_warning=(
                "Other registered versions use a different evaluation split "
                "or metric definition, so a direct comparison would be "
                "misleading."
                if incompatible and len({row["Version"] for row in comparisons}) < 2
                else None
            ),
            runtime_components=runtime_components,
            latest_run_id=(
                str(latest_run["run_id"]) if latest_run else None
            ),
            technical_details={
                "registry": option.__dict__,
                "metadata": metadata,
                "latest_run": latest_run or {},
                "evaluation_signature": signature,
                "availability_error": availability_error,
            },
        )
