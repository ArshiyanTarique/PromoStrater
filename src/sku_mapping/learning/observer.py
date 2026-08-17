"""Non-training persistence adapter for Phase 6 unified inference results."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sku_mapping.config import PipelineConfig
from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.learning.models import LabelQuality
from sku_mapping.learning.store import LearningStore

LOGGER = logging.getLogger(__name__)

CONFLICT_COLUMNS = (
    "protein_conflict",
    "mixed_protein_ambiguity",
    "strong_family_conflict",
    "strong_size_weight_conflict",
    "strong_pack_format_conflict",
    "feature_generation_failure",
    "missing_master",
    "pack_conflict",
)


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else None


def _json_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_nonempty(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    values = frame[column].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return str(values.iloc[0]) if not values.empty else None


def observe_unified_result(
    result: Any,
    *,
    config: PipelineConfig,
    source_path: str | Path | None = None,
) -> str | None:
    """Persist one completed/failed run and its candidate observations.

    This function only inserts observational records and governed label
    proposals. It never imports a trainer or calls ``fit``.
    """
    if not config.learning_store.enabled or not result.run_id:
        return None
    store = LearningStore(config.learning_store.database_path)
    now = datetime.now(timezone.utc).isoformat()
    candidates = result.candidates
    decisions = result.decisions
    source = Path(source_path) if source_path is not None else None
    decision_by_offer = (
        {
            str(row["offer_id"]): row
            for _, row in decisions.iterrows()
        }
        if not decisions.empty
        else {}
    )
    llm_model_id = _first_nonempty(candidates, "llm_model_id")
    package_hash = _first_nonempty(candidates, "model_package_sha256")
    store.upsert_pipeline_run(
        {
            "run_id": result.run_id,
            "started_at": now,
            "completed_at": now,
            "source_filename": source.name if source else None,
            "source_file_hash": _sha256_file(source),
            "source_row_count": result.statistics.get(
                "offers_processed", len(decisions)
            ),
            "unique_offer_count": int(
                decisions["offer_id"].nunique()
                if "offer_id" in decisions
                else len(decisions)
            ),
            "deployment_mode": config.ml.mode.value,
            "status": result.status,
            "model_id": config.ml.model_id,
            "llm_model_id": llm_model_id,
            "threshold": config.ml.auto_accept_threshold,
            "output_paths": {
                key: str(value)
                for key, value in result.output_paths.items()
            },
            "stage_runtimes": result.statistics.get(
                "stage_runtimes_seconds", {}
            ),
            "error_summary": result.error,
        }
    )
    if config.ml.model_id:
        store.register_model_version(
            model_id=config.ml.model_id,
            model_hash=package_hash,
            status="OBSERVED_IN_" + config.ml.mode.value.upper(),
            champion_status="OBSERVED_NOT_ACTIVATED_BY_LEARNING_STORE",
        )
    store.add_offer_decisions(
        str(result.run_id),
        decisions.to_dict(orient="records"),
    )
    prediction_records: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        offer_id = _safe_text(candidate.get("offer_group_id"))
        decision = decision_by_offer.get(offer_id)
        lightgbm_top = _safe_text(
            candidate.get("lightgbm_top_candidate")
        )
        final_selected = (
            _safe_text(decision.get("matched_master_sku"))
            if decision is not None
            else ""
        )
        suggested = final_selected or lightgbm_top
        candidate_id = _safe_text(candidate.get("master_itemcode"))
        is_suggested = bool(suggested and candidate_id == suggested)
        conflict_flags = [
            column
            for column in CONFLICT_COLUMNS
            if column in candidate
            and pd.notna(candidate[column])
            and bool(candidate[column])
        ]
        feature_snapshot = {
            column: _json_scalar(candidate.get(column))
            for column in MODEL_FEATURE_COLUMNS
        }
        prediction_records.append(
            {
                "offer_id": offer_id,
                "source_offer_id": _safe_text(
                    candidate.get("source_offer_id")
                ),
                "source_offer_text": _safe_text(
                    candidate.get("source_offer_text")
                ),
                "entity_id": _safe_text(candidate.get("entity_id")),
                "entity_index": candidate.get("entity_index"),
                "entity_count": candidate.get("entity_count"),
                "entity_text": _safe_text(candidate.get("entity_text")),
                "conjunction_type": _safe_text(
                    candidate.get("conjunction_type")
                ),
                "attribute_inheritance_flags": _safe_text(
                    candidate.get("attribute_inheritance_flags")
                ),
                "entity_parse_confidence": _safe_float(
                    candidate.get("entity_parse_confidence")
                ),
                "offer_description": _safe_text(
                    candidate.get("offer_text")
                ),
                "candidate_id": candidate_id,
                "candidate_description": _safe_text(
                    candidate.get("master_item_description")
                ),
                "candidate_rank": int(candidate.get("candidate_rank")),
                "lightgbm_probability": _safe_float(
                    candidate.get("calibrated_probability")
                ),
                "agreement_status": _safe_text(
                    candidate.get("agreement_status")
                ),
                "llm_decision": _safe_text(
                    candidate.get("llm_parsed_decision")
                ),
                "llm_confidence": _safe_float(
                    candidate.get("llm_confidence")
                ),
                "final_decision": (
                    _safe_text(decision.get("final_decision"))
                    if decision is not None and is_suggested
                    else "CANDIDATE_NOT_SELECTED"
                ),
                "decision_source": (
                    _safe_text(decision.get("decision_source"))
                    if decision is not None and is_suggested
                    else "CANDIDATE_RANKING_OBSERVATION"
                ),
                "conflict_flags": conflict_flags,
                "feature_snapshot": feature_snapshot,
            }
        )
    prediction_ids = store.add_predictions(
        str(result.run_id), prediction_records
    )
    for prediction_id, record in zip(
        prediction_ids, prediction_records, strict=True
    ):
        decision = str(record["final_decision"]).upper()
        if decision == "CANDIDATE_NOT_SELECTED":
            continue
        if decision == "LLM_ACCEPT":
            quality = LabelQuality.SILVER
            eligibility = "POLICY_QUALIFIED_REVIEW_REQUIRED_BEFORE_TRAINING"
            source_name = "STRUCTURED_LLM_REVIEW"
        elif decision == "AUTO_ACCEPT":
            quality = LabelQuality.PSEUDO
            eligibility = "NOT_TRAINING_ELIGIBLE"
            source_name = "MODEL_POLICY"
        else:
            quality = LabelQuality.REJECTED
            eligibility = "REJECTED"
            source_name = str(record["decision_source"])
        store.add_automated_label(
            prediction_id=prediction_id,
            source=source_name,
            proposed_label=decision,
            selected_candidate_id=str(record["candidate_id"]),
            confidence=record["llm_confidence"]
            if quality is LabelQuality.SILVER
            else record["lightgbm_probability"],
            label_quality=quality,
            eligibility_status=eligibility,
            rejection_reason=(
                None
                if quality in {LabelQuality.SILVER, LabelQuality.PSEUDO}
                else decision
            ),
        )
    if result.status.startswith("COMPLETED"):
        return store.create_review_session(
            str(result.run_id),
            threshold=config.ml.auto_accept_threshold,
            question_count=config.learning_store.questions_per_run,
        )
    return None


def observe_unified_result_non_blocking(
    result: Any,
    *,
    config: PipelineConfig,
    source_path: str | Path | None = None,
) -> str | None:
    """Keep learning-store failures nonfatal to production inference."""
    try:
        return observe_unified_result(
            result, config=config, source_path=source_path
        )
    except Exception:
        LOGGER.exception(
            "Learning-store observation failed; production output is unchanged"
        )
        return None
