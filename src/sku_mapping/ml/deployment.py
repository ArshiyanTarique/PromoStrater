"""Explicit, auditable, fail-safe ML deployment modes.

Assisted mode may alter only the candidate acceptance decision. It never
trains, updates, or persists model objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sku_mapping.config import PipelineConfig
from sku_mapping.constants import (
    FEATURE_GENERATOR_VERSION,
    MODEL_FEATURE_COLUMNS,
    MLDeploymentMode,
    MatchDecision,
)
from sku_mapping.data.preprocessing import preprocess_product_master
from sku_mapping.features.feature_generator import build_feature_vector
from sku_mapping.shadow.pipeline import (
    ShadowRunResult,
    run_shadow_observation_non_blocking,
)
from sku_mapping.shadow.predictor import (
    RegisteredShadowPackage,
    load_registered_shadow_package,
)

LOGGER = logging.getLogger(__name__)

THRESHOLD_SOURCE = "user_configured"
PRODUCTION_THRESHOLD_APPROVED = False

CONFLICT_NAMES = (
    "protein_conflict",
    "mixed_protein_ambiguity",
    "strong_family_conflict",
    "strong_size_weight_conflict",
    "strong_pack_format_conflict",
    "feature_generation_failure",
    "missing_master",
    "unregistered_or_invalid_package",
    "prediction_failure",
)


@dataclass(frozen=True)
class AssistedRunResult:
    """Assisted decisions and their immutable audit records."""

    status: str
    rows: pd.DataFrame
    decisions: pd.DataFrame
    run_id: str | None
    model_id: str | None
    model_package_sha256: str | None
    output_paths: dict[str, Path]
    error: str | None = None


def _prepared_master(master: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Itemcode",
        "Itemname",
        "master_measures_detailed",
    }
    return (
        master.copy(deep=True)
        if required.issubset(master.columns)
        else preprocess_product_master(master)
    )


def _identifier(row: pd.Series, position: int) -> str:
    for column in ("offer_group_id", "offerid", "record_id"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    payload = "|".join(
        str(row.get(column, ""))
        for column in (
            "match_text",
            "Offer Name",
            "Product",
            "Variant",
            "Base Packsize",
            "suggested_itemcode",
        )
    )
    return "assisted-offer-" + hashlib.sha256(
        f"{payload}|{position}".encode("utf-8")
    ).hexdigest()


def _empty_conflicts() -> dict[str, bool]:
    return {name: False for name in CONFLICT_NAMES}


def _has_measurements(row: pd.Series, key: str) -> bool:
    value = row.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def _feature_conflicts(
    feature_row: dict[str, Any],
    offer_row: pd.Series,
    master_row: pd.Series,
) -> dict[str, bool]:
    conflicts = _empty_conflicts()
    conflicts["protein_conflict"] = feature_row["protein_match"] == 0
    conflicts["mixed_protein_ambiguity"] = (
        feature_row["is_mixed_protein_offer"] == 1
    )
    conflicts["strong_family_conflict"] = feature_row["family_match"] == 0
    conflicts["strong_size_weight_conflict"] = bool(
        feature_row["size_match"] == 0
        and _has_measurements(offer_row, "offer_measures_detailed")
        and _has_measurements(master_row, "master_measures_detailed")
    )
    conflicts["strong_pack_format_conflict"] = (
        feature_row["pack_format_match"] == 0
    )
    return conflicts


def _record(
    *,
    run_id: str,
    timestamp: str,
    row: pd.Series,
    position: int,
    model_id: str | None,
    package_sha256: str | None,
    threshold: float,
    decision: MatchDecision,
    reason: str,
    override_applied: bool,
    conflicts: dict[str, bool],
    raw_probability: float | None = None,
    calibrated_probability: float | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "ml_mode": MLDeploymentMode.ASSISTED.value,
        "offer_identifier": _identifier(row, position),
        "candidate_identifier": str(row.get("suggested_itemcode", "")),
        "candidate_rank": 1,
        "model_id": model_id,
        "model_package_sha256": package_sha256,
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "auto_accept_threshold": float(threshold),
        "threshold_source": THRESHOLD_SOURCE,
        "production_threshold_approved": PRODUCTION_THRESHOLD_APPROVED,
        "decision": decision.value,
        "decision_reason": reason,
        "safety_override_applied": bool(override_applied),
        "conflict_flags": json.dumps(conflicts, sort_keys=True),
        **conflicts,
    }


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _attach_records(
    rows: pd.DataFrame,
    records: pd.DataFrame,
    *,
    successful_predictions: bool,
) -> pd.DataFrame:
    result = rows.copy(deep=True)
    assisted_columns = {
        "assisted_mode": records["ml_mode"].to_numpy(),
        "assisted_run_id": records["run_id"].to_numpy(),
        "assisted_model_id": records["model_id"].to_numpy(),
        "assisted_model_package_sha256": records[
            "model_package_sha256"
        ].to_numpy(),
        "assisted_raw_probability": records["raw_probability"].to_numpy(),
        "assisted_calibrated_probability": records[
            "calibrated_probability"
        ].to_numpy(),
        "assisted_auto_accept_threshold": records[
            "auto_accept_threshold"
        ].to_numpy(),
        "assisted_threshold_source": records["threshold_source"].to_numpy(),
        "assisted_production_threshold_approved": records[
            "production_threshold_approved"
        ].to_numpy(),
        "assisted_decision": records["decision"].to_numpy(),
        "assisted_decision_reason": records["decision_reason"].to_numpy(),
        "assisted_safety_override_applied": records[
            "safety_override_applied"
        ].to_numpy(),
        "assisted_conflict_flags": records["conflict_flags"].to_numpy(),
    }
    for column, values in assisted_columns.items():
        result[column] = values

    if successful_predictions:
        accepted = result["assisted_decision"].eq(
            MatchDecision.AUTO_ACCEPT.value
        )
        evaluable = result["assisted_decision"].isin(
            [
                MatchDecision.AUTO_ACCEPT.value,
                MatchDecision.MANUAL_REVIEW.value,
            ]
        )
        result.loc[evaluable, "ml_decision"] = np.where(
            accepted[evaluable], "AUTO_MATCH", "MANUAL_REVIEW"
        )
        result.loc[evaluable, "ml_probability"] = result.loc[
            evaluable, "assisted_calibrated_probability"
        ]
    return result


def _load_package(
    config: PipelineConfig,
    model_directory: str | Path | None,
) -> RegisteredShadowPackage:
    shadow = config.shadow_mode
    effective_model_directory = (
        Path(model_directory)
        if model_directory is not None
        else shadow.registry_path.parent / "registry"
    )
    return load_registered_shadow_package(
        registry_path=shadow.registry_path,
        model_directory=effective_model_directory,
        model_id=config.ml.model_id,
        require_package_status=shadow.require_package_status,
    )


def apply_assisted_decisions(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    config: PipelineConfig,
    model_directory: str | Path | None = None,
    run_id: str | None = None,
    persist_records: bool = True,
) -> AssistedRunResult:
    """Apply the configured assisted policy without ever fitting the model."""
    if config.ml.mode is not MLDeploymentMode.ASSISTED:
        return AssistedRunResult(
            status="DISABLED",
            rows=production_rows,
            decisions=pd.DataFrame(),
            run_id=None,
            model_id=None,
            model_package_sha256=None,
            output_paths={},
        )

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    effective_run_id = run_id or (
        f"assisted-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    threshold = config.ml.auto_accept_threshold
    master = _prepared_master(product_master)
    master_lookup = {
        str(row["Itemcode"]).strip(): row for _, row in master.iterrows()
    }

    registered: RegisteredShadowPackage | None = None
    load_error: Exception | None = None
    try:
        registered = _load_package(config, model_directory)
    except Exception as error:
        load_error = error
        LOGGER.exception(
            "Assisted model failed validation; retaining existing decisions"
        )
    if registered is not None and registered.package.get(
        "requires_group_features"
    ):
        # Ranked packages score a whole candidate shortlist together; this
        # single-candidate path cannot featurise one in isolation. Fail
        # closed to the existing-decision fallback instead of emitting
        # scores computed outside the model's training regime. The unified
        # inference pipeline is the supported path for ranked packages.
        LOGGER.error(
            "Registered package %s requires group features; "
            "apply_assisted_decisions scores single candidates only, so "
            "rows keep their existing decisions",
            registered.package.get("model_id"),
        )
        load_error = ValueError(
            "Ranked package requires group scoring; use the unified pipeline"
        )
        registered = None

    records: list[dict[str, Any]] = []
    successful_predictions = registered is not None
    for position, (_, row) in enumerate(production_rows.iterrows()):
        conflicts = _empty_conflicts()
        itemcode = str(row.get("suggested_itemcode", "")).strip()
        if itemcode in {"", "NO_MATCH", "REVIEW_REQUIRED", "nan", "None"}:
            records.append(
                _record(
                    run_id=effective_run_id,
                    timestamp=timestamp,
                    row=row,
                    position=position,
                    model_id=config.ml.model_id,
                    package_sha256=(
                        registered.package_sha256 if registered else None
                    ),
                    threshold=threshold,
                    decision=MatchDecision.NO_CANDIDATE,
                    reason="no_candidate_available",
                    override_applied=False,
                    conflicts=conflicts,
                )
            )
            continue
        if itemcode not in master_lookup:
            conflicts["missing_master"] = True
            records.append(
                _record(
                    run_id=effective_run_id,
                    timestamp=timestamp,
                    row=row,
                    position=position,
                    model_id=config.ml.model_id,
                    package_sha256=(
                        registered.package_sha256 if registered else None
                    ),
                    threshold=threshold,
                    decision=MatchDecision.MASTER_SKU_NOT_FOUND,
                    reason="candidate_missing_from_product_master",
                    override_applied=True,
                    conflicts=conflicts,
                )
            )
            continue
        if registered is None:
            conflicts["unregistered_or_invalid_package"] = True
            records.append(
                _record(
                    run_id=effective_run_id,
                    timestamp=timestamp,
                    row=row,
                    position=position,
                    model_id=config.ml.model_id,
                    package_sha256=None,
                    threshold=threshold,
                    decision=MatchDecision.MODEL_ERROR,
                    reason="model_load_or_compatibility_failure_safe_fallback",
                    override_applied=True,
                    conflicts=conflicts,
                )
            )
            continue

        master_row = master_lookup[itemcode]
        try:
            feature_row = build_feature_vector(row, master_row)
            features = pd.DataFrame(
                [[feature_row[name] for name in MODEL_FEATURE_COLUMNS]],
                columns=MODEL_FEATURE_COLUMNS,
            )
        except Exception:
            conflicts["feature_generation_failure"] = True
            records.append(
                _record(
                    run_id=effective_run_id,
                    timestamp=timestamp,
                    row=row,
                    position=position,
                    model_id=str(registered.package["model_id"]),
                    package_sha256=registered.package_sha256,
                    threshold=threshold,
                    decision=MatchDecision.MANUAL_REVIEW,
                    reason="feature_generation_failure",
                    override_applied=True,
                    conflicts=conflicts,
                )
            )
            continue

        conflicts.update(_feature_conflicts(feature_row, row, master_row))
        try:
            raw_probability = float(
                registered.package["model"].predict_proba(features)[0, 1]
            )
            calibrated_probability = float(
                registered.package["predictor"]
                .predict_calibrated_proba(features)[0]
            )
            if not (
                np.isfinite(raw_probability)
                and 0 <= raw_probability <= 1
                and np.isfinite(calibrated_probability)
                and 0 <= calibrated_probability <= 1
            ):
                raise ValueError("Model probabilities must be within [0, 1]")
        except Exception:
            LOGGER.exception("Assisted prediction failed; routing to review")
            conflicts["prediction_failure"] = True
            records.append(
                _record(
                    run_id=effective_run_id,
                    timestamp=timestamp,
                    row=row,
                    position=position,
                    model_id=str(registered.package["model_id"]),
                    package_sha256=registered.package_sha256,
                    threshold=threshold,
                    decision=MatchDecision.MODEL_ERROR,
                    reason="prediction_failure_safe_existing_fallback",
                    override_applied=True,
                    conflicts=conflicts,
                )
            )
            continue

        blocking = [
            name for name, active in conflicts.items() if active
        ]
        if (
            calibrated_probability >= threshold
            and blocking
            and config.ml.apply_safety_overrides
        ):
            decision = MatchDecision.MANUAL_REVIEW
            reason = "safety_override:" + ",".join(blocking)
            override = True
        elif calibrated_probability >= threshold:
            decision = MatchDecision.AUTO_ACCEPT
            reason = "calibrated_probability_meets_user_configured_threshold"
            override = False
        else:
            decision = MatchDecision.MANUAL_REVIEW
            reason = "below_user_configured_auto_accept_threshold"
            override = False
        records.append(
            _record(
                run_id=effective_run_id,
                timestamp=timestamp,
                row=row,
                position=position,
                model_id=str(registered.package["model_id"]),
                package_sha256=registered.package_sha256,
                threshold=threshold,
                decision=decision,
                reason=reason,
                override_applied=override,
                conflicts=conflicts,
                raw_probability=raw_probability,
                calibrated_probability=calibrated_probability,
            )
        )

    decisions = pd.DataFrame(records)
    result_rows = _attach_records(
        production_rows,
        decisions,
        successful_predictions=successful_predictions,
    )
    output_paths: dict[str, Path] = {}
    if persist_records:
        destination = (
            config.output.output_dir
            / "assisted"
            / effective_run_id
            / "assisted_decisions.csv"
        )
        try:
            _atomic_csv(decisions, destination)
            output_paths["assisted_decisions"] = destination
        except Exception:
            LOGGER.exception(
                "Could not persist assisted audit records; pipeline continues"
            )
    return AssistedRunResult(
        status=(
            "COMPLETED_ASSISTED"
            if registered is not None
            else "MODEL_ERROR_SAFE_FALLBACK"
        ),
        rows=result_rows,
        decisions=decisions,
        run_id=effective_run_id,
        model_id=(
            str(registered.package["model_id"])
            if registered is not None
            else config.ml.model_id
        ),
        model_package_sha256=(
            registered.package_sha256 if registered is not None else None
        ),
        output_paths=output_paths,
        error=str(load_error) if load_error is not None else None,
    )


def run_assisted_monitoring_non_blocking(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    config: PipelineConfig,
    model_directory: str | Path | None = None,
) -> ShadowRunResult:
    """Record assisted outcomes with the unchanged observational sidecar."""
    if (
        config.ml.mode is not MLDeploymentMode.ASSISTED
        or not config.ml.continue_shadow_monitoring
    ):
        return ShadowRunResult(
            status="DISABLED",
            shadow_run_id=None,
            prediction_rows=0,
            offer_groups=0,
            failed_shadow_predictions=0,
            output_paths={},
        )
    shadow = replace(
        config.shadow_mode,
        enabled=True,
        model_id=config.ml.model_id,
        package_reference=None,
    )
    return run_shadow_observation_non_blocking(
        production_rows,
        product_master,
        config=replace(config, shadow_mode=shadow),
        model_directory=model_directory,
    )

