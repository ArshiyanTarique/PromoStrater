"""Leakage-safe challenger fitting without registry activation."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from sku_mapping.config import PipelineConfig
from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.learning.store import LearningStore
from sku_mapping.ml.calibration import fit_probability_calibrator
from sku_mapping.ml.leakage import build_leakage_groups
from sku_mapping.ml.leakage_split import (
    LeakageSafeSplits,
    assert_zero_leakage,
    create_leakage_safe_splits,
)
from sku_mapping.ml.model_package import save_model_package, validate_model_package
from sku_mapping.ml.safety_thresholds import tune_shadow_thresholds
from sku_mapping.ml.shadow_trainer import ShadowTrainingConfig
from sku_mapping.retraining.artifacts import atomic_csv, atomic_json, sha256_file
from sku_mapping.retraining.snapshot import load_training_snapshot


@dataclass(frozen=True)
class ChallengerTrainingResult:
    model_id: str
    package_path: Path
    metadata_path: Path
    training_report_path: Path
    feature_importance_path: Path
    package_sha256: str
    dataset_id: str
    split_assignment_hash: str


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.loc[:, MODEL_FEATURE_COLUMNS].copy()
    for column in MODEL_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    if list(features.columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Challenger feature order differs from model schema")
    return features


def _fit_model(
    splits: LeakageSafeSplits,
    training_config: ShadowTrainingConfig,
) -> tuple[LGBMClassifier, list[dict[str, Any]], dict[str, Any]]:
    train_x = _feature_frame(splits.train)
    validation_x = _feature_frame(splits.validation)
    weights = pd.to_numeric(
        splits.train["training_sample_weight"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Challenger train split contains invalid sample weights")
    models: list[LGBMClassifier] = []
    results: list[dict[str, Any]] = []
    labels = splits.validation["pair_label"].to_numpy(dtype=int)
    for index, candidate in enumerate(training_config.hyperparameter_candidates):
        model = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=training_config.random_seed,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **candidate,
        )
        model.fit(
            train_x,
            splits.train["pair_label"],
            sample_weight=weights,
            eval_set=[(validation_x, splits.validation["pair_label"])],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    training_config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        probability = model.predict_proba(validation_x)[:, 1]
        results.append(
            {
                "candidate_index": index,
                "parameters": dict(candidate),
                "best_iteration": int(
                    model.best_iteration_ or candidate["n_estimators"]
                ),
                "validation_pr_auc": float(
                    average_precision_score(labels, probability)
                ),
                "validation_roc_auc": float(
                    roc_auc_score(labels, probability)
                ),
            }
        )
        models.append(model)
    best = max(
        range(len(results)),
        key=lambda index: (
            results[index]["validation_pr_auc"],
            results[index]["validation_roc_auc"],
            -index,
        ),
    )
    return models[best], results, dict(
        training_config.hyperparameter_candidates[best]
    )


def train_challenger(
    *,
    snapshot_manifest_path: str | Path,
    champion_model_id: str,
    config: PipelineConfig,
    store: LearningStore,
    training_config: ShadowTrainingConfig | None = None,
) -> ChallengerTrainingResult:
    """Train and package an inert challenger; never touch the active pointer."""
    if not str(champion_model_id).strip():
        raise ValueError("champion_model_id must be explicit")
    manifest, frame, _ = load_training_snapshot(snapshot_manifest_path)
    effective = training_config or ShadowTrainingConfig.from_pipeline_config(
        config
    )
    grouping = build_leakage_groups(frame)
    splits = create_leakage_safe_splits(grouping.frame, effective.split)
    repeated = create_leakage_safe_splits(grouping.frame, effective.split)
    if repeated.assignment_sha256 != splits.assignment_sha256:
        raise AssertionError("Challenger split is not reproducible")
    assert_zero_leakage(
        {
            "train": splits.train,
            "validation": splits.validation,
            "calibration": splits.calibration,
        }
    )

    model, candidates, selected = _fit_model(splits, effective)
    calibration_model, predictor, calibration_report, _ = (
        fit_probability_calibrator(
            model,
            splits.calibration,
            requested_method=effective.calibration_method,
            isotonic_min_rows=effective.isotonic_min_rows,
            isotonic_min_positive_rows=effective.isotonic_min_positive_rows,
            random_seed=effective.random_seed,
        )
    )
    calibration_probability = predictor.predict_calibrated_proba(
        _feature_frame(splits.calibration)
    )
    threshold_result = tune_shadow_thresholds(
        splits.calibration["pair_label"].to_numpy(dtype=int),
        calibration_probability,
        effective.threshold_policy,
    )

    timestamp = datetime.now(timezone.utc)
    timestamp_token = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    model_id = (
        f"alkabeer-sku-matcher-v4-challenger-{timestamp_token}-"
        f"{splits.assignment_sha256[:12]}"
    )
    package_filename = f"{model_id}.joblib"
    metadata_filename = f"{model_id}.json"
    directory = config.retraining.challenger_directory / model_id
    package_path = directory / package_filename
    metadata_path = directory / metadata_filename
    training_report_path = directory / "training_report.json"
    feature_importance_path = directory / "feature_importance.csv"

    metrics = {
        "model_selection": {
            "selection_split": "validation",
            "calibration_rows_used_for_model_selection": 0,
            "recent_gold_evaluation_rows_used": 0,
            "sealed_challenge_rows_used": 0,
            "candidate_results": candidates,
            "selected_parameters": selected,
        },
        "calibration": calibration_report,
        "split_audit": splits.audit,
        "snapshot_dataset_id": manifest["dataset_id"],
        "not_a_final_performance_estimate": True,
    }
    package: dict[str, Any] = {
        "package_schema_version": "3.0",
        "model": model,
        "calibration_model": calibration_model,
        "predictor": predictor,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "auto_match_threshold": threshold_result.auto_match_threshold,
        "manual_review_threshold": threshold_result.manual_review_threshold,
        "approved_auto_match_threshold": None,
        "auto_match_threshold_approved": False,
        "model_id": model_id,
        "package_version": f"4.0.0+{timestamp_token.lower()}",
        "model_version": "alkabeer_sku_matcher_v4_challenger",
        "parent_model": champion_model_id,
        "training_timestamp": timestamp.isoformat(),
        "training_dataset_hash": manifest["content_hash"],
        "processed_feature_table_hash": manifest["artifact_sha256"],
        "split_assignment_hash": splits.assignment_sha256,
        "metrics": metrics,
        "threshold_evidence": {
            "policy": threshold_result.policy,
            "selected_metrics": threshold_result.selected_metrics,
            "technical_evidence_requirements_met": (
                threshold_result.evidence_requirements_met
            ),
            "production_threshold_approved": False,
        },
        "calibration_method": calibration_report["method_selected"],
        "lightgbm_version": lgb.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "compatibility_policy": effective.compatibility_policy,
        "training_config": {
            "shadow_training": asdict(effective),
            "retraining": asdict(config.retraining),
            "sample_weight_column": "training_sample_weight",
            "pseudo_rows_used": 0,
        },
        "random_seed": effective.random_seed,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
    }
    validate_model_package(
        package,
        expected_training_dataset_hash=manifest["content_hash"],
        expected_processed_feature_table_hash=manifest["artifact_sha256"],
        expected_split_assignment_hash=splits.assignment_sha256,
    )
    save_model_package(package, package_path, metadata_path)
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "importance_gain": model.booster_.feature_importance(
                importance_type="gain"
            ),
            "importance_split": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("importance_gain", ascending=False, kind="stable")
    atomic_csv(importance, feature_importance_path)
    package_hash = sha256_file(package_path)
    report = {
        "model_id": model_id,
        "parent_champion_model_id": champion_model_id,
        "dataset_id": manifest["dataset_id"],
        "package_sha256": package_hash,
        "package_validation_passed": True,
        "leakage_checks_passed": True,
        "split_assignment_hash": splits.assignment_sha256,
        "counts_by_trust_level": manifest["counts_by_trust_level"],
        "sample_weight_policy": manifest["inclusion_policy"],
        "threshold_evidence": package["threshold_evidence"],
        "active_model_changed": False,
        "registration_status": "CHALLENGER_NOT_REGISTERED",
        "evaluation_status": "PENDING_CHAMPION_COMPARISON",
    }
    atomic_json(report, training_report_path)
    store.register_model_version(
        model_id=model_id,
        model_hash=package_hash,
        status="CHALLENGER_TRAINED",
        parent_model_id=champion_model_id,
        training_dataset_id=str(manifest["dataset_id"]),
        evaluation_summary=report,
        champion_status="CHALLENGER_NOT_ACTIVE",
        created_at=timestamp.isoformat(),
    )
    return ChallengerTrainingResult(
        model_id=model_id,
        package_path=package_path,
        metadata_path=metadata_path,
        training_report_path=training_report_path,
        feature_importance_path=feature_importance_path,
        package_sha256=package_hash,
        dataset_id=str(manifest["dataset_id"]),
        split_assignment_hash=splits.assignment_sha256,
    )
