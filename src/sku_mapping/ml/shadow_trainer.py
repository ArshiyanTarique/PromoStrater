"""Leakage-safe experimental v3 training for shadow-mode use only."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from sku_mapping.config import PipelineConfig
from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.ml.calibration import fit_probability_calibrator
from sku_mapping.ml.diagnostics import build_missingness_diagnostics
from sku_mapping.ml.leakage import build_leakage_groups
from sku_mapping.ml.leakage_split import (
    LeakageSafeSplits,
    LeakageSplitConfig,
    create_leakage_safe_splits,
)
from sku_mapping.ml.model_package import (
    save_model_package,
    update_model_registry,
    validate_model_package,
)
from sku_mapping.ml.provenance import build_provenance_weights
from sku_mapping.ml.safety_thresholds import (
    SafetyThresholdResult,
    ThresholdEvidencePolicy,
    tune_shadow_thresholds,
)
from sku_mapping.paths import PROJECT_ROOT
from sku_mapping.shadow.challenge import assert_not_sealed_challenge_input


@dataclass(frozen=True)
class ShadowTrainingConfig:
    """Complete pre-registered v3 shadow training policy."""

    random_seed: int
    split: LeakageSplitConfig
    calibration_method: str
    isotonic_min_rows: int
    isotonic_min_positive_rows: int
    threshold_policy: ThresholdEvidencePolicy
    provenance_weights: dict[str, float]
    compatibility_policy: dict[str, str]
    model_version: str = "alkabeer_sku_matcher_v3"
    package_schema_version: str = "3.0"
    early_stopping_rounds: int = 50
    hyperparameter_candidates: tuple[dict[str, Any], ...] = field(
        default_factory=lambda: (
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "min_child_samples": 30,
                "subsample": 0.9,
                "colsample_bytree": 1.0,
                "reg_lambda": 2.0,
            },
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_child_samples": 40,
                "subsample": 1.0,
                "colsample_bytree": 0.9,
                "reg_lambda": 5.0,
            },
        )
    )

    @classmethod
    def from_pipeline_config(cls, config: PipelineConfig) -> "ShadowTrainingConfig":
        policy = config.training
        return cls(
            random_seed=config.runtime.random_seed,
            split=LeakageSplitConfig(
                random_seed=config.runtime.random_seed,
                train_fraction=policy.train_fraction,
                validation_fraction=policy.validation_fraction,
                calibration_fraction=policy.calibration_fraction,
                candidate_splits=policy.split_search_candidates,
            ),
            calibration_method=policy.calibration_method,
            isotonic_min_rows=policy.isotonic_min_rows,
            isotonic_min_positive_rows=policy.isotonic_min_positive_rows,
            threshold_policy=ThresholdEvidencePolicy(
                target_auto_precision=policy.target_auto_precision,
                min_auto_match_rows=policy.min_auto_match_rows,
                max_auto_match_false_positives=policy.max_auto_match_false_positives,
                min_auto_precision_lower_bound=policy.min_auto_precision_lower_bound,
                precision_confidence_level=policy.precision_confidence_level,
                min_calibration_rows=policy.min_calibration_rows,
                min_calibration_positive_rows=policy.min_calibration_positive_rows,
                auto_threshold_min=policy.auto_threshold_min,
                auto_threshold_max=policy.auto_threshold_max,
                auto_threshold_steps=policy.auto_threshold_steps,
                manual_threshold_min=policy.manual_threshold_min,
                manual_threshold_max=policy.manual_threshold_max,
                manual_threshold_steps=policy.manual_threshold_steps,
            ),
            provenance_weights={
                str(key): float(value)
                for key, value in policy.provenance_weights.items()
            },
            compatibility_policy={
                "python_major_minor": policy.python_major_minor,
                "lightgbm_major_minor": policy.lightgbm_major_minor,
                "sklearn_major_minor": policy.sklearn_major_minor,
            },
        )


@dataclass
class ShadowTrainingResult:
    """Completed v3 shadow experiment and persisted artifacts."""

    package: dict[str, Any]
    splits: LeakageSafeSplits
    threshold_result: SafetyThresholdResult
    output_paths: dict[str, Path]
    technical_checks_passed: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_training_dataset_hash(
    manifest_path: Path,
    processed_hash: str,
) -> tuple[str, dict[str, Any]]:
    if not manifest_path.is_file():
        return processed_hash, {"source": "processed_feature_table_fallback"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_processed_hash = (
        manifest.get("output_hashes", {}).get("training_features_parquet")
    )
    if manifest_processed_hash and manifest_processed_hash != processed_hash:
        raise ValueError(
            "Processed feature-table hash differs from its generation manifest"
        )
    training_hash = manifest.get("input_hashes", {}).get("gold_pairs")
    if not isinstance(training_hash, str) or len(training_hash) != 64:
        raise ValueError("Training manifest lacks the gold-pair SHA-256")
    return training_hash, {
        "source": "training_feature_manifest",
        "manifest_path": str(manifest_path),
    }


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Training table is missing model features: {missing}")
    features = frame.loc[:, MODEL_FEATURE_COLUMNS].copy()
    for column in MODEL_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    if list(features.columns) != MODEL_FEATURE_COLUMNS:
        raise AssertionError("Model feature order changed")
    return features


def _fit_model_candidates(
    splits: LeakageSafeSplits,
    config: ShadowTrainingConfig,
) -> tuple[LGBMClassifier, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_features = _feature_frame(splits.train)
    validation_features = _feature_frame(splits.validation)
    train_weights, _, provenance_report = build_provenance_weights(
        splits.train, config.provenance_weights
    )
    validation_labels = splits.validation["pair_label"].to_numpy(dtype=int)
    models: list[LGBMClassifier] = []
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(config.hyperparameter_candidates):
        parameters = {
            "objective": "binary",
            "class_weight": "balanced",
            "random_state": config.random_seed,
            "n_jobs": -1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            **candidate,
        }
        model = LGBMClassifier(**parameters)
        model.fit(
            train_features,
            splits.train["pair_label"],
            sample_weight=train_weights,
            eval_set=[(validation_features, splits.validation["pair_label"])],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        probabilities = model.predict_proba(validation_features)[:, 1]
        results.append(
            {
                "candidate_index": index,
                "parameters": candidate,
                "best_iteration": int(
                    model.best_iteration_ or candidate["n_estimators"]
                ),
                "validation_pr_auc": float(
                    average_precision_score(validation_labels, probabilities)
                ),
                "validation_roc_auc": float(
                    roc_auc_score(validation_labels, probabilities)
                ),
            }
        )
        models.append(model)
    best_index = max(
        range(len(results)),
        key=lambda index: (
            results[index]["validation_pr_auc"],
            results[index]["validation_roc_auc"],
            -index,
        ),
    )
    return (
        models[best_index],
        results,
        dict(config.hyperparameter_candidates[best_index]),
        provenance_report,
    )


def _legacy_v2_registry_entries(registry_directory: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(registry_directory.glob("alkabeer_sku_matcher_v2_*.joblib")):
        package_hash = _sha256_file(path)
        package = joblib.load(path)
        timestamp = str(package.get("training_timestamp", ""))
        timestamp_token = re_safe_token(timestamp)
        entries.append(
            {
                "package_filename": path.name,
                "model_id": f"legacy-v2-{package_hash[:20]}",
                "model_version": str(package.get("model_version", "v2")),
                "package_version": f"legacy-v2+{timestamp_token}-{package_hash[:8]}",
                "creation_timestamp": timestamp,
                "training_dataset_hash": str(
                    package.get("training_dataset_hash", "")
                ),
                "feature_generator_version": str(
                    package.get("feature_generator_version", "unknown")
                ),
                "deployment_status": "SHADOW_MODE_ONLY",
                "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
                "automatic_production_matching_approved": False,
                "notes": (
                    "Frozen v2 package; adversarial review prohibits automatic "
                    "production matching. Existing package bytes remain unchanged."
                ),
                "parent_model": None,
            }
        )
    return entries


def re_safe_token(value: str) -> str:
    return "".join(character for character in value if character.isalnum()) or "unknown"


def _template_audit(grouping_audit: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    relevant = frame[frame["provenance_category"].isin({"synthetic", "rule_generated"})]
    examples = (
        relevant.groupby(
            ["template_group_id", "normalized_offer_template"], dropna=False
        )
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "template_group_id"], ascending=[False, True])
        .head(25)
        .to_dict(orient="records")
    )
    return {
        "deterministic": True,
        "normalization_rules": grouping_audit["normalization_rules"],
        "synthetic_or_rule_rows": int(len(relevant)),
        "template_group_count": int(relevant["template_group_id"].nunique()),
        "largest_templates": examples,
        "real_row_policy": (
            "stable record/offer/master/text identity; real rows are connected "
            "through offer_group_id and feature hash, not broad templates"
        ),
    }


def _phase_report(
    package: dict[str, Any],
    splits: LeakageSafeSplits,
    threshold_result: SafetyThresholdResult,
    output_paths: dict[str, Path],
) -> str:
    split_lines = "\n".join(
        f"- {name}: {details['rows']} rows, {details['positive_rows']} positive, "
        f"{details['leakage_groups']} leakage groups"
        for name, details in splits.audit["splits"].items()
    )
    return f"""# Phase 5C Completion Report

## Status

- Experimental package: `{output_paths['model'].name}`
- Deployment status: `SHADOW_MODE_ONLY`
- Automatic production matching approved: **false**
- Final challenge-set claim: **none**
- Production inference behavior changed: **no**
- Existing v2 packages: **preserved byte-for-byte and registered shadow-only**

## Leakage-safe splits

{split_lines}

All offer-group, relevant template-group, exact feature-hash, and transitive
leakage-group overlap checks passed with zero overlap. Every eligible row was
assigned exactly once. Assignment SHA-256:
`{splits.assignment_sha256}`.

## Model selection and calibration

Model fitting and early-stopping selection used train/validation only.
Probability calibration and threshold evidence used calibration only.
Calibration method: `{package['calibration_method']}`.

The shadow AUTO_MATCH candidate threshold is
`{threshold_result.auto_match_threshold:.6f}` and the manual-review threshold is
`{threshold_result.manual_review_threshold:.6f}`. Technical threshold evidence
requirements met: `{str(threshold_result.evidence_requirements_met).lower()}`.
No AUTO_MATCH threshold is approved for production.

The selected calibration candidate accepted
`{threshold_result.selected_metrics['auto_match_rows']}` rows with
`{threshold_result.selected_metrics['auto_match_false_positives']}` observed
false positives and precision
`{threshold_result.selected_metrics['auto_match_precision']:.6f}`. Its
one-sided precision lower bound was
`{threshold_result.selected_metrics['auto_match_precision_lower_bound']:.6f}`;
the configured evidence floor was
`{threshold_result.policy['min_auto_precision_lower_bound']:.6f}` and the
minimum accepted-row count was
`{threshold_result.policy['min_auto_match_rows']}`.

## Unresolved risks

- No sealed human-reviewed production challenge set exists.
- Synthetic/rule provenance remains the majority of available labelled rows.
- Calibration and threshold evidence are development evidence, not final
  performance estimates.
- `400+60 G X 20 Pkts` remains an unresolved business-rule interpretation.
- Business features with zero importance require more authoritative labels.
"""


def run_shadow_training_pipeline(
    feature_path: str | Path,
    *,
    config: ShadowTrainingConfig,
    processed_dir: str | Path = PROJECT_ROOT / "data" / "processed",
    model_registry_dir: str | Path = PROJECT_ROOT / "models" / "registry",
    metadata_dir: str | Path = PROJECT_ROOT / "models" / "metadata",
    registry_path: str | Path = PROJECT_ROOT / "models" / "model_registry.json",
    reports_dir: str | Path = PROJECT_ROOT / "reports",
    manifest_path: str | Path = (
        PROJECT_ROOT / "data" / "processed" / "training_feature_manifest.json"
    ),
) -> ShadowTrainingResult:
    """Run Phase 5C without a test/challenge set or production integration."""
    source_path = Path(feature_path)
    assert_not_sealed_challenge_input(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Processed feature table not found: {source_path}")
    registry_directory = Path(model_registry_dir)
    update_model_registry(
        Path(registry_path),
        _legacy_v2_registry_entries(registry_directory),
    )
    processed_hash = _sha256_file(source_path)
    training_hash, hash_source = _load_training_dataset_hash(
        Path(manifest_path), processed_hash
    )
    frame = pd.read_parquet(source_path)
    if len(frame) == 0:
        raise ValueError("Processed feature table is empty")
    labels = pd.to_numeric(frame["pair_label"], errors="coerce")
    if labels.isna().any() or set(labels.astype(int).unique()) != {0, 1}:
        raise ValueError("Processed feature table must contain both binary labels")
    frame = frame.copy()
    frame["pair_label"] = labels.astype(int)
    _feature_frame(frame)

    grouping = build_leakage_groups(frame)
    splits = create_leakage_safe_splits(grouping.frame, config.split)
    regenerated = create_leakage_safe_splits(grouping.frame, config.split)
    if regenerated.assignment_sha256 != splits.assignment_sha256:
        raise AssertionError("Leakage-safe split regeneration is not deterministic")

    model, candidates, selected_parameters, provenance_report = _fit_model_candidates(
        splits, config
    )
    validation_features = _feature_frame(splits.validation)
    validation_raw_scores = np.asarray(
        model.predict(
            validation_features,
            raw_score=True,
            num_iteration=model.best_iteration_,
        ),
        dtype=float,
    )
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    validation_predictions = splits.validation.copy()
    validation_predictions["raw_model_score"] = validation_raw_scores
    validation_predictions["raw_model_probability"] = validation_probabilities
    model_selection_report = {
        "selection_split": "validation",
        "calibration_rows_used_for_model_selection": 0,
        "challenge_rows_used": 0,
        "candidate_results": candidates,
        "selected_parameters": selected_parameters,
        "selected_best_iteration": int(model.best_iteration_),
        "selected_validation_pr_auc": float(
            average_precision_score(
                splits.validation["pair_label"], validation_probabilities
            )
        ),
        "selected_validation_roc_auc": float(
            roc_auc_score(
                splits.validation["pair_label"], validation_probabilities
            )
        ),
        "not_a_final_performance_estimate": True,
    }

    calibration_model, predictor, calibration_report, calibration_predictions = (
        fit_probability_calibrator(
            model,
            splits.calibration,
            requested_method=config.calibration_method,
            isotonic_min_rows=config.isotonic_min_rows,
            isotonic_min_positive_rows=config.isotonic_min_positive_rows,
            random_seed=config.random_seed,
        )
    )
    threshold_result = tune_shadow_thresholds(
        splits.calibration["pair_label"].to_numpy(dtype=int),
        calibration_predictions["calibrated_probability"].to_numpy(dtype=float),
        config.threshold_policy,
    )
    calibration_predictions["shadow_decision"] = np.where(
        calibration_predictions["calibrated_probability"]
        >= threshold_result.auto_match_threshold,
        "AUTO_MATCH",
        np.where(
            calibration_predictions["calibrated_probability"]
            >= threshold_result.manual_review_threshold,
            "MANUAL_REVIEW",
            "NO_MATCH",
        ),
    )

    gain_values = model.booster_.feature_importance(importance_type="gain")
    split_values = model.booster_.feature_importance(importance_type="split")
    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "importance_gain": gain_values,
            "importance_split": split_values,
        }
    ).sort_values("importance_gain", ascending=False, kind="stable")
    missingness_report = build_missingness_diagnostics(
        {
            "train": splits.train,
            "validation": splits.validation,
            "calibration": splits.calibration,
        },
        predictor=predictor,
        auto_threshold=threshold_result.auto_match_threshold,
        manual_threshold=threshold_result.manual_review_threshold,
        feature_importance_gain=dict(zip(MODEL_FEATURE_COLUMNS, gain_values)),
    )

    timestamp = datetime.now(timezone.utc)
    timestamp_iso = timestamp.isoformat()
    timestamp_token = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    model_id = (
        f"alkabeer-sku-matcher-v3-{timestamp_token}-"
        f"{splits.assignment_sha256[:12]}"
    )
    package_version = f"3.0.0+{timestamp_token.lower()}"
    parent_model = "alkabeer_sku_matcher_v2_20260729T054028473791Z.joblib"
    package: dict[str, Any] = {
        "package_schema_version": config.package_schema_version,
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
        "package_version": package_version,
        "model_version": config.model_version,
        "parent_model": parent_model,
        "training_timestamp": timestamp_iso,
        "training_dataset_hash": training_hash,
        "processed_feature_table_hash": processed_hash,
        "split_assignment_hash": splits.assignment_sha256,
        "metrics": {
            "model_selection": model_selection_report,
            "calibration": calibration_report,
            "no_final_challenge_metrics": True,
        },
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
        "compatibility_policy": config.compatibility_policy,
        "training_config": asdict(config),
        "random_seed": config.random_seed,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
    }
    validate_model_package(
        package,
        expected_training_dataset_hash=training_hash,
        expected_processed_feature_table_hash=processed_hash,
        expected_split_assignment_hash=splits.assignment_sha256,
    )

    processed = Path(processed_dir)
    reports = Path(reports_dir)
    model_path = registry_directory / f"{config.model_version}_{timestamp_token}.joblib"
    metadata_path = Path(metadata_dir) / f"{config.model_version}_{timestamp_token}.json"
    output_paths = {
        "model": model_path,
        "metadata": metadata_path,
        "registry": Path(registry_path),
        "split_assignments_parquet": processed
        / "leakage_safe_split_assignments.parquet",
        "split_assignments_csv": processed / "leakage_safe_split_assignments.csv",
        "train_split": processed / "shadow_train_split.parquet",
        "validation_split": processed / "shadow_validation_split.parquet",
        "calibration_split": processed / "shadow_calibration_split.parquet",
        "leakage_group_audit": reports / "leakage_group_audit.json",
        "template_normalization_audit": reports
        / "template_normalization_audit.json",
        "model_selection_report": reports / "model_selection_report.json",
        "calibration_report": reports / "calibration_report.json",
        "threshold_evidence_csv": reports / "threshold_evidence_report.csv",
        "threshold_evidence_json": reports / "threshold_evidence_report.json",
        "missingness_report": reports / "missingness_drift_report.json",
        "provenance_report": reports / "provenance_weighting_report.json",
        "package_validation_report": reports / "package_validation_report.json",
        "validation_predictions": reports / "shadow_validation_predictions.csv",
        "calibration_predictions": reports / "shadow_calibration_predictions.csv",
        "feature_importance": reports / "shadow_feature_importance.csv",
        "completion_report": reports / "phase_5c_completion.md",
    }

    _atomic_parquet(splits.assignments, output_paths["split_assignments_parquet"])
    _atomic_csv(splits.assignments, output_paths["split_assignments_csv"])
    _atomic_parquet(splits.train, output_paths["train_split"])
    _atomic_parquet(splits.validation, output_paths["validation_split"])
    _atomic_parquet(splits.calibration, output_paths["calibration_split"])
    _atomic_json(
        {
            "grouping": grouping.audit,
            "splitting": splits.audit,
            "technical_checks_passed": True,
        },
        output_paths["leakage_group_audit"],
    )
    _atomic_json(
        _template_audit(grouping.audit, grouping.frame),
        output_paths["template_normalization_audit"],
    )
    _atomic_json(model_selection_report, output_paths["model_selection_report"])
    _atomic_json(calibration_report, output_paths["calibration_report"])
    _atomic_csv(threshold_result.analysis, output_paths["threshold_evidence_csv"])
    _atomic_json(
        {
            "policy": threshold_result.policy,
            "selected_metrics": threshold_result.selected_metrics,
            "evidence_requirements_met": threshold_result.evidence_requirements_met,
            "approved_auto_match_threshold": None,
            "shadow_auto_match_threshold": threshold_result.auto_match_threshold,
            "manual_review_threshold": threshold_result.manual_review_threshold,
            "automatic_production_matching_approved": False,
        },
        output_paths["threshold_evidence_json"],
    )
    _atomic_json(missingness_report, output_paths["missingness_report"])
    _atomic_json(provenance_report, output_paths["provenance_report"])
    _atomic_csv(validation_predictions, output_paths["validation_predictions"])
    _atomic_csv(calibration_predictions, output_paths["calibration_predictions"])
    _atomic_csv(feature_importance, output_paths["feature_importance"])

    save_model_package(package, model_path, metadata_path)
    reloaded = joblib.load(model_path)
    validate_model_package(
        reloaded,
        expected_training_dataset_hash=training_hash,
        expected_processed_feature_table_hash=processed_hash,
        expected_split_assignment_hash=splits.assignment_sha256,
    )
    package_report = {
        "validation_passed": True,
        "package_filename": model_path.name,
        "model_id": model_id,
        "package_version": package_version,
        "deployment_status": "SHADOW_MODE_ONLY",
        "automatic_production_matching_approved": False,
        "validated_hashes": {
            "training_dataset_sha256": training_hash,
            "processed_feature_table_sha256": processed_hash,
            "split_assignment_sha256": splits.assignment_sha256,
        },
        "hash_source": hash_source,
        "runtime_compatibility_enforced": config.compatibility_policy,
    }
    _atomic_json(package_report, output_paths["package_validation_report"])

    v3_registry_entry = {
        "package_filename": model_path.name,
        "model_id": model_id,
        "model_version": config.model_version,
        "package_version": package_version,
        "creation_timestamp": timestamp_iso,
        "training_dataset_hash": training_hash,
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
        "notes": (
            "Experimental leakage-safe calibrated v3 package. No sealed "
            "human-reviewed final challenge set; shadow mode only."
        ),
        "parent_model": parent_model,
    }
    update_model_registry(
        output_paths["registry"],
        [v3_registry_entry],
    )

    completion_text = _phase_report(
        package, splits, threshold_result, output_paths
    )
    output_paths["completion_report"].parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_paths["completion_report"].parent,
        prefix=f".{output_paths['completion_report'].name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(completion_text)
        os.replace(temporary, output_paths["completion_report"])
    finally:
        temporary.unlink(missing_ok=True)

    return ShadowTrainingResult(
        package=package,
        splits=splits,
        threshold_result=threshold_result,
        output_paths=output_paths,
        technical_checks_passed=True,
    )
