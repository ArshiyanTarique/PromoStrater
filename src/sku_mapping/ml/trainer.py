"""Reproducible, group-isolated LightGBM training orchestration."""

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

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.ml.evaluator import evaluate_binary_classifier
from sku_mapping.ml.model_package import (
    save_model_package,
    validate_feature_frame_columns,
)
from sku_mapping.ml.threshold_tuning import ThresholdTuningResult, tune_thresholds
from sku_mapping.paths import PROJECT_ROOT
from sku_mapping.shadow.challenge import assert_not_sealed_challenge_input

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class TrainingConfig:
    """Deterministic model, split, and threshold-tuning settings."""

    random_seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    early_stopping_rounds: int = 50
    target_auto_precision: float = 0.99
    model_version: str = "alkabeer_sku_matcher_v2"
    calibration_bins: int = 10
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

    def __post_init__(self) -> None:
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, int
        ):
            raise ValueError("random_seed must be an integer")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be below 1")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if not 0 <= self.target_auto_precision <= 1:
            raise ValueError("target_auto_precision must be between 0 and 1")
        if not self.hyperparameter_candidates:
            raise ValueError("At least one hyperparameter candidate is required")


@dataclass(frozen=True)
class DatasetSplits:
    """Train, validation, and untouched test tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    method: str


@dataclass
class ModelTrainingResult:
    """All in-memory and persisted outputs of a completed training run."""

    model: LGBMClassifier
    splits: DatasetSplits
    threshold_result: ThresholdTuningResult
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    hyperparameter_results: list[dict[str, Any]]
    selected_hyperparameters: dict[str, Any]
    test_predictions: pd.DataFrame
    package: dict[str, Any]
    output_paths: dict[str, Path] = field(default_factory=dict)


def _validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["offer_group_id", "pair_label", *MODEL_FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training feature table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Training feature table is empty")
    if frame["offer_group_id"].isna().any() or (
        frame["offer_group_id"].astype("string").str.strip() == ""
    ).any():
        raise ValueError("offer_group_id must be populated for every row")
    labels = pd.to_numeric(frame["pair_label"], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise ValueError("pair_label must contain only binary labels 0 and 1")
    validated = frame.copy()
    validated["pair_label"] = labels.astype(int)
    for column in MODEL_FEATURE_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    all_missing = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if validated[column].isna().all()
    ]
    if all_missing:
        raise ValueError(f"Feature columns contain no numeric values: {all_missing}")
    return validated


def assert_no_group_leakage(splits: DatasetSplits) -> None:
    """Raise if an offer group occurs in more than one split."""
    group_sets = {
        name: set(getattr(splits, name)["offer_group_id"].astype(str))
        for name in SPLIT_NAMES
    }
    overlaps = {
        f"{left}_{right}": sorted(group_sets[left] & group_sets[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
        if group_sets[left] & group_sets[right]
    }
    if overlaps:
        raise ValueError(f"offer_group_id leakage detected: {overlaps}")


def _split_has_both_classes(frame: pd.DataFrame) -> bool:
    return set(frame["pair_label"].unique()) == {0, 1}


def _recommended_splits(frame: pd.DataFrame) -> DatasetSplits | None:
    if "recommended_split" not in frame.columns:
        return None
    values = frame["recommended_split"].astype("string").str.strip().str.lower()
    if values.isna().any() or not values.isin(SPLIT_NAMES).all():
        return None
    candidate = DatasetSplits(
        train=frame.loc[values == "train"].copy(),
        validation=frame.loc[values == "validation"].copy(),
        test=frame.loc[values == "test"].copy(),
        method="recommended_split",
    )
    if any(getattr(candidate, name).empty for name in SPLIT_NAMES):
        return None
    try:
        assert_no_group_leakage(candidate)
    except ValueError:
        return None
    if not all(_split_has_both_classes(getattr(candidate, name)) for name in SPLIT_NAMES):
        return None
    return candidate


def _generated_group_splits(
    frame: pd.DataFrame,
    config: TrainingConfig,
) -> DatasetSplits:
    groups = frame["offer_group_id"].astype(str)
    if groups.nunique() < 3:
        raise ValueError("At least three offer groups are required for splitting")
    validation_relative = config.validation_fraction / (1.0 - config.test_fraction)
    for attempt in range(128):
        attempt_seed = config.random_seed + attempt
        first = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_fraction,
            random_state=attempt_seed,
        )
        remaining_indices, test_indices = next(
            first.split(frame, frame["pair_label"], groups)
        )
        remaining = frame.iloc[remaining_indices]
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_relative,
            random_state=attempt_seed,
        )
        train_relative, validation_relative_indices = next(
            second.split(
                remaining,
                remaining["pair_label"],
                remaining["offer_group_id"].astype(str),
            )
        )
        candidate = DatasetSplits(
            train=remaining.iloc[train_relative].copy(),
            validation=remaining.iloc[validation_relative_indices].copy(),
            test=frame.iloc[test_indices].copy(),
            method="group_shuffle_split",
        )
        if all(
            _split_has_both_classes(getattr(candidate, name))
            for name in SPLIT_NAMES
        ):
            assert_no_group_leakage(candidate)
            return candidate
    raise ValueError(
        "Unable to produce group-aware train/validation/test splits with both "
        "classes; add more labelled offer groups"
    )


def create_group_splits(
    frame: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> DatasetSplits:
    """Use a complete valid recommended split, otherwise GroupShuffleSplit."""
    effective_config = config or TrainingConfig()
    validated = _validate_training_frame(frame)
    recommended = _recommended_splits(validated)
    splits = recommended or _generated_group_splits(validated, effective_config)
    assert_no_group_leakage(splits)
    return splits


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.loc[:, MODEL_FEATURE_COLUMNS]
    validate_feature_frame_columns(features.columns)
    return features


def _base_model_parameters(config: TrainingConfig) -> dict[str, Any]:
    return {
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": config.random_seed,
        "n_jobs": -1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _fit_candidates(
    splits: DatasetSplits,
    config: TrainingConfig,
) -> tuple[LGBMClassifier, list[dict[str, Any]], dict[str, Any]]:
    train_x = _feature_frame(splits.train)
    validation_x = _feature_frame(splits.validation)
    train_y = splits.train["pair_label"]
    validation_y = splits.validation["pair_label"].to_numpy(dtype=int)
    results: list[dict[str, Any]] = []
    fitted: list[LGBMClassifier] = []
    for index, candidate in enumerate(config.hyperparameter_candidates):
        parameters = {**_base_model_parameters(config), **candidate}
        model = LGBMClassifier(**parameters)
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, splits.validation["pair_label"])],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        probabilities = model.predict_proba(validation_x)[:, 1]
        result = {
            "candidate_index": index,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "validation_pr_auc": float(
                average_precision_score(validation_y, probabilities)
            ),
            "validation_roc_auc": float(
                roc_auc_score(validation_y, probabilities)
            ),
            "parameters": candidate,
        }
        results.append(result)
        fitted.append(model)
    best_index = max(
        range(len(results)),
        key=lambda index: (
            results[index]["validation_pr_auc"],
            results[index]["validation_roc_auc"],
            -results[index]["candidate_index"],
        ),
    )
    return fitted[best_index], results, dict(
        config.hyperparameter_candidates[best_index]
    )


def _decision_labels(
    probabilities: np.ndarray,
    auto_threshold: float,
    manual_threshold: float,
) -> np.ndarray:
    return np.where(
        probabilities >= auto_threshold,
        "AUTO_MATCH",
        np.where(
            probabilities >= manual_threshold,
            "MANUAL_REVIEW",
            "NO_MATCH",
        ),
    )


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(splits: DatasetSplits) -> dict[str, Any]:
    return {
        name: {
            "rows": int(len(split)),
            "offer_groups": int(split["offer_group_id"].nunique()),
            "class_distribution": {
                str(label): int(count)
                for label, count in split["pair_label"]
                .value_counts()
                .sort_index()
                .items()
            },
        }
        for name in SPLIT_NAMES
        for split in [getattr(splits, name)]
    }


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_training_pipeline(
    feature_path: str | Path,
    *,
    processed_dir: str | Path = PROJECT_ROOT / "data" / "processed",
    model_registry_dir: str | Path = PROJECT_ROOT / "models" / "registry",
    metadata_dir: str | Path = PROJECT_ROOT / "models" / "metadata",
    reports_dir: str | Path = PROJECT_ROOT / "reports",
    config: TrainingConfig | None = None,
) -> ModelTrainingResult:
    """Train, tune on validation, evaluate untouched test data, and persist."""
    effective_config = config or TrainingConfig()
    source_path = Path(feature_path)
    assert_not_sealed_challenge_input(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Training feature table not found: {source_path}")
    frame = pd.read_parquet(source_path)
    splits = create_group_splits(frame, effective_config)
    model, candidate_results, selected_parameters = _fit_candidates(
        splits, effective_config
    )

    validation_probabilities = model.predict_proba(
        _feature_frame(splits.validation)
    )[:, 1]
    threshold_result = tune_thresholds(
        splits.validation["pair_label"].to_numpy(dtype=int),
        validation_probabilities,
        target_auto_precision=effective_config.target_auto_precision,
    )
    validation_metrics = evaluate_binary_classifier(
        splits.validation,
        validation_probabilities,
        threshold=threshold_result.auto_match_threshold,
        calibration_bins=effective_config.calibration_bins,
    )

    # This is deliberately the first and only use of test labels/probabilities.
    test_probabilities = model.predict_proba(_feature_frame(splits.test))[:, 1]
    test_metrics = evaluate_binary_classifier(
        splits.test,
        test_probabilities,
        threshold=threshold_result.auto_match_threshold,
        calibration_bins=effective_config.calibration_bins,
    )
    test_predictions = splits.test.copy()
    test_predictions["probability"] = test_probabilities
    test_predictions["decision"] = _decision_labels(
        test_probabilities,
        threshold_result.auto_match_threshold,
        threshold_result.manual_review_threshold,
    )

    timestamp = datetime.now(timezone.utc)
    timestamp_iso = timestamp.isoformat()
    timestamp_token = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    dataset_hash = _dataset_hash(source_path)
    metrics = {
        "dataset": {
            "rows": int(len(frame)),
            "offer_groups": int(frame["offer_group_id"].nunique()),
            "class_distribution": {
                str(label): int(count)
                for label, count in frame["pair_label"]
                .value_counts()
                .sort_index()
                .items()
            },
            "sha256": dataset_hash,
        },
        "split_method": splits.method,
        "splits": _split_summary(splits),
        "split_leakage_check": "passed",
        "hyperparameter_candidates": candidate_results,
        "selected_hyperparameters": selected_parameters,
        "validation": validation_metrics,
        "selected_thresholds": {
            "auto_match_threshold": threshold_result.auto_match_threshold,
            "manual_review_threshold": threshold_result.manual_review_threshold,
            **threshold_result.selected_metrics,
        },
        "test": test_metrics,
        "test_decision_distribution": {
            str(decision): int(count)
            for decision, count in test_predictions["decision"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    package: dict[str, Any] = {
        "model": model,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "auto_match_threshold": threshold_result.auto_match_threshold,
        "manual_review_threshold": threshold_result.manual_review_threshold,
        "model_version": effective_config.model_version,
        "training_timestamp": timestamp_iso,
        "training_dataset_hash": dataset_hash,
        "metrics": metrics,
        "lightgbm_version": lgb.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "training_config": asdict(effective_config),
        "random_seed": effective_config.random_seed,
    }

    processed = Path(processed_dir)
    reports = Path(reports_dir)
    model_path = (
        Path(model_registry_dir)
        / f"{effective_config.model_version}_{timestamp_token}.joblib"
    )
    metadata_path = (
        Path(metadata_dir)
        / f"{effective_config.model_version}_{timestamp_token}.json"
    )
    output_paths = {
        "model": model_path,
        "metadata": metadata_path,
        "train_split": processed / "train_split.parquet",
        "validation_split": processed / "validation_split.parquet",
        "test_split": processed / "test_split.parquet",
        "training_metrics": reports / "training_metrics.json",
        "threshold_analysis": reports / "threshold_analysis.csv",
        "test_predictions": reports / "test_predictions.csv",
        "feature_importance": reports / "feature_importance.csv",
    }
    _atomic_parquet(splits.train, output_paths["train_split"])
    _atomic_parquet(splits.validation, output_paths["validation_split"])
    _atomic_parquet(splits.test, output_paths["test_split"])
    _atomic_json(metrics, output_paths["training_metrics"])
    _atomic_csv(threshold_result.analysis, output_paths["threshold_analysis"])
    _atomic_csv(test_predictions, output_paths["test_predictions"])
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
    _atomic_csv(importance, output_paths["feature_importance"])
    save_model_package(package, model_path, metadata_path)
    return ModelTrainingResult(
        model=model,
        splits=splits,
        threshold_result=threshold_result,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        hyperparameter_results=candidate_results,
        selected_hyperparameters=selected_parameters,
        test_predictions=test_predictions,
        package=package,
        output_paths=output_paths,
    )
