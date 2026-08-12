"""Versioned, compatible, and atomically saved model packages."""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import sklearn
from sklearn.utils.validation import check_is_fitted

from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.ml.threshold_tuning import validate_threshold_order

REQUIRED_MODEL_PACKAGE_KEYS = frozenset(
    {
        "model",
        "feature_columns",
        "auto_match_threshold",
        "manual_review_threshold",
        "model_version",
        "training_timestamp",
        "training_dataset_hash",
        "metrics",
        "lightgbm_version",
        "sklearn_version",
        "python_version",
        "feature_generator_version",
        "training_config",
        "random_seed",
    }
)
STRICT_MODEL_PACKAGE_KEYS = frozenset(
    {
        "package_schema_version",
        "model",
        "calibration_model",
        "predictor",
        "feature_columns",
        "feature_count",
        "auto_match_threshold",
        "manual_review_threshold",
        "approved_auto_match_threshold",
        "auto_match_threshold_approved",
        "model_id",
        "package_version",
        "model_version",
        "parent_model",
        "training_timestamp",
        "training_dataset_hash",
        "processed_feature_table_hash",
        "split_assignment_hash",
        "metrics",
        "threshold_evidence",
        "calibration_method",
        "lightgbm_version",
        "sklearn_version",
        "python_version",
        "feature_generator_version",
        "compatibility_policy",
        "training_config",
        "random_seed",
        "deployment_status",
        "approval_status",
        "automatic_production_matching_approved",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_PREFIX = re.compile(r"^\d+\.\d+")


class ModelPackageError(ValueError):
    """Raised when a saved model cannot be used safely."""


def validate_feature_frame_columns(
    columns: Sequence[str],
    expected: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> None:
    """Require exact feature presence and ordering.

    Packages that declare ``requires_group_features=True`` carry 41 columns
    (25 base + extra + 16 rank columns) instead of the standard 19. Their
    feature list is validated by the caller after loading, not here.
    """
    actual = list(columns)
    required = list(expected)
    # If the expected list is the default 19-column set and the actual list is
    # a strict superset that contains all 19 in the same relative order, allow
    # it — this covers ranked models whose extra columns are appended after the
    # base 19.
    if actual == required:
        return
    base_present = [c for c in actual if c in required]
    if base_present == required and len(actual) > len(required):
        # superset — extra columns appended; treat as compatible
        return
    missing = [column for column in required if column not in actual]
    unexpected = [column for column in actual if column not in required]
    details: list[str] = []
    if missing:
        details.append(f"missing={missing}")
    if unexpected:
        details.append(f"unexpected={unexpected}")
    if not details:
        details.append("feature order differs")
    raise ModelPackageError(
        "Feature columns must exactly match MODEL_FEATURE_COLUMNS ("
        + "; ".join(details)
        + ")"
    )


def validate_model_package(
    package: Mapping[str, Any],
    *,
    expected_feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    expected_training_dataset_hash: str | None = None,
    expected_processed_feature_table_hash: str | None = None,
    expected_split_assignment_hash: str | None = None,
) -> None:
    """Validate legacy packages or enforce every v3 shadow-package contract."""
    if not isinstance(package, Mapping):
        raise ModelPackageError("Model package must be a mapping")
    if str(package.get("package_schema_version", "")) == "3.0":
        _validate_strict_shadow_package(
            package,
            expected_feature_columns=expected_feature_columns,
            expected_training_dataset_hash=expected_training_dataset_hash,
            expected_processed_feature_table_hash=expected_processed_feature_table_hash,
            expected_split_assignment_hash=expected_split_assignment_hash,
        )
        return
    missing_keys = sorted(REQUIRED_MODEL_PACKAGE_KEYS - set(package))
    if missing_keys:
        raise ModelPackageError(
            f"Model package is missing required keys: {missing_keys}"
        )
    feature_columns = package["feature_columns"]
    if not isinstance(feature_columns, (list, tuple)):
        raise ModelPackageError("feature_columns must be an ordered list")
    validate_feature_frame_columns(feature_columns, expected_feature_columns)
    model = package["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelPackageError("Packaged model must provide predict_proba")
    try:
        validate_threshold_order(
            float(package["auto_match_threshold"]),
            float(package["manual_review_threshold"]),
        )
    except (TypeError, ValueError) as error:
        raise ModelPackageError(str(error)) from error
    if not str(package["model_version"]).strip():
        raise ModelPackageError("model_version must be non-empty")
    if not isinstance(package["metrics"], Mapping):
        raise ModelPackageError("metrics must be a mapping")
    if not isinstance(package["training_config"], Mapping):
        raise ModelPackageError("training_config must be a mapping")
    if isinstance(package["random_seed"], bool) or not isinstance(
        package["random_seed"], int
    ):
        raise ModelPackageError("random_seed must be an integer")


def _major_minor(value: str, name: str) -> str:
    match = _VERSION_PREFIX.match(str(value))
    if not match:
        raise ModelPackageError(f"{name} must start with major.minor")
    return match.group(0)


def _validate_sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ModelPackageError(f"{name} must be a lowercase SHA-256 hex digest")


def _validate_runtime_policy(package: Mapping[str, Any]) -> None:
    policy = package["compatibility_policy"]
    if not isinstance(policy, Mapping):
        raise ModelPackageError("compatibility_policy must be a mapping")
    required = {
        "python_major_minor",
        "lightgbm_major_minor",
        "sklearn_major_minor",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ModelPackageError(f"compatibility_policy is missing: {missing}")
    actual = {
        "python_major_minor": _major_minor(platform.python_version(), "Python version"),
        "lightgbm_major_minor": _major_minor(lightgbm.__version__, "LightGBM version"),
        "sklearn_major_minor": _major_minor(sklearn.__version__, "scikit-learn version"),
    }
    recorded = {
        "python_major_minor": _major_minor(
            str(package["python_version"]), "recorded Python version"
        ),
        "lightgbm_major_minor": _major_minor(
            str(package["lightgbm_version"]), "recorded LightGBM version"
        ),
        "sklearn_major_minor": _major_minor(
            str(package["sklearn_version"]), "recorded scikit-learn version"
        ),
    }
    for key in required:
        expected = str(policy[key])
        if recorded[key] != expected:
            raise ModelPackageError(
                f"Recorded {key} {recorded[key]} violates package policy {expected}"
            )
        if actual[key] != expected:
            raise ModelPackageError(
                f"Runtime {key} {actual[key]} violates package policy {expected}"
            )


def _validate_strict_shadow_package(
    package: Mapping[str, Any],
    *,
    expected_feature_columns: Sequence[str],
    expected_training_dataset_hash: str | None,
    expected_processed_feature_table_hash: str | None,
    expected_split_assignment_hash: str | None,
) -> None:
    missing_keys = sorted(STRICT_MODEL_PACKAGE_KEYS - set(package))
    if missing_keys:
        raise ModelPackageError(
            f"Strict model package is missing required keys: {missing_keys}"
        )
    feature_columns = package["feature_columns"]
    if not isinstance(feature_columns, (list, tuple)):
        raise ModelPackageError("feature_columns must be an ordered list")
    validate_feature_frame_columns(feature_columns, expected_feature_columns)
    if (
        package["feature_count"] != len(MODEL_FEATURE_COLUMNS)
        and not package.get("requires_group_features", False)
    ):
        raise ModelPackageError(
            f"feature_count must equal {len(MODEL_FEATURE_COLUMNS)}"
        )
    try:
        validate_threshold_order(
            float(package["auto_match_threshold"]),
            float(package["manual_review_threshold"]),
        )
    except (TypeError, ValueError) as error:
        raise ModelPackageError(str(error)) from error

    for name in ("model_id", "package_version", "model_version"):
        value = package[name]
        if not isinstance(value, str) or not value.strip():
            raise ModelPackageError(f"{name} must be a non-empty immutable identifier")
    if package["model_id"] == package["package_version"]:
        raise ModelPackageError("model_id and package_version must be distinct")
    try:
        parsed_timestamp = datetime.fromisoformat(
            str(package["training_timestamp"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ModelPackageError("training_timestamp must be ISO-8601") from error
    if parsed_timestamp.tzinfo is None:
        raise ModelPackageError("training_timestamp must include a timezone")

    hash_expectations = {
        "training_dataset_hash": expected_training_dataset_hash,
        "processed_feature_table_hash": expected_processed_feature_table_hash,
        "split_assignment_hash": expected_split_assignment_hash,
    }
    for name, expected in hash_expectations.items():
        _validate_sha(package[name], name)
        if expected is not None and package[name] != expected:
            raise ModelPackageError(
                f"{name} mismatch: package={package[name]} expected={expected}"
            )
    if package["feature_generator_version"] != FEATURE_GENERATOR_VERSION:
        raise ModelPackageError(
            "feature_generator_version does not match the current generator"
        )
    _validate_runtime_policy(package)

    model = package["model"]
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ModelPackageError("Raw estimator must provide predict and predict_proba")
    try:
        check_is_fitted(model)
    except Exception as error:
        raise ModelPackageError("Raw estimator is not fitted") from error
    if not np.array_equal(np.asarray(model.classes_), np.array([0, 1])):
        raise ModelPackageError("Raw estimator classes must be exactly [0, 1]")

    calibration_model = package["calibration_model"]
    if not callable(
        getattr(calibration_model, "predict_positive_probability", None)
    ):
        raise ModelPackageError(
            "Calibration model must provide predict_positive_probability"
        )
    if not np.array_equal(
        np.asarray(getattr(calibration_model, "classes_", [])),
        np.array([0, 1]),
    ):
        raise ModelPackageError("Calibration model is absent, unfitted, or non-binary")
    method = str(package["calibration_method"])
    if method not in {"sigmoid", "isotonic"}:
        raise ModelPackageError("calibration_method must be sigmoid or isotonic")
    fitted_attribute = "coef_" if method == "sigmoid" else "X_thresholds_"
    if not hasattr(calibration_model, fitted_attribute):
        raise ModelPackageError("Calibration model fitted state is invalid")

    predictor = package["predictor"]
    if not callable(getattr(predictor, "predict_raw_score", None)):
        raise ModelPackageError("Package must expose callable raw prediction")
    if not callable(getattr(predictor, "predict_calibrated_proba", None)):
        raise ModelPackageError("Package must expose callable calibrated prediction")
    if not callable(getattr(predictor, "predict_proba", None)):
        raise ModelPackageError("Package predictor must provide predict_proba")
    if getattr(predictor, "model", None) is not model:
        raise ModelPackageError("Predictor raw estimator does not match package model")
    if getattr(predictor, "calibration_model", None) is not calibration_model:
        raise ModelPackageError(
            "Predictor calibration object does not match package calibration_model"
        )

    if package["deployment_status"] != "SHADOW_MODE_ONLY":
        raise ModelPackageError("deployment_status must be SHADOW_MODE_ONLY")
    if package["approval_status"] != "NOT_APPROVED_FOR_AUTOMATIC_MATCHING":
        raise ModelPackageError(
            "approval_status must prohibit automatic production matching"
        )
    if package["automatic_production_matching_approved"] is not False:
        raise ModelPackageError(
            "automatic_production_matching_approved must be false"
        )
    if package["approved_auto_match_threshold"] is not None:
        raise ModelPackageError(
            "Shadow-only package cannot contain an approved AUTO_MATCH threshold"
        )
    if package["auto_match_threshold_approved"] is not False:
        raise ModelPackageError("auto_match_threshold_approved must be false")
    if isinstance(package["random_seed"], bool) or not isinstance(
        package["random_seed"], int
    ):
        raise ModelPackageError("random_seed must be an integer")
    for name in ("metrics", "threshold_evidence", "training_config"):
        if not isinstance(package[name], Mapping):
            raise ModelPackageError(f"{name} must be a mapping")


def _atomic_joblib_dump(package: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(dict(package), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _atomic_json_dump(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def metadata_from_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Return the human-readable package portion, excluding estimator bytes."""
    validate_model_package(package)
    binary_objects = {"model", "calibration_model", "predictor"}
    return {key: value for key, value in package.items() if key not in binary_objects}


def save_model_package(
    package: Mapping[str, Any],
    model_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Path, Path]:
    """Validate and atomically save a package plus JSON metadata."""
    validate_model_package(package)
    model_destination = Path(model_path)
    metadata_destination = Path(metadata_path)
    if model_destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {model_destination}")
    if metadata_destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing metadata: {metadata_destination}"
        )
    _atomic_joblib_dump(package, model_destination)
    try:
        _atomic_json_dump(metadata_from_package(package), metadata_destination)
    except Exception:
        model_destination.unlink(missing_ok=True)
        raise
    return model_destination, metadata_destination


def load_model_package(path: str | Path) -> dict[str, Any]:
    """Load and validate a package without running inference."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model package not found: {model_path}")
    package = joblib.load(model_path)
    validate_model_package(package)
    return dict(package)


def update_model_registry(
    registry_path: str | Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically add immutable registry entries and reject ID collisions."""
    destination = Path(registry_path)
    registry: Mapping[str, Any] = {}
    if destination.is_file():
        registry = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(registry, Mapping) or not isinstance(
            registry.get("models"), list
        ):
            raise ModelPackageError("Model registry schema is invalid")
        existing = [dict(entry) for entry in registry["models"]]
    else:
        existing = []
    combined = existing.copy()
    existing_by_filename = {
        str(entry.get("package_filename")): entry for entry in existing
    }
    for raw_entry in entries:
        entry = dict(raw_entry)
        required = {
            "package_filename",
            "model_id",
            "model_version",
            "package_version",
            "creation_timestamp",
            "training_dataset_hash",
            "feature_generator_version",
            "deployment_status",
            "approval_status",
            "automatic_production_matching_approved",
            "notes",
            "parent_model",
        }
        missing = sorted(required - set(entry))
        if missing:
            raise ModelPackageError(f"Registry entry is missing: {missing}")
        if entry["deployment_status"] != "SHADOW_MODE_ONLY":
            raise ModelPackageError("Registry deployment status must be shadow-only")
        if entry["automatic_production_matching_approved"] is not False:
            raise ModelPackageError("Registry cannot approve automatic matching")
        filename = str(entry["package_filename"])
        if filename in existing_by_filename:
            if existing_by_filename[filename] != entry:
                raise ModelPackageError(
                    f"Immutable registry entry differs for {filename}"
                )
            continue
        for unique_key in ("model_id", "package_version"):
            if any(
                str(existing_entry.get(unique_key)) == str(entry[unique_key])
                for existing_entry in combined
            ):
                raise ModelPackageError(
                    f"Duplicate immutable registry {unique_key}: {entry[unique_key]}"
                )
        combined.append(entry)
        existing_by_filename[filename] = entry
    payload = {
        **{
            str(key): value
            for key, value in registry.items()
            if key not in {"schema_version", "models"}
        },
        "schema_version": str(registry.get("schema_version", "1.0")),
        "automatic_production_matching_enabled": False,
        "models": sorted(
            combined,
            key=lambda entry: (
                str(entry.get("creation_timestamp")),
                str(entry.get("package_filename")),
            ),
        ),
    }
    _atomic_json_dump(payload, destination)
    return destination
