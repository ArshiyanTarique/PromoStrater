"""Strict v3 shadow-package and immutable registry validation."""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import lightgbm
import numpy as np
import pytest
import sklearn
from sklearn.dummy import DummyClassifier

from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.ml.calibration import (
    ShadowModelPredictor,
    SigmoidScoreCalibrator,
)
from sku_mapping.ml.model_package import (
    ModelPackageError,
    update_model_registry,
    validate_model_package,
)


def _major_minor(value: str) -> str:
    return ".".join(value.split(".")[:2])


def _strict_package() -> dict:
    features = np.array([[0.0] * 19, [1.0] * 19])
    labels = np.array([0, 1])
    model = DummyClassifier(strategy="prior").fit(features, labels)
    calibration = SigmoidScoreCalibrator(42).fit(
        np.array([-1.0, 1.0]), labels
    )
    predictor = ShadowModelPredictor(
        model=model,
        calibration_model=calibration,
        feature_columns=list(MODEL_FEATURE_COLUMNS),
    )
    digest = "a" * 64
    return {
        "package_schema_version": "3.0",
        "model": model,
        "calibration_model": calibration,
        "predictor": predictor,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "feature_count": 19,
        "auto_match_threshold": 0.95,
        "manual_review_threshold": 0.5,
        "approved_auto_match_threshold": None,
        "auto_match_threshold_approved": False,
        "model_id": "shadow-v3-model-unique",
        "package_version": "3.0.0+test",
        "model_version": "alkabeer_sku_matcher_v3",
        "parent_model": "v2.joblib",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_dataset_hash": digest,
        "processed_feature_table_hash": digest,
        "split_assignment_hash": digest,
        "metrics": {},
        "threshold_evidence": {},
        "calibration_method": "sigmoid",
        "lightgbm_version": lightgbm.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "compatibility_policy": {
            "python_major_minor": _major_minor(platform.python_version()),
            "lightgbm_major_minor": _major_minor(lightgbm.__version__),
            "sklearn_major_minor": _major_minor(sklearn.__version__),
        },
        "training_config": {},
        "random_seed": 42,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
    }


def test_strict_shadow_package_passes_current_runtime_policy() -> None:
    validate_model_package(_strict_package())


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("deployment_status", "PRODUCTION", "SHADOW_MODE_ONLY"),
        ("automatic_production_matching_approved", True, "must be false"),
        ("feature_count", 18, "feature_count"),
        ("training_dataset_hash", "bad", "SHA-256"),
        ("approved_auto_match_threshold", 0.95, "cannot contain an approved"),
    ],
)
def test_strict_shadow_package_rejects_incompatible_fields(
    key: str, value: object, message: str
) -> None:
    package = _strict_package()
    package[key] = value
    with pytest.raises(ModelPackageError, match=message):
        validate_model_package(package)


def test_strict_shadow_package_enforces_runtime_version_policy() -> None:
    package = _strict_package()
    package["compatibility_policy"]["sklearn_major_minor"] = "0.0"
    with pytest.raises(ModelPackageError, match="violates package policy"):
        validate_model_package(package)


def test_registry_rejects_duplicate_immutable_model_id(tmp_path) -> None:
    base = {
        "package_filename": "first.joblib",
        "model_id": "immutable-id",
        "model_version": "v3",
        "package_version": "3.0.0+one",
        "creation_timestamp": "2026-01-01T00:00:00+00:00",
        "training_dataset_hash": "a" * 64,
        "feature_generator_version": "1.0.0",
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
        "notes": "test",
        "parent_model": None,
    }
    path = tmp_path / "registry.json"
    update_model_registry(path, [base])
    duplicate = {**base, "package_filename": "second.joblib", "package_version": "3.0.0+two"}
    with pytest.raises(ModelPackageError, match="Duplicate immutable registry model_id"):
        update_model_registry(path, [duplicate])
