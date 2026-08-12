"""Strict model-package compatibility tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sklearn.dummy import DummyClassifier

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.model_package import (
    ModelPackageError,
    load_model_package,
    save_model_package,
    validate_model_package,
)


def _package(model=None) -> dict:
    return {
        "model": model or DummyClassifier(),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "auto_match_threshold": 0.9,
        "manual_review_threshold": 0.5,
        "model_version": "test-v2",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_dataset_hash": "abc123",
        "metrics": {},
        "lightgbm_version": "test",
        "sklearn_version": "test",
        "python_version": "test",
        "feature_generator_version": "test",
        "training_config": {},
        "random_seed": 42,
    }


def test_feature_order_validation_rejects_reordered_columns() -> None:
    package = _package()
    package["feature_columns"] = list(reversed(MODEL_FEATURE_COLUMNS))
    with pytest.raises(ModelPackageError, match="feature order"):
        validate_model_package(package)


def test_invalid_model_package_rejects_missing_keys() -> None:
    package = _package()
    del package["training_dataset_hash"]
    with pytest.raises(ModelPackageError, match="missing required keys"):
        validate_model_package(package)


def test_model_package_requires_predict_proba() -> None:
    class PredictOnly:
        def predict(self, values):
            return values

    with pytest.raises(ModelPackageError, match="predict_proba"):
        validate_model_package(_package(PredictOnly()))


def test_model_package_rejects_invalid_threshold_order() -> None:
    package = _package()
    package["manual_review_threshold"] = package["auto_match_threshold"]
    with pytest.raises(ModelPackageError, match="manual_review_threshold"):
        validate_model_package(package)


def test_model_package_round_trip(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    package = _package()
    save_model_package(package, model_path, metadata_path)
    loaded = load_model_package(model_path)
    assert loaded["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert metadata_path.is_file()
