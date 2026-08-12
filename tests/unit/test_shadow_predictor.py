"""Registered shadow-package loading and diagnostic prediction tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.shadow import predictor as predictor_module
from sku_mapping.shadow.predictor import (
    RegisteredShadowPackage,
    ShadowPackageError,
    ShadowPredictor,
    load_registered_shadow_package,
)


class _FakePredictor:
    def predict_raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([-2.0, 0.0, 2.0])[: len(frame)]

    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([0.01, 0.5, 0.99])[: len(frame)]


def _package() -> dict:
    return {
        "model_id": "explicit-shadow-id",
        "deployment_status": "SHADOW_MODE_ONLY",
        "automatic_production_matching_approved": False,
        "approved_auto_match_threshold": None,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "auto_match_threshold": 0.9,
        "manual_review_threshold": 0.1,
        "predictor": _FakePredictor(),
    }


def _registry(tmp_path) -> tuple:
    model_directory = tmp_path / "models" / "registry"
    model_directory.mkdir(parents=True)
    package_path = model_directory / "explicit.joblib"
    package_path.write_bytes(b"immutable-test-package")
    registry_path = tmp_path / "models" / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "automatic_production_matching_enabled": False,
                "models": [
                    {
                        "model_id": "explicit-shadow-id",
                        "package_filename": package_path.name,
                        "deployment_status": "SHADOW_MODE_ONLY",
                        "approval_status": (
                            "NOT_APPROVED_FOR_AUTOMATIC_MATCHING"
                        ),
                        "automatic_production_matching_approved": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return registry_path, model_directory, package_path


def test_only_explicit_registered_shadow_package_loads(
    tmp_path, monkeypatch
) -> None:
    registry_path, model_directory, package_path = _registry(tmp_path)
    validation_calls: list[dict] = []
    monkeypatch.setattr(
        predictor_module.joblib, "load", lambda _: _package()
    )
    monkeypatch.setattr(
        predictor_module,
        "validate_model_package",
        lambda package: validation_calls.append(package),
    )

    loaded = load_registered_shadow_package(
        registry_path=registry_path,
        model_directory=model_directory,
        model_id="explicit-shadow-id",
    )
    assert loaded.package_path == package_path.resolve()
    assert validation_calls == [loaded.package]

    with pytest.raises(ShadowPackageError, match="exactly one explicit"):
        load_registered_shadow_package(
            registry_path=registry_path,
            model_directory=model_directory,
        )


def test_package_validation_failure_occurs_before_prediction(
    tmp_path, monkeypatch
) -> None:
    registry_path, model_directory, _ = _registry(tmp_path)
    monkeypatch.setattr(
        predictor_module.joblib, "load", lambda _: _package()
    )
    monkeypatch.setattr(
        predictor_module,
        "validate_model_package",
        lambda _: (_ for _ in ()).throw(ValueError("invalid package")),
    )
    with pytest.raises(ShadowPackageError, match="compatibility validation"):
        load_registered_shadow_package(
            registry_path=registry_path,
            model_directory=model_directory,
            model_id="explicit-shadow-id",
        )


def test_shadow_prediction_preserves_feature_order_and_has_no_auto_action(
    tmp_path,
) -> None:
    path = tmp_path / "package.joblib"
    path.write_bytes(b"x")
    registered = RegisteredShadowPackage(
        package=_package(),
        registry_entry={"model_id": "explicit-shadow-id"},
        package_path=path,
        package_sha256="a" * 64,
    )
    predictor = ShadowPredictor(registered)
    features = pd.DataFrame(
        [[0.0] * len(MODEL_FEATURE_COLUMNS) for _ in range(3)],
        columns=MODEL_FEATURE_COLUMNS,
    )
    output = predictor.predict(features)
    assert output["shadow_decision_bucket"].tolist() == [
        "SHADOW_LOW_SCORE",
        "SHADOW_REVIEW",
        "SHADOW_HIGH_SCORE",
    ]
    assert not output["shadow_decision_bucket"].str.contains(
        "AUTO_MATCH", regex=False
    ).any()

    with pytest.raises(ShadowPackageError, match="exact"):
        predictor.predict(features.loc[:, list(reversed(MODEL_FEATURE_COLUMNS))])
