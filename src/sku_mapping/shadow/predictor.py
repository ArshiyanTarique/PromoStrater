"""Fail-closed loading and prediction for registered shadow-only packages."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.model_package import validate_model_package

LOGGER = logging.getLogger(__name__)

SHADOW_BUCKETS = (
    "SHADOW_HIGH_SCORE",
    "SHADOW_REVIEW",
    "SHADOW_LOW_SCORE",
)


class ShadowPackageError(ValueError):
    """Raised when an explicitly requested shadow package is unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RegisteredShadowPackage:
    """A validated immutable model package and its registry identity."""

    package: dict[str, Any]
    registry_entry: dict[str, Any]
    package_path: Path
    package_sha256: str


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ShadowPackageError(f"Model registry not found: {path}")
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShadowPackageError(f"Model registry is unreadable: {path}") from error
    if not isinstance(registry, dict) or not isinstance(
        registry.get("models"), list
    ):
        raise ShadowPackageError("Model registry schema is invalid")
    if registry.get("automatic_production_matching_enabled") is not False:
        raise ShadowPackageError(
            "Registry must keep automatic production matching disabled"
        )
    return registry


def load_registered_shadow_package(
    *,
    registry_path: str | Path,
    model_directory: str | Path,
    model_id: str | None = None,
    package_reference: str | Path | None = None,
    require_package_status: str = "SHADOW_MODE_ONLY",
) -> RegisteredShadowPackage:
    """Load only one explicitly identified registered shadow package.

    No newest-package fallback exists. Both model-ID and package-reference
    selection resolve through the registry before any joblib bytes are loaded.
    """
    if bool(model_id) == bool(package_reference):
        raise ShadowPackageError(
            "Specify exactly one explicit model_id or immutable package_reference"
        )
    if require_package_status != "SHADOW_MODE_ONLY":
        raise ShadowPackageError("Only SHADOW_MODE_ONLY packages may load")

    registry_file = Path(registry_path)
    registry = _read_registry(registry_file)
    entries = [dict(entry) for entry in registry["models"]]
    if model_id:
        matches = [entry for entry in entries if entry.get("model_id") == model_id]
    else:
        reference = Path(str(package_reference))
        matches = [
            entry
            for entry in entries
            if entry.get("package_filename") == reference.name
        ]
    if len(matches) != 1:
        selector = model_id or Path(str(package_reference)).name
        raise ShadowPackageError(
            f"Explicit shadow package selector must match exactly one registry "
            f"entry: {selector!r}"
        )
    entry = matches[0]
    if entry.get("deployment_status") != require_package_status:
        raise ShadowPackageError("Registry entry is not SHADOW_MODE_ONLY")
    if entry.get("automatic_production_matching_approved") is not False:
        raise ShadowPackageError(
            "Registry entry permits automatic production matching"
        )
    if entry.get("approval_status") != "NOT_APPROVED_FOR_AUTOMATIC_MATCHING":
        raise ShadowPackageError("Registry approval status is unsafe")

    registered_path = (
        Path(model_directory) / str(entry.get("package_filename", ""))
    ).resolve()
    if package_reference is not None:
        requested = Path(package_reference)
        if not requested.is_absolute():
            requested = Path(model_directory) / requested
        if requested.resolve() != registered_path:
            raise ShadowPackageError(
                "Package reference does not match its immutable registry path"
            )
    if not registered_path.is_file():
        raise ShadowPackageError(
            f"Registered model package is missing: {registered_path}"
        )

    try:
        package = joblib.load(registered_path)
        validate_model_package(package)
    except Exception as error:
        LOGGER.exception("Shadow model package validation failed")
        raise ShadowPackageError(
            f"Registered shadow package failed compatibility validation: "
            f"{registered_path.name}"
        ) from error
    if package.get("model_id") != entry.get("model_id"):
        raise ShadowPackageError("Package model_id differs from registry entry")
    if package.get("deployment_status") != "SHADOW_MODE_ONLY":
        raise ShadowPackageError("Package deployment status is not shadow-only")
    if package.get("automatic_production_matching_approved") is not False:
        raise ShadowPackageError("Package improperly permits automatic matching")
    if package.get("approved_auto_match_threshold") is not None:
        raise ShadowPackageError("Package contains a production-approved threshold")
    if (
        list(package.get("feature_columns", [])) != MODEL_FEATURE_COLUMNS
        and not package.get("requires_group_features", False)
    ):
        raise ShadowPackageError("Package feature order is incompatible")

    return RegisteredShadowPackage(
        package=dict(package),
        registry_entry=entry,
        package_path=registered_path,
        package_sha256=_sha256_file(registered_path),
    )


class ShadowPredictor:
    """Observational predictor that can emit only diagnostic shadow buckets."""

    def __init__(self, registered: RegisteredShadowPackage) -> None:
        self.registered = registered
        self.package = registered.package
        self.predictor = self.package["predictor"]
        self.feature_columns = list(self.package["feature_columns"])
        # Ranked models carry 41 columns (19 base + 6 extra + 16 rank).
        # They are compatible as long as the base 19 are present in order.
        base_present = [c for c in self.feature_columns if c in MODEL_FEATURE_COLUMNS]
        if (
            self.feature_columns != MODEL_FEATURE_COLUMNS
            and base_present != MODEL_FEATURE_COLUMNS
        ):
            raise ShadowPackageError("Shadow feature order is incompatible")

    @property
    def model_id(self) -> str:
        return str(self.package["model_id"])

    @property
    def high_score_threshold(self) -> float:
        return float(self.package["auto_match_threshold"])

    @property
    def review_threshold(self) -> float:
        return float(self.package["manual_review_threshold"])

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return scores and diagnostic buckets without production actions."""
        expected = self.feature_columns
        if list(features.columns) not in (MODEL_FEATURE_COLUMNS, expected):
            raise ShadowPackageError(
                "Shadow prediction requires exact MODEL_FEATURE_COLUMNS order"
            )
        raw_scores = np.asarray(
            self.predictor.predict_raw_score(features), dtype=float
        )
        model = self.package.get("model")
        if callable(getattr(model, "predict_proba", None)):
            raw_probabilities = np.asarray(
                model.predict_proba(features), dtype=float
            )[:, 1]
        else:
            # Compatibility for observational test doubles and early v3
            # packages: LightGBM raw scores use the logistic link.
            raw_probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
        probabilities = np.asarray(
            self.predictor.predict_calibrated_proba(features), dtype=float
        )
        if (
            raw_scores.shape != (len(features),)
            or raw_probabilities.shape != (len(features),)
            or probabilities.shape != (len(features),)
        ):
            raise ShadowPackageError("Shadow predictor returned invalid shapes")
        if not np.isfinite(raw_scores).all():
            raise ShadowPackageError("Shadow raw scores must be finite")
        if (
            not np.isfinite(raw_probabilities).all()
            or (raw_probabilities < 0).any()
            or (raw_probabilities > 1).any()
            or
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ShadowPackageError(
                "Shadow calibrated probabilities must be within [0, 1]"
            )
        buckets = np.where(
            probabilities >= self.high_score_threshold,
            "SHADOW_HIGH_SCORE",
            np.where(
                probabilities >= self.review_threshold,
                "SHADOW_REVIEW",
                "SHADOW_LOW_SCORE",
            ),
        )
        if not set(buckets).issubset(SHADOW_BUCKETS):
            raise AssertionError("Shadow predictor exposed a non-shadow action")
        return pd.DataFrame(
            {
                "raw_model_score": raw_scores,
                "raw_probability": raw_probabilities,
                "calibrated_probability": probabilities,
                "shadow_decision_bucket": buckets,
            },
            index=features.index,
        )
