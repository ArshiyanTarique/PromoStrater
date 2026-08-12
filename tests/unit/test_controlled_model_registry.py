"""Explicit activation, rollback, and immutable registry tests."""

from __future__ import annotations

import json
from pathlib import Path

from sku_mapping.learning.store import LearningStore
from sku_mapping.ml.model_package import load_model_package, save_model_package
from sku_mapping.retraining.registry import (
    ControlledModelRegistry,
    RegistryTransitionError,
)
from sku_mapping.retraining.artifacts import sha256_file
import pytest


from tests.retraining_fixtures import (  # noqa: E402
    CHAMPION_ID,
    CHAMPION_PACKAGE_FILENAME,
    build_champion_package,
    champion_registry_entry,
)


def _registry_fixture(
    tmp_path: Path,
) -> tuple[ControlledModelRegistry, Path, Path]:
    models = tmp_path / "models"
    registry_directory = models / "registry"
    metadata_directory = models / "metadata"
    registry_directory.mkdir(parents=True)
    metadata_directory.mkdir(parents=True)
    champion_path = build_champion_package(
        registry_directory, metadata_directory
    )
    champion_entry = champion_registry_entry()
    registry_path = models / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "automatic_production_matching_enabled": False,
                "models": [champion_entry],
            }
        ),
        encoding="utf-8",
    )
    store = LearningStore(tmp_path / "learning.db")
    store.register_model_version(
        model_id=CHAMPION_ID,
        model_hash=sha256_file(champion_path),
        status="EXISTING_CHAMPION",
        champion_status="PREVIOUS_CHAMPION",
    )
    registry = ControlledModelRegistry(
        registry_path=registry_path,
        model_directory=registry_directory,
        metadata_directory=metadata_directory,
        store=store,
    )
    package = load_model_package(champion_path)
    package["model_id"] = "challenger-test-id"
    package["package_version"] = "4.0.0+test"
    package["model_version"] = "alkabeer_sku_matcher_v4_challenger"
    package["parent_model"] = CHAMPION_ID
    challenger_directory = tmp_path / "challenger"
    challenger_path = challenger_directory / "challenger-test-id.joblib"
    metadata_path = challenger_directory / "challenger-test-id.json"
    save_model_package(package, challenger_path, metadata_path)
    store.register_model_version(
        model_id="challenger-test-id",
        model_hash=sha256_file(challenger_path),
        status="CHALLENGER_TRAINED",
        parent_model_id=CHAMPION_ID,
        training_dataset_id="fixture-dataset",
        champion_status="CHALLENGER_NOT_ACTIVE",
    )
    return registry, challenger_path, metadata_path


def _passing_summary() -> dict[str, object]:
    return {
        "comparison_id": "comparison-test",
        "challenger_model_id": "challenger-test-id",
        "promotion_policy_passed": True,
    }


def test_registration_does_not_activate_and_activation_is_explicit(
    tmp_path: Path,
) -> None:
    registry, package, metadata = _registry_fixture(tmp_path)
    registry.register_approved_challenger(
        package_path=package,
        metadata_path=metadata,
        comparison_summary=_passing_summary(),
    )
    assert registry.active_model_id() is None
    previous = registry.activate(
        model_id="challenger-test-id",
        expected_current_model_id=CHAMPION_ID,
        actor="unit-test",
        reason="explicit controlled activation",
    )
    assert previous == CHAMPION_ID
    assert registry.active_model_id() == "challenger-test-id"


def test_rollback_restores_previous_champion(tmp_path: Path) -> None:
    registry, package, metadata = _registry_fixture(tmp_path)
    registry.register_approved_challenger(
        package_path=package,
        metadata_path=metadata,
        comparison_summary=_passing_summary(),
    )
    registry.activate(
        model_id="challenger-test-id",
        expected_current_model_id=CHAMPION_ID,
        actor="unit-test",
        reason="activation before rollback",
    )
    restored = registry.rollback(
        actor="unit-test",
        reason="observed regression",
    )
    assert restored == CHAMPION_ID
    assert registry.active_model_id() == CHAMPION_ID


def test_challenger_cannot_overwrite_champion(tmp_path: Path) -> None:
    registry, package, metadata = _registry_fixture(tmp_path)
    collision = tmp_path / "collision" / CHAMPION_PACKAGE_FILENAME
    collision.parent.mkdir()
    collision.write_bytes(package.read_bytes())
    with pytest.raises(RegistryTransitionError, match="differs"):
        registry.register_approved_challenger(
            package_path=collision,
            metadata_path=metadata,
            comparison_summary=_passing_summary(),
        )
