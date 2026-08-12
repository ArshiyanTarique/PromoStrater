"""Immutable model registration with explicit activation and rollback."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sku_mapping.learning.store import LearningStore, LearningStoreError
from sku_mapping.ml.model_package import (
    ModelPackageError,
    load_model_package,
)
from sku_mapping.retraining.artifacts import atomic_json, sha256_file


class RegistryTransitionError(ValueError):
    """Raised when registration or activation would violate lifecycle policy."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ControlledModelRegistry:
    """Registry service whose active pointer changes only through activation."""

    def __init__(
        self,
        *,
        registry_path: str | Path,
        model_directory: str | Path,
        metadata_directory: str | Path,
        store: LearningStore,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.model_directory = Path(model_directory)
        self.metadata_directory = Path(metadata_directory)
        self.store = store

    def _read(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            raise RegistryTransitionError(
                f"Model registry not found: {self.registry_path}"
            )
        try:
            payload = json.loads(
                self.registry_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RegistryTransitionError("Model registry is unreadable") from error
        if not isinstance(payload, dict) or not isinstance(
            payload.get("models"), list
        ):
            raise RegistryTransitionError("Model registry schema is invalid")
        if payload.get("automatic_production_matching_enabled") is not False:
            raise RegistryTransitionError(
                "Automatic production matching must remain disabled"
            )
        payload.setdefault("active_assisted_model_id", None)
        payload.setdefault("activation_history", [])
        return payload

    def _entry(self, registry: Mapping[str, Any], model_id: str) -> dict[str, Any]:
        matches = [
            dict(entry)
            for entry in registry["models"]
            if str(entry.get("model_id")) == model_id
        ]
        if len(matches) != 1:
            raise RegistryTransitionError(
                f"Model ID must identify one registry entry: {model_id!r}"
            )
        return matches[0]

    def active_model_id(self) -> str | None:
        value = self._read().get("active_assisted_model_id")
        return str(value) if value else None

    def _require_learning_model(self, model_id: str) -> None:
        if not any(
            str(record.get("model_id")) == model_id
            for record in self.store.list_model_versions()
        ):
            raise RegistryTransitionError(
                f"Learning-store model lifecycle is missing: {model_id!r}"
            )

    def load_registered_package(
        self, model_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        registry = self._read()
        entry = self._entry(registry, model_id)
        path = (
            self.model_directory / str(entry.get("package_filename") or "")
        ).resolve()
        root = self.model_directory.resolve()
        if path.parent != root or not path.is_file():
            raise RegistryTransitionError(
                "Registered package is missing or outside the registry directory"
            )
        package = load_model_package(path)
        if str(package.get("model_id")) != model_id:
            raise RegistryTransitionError(
                "Registered package model ID differs from registry"
            )
        recorded_hash = entry.get("package_sha256")
        if recorded_hash and recorded_hash != sha256_file(path):
            raise RegistryTransitionError("Registered package hash mismatch")
        return package, entry, path

    @staticmethod
    def _copy_immutable(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(source) != sha256_file(destination):
                raise FileExistsError(
                    f"Refusing to overwrite immutable registry file: {destination}"
                )
            return
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def register_approved_challenger(
        self,
        *,
        package_path: str | Path,
        metadata_path: str | Path,
        comparison_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Register passing bytes without activating or approving automation."""
        if comparison_summary.get("promotion_policy_passed") is not True:
            raise RegistryTransitionError(
                "Only a policy-passing challenger may be registered"
            )
        package_source = Path(package_path)
        metadata_source = Path(metadata_path)
        package = load_model_package(package_source)
        model_id = str(package["model_id"])
        self._require_learning_model(model_id)
        if comparison_summary.get("challenger_model_id") != model_id:
            raise RegistryTransitionError(
                "Comparison report challenger ID differs from package"
            )
        registry = self._read()
        existing = [
            entry
            for entry in registry["models"]
            if entry.get("model_id") == model_id
            or entry.get("package_filename") == package_source.name
        ]
        package_destination = self.model_directory / package_source.name
        metadata_destination = self.metadata_directory / metadata_source.name
        entry = {
            "package_filename": package_destination.name,
            "metadata_filename": metadata_destination.name,
            "package_sha256": sha256_file(package_source),
            "model_id": model_id,
            "model_version": str(package["model_version"]),
            "package_version": str(package["package_version"]),
            "creation_timestamp": str(package["training_timestamp"]),
            "training_dataset_hash": str(package["training_dataset_hash"]),
            "feature_generator_version": str(
                package["feature_generator_version"]
            ),
            "deployment_status": "ASSISTED_USE_ONLY",
            "approval_status": "APPROVED_FOR_ASSISTED_USE",
            "automatic_production_matching_approved": False,
            "notes": (
                "Passed Phase 7C champion–challenger policy. Explicit "
                "activation is still required; automatic production matching "
                "remains prohibited."
            ),
            "parent_model": str(package["parent_model"]),
            "comparison_id": str(comparison_summary["comparison_id"]),
        }
        if existing:
            if len(existing) != 1 or dict(existing[0]) != entry:
                raise RegistryTransitionError(
                    "Immutable approved registry entry differs"
                )
            return entry
        self._copy_immutable(package_source, package_destination)
        self._copy_immutable(metadata_source, metadata_destination)
        updated = dict(registry)
        updated["schema_version"] = "2.0"
        updated["models"] = sorted(
            [*registry["models"], entry],
            key=lambda item: (
                str(item.get("creation_timestamp")),
                str(item.get("model_id")),
            ),
        )
        try:
            atomic_json(updated, self.registry_path)
        except Exception:
            package_destination.unlink(missing_ok=True)
            metadata_destination.unlink(missing_ok=True)
            raise
        self.store.update_model_lifecycle(
            model_id=model_id,
            status="APPROVED_FOR_ASSISTED_USE",
            champion_status="APPROVED_CHALLENGER_NOT_ACTIVE",
            evaluation_summary=comparison_summary,
        )
        return entry

    def activate(
        self,
        *,
        model_id: str,
        expected_current_model_id: str,
        actor: str,
        reason: str,
    ) -> str | None:
        """Atomically switch the assisted pointer after explicit confirmation."""
        if not actor.strip() or not reason.strip():
            raise RegistryTransitionError(
                "Activation requires an actor and reason"
            )
        registry = self._read()
        target = self._entry(registry, model_id)
        self._require_learning_model(model_id)
        if target.get("approval_status") != "APPROVED_FOR_ASSISTED_USE":
            raise RegistryTransitionError(
                "Target model is not approved for assisted use"
            )
        current = registry.get("active_assisted_model_id")
        if current is None:
            # The pre-7C champion was selected explicitly by runtime config.
            self._entry(registry, expected_current_model_id)
            previous = expected_current_model_id
        else:
            previous = str(current)
            if previous != expected_current_model_id:
                raise RegistryTransitionError(
                    "Active model changed since the activation was prepared"
                )
        if model_id == previous:
            raise RegistryTransitionError("Target model is already active")
        timestamp = _now()
        updated = dict(registry)
        updated["schema_version"] = "2.0"
        updated["active_assisted_model_id"] = model_id
        updated["activation_history"] = [
            *registry.get("activation_history", []),
            {
                "activated_at": timestamp,
                "from_model_id": previous,
                "to_model_id": model_id,
                "actor": actor.strip(),
                "reason": reason.strip(),
                "action": "ACTIVATE",
            },
        ]
        atomic_json(updated, self.registry_path)
        try:
            self.store.update_model_lifecycle(
                model_id=model_id,
                status="ACTIVE_ASSISTED",
                champion_status="CHAMPION_ACTIVE",
                evaluation_summary={
                    "activation_reason": reason.strip(),
                    "previous_model_id": previous,
                },
                activated_at=timestamp,
            )
        except LearningStoreError:
            # Registry activation is authoritative; an older champion may only
            # have existed in the file registry. Never roll back the pointer
            # silently after its atomic commit.
            raise
        return previous

    def rollback(
        self,
        *,
        actor: str,
        reason: str,
    ) -> str:
        """Explicitly return to the immediately previous assisted champion."""
        if not actor.strip() or not reason.strip():
            raise RegistryTransitionError("Rollback requires an actor and reason")
        registry = self._read()
        current = registry.get("active_assisted_model_id")
        if not current:
            raise RegistryTransitionError("No active assisted model to roll back")
        history = list(registry.get("activation_history", []))
        previous = None
        for event in reversed(history):
            if (
                event.get("to_model_id") == current
                and event.get("from_model_id")
            ):
                previous = str(event["from_model_id"])
                break
        if previous is None:
            raise RegistryTransitionError("No rollback target is recorded")
        self._entry(registry, previous)
        timestamp = _now()
        updated = dict(registry)
        updated["active_assisted_model_id"] = previous
        updated["activation_history"] = [
            *history,
            {
                "activated_at": timestamp,
                "from_model_id": str(current),
                "to_model_id": previous,
                "actor": actor.strip(),
                "reason": reason.strip(),
                "action": "ROLLBACK",
            },
        ]
        atomic_json(updated, self.registry_path)
        try:
            self.store.update_model_lifecycle(
                model_id=str(current),
                status="RETIRED_ROLLBACK_AVAILABLE",
                champion_status="PREVIOUS_CHAMPION",
                evaluation_summary={"rollback_to": previous},
                retired_at=timestamp,
            )
            # Older champions observed before Phase 7C may not be in SQLite.
            self.store.update_model_lifecycle(
                model_id=previous,
                status="ACTIVE_ASSISTED",
                champion_status="CHAMPION_ACTIVE",
                evaluation_summary={"rollback_from": str(current)},
                activated_at=timestamp,
            )
        except LearningStoreError:
            pass
        return previous
