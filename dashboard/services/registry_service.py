"""Read-only, registry-constrained model selection for the dashboard."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sku_mapping.config import PipelineConfig
from sku_mapping.shadow.predictor import load_registered_shadow_package

_VERSION_NUMBER = re.compile(r"v(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DashboardModelOption:
    """Safe model metadata suitable for display in a selector."""

    model_id: str
    model_version: str
    deployment_status: str
    approval_status: str
    package_filename: str
    display_label: str


def _metadata_document(
    config: PipelineConfig, package_filename: str
) -> dict[str, Any]:
    path = (
        config.shadow_mode.registry_path.parent
        / "metadata"
        / f"{Path(package_filename).stem}.json"
    )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def build_display_label(
    entry: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    """Return an operator-readable model name for selectors.

    For example ``Version 5 Ranked-Calibrated (2026-08-10)``. Whether a
    package is calibrated is taken from its recorded ``calibration_method``
    rather than from its name, so a package that never ran the calibration
    stage cannot present itself as calibrated.
    """
    version_text = str(entry.get("model_version") or "")
    match = _VERSION_NUMBER.search(version_text) or _VERSION_NUMBER.search(
        str(entry.get("model_id") or "")
    )
    label = (
        f"Version {match.group(1)}"
        if match
        else (version_text or "Unnamed model")
    )
    if "ranked" in version_text.lower():
        label = f"{label} Ranked"
    label += (
        "-Calibrated" if metadata.get("calibration_method") else "-Uncalibrated"
    )
    created = str(
        entry.get("creation_timestamp")
        or metadata.get("training_timestamp")
        or ""
    )[:10]
    return f"{label} ({created})" if created else label


class DashboardRegistryService:
    """Expose only explicitly registered and structurally safe packages."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def list_models(self) -> list[DashboardModelOption]:
        path = self.config.shadow_mode.registry_path
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Model registry is unavailable") from error
        models = []
        for entry in registry.get("models", []):
            if (
                entry.get("deployment_status") == "SHADOW_MODE_ONLY"
                and entry.get("automatic_production_matching_approved")
                is False
                and entry.get("approval_status")
                == "NOT_APPROVED_FOR_AUTOMATIC_MATCHING"
            ):
                package_filename = str(entry["package_filename"])
                models.append(
                    DashboardModelOption(
                        model_id=str(entry["model_id"]),
                        model_version=str(entry.get("model_version") or ""),
                        deployment_status=str(
                            entry["deployment_status"]
                        ),
                        approval_status=str(entry["approval_status"]),
                        package_filename=package_filename,
                        display_label=build_display_label(
                            entry,
                            _metadata_document(self.config, package_filename),
                        ),
                    )
                )
        return sorted(models, key=lambda item: item.model_id)

    def validate_model_id(self, model_id: str) -> DashboardModelOption:
        options = {item.model_id: item for item in self.list_models()}
        option = options.get(model_id)
        if option is None:
            raise ValueError("Selected model ID is not in the safe registry")
        model_directory = (
            self.config.shadow_mode.registry_path.parent / "registry"
        )
        load_registered_shadow_package(
            registry_path=self.config.shadow_mode.registry_path,
            model_directory=model_directory,
            model_id=model_id,
            require_package_status="SHADOW_MODE_ONLY",
        )
        return option

