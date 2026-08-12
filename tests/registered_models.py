"""Resolve a live registered model id for tests.

Tests used to hardcode a specific registered model id. That made them fail
the moment a superseded package was pruned from models/, which is a routine
housekeeping action rather than a real regression. Resolving the id from the
registry keeps the tests exercising a package that actually exists, and
prefers the configured champion so they exercise what the project ships.
"""

from __future__ import annotations

import json
from pathlib import Path

from sku_mapping.config import load_config

REGISTRY_PATH = Path("models/model_registry.json")


def registered_model_ids() -> list[str]:
    """Return every shadow-safe model id currently in the registry."""
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [
        str(entry["model_id"])
        for entry in registry.get("models", [])
        if entry.get("deployment_status") == "SHADOW_MODE_ONLY"
        and entry.get("automatic_production_matching_approved") is False
    ]


def registered_model_id() -> str:
    """Return the configured champion, or any registered model as fallback."""
    available = registered_model_ids()
    if not available:
        raise RuntimeError(
            "No registered shadow model is available; tests need at least one "
            "entry in models/model_registry.json"
        )
    configured = str(load_config("config/default.yaml").ml.model_id or "")
    return configured if configured in available else available[0]
