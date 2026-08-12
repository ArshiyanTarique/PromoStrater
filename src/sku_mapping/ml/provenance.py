"""Explicit provenance reliability policy for training sample weights."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.leakage import classify_provenance


def build_provenance_weights(
    frame: pd.DataFrame,
    configured_weights: Mapping[str, float],
) -> tuple[np.ndarray, pd.Series, dict[str, Any]]:
    """Return row weights and an auditable category distribution."""
    categories = (
        frame["provenance_category"].astype(str)
        if "provenance_category" in frame
        else pd.Series(
            [classify_provenance(row) for _, row in frame.iterrows()],
            index=frame.index,
            dtype="string",
        )
    )
    missing = sorted(set(categories) - set(configured_weights))
    if missing:
        raise ValueError(f"No configured provenance weight for categories: {missing}")
    weights = categories.map(
        {key: float(value) for key, value in configured_weights.items()}
    ).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Provenance sample weights must be finite and positive")
    report = {
        "policy": "multiplicative_sample_weight_with_balanced_class_weight",
        "model_feature_columns_unchanged": list(MODEL_FEATURE_COLUMNS),
        "provenance_is_model_feature": False,
        "configured_weights": {
            str(key): float(value)
            for key, value in sorted(configured_weights.items())
        },
        "categories": {
            str(category): {
                "rows": int((categories == category).sum()),
                "weight": float(configured_weights[str(category)]),
                "total_weight": float(weights[categories.to_numpy() == category].sum()),
            }
            for category in sorted(set(categories))
        },
    }
    return weights, categories, report
