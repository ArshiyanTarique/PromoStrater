"""Provenance reliability affects weights, never model inputs."""

from __future__ import annotations

import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.provenance import build_provenance_weights


def test_provenance_changes_sample_weight_but_not_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "provenance_category": ["human_audited", "synthetic", "rule_generated"],
            **{
                feature: [1.0, 1.0, 1.0]
                for feature in MODEL_FEATURE_COLUMNS
            },
        }
    )
    weights, categories, report = build_provenance_weights(
        frame,
        {
            "human_audited": 1.5,
            "synthetic": 0.6,
            "rule_generated": 0.5,
        },
    )
    assert weights.tolist() == [1.5, 0.6, 0.5]
    assert categories.tolist() == [
        "human_audited",
        "synthetic",
        "rule_generated",
    ]
    assert "provenance_category" not in MODEL_FEATURE_COLUMNS
    assert report["provenance_is_model_feature"] is False
