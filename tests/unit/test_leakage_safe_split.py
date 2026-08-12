"""Leakage-safe split isolation and reproducibility tests."""

from __future__ import annotations

import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.leakage import build_leakage_groups
from sku_mapping.ml.leakage_split import (
    LeakageSplitConfig,
    assert_zero_leakage,
    create_leakage_safe_splits,
)


def _fixture(group_count: int = 45) -> pd.DataFrame:
    rows = []
    for group_index in range(group_count):
        for label in (0, 1):
            row = {
                "record_id": f"r-{group_index}-{label}",
                "offer_group_id": f"g-{group_index}",
                "offer_text": f"Unique family {group_index} variant {label} 400g",
                "master_itemcode": f"sku-{group_index}-{label}",
                "pair_label": label,
                "source_dataset": "REAL",
                "label_provenance": "human_audited",
            }
            row.update(
                {
                    feature: float(group_index * 10 + label)
                    for feature in MODEL_FEATURE_COLUMNS
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_leakage_safe_split_has_zero_overlap_and_no_dropped_rows() -> None:
    augmented = build_leakage_groups(_fixture()).frame
    result = create_leakage_safe_splits(
        augmented,
        LeakageSplitConfig(random_seed=13, candidate_splits=32),
    )
    frames = {
        "train": result.train,
        "validation": result.validation,
        "calibration": result.calibration,
    }
    assert_zero_leakage(frames)
    assert len(result.assignments) == len(augmented)
    assert result.assignments["input_row_number"].nunique() == len(augmented)
    assert set(result.assignments["split"]) == {
        "train",
        "validation",
        "calibration",
    }


def test_leakage_safe_split_regeneration_is_deterministic() -> None:
    augmented = build_leakage_groups(_fixture()).frame
    config = LeakageSplitConfig(random_seed=99, candidate_splits=32)
    first = create_leakage_safe_splits(augmented, config)
    second = create_leakage_safe_splits(augmented, config)
    assert first.assignment_sha256 == second.assignment_sha256
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
