"""Group-aware split safety and reproducibility tests."""

from __future__ import annotations

import pandas as pd
import pytest

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.trainer import TrainingConfig, create_group_splits


def _feature_table(group_count: int = 30) -> pd.DataFrame:
    rows = []
    for group_index in range(group_count):
        for label in (0, 1):
            row = {
                "offer_group_id": f"group-{group_index}",
                "pair_label": label,
                "recommended_split": "",
            }
            row.update(
                {
                    feature: float(group_index + label)
                    for feature in MODEL_FEATURE_COLUMNS
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_group_split_has_no_offer_group_leakage() -> None:
    splits = create_group_splits(
        _feature_table(), TrainingConfig(random_seed=17)
    )
    groups = [
        set(getattr(splits, name)["offer_group_id"])
        for name in ("train", "validation", "test")
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_group_split_is_reproducible() -> None:
    frame = _feature_table()
    first = create_group_splits(frame, TrainingConfig(random_seed=123))
    second = create_group_splits(frame, TrainingConfig(random_seed=123))
    for name in ("train", "validation", "test"):
        assert getattr(first, name).index.tolist() == getattr(second, name).index.tolist()


def test_complete_valid_recommended_split_is_used() -> None:
    frame = _feature_table(6)
    allocation = {
        "group-0": "train",
        "group-1": "train",
        "group-2": "validation",
        "group-3": "validation",
        "group-4": "test",
        "group-5": "test",
    }
    frame["recommended_split"] = frame["offer_group_id"].map(allocation)
    splits = create_group_splits(frame)
    assert splits.method == "recommended_split"
    assert set(splits.validation["offer_group_id"]) == {"group-2", "group-3"}


def test_missing_feature_columns_fail_before_splitting() -> None:
    frame = _feature_table().drop(columns=[MODEL_FEATURE_COLUMNS[-1]])
    with pytest.raises(ValueError, match="missing columns"):
        create_group_splits(frame)
