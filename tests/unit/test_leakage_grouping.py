"""Canonical template and transitive leakage-group tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.leakage import (
    build_leakage_groups,
    feature_vector_hash,
    normalize_offer_template,
)


def _row(
    record_id: str,
    offer_group_id: str,
    offer_text: str,
    feature_value: float,
    *,
    provenance: str = "v2_synthetic_by_construction",
) -> dict[str, object]:
    row: dict[str, object] = {
        "record_id": record_id,
        "offer_group_id": offer_group_id,
        "offer_text": offer_text,
        "master_itemcode": f"sku-{record_id}",
        "pair_label": int(feature_value > 0),
        "source_dataset": "GOLD_V2",
        "label_provenance": provenance,
    }
    row.update({feature: feature_value for feature in MODEL_FEATURE_COLUMNS})
    return row


def test_template_normalization_handles_units_multipliers_and_bonus() -> None:
    left = normalize_offer_template(
        "Al-Kabeer Spicy Chicken Nuggets 2 × 500 G + 100 gm"
    )
    right = normalize_offer_template(
        "al kabeer spicy chicken nuggets 2x500g+100 grams"
    )
    assert left == right
    assert "chicken" in left
    assert "nuggets" in left
    assert "spicy" in left
    assert "bonus plus" in left
    assert normalize_offer_template("Soup 1 L") != normalize_offer_template(
        "Soup 1 kg"
    )


def test_feature_hash_is_ordered_stable_and_excludes_labels() -> None:
    row = {feature: float(index) for index, feature in enumerate(MODEL_FEATURE_COLUMNS)}
    row["bonus_weight_g"] = np.nan
    first = feature_vector_hash(row)
    second = feature_vector_hash({**row, "pair_label": 1, "source_dataset": "x"})
    changed = dict(row)
    changed[MODEL_FEATURE_COLUMNS[0]] = 99.0
    assert first == second
    assert first != feature_vector_hash(changed)


def test_connected_components_apply_relationships_transitively() -> None:
    rows = [
        _row("a", "offer-ab", "Chicken burger 400g", 1.0),
        _row("b", "offer-ab", "Chicken nuggets 400g", 2.0),
        _row("c", "offer-c", "Chicken nuggets 750gm", 3.0),
        _row("d", "offer-d", "Beef kofta 1kg", 3.0),
        _row("e", "offer-e", "Fish fingers 500g", 5.0),
    ]
    result = build_leakage_groups(pd.DataFrame(rows))
    component_ids = result.frame.set_index("record_id")["leakage_group_id"]
    assert len({component_ids["a"], component_ids["b"], component_ids["c"], component_ids["d"]}) == 1
    assert component_ids["e"] != component_ids["a"]


def test_real_rows_do_not_share_broad_template_ids() -> None:
    rows = [
        _row(
            "real-a",
            "real-offer-a",
            "Chicken nuggets 400g",
            1.0,
            provenance="real_flyer_catalogue_forced",
        ),
        _row(
            "real-b",
            "real-offer-b",
            "Chicken nuggets 750g",
            2.0,
            provenance="real_flyer_catalogue_forced",
        ),
    ]
    result = build_leakage_groups(pd.DataFrame(rows))
    assert result.frame["template_group_id"].nunique() == 2
