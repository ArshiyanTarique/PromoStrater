"""Unit tests for governed training feature generation."""

from __future__ import annotations

import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.training import feature_builder
from sku_mapping.training.feature_builder import build_training_feature_dataset


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Itemcode": "001",
                "Itemname": "CHICKEN NUGGETS",
                "Item-Cat-2": "Chicken",
                "Item-Cat-4": "NUGGETS",
                "Item Description": "CHICKEN NUGGETS",
                "Item-Spec": "400g x 20 Pkts",
            },
            {
                "Itemcode": "BEEF",
                "Itemname": "BEEF BURGER",
                "Item-Cat-2": "Meat",
                "Item-Cat-4": "BURGERS",
                "Item Description": "BEEF BURGER",
                "Item-Spec": "500g x 20 Pkts",
            },
        ]
    )


def _gold(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "record_id": "r1",
        "source_dataset": "SYNTHETIC",
        "offer_group_id": "g1",
        "offer_text": "Al Kabeer Chicken Nuggets 400g",
        "master_itemcode": "001",
        "pair_label": 1,
        "use_for_binary_pair_training": 1,
        "label_provenance": "human",
        "label_confidence": 1.0,
        "recommended_split": "train",
        "split_group": "group-1",
        "product_class_offer": "Chicken Nuggets-Frozen",
        "variant_offer": "No Variant",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_valid_positive_and_negative_synthetic_pairs_use_gold_master_codes() -> None:
    result = build_training_feature_dataset(
        _gold(
            [
                {},
                {
                    "record_id": "r2",
                    "offer_group_id": "g2",
                    "master_itemcode": "BEEF",
                    "pair_label": 0,
                },
            ]
        ),
        _master(),
    )

    assert result.accepted["master_itemcode"].tolist() == ["001", "BEEF"]
    assert result.accepted["pair_label"].tolist() == [1, 0]
    assert result.manifest["class_distribution"] == {"0": 1, "1": 1}
    assert result.offer_reconstruction_counts == {"synthetic_or_text_fallback": 2}


def test_unknown_master_abstain_and_disabled_rows_are_rejected() -> None:
    result = build_training_feature_dataset(
        _gold(
            [
                {"record_id": "unknown", "master_itemcode": "DOES-NOT-EXIST"},
                {"record_id": "abstain", "pair_label": -1},
                {"record_id": "disabled", "use_for_binary_pair_training": 0},
            ]
        ),
        _master(),
    )

    assert result.accepted.empty
    reasons = dict(zip(result.rejected["record_id"], result.rejected["rejection_reason"]))
    assert "unknown_master_itemcode" in reasons["unknown"]
    assert "abstain_pair_label" in reasons["abstain"]
    assert "use_for_binary_pair_training_not_1" in reasons["disabled"]
    assert result.eligible_binary_rows == 1


def test_reliable_real_row_uses_rich_feature_api_and_ambiguous_row_falls_back(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def rich(offer_row, master_row):
        calls.append(("rich", str(offer_row["Product"])))
        return {name: 1 for name in MODEL_FEATURE_COLUMNS}

    def fallback(offer_text, master_row, product="", variant="", base_packsize=""):
        calls.append(("fallback", product))
        return {name: 2 for name in MODEL_FEATURE_COLUMNS}

    monkeypatch.setattr(feature_builder, "build_feature_vector", rich)
    monkeypatch.setattr(feature_builder, "build_feature_vector_from_text", fallback)
    clickflyer = pd.DataFrame(
        [
            {
                "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                "Product": "Chicken Nuggets-Frozen",
                "Variant": "No Variant",
                "Base Packsize": "400g",
            },
            {
                "Offer Name": "Ambiguous Offer",
                "Product": "Chicken Nuggets-Frozen",
                "Variant": "Regular",
                "Base Packsize": "400g",
            },
            {
                "Offer Name": "Ambiguous Offer",
                "Product": "Chicken Strips-Frozen",
                "Variant": "Spicy",
                "Base Packsize": "400g",
            },
        ]
    )
    gold = _gold(
        [
            {"source_dataset": "REAL"},
            {
                "record_id": "ambiguous",
                "offer_group_id": "g2",
                "offer_text": "Ambiguous Offer",
                "source_dataset": "REAL",
            },
        ]
    )

    result = build_training_feature_dataset(gold, _master(), clickflyer)

    assert calls == [
        ("rich", "Chicken Nuggets-Frozen"),
        ("fallback", "Chicken Nuggets-Frozen"),
    ]
    assert result.offer_reconstruction_counts == {
        "exact_clickflyer": 1,
        "synthetic_or_text_fallback": 1,
    }


def test_synthetic_row_uses_text_api_even_when_exact_flyer_text_exists(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        feature_builder,
        "build_feature_vector",
        lambda offer, master: calls.append("rich")
        or {name: 1 for name in MODEL_FEATURE_COLUMNS},
    )
    monkeypatch.setattr(
        feature_builder,
        "build_feature_vector_from_text",
        lambda *args, **kwargs: calls.append("text")
        or {name: 2 for name in MODEL_FEATURE_COLUMNS},
    )
    flyer = pd.DataFrame(
        [
            {
                "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                "Product": "Chicken Nuggets-Frozen",
                "Variant": "No Variant",
                "Base Packsize": "400g",
            }
        ]
    )

    result = build_training_feature_dataset(_gold([{}]), _master(), flyer)

    assert calls == ["text"]
    assert result.accepted.loc[0, MODEL_FEATURE_COLUMNS[0]] == 2


def test_feature_order_and_provenance_fields_are_preserved() -> None:
    result = build_training_feature_dataset(_gold([{}]), _master())
    metadata = [
        "record_id",
        "offer_group_id",
        "offer_text",
        "master_itemcode",
        "pair_label",
        "source_dataset",
        "label_provenance",
        "label_confidence",
        "recommended_split",
        "split_group",
        "product_class_offer",
        "variant_offer",
    ]

    assert result.accepted.columns.tolist() == metadata + list(MODEL_FEATURE_COLUMNS)
    row = result.accepted.iloc[0]
    assert row["source_dataset"] == "SYNTHETIC"
    assert row["label_provenance"] == "human"
    assert row["split_group"] == "group-1"
    assert row["master_itemcode"] == "001"


def test_duplicate_and_conflicting_pairs_are_audited_but_retained() -> None:
    result = build_training_feature_dataset(
        _gold([{}, {"record_id": "r2", "pair_label": 0}]),
        _master(),
    )

    assert len(result.accepted) == 2
    assert result.audit["duplicate_offer_master_pairs"]["group_count"] == 1
    assert result.audit["conflicting_labels"]["pair_count"] == 1
