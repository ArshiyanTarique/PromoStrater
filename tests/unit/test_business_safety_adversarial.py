"""Adversarial regression tests for known business-safety conditions."""

from __future__ import annotations

import math

from sku_mapping.features import build_feature_vector_from_text
from sku_mapping.features.measurement_features import (
    _extract_bonus_weight,
    _extract_piece_count,
    collapse_to_simple,
    extract_flyer_measures,
    extract_master_measures,
    pack_is_compatible,
    pack_structure_agrees,
)


def _master(spec: str, text: str = "CHICKEN NUGGETS") -> dict[str, str]:
    return {
        "Itemname": text,
        "Item-Cat-4": text,
        "Item Description": text,
        "Item-Spec": spec,
    }


def test_mixed_protein_and_pack_format_conflict_remain_explicit() -> None:
    mixed = build_feature_vector_from_text(
        "Chicken and beef burgers 400g", _master("400g", "CHICKEN BURGER")
    )
    conflict = build_feature_vector_from_text(
        "Chicken nuggets 2 x 500g", _master("1kg")
    )
    assert mixed["is_mixed_protein_offer"] == 1
    assert conflict["size_match"] == 1
    assert conflict["pack_format_match"] == 0


def test_missing_offer_and_master_weights_remain_nan() -> None:
    missing_offer = build_feature_vector_from_text(
        "Chicken nuggets", _master("400g")
    )
    missing_master = build_feature_vector_from_text(
        "Chicken nuggets 400g", _master("")
    )
    assert math.isnan(float(missing_offer["unit_pack_weight_g"]))
    assert math.isnan(float(missing_master["master_unit_weight_g"]))


def test_bonus_piece_and_weight_multiplier_are_not_conflated() -> None:
    assert _extract_bonus_weight("750g + 250g bonus pack") == 250.0
    assert _extract_piece_count("Chicken nuggets 20 pieces") == 20.0
    assert math.isnan(_extract_piece_count("Chicken nuggets 2 x 500g"))


def test_litre_never_matches_kilogram() -> None:
    assert (
        pack_is_compatible(
            collapse_to_simple(extract_flyer_measures("1 L")),
            collapse_to_simple(extract_master_measures("1 kg")),
        )
        is False
    )


def test_carton_quantity_is_not_retail_pack_quantity() -> None:
    master = extract_master_measures("270 Gms x 20 Pkts")
    offer = extract_flyer_measures("5.4kg")
    assert all(value != 5400.0 for value, _, _ in master)
    assert pack_is_compatible(
        collapse_to_simple(offer), collapse_to_simple(master)
    ) is False
    assert pack_structure_agrees(offer, master) is None


def test_legacy_400_plus_60_case_rule_remains_documented_not_reinterpreted() -> None:
    # Phase 5C deliberately does not invent an authoritative interpretation
    # for this ambiguous master expression.
    details = extract_master_measures("400+60 G X 20 Pkts")
    assert details == [(60.0, "weight", 1)]
