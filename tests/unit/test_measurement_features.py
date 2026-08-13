"""Tests for source-specific measurement behavior."""

import math

from sku_mapping.features.measurement_features import (
    _extract_bonus_weight,
    _extract_piece_count,
    _first_weight_value,
    _master_units_per_carton,
    _offer_total_weight,
    _offer_unit_count,
    collapse_to_simple,
    extract_flyer_measures,
    extract_master_measures,
    pack_is_compatible,
    pack_structure_agrees,
    unit_dim_value,
)


def test_unit_dimensions_keep_weight_and_volume_separate() -> None:
    assert unit_dim_value("kg") == ("weight", 1000.0)
    assert unit_dim_value("l") == ("volume", 1000.0)
    assert (
        pack_is_compatible(
            collapse_to_simple(extract_flyer_measures("1L")),
            collapse_to_simple(extract_master_measures("1kg")),
        )
        is False
    )


def test_matching_and_mismatching_pack_weights() -> None:
    offer = collapse_to_simple(extract_flyer_measures("400g"))
    matching = collapse_to_simple(extract_master_measures("400 Gms x 20 Pkts"))
    mismatching = collapse_to_simple(extract_master_measures("1 kg x 10 Pkts"))
    assert pack_is_compatible(offer, matching) is True
    assert pack_is_compatible(offer, mismatching) is False


def test_flyer_multipack_preserves_total_and_unit_size() -> None:
    details = extract_flyer_measures("2 x 500g")
    assert (500.0, "weight", 1) in details
    assert (1000.0, "weight", 2) in details
    assert _first_weight_value(details) == 500.0
    assert _offer_unit_count(details) == 2.0
    assert _offer_total_weight(details) == 1000.0


def test_master_case_excludes_carton_total() -> None:
    details = extract_master_measures("270 Gms x 20 Pkts")
    assert details == [(270.0, "weight", 1)]
    assert _master_units_per_carton("270 Gms x 20 Pkts") == 20.0
    assert all(value != 5400.0 for value, _, _ in details)


def test_compact_twin_pack_master_spec_keeps_unit_and_outer_carton_count() -> None:
    details = extract_master_measures("400 Gmsx2Pktx6")
    assert details == [(400.0, "weight", 1)]
    assert _master_units_per_carton("400 Gmsx2Pktx6") == 6.0


def test_bonus_and_piece_count_extraction() -> None:
    assert _extract_bonus_weight("Chicken Nuggets 750g + 250g") == 250.0
    assert _extract_piece_count("Chicken Nuggets 20 pcs") == 20.0
    assert _extract_piece_count("Chicken Nuggets x 20") == 20.0
    assert math.isnan(_extract_piece_count("Chicken Nuggets 2 x 500g"))


def test_missing_measurements_are_unknown() -> None:
    assert extract_flyer_measures(None) == []
    assert pack_is_compatible([], []) is None
    assert pack_structure_agrees([], []) is None
    assert math.isnan(_first_weight_value([]))


def test_structure_detects_twin_pack_against_plain_one_kilogram() -> None:
    offer = extract_flyer_measures("2 x 500g")
    master = extract_master_measures("1kg")
    assert pack_is_compatible(collapse_to_simple(offer), collapse_to_simple(master)) is True
    assert pack_structure_agrees(offer, master) is False


def test_master_gram_spelling_is_recognised() -> None:
    """The master data spells grams "GRM"; the flyers spell it "gm".

    While "grm" was unknown the master side parsed to nothing, so
    ``pack_is_compatible`` answered None rather than False and no size
    conflict could be raised for that SKU at all - a 550gm competitor sat
    unchallenged under a 400 GRM Al Kabeer SKU.
    """
    assert collapse_to_simple(extract_master_measures("400 GRM")) == [
        (400.0, "weight")
    ]
    assert collapse_to_simple(
        extract_master_measures("12 PKTS X 400 GRM")
    ) == [(400.0, "weight")]

    master = collapse_to_simple(extract_master_measures("12 PKTS X 400 GRM"))
    oversized = collapse_to_simple(
        extract_flyer_measures("Cucina Tempura Chicken Fries 550gm")
    )
    matching = collapse_to_simple(
        extract_flyer_measures("Americana Chicken Fries 400 gm")
    )
    assert pack_is_compatible(oversized, master) is False
    assert pack_is_compatible(matching, master) is True


def test_gram_spellings_agree_on_one_value() -> None:
    for spelling in ("400 g", "400 gm", "400 gms", "400 grm", "400 grams"):
        assert collapse_to_simple(extract_master_measures(spelling)) == [
            (400.0, "weight")
        ]
