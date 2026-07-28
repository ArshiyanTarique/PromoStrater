"""Tests for semantic feature extraction."""

from sku_mapping.features.semantic_features import (
    _compatibility_flag,
    _expected_match_count,
    _family_set,
    _protein_set,
    _variant_set,
)


def test_protein_detection_and_compatibility() -> None:
    chicken = _protein_set("Spicy chicken nuggets")
    beef = _protein_set("Beef burger")
    assert chicken == {"chicken"}
    assert beef == {"beef"}
    assert _compatibility_flag(chicken, chicken) == 1
    assert _compatibility_flag(chicken, beef) == 0


def test_family_and_variant_detection_preserve_legacy_overlaps() -> None:
    assert _family_set("Spicy chicken nuggets") == {"chicken nuggets", "nuggets"}
    assert _variant_set("Non-spicy chicken") == {"non spicy", "spicy"}


def test_missing_semantics_are_compatible() -> None:
    assert _compatibility_flag(set(), {"chicken"}) == 1


def test_expected_match_count_uses_spaced_separators() -> None:
    assert _expected_match_count("Chicken nuggets / beef burger") == 2.0
    assert _expected_match_count("750g+250g") == 1.0
