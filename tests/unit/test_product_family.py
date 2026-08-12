"""Guards for the product-family alias table.

The table widens competitor-discovery buckets, so its failure mode is
inventing competitors that do not exist. These tests pin the two
properties that keep that from happening: variants of one dish merge, and
different products never do.
"""

from __future__ import annotations

import pytest

from sku_mapping.features.product_family import (
    MUST_STAY_DISTINCT,
    PRODUCT_TOKEN_ALIASES,
    audit_alias_collisions,
    normalize_product_family,
)

LEGACY_ALIASES = {
    "nugget": "nuggets", "burger": "burgers", "patty": "patties",
    "sausage": "sausages", "strip": "strips", "wing": "wings",
    "prawn": "prawns", "shrimp": "shrimps", "paratha": "parathas",
    "samosa": "samosas", "roll": "rolls",
}


@pytest.mark.parametrize("left,right", MUST_STAY_DISTINCT)
def test_different_products_never_collapse(left, right):
    assert normalize_product_family(left) != normalize_product_family(right)


@pytest.mark.parametrize(
    "left,right",
    [
        ("Chicken Sambosa", "Chicken Samosa"),
        ("Beef Kubbe", "Beef Kibbeh"),
        ("Seekh Kabab", "Seekh Kebabs"),
        ("Chicken Shawerma", "Chicken Shawarma"),
        ("Veg Felafel", "Veg Falafel"),
        ("Shish Tawouk", "Sheesh Tawook"),
        ("White Fish Fillet", "White Fish Fillets"),
        ("Chicken Tender", "Chicken Tenders"),
        ("Chapati Plain", "Chappati Plain"),
        ("Meat Kufta", "Meat Kofta"),
    ],
)
def test_spelling_variants_of_one_dish_merge(left, right):
    assert normalize_product_family(left) == normalize_product_family(right)


def test_kibbeh_and_kebab_stay_separate_dishes():
    """Both are transliterated from Arabic but are different products."""
    assert normalize_product_family("kubba") == "kibbeh"
    assert normalize_product_family("kabab") == "kebabs"
    assert normalize_product_family("kubba") != normalize_product_family("kabab")


def test_frozen_marker_is_stripped_regardless_of_separator():
    for raw in ("Chicken Nuggets Frozen", "Chicken Nuggets-Frozen",
                "Chicken Nuggets_Frozen"):
        assert normalize_product_family(raw) == "chicken nuggets"


def test_legacy_aliases_are_preserved():
    """The original 11 mappings must keep their exact behaviour."""
    for token, canonical in LEGACY_ALIASES.items():
        assert PRODUCT_TOKEN_ALIASES[token] == canonical


def test_no_alias_chains_to_another_alias():
    """A -> B -> C would make the result depend on iteration order."""
    chained = {
        key: value for key, value in PRODUCT_TOKEN_ALIASES.items()
        if value in PRODUCT_TOKEN_ALIASES
        and PRODUCT_TOKEN_ALIASES[value] != value
    }
    assert chained == {}


def test_normalisation_is_idempotent():
    for canonical in set(PRODUCT_TOKEN_ALIASES.values()):
        assert normalize_product_family(canonical) == canonical


def test_audit_reports_merges_it_performs():
    collisions = audit_alias_collisions(
        ["Beef Kibbeh", "Beef Kubbe", "Chicken Nuggets"]
    )
    assert len(collisions) == 1
    assert collisions[0]["canonical"] == "beef kibbeh"
    assert collisions[0]["merged"] == ["beef kibbeh", "beef kubbe"]


def test_audit_is_silent_when_nothing_merges():
    assert audit_alias_collisions(["Chicken Nuggets", "Beef Burger"]) == []


def test_audit_catches_an_unsafe_table():
    """The audit must surface a bad alias, not just good ones."""
    unsafe = {**PRODUCT_TOKEN_ALIASES, "fingers": "fries"}
    collisions = audit_alias_collisions(
        ["Chicken Fingers", "Chicken Fries"], unsafe
    )
    assert collisions and collisions[0]["merged"] == [
        "chicken fingers", "chicken fries"
    ]


def test_empty_and_missing_values_are_safe():
    assert normalize_product_family("") == ""
    assert normalize_product_family(None) == "none"
    assert audit_alias_collisions(["", "   "]) == []
