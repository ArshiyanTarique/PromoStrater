"""Tests for shared preprocessing copied from the production pipeline."""

import pandas as pd

from sku_mapping.data.preprocessing import (
    categorize,
    normalize_brand,
    normalize_product_family,
    preprocess_clickflyer,
    preprocess_product_master,
)


def test_legacy_brand_category_and_family_normalization() -> None:
    assert normalize_brand(" Al-Kabeer Foods! ") == "al kabeer foods"
    assert categorize("chicken nuggets") == "Chicken"
    assert categorize("corned beef") == "Meat"
    assert normalize_product_family("Chicken Nugget-Frozen") == "chicken nuggets"


def test_clickflyer_preprocessing_builds_legacy_fields_without_mutating_input() -> None:
    original = pd.DataFrame(
        [{
            "Offer Name": "Al Kabeer Chicken Nuggets 2 x 500g",
            "Product": "Chicken Nugget-Frozen",
            "Brand Name": "Al-Kabeer",
            "Variant": "No Variant",
            "Base Packsize": "2 x 500g",
        }]
    )
    prepared = preprocess_clickflyer(original)
    assert "match_text" not in original.columns
    assert prepared.loc[0, "is_own"]
    assert prepared.loc[0, "category"] == "Chicken"
    assert prepared.loc[0, "product_family"] == "chicken nuggets"
    assert (1000.0, "weight", 2) in prepared.loc[0, "offer_measures_detailed"]
    assert (500.0, "weight") in prepared.loc[0, "offer_measures"]


def test_product_master_preprocessing_never_uses_carton_total_as_retail_pack() -> None:
    master = pd.DataFrame(
        [{
            "Itemcode": " 001 ",
            "Itemname": "CHICKEN NUGGETS",
            "Item-Cat-2": "Chicken",
            "Item-Cat-4": "Nuggets",
            "Item Description": "Chicken nuggets",
            "Item-Spec": "270 Gms x 20 Pkts",
        }]
    )
    prepared = preprocess_product_master(master)
    assert prepared.loc[0, "Itemcode"] == "001"
    assert prepared.loc[0, "master_measures_detailed"] == [(270.0, "weight", 1)]
    assert (5400.0, "weight") not in prepared.loc[0, "master_measures"]
