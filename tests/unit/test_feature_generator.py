"""Feature-vector contract, behavior, and legacy parity tests."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.features import build_feature_vector, build_feature_vector_from_text
from sku_mapping.features.measurement_features import (
    extract_flyer_measures,
    extract_master_measures,
)

EXPECTED_COLUMNS = [
    "protein_match",
    "family_match",
    "variant_match",
    "size_match",
    "pack_format_match",
    "word_similarity",
    "character_similarity",
    "token_similarity",
    "unit_pack_weight_g",
    "number_of_units",
    "bonus_weight_g",
    "total_offer_weight_g",
    "piece_count",
    "master_unit_weight_g",
    "master_units_per_carton",
    "is_mixed_protein_offer",
    "is_multi_family_offer",
    "contains_non_meat_product",
    "expected_match_count",
]


def _offer(
    name: str,
    product: str = "Chicken Nuggets-Frozen",
    variant: str = "No Variant",
    pack: str = "400g",
) -> dict[str, Any]:
    return {
        "Offer Name": name,
        "Product": product,
        "Variant": variant,
        "Base Packsize": pack,
        "offer_measures_detailed": extract_flyer_measures(f"{pack} {name}"),
    }


def _master(
    name: str,
    category: str,
    description: str,
    spec: str,
) -> dict[str, Any]:
    return {
        "Itemname": name,
        "Item-Cat-4": category,
        "Item Description": description,
        "Item-Spec": spec,
        "master_measures_detailed": extract_master_measures(spec),
    }


def _assert_values_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert list(actual) == list(expected)
    for key in actual:
        actual_value = actual[key]
        expected_value = expected[key]
        if isinstance(actual_value, (float, np.floating)) and math.isnan(actual_value):
            assert isinstance(expected_value, (float, np.floating))
            assert math.isnan(expected_value), key
        else:
            assert actual_value == expected_value, key


def _load_legacy_feature_namespace() -> dict[str, Any]:
    """Compile only the legacy feature definitions, never module-level pipeline code."""
    source_path = Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    assignments = {
        "FILLER_WORDS",
        "WEIGHT_UNITS",
        "VOLUME_UNITS",
        "ALL_UNITS",
        "PROTEIN_WORDS",
        "NON_MEAT_WORDS",
        "FAMILY_PHRASES",
        "FAMILY_CONCEPT_ALIASES",
        "VARIANT_WORDS",
    }
    functions = {
        "clean_offer_text",
        "unit_dim_value",
        "pack_is_compatible",
        "extract_measures_detailed",
        "extract_flyer_measures",
        "extract_master_measures",
        "collapse_to_simple",
        "pack_structure_agrees",
        "_first_weight_value",
        "_offer_unit_count",
        "_offer_total_weight",
        "_extract_piece_count",
        "_extract_bonus_weight",
        "_master_units_per_carton",
        "_protein_set",
        "_family_set",
        "_family_concept_set",
        "_resolve_semantic_variant",
        "_variant_set",
        "_compatibility_flag",
        "_expected_match_count",
        "_build_ml_feature_row",
    }
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if target_names & assignments:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {"pd": pd, "np": np, "re": re, "fuzz": fuzz}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


def test_model_feature_columns_are_exact_and_ordered() -> None:
    assert MODEL_FEATURE_COLUMNS == EXPECTED_COLUMNS
    assert len(MODEL_FEATURE_COLUMNS) == 19


def test_chicken_offer_matches_chicken_but_not_beef() -> None:
    offer = _offer("Al Kabeer Chicken Nuggets 400g")
    chicken = _master("CHICKEN NUGGETS", "CHICKEN", "Chicken nuggets", "400 Gms x 20 Pkts")
    beef = _master("BEEF BURGER", "MEAT", "Beef burger", "400 Gms x 20 Pkts")
    assert build_feature_vector(offer, chicken)["protein_match"] == 1
    assert build_feature_vector(offer, beef)["protein_match"] == 0


def test_size_match_for_400g_and_mismatch_for_1kg() -> None:
    offer = _offer("Al Kabeer Chicken Nuggets 400g")
    matching = _master("CHICKEN NUGGETS", "CHICKEN", "", "400 Gms x 20 Pkts")
    mismatching = _master("CHICKEN NUGGETS", "CHICKEN", "", "1 kg x 10 Pkts")
    assert build_feature_vector(offer, matching)["size_match"] == 1
    assert build_feature_vector(offer, mismatching)["size_match"] == 0


def test_multipack_bonus_piece_and_master_case_features() -> None:
    offer = _offer(
        "Al Kabeer Chicken Nuggets 20 pcs 750g + 250g",
        pack="2 x 500g",
    )
    master = _master("CHICKEN NUGGETS", "CHICKEN", "", "270 Gms x 20 Pkts")
    features = build_feature_vector(offer, master)
    assert features["unit_pack_weight_g"] == 250.0
    assert features["number_of_units"] == 2.0
    assert features["bonus_weight_g"] == 250.0
    assert features["total_offer_weight_g"] == 1000.0
    assert features["piece_count"] == 20.0
    assert features["master_unit_weight_g"] == 270.0
    assert features["master_units_per_carton"] == 20.0


def test_synthetic_offer_generation_and_mapping_compatible_series() -> None:
    master = pd.Series(
        {
            "Itemname": "CHICKEN NUGGETS",
            "Item-Cat-4": "CHICKEN",
            "Item Description": "Chicken nuggets",
            "Item-Spec": "400 Gms x 20 Pkts",
        }
    )
    features = build_feature_vector_from_text(
        "Al Kabeer Chicken Nuggets",
        master,
        product="Chicken Nuggets-Frozen",
        base_packsize="400g",
    )
    assert features["protein_match"] == 1
    assert features["size_match"] == 1
    assert features["unit_pack_weight_g"] == 400.0


def test_missing_measurements_and_weight_volume_separation() -> None:
    missing = build_feature_vector(
        {"Offer Name": "Chicken Nuggets", "Product": "Chicken Nuggets"},
        {"Itemname": "CHICKEN NUGGETS", "Item-Cat-4": "CHICKEN"},
    )
    assert missing["size_match"] == 0
    assert missing["pack_format_match"] == 1
    assert math.isnan(float(missing["unit_pack_weight_g"]))

    volume = build_feature_vector_from_text(
        "Chicken Soup 1L",
        {
            "Itemname": "CHICKEN SOUP",
            "Item-Cat-4": "CHICKEN",
            "Item Description": "",
            "Item-Spec": "1kg",
        },
    )
    assert volume["size_match"] == 0


def test_mixed_protein_offer() -> None:
    features = build_feature_vector_from_text(
        "Chicken and beef burger 400g",
        {
            "Itemname": "CHICKEN BURGER",
            "Item-Cat-4": "CHICKEN",
            "Item Description": "",
            "Item-Spec": "400g",
        },
    )
    assert features["is_mixed_protein_offer"] == 1
    assert features["expected_match_count"] == 2.0


def test_contradictory_source_variant_cannot_override_offer_protein() -> None:
    offer = _offer(
        "Al Kabeer Chicken Samosas 240 gm",
        product="Samosa-Frozen",
        variant="Mutton",
        pack="240 gm",
    )
    chicken = _master(
        "12 CHICKEN SAMOSAS",
        "SAMOSA",
        "Chicken samosas",
        "240 Gms x 20 Pkts",
    )
    mutton = _master(
        "12 MUTTON SAMOSAS",
        "SAMOSA",
        "Mutton samosas",
        "240 Gms x 20 Pkts",
    )

    chicken_features = build_feature_vector(offer, chicken)
    mutton_features = build_feature_vector(offer, mutton)

    assert chicken_features["protein_match"] == 1
    assert mutton_features["protein_match"] == 0
    assert chicken_features["is_mixed_protein_offer"] == 0


def test_nested_family_aliases_are_one_product_family() -> None:
    features = build_feature_vector_from_text(
        "Al Kabeer Chicken Nuggets 400 gm",
        _master(
            "CHICKEN NUGGETS (400G)",
            "NUGGETS",
            "Chicken nuggets",
            "400 Gms x 12 Pkts",
        ),
        product="Chicken Nuggets-Frozen",
        base_packsize="400 gm",
    )
    assert features["family_match"] == 1
    assert features["is_multi_family_offer"] == 0


def test_all_feature_values_are_numeric_or_nan() -> None:
    features = build_feature_vector_from_text(
        "Chicken Nuggets",
        {"Itemname": "CHICKEN NUGGETS"},
    )
    assert list(features) == MODEL_FEATURE_COLUMNS
    assert all(
        isinstance(value, (int, float, np.integer, np.floating)) for value in features.values()
    )


def test_pandas_na_does_not_become_feature_text() -> None:
    features = build_feature_vector(
        pd.Series(
            {
                "Offer Name": pd.NA,
                "Product": "Chicken Nuggets",
                "Variant": pd.NA,
                "Base Packsize": pd.NA,
            }
        ),
        pd.Series(
            {
                "Itemname": "CHICKEN NUGGETS",
                "Item-Cat-4": pd.NA,
                "Item Description": pd.NA,
                "Item-Spec": pd.NA,
            }
        ),
    )
    assert list(features) == MODEL_FEATURE_COLUMNS
    assert all(
        isinstance(value, (int, float, np.integer, np.floating)) for value in features.values()
    )


def test_exact_parity_with_authoritative_legacy_implementation() -> None:
    legacy = _load_legacy_feature_namespace()
    pairs = [
        (
            {
                "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                "Product": "Chicken Nuggets-Frozen",
                "Variant": "Spicy",
                "offer_measures_detailed": legacy["extract_flyer_measures"]("400g"),
            },
            {
                "Itemname": "CHICKEN NUGGETS",
                "Item-Cat-4": "CHICKEN",
                "Item Description": "Spicy chicken nuggets",
                "Item-Spec": "400 Gms x 20 Pkts",
                "master_measures_detailed": legacy["extract_master_measures"](
                    "400 Gms x 20 Pkts"
                ),
            },
        ),
        (
            {
                "Offer Name": "Chicken and Beef Burger 20 pcs 750g + 250g",
                "Product": "Burgers-Frozen",
                "Variant": "No Variant",
                "offer_measures_detailed": legacy["extract_flyer_measures"](
                    "2 x 500g Chicken and Beef Burger"
                ),
            },
            {
                "Itemname": "CHICKEN BURGER",
                "Item-Cat-4": "CHICKEN",
                "Item Description": "Regular chicken burger",
                "Item-Spec": "270 Gms x 20 Pkts",
                "master_measures_detailed": legacy["extract_master_measures"](
                    "270 Gms x 20 Pkts"
                ),
            },
        ),
        (
            {
                "Offer Name": "Chicken Nuggets",
                "Product": "Chicken Nuggets-Frozen",
                "Variant": "",
                "offer_measures_detailed": [],
            },
            {
                "Itemname": "CHICKEN NUGGETS",
                "Item-Cat-4": "CHICKEN",
                "Item Description": "",
                "Item-Spec": "",
                "master_measures_detailed": [],
            },
        ),
    ]
    for offer, master in pairs:
        expected = legacy["_build_ml_feature_row"](offer, master)
        actual = build_feature_vector(offer, master)
        _assert_values_equal(actual, expected)
