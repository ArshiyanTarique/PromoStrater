"""Shared model-feature generation for training and inference."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.features.measurement_features import (
    DetailedMeasure,
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
)
from sku_mapping.features.semantic_features import (
    NON_MEAT_WORDS,
    _compatibility_flag,
    _expected_match_count,
    _family_concept_set,
    _family_set,
    _protein_set,
    _resolve_semantic_variant,
    _variant_set,
)
from sku_mapping.features.text_features import clean_offer_text, safe_text

FeatureValue = float | int | None


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    """Read a mapping value without relying on pandas-specific behavior."""
    try:
        return row.get(key, "")
    except AttributeError as error:
        raise TypeError("offer_row and master_row must be mapping-compatible") from error


def _valid_details(value: Any) -> list[DetailedMeasure] | None:
    """Normalize an existing detailed-measure collection, if supplied."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return list(value)
    except TypeError:
        return None


def _offer_details(offer_row: Mapping[str, Any]) -> list[DetailedMeasure]:
    if "offer_measures_detailed" in offer_row:
        supplied = _valid_details(_row_value(offer_row, "offer_measures_detailed"))
        if supplied is not None:
            return supplied
    source = " ".join(
        [
            safe_text(_row_value(offer_row, "Base Packsize")),
            safe_text(_row_value(offer_row, "Offer Name")),
        ]
    )
    return extract_flyer_measures(source)


def _master_details(master_row: Mapping[str, Any]) -> list[DetailedMeasure]:
    if "master_measures_detailed" in master_row:
        supplied = _valid_details(_row_value(master_row, "master_measures_detailed"))
        if supplied is not None:
            return supplied
    return extract_master_measures(safe_text(_row_value(master_row, "Item-Spec")))


def build_feature_vector(
    offer_row: Mapping[str, Any],
    master_row: Mapping[str, Any],
) -> dict[str, float | int | None]:
    """Build the ordered 19-feature vector for an offer/master pair.

    Existing precomputed detailed measurements are honored for exact inference
    parity. When absent, the same source-specific parsers derive them from the
    raw offer and master fields.
    """
    offer_name = safe_text(_row_value(offer_row, "Offer Name"))
    offer_product = safe_text(_row_value(offer_row, "Product"))
    semantic_variant = _resolve_semantic_variant(
        offer_name,
        offer_product,
        safe_text(_row_value(offer_row, "Variant")),
    )
    offer_text = " ".join(
        [
            offer_name,
            offer_product,
            semantic_variant,
        ]
    ).strip()
    master_text = " ".join(
        [
            safe_text(_row_value(master_row, "Itemname")),
            safe_text(_row_value(master_row, "Item-Cat-4")),
            safe_text(_row_value(master_row, "Item Description")),
        ]
    ).strip()

    offer_clean = clean_offer_text(offer_text)
    master_clean = clean_offer_text(master_text)
    offer_proteins = _protein_set(offer_text)
    master_proteins = _protein_set(master_text)
    offer_families = _family_set(offer_text)
    master_families = _family_set(master_text)
    offer_variants = _variant_set(offer_text)
    master_variants = _variant_set(master_text)

    offer_details = _offer_details(offer_row)
    master_details = _master_details(master_row)
    pack_ok = pack_is_compatible(
        collapse_to_simple(offer_details),
        collapse_to_simple(master_details),
    )
    structure_ok = pack_structure_agrees(offer_details, master_details)

    word_similarity = (
        fuzz.token_sort_ratio(offer_clean, master_clean)
        + fuzz.token_set_ratio(offer_clean, master_clean)
    ) / 2.0

    features: dict[str, FeatureValue] = {
        "protein_match": _compatibility_flag(offer_proteins, master_proteins),
        "family_match": _compatibility_flag(offer_families, master_families),
        "variant_match": _compatibility_flag(offer_variants, master_variants),
        "size_match": int(pack_ok is True),
        "pack_format_match": int(structure_ok is not False),
        "word_similarity": round(float(word_similarity), 1),
        "character_similarity": round(float(fuzz.ratio(offer_clean, master_clean)), 1),
        "token_similarity": round(float(fuzz.WRatio(offer_clean, master_clean)), 1),
        "unit_pack_weight_g": _first_weight_value(offer_details),
        "number_of_units": _offer_unit_count(offer_details),
        "bonus_weight_g": _extract_bonus_weight(offer_text),
        "total_offer_weight_g": _offer_total_weight(offer_details),
        "piece_count": _extract_piece_count(offer_text),
        "master_unit_weight_g": _first_weight_value(master_details),
        "master_units_per_carton": _master_units_per_carton(
            _row_value(master_row, "Item-Spec")
        ),
        "is_mixed_protein_offer": int(len(offer_proteins) > 1),
        "is_multi_family_offer": int(
            len(_family_concept_set(offer_text)) > 1
        ),
        "contains_non_meat_product": int(
            bool(set(re.findall(r"[a-z]+", offer_text.lower())) & NON_MEAT_WORDS)
        ),
        "expected_match_count": _expected_match_count(offer_text),
    }
    return {column: features[column] for column in MODEL_FEATURE_COLUMNS}


def build_feature_vector_from_text(
    offer_text: str,
    master_row: Mapping[str, Any],
    product: str = "",
    variant: str = "",
    base_packsize: str = "",
) -> dict[str, float | int | None]:
    """Build features for a synthetic offer absent from the flyer dataset."""
    offer_row: dict[str, Any] = {
        "Offer Name": safe_text(offer_text),
        "Product": safe_text(product),
        "Variant": safe_text(variant),
        "Base Packsize": safe_text(base_packsize),
    }
    return build_feature_vector(offer_row, master_row)
