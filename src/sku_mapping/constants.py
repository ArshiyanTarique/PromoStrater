"""Stable constants shared by SKU-mapping training and inference."""

MODEL_FEATURE_COLUMNS: list[str] = [
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
