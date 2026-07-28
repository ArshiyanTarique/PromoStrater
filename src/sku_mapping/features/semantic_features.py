"""Protein, product-family, and variant compatibility features."""

from __future__ import annotations

import re
from collections.abc import Collection

PROTEIN_WORDS = {
    "chicken",
    "beef",
    "veal",
    "lamb",
    "mutton",
    "fish",
    "shrimp",
    "shrimps",
    "prawn",
    "prawns",
    "turkey",
}
NON_MEAT_WORDS = {
    "cheese",
    "chocolate",
    "vegetable",
    "vegetables",
    "veg",
    "peas",
    "corn",
    "potato",
    "potatoes",
    "fries",
    "paratha",
    "spinach",
}
FAMILY_PHRASES = [
    "seekh kebab",
    "spring roll",
    "chicken fries",
    "fish fingers",
    "cheese sticks",
    "chicken strips",
    "chicken fillet",
    "chicken fillets",
    "chicken popcorn",
    "chicken nuggets",
    "beef burger",
    "chicken burger",
    "burger",
    "burgers",
    "nuggets",
    "fillet",
    "fillets",
    "strips",
    "popcorn",
    "kibbeh",
    "kofta",
    "samosa",
    "samosas",
    "paratha",
    "fries",
    "wedges",
    "patties",
    "sausages",
    "shrimps",
    "prawns",
]
VARIANT_WORDS = {
    "spicy",
    "non spicy",
    "non-spicy",
    "regular",
    "hot",
    "sriracha",
    "buffalo",
    "bbq",
    "barbecue",
    "onion",
    "breaded",
    "zing",
    "krazee",
    "jumbo",
    "mild",
}


def _protein_set(text: object) -> set[str]:
    """Return recognized protein tokens."""
    tokens = set(re.findall(r"[a-z]+", str(text).lower()))
    return tokens & PROTEIN_WORDS


def _family_set(text: object) -> set[str]:
    """Return recognized product-family phrases."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    found: set[str] = set()
    for phrase in FAMILY_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            found.add(phrase)
    return found


def _variant_set(text: object) -> set[str]:
    """Return recognized variant phrases."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    found: set[str] = set()
    for variant in VARIANT_WORDS:
        normalized_variant = variant.replace("-", " ")
        if re.search(rf"\b{re.escape(normalized_variant)}\b", normalized):
            found.add(normalized_variant)
    return found


def _compatibility_flag(
    left_values: Collection[str],
    right_values: Collection[str],
    unspecified_is_match: bool = True,
) -> int:
    """Return the production-compatible binary overlap flag."""
    if not left_values or not right_values:
        return 1 if unspecified_is_match else 0
    return int(bool(set(left_values) & set(right_values)))


def _expected_match_count(text: object) -> float:
    """Estimate expected matches from spaced offer separators."""
    normalized = str(text).lower()
    separators = len(re.findall(r"\s(?:/|\bor\b|\band\b|\+)\s", normalized))
    return float(max(1, min(4, separators + 1)))
