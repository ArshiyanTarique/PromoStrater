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
FAMILY_CONCEPT_ALIASES = {
    "beef burger": "burger",
    "burgers": "burger",
    "chicken burger": "burger",
    "chicken fillet": "fillet",
    "chicken fillets": "fillet",
    "chicken fries": "fries",
    "chicken nuggets": "nuggets",
    "chicken popcorn": "popcorn",
    "chicken strips": "strips",
    "fillets": "fillet",
    "samosas": "samosa",
}
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


# Fix: PROTEIN_WORDS lists both spellings of the same protein, so an offer
# saying "BREADED SHRIMP" produced {"shrimp"} while the master "BREADED SHRIMPS
# 250 / PRAWNS-SHRIMPS" produced {"shrimps", "prawns"}. The two sets did not
# intersect, so _compatibility_flag reported a protein CONTRADICTION for a pair
# that is in fact the same product. Canonicalizing to one spelling removes the
# false contradiction without widening what counts as a protein.
PROTEIN_CANONICAL = {"shrimp": "shrimps", "prawn": "prawns"}


def _protein_set(text: object) -> set[str]:
    """Return recognized protein tokens, canonicalized to one spelling."""
    tokens = set(re.findall(r"[a-z]+", str(text).lower()))
    return {PROTEIN_CANONICAL.get(word, word) for word in tokens & PROTEIN_WORDS}


def _family_set(text: object) -> set[str]:
    """Return recognized product-family phrases."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    # Fix: the master spells this family "POP-CORN", which the punctuation strip
    # above turns into "pop corn", but FAMILY_PHRASES only holds "popcorn".
    # No popcorn SKU could ever set family_match=1 against a "POPCORN" offer.
    normalized = re.sub(r"\bpop corn\b", "popcorn", normalized)
    found: set[str] = set()
    for phrase in FAMILY_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            found.add(phrase)
    return found


def _family_concept_set(text: object) -> set[str]:
    """Collapse nested and singular/plural phrases into product concepts."""
    return {
        FAMILY_CONCEPT_ALIASES.get(family, family)
        for family in _family_set(text)
    }


def _resolve_semantic_variant(
    offer_name: object,
    product: object,
    variant: object,
) -> str:
    """Ignore a protein Variant that contradicts explicit source meaning.

    The raw Variant remains untouched for traceability and exports. This
    helper only controls semantic parsing and therefore cannot copy values
    from, or depend on, a candidate Product Master row.
    """
    variant_text = "" if variant is None else str(variant).strip()
    primary_text = " ".join(
        value
        for value in (
            "" if offer_name is None else str(offer_name).strip(),
            "" if product is None else str(product).strip(),
        )
        if value
    )
    primary_proteins = _protein_set(primary_text)
    variant_proteins = _protein_set(variant_text)
    if (
        primary_proteins
        and variant_proteins
        and not primary_proteins.intersection(variant_proteins)
    ):
        return ""
    return variant_text


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
