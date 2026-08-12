"""Extra features aimed at the confusions the 19-column set cannot express.

Measured failures on held-out offers were almost all near-identical products:

    24 CHICKEN BURGERS      vs  8 CHICKEN BURGERS        piece count
    BREADED FILLET SPICY    vs  BREADED FILLET NON-SPICY flavour
    CHICKEN STICKS          vs  CHICKEN STICKS BULK      pack size
    TAYEBAT CHICKEN BURGER  vs  AK PRO CHICKEN BURGER    product line

The existing features cannot separate these:

  * ``size_match`` is binary - 400g vs 450g and 400g vs 2kg both read as 0,
    so the model cannot tell a rounding difference from a different product.
  * ``variant_match`` uses ``unspecified_is_match=True``, so a SKU that states
    no flavour counts as compatible with a SPICY offer. That is RB-4, and it is
    exactly why spicy/non-spicy pairs get confused.
  * nothing encodes piece counts or the AK PRO / Tayebat / Zing product lines.

Each function returns a plain number, so these append to the existing vector
without disturbing it.
"""

from __future__ import annotations

import re

WEIGHT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kgs?|kilograms?|gms?|g|grms?|grams?)\b", re.IGNORECASE)
# "12 PKTS X 400 GRM", "8 X 750 GMS", "24's"
COUNT = re.compile(r"\b(\d{1,3})\s*(?:x|pkts?|pcs?|pieces?|'s)\b", re.IGNORECASE)
LEADING_COUNT = re.compile(r"^\s*(\d{1,3})\s+[a-z]", re.IGNORECASE)

SPICE_WORDS = {
    "spicy": "spicy", "hot": "spicy", "sriracha": "spicy", "buffalo": "spicy",
    "tandoori": "spicy", "peri": "spicy",
    "non spicy": "nonspicy", "nonspicy": "nonspicy", "non-spicy": "nonspicy",
    "plain": "plain", "regular": "plain", "original": "plain",
}
PRODUCT_LINES = ("ak pro", "a.k pro", "akpro", "tayebat", "taybat", "zing",
                 "zinger", "krazee", "tabarruk", "chargrilled", "jumbo")
BULK_WORDS = ("bulk", "catering", "carton", "case")


def _grams(text: object) -> list[float]:
    out = []
    for value, unit in WEIGHT.findall(str(text).replace(",", "")):
        v = float(value)
        out.append(v * 1000 if unit.lower().startswith(("kg", "kilo")) else v)
    return out


def _counts(text: object) -> set[int]:
    t = str(text)
    found = {int(c) for c in COUNT.findall(t)}
    lead = LEADING_COUNT.match(t)
    if lead:
        found.add(int(lead.group(1)))
    return {c for c in found if 1 <= c <= 200}


def _spice(text: object) -> set[str]:
    t = re.sub(r"[^a-z ]+", " ", str(text).lower())
    t = re.sub(r"\b(?:non|not|no)[\s-]*spicy\b", " nonspicy ", t)
    found = set()
    for word, tag in SPICE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            found.add(tag)
    return found


def _lines(text: object) -> set[str]:
    t = str(text).lower()
    return {p for p in PRODUCT_LINES if p in t}


def _is_bulk(text: object) -> int:
    t = str(text).lower()
    return int(any(w in t for w in BULK_WORDS))


def size_ratio(offer_text: object, master_text: object) -> float:
    """Smaller weight over larger, 1.0 when they agree.

    Continuous, unlike ``size_match``: 400g vs 450g scores 0.89 while 400g vs
    2kg scores 0.20, so the model can tell a rounding difference from a
    different pack. -1.0 when either side states no weight.
    """
    a, b = _grams(offer_text), _grams(master_text)
    if not a or not b:
        return -1.0
    best = 0.0
    for x in a:
        for y in b:
            if x > 0 and y > 0:
                best = max(best, min(x, y) / max(x, y))
    return round(best, 4)


def count_agreement(offer_text: object, master_text: object) -> float:
    """1.0 identical piece counts, 0.0 both stated and different, -1.0 unknown.

    Separates "24 CHICKEN BURGERS" from "8 CHICKEN BURGERS", which are otherwise
    textually near-identical.
    """
    a, b = _counts(offer_text), _counts(master_text)
    if not a or not b:
        return -1.0
    return 1.0 if a & b else 0.0


def spice_conflict(offer_text: object, master_text: object) -> float:
    """1.0 when the two state contradictory flavours.

    The point of this feature: unlike ``variant_match`` it fires only when BOTH
    sides state something and they disagree, so a spicy offer no longer looks
    compatible with a SKU that is silent on flavour.
    """
    a, b = _spice(offer_text), _spice(master_text)
    if not a or not b:
        return 0.0
    return 0.0 if a & b else 1.0


def spice_stated_only_by_offer(offer_text: object, master_text: object) -> float:
    """1.0 when the offer names a flavour and the SKU names none.

    Directly encodes the RB-4 gap: such a pair is weaker evidence than a SKU
    that states the same flavour, but the old binary flag scored them alike.
    """
    return float(bool(_spice(offer_text)) and not bool(_spice(master_text)))


def product_line_conflict(offer_text: object, master_text: object) -> float:
    """1.0 when both name a product line and the lines differ."""
    a, b = _lines(offer_text), _lines(master_text)
    if not a or not b:
        return 0.0
    return 0.0 if a & b else 1.0


def bulk_mismatch(offer_text: object, master_text: object) -> float:
    """1.0 when exactly one side is a bulk/catering pack."""
    return float(_is_bulk(offer_text) != _is_bulk(master_text))


EXTRA_FEATURE_COLUMNS: tuple[str, ...] = (
    "size_ratio",
    "count_agreement",
    "spice_conflict",
    "spice_stated_only_by_offer",
    "product_line_conflict",
    "bulk_mismatch",
)


def build_extra_features(offer_text: object, master_text: object) -> dict[str, float]:
    """Return the extra features for one offer/master pair."""
    return {
        "size_ratio": size_ratio(offer_text, master_text),
        "count_agreement": count_agreement(offer_text, master_text),
        "spice_conflict": spice_conflict(offer_text, master_text),
        "spice_stated_only_by_offer": spice_stated_only_by_offer(
            offer_text, master_text),
        "product_line_conflict": product_line_conflict(offer_text, master_text),
        "bulk_mismatch": bulk_mismatch(offer_text, master_text),
    }
