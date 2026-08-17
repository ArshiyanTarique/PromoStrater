"""Competitor-side text normalisation shared by the rules and the re-ranker.

One implementation, two consumers. :func:`strip_competitor_brand` was
originally inlined in ``discovery._competitor_match_text``; the ML re-ranker
needs exactly the same treatment, and a second copy would be free to drift.
Both now call this.

Why competitors need it at all: an Al Kabeer master text can never contain a
rival's brand, so every rival brand token is a guaranteed miss that only
dilutes a token ratio. Measured on the production model, leaving the brand in
costs a competitor pair roughly 4 raw-margin points against the identical
product - a penalty for naming itself, not for being a worse match.

Own-brand text is untouched by this module. ``clean_offer_text`` already
removes the Al Kabeer token from both sides of an own-brand comparison.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Transliterations of the same product word. ClickFlyer carries retailer copy
#: verbatim, so the same item appears as "Sambosa" or "Samosa" depending on who
#: wrote the flyer, and a token-based ratio treats those as unrelated words.
COMPETITOR_TOKEN_ALIASES: dict[str, str] = {
    "sambosa": "samosa",
    "sambosas": "samosa",
    "samboosa": "samosa",
    "sambousa": "samosa",
    "samosas": "samosa",
}


def _tokens(text: str) -> set[str]:
    return set(str(text or "").replace("-", " ").lower().split())


def strip_competitor_brand(
    text: object,
    brand: object,
    *,
    protected: object = "",
    aliases: dict[str, str] | None = None,
) -> str:
    """Drop a competitor's own brand tokens and fold transliterations.

    ``protected`` names tokens that must survive even when they also appear in
    the brand - in practice the row's ``Product``, because a brand word that is
    also a product word ("Cucina Tempura" against a "Tempura" product) is
    carrying product meaning rather than branding.

    Nothing is invented: tokens are only removed or spelling-folded, never
    added, so a row with no brand comes back unchanged apart from aliasing.
    """
    table = COMPETITOR_TOKEN_ALIASES if aliases is None else aliases
    lowered = str(text or "").lower()
    brand_tokens = _tokens(brand)
    protected_tokens = _tokens(protected)
    kept: Iterable[str] = (
        table.get(token, token)
        for token in lowered.split()
        if token not in brand_tokens or token in protected_tokens
    )
    return " ".join(kept)
