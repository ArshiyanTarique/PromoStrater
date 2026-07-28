"""Text normalization used by the legacy SKU feature generator."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

FILLER_WORDS = {
    "krazee",
    "jumbo",
    "promo",
    "free",
    "new",
    "nispc",
    "special",
    "offer",
    "combo",
    "value",
    "mega",
    "family",
}


def safe_text(value: Any) -> str:
    """Return a usable string, treating common missing values as empty."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def clean_offer_text(text: Any) -> str:
    """Normalize matching text exactly as the production feature logic does."""
    normalized = re.sub(r"al[\s\-]?kabeer", " ", safe_text(text).lower())
    words = re.findall(r"[a-z]+", normalized)
    excluded_units = {"gm", "kg", "g", "kgs", "gms", "lb", "lbs", "oz", "ml", "l"}
    return " ".join(
        word for word in words if word not in FILLER_WORDS and word not in excluded_units
    )
