"""Canonical offer identity shared by dashboard and inference workflows."""

from __future__ import annotations

import hashlib
import math
import numbers
import re
from dataclasses import dataclass

import pandas as pd

FALLBACK_IDENTITY_VERSION = "stable_offer_fingerprint_v1"
_WHOLE_DECIMAL_TEXT = re.compile(r"^([+-]?\d+)\.0+$")
_MISSING_TEXT = frozenset({"", "nan", "none", "null", "<na>", "nat"})


@dataclass(frozen=True)
class OfferIdentityAssignment:
    """Canonical identities plus auditable source/count metadata."""

    identities: pd.Series
    source: str
    unique_offer_count: int
    valid_offer_id_count: int
    missing_offer_id_count: int


def normalize_offer_id(value: object) -> str | None:
    """Return a stable scalar ID without changing meaningful string IDs."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        return str(int(number)) if number.is_integer() else str(value).strip()

    text = str(value).strip()
    if text.casefold() in _MISSING_TEXT:
        return None
    whole_decimal = _WHOLE_DECIMAL_TEXT.fullmatch(text)
    if whole_decimal:
        return whole_decimal.group(1)
    return text


def fallback_offer_identity(row: pd.Series, position: int) -> str:
    """Use the repository's deterministic row fingerprint fallback."""
    existing = normalize_offer_id(row.get("offer_group_id"))
    if existing is not None:
        return existing
    source_identifier = normalize_offer_id(row.get("record_id")) or ""
    payload = "|".join(
        [
            source_identifier,
            str(row.get("Offer Name", "")),
            str(row.get("Retailer Name", "")),
            str(row.get("Product", "")),
            str(row.get("Variant", "")),
            str(row.get("Base Packsize", "")),
            str(position),
        ]
    )
    return "shadow_offer_" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def canonical_offer_identity(row: pd.Series, position: int) -> str:
    """Prefer ClickFlyer ``offerid`` and otherwise use the stable fallback."""
    offer_id = normalize_offer_id(row.get("offerid"))
    if offer_id is not None:
        return offer_id
    return fallback_offer_identity(row, position)


def assign_offer_identities(frame: pd.DataFrame) -> OfferIdentityAssignment:
    """Assign the identity used for counting, inference, review, and storage."""
    if "offerid" in frame.columns:
        normalized = frame["offerid"].map(normalize_offer_id)
    else:
        normalized = pd.Series(
            [None] * len(frame), index=frame.index, dtype="object"
        )
    valid = normalized.notna()
    valid_count = int(valid.sum())
    missing_count = int(len(frame) - valid_count)

    identities = normalized.astype("object").copy()
    missing_positions = [
        position
        for position, is_valid in enumerate(valid.to_numpy())
        if not is_valid
    ]
    for position in missing_positions:
        identities.iloc[position] = fallback_offer_identity(
            frame.iloc[position], position
        )
    identities.name = "offer_group_id"
    if valid_count:
        source = (
            "offerid"
            if missing_count == 0
            else "offerid_with_stable_fallback"
        )
    else:
        source = FALLBACK_IDENTITY_VERSION
    # Missing source IDs never become a shared "nan" bucket. Each receives the
    # documented deterministic fallback and remains a traceable canonical offer.
    unique_count = int(identities.nunique(dropna=True))
    return OfferIdentityAssignment(
        identities=identities,
        source=source,
        unique_offer_count=unique_count,
        valid_offer_id_count=valid_count,
        missing_offer_id_count=missing_count,
    )
