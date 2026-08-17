"""Weight, volume, multipack, and carton feature helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, TypeAlias

import numpy as np

Dimension: TypeAlias = Literal["weight", "volume"]
SimpleMeasure: TypeAlias = tuple[float, Dimension]
DetailedMeasure: TypeAlias = tuple[float, Dimension, int]

WEIGHT_UNITS = {
    "kg": 1000,
    "kgs": 1000,
    "g": 1,
    "gm": 1,
    "gms": 1,
    # The master data spells grams "GRM"; the flyers spell it "gm" or "g".
    # Without these the master side of a pack comparison parses to nothing,
    # pack_is_compatible answers None instead of False, and no size conflict
    # can ever be raised for that SKU - which is how a 550gm competitor came
    # to sit under a 400 GRM Al Kabeer SKU.
    "grm": 1,
    "grms": 1,
    "gram": 1,
    "grams": 1,
    "lb": 453.592,
    "lbs": 453.592,
    "oz": 28.3495,
}
VOLUME_UNITS = {
    "ml": 1,
    "l": 1000,
    "ltr": 1000,
    "litre": 1000,
    "liter": 1000,
}
ALL_UNITS = "|".join(
    sorted(set(WEIGHT_UNITS) | set(VOLUME_UNITS), key=len, reverse=True)
)


def unit_dim_value(unit: str) -> tuple[Dimension, float]:
    """Return the measurement dimension and base-unit multiplier."""
    if unit in WEIGHT_UNITS:
        return "weight", float(WEIGHT_UNITS[unit])
    return "volume", float(VOLUME_UNITS[unit])


def extract_measures_detailed(
    text: object,
    include_total: bool = True,
    include_unit: bool = True,
) -> list[DetailedMeasure]:
    """Extract normalized values, dimensions, and explicit unit counts.

    For ``N x size``, ``include_total`` controls the bundle/case total and
    ``include_unit`` controls the consumer-facing per-unit size.
    """
    if not isinstance(text, str):
        return []
    normalized = text.lower().replace(",", "")
    normalized = re.sub(
        (
            rf"(\d+(?:\.\d+)?)\s*({ALL_UNITS})\s*(?:x|\*|\u00d7)\s*"
            rf"(\d+(?:\.\d+)?)\s*(?:pkts?|packets?|packs?)\s*"
            rf"(?:x|\*|\u00d7)\s*(\d+(?:\.\d+)?)"
        ),
        r"\1 \2 x \3 x \4",
        normalized,
    )
    output: list[DetailedMeasure] = []
    occupied: list[tuple[int, int]] = []

    for match in re.finditer(
        rf"(\d+(?:\.\d+)?)\s*[x×\*]\s*(\d+(?:\.\d+)?)\s*({ALL_UNITS})\b",
        normalized,
    ):
        occupied.append(match.span())
        multiplier, size, unit = match.groups()
        dimension, factor = unit_dim_value(unit)
        count = int(round(float(multiplier)))
        if include_total:
            output.append((float(multiplier) * float(size) * factor, dimension, count))
        if include_unit:
            output.append((float(size) * factor, dimension, 1))

    for match in re.finditer(
        rf"(\d+(?:\.\d+)?)\s*({ALL_UNITS})\s*[x×\*]\s*(\d+(?:\.\d+)?)\b",
        normalized,
    ):
        occupied.append(match.span())
        size, unit, multiplier = match.groups()
        dimension, factor = unit_dim_value(unit)
        count = int(round(float(multiplier)))
        if include_total:
            output.append((float(size) * float(multiplier) * factor, dimension, count))
        if include_unit:
            output.append((float(size) * factor, dimension, 1))

    def inside_multipack(span: tuple[int, int]) -> bool:
        return any(span[0] >= start and span[1] <= end for start, end in occupied)

    for match in re.finditer(
        rf"(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*({ALL_UNITS})\b",
        normalized,
    ):
        if inside_multipack(match.span()):
            continue
        first, second, unit = match.groups()
        dimension, factor = unit_dim_value(unit)
        output.append((float(first) * factor, dimension, 1))
        if second:
            output.append((float(second) * factor, dimension, 1))

    return sorted(
        set(
            (round(value, 1), dimension, count)
            for value, dimension, count in output
            if 1 <= value <= 30000
        )
    )


def extract_flyer_measures(text: object) -> list[DetailedMeasure]:
    """Extract both promotional bundle totals and retail unit sizes."""
    return extract_measures_detailed(text, include_total=True, include_unit=True)


def extract_master_measures(text: object) -> list[DetailedMeasure]:
    """Extract retail units without treating master carton totals as packs."""
    return extract_measures_detailed(text, include_total=False, include_unit=True)


def collapse_to_simple(
    detailed_measures: Iterable[DetailedMeasure],
) -> list[SimpleMeasure]:
    """Remove counts while preserving value and measurement dimension."""
    return sorted(set((value, dimension) for value, dimension, _ in detailed_measures))


def pack_is_compatible(
    offer_measures: Iterable[SimpleMeasure],
    master_measures: Iterable[SimpleMeasure],
    tol: float = 0.10,
) -> bool | None:
    """Return pack compatibility, comparing only like dimensions."""
    offer = list(offer_measures)
    master = list(master_measures)
    if not offer or not master:
        return None
    for offer_value, offer_dimension in offer:
        for master_value, master_dimension in master:
            if (
                offer_dimension == master_dimension
                and abs(offer_value - master_value) / max(offer_value, master_value) <= tol
            ):
                return True
    return False


def pack_structure_agrees(
    offer_details: Iterable[DetailedMeasure],
    master_details: Iterable[DetailedMeasure],
    tol: float = 0.10,
) -> bool | None:
    """Compare pack values and unit counts using legacy tri-state semantics."""
    offer = list(offer_details)
    master = list(master_details)
    if not offer or not master:
        return None
    any_total_match = False
    for offer_value, offer_dimension, offer_count in offer:
        for master_value, master_dimension, master_count in master:
            if (
                offer_dimension == master_dimension
                and abs(offer_value - master_value) / max(offer_value, master_value) <= tol
            ):
                any_total_match = True
                if offer_count == master_count:
                    return True
    return False if any_total_match else None


def _first_weight_value(details: Iterable[DetailedMeasure]) -> float:
    """Return the smallest consumer-facing weight in grams."""
    values = list(details)
    if not values:
        return np.nan
    retail = [
        float(value)
        for value, dimension, count in values
        if dimension == "weight" and count == 1
    ]
    if retail:
        return min(retail)
    weights = [float(value) for value, dimension, _ in values if dimension == "weight"]
    return min(weights) if weights else np.nan


def _offer_unit_count(details: Iterable[DetailedMeasure]) -> float:
    """Return the largest explicit flyer multipack count, otherwise one."""
    values = list(details)
    if not values:
        return np.nan
    counts = [int(count) for _, _, count in values if int(count) > 1]
    return float(max(counts)) if counts else 1.0


def _offer_total_weight(details: Iterable[DetailedMeasure]) -> float:
    """Return the largest parsed flyer weight."""
    values = list(details)
    if not values:
        return np.nan
    weights = [float(value) for value, dimension, _ in values if dimension == "weight"]
    return max(weights) if weights else np.nan


def _extract_piece_count(text: object) -> float:
    """Extract explicit piece counts without treating multipack weights as pieces."""
    normalized = str(text).lower()
    patterns = [
        r"(\d+)\s*(?:pcs?|pieces?)\b",
        r"\b(\d+)\s*['’]s\b",
        rf"\bx\s*(\d+)(?!\d)(?!\s*(?:{ALL_UNITS})\b)\s*(?:pcs?|pieces?)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return np.nan


def _extract_bonus_weight(text: object) -> float:
    """Extract the second weight in an expression such as ``750g+250g``."""
    normalized = str(text).lower().replace(" ", "")
    match = re.search(
        rf"\d+(?:\.\d+)?(?:{ALL_UNITS})\+(\d+(?:\.\d+)?)({ALL_UNITS})",
        normalized,
    )
    if not match:
        return np.nan
    amount, unit = match.groups()
    dimension, factor = unit_dim_value(unit)
    return float(amount) * factor if dimension == "weight" else np.nan


def _master_units_per_carton(spec: object) -> float:
    """Extract the explicit carton multiplier from a master specification."""
    normalized = str(spec).lower().replace(",", "")
    chained = re.search(
        (
            rf"\d+(?:\.\d+)?\s*(?:{ALL_UNITS})\s*(?:x|\*|\u00d7)\s*"
            rf"\d+(?:\.\d+)?\s*(?:pkts?|packets?|packs?)\s*"
            rf"(?:x|\*|\u00d7)\s*(\d+(?:\.\d+)?)"
        ),
        normalized,
    )
    if chained:
        return float(chained.group(1))
    patterns = [
        rf"\d+(?:\.\d+)?\s*(?:{ALL_UNITS})\s*[x×*]\s*(\d+(?:\.\d+)?)",
        rf"(\d+(?:\.\d+)?)\s*[x×*]\s*\d+(?:\.\d+)?\s*(?:{ALL_UNITS})",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return np.nan
