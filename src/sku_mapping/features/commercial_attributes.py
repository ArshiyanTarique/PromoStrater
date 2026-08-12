"""Independent commercial parsing and deterministic candidate comparison."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from sku_mapping.features.measurement_features import (
    ALL_UNITS,
    _extract_bonus_weight,
    _extract_piece_count,
    unit_dim_value,
)
from sku_mapping.features.semantic_features import (
    _family_concept_set,
    _protein_set,
    _variant_set,
)
from sku_mapping.features.text_features import safe_text


class MappingOutcome(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    ADAPTED_MATCH = "ADAPTED_MATCH"
    UNACCEPTABLE_MATCH = "UNACCEPTABLE_MATCH"
    UNKNOWN = "UNKNOWN"


class MeasurementMatch(str, Enum):
    EXACT = "EXACT"
    CONVERSION_EQUIVALENT = "CONVERSION_EQUIVALENT"
    TOLERANCE_MATCH = "TOLERANCE_MATCH"
    PROMOTION_MISMATCH = "PROMOTION_MISMATCH"
    UNIT_WEIGHT_MISMATCH = "UNIT_WEIGHT_MISMATCH"
    TOTAL_WEIGHT_MISMATCH = "TOTAL_WEIGHT_MISMATCH"
    PACK_FORMAT_MISMATCH = "PACK_FORMAT_MISMATCH"
    UNIT_SIZE_MISMATCH = "UNIT_SIZE_MISMATCH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CommercialTaxonomy:
    """Injectable, catalogue-independent vocabulary and tolerance."""

    flavour_words: frozenset[str] = frozenset(
        {"bbq", "barbecue", "buffalo", "butter", "cheese", "garlic", "herb",
         "onion", "pepper", "sriracha", "tandoori"}
    )
    product_line_words: frozenset[str] = frozenset(
        {"classic", "jumbo", "krazee", "premium", "value"}
    )
    promo_words: frozenset[str] = frozenset(
        {"bonus", "combo", "free", "offer", "promo", "promotional",
         "twin", "value pack"}
    )
    format_words: frozenset[str] = frozenset(
        {"bag", "box", "bulk", "carton", "family pack", "packet", "pack",
         "pouch", "tray"}
    )
    family_words: frozenset[str] = frozenset(
        {
            "breast", "cubes", "meat balls", "mixed vegetables", "paneer",
            "peas", "sticks",
        }
    )
    tolerance: float = 0.10


DEFAULT_COMMERCIAL_TAXONOMY = CommercialTaxonomy()


@dataclass(frozen=True)
class CommercialAttributes:
    family: tuple[str, ...]
    subfamily: tuple[str, ...]
    protein: tuple[str, ...]
    mixed_protein: bool
    variants: tuple[str, ...]
    flavour: tuple[str, ...]
    spicy_state: str
    product_line: tuple[str, ...]
    commercial_format: tuple[str, ...]
    promotional: bool
    base_measure: float | None
    bonus_measure: float | None
    total_measure: float | None
    measurement_dimension: str | None
    pack_count: int | None
    piece_count: int | None
    per_piece_measure: float | None
    bundle_structure: str
    multi_product: bool
    slash_ambiguity: bool
    source_field_conflict: bool
    confidence: float
    evidence_text: str


@dataclass(frozen=True)
class CommercialComparison:
    outcome: str
    severity: str
    measurement_match: str
    hard_conflict: bool
    exact_match_eligible: bool
    family_relation: str
    protein_relation: str
    variant_relation: str
    reason_codes: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["reason_codes"] = "|".join(self.reason_codes)
        return {f"commercial_{key}": value for key, value in raw.items()}


def _row_text(row: Mapping[str, Any], columns: tuple[str, ...]) -> str:
    return " ".join(safe_text(row.get(column, "")) for column in columns).strip()


def _phrases(text: str, words: frozenset[str]) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return tuple(sorted(
        word for word in words
        if re.search(rf"\b{re.escape(word)}\b", normalized)
    ))


def _measure_profile(
    text: str,
    *,
    master_mode: bool = False,
) -> tuple[float | None, float | None, float | None, str | None, int | None]:
    normalized = text.lower().replace(",", "")
    normalized = re.sub(r"\bgrams?\b", "g", normalized)
    normalized = re.sub(r"\bkilograms?\b", "kg", normalized)
    normalized = re.sub(rf"\b({ALL_UNITS})\.", r"\1", normalized)
    # Catalogue descriptions can concatenate a short item-name size with the
    # same description size (for example ``400400 g``). Collapse only an
    # immediately repeated 2-4 digit token before a recognized unit.
    normalized = re.sub(
        rf"\b(\d{{2,4}})\1\s*({ALL_UNITS})\b", r"\1 \2", normalized
    )
    raw_bonus = _extract_bonus_weight(normalized)
    bonus = None if math.isnan(raw_bonus) else float(raw_bonus)
    if master_mode:
        direct: list[tuple[float, str]] = []
        for match in re.finditer(
            rf"(\d+(?:\.\d+)?)\s*({ALL_UNITS})\b", normalized
        ):
            amount, unit = match.groups()
            dimension, factor = unit_dim_value(unit)
            value = float(amount) * factor
            if 1 <= value <= 30000:
                direct.append((value, dimension))
        if not direct:
            return None, bonus, None, None, None
        if bonus is not None:
            base, dimension = direct[0]
        elif re.search(r"\bbulk\b", normalized):
            base, dimension = max(direct, key=lambda item: item[0])
        else:
            base, dimension = min(direct, key=lambda item: item[0])
        return base, bonus, base + (bonus or 0.0), dimension, 1
    forward = re.search(
        rf"(\d+(?:\.\d+)?)\s*[x×*]\s*(\d+(?:\.\d+)?)\s*({ALL_UNITS})\b",
        normalized,
    )
    reverse = re.search(
        rf"(\d+(?:\.\d+)?)\s*({ALL_UNITS})\s*[x×*]\s*(\d+(?:\.\d+)?)\b",
        normalized,
    )
    if forward:
        count_text, size_text, unit = forward.groups()
    elif reverse:
        size_text, unit, count_text = reverse.groups()
    else:
        single = re.search(rf"(\d+(?:\.\d+)?)\s*({ALL_UNITS})\b", normalized)
        if not single:
            return None, None, None, None, None
        size_text, unit = single.groups()
        count_text = "1"
    dimension, factor = unit_dim_value(unit)
    count = max(1, int(round(float(count_text))))
    base = float(size_text) * factor
    return base, bonus, base * count + (bonus or 0.0), dimension, count


def _spicy_state(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    if re.search(r"\b(?:non spicy|no spicy|mild|plain)\b", normalized):
        return "NON_SPICY"
    if re.search(r"\b(?:spicy|hot|sriracha|buffalo)\b", normalized):
        return "SPICY"
    return "UNKNOWN"


def _parse(
    primary_text: str,
    auxiliary_text: str,
    taxonomy: CommercialTaxonomy,
    *,
    check_field_conflict: bool,
    master_mode: bool = False,
) -> CommercialAttributes:
    combined = " ".join(filter(None, (primary_text, auxiliary_text)))
    primary_proteins = _protein_set(primary_text)
    auxiliary_proteins = _protein_set(auxiliary_text)
    field_conflict = bool(
        check_field_conflict and primary_proteins and auxiliary_proteins
        and primary_proteins.isdisjoint(auxiliary_proteins)
    )
    proteins = primary_proteins or auxiliary_proteins
    proteins = set(proteins)
    semantic_families = set(_family_concept_set(combined))
    families = tuple(sorted(
        semantic_families
        or set(_phrases(combined, taxonomy.family_words))
    ))
    variants = tuple(sorted(_variant_set(combined)))
    base, bonus, total, dimension, pack_count = _measure_profile(
        combined, master_mode=master_mode
    )
    raw_piece = _extract_piece_count(combined)
    piece_count = None if math.isnan(raw_piece) else int(raw_piece)
    slash = bool(re.search(r"\s/\s", combined))
    explicit_bundle_separator = bool(re.search(r"\s(?:/|\+)\s", combined))
    multi = (
        len(families) > 1 or len(proteins) > 1 or explicit_bundle_separator
    )
    promotional = bool(_phrases(combined, taxonomy.promo_words) or bonus)
    bundle = (
        "MULTI_PRODUCT" if multi else
        "PROMOTIONAL_OR_MULTIPACK" if promotional or (pack_count or 1) > 1
        else "SINGLE_PRODUCT"
    )
    evidence_groups = sum(bool(v) for v in (families, proteins, variants, base, piece_count))
    confidence = min(1.0, 0.25 + evidence_groups * 0.15)
    if field_conflict or slash:
        confidence = max(0.0, confidence - 0.25)
    return CommercialAttributes(
        family=families,
        subfamily=families,
        protein=tuple(sorted(proteins)),
        mixed_protein=len(proteins) > 1,
        variants=variants,
        flavour=_phrases(combined, taxonomy.flavour_words),
        spicy_state=_spicy_state(combined),
        product_line=_phrases(combined, taxonomy.product_line_words),
        commercial_format=_phrases(combined, taxonomy.format_words),
        promotional=promotional,
        base_measure=base,
        bonus_measure=bonus,
        total_measure=total,
        measurement_dimension=dimension,
        pack_count=pack_count,
        piece_count=piece_count,
        per_piece_measure=(
            total / piece_count if total is not None and piece_count else None
        ),
        bundle_structure=bundle,
        multi_product=multi,
        slash_ambiguity=slash,
        source_field_conflict=field_conflict,
        confidence=round(confidence, 2),
        evidence_text=combined,
    )


def parse_source_attributes(
    row: Mapping[str, Any],
    taxonomy: CommercialTaxonomy = DEFAULT_COMMERCIAL_TAXONOMY,
) -> CommercialAttributes:
    """Parse source columns only, making candidate-value inheritance impossible."""
    return _parse(
        _row_text(row, ("Offer Name", "Product", "Base Packsize")),
        _row_text(row, ("Variant",)),
        taxonomy,
        check_field_conflict=True,
        master_mode=False,
    )


def parse_master_attributes(
    row: Mapping[str, Any],
    taxonomy: CommercialTaxonomy = DEFAULT_COMMERCIAL_TAXONOMY,
) -> CommercialAttributes:
    return _parse(
        _row_text(row, (
            "Itemname", "Item-Cat-2", "Item-Cat-4",
            "Item Description", "Item-Spec",
        )),
        "",
        taxonomy,
        check_field_conflict=False,
        master_mode=True,
    )


def _relation(source: tuple[str, ...], master: tuple[str, ...]) -> str:
    if not source or not master:
        return "UNKNOWN"
    return "MATCH" if set(source) & set(master) else "CONFLICT"


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) / max(abs(left), abs(right), 1.0) <= tolerance


def _measurement_relation(
    source: CommercialAttributes,
    master: CommercialAttributes,
    tolerance: float,
) -> MeasurementMatch:
    if source.base_measure is None or master.base_measure is None:
        return MeasurementMatch.UNKNOWN
    if source.measurement_dimension != master.measurement_dimension:
        return MeasurementMatch.UNIT_SIZE_MISMATCH
    if _close(source.base_measure, master.base_measure, 0.001):
        source_promo = source.promotional or (source.pack_count or 1) > 1
        if source_promo != master.promotional:
            return MeasurementMatch.PROMOTION_MISMATCH
        return MeasurementMatch.EXACT
    if _close(source.base_measure, master.base_measure, tolerance):
        return MeasurementMatch.TOLERANCE_MATCH
    if (
        source.total_measure is not None and master.total_measure is not None
        and _close(source.total_measure, master.total_measure, tolerance)
    ):
        return MeasurementMatch.UNIT_WEIGHT_MISMATCH
    return MeasurementMatch.UNIT_SIZE_MISMATCH


def compare_commercial_attributes(
    source: CommercialAttributes,
    master: CommercialAttributes,
    taxonomy: CommercialTaxonomy = DEFAULT_COMMERCIAL_TAXONOMY,
) -> CommercialComparison:
    family = _relation(source.family, master.family)
    protein = _relation(source.protein, master.protein)
    variant = _relation(source.variants, master.variants)
    measurement = _measurement_relation(source, master, taxonomy.tolerance)
    reasons: list[str] = []
    hard = False
    for relation, code in ((family, "FAMILY_CONFLICT"), (protein, "PROTEIN_CONFLICT")):
        if relation == "CONFLICT":
            reasons.append(code)
            hard = True
    if (
        source.spicy_state != "UNKNOWN" and master.spicy_state != "UNKNOWN"
        and source.spicy_state != master.spicy_state
    ):
        reasons.append("SPICY_POLARITY_CONFLICT")
        hard = True
        variant = "CONFLICT"
    if source.source_field_conflict:
        reasons.append("SOURCE_FIELD_CONTRADICTION")
    source_flavour = set(source.flavour)
    master_flavour = set(master.flavour)
    if master_flavour and not master_flavour.issubset(source_flavour):
        reasons.append("MASTER_ONLY_FLAVOUR")
    source_line = set(source.product_line)
    master_line = set(master.product_line)
    if master_line and not master_line.issubset(source_line):
        reasons.append(
            "PRODUCT_LINE_CONFLICT"
            if source_line
            else "MASTER_ONLY_PRODUCT_LINE"
        )
    if source.promotional != master.promotional:
        if "PROMOTIONAL_STRUCTURE_MISMATCH" not in reasons:
            reasons.append("PROMOTIONAL_STRUCTURE_MISMATCH")
    if source.multi_product or source.slash_ambiguity:
        reasons.append("SOURCE_BUNDLE_AMBIGUITY")
    if measurement is MeasurementMatch.UNIT_SIZE_MISMATCH:
        reasons.append(measurement.value)
        hard = True
    elif measurement is MeasurementMatch.PROMOTION_MISMATCH:
        reasons.append("PROMOTIONAL_STRUCTURE_MISMATCH")
    unknown = family == protein == "UNKNOWN" and measurement is MeasurementMatch.UNKNOWN
    adapted = bool(
        source.multi_product or source.slash_ambiguity
        or source.source_field_conflict
        or "MASTER_ONLY_FLAVOUR" in reasons
        or "PRODUCT_LINE_CONFLICT" in reasons
        or "MASTER_ONLY_PRODUCT_LINE" in reasons
        or "PROMOTIONAL_STRUCTURE_MISMATCH" in reasons
        or measurement is MeasurementMatch.PROMOTION_MISMATCH
    )
    if hard:
        outcome, severity = MappingOutcome.UNACCEPTABLE_MATCH, "HARD"
    elif adapted:
        outcome, severity = MappingOutcome.ADAPTED_MATCH, "SEVERE"
    elif unknown:
        outcome, severity = MappingOutcome.UNKNOWN, "UNKNOWN"
    else:
        outcome, severity = MappingOutcome.EXACT_MATCH, "NONE"
    return CommercialComparison(
        outcome=outcome.value,
        severity=severity,
        measurement_match=measurement.value,
        hard_conflict=hard,
        exact_match_eligible=outcome is MappingOutcome.EXACT_MATCH,
        family_relation=family,
        protein_relation=protein,
        variant_relation=variant,
        reason_codes=tuple(reasons),
    )


def attributes_json(attributes: CommercialAttributes) -> str:
    return json.dumps(asdict(attributes), sort_keys=True, ensure_ascii=False)
