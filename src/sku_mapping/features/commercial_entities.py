"""Deterministic decomposition of one source offer into commercial entities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from sku_mapping.features.commercial_attributes import parse_source_attributes
from sku_mapping.features.measurement_features import (
    ALL_UNITS,
    collapse_to_simple,
    extract_flyer_measures,
    unit_dim_value,
)
from sku_mapping.features.semantic_features import _protein_set
from sku_mapping.features.text_features import clean_offer_text, safe_text

ENTITY_PARSER_VERSION = "1.0.0"

_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bburger\s+patt(?:y|ies)\b", "burger patty"),
    (r"\bmeat\s*balls?\b", "meatballs"),
    (r"\bchicken\s+wings?\b|\bwings?\b", "wings"),
    (r"\bsausages?\b", "sausages"),
    (r"\bsamosas?\b", "samosa"),
    (r"\bnuggets?\b", "nuggets"),
    (r"\bpopcorn\b", "popcorn"),
    (r"\bfillets?\b", "fillet"),
    (r"\bstrips?\b", "strips"),
    (r"\bparathas?\b", "paratha"),
    (r"\bfries\b", "fries"),
    (r"\bkibbeh\b", "kibbeh"),
    (r"\bkofta\b", "kofta"),
)

_DESCRIPTOR_CONJUNCTIONS = frozenset(
    {
        "sweet and spicy",
        "hot and spicy",
        "chicken and cheese",
        "beef and herb",
    }
)


class ConjunctionType(str, Enum):
    SINGLE = "SINGLE"
    AND = "AND"
    OR = "OR"
    BUNDLE = "BUNDLE"
    PROMOTIONAL_BONUS = "PROMOTIONAL_BONUS"
    CHOICE = "CHOICE"
    MIXED_PACK = "MIXED_PACK"
    UNKNOWN_MULTI_PRODUCT = "UNKNOWN_MULTI_PRODUCT"


@dataclass(frozen=True)
class CommercialEntity:
    entity_id: str
    entity_index: int
    entity_count: int
    entity_text: str
    entity_type: str
    conjunction_type: str
    protein: tuple[str, ...]
    product_family: tuple[str, ...]
    retail_weight_g: float | None
    pack_count: int | None
    bonus_weight_g: float | None
    attribute_inheritance_flags: tuple[str, ...]
    weight_inheritance_confidence: str
    parse_confidence: float
    explicitly_stated_attributes: tuple[str, ...]
    inherited_attributes: tuple[str, ...]
    original_source_text: str
    original_source_offer_id: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["protein"] = "|".join(self.protein)
        record["product_family"] = "|".join(self.product_family)
        record["attribute_inheritance_flags"] = "|".join(
            self.attribute_inheritance_flags
        )
        record["explicitly_stated_attributes"] = "|".join(
            self.explicitly_stated_attributes
        )
        record["inherited_attributes"] = "|".join(
            self.inherited_attributes
        )
        return record


def _normalize(text: object) -> str:
    value = safe_text(text)
    value = value.replace("×", " x ").replace("Ã—", " x ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _families(text: str) -> tuple[str, ...]:
    normalized = text.lower()
    return tuple(
        family
        for pattern, family in _FAMILY_PATTERNS
        if re.search(pattern, normalized)
    )


def _weights(text: str) -> tuple[float, ...]:
    normalized = text.lower()
    normalized = re.sub(r"\bgrams?\b", "g", normalized)
    normalized = re.sub(r"\bkilograms?\b", "kg", normalized)
    values: list[float] = []
    for amount, unit in re.findall(
        rf"(\d+(?:\.\d+)?)\s*({ALL_UNITS})\b", normalized
    ):
        dimension, factor = unit_dim_value(unit)
        if dimension == "weight":
            values.append(round(float(amount) * factor, 3))
    return tuple(values)


def _source_id(row: Mapping[str, Any]) -> str:
    for column in ("source_offer_id", "offerid", "offer_group_id"):
        value = safe_text(row.get(column, ""))
        if value:
            return value
    digest = hashlib.sha256(
        _normalize(row.get("Offer Name", "")).encode("utf-8")
    ).hexdigest()
    return f"offer-{digest[:16]}"


def _separator(text: str) -> tuple[str | None, ConjunctionType]:
    lowered = text.lower()
    for phrase in _DESCRIPTOR_CONJUNCTIONS:
        if phrase in lowered and len(_families(text)) <= 1:
            return None, ConjunctionType.SINGLE
    patterns = (
        (r"\s+\+\s+", ConjunctionType.BUNDLE),
        (r"\s+or\s+", ConjunctionType.OR),
        (r"\s+with\s+", ConjunctionType.AND),
        (r"\s+and\s+", ConjunctionType.AND),
        (r"\s*/\s*", ConjunctionType.UNKNOWN_MULTI_PRODUCT),
        (r"\s*,\s*", ConjunctionType.AND),
        (r"\s*&\s*", ConjunctionType.AND),
    )
    for pattern, conjunction in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            if conjunction is ConjunctionType.BUNDLE and re.search(
                rf"\+\s*\d+(?:\.\d+)?\s*(?:{ALL_UNITS})\s*(?:free)?\b",
                lowered,
            ):
                return None, ConjunctionType.SINGLE
            if conjunction is ConjunctionType.BUNDLE and re.search(
                r"\+\s*free\b", lowered
            ):
                conjunction = ConjunctionType.PROMOTIONAL_BONUS
            return pattern, conjunction
    return None, ConjunctionType.SINGLE


def _should_split(parts: list[str], full_text: str) -> bool:
    if len(parts) < 2:
        return False
    family_sets = [set(_families(part)) for part in parts]
    protein_sets = [_protein_set(part) for part in parts]
    full_families = set(_families(full_text))
    informative = [
        bool(families or proteins)
        for families, proteins in zip(family_sets, protein_sets, strict=True)
    ]
    if sum(informative) < 2:
        return False
    if len(full_families) == 1:
        only_family = next(iter(full_families))
        distinct_proteins = set().union(*protein_sets)
        if len(distinct_proteins) > 1:
            return True
        if sum(bool(families) for families in family_sets) >= 2:
            return True
        # "Chicken and Cheese Sausages" has only one protein-bearing side and
        # one independently non-family descriptor side.
        return False
    return sum(bool(families) for families in family_sets) >= 2


def _entity_text(
    part: str,
    proteins: tuple[str, ...],
    families: tuple[str, ...],
    weight: float | None,
    *,
    bonus: bool,
) -> str:
    tokens: list[str] = []
    if bonus:
        tokens.append("free")
    tokens.extend(proteins)
    tokens.extend(families)
    semantic_words = {
        (
            token[:-3] + "y"
            if token.endswith("ies")
            else token[:-1]
            if token.endswith("s")
            else token
        )
        for value in (*proteins, *families)
        for token in value.split()
    }
    normalized_part = _normalize(part).lower()
    for token in re.findall(r"[a-z0-9]+", normalized_part):
        canonical = (
            token[:-3] + "y"
            if token.endswith("ies")
            else token[:-1]
            if token.endswith("s")
            else token
        )
        if (
            canonical not in semantic_words
            and token not in tokens
            and token not in {"al", "kabeer"}
        ):
            tokens.append(token)
    if weight is not None and not _weights(part):
        tokens.append(f"{weight:g} g")
    return " ".join(tokens).strip()


def decompose_commercial_entities(
    row: Mapping[str, Any],
) -> tuple[CommercialEntity, ...]:
    """Return one or more independently matchable entities for a source row."""
    original = _normalize(row.get("Offer Name", row.get("offer_text", "")))
    source_id = _source_id(row)
    separator, conjunction = _separator(original)
    if separator is None:
        attributes = parse_source_attributes(row)
        entity = CommercialEntity(
            entity_id=f"{source_id}_1",
            entity_index=1,
            entity_count=1,
            entity_text=original,
            entity_type=(
                "MIXED_PACK"
                if attributes.bundle_structure == "MULTI_PRODUCT"
                else "SINGLE_PRODUCT"
            ),
            conjunction_type=ConjunctionType.SINGLE.value,
            protein=attributes.protein,
            product_family=attributes.family,
            retail_weight_g=attributes.base_measure,
            pack_count=attributes.pack_count,
            bonus_weight_g=attributes.bonus_measure,
            attribute_inheritance_flags=(),
            weight_inheritance_confidence="NOT_APPLICABLE",
            parse_confidence=attributes.confidence,
            explicitly_stated_attributes=("source_text",),
            inherited_attributes=(),
            original_source_text=original,
            original_source_offer_id=source_id,
        )
        return (entity,)

    parts = [
        part.strip(" ,/+&")
        for part in re.split(separator, original, flags=re.IGNORECASE)
        if part.strip(" ,/+&")
    ]
    if not _should_split(parts, original):
        attributes = parse_source_attributes(row)
        return (
            CommercialEntity(
                entity_id=f"{source_id}_1",
                entity_index=1,
                entity_count=1,
                entity_text=original,
                entity_type="SINGLE_PRODUCT",
                conjunction_type=ConjunctionType.SINGLE.value,
                protein=attributes.protein,
                product_family=attributes.family,
                retail_weight_g=attributes.base_measure,
                pack_count=attributes.pack_count,
                bonus_weight_g=attributes.bonus_measure,
                attribute_inheritance_flags=(),
                weight_inheritance_confidence="NOT_APPLICABLE",
                parse_confidence=attributes.confidence,
                explicitly_stated_attributes=("source_text",),
                inherited_attributes=(),
                original_source_text=original,
                original_source_offer_id=source_id,
            ),
        )

    full_families = _families(original)
    full_weights = _weights(original)
    full_proteins = tuple(sorted(_protein_set(original)))
    entities: list[CommercialEntity] = []
    for position, part in enumerate(parts, start=1):
        explicit_families = _families(part)
        explicit_proteins = tuple(sorted(_protein_set(part)))
        explicit_weights = _weights(part)
        families = explicit_families or (
            full_families if len(set(full_families)) == 1 else ()
        )
        proteins = explicit_proteins or (
            full_proteins if len(set(full_proteins)) == 1 else ()
        )
        inherited: list[str] = []
        flags: list[str] = []
        if families and not explicit_families:
            inherited.append("product_family")
            flags.append("family:inherited_shared")
        if proteins and not explicit_proteins:
            inherited.append("protein")
            flags.append("protein:inherited_shared")
        if explicit_weights:
            weight = explicit_weights[0]
            weight_confidence = "HIGH"
        elif len(set(full_weights)) == 1:
            weight = full_weights[0]
            inherited.append("retail_weight_g")
            flags.append("retail_weight_g:inherited_shared")
            distinct_families = len(set(full_families)) > 1
            weight_confidence = "LOW" if distinct_families else "HIGH"
        else:
            weight = None
            weight_confidence = "LOW"
        bonus = (
            conjunction is ConjunctionType.PROMOTIONAL_BONUS
            and position > 1
        )
        explicit = ["source_segment"]
        if explicit_families:
            explicit.append("product_family")
        if explicit_proteins:
            explicit.append("protein")
        if explicit_weights:
            explicit.append("retail_weight_g")
        confidence = 0.9
        if weight_confidence == "LOW" or not families:
            confidence -= 0.3
        entity = CommercialEntity(
            entity_id=f"{source_id}_{position}",
            entity_index=position,
            entity_count=len(parts),
            entity_text=_entity_text(
                part,
                proteins,
                families,
                weight,
                bonus=bonus,
            ),
            entity_type=(
                "PROMOTIONAL_BONUS" if bonus else "COMMERCIAL_PRODUCT"
            ),
            conjunction_type=conjunction.value,
            protein=proteins,
            product_family=families,
            retail_weight_g=weight,
            pack_count=None,
            bonus_weight_g=None,
            attribute_inheritance_flags=tuple(flags),
            weight_inheritance_confidence=weight_confidence,
            parse_confidence=round(max(0.0, confidence), 2),
            explicitly_stated_attributes=tuple(explicit),
            inherited_attributes=tuple(inherited),
            original_source_text=original,
            original_source_offer_id=source_id,
        )
        entities.append(entity)
    return tuple(entities)


def expand_offer_entities(rows: Any, *, progress: Any = None) -> Any:
    """Expand a DataFrame to one normalized row per commercial entity.

    ``progress`` is an optional ``callable(completed, total)`` invoked while
    the rows are decomposed. Decomposition is per-row work that can run for
    minutes on a large upload, so a caller driving a progress bar needs to
    observe it rather than wait for the whole frame.
    """
    import pandas as pd

    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    expanded: list[dict[str, Any]] = []
    total_rows = len(rows)
    update_interval = max(1, min(2_000, total_rows // 50 or 1))
    for processed, (_, source) in enumerate(rows.iterrows(), start=1):
        source_record = source.to_dict()
        entities = decompose_commercial_entities(source_record)
        for entity in entities:
            record = dict(source_record)
            record.update(
                {
                    "source_offer_id": entity.original_source_offer_id,
                    "source_offer_text": entity.original_source_text,
                    "entity_id": entity.entity_id,
                    "entity_index": entity.entity_index,
                    "entity_count": entity.entity_count,
                    "entity_text": entity.entity_text,
                    "entity_type": entity.entity_type,
                    "conjunction_type": entity.conjunction_type,
                    "entity_protein": "|".join(entity.protein),
                    "entity_product_family": "|".join(
                        entity.product_family
                    ),
                    "entity_retail_weight_g": entity.retail_weight_g,
                    "entity_pack_count": entity.pack_count,
                    "entity_bonus_weight_g": entity.bonus_weight_g,
                    "attribute_inheritance_flags": "|".join(
                        entity.attribute_inheritance_flags
                    ),
                    "weight_inheritance_confidence": (
                        entity.weight_inheritance_confidence
                    ),
                    "entity_parse_confidence": entity.parse_confidence,
                    "entity_attributes_json": json.dumps(
                        entity.to_record(), sort_keys=True
                    ),
                }
            )
            if len(entities) > 1:
                record["Offer Name"] = entity.entity_text
                record["Product"] = " ".join(entity.product_family)
                record["Variant"] = ""
                record["Base Packsize"] = (
                    f"{entity.retail_weight_g:g} g"
                    if entity.retail_weight_g is not None
                    and math.isfinite(entity.retail_weight_g)
                    else ""
                )
                # All candidate retrieval and feature extraction must operate
                # on the entity, never on stale values parsed from the combined
                # source offer.
                entity_measure_text = (
                    f"{record['Base Packsize']} {entity.entity_text}"
                )
                detailed = extract_flyer_measures(entity_measure_text)
                record["offer_measures_detailed"] = detailed
                record["offer_measures"] = collapse_to_simple(detailed)
                record["match_text"] = clean_offer_text(
                    f"{entity.entity_text} {record['Product']}"
                )
                # A mixed source row can span catalogue categories.  Using the
                # original row category would silently hide valid candidates.
                # The existing "Other" category deliberately searches the
                # bounded full catalogue; commercial compatibility remains the
                # authoritative filter.
                record["category"] = "Other"
                record["product_family"] = record["Product"]
            expanded.append(record)
        if progress is not None and (
            processed == total_rows or processed % update_interval == 0
        ):
            progress(processed, total_rows)
    return pd.DataFrame(expanded)
