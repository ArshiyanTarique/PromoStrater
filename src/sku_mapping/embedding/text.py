"""Deterministic, information-preserving embedding text preparation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from sku_mapping.features.text_features import safe_text

TEXT_CONSTRUCTION_VERSION = "2.0.0"
TEXT_NORMALIZATION_VERSION = "2.0.0"


def normalize_embedding_text(value: object) -> str:
    """Normalize formatting while retaining commercial distinctions."""
    text = unicodedata.normalize("NFKC", safe_text(value)).lower()
    text = text.replace("Ã—", " x ").replace("×", " x ")
    text = re.sub(r"\bgrams?\b|\bgms?\b|\bgm\b", "g", text)
    text = re.sub(r"\bkilograms?\b|\bkgs?\b", "kg", text)
    text = re.sub(r"(\d)\s*kg\b", r"\1 kg", text)
    text = re.sub(r"(\d)\s*g\b", r"\1 g", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[,()]+", " ", text)
    text = re.sub(r"[;:]+", " ", text)
    text = re.sub(r"\s*[|]\s*", " | ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" |.,")


def _parts(
    fields: tuple[tuple[str, str], ...],
    row: Mapping[str, Any],
) -> str:
    prepared: list[str] = []
    for label, column in fields:
        value = normalize_embedding_text(row.get(column, ""))
        if value:
            prepared.append(f"{label}={value}")
    return " | ".join(prepared)


def _structured_parts(value: object) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    keys = (
        "protein",
        "family",
        "variants",
        "flavour",
        "spicy_state",
        "product_line",
        "commercial_format",
        "promotional",
        "base_measure",
        "bonus_measure",
        "total_measure",
        "pack_count",
        "piece_count",
        "bundle_structure",
        "source_field_conflict",
    )
    parts: list[str] = []
    for key in keys:
        item = parsed.get(key)
        if item in (None, "", [], (), False):
            continue
        if isinstance(item, list):
            item = " ".join(map(str, item))
        parts.append(f"{key}={normalize_embedding_text(item)}")
    return " ".join(parts)


def prepare_offer_embedding_text(row: Mapping[str, Any]) -> str:
    """Construct entity-aware source text with structured commercial evidence."""
    prepared = _parts(
        (
            ("brand", "offer_brand"),
            ("entity", "entity_text"),
            ("entity_protein", "entity_protein"),
            ("entity_family", "entity_product_family"),
            ("entity_weight_g", "entity_retail_weight_g"),
            ("conjunction", "conjunction_type"),
            ("product", "offer_product"),
            ("family", "product_family"),
            ("variant", "offer_variant"),
            ("offer", "offer_text"),
            ("pack", "offer_base_packsize"),
        ),
        row,
    )
    structured = _structured_parts(
        row.get("source_commercial_attributes", "")
    )
    return (
        f"{prepared} | commercial={structured}"
        if prepared and structured
        else prepared or structured
    )


def prepare_candidate_embedding_text(row: Mapping[str, Any]) -> str:
    """Construct master text while preserving commercial unit semantics."""
    prepared = _parts(
        (
            ("brand", "master_brand"),
            ("item", "master_item_description"),
            ("family", "master_item_family"),
            ("category", "master_item_category"),
            ("description", "master_item_long_description"),
            ("pack", "master_item_spec"),
        ),
        row,
    )
    structured = _structured_parts(
        row.get("master_commercial_attributes", "")
    )
    return (
        f"{prepared} | commercial={structured}"
        if prepared and structured
        else prepared or structured
    )
