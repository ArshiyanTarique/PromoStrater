"""Deterministic template normalization and transitive leakage grouping."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS

SYNTHETIC_PROVENANCE_CATEGORIES = frozenset({"synthetic", "rule_generated"})

_WEIGHT_UNITS = re.compile(
    r"(?<![a-z])(?:kilograms?|kilos?|kgs?|kg|grams?|gms?|gm|gr|g)\b",
    flags=re.IGNORECASE,
)
_VOLUME_UNITS = re.compile(
    r"(?<![a-z])(?:millilit(?:er|re)s?|ml|lit(?:er|re)s?|ltrs?|ltr|l)\b",
    flags=re.IGNORECASE,
)
_PACK_WORDS = re.compile(
    r"\b(?:packets?|pkts?|packs?|pks?|pieces?|pcs?|counts?|cts?)\b",
    flags=re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![a-z])\d+(?:\.\d+)?(?![a-z])", flags=re.IGNORECASE)


def normalize_offer_template(offer_text: object) -> str:
    """Return a canonical, business-preserving offer template.

    Rules are deliberately narrow: case, multiplication glyphs, measurement
    unit spellings, pack-count spellings, punctuation, quantities, bonus-plus
    notation, and whitespace are normalized. Protein, family, variant, and
    pack-type words are retained.
    """
    text = "" if offer_text is None or pd.isna(offer_text) else str(offer_text)
    text = text.lower()
    text = text.replace("Ã—", " x ").replace("×", " x ").replace("*", " x ")
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " x ", text)
    text = re.sub(r"\s*\+\s*", " bonus_plus ", text)
    text = _WEIGHT_UNITS.sub(" weight_unit ", text)
    text = _VOLUME_UNITS.sub(" volume_unit ", text)
    text = _PACK_WORDS.sub(" pack_unit ", text)
    text = _NUMBER.sub(" quantity ", text)
    text = re.sub(r"[/|&,;:()\[\]{}]", " ", text)
    text = re.sub(r"[-_.]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_provenance(row: pd.Series | dict[str, Any]) -> str:
    """Map source/provenance metadata to an explicit reliability category."""
    source = str(row.get("source_dataset", "") or "").lower()
    provenance = str(row.get("label_provenance", "") or "").lower()
    combined = f"{source} {provenance}"
    if "synthetic" in combined:
        return "synthetic"
    if "rule_generated" in combined or "rule-generated" in combined:
        return "rule_generated"
    if "real_clickflyer_audit" in source and "contradiction" in provenance:
        return "human_audited_contradiction"
    if "audit" in combined:
        return "human_audited"
    if "clickflyer" in source and ("autolabel" in source or "forced" in provenance):
        return "clickflyer_autolabel_forced"
    if "contradiction" in combined or "forced" in combined:
        return "forced_or_contradiction"
    if "clickflyer" in source:
        return "clickflyer_autolabel"
    return "unknown"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def template_group_id_for_row(
    row: pd.Series | dict[str, Any],
    *,
    provenance_category: str | None = None,
) -> str:
    """Create a shared synthetic template ID or a row-stable real ID."""
    category = provenance_category or classify_provenance(row)
    template = normalize_offer_template(row.get("offer_text", ""))
    if category in SYNTHETIC_PROVENANCE_CATEGORIES:
        return f"template_{_sha256_text(template)}"
    identity = "|".join(
        [
            _stable_text(row.get("record_id")),
            _stable_text(row.get("offer_group_id")),
            _stable_text(row.get("master_itemcode")),
            template,
        ]
    )
    return f"real_row_{_sha256_text(identity)}"


def _serialize_feature_value(value: object) -> str:
    """Serialize a numeric feature deterministically, including missing data."""
    if value is None or pd.isna(value):
        return "NaN"
    numeric = float(value)
    if not math.isfinite(numeric):
        if math.isnan(numeric):
            return "NaN"
        return "Infinity" if numeric > 0 else "-Infinity"
    if numeric == 0.0:
        numeric = 0.0
    return numeric.hex()


def feature_vector_hash(row: pd.Series | dict[str, Any]) -> str:
    """Hash the ordered 19 numeric model inputs with stable NaN semantics."""
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Cannot hash feature vector; missing columns: {missing}")
    serialized = [_serialize_feature_value(row[column]) for column in MODEL_FEATURE_COLUMNS]
    payload = json.dumps(serialized, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class LeakageGroupingResult:
    """Augmented rows and deterministic connected-component audit."""

    frame: pd.DataFrame
    audit: dict[str, Any]


def build_leakage_groups(frame: pd.DataFrame) -> LeakageGroupingResult:
    """Create transitive components across offers, templates, and features."""
    required = ["offer_group_id", "offer_text", *MODEL_FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Leakage grouping is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot build leakage groups for an empty table")
    if frame["offer_group_id"].isna().any():
        raise ValueError("offer_group_id must be populated before leakage grouping")

    augmented = frame.copy()
    augmented["input_row_number"] = np.arange(len(augmented), dtype=np.int64)
    augmented["provenance_category"] = [
        classify_provenance(row) for _, row in augmented.iterrows()
    ]
    augmented["normalized_offer_template"] = augmented["offer_text"].map(
        normalize_offer_template
    )
    augmented["template_group_id"] = [
        template_group_id_for_row(row, provenance_category=str(category))
        for (_, row), category in zip(
            augmented.iterrows(), augmented["provenance_category"]
        )
    ]
    augmented["feature_vector_hash"] = [
        feature_vector_hash(row) for _, row in augmented.iterrows()
    ]

    union_find = _UnionFind(len(augmented))
    first_for_key: dict[tuple[str, str], int] = {}
    component_tokens: list[list[str]] = [[] for _ in range(len(augmented))]
    for position, (_, row) in enumerate(augmented.iterrows()):
        keys = [
            ("offer", _stable_text(row["offer_group_id"])),
            ("feature", str(row["feature_vector_hash"])),
        ]
        if row["provenance_category"] in SYNTHETIC_PROVENANCE_CATEGORIES:
            keys.append(("template", str(row["template_group_id"])))
        for key_type, key_value in keys:
            token = f"{key_type}:{key_value}"
            component_tokens[position].append(token)
            previous = first_for_key.setdefault((key_type, key_value), position)
            union_find.union(position, previous)

    members: dict[int, list[int]] = {}
    for position in range(len(augmented)):
        members.setdefault(union_find.find(position), []).append(position)

    leakage_ids = [""] * len(augmented)
    component_sizes: list[int] = []
    for positions in members.values():
        tokens = sorted(
            {
                token
                for position in positions
                for token in component_tokens[position]
            }
        )
        digest = _sha256_text("\n".join(tokens))
        leakage_id = f"leakage_{digest}"
        component_sizes.append(len(positions))
        for position in positions:
            leakage_ids[position] = leakage_id
    augmented["leakage_group_id"] = leakage_ids

    relevant_templates = augmented[
        augmented["provenance_category"].isin(SYNTHETIC_PROVENANCE_CATEGORIES)
    ]
    audit = {
        "rows": int(len(augmented)),
        "leakage_group_count": int(augmented["leakage_group_id"].nunique()),
        "largest_leakage_group_rows": int(max(component_sizes)),
        "multirow_leakage_groups": int(sum(size > 1 for size in component_sizes)),
        "offer_group_count": int(augmented["offer_group_id"].nunique()),
        "relevant_template_group_count": int(
            relevant_templates["template_group_id"].nunique()
        ),
        "feature_vector_hash_count": int(
            augmented["feature_vector_hash"].nunique()
        ),
        "provenance_category_distribution": {
            str(key): int(value)
            for key, value in augmented["provenance_category"]
            .value_counts()
            .sort_index()
            .items()
        },
        "normalization_rules": [
            "lowercase",
            "normalize multiplication glyphs to x",
            "preserve bonus plus as bonus_plus",
            "canonicalize weight units",
            "canonicalize volume units",
            "canonicalize pack-count units",
            "replace numeric quantities",
            "normalize punctuation and whitespace",
            "preserve product-family, protein, variant, and pack-type words",
        ],
    }
    return LeakageGroupingResult(frame=augmented, audit=audit)
