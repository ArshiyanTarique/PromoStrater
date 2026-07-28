"""Governance audit for gold SKU-matching pairs."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any

import pandas as pd

from sku_mapping.data.validators import normalize_itemcode
from sku_mapping.features.text_features import safe_text


def normalized_offer_text(value: object) -> str:
    """Normalize text only for exact identity checks, never fuzzy matching."""
    return re.sub(r"\s+", " ", safe_text(value).strip().lower())


def normalized_pair_label(value: object) -> int | None:
    """Return an integral label when represented exactly, otherwise ``None``."""
    if value is None or pd.isna(value) or isinstance(value, bool):
        return None
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def normalized_training_flag(value: object) -> int | None:
    """Coerce documented binary flag representations without guessing."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    values = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
    }
    return values.get(normalized)


def _distribution(values: pd.Series) -> dict[str, int]:
    normalized = values.astype("string").fillna("<missing>")
    counts = normalized.value_counts(dropna=False, sort=False)
    return {
        str(key): int(value)
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def _imbalance(values: pd.Series) -> dict[str, Any]:
    distribution = _distribution(values)
    total = sum(distribution.values())
    majority = max(distribution.values(), default=0)
    return {
        "distribution": distribution,
        "majority_fraction": round(majority / total, 6) if total else None,
    }


def _records(frame: pd.DataFrame, columns: list[str], limit: int = 100) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    records = frame.loc[:, available].head(limit).astype(object)
    records = records.where(pd.notna(records), None)
    return records.to_dict(orient="records")


def audit_training_data(
    gold_pairs: pd.DataFrame,
    master_itemcodes: Collection[str],
) -> dict[str, Any]:
    """Audit gold-pair quality without deleting or mutating source rows."""
    audited = gold_pairs.copy()
    audited["_normalized_offer_text"] = audited["offer_text"].map(normalized_offer_text)
    audited["_normalized_master_itemcode"] = audited["master_itemcode"].map(
        normalize_itemcode
    )
    audited["_normalized_pair_label"] = audited["pair_label"].map(normalized_pair_label)
    pair_columns = ["offer_group_id", "_normalized_master_itemcode"]

    duplicate_mask = audited.duplicated(pair_columns, keep=False)
    duplicate_rows = audited.loc[duplicate_mask]
    duplicate_groups = duplicate_rows.drop_duplicates(pair_columns)

    label_counts = audited.groupby(pair_columns, dropna=False)[
        "_normalized_pair_label"
    ].nunique(dropna=True)
    conflicting_index = label_counts.loc[label_counts > 1].index
    conflicting = audited.set_index(pair_columns).loc[conflicting_index].reset_index()

    invalid_labels = audited.loc[
        ~audited["_normalized_pair_label"].isin([-1, 0, 1])
    ]
    known_codes = {str(code) for code in master_itemcodes}
    unknown_master = audited.loc[
        audited["_normalized_master_itemcode"].isna()
        | ~audited["_normalized_master_itemcode"].astype("string").isin(known_codes)
    ]
    null_offer = audited.loc[audited["_normalized_offer_text"].eq("")]

    split_conflicts = pd.DataFrame()
    if "recommended_split" in audited.columns:
        split_values = audited.assign(
            _split=audited["recommended_split"].astype("string").str.strip()
        )
        split_counts = split_values.loc[
            split_values["_split"].notna() & split_values["_split"].ne("")
        ].groupby("offer_group_id", dropna=False)["_split"].nunique()
        groups = split_counts.loc[split_counts > 1].index
        split_conflicts = split_values.loc[
            split_values["offer_group_id"].isin(groups)
        ]

    synthetic_repeats = pd.DataFrame()
    if "source_dataset" in audited.columns or "label_provenance" in audited.columns:
        source = (
            audited["source_dataset"].astype("string").fillna("")
            if "source_dataset" in audited.columns
            else pd.Series("", index=audited.index, dtype="string")
        )
        provenance = (
            audited["label_provenance"].astype("string").fillna("")
            if "label_provenance" in audited.columns
            else pd.Series("", index=audited.index, dtype="string")
        )
        synthetic = audited.loc[
            source.str.contains("synthetic", case=False)
            | provenance.str.contains("synthetic", case=False)
        ]
        synthetic_repeats = synthetic.loc[
            synthetic.duplicated("_normalized_offer_text", keep=False)
            & synthetic["_normalized_offer_text"].ne("")
        ]

    exact_duplicate_mask = audited.drop(
        columns=[
            "_normalized_offer_text",
            "_normalized_master_itemcode",
            "_normalized_pair_label",
        ]
    ).duplicated(keep=False)
    exact_duplicates = audited.loc[exact_duplicate_mask]

    class_values = audited["_normalized_pair_label"].astype("Int64")
    source_values = (
        audited["source_dataset"]
        if "source_dataset" in audited.columns
        else pd.Series(pd.NA, index=audited.index, dtype="string")
    )
    provenance_values = (
        audited["label_provenance"]
        if "label_provenance" in audited.columns
        else pd.Series(pd.NA, index=audited.index, dtype="string")
    )

    return {
        "total_rows": int(len(audited)),
        "policy": {
            "duplicate_pairs": "report_and_retain_when_otherwise_eligible",
            "conflicting_labels": "report_and_retain_when_otherwise_eligible",
            "exact_duplicates": "report_and_retain_when_otherwise_eligible",
            "invalid_or_ineligible_rows": "write_to_rejected_training_rows",
        },
        "duplicate_offer_master_pairs": {
            "group_count": int(len(duplicate_groups)),
            "row_count": int(len(duplicate_rows)),
            "examples": _records(
                duplicate_rows,
                ["record_id", "offer_group_id", "offer_text", "master_itemcode", "pair_label"],
            ),
        },
        "conflicting_labels": {
            "pair_count": int(len(conflicting_index)),
            "row_count": int(len(conflicting)),
            "examples": _records(
                conflicting,
                ["record_id", "offer_group_id", "offer_text", "master_itemcode", "pair_label"],
            ),
        },
        "offer_groups_across_recommended_splits": {
            "group_count": int(split_conflicts["offer_group_id"].nunique())
            if not split_conflicts.empty
            else 0,
            "row_count": int(len(split_conflicts)),
            "examples": _records(
                split_conflicts,
                ["record_id", "offer_group_id", "recommended_split"],
            ),
        },
        "invalid_labels": {
            "row_count": int(len(invalid_labels)),
            "examples": _records(
                invalid_labels,
                ["record_id", "offer_group_id", "master_itemcode", "pair_label"],
            ),
        },
        "unknown_master_skus": {
            "row_count": int(len(unknown_master)),
            "unique_codes": sorted(
                str(code)
                for code in unknown_master["_normalized_master_itemcode"].dropna().unique()
            ),
            "examples": _records(
                unknown_master,
                ["record_id", "offer_group_id", "master_itemcode"],
            ),
        },
        "null_offer_text": {
            "row_count": int(len(null_offer)),
            "examples": _records(
                null_offer,
                ["record_id", "offer_group_id", "master_itemcode"],
            ),
        },
        "class_imbalance": _imbalance(class_values),
        "source_imbalance": _imbalance(source_values),
        "provenance_imbalance": _imbalance(provenance_values),
        "repeated_synthetic_templates": {
            "template_count": int(
                synthetic_repeats["_normalized_offer_text"].nunique()
            )
            if not synthetic_repeats.empty
            else 0,
            "row_count": int(len(synthetic_repeats)),
            "examples": _records(
                synthetic_repeats,
                ["record_id", "offer_group_id", "offer_text", "master_itemcode"],
            ),
        },
        "exact_duplicate_rows": {
            "row_count": int(len(exact_duplicates)),
            "examples": _records(
                exact_duplicates,
                ["record_id", "offer_group_id", "offer_text", "master_itemcode", "pair_label"],
            ),
        },
    }
