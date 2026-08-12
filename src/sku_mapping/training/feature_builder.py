"""Build the governed 19-feature binary-pair training dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.data.loaders import (
    load_clickflyer,
    load_gold_pairs_for_audit,
    load_product_master,
)
from sku_mapping.data.preprocessing import preprocess_product_master
from sku_mapping.data.validators import (
    normalize_itemcode,
    validate_gold_pairs_for_audit,
    validate_product_master,
)
from sku_mapping.features import build_feature_vector, build_feature_vector_from_text
from sku_mapping.features.text_features import safe_text
from sku_mapping.paths import PROJECT_ROOT
from sku_mapping.training.data_audit import (
    audit_training_data,
    normalized_offer_text,
    normalized_pair_label,
    normalized_training_flag,
)

ACCEPTED_METADATA_COLUMNS = (
    "record_id",
    "offer_group_id",
    "offer_text",
    "master_itemcode",
    "pair_label",
    "source_dataset",
    "label_provenance",
    "label_confidence",
    "recommended_split",
    "split_group",
    "product_class_offer",
    "variant_offer",
)
FLYER_FEATURE_COLUMNS = ("Offer Name", "Product", "Variant", "Base Packsize")


@dataclass
class TrainingFeatureBuildResult:
    """In-memory tables and metadata produced before or during output writing."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame
    audit: dict[str, Any]
    manifest: dict[str, Any]
    eligible_binary_rows: int
    offer_reconstruction_counts: dict[str, int]
    output_paths: dict[str, Path] = field(default_factory=dict)


def _distribution(values: pd.Series) -> dict[str, int]:
    counts = values.astype("string").fillna("<missing>").value_counts(sort=False)
    return {
        str(key): int(value)
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def _flyer_metadata_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(safe_text(row.get(column, "")).strip() for column in FLYER_FEATURE_COLUMNS)


def _store_unique(
    lookup: dict[object, tuple[str, str, str, str] | None],
    key: object,
    metadata: tuple[str, str, str, str],
) -> None:
    if not key:
        return
    existing = lookup.get(key)
    if key not in lookup:
        lookup[key] = metadata
    elif existing != metadata:
        lookup[key] = None


def _build_reliable_flyer_lookup(
    clickflyer: pd.DataFrame | None,
) -> tuple[
    dict[str, tuple[str, str, str, str] | None],
    dict[tuple[str, str], tuple[str, str, str, str] | None],
]:
    """Index exact flyer identities; ambiguous metadata is represented by ``None``."""
    if clickflyer is None:
        return {}, {}
    missing = [column for column in FLYER_FEATURE_COLUMNS if column not in clickflyer.columns]
    if missing:
        raise ValueError(
            "ClickFlyer enrichment data is missing required columns: "
            + ", ".join(missing)
        )
    by_text: dict[str, tuple[str, str, str, str] | None] = {}
    by_text_retailer: dict[
        tuple[str, str], tuple[str, str, str, str] | None
    ] = {}
    has_retailer = "Retailer Name" in clickflyer.columns
    columns = list(FLYER_FEATURE_COLUMNS) + (["Retailer Name"] if has_retailer else [])
    for values in clickflyer.loc[:, columns].itertuples(index=False, name=None):
        metadata = tuple(safe_text(value).strip() for value in values[:4])
        text_key = normalized_offer_text(metadata[0])
        _store_unique(by_text, text_key, metadata)
        if has_retailer:
            retailer_key = normalized_offer_text(values[4])
            if retailer_key:
                _store_unique(by_text_retailer, (text_key, retailer_key), metadata)
    return by_text, by_text_retailer


def _reliable_flyer_row(
    gold_row: Mapping[str, Any],
    by_text: dict[str, tuple[str, str, str, str] | None],
    by_text_retailer: dict[
        tuple[str, str], tuple[str, str, str, str] | None
    ],
) -> dict[str, str] | None:
    text_key = normalized_offer_text(gold_row.get("offer_text"))
    retailer_key = normalized_offer_text(gold_row.get("retailer"))
    metadata: tuple[str, str, str, str] | None
    if retailer_key:
        metadata = by_text_retailer.get((text_key, retailer_key))
    else:
        metadata = by_text.get(text_key)
    if metadata is None:
        return None
    return dict(zip(FLYER_FEATURE_COLUMNS, metadata))


def _is_synthetic_gold_row(row: Mapping[str, Any]) -> bool:
    """Identify synthetic provenance that must use the text-only feature API."""
    provenance = " ".join(
        [
            safe_text(row.get("source_dataset")),
            safe_text(row.get("label_provenance")),
        ]
    ).lower()
    return "synthetic" in provenance


def _row_rejection_reasons(
    row: Mapping[str, Any],
    known_master_codes: set[str],
) -> tuple[list[str], int | None, int | None, str | None]:
    flag = normalized_training_flag(row.get("use_for_binary_pair_training"))
    label = normalized_pair_label(row.get("pair_label"))
    code_value = normalize_itemcode(row.get("master_itemcode"))
    code = None if pd.isna(code_value) else str(code_value)
    reasons: list[str] = []
    if flag is None:
        reasons.append("invalid_use_for_binary_pair_training")
    elif flag != 1:
        reasons.append("use_for_binary_pair_training_not_1")
    if label is None or label not in {-1, 0, 1}:
        reasons.append("invalid_pair_label")
    elif label == -1:
        reasons.append("abstain_pair_label")
    if not normalized_offer_text(row.get("offer_text")):
        reasons.append("null_offer_text")
    if code is None:
        reasons.append("missing_master_itemcode")
    elif code not in known_master_codes:
        reasons.append("unknown_master_itemcode")
    return reasons, flag, label, code


def build_training_feature_dataset(
    gold_pairs: pd.DataFrame,
    product_master: pd.DataFrame,
    clickflyer: pd.DataFrame | None = None,
    *,
    input_filenames: Mapping[str, str] | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> TrainingFeatureBuildResult:
    """Build accepted/rejected training rows without reading or writing files."""
    gold = validate_gold_pairs_for_audit(gold_pairs)
    master = preprocess_product_master(validate_product_master(product_master))
    master_lookup = {
        str(row["Itemcode"]): row
        for _, row in master.iterrows()
    }
    known_master_codes = set(master_lookup)
    audit = audit_training_data(gold, known_master_codes)
    by_text, by_text_retailer = _build_reliable_flyer_lookup(clickflyer)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    reconstruction_counts: Counter[str] = Counter()
    eligible_binary_rows = 0

    metadata_columns = [
        column for column in ACCEPTED_METADATA_COLUMNS if column in gold.columns
    ]
    for _, gold_row in gold.iterrows():
        reasons, flag, label, master_code = _row_rejection_reasons(
            gold_row, known_master_codes
        )
        if flag == 1 and label in {0, 1}:
            eligible_binary_rows += 1
        if reasons:
            rejected = gold_row.to_dict()
            rejected["rejection_reason"] = ";".join(reasons)
            rejected_rows.append(rejected)
            exclusion_reasons.update(reasons)
            continue

        # The lookup is deliberately and exclusively keyed by the gold code.
        master_row = master_lookup[master_code]  # type: ignore[index]
        flyer_row = None
        if not _is_synthetic_gold_row(gold_row):
            flyer_row = _reliable_flyer_row(gold_row, by_text, by_text_retailer)
        if flyer_row is not None:
            features = build_feature_vector(flyer_row, master_row)
            reconstruction_counts["exact_clickflyer"] += 1
        else:
            features = build_feature_vector_from_text(
                safe_text(gold_row.get("offer_text")),
                master_row,
                product=safe_text(gold_row.get("product_class_offer")),
                variant=safe_text(gold_row.get("variant_offer")),
            )
            reconstruction_counts["synthetic_or_text_fallback"] += 1

        accepted = {column: gold_row.get(column) for column in metadata_columns}
        accepted["master_itemcode"] = master_code
        accepted["pair_label"] = int(label)  # type: ignore[arg-type]
        accepted.update(features)
        accepted_rows.append(accepted)

    accepted_columns = metadata_columns + list(MODEL_FEATURE_COLUMNS)
    accepted = pd.DataFrame(accepted_rows, columns=accepted_columns)
    rejected_columns = list(gold.columns) + ["rejection_reason"]
    rejected = pd.DataFrame(rejected_rows, columns=rejected_columns)
    feature_schema = {
        feature: str(accepted[feature].dtype)
        for feature in MODEL_FEATURE_COLUMNS
    }
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_filenames": dict(input_filenames or {}),
        "input_hashes": dict(input_hashes or {}),
        "total_rows": int(len(gold)),
        "eligible_binary_rows": int(eligible_binary_rows),
        "accepted_rows": int(len(accepted)),
        "rejected_rows": int(len(rejected)),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "class_distribution": _distribution(accepted["pair_label"])
        if not accepted.empty
        else {},
        "source_distribution": _distribution(accepted["source_dataset"])
        if "source_dataset" in accepted.columns
        else {},
        "provenance_distribution": _distribution(accepted["label_provenance"])
        if "label_provenance" in accepted.columns
        else {},
        "offer_reconstruction": dict(sorted(reconstruction_counts.items())),
        "feature_schema": feature_schema,
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "output_hashes": {},
    }
    return TrainingFeatureBuildResult(
        accepted=accepted,
        rejected=rejected,
        audit=audit,
        manifest=manifest,
        eligible_binary_rows=eligible_binary_rows,
        offer_reconstruction_counts=dict(sorted(reconstruction_counts.items())),
    )


def sha256_file(path: str | Path) -> str:
    """Compute a streaming SHA-256 digest for an input or generated artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_training_feature_outputs(
    result: TrainingFeatureBuildResult,
    output_dir: str | Path,
    *,
    output_encoding: str = "utf-8-sig",
) -> dict[str, Path]:
    """Write accepted, rejected, manifest, and audit artifacts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_features_parquet": destination / "training_features.parquet",
        "training_features_csv": destination / "training_features.csv",
        "rejected_training_rows_csv": destination / "rejected_training_rows.csv",
        "training_feature_manifest_json": destination
        / "training_feature_manifest.json",
        "training_data_audit_json": destination / "training_data_audit.json",
    }
    result.accepted.to_parquet(paths["training_features_parquet"], index=False)
    result.accepted.to_csv(
        paths["training_features_csv"], index=False, encoding=output_encoding
    )
    result.rejected.to_csv(
        paths["rejected_training_rows_csv"], index=False, encoding=output_encoding
    )
    _write_json(paths["training_data_audit_json"], result.audit)

    hash_keys = (
        "training_features_parquet",
        "training_features_csv",
        "rejected_training_rows_csv",
        "training_data_audit_json",
    )
    result.manifest["output_hashes"] = {
        key: sha256_file(paths[key]) for key in hash_keys
    }
    _write_json(paths["training_feature_manifest_json"], result.manifest)
    result.output_paths = paths
    return paths


def build_training_features_from_paths(
    gold_path: str | Path,
    master_path: str | Path,
    *,
    clickflyer_path: str | Path | None = None,
    output_dir: str | Path = PROJECT_ROOT / "data" / "processed",
    output_encoding: str = "utf-8-sig",
) -> TrainingFeatureBuildResult:
    """Load validated inputs, build feature rows, and persist Phase 4 outputs."""
    resolved_inputs = {
        "gold_pairs": Path(gold_path),
        "product_master": Path(master_path),
    }
    clickflyer = None
    if clickflyer_path is not None:
        resolved_inputs["clickflyer"] = Path(clickflyer_path)
        clickflyer = load_clickflyer(clickflyer_path)
    filenames = {name: path.name for name, path in resolved_inputs.items()}
    hashes = {name: sha256_file(path) for name, path in resolved_inputs.items()}
    result = build_training_feature_dataset(
        load_gold_pairs_for_audit(gold_path),
        load_product_master(master_path),
        clickflyer,
        input_filenames=filenames,
        input_hashes=hashes,
    )
    write_training_feature_outputs(
        result, output_dir, output_encoding=output_encoding
    )
    return result
