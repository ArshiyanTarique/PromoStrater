"""Fail-fast validation for external SKU-mapping inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from sku_mapping.schemas import (
    CLICKFLYER_SCHEMA,
    GOLD_PAIRS_SCHEMA,
    MODEL_PACKAGE_REQUIRED_KEYS,
    PRODUCT_MASTER_SCHEMA,
    TableSchema,
)


class InputValidationError(ValueError):
    """Raised when an input does not satisfy the documented schema."""


class EmptyInputError(InputValidationError):
    """Raised when an input has no data records."""


class DuplicateItemCodeError(InputValidationError):
    """Raised when Product Master identifiers are not unique."""


class ModelPackageValidationError(InputValidationError):
    """Raised when a deserialized model package lacks its basic contract."""


def normalize_itemcode(value: object) -> str | pd.NA:
    """Normalize an identifier without dropping textual leading zeros."""
    if value is None or pd.isna(value):
        return pd.NA
    normalized = str(value).strip()
    if not normalized:
        return pd.NA
    if normalized.endswith(".0") and normalized[:-2].isdigit():
        normalized = normalized[:-2]
    return normalized


def normalize_itemcode_series(values: pd.Series) -> pd.Series:
    """Normalize an Itemcode column using pandas' nullable string dtype."""
    return values.map(normalize_itemcode).astype("string")


def _validate_table_shape(frame: pd.DataFrame, schema: TableSchema) -> None:
    if frame.empty:
        raise EmptyInputError(f"{schema.name} contains no data rows")
    missing = [column for column in schema.required_columns if column not in frame.columns]
    if missing:
        raise InputValidationError(
            f"{schema.name} is missing required columns: {', '.join(missing)}"
        )


def _validate_nonempty_identifier(frame: pd.DataFrame, column: str, schema_name: str) -> None:
    missing_count = int(frame[column].isna().sum())
    if missing_count:
        raise InputValidationError(
            f"{schema_name} contains {missing_count} missing or blank '{column}' values"
        )


def validate_clickflyer(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a ClickFlyer table without changing its columns or rows."""
    _validate_table_shape(frame, CLICKFLYER_SCHEMA)
    return frame


def validate_product_master(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate master schema and reject duplicate or missing item codes."""
    _validate_table_shape(frame, PRODUCT_MASTER_SCHEMA)
    validated = frame.copy()
    validated["Itemcode"] = normalize_itemcode_series(validated["Itemcode"])
    _validate_nonempty_identifier(validated, "Itemcode", PRODUCT_MASTER_SCHEMA.name)
    duplicate_codes = sorted(
        validated.loc[validated["Itemcode"].duplicated(keep=False), "Itemcode"].unique()
    )
    if duplicate_codes:
        preview = ", ".join(map(str, duplicate_codes[:10]))
        suffix = "..." if len(duplicate_codes) > 10 else ""
        raise DuplicateItemCodeError(
            f"{PRODUCT_MASTER_SCHEMA.name} contains duplicate Itemcode values: {preview}{suffix}"
        )
    return validated


def _coerce_binary_training_flag(value: object) -> int:
    if value is None or pd.isna(value):
        raise InputValidationError("use_for_binary_pair_training cannot be missing")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) in {0.0, 1.0}:
            return int(value)
    normalized = str(value).strip().lower()
    mapping = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
    }
    if normalized not in mapping:
        raise InputValidationError(
            "use_for_binary_pair_training must contain only 0/1 or boolean values"
        )
    return mapping[normalized]


def validate_gold_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate gold-pair schema and coerce its binary-training flag."""
    _validate_table_shape(frame, GOLD_PAIRS_SCHEMA)
    validated = frame.copy()
    validated["master_itemcode"] = normalize_itemcode_series(validated["master_itemcode"])
    _validate_nonempty_identifier(validated, "master_itemcode", GOLD_PAIRS_SCHEMA.name)
    invalid_labels = validated["pair_label"].dropna().loc[
        ~validated["pair_label"].dropna().isin([0, 1, -1])
    ]
    if not invalid_labels.empty:
        values = ", ".join(map(str, sorted(invalid_labels.unique())))
        raise InputValidationError(
            f"{GOLD_PAIRS_SCHEMA.name} has unsupported pair_label values: {values}"
        )
    validated["use_for_binary_pair_training"] = validated[
        "use_for_binary_pair_training"
    ].map(_coerce_binary_training_flag).astype("int8")
    return validated


def validate_gold_pairs_for_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate gold table structure while preserving row-level errors for reporting.

    Phase 4 must write invalid labels, flags, and missing master identifiers to
    the rejected-row report. The strict ``validate_gold_pairs`` API remains the
    default for callers that need fail-fast row validation.
    """
    _validate_table_shape(frame, GOLD_PAIRS_SCHEMA)
    validated = frame.copy()
    validated["master_itemcode"] = normalize_itemcode_series(validated["master_itemcode"])
    return validated


def validate_model_package(package: object) -> Mapping[str, Any]:
    """Validate the minimal model-package contract without invoking the model."""
    if not isinstance(package, Mapping):
        raise ModelPackageValidationError("Model package must deserialize to a mapping")
    missing = [key for key in MODEL_PACKAGE_REQUIRED_KEYS if key not in package]
    if missing:
        raise ModelPackageValidationError(
            f"Model package is missing required keys: {', '.join(missing)}"
        )
    if not isinstance(package["feature_columns"], (list, tuple)):
        raise ModelPackageValidationError("Model package feature_columns must be a list or tuple")
    return package
