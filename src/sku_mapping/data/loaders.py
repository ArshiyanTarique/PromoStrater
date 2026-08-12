"""Path-based, validated loaders with no import-time I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sku_mapping.data.validators import (
    EmptyInputError,
    InputValidationError,
    validate_clickflyer,
    validate_gold_pairs,
    validate_gold_pairs_for_audit,
    validate_model_package,
    validate_product_master,
)


class UnsupportedFileTypeError(ValueError):
    """Raised when a loader receives an unsupported path suffix."""


def _existing_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def _read_csv(path: Path, *, dtype: dict[str, str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, dtype=dtype)
    except pd.errors.EmptyDataError as error:
        raise EmptyInputError(f"CSV file is empty: {path}") from error


def load_clickflyer(path: str | Path) -> pd.DataFrame:
    """Load and validate ClickFlyer CSV data, preserving offer identifiers as text."""
    input_path = _existing_path(path)
    if input_path.suffix.lower() != ".csv":
        raise UnsupportedFileTypeError("ClickFlyer input must be a .csv file")
    return validate_clickflyer(_read_csv(input_path, dtype={"offerid": "string"}))


def load_product_master(path: str | Path) -> pd.DataFrame:
    """Load, normalize, and validate a Product Master Excel workbook."""
    input_path = _existing_path(path)
    if input_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise UnsupportedFileTypeError("Product Master input must be an .xlsx or .xls file")
    try:
        frame = pd.read_excel(input_path, dtype={"Itemcode": "string"})
    except ValueError as error:
        raise InputValidationError(f"Unable to read Product Master workbook: {input_path}") from error
    return validate_product_master(frame)


def load_gold_pairs(path: str | Path) -> pd.DataFrame:
    """Load, normalize, and validate gold pair labels without resolving master rows."""
    input_path = _existing_path(path)
    if input_path.suffix.lower() != ".csv":
        raise UnsupportedFileTypeError("Gold training pairs input must be a .csv file")
    return validate_gold_pairs(_read_csv(input_path, dtype={"master_itemcode": "string"}))


def load_gold_pairs_for_audit(path: str | Path) -> pd.DataFrame:
    """Load gold pairs while retaining row-level errors for Phase 4 rejection."""
    input_path = _existing_path(path)
    if input_path.suffix.lower() != ".csv":
        raise UnsupportedFileTypeError("Gold training pairs input must be a .csv file")
    return validate_gold_pairs_for_audit(
        _read_csv(input_path, dtype={"master_itemcode": "string"})
    )


def load_model_package(path: str | Path) -> dict[str, Any]:
    """Load a joblib model package and validate only its structural contract."""
    input_path = _existing_path(path)
    if input_path.suffix.lower() not in {".joblib", ".pkl", ".pickle"}:
        raise UnsupportedFileTypeError("Model package must be a .joblib, .pkl, or .pickle file")
    try:
        package = joblib.load(input_path)
    except Exception as error:
        raise InputValidationError(f"Unable to load model package: {input_path}") from error
    return dict(validate_model_package(package))
