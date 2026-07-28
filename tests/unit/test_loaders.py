"""Tests for path-based loaders using small fixture inputs only."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from sku_mapping.data.loaders import (
    UnsupportedFileTypeError,
    load_clickflyer,
    load_gold_pairs,
    load_model_package,
    load_product_master,
)
from sku_mapping.data.validators import EmptyInputError, InputValidationError

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _master_row(itemcode: str = "001") -> dict[str, object]:
    return {
        "Itemcode": itemcode,
        "Itemname": "CHICKEN NUGGETS",
        "Item-Cat-2": "Chicken",
        "Item-Cat-4": "Nuggets",
        "Item Description": "Chicken nuggets",
        "Item-Spec": "400g x 20 Pkts",
    }


def test_load_clickflyer_fixture_preserves_offerid_text() -> None:
    frame = load_clickflyer(FIXTURES / "clickflyer_valid.csv")
    assert frame.loc[0, "offerid"] == "001"


def test_load_gold_fixture_normalizes_code_and_flag() -> None:
    frame = load_gold_pairs(FIXTURES / "gold_pairs_valid.csv")
    assert frame.loc[0, "master_itemcode"] == "001"
    assert frame.loc[0, "use_for_binary_pair_training"] == 1


def test_load_product_master_preserves_leading_zero_itemcode(tmp_path: Path) -> None:
    path = tmp_path / "master.xlsx"
    pd.DataFrame([_master_row("001")]).to_excel(path, index=False)
    frame = load_product_master(path)
    assert frame.loc[0, "Itemcode"] == "001"


def test_load_product_master_reports_duplicate_codes(tmp_path: Path) -> None:
    path = tmp_path / "master.xlsx"
    pd.DataFrame([_master_row("001"), _master_row("001")]).to_excel(path, index=False)
    with pytest.raises(InputValidationError, match="duplicate Itemcode"):
        load_product_master(path)


def test_empty_and_missing_files_fail_clearly(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(EmptyInputError):
        load_clickflyer(empty)
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        load_gold_pairs(tmp_path / "missing.csv")


def test_unsupported_file_types_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        load_clickflyer(path)
    with pytest.raises(UnsupportedFileTypeError):
        load_product_master(path)
    with pytest.raises(UnsupportedFileTypeError):
        load_gold_pairs(path)


def test_load_model_package_validates_structure(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump({"model": "fake", "feature_columns": ["feature"]}, path)
    assert load_model_package(path)["feature_columns"] == ["feature"]


def test_invalid_model_package_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump({"model": "fake"}, path)
    with pytest.raises(InputValidationError, match="feature_columns"):
        load_model_package(path)
