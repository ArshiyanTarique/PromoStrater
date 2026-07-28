"""Tests for input validation boundaries."""

import pandas as pd
import pytest

from sku_mapping.data.validators import (
    DuplicateItemCodeError,
    EmptyInputError,
    InputValidationError,
    normalize_itemcode,
    validate_clickflyer,
    validate_gold_pairs,
    validate_product_master,
)
from sku_mapping.schemas import (
    CLICKFLYER_SCHEMA,
    GOLD_PAIRS_SCHEMA,
    PRODUCT_MASTER_SCHEMA,
)


def _frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([{column: "x" for column in columns}])


def test_required_columns_are_enforced() -> None:
    with pytest.raises(InputValidationError, match="Offer Name"):
        validate_clickflyer(_frame(CLICKFLYER_SCHEMA.required_columns[1:]))
    with pytest.raises(InputValidationError, match="Itemname"):
        validate_product_master(_frame(("Itemcode",)))
    with pytest.raises(InputValidationError, match="offer_text"):
        validate_gold_pairs(_frame(("offer_group_id", "master_itemcode", "pair_label", "use_for_binary_pair_training")))


def test_empty_tables_fail_early() -> None:
    with pytest.raises(EmptyInputError):
        validate_clickflyer(pd.DataFrame(columns=CLICKFLYER_SCHEMA.required_columns))


def test_duplicate_product_master_codes_are_explicit() -> None:
    frame = _frame(PRODUCT_MASTER_SCHEMA.required_columns)
    duplicate = pd.concat([frame, frame], ignore_index=True)
    duplicate["Itemcode"] = ["001", "001"]
    with pytest.raises(DuplicateItemCodeError, match="001"):
        validate_product_master(duplicate)


def test_normalized_product_master_codes_are_checked_for_duplicates() -> None:
    frame = pd.concat(
        [_frame(PRODUCT_MASTER_SCHEMA.required_columns), _frame(PRODUCT_MASTER_SCHEMA.required_columns)],
        ignore_index=True,
    )
    frame["Itemcode"] = ["001.0", "001"]
    with pytest.raises(DuplicateItemCodeError, match="001"):
        validate_product_master(frame)


def test_itemcode_normalization_preserves_leading_zeros() -> None:
    assert normalize_itemcode(" 001 ") == "001"
    assert normalize_itemcode("001.0") == "001"
    assert normalize_itemcode(123.0) == "123"
    assert pd.isna(normalize_itemcode("  "))


def test_gold_optional_columns_are_optional_and_flags_are_coerced() -> None:
    frame = pd.DataFrame(
        [{
            "offer_group_id": "offer-1",
            "offer_text": "Chicken nuggets",
            "master_itemcode": " 001 ",
            "pair_label": 1,
            "use_for_binary_pair_training": "true",
        }]
    )
    validated = validate_gold_pairs(frame)
    assert validated.loc[0, "master_itemcode"] == "001"
    assert validated.loc[0, "use_for_binary_pair_training"] == 1
    assert "source_dataset" not in validated.columns


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (1, 1), (1.0, 1), ("1", 1), ("1.0", 1), (True, 1),
        (0, 0), (0.0, 0), ("0", 0), ("0.0", 0), (False, 0),
    ],
)
def test_binary_training_flag_accepts_equivalent_boolean_representations(
    flag: object,
    expected: int,
) -> None:
    frame = pd.DataFrame(
        [{
            "offer_group_id": "offer-1",
            "offer_text": "Chicken nuggets",
            "master_itemcode": "001",
            "pair_label": 1,
            "use_for_binary_pair_training": flag,
        }]
    )
    validated = validate_gold_pairs(frame)
    assert validated.loc[0, "use_for_binary_pair_training"] == expected


def test_gold_validator_returns_a_copy_without_mutating_input() -> None:
    frame = pd.DataFrame(
        [{
            "offer_group_id": "offer-1",
            "offer_text": "Chicken nuggets",
            "master_itemcode": " 001 ",
            "pair_label": 1,
            "use_for_binary_pair_training": "true",
        }]
    )
    validate_gold_pairs(frame)
    assert frame.loc[0, "master_itemcode"] == " 001 "
    assert frame.loc[0, "use_for_binary_pair_training"] == "true"


@pytest.mark.parametrize("label", [2, "positive", 0.5])
def test_invalid_gold_labels_are_rejected(label: object) -> None:
    frame = pd.DataFrame(
        [{
            "offer_group_id": "offer-1",
            "offer_text": "Chicken nuggets",
            "master_itemcode": "001",
            "pair_label": label,
            "use_for_binary_pair_training": 1,
        }]
    )
    with pytest.raises(InputValidationError, match="pair_label"):
        validate_gold_pairs(frame)


def test_invalid_binary_training_flag_is_rejected() -> None:
    frame = pd.DataFrame(
        [{
            "offer_group_id": "offer-1",
            "offer_text": "Chicken nuggets",
            "master_itemcode": "001",
            "pair_label": 1,
            "use_for_binary_pair_training": "maybe",
        }]
    )
    with pytest.raises(InputValidationError, match="use_for_binary_pair_training"):
        validate_gold_pairs(frame)
