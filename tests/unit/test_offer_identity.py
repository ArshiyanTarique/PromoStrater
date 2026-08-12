"""Regression tests for ClickFlyer canonical offer identity."""

from __future__ import annotations

import pandas as pd
import pytest

from sku_mapping.data.offer_identity import (
    FALLBACK_IDENTITY_VERSION,
    assign_offer_identities,
    normalize_offer_id,
)
from sku_mapping.data.validators import validate_clickflyer


def test_40_distinct_offerids_count_as_40_despite_shared_fields() -> None:
    frame = pd.read_csv(
        "tests/fixtures/clickflyer_40_distinct_offerids.csv"
    )
    assignment = assign_offer_identities(frame)

    assert len(frame) == 40
    assert assignment.unique_offer_count == 40
    assert assignment.identities.nunique() == 40
    assert assignment.source == "offerid"
    assert frame["Flyer Name"].nunique() == 1
    assert frame["Offer Name"].nunique() == 1


def test_repeated_rows_with_same_offerid_count_once() -> None:
    frame = pd.DataFrame(
        {
            "offerid": ["offer-1", " offer-1 ", "offer-2"],
            "Offer Name": ["Shared"] * 3,
        }
    )
    assignment = assign_offer_identities(frame)

    assert assignment.identities.tolist() == [
        "offer-1",
        "offer-1",
        "offer-2",
    ]
    assert assignment.unique_offer_count == 2


def test_numeric_and_string_integer_ids_normalize_consistently() -> None:
    values = [31862126, 31862126.0, "31862126", " 31862126.0 "]

    assert {normalize_offer_id(value) for value in values} == {"31862126"}


def test_missing_offerid_column_uses_documented_stable_fallback() -> None:
    frame = pd.DataFrame(
        {
            "Flyer Name": ["Shared", "Shared"],
            "Offer Name": ["Same", "Same"],
            "Retailer Name": ["R", "R"],
        }
    )
    first = assign_offer_identities(frame)
    second = assign_offer_identities(frame)

    assert first.source == FALLBACK_IDENTITY_VERSION
    assert first.unique_offer_count == 2
    assert first.identities.tolist() == second.identities.tolist()
    assert all(
        value.startswith("shadow_offer_") for value in first.identities
    )
    required = pd.DataFrame(
        {
            "Offer Name": ["Same"],
            "Product": ["Nuggets"],
            "Brand Name": ["Al-Kabeer"],
            "Variant": [""],
            "Base Packsize": ["400g"],
            "Country": ["Saudi Arabia"],
            "Retailer Name": ["R"],
            "Flyer Name": ["F"],
            "Offer Price": [10.0],
            "Regular Price": [12.0],
        }
    )
    assert validate_clickflyer(required).equals(required)


def test_genuinely_missing_ids_receive_distinct_fallback_identities() -> None:
    frame = pd.DataFrame(
        {
            "offerid": ["1", None, " ", "2"],
            "Offer Name": ["A", "B", "C", "D"],
        }
    )
    assignment = assign_offer_identities(frame)

    assert assignment.unique_offer_count == 4
    assert assignment.missing_offer_id_count == 2
    assert assignment.source == "offerid_with_stable_fallback"


@pytest.mark.parametrize(
    ("offer_ids", "offer_names", "retailers", "flyers", "expected"),
    [
        (
            ["1", "2", "3"],
            ["Shared"] * 3,
            ["Same"] * 3,
            ["Same"] * 3,
            3,
        ),
        (
            ["1", "1", "2"],
            ["Different A", "Different B", "Different C"],
            ["R1", "R2", "R3"],
            ["F1", "F2", "F3"],
            2,
        ),
        (
            [1, "1.0", 2.0, "2"],
            ["One", "Two", "Three", "Four"],
            ["R"] * 4,
            ["F"] * 4,
            2,
        ),
    ],
)
def test_identity_depends_on_offerid_not_shared_descriptive_fields(
    offer_ids: list[object],
    offer_names: list[str],
    retailers: list[str],
    flyers: list[str],
    expected: int,
) -> None:
    assignment = assign_offer_identities(
        pd.DataFrame(
            {
                "offerid": offer_ids,
                "Offer Name": offer_names,
                "Retailer Name": retailers,
                "Flyer Name": flyers,
            }
        )
    )

    assert assignment.unique_offer_count == expected


def test_large_identity_assignment_does_not_collapse_shared_templates() -> None:
    size = 20_000
    assignment = assign_offer_identities(
        pd.DataFrame(
            {
                "offerid": range(size),
                "Offer Name": ["Shared synthetic template"] * size,
                "Retailer Name": ["Retailer"] * size,
                "Flyer Name": ["Flyer"] * size,
            }
        )
    )

    assert assignment.unique_offer_count == size
    assert assignment.identities.nunique() == size
