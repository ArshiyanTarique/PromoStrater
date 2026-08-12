"""Safe parity tests for the extracted Stage 2 candidate logic."""

from __future__ import annotations

import ast
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rapidfuzz import fuzz, process

from sku_mapping.data.preprocessing import preprocess_clickflyer, preprocess_product_master
from sku_mapping.matching.matcher import match_preprocessed_offers


def _legacy_match_batch() -> tuple[dict[str, Any], Any]:
    """Compile only Stage 2 helpers from the authoritative script."""
    source_path = Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names = {"pack_is_compatible", "pack_structure_agrees", "match_batch"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "fuzz": fuzz,
        "process": process,
        # match_batch reads the module-level shortlist size; the ranked
        # package's default is 20 and parity only compares Stage 2 fields.
        "RETRIEVAL_K": 20,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace, namespace["match_batch"]


def _master(rows: list[dict[str, str]]) -> pd.DataFrame:
    return preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": row["Itemcode"],
                    "Itemname": row["Itemname"],
                    "Item-Cat-2": row["Item-Cat-2"],
                    "Item-Cat-4": row.get("Item-Cat-4", ""),
                    "Item Description": row.get("Item Description", ""),
                    "Item-Spec": row.get("Item-Spec", ""),
                }
                for row in rows
            ]
        )
    )


def _offers(rows: list[dict[str, str]]) -> pd.DataFrame:
    return preprocess_clickflyer(pd.DataFrame(rows))


@pytest.mark.parametrize(
    "rows,offer_rows",
    [
        (
            [
                {"Itemcode": "NUG-400", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
                {"Itemcode": "STR-750", "Itemname": "CHICKEN STRIPS", "Item-Cat-2": "Chicken", "Item-Spec": "750g"},
            ],
            [{"Offer Name": "Al Kabeer Chicken Nuggets 400g", "Product": "Chicken Nuggets-Frozen", "Brand Name": "Al Kabeer", "Variant": "No Variant", "Base Packsize": "400g"}],
        ),
        (
            [
                {"Itemcode": "CH-FILLET", "Itemname": "CHICKEN FILLET", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
                {"Itemcode": "FS-FILLET", "Itemname": "FISH FILLET", "Item-Cat-2": "Seafood", "Item-Spec": "400g"},
            ],
            [{"Offer Name": "Chicken Fillet 400g", "Product": "Chicken Fillet-Frozen", "Brand Name": "Al Kabeer", "Variant": "No Variant", "Base Packsize": "400g"}],
        ),
        (
            [{"Itemcode": "NUG-1K", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "1kg"}],
            [{"Offer Name": "Chicken Nuggets 2 x 500g", "Product": "Chicken Nuggets-Frozen", "Brand Name": "Al Kabeer", "Variant": "No Variant", "Base Packsize": "2 x 500g"}],
        ),
        (
            [{"Itemcode": "NUG-400", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"}],
            [{"Offer Name": "Chicken Nuggets 1kg", "Product": "Chicken Nuggets-Frozen", "Brand Name": "Al Kabeer", "Variant": "No Variant", "Base Packsize": "1kg"}],
        ),
        (
            [{"Itemcode": "MYST", "Itemname": "MYSTERY SNACK", "Item-Cat-2": "Chicken", "Item-Spec": "400g"}],
            [{"Offer Name": "Mystery Snack", "Product": "Unknown Product", "Brand Name": "Al Kabeer", "Variant": "No Variant", "Base Packsize": "400g"}],
        ),
    ],
)
def test_fixture_results_match_authoritative_stage_two(
    rows: list[dict[str, str]],
    offer_rows: list[dict[str, str]],
) -> None:
    master = _master(rows)
    offers = _offers(offer_rows)
    namespace, legacy_match = _legacy_match_batch()
    namespace["master"] = master
    namespace["master_by_cat"] = {
        category: group.reset_index(drop=True) for category, group in master.groupby("category")
    }

    legacy = legacy_match(offers.reset_index(drop=True), str(offers.loc[0, "category"]))[0]
    actual = match_preprocessed_offers(offers, master)[0]
    assert actual.itemcode == legacy["matched_itemcode"]
    assert actual.itemname == legacy["matched_itemname"]
    assert actual.text_score == legacy["match_score"]
    assert actual.margin == legacy["margin"]
    assert actual.raw_margin == legacy["raw_margin"]
    assert actual.pack_status == legacy["pack_status"]
    assert actual.confidence_tier == legacy["confidence_tier"]


def test_reproducible_limited_real_data_parity_sample() -> None:
    """Compare 24 fixed-seed own-brand rows without scanning the full flyer dump."""
    root = Path(__file__).parents[2]
    flyer_path = root / "Alkabeer_Export_Data_Clickflyer.csv"
    master_path = root / "Product_Master.xlsx"
    if not flyer_path.exists() or not master_path.exists():
        pytest.skip("Repository data fixtures are unavailable")

    raw_flyer = pd.read_csv(flyer_path, nrows=8_000, dtype={"offerid": "string"}, low_memory=False)
    offers = preprocess_clickflyer(raw_flyer)
    offers = offers.loc[offers["is_own"]].sample(
        n=min(24, int(offers["is_own"].sum())),
        random_state=42,
    ).reset_index(drop=True)
    if offers.empty:
        pytest.skip("No own-brand rows in the reproducible limited sample")
    master = preprocess_product_master(pd.read_excel(master_path, dtype={"Itemcode": "string"}))

    namespace, legacy_match = _legacy_match_batch()
    namespace["master"] = master
    namespace["master_by_cat"] = {
        category: group.reset_index(drop=True) for category, group in master.groupby("category")
    }
    legacy_start = perf_counter()
    expected: list[dict[str, Any]] = []
    for category, group in offers.groupby("category", sort=True):
        for position, result in zip(group.index, legacy_match(group.reset_index(drop=True), category)):
            expected.append({"position": position, **result})
    expected = sorted(expected, key=lambda row: row["position"])
    legacy_elapsed = perf_counter() - legacy_start

    start = perf_counter()
    actual = match_preprocessed_offers(offers, master)
    elapsed = perf_counter() - start
    assert elapsed < 10.0
    assert legacy_elapsed < 10.0
    print(
        f"candidate parity sample (24 max rows): "
        f"legacy={legacy_elapsed:.4f}s extracted={elapsed:.4f}s"
    )
    for expected_row, actual_match in zip(expected, actual):
        assert actual_match.itemcode == expected_row["matched_itemcode"]
        assert actual_match.text_score == expected_row["match_score"]
        assert actual_match.margin == expected_row["margin"]
        assert actual_match.raw_margin == expected_row["raw_margin"]
        assert actual_match.confidence_tier == expected_row["confidence_tier"]
