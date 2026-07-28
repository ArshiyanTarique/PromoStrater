"""End-to-end fixture test for Phase 4 training artifacts."""

from __future__ import annotations

import json

import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.training.feature_builder import build_training_features_from_paths


def test_build_training_features_writes_manifest_and_rejected_rows(tmp_path) -> None:
    gold_path = tmp_path / "gold.csv"
    master_path = tmp_path / "master.xlsx"
    flyer_path = tmp_path / "flyer.csv"
    output_dir = tmp_path / "processed"
    pd.DataFrame(
        [
            {
                "record_id": "accepted",
                "source_dataset": "REAL",
                "offer_group_id": "g1",
                "offer_text": "Al Kabeer Chicken Nuggets 400g",
                "master_itemcode": "001",
                "pair_label": 1,
                "use_for_binary_pair_training": 1,
                "label_provenance": "human",
            },
            {
                "record_id": "rejected",
                "source_dataset": "SYNTHETIC",
                "offer_group_id": "g2",
                "offer_text": "Unknown product",
                "master_itemcode": "UNKNOWN",
                "pair_label": 0,
                "use_for_binary_pair_training": 1,
                "label_provenance": "synthetic_rule",
            },
        ]
    ).to_csv(gold_path, index=False)
    pd.DataFrame(
        [
            {
                "Itemcode": "001",
                "Itemname": "CHICKEN NUGGETS",
                "Item-Cat-2": "Chicken",
                "Item-Cat-4": "NUGGETS",
                "Item Description": "CHICKEN NUGGETS",
                "Item-Spec": "400g x 20 Pkts",
            }
        ]
    ).to_excel(master_path, index=False)
    pd.DataFrame(
        [
            {
                "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                "Product": "Chicken Nuggets-Frozen",
                "Brand Name": "Al Kabeer",
                "Variant": "No Variant",
                "Base Packsize": "400g",
                "Country": "Saudi Arabia",
                "Retailer Name": "Retailer",
                "Flyer Name": "Flyer",
                "offerid": "0001",
                "Offer Price": 10,
                "Regular Price": 12,
            }
        ]
    ).to_csv(flyer_path, index=False)

    result = build_training_features_from_paths(
        gold_path,
        master_path,
        clickflyer_path=flyer_path,
        output_dir=output_dir,
    )

    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert result.output_paths["training_features_parquet"].is_file()
    assert result.output_paths["training_features_csv"].is_file()
    assert result.output_paths["rejected_training_rows_csv"].is_file()
    assert result.output_paths["training_feature_manifest_json"].is_file()
    assert result.output_paths["training_data_audit_json"].is_file()

    accepted = pd.read_parquet(result.output_paths["training_features_parquet"])
    rejected = pd.read_csv(result.output_paths["rejected_training_rows_csv"])
    manifest = json.loads(
        result.output_paths["training_feature_manifest_json"].read_text(
            encoding="utf-8"
        )
    )
    assert accepted.columns[-19:].tolist() == list(MODEL_FEATURE_COLUMNS)
    assert rejected.loc[0, "rejection_reason"] == "unknown_master_itemcode"
    assert manifest["accepted_rows"] == 1
    assert manifest["rejected_rows"] == 1
    assert manifest["input_filenames"]["gold_pairs"] == "gold.csv"
    assert set(manifest["output_hashes"]) == {
        "training_features_parquet",
        "training_features_csv",
        "rejected_training_rows_csv",
        "training_data_audit_json",
    }
