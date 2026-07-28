"""Unit tests for Phase 4 training-data governance."""

from __future__ import annotations

import pandas as pd

from sku_mapping.training.data_audit import audit_training_data


def _gold_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "r1",
                "source_dataset": "SYNTHETIC_V1",
                "offer_group_id": "g1",
                "offer_text": "Chicken Nuggets 400g",
                "master_itemcode": "001",
                "pair_label": 1,
                "use_for_binary_pair_training": 1,
                "recommended_split": "train",
                "label_provenance": "human",
            },
            {
                "record_id": "r2",
                "source_dataset": "SYNTHETIC_V1",
                "offer_group_id": "g1",
                "offer_text": "Chicken Nuggets 400g",
                "master_itemcode": "001",
                "pair_label": 0,
                "use_for_binary_pair_training": 1,
                "recommended_split": "test",
                "label_provenance": "rule",
            },
            {
                "record_id": "r3",
                "source_dataset": "REAL",
                "offer_group_id": "g2",
                "offer_text": None,
                "master_itemcode": "UNKNOWN",
                "pair_label": "bad",
                "use_for_binary_pair_training": 1,
                "recommended_split": "train",
                "label_provenance": "human",
            },
        ]
    )


def test_audit_reports_duplicate_conflicting_unknown_and_split_issues() -> None:
    audit = audit_training_data(_gold_rows(), {"001"})

    assert audit["duplicate_offer_master_pairs"]["group_count"] == 1
    assert audit["duplicate_offer_master_pairs"]["row_count"] == 2
    assert audit["conflicting_labels"]["pair_count"] == 1
    assert audit["conflicting_labels"]["row_count"] == 2
    assert audit["offer_groups_across_recommended_splits"]["group_count"] == 1
    assert audit["invalid_labels"]["row_count"] == 1
    assert audit["unknown_master_skus"]["unique_codes"] == ["UNKNOWN"]
    assert audit["null_offer_text"]["row_count"] == 1
    assert audit["repeated_synthetic_templates"]["template_count"] == 1


def test_audit_reports_exact_duplicates_without_deleting_them() -> None:
    row = _gold_rows().iloc[0].to_dict()
    frame = pd.DataFrame([row, row])

    audit = audit_training_data(frame, {"001"})

    assert audit["exact_duplicate_rows"]["row_count"] == 2
    assert audit["policy"]["exact_duplicates"] == (
        "report_and_retain_when_otherwise_eligible"
    )
    assert len(frame) == 2
