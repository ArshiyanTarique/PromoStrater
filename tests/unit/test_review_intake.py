"""Immutable human-review validation and staging tests."""

from __future__ import annotations

import pandas as pd
import pytest

from sku_mapping.shadow.intake import (
    DuplicateReviewError,
    ReviewIntakeError,
    stage_completed_review_file,
)


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Itemcode": ["001", "002"],
            "Itemname": ["Chicken Nuggets", "Chicken Strips"],
        }
    )


def _review(**overrides: str) -> pd.DataFrame:
    row = {
        "shadow_run_id": "shadow-run-1",
        "offer_group_id": "offer-group-1",
        "offer_text": "Al Kabeer Chicken Nuggets 400g",
        "top_candidate_1_itemcode": "001",
        "top_candidate_2_itemcode": "002",
        "human_label": "CORRECT_TOP_CANDIDATE",
        "selected_master_itemcode": "001",
        "reviewer_code": "R-17",
        "reviewer_notes": "Clear match",
        "review_timestamp": "2026-07-29T12:00:00+00:00",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_completed_review_is_preserved_and_staged_immutably(tmp_path) -> None:
    review_path = tmp_path / "completed.csv"
    _review().to_csv(review_path, index=False, encoding="utf-8-sig")
    result = stage_completed_review_file(
        review_path,
        product_master=_master(),
        staging_directory=tmp_path / "staging",
    )
    assert result.raw_submission_path.is_file()
    assert len(result.normalized_record_paths) == 1
    assert result.audit["training_data_updated"] is False
    assert result.audit["status"] == "ACCEPTED_TO_REVIEW_STAGING"

    with pytest.raises(DuplicateReviewError, match="overwrite"):
        stage_completed_review_file(
            review_path,
            product_master=_master(),
            staging_directory=tmp_path / "staging",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"human_label": "MODEL_LOOKS_GOOD"}, "invalid human_label"),
        (
            {
                "human_label": "NO_VALID_MASTER_SKU",
                "selected_master_itemcode": "001",
            },
            "contradicts",
        ),
        (
            {
                "human_label": "CORRECT_OTHER_CANDIDATE",
                "selected_master_itemcode": "999",
            },
            "unknown Product Master SKU",
        ),
        ({"reviewer_code": ""}, "reviewer_code"),
        ({"review_timestamp": ""}, "review_timestamp"),
    ],
)
def test_invalid_review_rows_are_rejected_but_raw_file_is_preserved(
    tmp_path, overrides: dict[str, str], message: str
) -> None:
    review_path = tmp_path / "invalid.csv"
    _review(**overrides).to_csv(
        review_path, index=False, encoding="utf-8-sig"
    )
    staging = tmp_path / "staging"
    with pytest.raises(ReviewIntakeError, match=message):
        stage_completed_review_file(
            review_path,
            product_master=_master(),
            staging_directory=staging,
        )
    assert len(list((staging / "raw").glob("*.csv"))) == 1
    audit_paths = list((staging / "audits").glob("*.json"))
    assert len(audit_paths) == 1
