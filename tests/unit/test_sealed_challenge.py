"""SEALED_UNOPENED challenge-set governance tests."""

from __future__ import annotations

import pandas as pd
import pytest

from sku_mapping.ml.trainer import run_training_pipeline
from sku_mapping.shadow.challenge import (
    SealedChallengeSetError,
    assert_not_sealed_challenge_input,
    build_sealed_challenge_set,
)
from sku_mapping.shadow.intake import stage_completed_review_file


def _stage_review(tmp_path):
    review_path = tmp_path / "completed.csv"
    pd.DataFrame(
        [
            {
                "shadow_run_id": "shadow-1",
                "offer_group_id": "offer-1",
                "offer_text": "Chicken nuggets 400g",
                "top_candidate_1_itemcode": "001",
                "top_candidate_2_itemcode": "002",
                "human_label": "CORRECT_TOP_CANDIDATE",
                "selected_master_itemcode": "001",
                "reviewer_code": "R1",
                "reviewer_notes": "",
                "review_timestamp": "2026-07-29T12:00:00Z",
            }
        ]
    ).to_csv(review_path, index=False)
    return stage_completed_review_file(
        review_path,
        product_master=pd.DataFrame(
            {"Itemcode": ["001", "002"], "Itemname": ["Nuggets", "Strips"]}
        ),
        staging_directory=tmp_path / "staging",
    )


def test_sealed_challenge_cannot_enter_ordinary_training(tmp_path) -> None:
    intake = _stage_review(tmp_path)
    predictions_path = tmp_path / "shadow_predictions.parquet"
    pd.DataFrame(
        [
            {
                "shadow_run_id": "shadow-1",
                "offer_group_id": "offer-1",
                "candidate_rank": 1,
                "master_itemcode": "001",
                "product_family": "nuggets",
            },
            {
                "shadow_run_id": "shadow-1",
                "offer_group_id": "offer-1",
                "candidate_rank": 2,
                "master_itemcode": "002",
                "product_family": "nuggets",
            },
        ]
    ).to_parquet(predictions_path, index=False)
    result = build_sealed_challenge_set(
        normalized_review_paths=intake.normalized_record_paths,
        shadow_predictions_path=predictions_path,
        challenge_root=tmp_path / "challenge_sets",
    )
    sealed_path = result.directory / "sealed_challenge_records.parquet"

    assert result.manifest["status"] == "SEALED_UNOPENED"
    assert result.manifest["evaluation_approved"] is False
    assert result.manifest["opened_at"] is None
    assert result.manifest["evaluated_at"] is None
    assert result.manifest["labels_exposed_to_training"] is False
    with pytest.raises(SealedChallengeSetError, match="cannot be loaded"):
        assert_not_sealed_challenge_input(sealed_path)
    with pytest.raises(SealedChallengeSetError, match="cannot be loaded"):
        run_training_pipeline(sealed_path)
