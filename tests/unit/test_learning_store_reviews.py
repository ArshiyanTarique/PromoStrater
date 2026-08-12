"""Review-session, answer-governance, and challenge-exclusion tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sku_mapping.learning.models import HumanReviewAnswer, LabelQuality
from sku_mapping.learning.store import (
    DuplicateHumanReviewError,
    LearningStore,
    LearningStoreError,
)


def _populated_store(tmp_path: Path) -> tuple[LearningStore, str]:
    store = LearningStore(tmp_path / "learning.db")
    run_id = "run-1"
    store.upsert_pipeline_run(
        {
            "run_id": run_id,
            "status": "COMPLETED_ASSISTED",
            "deployment_mode": "assisted",
            "unique_offer_count": 6,
            "source_row_count": 6,
        }
    )
    records = []
    for number in range(6):
        offer = f"offer-{number}"
        for rank in (1, 2):
            records.append(
                {
                    "offer_id": offer,
                    "offer_description": f"Offer {number}",
                    "candidate_id": f"sku-{number}-{rank}",
                    "candidate_description": f"SKU {number}-{rank}",
                    "candidate_rank": rank,
                    "lightgbm_probability": 0.95 - number * 0.05
                    if rank == 1
                    else 0.1,
                    "embedding_similarity": 0.8 - number * 0.05,
                    "agreement_status": (
                        "DISAGREEMENT" if number == 2 else "WEAK_AGREEMENT"
                    ),
                    "llm_decision": (
                        "UNCERTAIN" if number == 3 else ""
                    ),
                    "final_decision": (
                        "AUTO_ACCEPT"
                        if number == 0 and rank == 1
                        else (
                            "MANUAL_REVIEW"
                            if rank == 1
                            else "CANDIDATE_NOT_SELECTED"
                        )
                    ),
                    "decision_source": (
                        "STRUCTURED_LLM_REVIEW"
                        if number == 3 and rank == 1
                        else "AGREEMENT_POLICY"
                    ),
                    "conflict_flags": (
                        ["pack_conflict"] if number == 4 else []
                    ),
                    "feature_snapshot": {"protein_match": 1},
                }
            )
    store.add_predictions(run_id, records)
    session_id = store.create_review_session(run_id)
    assert session_id is not None
    return store, session_id


def test_session_has_exactly_five_unique_questions(tmp_path: Path) -> None:
    store, session_id = _populated_store(tmp_path)
    connection = sqlite3.connect(store.path)
    try:
        rows = connection.execute(
            """
            SELECT offer_id, question_text, selection_reason
            FROM human_reviews WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 5
    assert len({row[0] for row in rows}) == 5
    assert {row[1] for row in rows} == {
        "Is this suggested SKU match correct?"
    }
    assert all(row[2] for row in rows)


def test_false_answer_requires_explicit_resolution(tmp_path: Path) -> None:
    store, session_id = _populated_store(tmp_path)
    question = store.next_unanswered_question(session_id)
    assert question is not None
    with pytest.raises(LearningStoreError, match="exactly one"):
        store.save_answer(
            question.review_id,
            HumanReviewAnswer(is_correct=False),
        )
    store.save_answer(
        question.review_id,
        HumanReviewAnswer(
            is_correct=False,
            corrected_candidate_id="valid-master-outside-top-k",
            decomposition_action="CONFIRM",
        ),
    )
    saved = store.review_questions(session_id)[0]
    assert saved["suggested_candidate_id"] != (
        "valid-master-outside-top-k"
    )
    assert saved["corrected_candidate_id"] == (
        "valid-master-outside-top-k"
    )


def test_gold_answer_and_duplicate_prevention(tmp_path: Path) -> None:
    store, session_id = _populated_store(tmp_path)
    question = store.next_unanswered_question(session_id)
    assert question is not None
    store.save_answer(
        question.review_id,
        HumanReviewAnswer(is_correct=True, reviewer_id="reviewer-1"),
    )
    with pytest.raises(DuplicateHumanReviewError):
        store.save_answer(
            question.review_id,
            HumanReviewAnswer(is_correct=True, reviewer_id="reviewer-1"),
        )
    exported = store.reviewed_labels()
    assert exported[0]["label_quality"] == LabelQuality.GOLD.value


def test_all_five_answers_complete_session_and_count_gold(
    tmp_path: Path,
) -> None:
    store, session_id = _populated_store(tmp_path)
    for _ in range(5):
        question = store.next_unanswered_question(session_id)
        assert question is not None
        assert len(question.supplied_candidates) == 2
        store.save_answer(
            question.review_id,
            HumanReviewAnswer(is_correct=True),
        )
    assert store.next_unanswered_question(session_id) is None
    assert store.review_session_complete(session_id)
    assert store.count_new_gold_labels_since_last_model() == 5


def test_cannot_determine_is_not_gold(tmp_path: Path) -> None:
    store, session_id = _populated_store(tmp_path)
    question = store.next_unanswered_question(session_id)
    assert question is not None
    store.save_answer(
        question.review_id,
        HumanReviewAnswer(is_correct=False, cannot_determine=True),
    )
    assert store.reviewed_labels()[0]["label_quality"] == "REJECTED"


def test_training_dataset_blocks_sealed_challenge_reviews(
    tmp_path: Path,
) -> None:
    store, session_id = _populated_store(tmp_path)
    question = store.next_unanswered_question(session_id)
    assert question is not None
    store.save_answer(
        question.review_id,
        HumanReviewAnswer(is_correct=True),
    )
    manifest = tmp_path / "challenge_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "SEALED_UNOPENED",
                "review_ids": [question.review_id],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LearningStoreError, match="sealed challenge"):
        store.create_training_dataset_record(
            included_review_ids=[question.review_id],
            challenge_manifest_paths=[manifest],
        )


def test_pseudo_label_cannot_be_marked_training_eligible(
    tmp_path: Path,
) -> None:
    store, _ = _populated_store(tmp_path)
    connection = sqlite3.connect(store.path)
    prediction_id = connection.execute(
        "SELECT prediction_id FROM predictions LIMIT 1"
    ).fetchone()[0]
    connection.close()
    with pytest.raises(LearningStoreError, match="PSEUDO"):
        store.add_automated_label(
            prediction_id=prediction_id,
            source="MODEL",
            proposed_label="MATCH",
            selected_candidate_id="sku",
            confidence=0.99,
            label_quality=LabelQuality.PSEUDO,
            eligibility_status="ELIGIBLE",
        )
