"""Regression tests for durable run and Human Validation visibility."""

from __future__ import annotations

from pathlib import Path

from dashboard.services.review_service import DashboardReviewService
from sku_mapping.learning.store import LearningStore


def _add_manual_review_run(
    store: LearningStore,
    *,
    run_id: str,
    timestamp: str,
) -> None:
    """Persist one completed run containing only unreviewed proposals."""
    store.upsert_pipeline_run(
        {
            "run_id": run_id,
            "started_at": timestamp,
            "completed_at": timestamp,
            "status": "COMPLETED_DASHBOARD_ASSISTED",
            "deployment_mode": "assisted",
            "source_row_count": 40,
            "unique_offer_count": 40,
        }
    )
    decisions = []
    predictions = []
    for number in (1, 2):
        offer_id = f"{run_id}-offer-{number}"
        candidate_id = f"{run_id}-sku-{number}"
        decisions.append(
            {
                "offer_id": offer_id,
                "offer_description": f"Al Kabeer offer {number}",
                "source_row_count": 1,
                "is_own_brand": True,
                "proposed_master_sku": candidate_id,
                "proposed_master_description": f"Master SKU {number}",
                "proposed_candidate_rank": 1,
                "matched_master_sku": None,
                "final_decision": "MANUAL_REVIEW",
                "decision_source": "AGREEMENT_POLICY",
                "final_decision_reason": "human confirmation required",
                "final_eligible_mapping": False,
                "lightgbm_probability": 0.74 + number / 100,
            }
        )
        predictions.append(
            {
                "offer_id": offer_id,
                "offer_description": f"Al Kabeer offer {number}",
                "candidate_id": candidate_id,
                "candidate_description": f"Master SKU {number}",
                "candidate_rank": 1,
                "lightgbm_probability": 0.74 + number / 100,
                "embedding_similarity": None,
                "agreement_status": "UNAVAILABLE",
                "final_decision": "MANUAL_REVIEW",
                "decision_source": "AGREEMENT_POLICY",
            }
        )
    store.add_offer_decisions(run_id, decisions)
    store.add_predictions(run_id, predictions)


def test_runs_and_manual_proposals_survive_reopen_and_service_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "learning.db"
    first_store = LearningStore(database_path)
    _add_manual_review_run(
        first_store,
        run_id="run-a",
        timestamp="2026-07-30T00:00:00+00:00",
    )

    reopened_store = LearningStore(database_path)
    assert reopened_store.get_pipeline_run("run-a") is not None
    _add_manual_review_run(
        reopened_store,
        run_id="run-b",
        timestamp="2026-07-30T01:00:00+00:00",
    )

    restarted_store = LearningStore(database_path)
    assert {
        run["run_id"] for run in restarted_store.list_pipeline_runs()
    } == {"run-a", "run-b"}

    first_service = DashboardReviewService(restarted_store)
    visible_runs = first_service.runs_with_review_sessions()
    assert {run["run_id"] for run in visible_runs} == {"run-a", "run-b"}
    for run in visible_runs:
        session_id = str(run["review_session"]["session_id"])
        questions = restarted_store.review_questions(session_id)
        assert len(questions) == 2
        assert all(question["answered_at"] is None for question in questions)
        assert all(
            str(question["suggested_candidate_id"]).startswith(
                f"{run['run_id']}-sku-"
            )
            for question in questions
        )

    # Rebuilding dashboard services must reuse, not recreate, durable state.
    second_service = DashboardReviewService(LearningStore(database_path))
    second_visible = second_service.runs_with_review_sessions()
    assert {run["run_id"] for run in second_visible} == {"run-a", "run-b"}

    diagnostics = LearningStore(database_path).summary()
    assert diagnostics["database_path"] == str(database_path.resolve())
    assert diagnostics["stored_run_count"] == 2
    assert diagnostics["stored_decision_count"] == 4
    assert diagnostics["counts"]["review_sessions"] == 2
    assert diagnostics["counts"]["human_reviews"] == 4
    assert diagnostics["latest_run_id"] == "run-b"
    assert (
        diagnostics["latest_run_timestamp"]
        == "2026-07-30T01:00:00+00:00"
    )


def test_learning_store_path_does_not_follow_working_directory_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_working_directory = tmp_path / "original"
    later_working_directory = tmp_path / "later"
    original_working_directory.mkdir()
    later_working_directory.mkdir()

    monkeypatch.chdir(original_working_directory)
    store = LearningStore(Path("state") / "learning.db")
    expected_path = (
        original_working_directory / "state" / "learning.db"
    ).resolve()

    monkeypatch.chdir(later_working_directory)
    store.upsert_pipeline_run(
        {
            "run_id": "stable-path-run",
            "status": "COMPLETED_ASSISTED",
            "deployment_mode": "assisted",
        }
    )

    assert store.path == expected_path
    assert expected_path.is_file()
    assert not (later_working_directory / "state" / "learning.db").exists()
    assert (
        LearningStore(expected_path).get_pipeline_run("stable-path-run")
        is not None
    )
