"""The competitor ground-truth table: staging, verdicts, and preservation.

Nothing in the pipeline reads this table to decide anything. It exists so the
labels that neither the rules nor the borrowed model have ever been measured
against can start accumulating.
"""

from __future__ import annotations

import pytest

from sku_mapping.learning.store import LearningStore, LearningStoreError


@pytest.fixture()
def store(tmp_path) -> LearningStore:
    return LearningStore(tmp_path / "learning.db")


def _proposal(**overrides):
    base = {
        "run_id": "run-1",
        "master_sku": "SKU-A",
        "competitor_offer_id": "offer-1",
        "competitor_offer_name": "Rival Chicken Samosa 240g",
        "competitor_brand": "Rival",
        "master_name": "Chicken Samosa",
        "proposed_status": "MATCHED",
        "proposed_reason": "SUPPORTED_COMPATIBLE_COMPETITOR",
        "rule_score": 72.5,
        "rule_adjusted_score": 75.5,
        "lightgbm_score": -1.25,
        "lightgbm_rank": 2,
        "ranking_source": "lightgbm",
        "model_id": "ranked-v5-cal",
    }
    base.update(overrides)
    return base


def test_staging_records_the_proposal_as_pending(store: LearningStore) -> None:
    assert store.stage_competitor_proposals([_proposal()]) == 1
    (row,) = store.competitor_decisions()
    assert row["decision"] == "PENDING"
    assert row["lightgbm_score"] == pytest.approx(-1.25)
    assert row["ranking_source"] == "lightgbm"
    assert row["decided_at"] is None


def test_restaging_a_run_never_discards_a_human_verdict(
    store: LearningStore,
) -> None:
    """A rerun refreshes evidence; it must not erase what a reviewer said."""
    store.stage_competitor_proposals([_proposal()])
    store.record_competitor_decision(
        run_id="run-1",
        master_sku="SKU-A",
        competitor_offer_id="offer-1",
        decision="REJECTED",
        reviewer="reviewer@example.com",
        notes="different pack format",
    )
    store.stage_competitor_proposals(
        [_proposal(lightgbm_score=4.0, lightgbm_rank=1)]
    )

    (row,) = store.competitor_decisions()
    assert row["decision"] == "REJECTED"
    assert row["reviewer"] == "reviewer@example.com"
    assert row["notes"] == "different pack format"
    assert row["lightgbm_score"] == pytest.approx(4.0)


def test_unknown_decision_is_refused(store: LearningStore) -> None:
    store.stage_competitor_proposals([_proposal()])
    with pytest.raises(LearningStoreError):
        store.record_competitor_decision(
            run_id="run-1",
            master_sku="SKU-A",
            competitor_offer_id="offer-1",
            decision="MAYBE",
            reviewer="r",
        )


def test_verdict_on_unstaged_relationship_is_refused(
    store: LearningStore,
) -> None:
    with pytest.raises(LearningStoreError):
        store.record_competitor_decision(
            run_id="nope",
            master_sku="SKU-A",
            competitor_offer_id="offer-1",
            decision="CONFIRMED",
            reviewer="r",
        )


def test_missing_identity_is_refused(store: LearningStore) -> None:
    with pytest.raises(LearningStoreError):
        store.stage_competitor_proposals([_proposal(master_sku="")])


def test_pending_queue_can_be_filtered(store: LearningStore) -> None:
    store.stage_competitor_proposals(
        [_proposal(), _proposal(competitor_offer_id="offer-2")]
    )
    store.record_competitor_decision(
        run_id="run-1",
        master_sku="SKU-A",
        competitor_offer_id="offer-2",
        decision="CONFIRMED",
        reviewer="r",
    )
    assert len(store.competitor_decisions(decision="PENDING")) == 1
    assert len(store.competitor_decisions(decision="CONFIRMED")) == 1
    assert len(store.competitor_decisions()) == 2


def test_existing_own_brand_tables_survive_the_migration(
    store: LearningStore,
) -> None:
    """Adding the competitor table must not disturb the own-brand store."""
    store.stage_competitor_proposals([_proposal()])
    store.upsert_pipeline_run({"run_id": "run-1", "status": "PROCESSING"})
    assert store.get_pipeline_run("run-1")["run_id"] == "run-1"
