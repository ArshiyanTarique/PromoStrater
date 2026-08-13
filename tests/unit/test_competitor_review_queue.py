"""Building a human-sized review queue from a finished competitor run."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sku_mapping.competitors.review import (
    competitor_review_proposals,
    review_queue_summary,
    stage_competitor_review_queue,
)
from sku_mapping.learning.store import LearningStore


def _export(offer_count: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "master_sku": "SKU-A",
                "master_name": "Chicken Samosa",
                "competitor_offer_ids": json.dumps(
                    [f"offer-{i}" for i in range(offer_count)]
                ),
                "competitor_offer_names": json.dumps(
                    [f"Rival Samosa {i}" for i in range(offer_count)]
                ),
                "competitor_brand_names": json.dumps(
                    [f"Brand{i}" for i in range(offer_count)]
                ),
                "competitor_status": "COMPETITORS_FOUND",
                "competitor_reason": "SUPPORTED_COMPATIBLE_COMPETITOR",
            },
            {
                "master_sku": "SKU-B",
                "master_name": "Chicken Fries",
                "competitor_offer_ids": json.dumps([]),
                "competitor_offer_names": json.dumps([]),
                "competitor_brand_names": json.dumps([]),
                "competitor_status": "NO_COMPETITOR_FOUND",
                "competitor_reason": "no compatible competitor offer in upload",
            },
        ]
    )


def test_queue_is_bounded_per_master_sku() -> None:
    """A queue nobody can work is not ground truth."""
    proposals = competitor_review_proposals(
        _export(50), run_id="r1", per_target=3
    )
    assert len(proposals) == 3
    assert [p["lightgbm_rank"] for p in proposals] == [1, 2, 3]
    assert {p["master_sku"] for p in proposals} == {"SKU-A"}


def test_disabled_by_default_stages_nothing() -> None:
    assert competitor_review_proposals(_export(), run_id="r1", per_target=0) == []


def test_targets_without_competitors_are_skipped() -> None:
    proposals = competitor_review_proposals(
        _export(2), run_id="r1", per_target=10
    )
    assert {p["master_sku"] for p in proposals} == {"SKU-A"}


def test_queue_records_which_ordering_produced_it() -> None:
    proposals = competitor_review_proposals(
        _export(2),
        run_id="r1",
        per_target=2,
        model_id="ranked-v5-cal",
        ranking_source="lightgbm",
    )
    assert all(p["ranking_source"] == "lightgbm" for p in proposals)
    assert all(p["model_id"] == "ranked-v5-cal" for p in proposals)


def test_staging_round_trips_into_the_store(tmp_path) -> None:
    store = LearningStore(tmp_path / "learning.db")
    staged = stage_competitor_review_queue(
        store, _export(4), run_id="r1", per_target=2
    )
    assert staged == 2
    rows = store.competitor_decisions()
    assert review_queue_summary(rows) == {"total": 2, "PENDING": 2}
    assert rows[0]["master_name"] == "Chicken Samosa"


def test_restaging_the_same_run_is_idempotent(tmp_path) -> None:
    store = LearningStore(tmp_path / "learning.db")
    stage_competitor_review_queue(store, _export(4), run_id="r1", per_target=2)
    stage_competitor_review_queue(store, _export(4), run_id="r1", per_target=2)
    assert len(store.competitor_decisions()) == 2


def test_malformed_json_does_not_raise() -> None:
    broken = _export(1)
    broken.loc[0, "competitor_offer_ids"] = "not json"
    assert competitor_review_proposals(broken, run_id="r1", per_target=5) == []
