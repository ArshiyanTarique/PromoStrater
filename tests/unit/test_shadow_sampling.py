"""Deterministic multi-stratum shadow review sampling tests."""

from __future__ import annotations

import pandas as pd

from sku_mapping.shadow.sampling import (
    SAMPLING_STRATA,
    sample_offers_for_review,
)


def _candidates() -> pd.DataFrame:
    rows = []
    families = ["cube", "frank", "nuggets", "rare family", "fries", "stick"]
    for offer_index in range(12):
        for rank in (1, 2):
            probability = max(
                0.01,
                min(0.99, 0.98 - offer_index * 0.07 - (rank - 1) * 0.04),
            )
            rows.append(
                {
                    "offer_group_id": f"offer-{offer_index:02d}",
                    "candidate_rank": rank,
                    "master_itemcode": f"sku-{offer_index}-{rank}",
                    "calibrated_probability": probability,
                    "shadow_decision_bucket": (
                        "SHADOW_HIGH_SCORE"
                        if probability >= 0.9
                        else "SHADOW_REVIEW"
                    ),
                    "existing_production_decision": (
                        "AUTO_MATCH" if offer_index % 3 == 0 else "NO_MATCH"
                    ),
                    "product_family": families[offer_index % len(families)],
                    "is_mixed_protein_offer": int(offer_index % 5 == 0),
                    "pack_conflict": offer_index % 4 == 0,
                    "feature_missingness_count": offer_index % 3,
                }
            )
    return pd.DataFrame(rows)


def test_sampling_is_reproducible_deduplicated_and_retains_reasons() -> None:
    counts = {stratum: 3 for stratum in SAMPLING_STRATA}
    first = sample_offers_for_review(
        _candidates(),
        counts_by_stratum=counts,
        random_seed=42,
        diagnostic_threshold=0.9,
    )
    second = sample_offers_for_review(
        _candidates(),
        counts_by_stratum=counts,
        random_seed=42,
        diagnostic_threshold=0.9,
    )
    pd.testing.assert_frame_equal(first.offers, second.offers)
    assert first.report == second.report
    assert first.offers["offer_group_id"].is_unique
    assert first.offers["selection_reasons"].str.len().gt(0).all()
    assert set(first.report["strata"]) == set(SAMPLING_STRATA)
