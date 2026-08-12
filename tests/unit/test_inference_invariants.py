"""Behavioral tests for terminal lifecycle invariants."""

from __future__ import annotations

import pandas as pd
import pytest

from sku_mapping.inference.invariants import (
    PipelineInvariantError,
    validate_terminal_decisions,
)


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "offer_id": ["own-1", "competitor-1"],
            "final_decision": ["MANUAL_REVIEW", "COMPETITOR_OFFER"],
            "final_eligible_mapping": [False, False],
            "proposed_master_sku": ["SKU-1", ""],
            "matched_master_sku": ["", ""],
        }
    )


def test_complete_lifecycle_accepts_proposal_without_accepted_mapping() -> None:
    offers = pd.DataFrame(
        {
            "offer_group_id": ["own-1", "competitor-1"],
            "is_own": [True, False],
        }
    )
    candidates = pd.DataFrame(
        {
            "offer_group_id": ["own-1"],
            "master_itemcode": ["SKU-1"],
        }
    )

    validate_terminal_decisions(offers, candidates, _decisions())


@pytest.mark.parametrize(
    "mutation",
    ["missing_offer", "duplicate_offer", "erased_proposal", "false_accept"],
)
def test_lifecycle_invariant_rejects_silent_or_inconsistent_state(
    mutation: str,
) -> None:
    offers = pd.DataFrame(
        {
            "offer_group_id": ["own-1", "competitor-1"],
            "is_own": [True, False],
        }
    )
    candidates = pd.DataFrame(
        {
            "offer_group_id": ["own-1"],
            "master_itemcode": ["SKU-1"],
        }
    )
    decisions = _decisions()
    if mutation == "missing_offer":
        decisions = decisions.iloc[:1].copy()
    elif mutation == "duplicate_offer":
        decisions.loc[1, "offer_id"] = "own-1"
    elif mutation == "erased_proposal":
        decisions.loc[0, "proposed_master_sku"] = ""
    else:
        decisions.loc[1, "final_eligible_mapping"] = True
        decisions.loc[1, "matched_master_sku"] = "SKU-X"

    with pytest.raises(PipelineInvariantError):
        validate_terminal_decisions(offers, candidates, decisions)
