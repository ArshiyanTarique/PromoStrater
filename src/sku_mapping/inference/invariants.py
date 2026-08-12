"""Runtime invariants for canonical offers and terminal inference decisions."""

from __future__ import annotations

import pandas as pd

from sku_mapping.constants import FinalMatchDecision


class PipelineInvariantError(RuntimeError):
    """Raised when an offer silently disappears or a decision is inconsistent."""


def _text_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="string")
    return frame[column].fillna("").astype(str).str.strip()


def validate_terminal_decisions(
    offers: pd.DataFrame,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    """Require one traceable terminal decision for every canonical offer."""
    offer_ids = _text_values(offers, "offer_group_id")
    decision_ids = _text_values(decisions, "offer_id")
    if offer_ids.eq("").any() or offer_ids.duplicated().any():
        raise PipelineInvariantError(
            "Canonical offers must have unique non-empty offer_group_id values"
        )
    if decision_ids.eq("").any() or decision_ids.duplicated().any():
        raise PipelineInvariantError(
            "Terminal decisions must have unique non-empty offer_id values"
        )
    missing = sorted(set(offer_ids) - set(decision_ids))
    unexpected = sorted(set(decision_ids) - set(offer_ids))
    if missing or unexpected or len(decisions) != len(offers):
        raise PipelineInvariantError(
            "Terminal decision coverage differs from canonical offers: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    terminal_states = _text_values(decisions, "final_decision")
    allowed_states = {state.value for state in FinalMatchDecision}
    invalid_states = sorted(set(terminal_states) - allowed_states)
    if terminal_states.eq("").any() or invalid_states:
        raise PipelineInvariantError(
            f"Invalid or missing terminal decision states: {invalid_states}"
        )

    decision_by_offer = decisions.set_index(
        decisions["offer_id"].astype(str)
    )
    if "is_own" in offers:
        competitor_ids = set(
            offer_ids.loc[
                ~offers["is_own"].fillna(False).astype(bool)
            ]
        )
        if competitor_ids:
            competitor_decisions = decision_by_offer.loc[
                sorted(competitor_ids)
            ]
            if not _text_values(
                competitor_decisions, "final_decision"
            ).eq(FinalMatchDecision.COMPETITOR_OFFER.value).all():
                raise PipelineInvariantError(
                    "Non-own-brand offers require COMPETITOR_OFFER state"
                )
            if competitor_decisions[
                "final_eligible_mapping"
            ].fillna(False).astype(bool).any():
                raise PipelineInvariantError(
                    "Competitor offers cannot be accepted SKU mappings"
                )

    eligible = decisions["final_eligible_mapping"].fillna(False).astype(bool)
    matched = _text_values(decisions, "matched_master_sku")
    if matched.loc[eligible].eq("").any():
        raise PipelineInvariantError(
            "Eligible mappings require an accepted master SKU"
        )
    if matched.loc[~eligible].ne("").any():
        raise PipelineInvariantError(
            "Non-eligible decisions cannot expose an accepted master SKU"
        )

    if not candidates.empty and "offer_group_id" in candidates:
        candidate_offer_ids = set(
            _text_values(candidates, "offer_group_id").loc[
                _text_values(candidates, "master_itemcode").ne("")
            ]
        )
        proposed = _text_values(decisions, "proposed_master_sku")
        proposal_by_offer = pd.Series(
            proposed.to_numpy(), index=decision_ids.to_numpy()
        )
        missing_proposals = sorted(
            offer_id
            for offer_id in candidate_offer_ids
            if not proposal_by_offer.get(offer_id, "")
        )
        if missing_proposals:
            raise PipelineInvariantError(
                "Retained candidates require a proposed master SKU: "
                f"{missing_proposals[:5]}"
            )
