"""Apply the automatic competitor decision to a discovery long frame.

Discovery loops one master SKU against many offers, but a decision belongs to
one competitor OFFER: "which Al Kabeer SKU, if any, does this rival product
compete with" has a single answer. So the decision runs here, once, over the
assembled long frame, after every target has been evaluated and ranked.

Three columns are added and none are overwritten. The rule status and the
model score stay exactly as discovery recorded them, so an audit can still see
what the rules admitted and what the model thought before the decision was
taken.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

from sku_mapping.competitors.adjudicator import (
    AdjudicationCandidate,
    build_request,
    disabled_verdict,
)
from sku_mapping.competitors.policy import (
    DECISION_COLUMNS,
    CandidateSignal,
    CompetitorBand,
    CompetitorDecision,
    CompetitorDecisionReason,
    classify_offer,
    resolve_outcomes,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DECISION_COLUMNS",
    "NOT_DECIDED",
    "accepted_only",
    "apply_automatic_decisions",
]

#: Emitted when the decision layer is switched off, so a downstream reader can
#: tell "not decided" apart from "decided to reject".
NOT_DECIDED = "NOT_DECIDED"


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _number(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _signal(row: pd.Series) -> CandidateSignal:
    status = _text(row.get("competitor_match_status"))
    return CandidateSignal(
        master_sku=_text(row.get("master_sku")),
        status=status,
        fuzzy_score=_number(row.get("competitor_match_score")) or 0.0,
        adjusted_score=_number(row.get("competitor_adjusted_score")) or 0.0,
        model_score=_number(row.get("competitor_lightgbm_score")),
        model_rank=(
            int(rank)
            if (rank := _number(row.get("competitor_lightgbm_rank"))) is not None
            else None
        ),
        # The rules mark a relationship AMBIGUOUS precisely when they could not
        # verify the pack. A known pack conflict is already HARD_CONFLICT and
        # never reaches the supported set.
        pack_verified=status != "AMBIGUOUS",
    )


def _candidate(row: pd.Series) -> AdjudicationCandidate:
    return AdjudicationCandidate(
        master_sku=_text(row.get("master_sku")),
        master_name=_text(row.get("master_name")),
        master_description=_text(row.get("master_description")),
        fuzzy_score=_number(row.get("competitor_match_score")),
        adjusted_score=_number(row.get("competitor_adjusted_score")),
        model_score=_number(row.get("competitor_lightgbm_score")),
        model_rank=(
            int(rank)
            if (rank := _number(row.get("competitor_lightgbm_rank"))) is not None
            else None
        ),
        pack_status=(
            "UNVERIFIED"
            if _text(row.get("competitor_match_status")) == "AMBIGUOUS"
            else "COMPATIBLE"
        ),
        rule_reason=_text(row.get("competitor_match_reason")),
    )


def apply_automatic_decisions(
    long_frame: pd.DataFrame,
    *,
    clear_margin: float = 0.0,
    clear_gap: float = 2.0,
    adjudicator: Any | None = None,
    max_adjudicated_candidates: int = 5,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Decide every relationship, returning the frame plus counters.

    Every row leaves with ACCEPTED or REJECTED. Nothing is routed to a human,
    and no row is dropped: a rejection is recorded, not deleted, so the audit
    keeps the full picture of what was considered.
    """
    stats: dict[str, int] = {
        "offers": 0,
        "clear_offers": 0,
        "ambiguous_offers": 0,
        "rejected_offers": 0,
        "adjudicated_offers": 0,
        "accepted_relationships": 0,
        "rejected_relationships": 0,
        "llm_accepts": 0,
        "llm_rejects": 0,
        "llm_failures": 0,
    }
    frame = long_frame.copy()
    if frame.empty:
        for column in DECISION_COLUMNS:
            frame[column] = pd.Series(dtype="object")
        return frame, stats

    decisions = pd.Series(NOT_DECIDED, index=frame.index, dtype="object")
    reasons = pd.Series("", index=frame.index, dtype="object")
    sources = pd.Series("", index=frame.index, dtype="object")

    offer_ids = frame["competitor_offer_id"].map(_text)
    grouped = list(frame.groupby(offer_ids, sort=True))
    for position, (offer_id, group) in enumerate(grouped, start=1):
        signals = [_signal(row) for _, row in group.iterrows()]
        classification = classify_offer(
            signals,
            clear_margin=clear_margin,
            clear_gap=clear_gap,
            max_adjudicated_candidates=max_adjudicated_candidates,
        )
        stats["offers"] += 1

        selected_master: str | None = None
        selected_reason: CompetitorDecisionReason | None = None
        source = "rules_ml"

        if classification.band is CompetitorBand.CLEAR:
            stats["clear_offers"] += 1
        elif classification.band is CompetitorBand.REJECT:
            stats["rejected_offers"] += 1
        else:
            stats["ambiguous_offers"] += 1
            source = "llm"
            if adjudicator is None:
                verdict = disabled_verdict(offer_id)
            else:
                first = group.iloc[0]
                admitted = {
                    _text(row.get("master_sku")): row
                    for _, row in group.iterrows()
                    if _text(row.get("competitor_match_status"))
                    in {"MATCHED", "AMBIGUOUS"}
                }
                verdict = adjudicator.adjudicate(
                    build_request(
                        classification,
                        offer_id=offer_id,
                        offer_name=_text(first.get("competitor_offer_name")),
                        candidates=tuple(
                            _candidate(row) for row in admitted.values()
                        ),
                        competitor_brand=_text(first.get("competitor_brand")),
                        competitor_product=_text(first.get("competitor_product")),
                        competitor_variant=_text(first.get("competitor_variant")),
                        competitor_pack_size=_text(first.get("competitor_pack_size")),
                        competitor_retailer=_text(first.get("competitor_retailer")),
                    )
                )
                stats["adjudicated_offers"] += 1
                if verdict.reason is CompetitorDecisionReason.LLM_ACCEPTED:
                    stats["llm_accepts"] += 1
                elif verdict.reason in {
                    CompetitorDecisionReason.LLM_REJECTED,
                    CompetitorDecisionReason.LLM_UNCERTAIN,
                }:
                    stats["llm_rejects"] += 1
                else:
                    stats["llm_failures"] += 1
            selected_master = verdict.selected_master
            selected_reason = verdict.reason

        outcomes = resolve_outcomes(
            classification,
            signals,
            selected_master=selected_master,
            selected_reason=selected_reason,
            source=source,
        )
        for index, row in group.iterrows():
            outcome = outcomes.get(_text(row.get("master_sku")))
            if outcome is None:
                continue
            decisions.loc[index] = outcome.decision.value
            reasons.loc[index] = outcome.reason.value
            sources.loc[index] = outcome.source
            if outcome.accepted:
                stats["accepted_relationships"] += 1
            else:
                stats["rejected_relationships"] += 1
        if progress is not None:
            progress(position, len(grouped))

    frame["competitor_decision"] = decisions
    frame["competitor_decision_reason"] = reasons
    frame["competitor_decision_source"] = sources
    return frame, stats


def accepted_only(frame: pd.DataFrame) -> pd.DataFrame:
    """The accepted competitor relationships, for a business export.

    A frame that was never decided is returned untouched rather than emptied,
    so a rules-only run does not silently export nothing.
    """
    if "competitor_decision" not in frame.columns:
        return frame
    if (frame["competitor_decision"] == NOT_DECIDED).all():
        return frame
    return frame[frame["competitor_decision"] == CompetitorDecision.ACCEPTED.value]
