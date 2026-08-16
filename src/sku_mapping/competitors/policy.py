"""Automatic competitor decision policy.

Competitor discovery ends in an automatic ACCEPT or REJECT. There is no
production human route: everything this module cannot settle is handed to the
LLM adjudicator, and everything the adjudicator cannot settle is rejected.
Rejection is the safe direction because a missed competitor understates a
rival's presence, while a wrong one asserts a rivalry that does not exist.

The thresholds here are PROVISIONAL. They were taken from a measured
distribution rather than chosen, and they have never been validated against
human competitor labels, because none exist. See ``docs/`` and the module
constants for the measurement they came from.

Nothing in this module reads the own-brand ``lightgbm_auto_accept_threshold``.
That number is a calibrated probability for a different question - "is this
Al Kabeer offer this Al Kabeer SKU" - and the competitor signal is an
uncalibrated raw margin comparable only within one offer's shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import inf, isfinite

#: Statuses the rules admit as a real competitor relationship. Mirrors
#: ``discovery.SUPPORTED_COMPETITOR_STATUSES``; duplicated as a parameter
#: default rather than imported so this module stays free of discovery.
SUPPORTED_STATUSES = frozenset({"MATCHED", "AMBIGUOUS"})

#: Columns the decision layer appends to a discovery long frame. Declared here
#: rather than beside the frame code so discovery can name them without
#: importing the LLM stack.
DECISION_COLUMNS = (
    "competitor_decision",
    "competitor_decision_reason",
    "competitor_decision_source",
)

#: Where the shipped defaults come from. An 8,000-row real slice produced
#: 2,683 supported relationships over 1,302 competitor offers, 99.1% of them
#: scored by the model:
#:
#:   rank-1 raw margin   p10 -8.49  p25 -4.99  p50 -1.90  p75 +2.05  p90 +5.11
#:   rank1-rank2 gap     p10  0.00  p25  0.00  p50  2.12  p75  5.35  p90  9.48
#:
#: A quarter of offers have a gap of exactly zero - the model cannot separate
#: its first and second choice at all - which is the population the adjudicator
#: exists to resolve. The defaults are the sign boundary of the margin and the
#: median of the gap, not round numbers chosen for looking tidy.
MEASURED_SLICE_ROWS = 8_000
MEASURED_SUPPORTED_RELATIONSHIPS = 2_683
MEASURED_OFFERS = 1_302
DEFAULT_CLEAR_MARGIN = 0.0
DEFAULT_CLEAR_GAP = 2.0


class CompetitorBand(str, Enum):
    """Routing band for one competitor offer's shortlist."""

    #: The model both likes its top candidate and separates it from the rest.
    CLEAR = "CLEAR"
    #: Admitted by the rules but not separated by the model. Goes to the LLM.
    AMBIGUOUS = "AMBIGUOUS"
    #: Nothing survived the rules. Never reaches the LLM.
    REJECT = "REJECT"


class CompetitorDecision(str, Enum):
    """Terminal automatic outcome. There is no third value by design."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class CompetitorDecisionReason(str, Enum):
    """Stable audit vocabulary for why a relationship ended as it did."""

    # Settled without the adjudicator.
    CLEAR_MARGIN_AND_GAP = "CLEAR_MARGIN_AND_GAP"
    OUTRANKED_BY_CLEAR_WINNER = "OUTRANKED_BY_CLEAR_WINNER"
    OUTRANKED_BY_LLM_SELECTION = "OUTRANKED_BY_LLM_SELECTION"
    BELOW_ELIGIBILITY_POLICY = "BELOW_ELIGIBILITY_POLICY"
    HARD_CONFLICT = "HARD_CONFLICT"

    # Why the adjudicator was consulted.
    AMBIGUOUS_LOW_MARGIN = "AMBIGUOUS_LOW_MARGIN"
    AMBIGUOUS_NARROW_GAP = "AMBIGUOUS_NARROW_GAP"
    AMBIGUOUS_NO_MODEL_SCORE = "AMBIGUOUS_NO_MODEL_SCORE"
    AMBIGUOUS_PACK_UNVERIFIED = "AMBIGUOUS_PACK_UNVERIFIED"

    # What the adjudicator said. Every non-accept lands on REJECTED.
    LLM_ACCEPTED = "LLM_ACCEPTED"
    LLM_REJECTED = "LLM_REJECTED"
    LLM_UNCERTAIN = "LLM_UNCERTAIN"
    LLM_UNKNOWN_CANDIDATE = "LLM_UNKNOWN_CANDIDATE"
    LLM_MALFORMED_RESPONSE = "LLM_MALFORMED_RESPONSE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_PROVIDER_FAILURE = "LLM_PROVIDER_FAILURE"
    LLM_DISABLED = "LLM_DISABLED"


#: Every reason that terminates in ACCEPTED. Anything not listed rejects, so a
#: reason code added later cannot accidentally become an acceptance.
ACCEPTING_REASONS = frozenset(
    {
        CompetitorDecisionReason.CLEAR_MARGIN_AND_GAP,
        CompetitorDecisionReason.LLM_ACCEPTED,
    }
)


@dataclass(frozen=True)
class CandidateSignal:
    """One (master SKU, competitor offer) relationship the rules produced.

    ``model_score`` is the raw margin from the borrowed own-brand ranker. It is
    NOT a probability and is comparable only against other candidates for the
    same offer.
    """

    master_sku: str
    status: str
    fuzzy_score: float
    adjusted_score: float
    model_score: float | None = None
    model_rank: int | None = None
    #: ``False`` only when the rules positively could not verify the pack, i.e.
    #: the ``AMBIGUOUS``/``SUPPORTED_PACK_UNVERIFIED`` case. A known conflict
    #: never reaches here - it is already ``HARD_CONFLICT``.
    pack_verified: bool = True

    @property
    def is_supported(self) -> bool:
        return self.status in SUPPORTED_STATUSES


@dataclass(frozen=True)
class OfferClassification:
    """How one competitor offer's shortlist was routed."""

    band: CompetitorBand
    reason: CompetitorDecisionReason
    #: The candidate that would be accepted if the band is CLEAR, or the one
    #: proposed to the adjudicator first if AMBIGUOUS. ``None`` when nothing
    #: was admitted at all.
    winner: str | None
    top_score: float | None
    runner_up_score: float | None
    #: ``inf`` when a lone candidate has nothing to be separated from.
    gap: float | None
    #: Every admitted candidate, best first. Bounded by the caller before it
    #: ever reaches a provider.
    ranked: tuple[str, ...]

    @property
    def needs_adjudication(self) -> bool:
        return self.band is CompetitorBand.AMBIGUOUS


def _ordering_key(candidate: CandidateSignal) -> tuple[float, float, str]:
    """Deterministic best-first order.

    Ties are broken by master SKU so two runs over the same data cannot
    disagree about which of two identically scored candidates won. Sorting on
    the score alone would leave the order at the mercy of dict insertion.
    """
    score = candidate.model_score
    return (
        -(score if score is not None else -inf),
        -candidate.adjusted_score,
        candidate.master_sku,
    )


def classify_offer(
    candidates: list[CandidateSignal],
    *,
    clear_margin: float = DEFAULT_CLEAR_MARGIN,
    clear_gap: float = DEFAULT_CLEAR_GAP,
    max_adjudicated_candidates: int = 5,
) -> OfferClassification:
    """Route one competitor offer's shortlist into a band.

    Pure. No model, no network, no frame. The gates are applied in order of
    how much they prove, so the recorded reason is the strongest thing that
    was wrong rather than whichever check happened to run last.
    """
    supported = [candidate for candidate in candidates if candidate.is_supported]
    if not supported:
        return OfferClassification(
            band=CompetitorBand.REJECT,
            reason=CompetitorDecisionReason.BELOW_ELIGIBILITY_POLICY,
            winner=None,
            top_score=None,
            runner_up_score=None,
            gap=None,
            ranked=(),
        )

    # One master SKU, one candidate. The same relationship is recorded once per
    # source Al Kabeer offer, so a shortlist can hold several rows for one SKU.
    # Left in, they become each other's runner-up: the gap collapses to zero
    # and a relationship the model was certain about is forced into ambiguity
    # by nothing but its own duplicates. Keep the best row per SKU.
    ordered: list[CandidateSignal] = []
    seen: set[str] = set()
    for candidate in sorted(supported, key=_ordering_key):
        if candidate.master_sku in seen:
            continue
        seen.add(candidate.master_sku)
        ordered.append(candidate)
    ranked = tuple(candidate.master_sku for candidate in ordered)
    bounded = ranked[: max(1, max_adjudicated_candidates)]
    scored = [
        candidate for candidate in ordered if candidate.model_score is not None
    ]

    # No opinion from the model at all: the rules admitted these on fuzzy
    # evidence alone. That is exactly the case the adjudicator is for, and
    # never something to accept automatically.
    if not scored:
        return OfferClassification(
            band=CompetitorBand.AMBIGUOUS,
            reason=CompetitorDecisionReason.AMBIGUOUS_NO_MODEL_SCORE,
            winner=ordered[0].master_sku,
            top_score=None,
            runner_up_score=None,
            gap=None,
            ranked=bounded,
        )

    top = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    top_score = float(top.model_score)  # type: ignore[arg-type]
    runner_up_score = (
        float(runner_up.model_score) if runner_up is not None else None  # type: ignore[arg-type]
    )
    # A lone candidate has nothing to be separated from, so the gap gate cannot
    # fail it. The margin gate still can.
    gap = inf if runner_up_score is None else top_score - runner_up_score

    def ambiguous(reason: CompetitorDecisionReason) -> OfferClassification:
        return OfferClassification(
            band=CompetitorBand.AMBIGUOUS,
            reason=reason,
            winner=top.master_sku,
            top_score=top_score,
            runner_up_score=runner_up_score,
            # A lone candidate's gap is infinite, which is true but not
            # serialisable - the adjudicator payload is strict JSON. "No
            # runner-up to measure against" is the honest reading, and it
            # travels.
            gap=gap if isfinite(gap) else None,
            ranked=bounded,
        )

    if not (top_score > clear_margin):
        return ambiguous(CompetitorDecisionReason.AMBIGUOUS_LOW_MARGIN)
    if gap < clear_gap:
        return ambiguous(CompetitorDecisionReason.AMBIGUOUS_NARROW_GAP)
    if not top.pack_verified:
        return ambiguous(CompetitorDecisionReason.AMBIGUOUS_PACK_UNVERIFIED)

    return OfferClassification(
        band=CompetitorBand.CLEAR,
        reason=CompetitorDecisionReason.CLEAR_MARGIN_AND_GAP,
        winner=top.master_sku,
        top_score=top_score,
        runner_up_score=runner_up_score,
        gap=None if not isfinite(gap) else gap,
        ranked=bounded,
    )


@dataclass(frozen=True)
class RelationshipOutcome:
    """The terminal automatic decision for one relationship row."""

    master_sku: str
    decision: CompetitorDecision
    reason: CompetitorDecisionReason
    source: str

    @property
    def accepted(self) -> bool:
        return self.decision is CompetitorDecision.ACCEPTED


def resolve_outcomes(
    classification: OfferClassification,
    candidates: list[CandidateSignal],
    *,
    selected_master: str | None = None,
    selected_reason: CompetitorDecisionReason | None = None,
    source: str = "rules_ml",
) -> dict[str, RelationshipOutcome]:
    """Turn a routed offer into one terminal outcome per relationship.

    At most one master SKU per competitor offer is accepted: an offer is one
    product on one flyer, so "which Al Kabeer SKU does it compete with" has a
    single answer. Every other admitted candidate is rejected as outranked
    rather than silently dropped, so the audit shows what lost and to what.

    ``selected_master`` carries the adjudicator's verdict. ``None`` with a
    ``selected_reason`` means it declined, failed, or was never asked - all of
    which reject the whole shortlist.
    """
    outcomes: dict[str, RelationshipOutcome] = {}
    if classification.band is CompetitorBand.REJECT:
        reason = CompetitorDecisionReason.BELOW_ELIGIBILITY_POLICY
        for candidate in candidates:
            outcomes[candidate.master_sku] = RelationshipOutcome(
                master_sku=candidate.master_sku,
                decision=CompetitorDecision.REJECTED,
                reason=reason,
                source=source,
            )
        return outcomes

    if classification.band is CompetitorBand.CLEAR:
        winner = classification.winner
        winning_reason = CompetitorDecisionReason.CLEAR_MARGIN_AND_GAP
        losing_reason = CompetitorDecisionReason.OUTRANKED_BY_CLEAR_WINNER
    else:
        winner = selected_master
        winning_reason = selected_reason or CompetitorDecisionReason.LLM_ACCEPTED
        losing_reason = (
            CompetitorDecisionReason.OUTRANKED_BY_LLM_SELECTION
            if winner is not None
            else (selected_reason or CompetitorDecisionReason.LLM_DISABLED)
        )

    accepted_is_real = (
        winner is not None and winning_reason in ACCEPTING_REASONS
    )
    for candidate in candidates:
        is_winner = accepted_is_real and candidate.master_sku == winner
        if is_winner:
            outcomes[candidate.master_sku] = RelationshipOutcome(
                master_sku=candidate.master_sku,
                decision=CompetitorDecision.ACCEPTED,
                reason=winning_reason,
                source=source,
            )
            continue
        if not candidate.is_supported:
            reason = (
                CompetitorDecisionReason.HARD_CONFLICT
                if candidate.status == "HARD_CONFLICT"
                else CompetitorDecisionReason.BELOW_ELIGIBILITY_POLICY
            )
        elif accepted_is_real:
            reason = losing_reason
        else:
            # Nothing was accepted for this offer, so every admitted row
            # carries the verdict that sank the whole shortlist.
            reason = winning_reason
        outcomes[candidate.master_sku] = RelationshipOutcome(
            master_sku=candidate.master_sku,
            decision=CompetitorDecision.REJECTED,
            reason=reason,
            source=source,
        )
    return outcomes
