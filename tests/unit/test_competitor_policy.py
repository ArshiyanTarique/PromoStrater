"""Automatic competitor decision policy.

These tests pin the bands and the terminal outcomes. They deliberately use no
model and no provider: the policy is pure, and a threshold change has to show
up here before it can show up in a run.
"""

from __future__ import annotations

import pytest

from sku_mapping.competitors.policy import (
    ACCEPTING_REASONS,
    DEFAULT_CLEAR_GAP,
    DEFAULT_CLEAR_MARGIN,
    CandidateSignal,
    CompetitorBand,
    CompetitorDecision,
    CompetitorDecisionReason,
    classify_offer,
    resolve_outcomes,
)


def signal(
    master_sku: str,
    *,
    status: str = "MATCHED",
    fuzzy: float = 80.0,
    adjusted: float = 83.0,
    model_score: float | None = None,
    pack_verified: bool = True,
) -> CandidateSignal:
    return CandidateSignal(
        master_sku=master_sku,
        status=status,
        fuzzy_score=fuzzy,
        adjusted_score=adjusted,
        model_score=model_score,
        pack_verified=pack_verified,
    )


class TestBands:
    def test_strong_margin_and_wide_gap_is_clear(self) -> None:
        result = classify_offer(
            [signal("A", model_score=5.0), signal("B", model_score=1.0)]
        )
        assert result.band is CompetitorBand.CLEAR
        assert result.winner == "A"
        assert result.reason is CompetitorDecisionReason.CLEAR_MARGIN_AND_GAP
        assert result.gap == pytest.approx(4.0)

    def test_zero_gap_is_ambiguous_even_with_a_strong_margin(self) -> None:
        """The measured 25% of offers where the model separates nothing."""
        result = classify_offer(
            [signal("A", model_score=6.0), signal("B", model_score=6.0)]
        )
        assert result.band is CompetitorBand.AMBIGUOUS
        assert result.reason is CompetitorDecisionReason.AMBIGUOUS_NARROW_GAP
        assert result.gap == pytest.approx(0.0)

    def test_wide_gap_below_the_margin_floor_is_ambiguous(self) -> None:
        result = classify_offer(
            [signal("A", model_score=-1.0), signal("B", model_score=-30.0)]
        )
        assert result.band is CompetitorBand.AMBIGUOUS
        assert result.reason is CompetitorDecisionReason.AMBIGUOUS_LOW_MARGIN

    def test_margin_is_checked_before_gap(self) -> None:
        """Both gates fail; the reason names the stronger evidence."""
        result = classify_offer(
            [signal("A", model_score=-4.0), signal("B", model_score=-4.0)]
        )
        assert result.reason is CompetitorDecisionReason.AMBIGUOUS_LOW_MARGIN

    def test_a_lone_candidate_has_no_gap_to_fail(self) -> None:
        result = classify_offer([signal("A", model_score=3.0)])
        assert result.band is CompetitorBand.CLEAR
        assert result.runner_up_score is None

    def test_a_lone_candidate_still_faces_the_margin_gate(self) -> None:
        result = classify_offer([signal("A", model_score=-3.0)])
        assert result.band is CompetitorBand.AMBIGUOUS
        assert result.reason is CompetitorDecisionReason.AMBIGUOUS_LOW_MARGIN

    def test_unscored_candidates_are_never_accepted_automatically(self) -> None:
        result = classify_offer([signal("A"), signal("B")])
        assert result.band is CompetitorBand.AMBIGUOUS
        assert result.reason is CompetitorDecisionReason.AMBIGUOUS_NO_MODEL_SCORE

    def test_unverified_pack_blocks_an_otherwise_clear_winner(self) -> None:
        result = classify_offer(
            [
                signal("A", status="AMBIGUOUS", model_score=9.0, pack_verified=False),
                signal("B", model_score=1.0),
            ]
        )
        assert result.band is CompetitorBand.AMBIGUOUS
        assert result.reason is CompetitorDecisionReason.AMBIGUOUS_PACK_UNVERIFIED

    def test_nothing_admitted_rejects_without_adjudication(self) -> None:
        result = classify_offer(
            [
                signal("A", status="HARD_CONFLICT", model_score=9.0),
                signal("B", status="UNRELATED", model_score=8.0),
            ]
        )
        assert result.band is CompetitorBand.REJECT
        assert result.winner is None
        assert result.needs_adjudication is False

    def test_conflicting_rows_cannot_win_on_a_high_model_score(self) -> None:
        """A hard conflict outscoring everything must not become the winner."""
        result = classify_offer(
            [
                signal("CONFLICT", status="HARD_CONFLICT", model_score=99.0),
                signal("OK", model_score=4.0),
            ]
        )
        assert result.winner == "OK"


class TestDeterminism:
    def test_tied_scores_break_on_master_sku_not_input_order(self) -> None:
        forward = classify_offer(
            [signal("Z", model_score=5.0), signal("A", model_score=5.0)]
        )
        reverse = classify_offer(
            [signal("A", model_score=5.0), signal("Z", model_score=5.0)]
        )
        assert forward.winner == reverse.winner == "A"
        assert forward.ranked == reverse.ranked

    def test_scored_candidates_outrank_unscored_ones(self) -> None:
        result = classify_offer(
            [signal("UNSCORED", adjusted=99.0), signal("SCORED", model_score=-9.0)]
        )
        assert result.ranked[0] == "SCORED"

    def test_the_adjudicated_shortlist_is_bounded(self) -> None:
        many = [signal(f"SKU{index}", model_score=float(index)) for index in range(40)]
        result = classify_offer(many, max_adjudicated_candidates=5)
        assert len(result.ranked) == 5
        assert result.ranked[0] == "SKU39"


class TestOutcomes:
    def test_a_clear_offer_accepts_exactly_one_and_outranks_the_rest(self) -> None:
        candidates = [
            signal("A", model_score=5.0),
            signal("B", model_score=1.0),
            signal("C", model_score=0.5),
        ]
        outcomes = resolve_outcomes(classify_offer(candidates), candidates)
        assert outcomes["A"].decision is CompetitorDecision.ACCEPTED
        assert [outcomes[sku].decision for sku in ("B", "C")] == [
            CompetitorDecision.REJECTED,
            CompetitorDecision.REJECTED,
        ]
        assert (
            outcomes["B"].reason
            is CompetitorDecisionReason.OUTRANKED_BY_CLEAR_WINNER
        )

    def test_every_relationship_gets_a_terminal_decision(self) -> None:
        """No production human route: nothing may be left unresolved."""
        candidates = [
            signal("A", model_score=5.0),
            signal("B", status="HARD_CONFLICT"),
            signal("C", status="UNRELATED"),
        ]
        outcomes = resolve_outcomes(classify_offer(candidates), candidates)
        assert set(outcomes) == {"A", "B", "C"}
        assert all(
            outcome.decision
            in (CompetitorDecision.ACCEPTED, CompetitorDecision.REJECTED)
            for outcome in outcomes.values()
        )
        assert outcomes["B"].reason is CompetitorDecisionReason.HARD_CONFLICT

    def test_an_adjudicated_acceptance_wins_the_offer(self) -> None:
        candidates = [
            signal("A", model_score=6.0),
            signal("B", model_score=6.0),
        ]
        classification = classify_offer(candidates)
        outcomes = resolve_outcomes(
            classification,
            candidates,
            selected_master="B",
            selected_reason=CompetitorDecisionReason.LLM_ACCEPTED,
            source="gemini",
        )
        assert outcomes["B"].decision is CompetitorDecision.ACCEPTED
        assert outcomes["A"].decision is CompetitorDecision.REJECTED
        assert (
            outcomes["A"].reason
            is CompetitorDecisionReason.OUTRANKED_BY_LLM_SELECTION
        )

    @pytest.mark.parametrize(
        "reason",
        [
            CompetitorDecisionReason.LLM_REJECTED,
            CompetitorDecisionReason.LLM_UNCERTAIN,
            CompetitorDecisionReason.LLM_TIMEOUT,
            CompetitorDecisionReason.LLM_MALFORMED_RESPONSE,
            CompetitorDecisionReason.LLM_UNKNOWN_CANDIDATE,
            CompetitorDecisionReason.LLM_PROVIDER_FAILURE,
            CompetitorDecisionReason.LLM_DISABLED,
        ],
    )
    def test_every_non_acceptance_rejects_the_whole_shortlist(
        self, reason: CompetitorDecisionReason
    ) -> None:
        candidates = [
            signal("A", model_score=6.0),
            signal("B", model_score=6.0),
        ]
        outcomes = resolve_outcomes(
            classify_offer(candidates),
            candidates,
            selected_master=None,
            selected_reason=reason,
            source="gemini",
        )
        assert all(
            outcome.decision is CompetitorDecision.REJECTED
            for outcome in outcomes.values()
        )
        assert {outcome.reason for outcome in outcomes.values()} == {reason}

    def test_a_selected_master_with_a_rejecting_reason_cannot_accept(self) -> None:
        """A verdict that names a SKU but did not accept it must not slip through."""
        candidates = [signal("A", model_score=6.0), signal("B", model_score=6.0)]
        outcomes = resolve_outcomes(
            classify_offer(candidates),
            candidates,
            selected_master="A",
            selected_reason=CompetitorDecisionReason.LLM_UNCERTAIN,
            source="gemini",
        )
        assert outcomes["A"].decision is CompetitorDecision.REJECTED

    def test_only_two_reasons_can_ever_accept(self) -> None:
        assert ACCEPTING_REASONS == {
            CompetitorDecisionReason.CLEAR_MARGIN_AND_GAP,
            CompetitorDecisionReason.LLM_ACCEPTED,
        }


class TestProvisionalDefaults:
    def test_defaults_match_the_measured_distribution(self) -> None:
        """Sign boundary of the margin, median of the gap. Change with data."""
        assert DEFAULT_CLEAR_MARGIN == 0.0
        assert DEFAULT_CLEAR_GAP == 2.0

    def test_the_own_brand_threshold_is_not_reused(self) -> None:
        assert DEFAULT_CLEAR_MARGIN != 0.95
        assert DEFAULT_CLEAR_GAP != 0.95

    def test_thresholds_are_injectable(self) -> None:
        candidates = [signal("A", model_score=1.0), signal("B", model_score=0.0)]
        assert classify_offer(candidates).band is CompetitorBand.AMBIGUOUS
        strict = classify_offer(candidates, clear_gap=0.5)
        assert strict.band is CompetitorBand.CLEAR
