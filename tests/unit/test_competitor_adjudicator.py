"""Competitor LLM adjudication and its failure modes.

The point of these tests is the safety direction: there is no production human
route, so every abnormal path must reject rather than accept or raise. No
network is touched - the provider is a stub.
"""

from __future__ import annotations

import json

import pytest

from sku_mapping.competitors.adjudicator import (
    AdjudicationCandidate,
    AdjudicationRequest,
    CompetitorAdjudicator,
    build_request,
    disabled_verdict,
)
from sku_mapping.competitors.policy import (
    CandidateSignal,
    CompetitorDecision,
    CompetitorDecisionReason,
    classify_offer,
    resolve_outcomes,
)
from sku_mapping.llm_review.provider import LLMProviderError, LLMProviderTimeout


class StubProvider:
    """Returns a canned response or raises a canned error."""

    provider_name = "stub"
    model_name = "stub-model"
    model_id = "stub:stub-model"

    def __init__(self, response: str | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.requests: list[str] = []

    def generate(self, *, structured_request: str, system_prompt: str, **_: object) -> str:
        self.requests.append(structured_request)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def response(
    decision: str = "ACCEPT_CANDIDATE",
    candidate: str | None = "SKU-A",
    confidence: float = 0.9,
    reasons: list[str] | None = None,
    explanation: str = "Same protein, family, and pack.",
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "selected_candidate_id": candidate,
            "confidence": confidence,
            "reason_codes": reasons if reasons is not None else ["FAMILY_MATCH"],
            "short_explanation": explanation,
        }
    )


def request(*skus: str) -> AdjudicationRequest:
    return AdjudicationRequest(
        offer_id="OFFER-1",
        offer_name="Rival Chicken Samosa 240g",
        competitor_brand="RivalBrand",
        candidates=tuple(
            AdjudicationCandidate(
                master_sku=sku, master_name=f"Al Kabeer {sku}", model_score=1.0
            )
            for sku in (skus or ("SKU-A", "SKU-B"))
        ),
        ambiguity_reason="AMBIGUOUS_NARROW_GAP",
    )


class TestAcceptance:
    def test_accept_selects_the_named_candidate(self) -> None:
        adjudicator = CompetitorAdjudicator(provider=StubProvider(response()))
        verdict = adjudicator.adjudicate(request())
        assert verdict.reason is CompetitorDecisionReason.LLM_ACCEPTED
        assert verdict.selected_master == "SKU-A"
        assert verdict.accepted is True
        assert adjudicator.accepts == 1

    def test_the_offer_and_candidates_reach_the_provider(self) -> None:
        provider = StubProvider(response())
        CompetitorAdjudicator(provider=provider).adjudicate(request())
        payload = json.loads(provider.requests[0])
        assert payload["competitor_offer"]["brand"] == "RivalBrand"
        assert {c["candidate_id"] for c in payload["candidates"]} == {"SKU-A", "SKU-B"}
        assert payload["why_this_needs_a_decision"]["ambiguity_reason"] == (
            "AMBIGUOUS_NARROW_GAP"
        )

    def test_the_model_score_is_never_labelled_a_probability(self) -> None:
        provider = StubProvider(response())
        CompetitorAdjudicator(provider=provider).adjudicate(request())
        payload = json.loads(provider.requests[0])
        keys = set(payload["candidates"][0])
        assert "model_raw_margin_within_this_offer" in keys
        assert not any("probab" in key.lower() for key in keys)


class TestRejectionPaths:
    @pytest.mark.parametrize(
        ("decision", "expected"),
        [
            ("REJECT_ALL", CompetitorDecisionReason.LLM_REJECTED),
            ("UNCERTAIN", CompetitorDecisionReason.LLM_UNCERTAIN),
        ],
    )
    def test_non_accept_decisions_reject(self, decision: str, expected) -> None:
        adjudicator = CompetitorAdjudicator(
            provider=StubProvider(response(decision, candidate=None))
        )
        verdict = adjudicator.adjudicate(request())
        assert verdict.reason is expected
        assert verdict.selected_master is None
        assert verdict.accepted is False

    def test_a_timeout_rejects_and_does_not_raise(self) -> None:
        adjudicator = CompetitorAdjudicator(
            provider=StubProvider(error=LLMProviderTimeout("timed out"))
        )
        verdict = adjudicator.adjudicate(request())
        assert verdict.reason is CompetitorDecisionReason.LLM_TIMEOUT
        assert adjudicator.failures == 1

    def test_a_provider_failure_rejects_and_does_not_raise(self) -> None:
        adjudicator = CompetitorAdjudicator(
            provider=StubProvider(error=LLMProviderError("endpoint down"))
        )
        assert (
            adjudicator.adjudicate(request()).reason
            is CompetitorDecisionReason.LLM_PROVIDER_FAILURE
        )

    def test_an_unexpected_exception_rejects_rather_than_killing_the_run(self) -> None:
        adjudicator = CompetitorAdjudicator(
            provider=StubProvider(error=RuntimeError("something unforeseen"))
        )
        assert (
            adjudicator.adjudicate(request()).reason
            is CompetitorDecisionReason.LLM_PROVIDER_FAILURE
        )

    @pytest.mark.parametrize(
        "payload",
        [
            "not json at all",
            "[]",
            '{"decision": "MAYBE", "selected_candidate_id": null, "confidence": 1, '
            '"reason_codes": [], "short_explanation": "x"}',
            '{"decision": "ACCEPT_CANDIDATE"}',
        ],
    )
    def test_malformed_responses_reject(self, payload: str) -> None:
        adjudicator = CompetitorAdjudicator(provider=StubProvider(payload))
        verdict = adjudicator.adjudicate(request())
        assert verdict.reason is CompetitorDecisionReason.LLM_MALFORMED_RESPONSE
        assert verdict.validation_errors

    def test_an_invented_master_sku_rejects_with_its_own_reason(self) -> None:
        """The provider may only choose from what the pipeline supplied."""
        adjudicator = CompetitorAdjudicator(
            provider=StubProvider(response(candidate="SKU-DOES-NOT-EXIST"))
        )
        verdict = adjudicator.adjudicate(request())
        assert verdict.reason is CompetitorDecisionReason.LLM_UNKNOWN_CANDIDATE
        assert verdict.selected_master is None

    def test_an_empty_candidate_set_never_calls_the_provider(self) -> None:
        provider = StubProvider(response())
        adjudicator = CompetitorAdjudicator(provider=provider)
        verdict = adjudicator.adjudicate(
            AdjudicationRequest(offer_id="O", offer_name="n")
        )
        assert verdict.reason is CompetitorDecisionReason.LLM_REJECTED
        assert provider.requests == []

    def test_a_disabled_adjudicator_rejects(self) -> None:
        assert (
            disabled_verdict("OFFER-1").reason
            is CompetitorDecisionReason.LLM_DISABLED
        )


class TestBounding:
    def test_the_candidate_universe_sent_is_capped(self) -> None:
        provider = StubProvider(response())
        adjudicator = CompetitorAdjudicator(provider=provider, max_candidates=3)
        adjudicator.adjudicate(request(*[f"SKU-{index}" for index in range(50)]))
        payload = json.loads(provider.requests[0])
        assert len(payload["candidates"]) == 3

    def test_a_candidate_beyond_the_cap_cannot_be_selected(self) -> None:
        """Capping must also narrow what counts as a supplied candidate."""
        adjudicator = CompetitorAdjudicator(
            provider=StubProvider(response(candidate="SKU-9")), max_candidates=2
        )
        verdict = adjudicator.adjudicate(
            request(*[f"SKU-{index}" for index in range(10)])
        )
        assert verdict.reason is CompetitorDecisionReason.LLM_UNKNOWN_CANDIDATE

    def test_identical_offers_are_only_asked_once(self) -> None:
        provider = StubProvider(response())
        adjudicator = CompetitorAdjudicator(provider=provider)
        adjudicator.adjudicate(request())
        adjudicator.adjudicate(request())
        assert len(provider.requests) == 1


class TestEndToEndRouting:
    """The policy and the adjudicator have to agree on the terminal outcome."""

    def _candidates(self) -> list[CandidateSignal]:
        return [
            CandidateSignal("SKU-A", "MATCHED", 80.0, 83.0, model_score=6.0),
            CandidateSignal("SKU-B", "MATCHED", 79.0, 82.0, model_score=6.0),
        ]

    def test_an_ambiguous_offer_accepted_by_the_llm_ends_accepted(self) -> None:
        candidates = self._candidates()
        classification = classify_offer(candidates)
        assert classification.needs_adjudication
        verdict = CompetitorAdjudicator(
            provider=StubProvider(response(candidate="SKU-B"))
        ).adjudicate(
            build_request(
                classification,
                offer_id="OFFER-1",
                offer_name="Rival Samosa",
                candidates=tuple(
                    AdjudicationCandidate(master_sku=c.master_sku, master_name=c.master_sku)
                    for c in candidates
                ),
            )
        )
        outcomes = resolve_outcomes(
            classification,
            candidates,
            selected_master=verdict.selected_master,
            selected_reason=verdict.reason,
            source="gemini",
        )
        assert outcomes["SKU-B"].decision is CompetitorDecision.ACCEPTED
        assert outcomes["SKU-A"].decision is CompetitorDecision.REJECTED

    @pytest.mark.parametrize(
        "provider",
        [
            StubProvider(response("UNCERTAIN", candidate=None)),
            StubProvider(error=LLMProviderTimeout("t")),
            StubProvider("garbage"),
            StubProvider(response(candidate="INVENTED")),
        ],
    )
    def test_every_unresolved_path_ends_with_nothing_accepted(
        self, provider: StubProvider
    ) -> None:
        candidates = self._candidates()
        classification = classify_offer(candidates)
        verdict = CompetitorAdjudicator(provider=provider).adjudicate(
            build_request(
                classification,
                offer_id="OFFER-1",
                offer_name="Rival Samosa",
                candidates=tuple(
                    AdjudicationCandidate(master_sku=c.master_sku, master_name=c.master_sku)
                    for c in candidates
                ),
            )
        )
        outcomes = resolve_outcomes(
            classification,
            candidates,
            selected_master=verdict.selected_master,
            selected_reason=verdict.reason,
            source="gemini",
        )
        assert all(
            outcome.decision is CompetitorDecision.REJECTED
            for outcome in outcomes.values()
        )

    def test_build_request_preserves_the_model_ordering(self) -> None:
        candidates = [
            CandidateSignal("LOW", "MATCHED", 70.0, 70.0, model_score=1.0),
            CandidateSignal("HIGH", "MATCHED", 70.0, 70.0, model_score=9.0),
        ]
        built = build_request(
            classify_offer(candidates),
            offer_id="O",
            offer_name="n",
            candidates=tuple(
                AdjudicationCandidate(master_sku=c.master_sku, master_name=c.master_sku)
                for c in candidates
            ),
        )
        assert [c.master_sku for c in built.candidates] == ["HIGH", "LOW"]
