"""LLM adjudication for ambiguous competitor offers.

This is the last stage before a terminal decision. It is consulted only for
the offers :mod:`sku_mapping.competitors.policy` could not settle, and it can
only ever choose from candidates the pipeline supplied - the strict parser it
reuses rejects a response naming anything else, so the provider cannot invent
a Master SKU.

Every failure mode ends the same way: REJECTED. A timeout, a malformed
envelope, an unknown candidate, an explicit UNCERTAIN, or a disabled reviewer
all reject the offer rather than escalating it, because there is no production
human route to escalate to.

The provider is the one built by :func:`create_llm_provider`. There is
deliberately no second Gemini client here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sku_mapping.competitors.policy import (
    CompetitorDecisionReason,
    OfferClassification,
)
from sku_mapping.llm_review.models import LLMDecision
from sku_mapping.llm_review.provider import LLMProviderTimeout
from sku_mapping.llm_review.reviewer import (
    LLMResponseValidationError,
    parse_llm_response,
)

LOGGER = logging.getLogger(__name__)

COMPETITOR_PROMPT_VERSION = "competitor-adjudication-v1"

SYSTEM_PROMPT = (
    "You decide whether a rival retailer's flyer offer competes with a "
    "specific Al Kabeer product.\n"
    "\n"
    "Two products compete when a shopper would buy one INSTEAD of the other: "
    "same food family, same protein, same form, and a comparable pack.\n"
    "\n"
    "A different protein, a different food family, or a different physical "
    "form means they do NOT compete, however similar the wording looks.\n"
    "\n"
    "Rules you must obey:\n"
    "- Choose ONLY from the supplied candidate_id values. Never output an "
    "identifier that is not in the list.\n"
    "- Return ACCEPT_CANDIDATE with exactly one candidate_id when one "
    "candidate clearly competes.\n"
    "- Return REJECT_ALL with a null candidate when none of them compete.\n"
    "- Return UNCERTAIN with a null candidate when the evidence does not let "
    "you decide.\n"
    "- Prefer REJECT_ALL or UNCERTAIN over a guess. A wrong competitor is "
    "worse than a missing one.\n"
    "- Respond with JSON only, matching the required schema exactly."
)


@dataclass(frozen=True)
class AdjudicationCandidate:
    """One Master SKU offered to the adjudicator for a single offer."""

    master_sku: str
    master_name: str
    master_description: str = ""
    fuzzy_score: float | None = None
    adjusted_score: float | None = None
    model_score: float | None = None
    model_rank: int | None = None
    pack_status: str = "UNKNOWN"
    rule_reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.master_sku,
            "master_name": self.master_name,
            "master_description": self.master_description,
            "rapidfuzz_score": self.fuzzy_score,
            "rapidfuzz_adjusted_score": self.adjusted_score,
            # Named so the provider cannot read it as a probability.
            "model_raw_margin_within_this_offer": self.model_score,
            "model_rank_within_this_offer": self.model_rank,
            "pack_compatibility": self.pack_status,
            "rule_reason": self.rule_reason,
        }


@dataclass(frozen=True)
class AdjudicationRequest:
    """Everything the adjudicator is allowed to see about one offer."""

    offer_id: str
    offer_name: str
    competitor_brand: str = ""
    competitor_product: str = ""
    competitor_variant: str = ""
    competitor_pack_size: str = ""
    competitor_retailer: str = ""
    candidates: tuple[AdjudicationCandidate, ...] = ()
    ambiguity_reason: str = ""
    top_score: float | None = None
    runner_up_score: float | None = None
    gap: float | None = None

    @property
    def candidate_ids(self) -> set[str]:
        return {candidate.master_sku for candidate in self.candidates}

    def to_payload(self) -> dict[str, Any]:
        return {
            "competitor_offer": {
                "offer_id": self.offer_id,
                "offer_name": self.offer_name,
                "brand": self.competitor_brand,
                "product": self.competitor_product,
                "variant": self.competitor_variant,
                "pack_size": self.competitor_pack_size,
                "retailer": self.competitor_retailer,
            },
            "why_this_needs_a_decision": {
                "ambiguity_reason": self.ambiguity_reason,
                "best_model_raw_margin": self.top_score,
                "runner_up_model_raw_margin": self.runner_up_score,
                "margin_gap_between_top_two": self.gap,
            },
            "candidates": [
                candidate.to_payload() for candidate in self.candidates
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(), ensure_ascii=False, allow_nan=False, sort_keys=True
        )


@dataclass(frozen=True)
class AdjudicationVerdict:
    """A terminal verdict. ``selected_master`` is set only on acceptance."""

    offer_id: str
    reason: CompetitorDecisionReason
    selected_master: str | None = None
    confidence: float | None = None
    explanation: str = ""
    provider: str = ""
    model_id: str = ""
    prompt_version: str = COMPETITOR_PROMPT_VERSION
    request_hash: str = ""
    raw_response_hash: str | None = None
    validation_errors: tuple[str, ...] = ()
    latency_seconds: float = 0.0
    called_provider: bool = False

    @property
    def accepted(self) -> bool:
        return (
            self.selected_master is not None
            and self.reason is CompetitorDecisionReason.LLM_ACCEPTED
        )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CompetitorAdjudicator:
    """Calls one provider per ambiguous offer and never raises at the caller."""

    provider: Any
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    #: Hard ceiling on how much of the candidate universe reaches a provider.
    max_candidates: int = 5
    calls: int = 0
    accepts: int = 0
    rejects: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    _cache: dict[str, AdjudicationVerdict] = field(default_factory=dict)

    @property
    def provider_name(self) -> str:
        return str(getattr(self.provider, "provider_name", "unknown"))

    @property
    def model_id(self) -> str:
        return str(getattr(self.provider, "model_id", "unknown"))

    def _verdict(
        self,
        request: AdjudicationRequest,
        reason: CompetitorDecisionReason,
        *,
        selected: str | None = None,
        confidence: float | None = None,
        explanation: str = "",
        request_hash: str = "",
        raw_response_hash: str | None = None,
        errors: tuple[str, ...] = (),
        latency: float = 0.0,
        called: bool = True,
    ) -> AdjudicationVerdict:
        return AdjudicationVerdict(
            offer_id=request.offer_id,
            reason=reason,
            selected_master=selected,
            confidence=confidence,
            explanation=explanation,
            provider=self.provider_name,
            model_id=self.model_id,
            request_hash=request_hash,
            raw_response_hash=raw_response_hash,
            validation_errors=errors,
            latency_seconds=latency,
            called_provider=called,
        )

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationVerdict:
        """Return a terminal verdict for one ambiguous offer.

        Never propagates a provider exception. Every abnormal path resolves to
        a rejecting reason so the caller always has a decision to record.
        """
        if not request.candidates:
            return self._verdict(
                request,
                CompetitorDecisionReason.LLM_REJECTED,
                explanation="No candidates were supplied to adjudicate.",
                called=False,
            )

        bounded = AdjudicationRequest(
            offer_id=request.offer_id,
            offer_name=request.offer_name,
            competitor_brand=request.competitor_brand,
            competitor_product=request.competitor_product,
            competitor_variant=request.competitor_variant,
            competitor_pack_size=request.competitor_pack_size,
            competitor_retailer=request.competitor_retailer,
            candidates=request.candidates[: max(1, self.max_candidates)],
            ambiguity_reason=request.ambiguity_reason,
            top_score=request.top_score,
            runner_up_score=request.runner_up_score,
            gap=request.gap,
        )
        structured_request = bounded.to_json()
        request_hash = _digest(structured_request)
        cached = self._cache.get(request_hash)
        if cached is not None:
            return cached

        started = time.perf_counter()
        try:
            raw_response = self.provider.generate(
                structured_request=structured_request,
                system_prompt=SYSTEM_PROMPT,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
        except LLMProviderTimeout as error:
            verdict = self._verdict(
                request,
                CompetitorDecisionReason.LLM_TIMEOUT,
                explanation=str(error),
                request_hash=request_hash,
                latency=time.perf_counter() - started,
            )
            self.calls += 1
            self.failures += 1
            self.total_latency_seconds += verdict.latency_seconds
            self._cache[request_hash] = verdict
            return verdict
        except Exception as error:
            # LLMProviderError lands here, and so does anything a provider
            # raises that we never anticipated. A failure must not take the run
            # down: it rejects the offer, exactly like a refusal would.
            verdict = self._verdict(
                request,
                CompetitorDecisionReason.LLM_PROVIDER_FAILURE,
                explanation=str(error),
                request_hash=request_hash,
                latency=time.perf_counter() - started,
            )
            LOGGER.warning(
                "Competitor adjudication failed for offer %s; rejecting",
                request.offer_id,
                exc_info=True,
            )
            self.calls += 1
            self.failures += 1
            self.total_latency_seconds += verdict.latency_seconds
            self._cache[request_hash] = verdict
            return verdict

        latency = time.perf_counter() - started
        self.calls += 1
        self.total_latency_seconds += latency
        response_hash = _digest(raw_response or "")

        try:
            parsed = parse_llm_response(
                raw_response, supplied_candidate_ids=bounded.candidate_ids
            )
        except LLMResponseValidationError as error:
            errors = tuple(str(item) for item in getattr(error, "errors", ()) or (str(error),))
            # A named-but-unsupplied candidate is the invention case, and is
            # worth its own reason code so it can be counted separately from
            # ordinary schema noise.
            invented = any("SELECTED_CANDIDATE_NOT_SUPPLIED" in item for item in errors)
            verdict = self._verdict(
                request,
                CompetitorDecisionReason.LLM_UNKNOWN_CANDIDATE
                if invented
                else CompetitorDecisionReason.LLM_MALFORMED_RESPONSE,
                request_hash=request_hash,
                raw_response_hash=response_hash,
                errors=errors,
                latency=latency,
            )
            self.failures += 1
            self._cache[request_hash] = verdict
            return verdict

        if parsed.decision is LLMDecision.ACCEPT_CANDIDATE:
            verdict = self._verdict(
                request,
                CompetitorDecisionReason.LLM_ACCEPTED,
                selected=parsed.selected_candidate_id,
                confidence=parsed.confidence,
                explanation=parsed.short_explanation,
                request_hash=request_hash,
                raw_response_hash=response_hash,
                latency=latency,
            )
            self.accepts += 1
        else:
            verdict = self._verdict(
                request,
                CompetitorDecisionReason.LLM_REJECTED
                if parsed.decision is LLMDecision.REJECT_ALL
                else CompetitorDecisionReason.LLM_UNCERTAIN,
                confidence=parsed.confidence,
                explanation=parsed.short_explanation,
                request_hash=request_hash,
                raw_response_hash=response_hash,
                latency=latency,
            )
            self.rejects += 1
        self._cache[request_hash] = verdict
        return verdict


def disabled_verdict(offer_id: str) -> AdjudicationVerdict:
    """The verdict for an ambiguous offer when no adjudicator is configured."""
    return AdjudicationVerdict(
        offer_id=offer_id,
        reason=CompetitorDecisionReason.LLM_DISABLED,
        explanation="LLM adjudication is disabled; ambiguous offers reject.",
        called_provider=False,
    )


def build_request(
    classification: OfferClassification,
    *,
    offer_id: str,
    offer_name: str,
    candidates: tuple[AdjudicationCandidate, ...],
    competitor_brand: str = "",
    competitor_product: str = "",
    competitor_variant: str = "",
    competitor_pack_size: str = "",
    competitor_retailer: str = "",
) -> AdjudicationRequest:
    """Assemble a request from a routed offer, preserving the model's order."""
    order = {sku: index for index, sku in enumerate(classification.ranked)}
    ordered = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.master_sku in order
            ),
            key=lambda candidate: order[candidate.master_sku],
        )
    )
    return AdjudicationRequest(
        offer_id=offer_id,
        offer_name=offer_name,
        competitor_brand=competitor_brand,
        competitor_product=competitor_product,
        competitor_variant=competitor_variant,
        competitor_pack_size=competitor_pack_size,
        competitor_retailer=competitor_retailer,
        candidates=ordered,
        ambiguity_reason=classification.reason.value,
        top_score=classification.top_score,
        runner_up_score=classification.runner_up_score,
        gap=classification.gap,
    )
