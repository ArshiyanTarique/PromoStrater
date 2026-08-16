"""The one place a threshold is chosen and a below-threshold row is routed.

Own-brand and competitor offers run the same engine, so they must not each
carry their own copy of "what counts as good enough". Both ask this module,
and the answer depends on a single global toggle:

    LLM review ON   ->  auto-accept at 0.95, everything below goes to Gemini
    LLM review OFF  ->  auto-accept at 0.85, everything below goes to a human

The two modes are deliberately different numbers. With a reviewer behind it the
cut can afford to be strict, because being sent to Gemini is cheap and every
case still ends in an automatic decision. With no reviewer, the same strictness
would bury a person in work, so the cut relaxes and the residue becomes an
explicit manual queue.

BOTH NUMBERS ARE MODEL-SCORE CUT-OFFS, NOT PROBABILITIES OF CORRECTNESS. 0.95
does not mean "95% accurate" and 0.85 does not mean "85% accurate". They are
operational choices about how much work to automate. Nothing in this repository
has ever measured accuracy at either point, and no user-facing surface may
describe them as confidence or accuracy.

Switching the toggle changes the threshold and the review destination and
nothing else. Candidate generation, the 41 features, the model, and the scores
are identical in both modes - that equivalence is asserted in the tests, and it
is the whole reason the toggle is safe to flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

#: Defaults. Changing either changes how much of the dump is automated, so they
#: live here and nowhere else.
LLM_ON_AUTO_ACCEPT_THRESHOLD = 0.95
LLM_OFF_AUTO_ACCEPT_THRESHOLD = 0.85


class ReviewDestination(str, Enum):
    """Where a below-threshold offer goes."""

    #: Adjudicated automatically. The run needs no human to finish.
    GEMINI = "GEMINI"
    #: Queued for a person. An explicit manual-validation mode, not a failure.
    HUMAN_VALIDATION = "HUMAN_VALIDATION"


class RouteOutcome(str, Enum):
    """What the threshold decided, before any reviewer is consulted."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class RoutingMode:
    """The active threshold and review destination for a whole run."""

    llm_review_enabled: bool
    auto_accept_threshold: float
    review_destination: ReviewDestination

    @classmethod
    def from_toggle(
        cls,
        llm_review_enabled: bool,
        *,
        llm_on_threshold: float = LLM_ON_AUTO_ACCEPT_THRESHOLD,
        llm_off_threshold: float = LLM_OFF_AUTO_ACCEPT_THRESHOLD,
    ) -> "RoutingMode":
        """Derive the whole mode from the single global switch.

        This is the only function that turns the toggle into a number. Anything
        that needs the active threshold asks for it here rather than deciding
        for itself, which is what stops the two modes drifting apart.
        """
        enabled = bool(llm_review_enabled)
        return cls(
            llm_review_enabled=enabled,
            auto_accept_threshold=float(
                llm_on_threshold if enabled else llm_off_threshold
            ),
            review_destination=(
                ReviewDestination.GEMINI
                if enabled
                else ReviewDestination.HUMAN_VALIDATION
            ),
        )

    @classmethod
    def from_config(cls, config: Any) -> "RoutingMode":
        """Build from a loaded config object.

        Reads the existing ``llm_review.enabled`` switch rather than inventing
        a parallel one, so a run cannot have the reviewer configured off and
        the routing think it is on.
        """
        llm_review = getattr(config, "llm_review", None)
        return cls.from_toggle(
            bool(getattr(llm_review, "enabled", False)),
            llm_on_threshold=float(
                getattr(
                    llm_review,
                    "on_auto_accept_threshold",
                    LLM_ON_AUTO_ACCEPT_THRESHOLD,
                )
                or LLM_ON_AUTO_ACCEPT_THRESHOLD
            ),
            llm_off_threshold=float(
                getattr(
                    llm_review,
                    "off_auto_accept_threshold",
                    LLM_OFF_AUTO_ACCEPT_THRESHOLD,
                )
                or LLM_OFF_AUTO_ACCEPT_THRESHOLD
            ),
        )

    def decide(self, score: float | None) -> RouteOutcome:
        """Route one model score.

        The comparison is ``>=``: a score exactly on the threshold is accepted.
        A missing score can never be accepted - there is nothing to judge - so
        it reviews.
        """
        if score is None:
            return RouteOutcome.REVIEW
        try:
            value = float(score)
        except (TypeError, ValueError):
            return RouteOutcome.REVIEW
        if value != value:  # NaN
            return RouteOutcome.REVIEW
        return (
            RouteOutcome.AUTO_ACCEPT
            if value >= self.auto_accept_threshold
            else RouteOutcome.REVIEW
        )

    @property
    def routes_to_llm(self) -> bool:
        """Whether a below-threshold offer should reach a provider at all.

        Callers gate provider construction on this, so an OFF run never builds
        a client, never needs an API key, and never makes a call.
        """
        return self.review_destination is ReviewDestination.GEMINI

    @property
    def threshold_percent(self) -> int:
        """The threshold as a whole number, for display only."""
        return int(round(self.auto_accept_threshold * 100))

    def describe(self) -> str:
        """One line for logs and the dashboard.

        Deliberately says "model score" - never "confidence" or "accuracy".
        """
        state = "ON" if self.llm_review_enabled else "OFF"
        below = (
            "Gemini review"
            if self.routes_to_llm
            else "Human Validation"
        )
        return (
            f"Gemini Review {state} - auto-accept at model score "
            f"{self.auto_accept_threshold:.2f}; below that: {below}"
        )
