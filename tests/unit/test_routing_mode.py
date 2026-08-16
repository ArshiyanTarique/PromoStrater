"""The global LLM toggle and the thresholds it selects.

These pin the exact boundary values. A change to 0.95 or 0.85 has to be made
here, deliberately, before it can reach a run.
"""

from __future__ import annotations

import math

import pytest

from sku_mapping.matching.routing import (
    LLM_OFF_AUTO_ACCEPT_THRESHOLD,
    LLM_ON_AUTO_ACCEPT_THRESHOLD,
    ReviewDestination,
    RouteOutcome,
    RoutingMode,
)

ON = RoutingMode.from_toggle(True)
OFF = RoutingMode.from_toggle(False)


class TestDefaults:
    def test_the_two_modes_use_the_specified_thresholds(self) -> None:
        assert LLM_ON_AUTO_ACCEPT_THRESHOLD == 0.95
        assert LLM_OFF_AUTO_ACCEPT_THRESHOLD == 0.85
        assert ON.auto_accept_threshold == 0.95
        assert OFF.auto_accept_threshold == 0.85

    def test_the_destination_follows_the_toggle(self) -> None:
        assert ON.review_destination is ReviewDestination.GEMINI
        assert OFF.review_destination is ReviewDestination.HUMAN_VALIDATION
        assert ON.routes_to_llm is True
        assert OFF.routes_to_llm is False


class TestRoutingBoundaries:
    @pytest.mark.parametrize(
        ("mode", "score", "expected"),
        [
            # LLM ON - 0.95 cut, inclusive.
            (ON, 0.96, RouteOutcome.AUTO_ACCEPT),
            (ON, 0.95, RouteOutcome.AUTO_ACCEPT),
            (ON, 0.9499, RouteOutcome.REVIEW),
            (ON, 0.85, RouteOutcome.REVIEW),
            # LLM OFF - 0.85 cut, inclusive.
            (OFF, 0.96, RouteOutcome.AUTO_ACCEPT),
            (OFF, 0.85, RouteOutcome.AUTO_ACCEPT),
            (OFF, 0.8499, RouteOutcome.REVIEW),
            (OFF, 0.0, RouteOutcome.REVIEW),
        ],
    )
    def test_scores_route_as_specified(
        self, mode: RoutingMode, score: float, expected: RouteOutcome
    ) -> None:
        assert mode.decide(score) is expected

    def test_the_band_between_the_two_thresholds_is_the_whole_difference(
        self,
    ) -> None:
        """0.90 automates only when the reviewer is off. That is the trade."""
        assert OFF.decide(0.90) is RouteOutcome.AUTO_ACCEPT
        assert ON.decide(0.90) is RouteOutcome.REVIEW

    @pytest.mark.parametrize("score", [None, float("nan"), "not a number"])
    def test_an_unusable_score_can_never_be_auto_accepted(
        self, score: object
    ) -> None:
        """A missing or broken score is reviewed, never waved through."""
        assert ON.decide(score) is RouteOutcome.REVIEW  # type: ignore[arg-type]
        assert OFF.decide(score) is RouteOutcome.REVIEW  # type: ignore[arg-type]


class TestConfigDerivation:
    class _LLM:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled

    class _Config:
        def __init__(self, enabled: bool) -> None:
            self.llm_review = TestConfigDerivation._LLM(enabled)

    def test_the_mode_comes_from_the_existing_llm_review_switch(self) -> None:
        assert RoutingMode.from_config(self._Config(True)).auto_accept_threshold == 0.95
        assert RoutingMode.from_config(self._Config(False)).auto_accept_threshold == 0.85

    def test_a_config_without_the_section_falls_back_to_off(self) -> None:
        """Absent configuration must not silently enable a paid provider."""
        mode = RoutingMode.from_config(object())
        assert mode.llm_review_enabled is False
        assert mode.routes_to_llm is False

    def test_thresholds_are_overridable_without_touching_the_defaults(self) -> None:
        custom = RoutingMode.from_toggle(
            True, llm_on_threshold=0.99, llm_off_threshold=0.5
        )
        assert custom.auto_accept_threshold == 0.99
        assert LLM_ON_AUTO_ACCEPT_THRESHOLD == 0.95


class TestDisplay:
    def test_the_description_never_claims_accuracy_or_confidence(self) -> None:
        for mode in (ON, OFF):
            text = mode.describe().lower()
            assert "model score" in text
            assert "accuracy" not in text
            assert "confidence" not in text

    def test_the_description_names_the_destination(self) -> None:
        assert "Gemini review" in ON.describe()
        assert "Human Validation" in OFF.describe()

    def test_the_percent_form_is_whole_and_matches_the_threshold(self) -> None:
        assert ON.threshold_percent == 95
        assert OFF.threshold_percent == 85
        for mode in (ON, OFF):
            assert math.isclose(
                mode.threshold_percent / 100, mode.auto_accept_threshold
            )
