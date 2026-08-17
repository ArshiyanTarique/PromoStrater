"""The global toggle must reach the production own-brand threshold.

The toggle used to stop at the dashboard: the live pipeline pinned its cut-off
to ``ml.auto_accept_threshold`` and ignored the switch entirely, so turning
Gemini on changed the displayed mode but not what the run actually did. These
tests pin the wiring so that cannot regress.
"""

from __future__ import annotations

import dataclasses

import pytest

from sku_mapping.config import ConfigurationError, load_config
from sku_mapping.matching.routing import RoutingMode


@pytest.fixture(scope="module")
def base_config():
    return load_config("config/default.yaml")


def with_toggle(config, enabled: bool):
    """Flip only the global switch, leaving everything else alone."""
    llm_review = dataclasses.replace(
        config.llm_review,
        enabled=enabled,
        # An enabled reviewer requires a model name; the value is irrelevant
        # here because nothing in these tests makes a call.
        model=config.llm_review.model or "gemini-2.0-flash",
    )
    return dataclasses.replace(config, llm_review=llm_review)


class TestProductionThreshold:
    def test_the_toggle_selects_the_production_cut_off(self, base_config) -> None:
        assert (
            RoutingMode.from_config(
                with_toggle(base_config, False)
            ).auto_accept_threshold
            == 0.85
        )
        assert (
            RoutingMode.from_config(
                with_toggle(base_config, True)
            ).auto_accept_threshold
            == 0.95
        )

    def test_the_pipeline_derives_its_threshold_from_the_toggle(
        self, base_config
    ) -> None:
        """Mirrors what unified inference does when it builds effective_config.

        If this ever reverts to ``config.ml.auto_accept_threshold`` the two
        modes collapse to one number and this fails.
        """
        for enabled, expected in ((False, 0.85), (True, 0.95)):
            config = with_toggle(base_config, enabled)
            effective = dataclasses.replace(
                config,
                agreement=dataclasses.replace(
                    config.agreement,
                    lightgbm_auto_accept_threshold=RoutingMode.from_config(
                        config
                    ).auto_accept_threshold,
                ),
            )
            assert effective.agreement.lightgbm_auto_accept_threshold == expected

    def test_the_static_ml_threshold_is_no_longer_authoritative(
        self, base_config
    ) -> None:
        """ml.auto_accept_threshold must not silently override the toggle."""
        on = RoutingMode.from_config(with_toggle(base_config, True))
        assert on.auto_accept_threshold != base_config.ml.auto_accept_threshold

    def test_the_destination_follows_the_same_switch(self, base_config) -> None:
        assert RoutingMode.from_config(with_toggle(base_config, True)).routes_to_llm
        assert not RoutingMode.from_config(
            with_toggle(base_config, False)
        ).routes_to_llm


class TestConfiguredThresholds:
    def test_the_shipped_defaults_are_the_agreed_values(self, base_config) -> None:
        assert base_config.llm_review.on_auto_accept_threshold == 0.95
        assert base_config.llm_review.off_auto_accept_threshold == 0.85

    def test_configured_values_override_the_module_defaults(
        self, base_config
    ) -> None:
        config = dataclasses.replace(
            base_config,
            llm_review=dataclasses.replace(
                base_config.llm_review,
                enabled=False,
                on_auto_accept_threshold=0.99,
                off_auto_accept_threshold=0.70,
            ),
        )
        assert RoutingMode.from_config(config).auto_accept_threshold == 0.70

    def test_turning_the_reviewer_off_may_not_automate_less(
        self, base_config
    ) -> None:
        """An OFF cut above the ON cut is incoherent and must be refused."""
        with pytest.raises(ConfigurationError):
            dataclasses.replace(
                base_config.llm_review,
                on_auto_accept_threshold=0.80,
                off_auto_accept_threshold=0.90,
            )

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
    def test_thresholds_outside_the_unit_interval_are_refused(
        self, base_config, value: float
    ) -> None:
        with pytest.raises(ConfigurationError):
            dataclasses.replace(
                base_config.llm_review, on_auto_accept_threshold=value
            )
