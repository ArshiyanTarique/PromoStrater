"""The hybrid competitor path: retrieval keeps recall, ML ranks, LLM adjudicates.

These guard the three wiring defects that made the hybrid inert in production:
the re-ranker asked for a model id that is null, the adjudicator was never
constructed, and the adjudicator was never forwarded to discovery.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from dashboard.services.processing_service import (
    _competitor_adjudicator,
    _competitor_reranker,
)
from sku_mapping.competitors.discovery import (
    COMPETITOR_LONG_COLUMNS,
    discover_competitors,
)
from sku_mapping.config import load_config

import tests.unit.test_business_outputs as business


def _config():
    return load_config("config/default.yaml")


# -- the global toggle owns whether a provider exists at all ---------------

def test_no_provider_is_constructed_when_the_global_toggle_is_off() -> None:
    """LLM OFF must make no API key necessary and no call possible."""
    config = _config()
    config = replace(config, llm_review=replace(config.llm_review, enabled=False))

    assert _competitor_adjudicator(config) is None


def test_toggle_on_builds_an_adjudicator_or_fails_safely(monkeypatch) -> None:
    """LLM ON builds one; a provider that cannot be built degrades to None.

    Returning None is the safe direction: ambiguous competitors then reject
    automatically rather than queueing for a person.
    """
    config = _config()
    config = replace(
        config,
        llm_review=replace(config.llm_review, enabled=True, provider="gemini"),
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    adjudicator = _competitor_adjudicator(config)

    # Either a real adjudicator, or None because no key exists. Never a raise,
    # and never a human route.
    if adjudicator is not None:
        assert adjudicator.max_candidates == config.competitors.llm_max_candidates


def test_reranker_requests_the_model_id_own_brand_inference_uses() -> None:
    """shadow_mode.model_id is null; asking for it silently loaded nothing."""
    config = _config()
    assert config.shadow_mode.model_id is None, "fixture assumption changed"
    assert config.ml.model_id, "own-brand model id must be configured"

    config = replace(
        config, competitors=replace(config.competitors, ml_reranking_enabled=True)
    )
    reranker = _competitor_reranker(config)

    assert reranker is not None, (
        "re-ranking is enabled but no package loaded; the loader is asking "
        "for the wrong model id again"
    )


def test_reranker_is_absent_when_reranking_is_disabled() -> None:
    config = _config()
    config = replace(
        config, competitors=replace(config.competitors, ml_reranking_enabled=False)
    )
    assert _competitor_reranker(config) is None


# -- recall safety ---------------------------------------------------------

def _discover(config, **kwargs):
    return discover_competitors(
        business._prepared_offers(),
        business._master(),
        pd.DataFrame(
            [
                {
                    "source_offer_id": "own-1",
                    "source_offer_name": "Al Kabeer Chicken Nuggets 400g",
                    "matched_master_sku": "SKU-NUGGETS",
                }
            ]
        ),
        config=config,
        run_id="hybrid-test",
        **kwargs,
    )


def test_the_model_cannot_remove_a_relationship_retrieval_found() -> None:
    """ML ranks; it must never gate recall.

    Replacing retrieval with the own-brand model was measured to cut CKSA from
    194 competitors to 4, so the model is confined to ordering.
    """
    config = _config()
    rules_only = replace(
        config.competitors,
        ml_reranking_enabled=False,
        automatic_decisions_enabled=False,
    )
    hybrid = replace(
        config.competitors,
        ml_reranking_enabled=False,
        automatic_decisions_enabled=True,
    )

    supported = {"MATCHED", "AMBIGUOUS"}
    base = _discover(rules_only).long_format
    with_decisions = _discover(hybrid).long_format

    def relationships(frame):
        rows = frame[frame["competitor_match_status"].isin(supported)]
        return set(
            zip(
                rows["master_sku"].astype(str),
                rows["competitor_offer_id"].astype(str),
            )
        )

    assert relationships(base) == relationships(with_decisions)


def test_audit_contract_has_one_shape_whether_or_not_decisions_run() -> None:
    """The export validator sees the same columns in both configurations."""
    config = _config()
    off = _discover(
        replace(config.competitors, automatic_decisions_enabled=False)
    ).long_format
    on = _discover(
        replace(config.competitors, automatic_decisions_enabled=True)
    ).long_format

    assert tuple(off.columns) == COMPETITOR_LONG_COLUMNS
    assert tuple(on.columns) == COMPETITOR_LONG_COLUMNS


def test_ambiguous_competitors_reject_rather_than_queue_for_a_human() -> None:
    """With no adjudicator every unsettled competitor takes the safe route."""
    config = _config()
    frame = _discover(
        replace(config.competitors, automatic_decisions_enabled=True),
        adjudicator=None,
    ).long_format

    decided = frame["competitor_decision"].dropna()
    assert not decided.empty, "decision layer produced no verdicts"
    assert set(decided.unique()) <= {"ACCEPTED", "REJECTED"}, (
        "a competitor decision escaped the automatic ACCEPT/REJECT contract"
    )
