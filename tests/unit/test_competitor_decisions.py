"""Frame-level automatic competitor decisions."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sku_mapping.competitors.adjudicator import CompetitorAdjudicator
from sku_mapping.competitors.decisions import (
    DECISION_COLUMNS,
    NOT_DECIDED,
    accepted_only,
    apply_automatic_decisions,
)
from sku_mapping.competitors.policy import CompetitorDecisionReason
from sku_mapping.llm_review.provider import LLMProviderTimeout

from tests.unit.test_competitor_adjudicator import StubProvider, response


def row(
    offer_id: str,
    master_sku: str,
    *,
    status: str = "MATCHED",
    fuzzy: float = 80.0,
    model_score: float | None = None,
) -> dict[str, object]:
    return {
        "competitor_offer_id": offer_id,
        "competitor_offer_name": f"Rival product for {offer_id}",
        "competitor_brand": "RivalBrand",
        "competitor_product": "Samosa",
        "competitor_variant": "",
        "competitor_pack_size": "240g",
        "competitor_retailer": "SomeRetailer",
        "master_sku": master_sku,
        "master_name": f"Al Kabeer {master_sku}",
        "master_description": "",
        "competitor_match_status": status,
        "competitor_match_reason": "SUPPORTED_COMPATIBLE_COMPETITOR",
        "competitor_match_score": fuzzy,
        "competitor_adjusted_score": fuzzy + 3,
        "competitor_lightgbm_score": model_score,
        "competitor_lightgbm_rank": None,
    }


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


class TestTerminality:
    def test_every_row_leaves_accepted_or_rejected(self) -> None:
        decided, _ = apply_automatic_decisions(
            frame(
                row("O1", "A", model_score=5.0),
                row("O1", "B", model_score=1.0),
                row("O2", "C", status="UNRELATED"),
            )
        )
        assert set(decided["competitor_decision"]) <= {"ACCEPTED", "REJECTED"}
        assert NOT_DECIDED not in set(decided["competitor_decision"])

    def test_no_row_is_dropped(self) -> None:
        source = frame(
            row("O1", "A", model_score=5.0),
            row("O1", "B", model_score=1.0),
            row("O2", "C", status="HARD_CONFLICT"),
        )
        decided, _ = apply_automatic_decisions(source)
        assert len(decided) == len(source)

    def test_the_decision_columns_are_added_without_overwriting_evidence(self) -> None:
        source = frame(row("O1", "A", model_score=5.0))
        decided, _ = apply_automatic_decisions(source)
        for column in DECISION_COLUMNS:
            assert column in decided.columns
        assert decided["competitor_match_status"].tolist() == ["MATCHED"]
        assert decided["competitor_lightgbm_score"].tolist() == [5.0]

    def test_an_empty_frame_is_handled(self) -> None:
        decided, stats = apply_automatic_decisions(pd.DataFrame())
        assert decided.empty
        assert stats["offers"] == 0


class TestWithoutAnAdjudicator:
    def test_a_clear_offer_accepts_one_master_sku(self) -> None:
        decided, stats = apply_automatic_decisions(
            frame(row("O1", "A", model_score=5.0), row("O1", "B", model_score=1.0))
        )
        accepted = decided[decided["competitor_decision"] == "ACCEPTED"]
        assert accepted["master_sku"].tolist() == ["A"]
        assert stats["clear_offers"] == 1
        assert stats["accepted_relationships"] == 1

    def test_an_ambiguous_offer_rejects_when_no_adjudicator_is_configured(self) -> None:
        decided, stats = apply_automatic_decisions(
            frame(row("O1", "A", model_score=6.0), row("O1", "B", model_score=6.0))
        )
        assert set(decided["competitor_decision"]) == {"REJECTED"}
        assert set(decided["competitor_decision_reason"]) == {"LLM_DISABLED"}
        assert stats["ambiguous_offers"] == 1
        assert stats["adjudicated_offers"] == 0

    def test_offers_are_decided_independently(self) -> None:
        decided, stats = apply_automatic_decisions(
            frame(
                row("O1", "A", model_score=9.0),
                row("O1", "B", model_score=1.0),
                row("O2", "A", model_score=6.0),
                row("O2", "B", model_score=6.0),
            )
        )
        first = decided[decided["competitor_offer_id"] == "O1"]
        second = decided[decided["competitor_offer_id"] == "O2"]
        assert set(first["competitor_decision"]) == {"ACCEPTED", "REJECTED"}
        assert set(second["competitor_decision"]) == {"REJECTED"}
        assert stats["offers"] == 2


class TestWithAnAdjudicator:
    def test_an_accepted_ambiguous_offer_records_the_llm_as_the_source(self) -> None:
        adjudicator = CompetitorAdjudicator(provider=StubProvider(response(candidate="B")))
        decided, stats = apply_automatic_decisions(
            frame(row("O1", "A", model_score=6.0), row("O1", "B", model_score=6.0)),
            adjudicator=adjudicator,
        )
        accepted = decided[decided["competitor_decision"] == "ACCEPTED"]
        assert accepted["master_sku"].tolist() == ["B"]
        assert accepted["competitor_decision_source"].tolist() == ["llm"]
        assert accepted["competitor_decision_reason"].tolist() == ["LLM_ACCEPTED"]
        assert stats["llm_accepts"] == 1

    def test_a_clear_offer_never_reaches_the_adjudicator(self) -> None:
        provider = StubProvider(response())
        apply_automatic_decisions(
            frame(row("O1", "A", model_score=9.0), row("O1", "B", model_score=1.0)),
            adjudicator=CompetitorAdjudicator(provider=provider),
        )
        assert provider.requests == []

    @pytest.mark.parametrize(
        ("provider", "expected"),
        [
            (StubProvider(response("REJECT_ALL", candidate=None)), "LLM_REJECTED"),
            (StubProvider(response("UNCERTAIN", candidate=None)), "LLM_UNCERTAIN"),
            (StubProvider(error=LLMProviderTimeout("t")), "LLM_TIMEOUT"),
            (StubProvider("garbage"), "LLM_MALFORMED_RESPONSE"),
            (StubProvider(response(candidate="NOT-A-REAL-SKU")), "LLM_UNKNOWN_CANDIDATE"),
        ],
    )
    def test_every_unresolved_llm_path_rejects_the_offer(
        self, provider: StubProvider, expected: str
    ) -> None:
        decided, _ = apply_automatic_decisions(
            frame(row("O1", "A", model_score=6.0), row("O1", "B", model_score=6.0)),
            adjudicator=CompetitorAdjudicator(provider=provider),
        )
        assert set(decided["competitor_decision"]) == {"REJECTED"}
        assert set(decided["competitor_decision_reason"]) == {expected}

    def test_only_admitted_candidates_are_sent(self) -> None:
        provider = StubProvider(response(candidate="A"))
        apply_automatic_decisions(
            frame(
                row("O1", "A", model_score=6.0),
                row("O1", "B", model_score=6.0),
                row("O1", "CONFLICTED", status="HARD_CONFLICT", model_score=99.0),
            ),
            adjudicator=CompetitorAdjudicator(provider=provider),
        )
        sent = {
            candidate["candidate_id"]
            for candidate in json.loads(provider.requests[0])["candidates"]
        }
        assert sent == {"A", "B"}

    def test_the_candidate_universe_sent_is_bounded(self) -> None:
        provider = StubProvider(response(candidate="SKU0"))
        apply_automatic_decisions(
            frame(
                *[
                    row("O1", f"SKU{index}", model_score=6.0)
                    for index in range(30)
                ]
            ),
            adjudicator=CompetitorAdjudicator(provider=provider, max_candidates=4),
            max_adjudicated_candidates=4,
        )
        assert len(json.loads(provider.requests[0])["candidates"]) == 4


class TestExport:
    def test_accepted_only_filters_a_decided_frame(self) -> None:
        decided, _ = apply_automatic_decisions(
            frame(row("O1", "A", model_score=5.0), row("O1", "B", model_score=1.0))
        )
        assert accepted_only(decided)["master_sku"].tolist() == ["A"]

    def test_accepted_only_leaves_an_undecided_frame_intact(self) -> None:
        source = frame(row("O1", "A", model_score=5.0))
        assert len(accepted_only(source)) == 1


class TestDeterminism:
    def test_row_order_does_not_change_the_outcome(self) -> None:
        rows = [row("O1", "A", model_score=6.0), row("O1", "B", model_score=6.0)]
        forward, _ = apply_automatic_decisions(
            frame(*rows), adjudicator=CompetitorAdjudicator(provider=StubProvider(response(candidate="A")))
        )
        backward, _ = apply_automatic_decisions(
            frame(*reversed(rows)),
            adjudicator=CompetitorAdjudicator(provider=StubProvider(response(candidate="A"))),
        )
        assert set(
            forward[forward["competitor_decision"] == "ACCEPTED"]["master_sku"]
        ) == set(
            backward[backward["competitor_decision"] == "ACCEPTED"]["master_sku"]
        )


class TestOnePerOfferInvariant:
    """A competitor offer competes with at most one Al Kabeer SKU."""

    def test_at_most_one_distinct_master_sku_is_accepted_per_offer(self) -> None:
        decided, _ = apply_automatic_decisions(
            frame(
                row("O1", "A", model_score=9.0),
                row("O1", "B", model_score=1.0),
                row("O1", "C", model_score=0.5),
                row("O2", "A", model_score=8.0),
                row("O2", "D", model_score=0.0),
            )
        )
        accepted = decided[decided["competitor_decision"] == "ACCEPTED"]
        per_offer = accepted.groupby("competitor_offer_id")["master_sku"].nunique()
        assert (per_offer <= 1).all()

    def test_a_repeated_relationship_row_shares_one_decision(self) -> None:
        """The same pair seen under two source offers is one relationship."""
        decided, _ = apply_automatic_decisions(
            frame(
                row("O1", "A", model_score=9.0),
                row("O1", "A", model_score=9.0),
                row("O1", "B", model_score=1.0),
            )
        )
        accepted = decided[decided["competitor_decision"] == "ACCEPTED"]
        assert accepted["master_sku"].nunique() == 1
        assert set(accepted["master_sku"]) == {"A"}
        assert len(accepted) == 2
