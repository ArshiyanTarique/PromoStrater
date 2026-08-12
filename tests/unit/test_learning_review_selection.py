"""Deterministic bounded review selection and fallback behavior."""

from __future__ import annotations

from sku_mapping.learning.review_selection import select_five_reviews


def _row(
    offer: str,
    *,
    probability: float,
    decision: str = "MANUAL_REVIEW",
    agreement: str = "WEAK_AGREEMENT",
    llm: str = "",
    conflicts: str = "[]",
) -> dict[str, object]:
    return {
        "prediction_id": f"p-{offer}",
        "offer_id": offer,
        "candidate_id": f"sku-{offer}",
        "candidate_rank": 1,
        "lightgbm_probability": probability,
        "embedding_similarity": 0.5,
        "agreement_status": agreement,
        "llm_decision": llm,
        "final_decision": decision,
        "decision_source": (
            "STRUCTURED_LLM_REVIEW" if llm else "AGREEMENT_POLICY"
        ),
        "conflict_flags_json": conflicts,
    }


def test_targeted_selection_is_deterministic_and_unique() -> None:
    rows = [
        _row("auto", probability=0.97, decision="AUTO_ACCEPT"),
        _row("near", probability=0.849),
        _row(
            "disagree",
            probability=0.72,
            agreement="DISAGREEMENT",
        ),
        _row("llm", probability=0.71, llm="ACCEPT_CANDIDATE"),
        _row("hard", probability=0.91, conflicts='["protein_conflict"]'),
        _row("spare", probability=0.42),
    ]
    first = select_five_reviews(rows)
    second = select_five_reviews(reversed(rows))
    assert first == second
    assert len({item.prediction["offer_id"] for item in first}) == 5
    assert [item.category for item in first] == [
        "HIGH_CONFIDENCE_AUTO_ACCEPT",
        "NEAR_AUTO_ACCEPT_THRESHOLD",
        "LIGHTGBM_EMBEDDING_DISAGREEMENT",
        "LLM_REVIEWED",
        "DIFFICULT_OR_CONFLICT_PRONE",
    ]


def test_missing_categories_use_audited_fallbacks() -> None:
    rows = [
        _row(f"offer-{number}", probability=0.1 * number)
        for number in range(1, 7)
    ]
    selected = select_five_reviews(rows)
    assert len(selected) == 5
    assert len({item.prediction["offer_id"] for item in selected}) == 5
    assert any(item.fallback_reason for item in selected)
    assert all(
        "NO_ELIGIBLE_" in item.fallback_reason
        for item in selected
        if item.fallback_reason
    )


def test_fewer_than_five_eligible_offers_remain_reviewable() -> None:
    rows = [_row(str(number), probability=0.5) for number in range(4)]
    selected = select_five_reviews(rows)
    assert len(selected) == 4
    assert {item.prediction["offer_id"] for item in selected} == {
        "0",
        "1",
        "2",
        "3",
    }
