"""All Phase 6C agreement-status and routing branches."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sku_mapping.agreement.policy import evaluate_candidate_agreement
from sku_mapping.config import load_config
from sku_mapping.constants import AgreementStatus, ReviewRoute


def _config():
    return load_config("config/default.yaml").agreement


def _candidates(
    *,
    probability: float = 0.90,
    same_top: bool = True,
    hard_conflict: bool = False,
    embedding_unavailable: bool = False,
    model_unavailable: bool = False,
    missing_master: bool = False,
) -> pd.DataFrame:
    first_code = "NO_MATCH" if missing_master else "SKU-A"
    first_similarity, second_similarity = (
        (0.80, 0.50) if same_top else (0.50, 0.80)
    )
    failure = "backend unavailable" if embedding_unavailable else ""
    return pd.DataFrame(
        [
            {
                "offer_group_id": "offer-1",
                "master_itemcode": first_code,
                "candidate_rank": 1,
                "candidate_margin": 9.0,
                "candidate_raw_margin": 7.0,
                "calibrated_probability": (
                    np.nan if model_unavailable else probability
                ),
                "embedding_similarity": (
                    np.nan
                    if embedding_unavailable
                    else first_similarity
                ),
                "embedding_failure_reason": failure,
                "protein_match": 0 if hard_conflict else 1,
                "family_match": 1,
                "pack_format_match": 1,
                "candidate_pack_status": True,
                "candidate_pack_structure_status": True,
                "is_mixed_protein_offer": 0,
            },
            {
                "offer_group_id": "offer-1",
                "master_itemcode": "SKU-B",
                "candidate_rank": 2,
                "candidate_margin": 100.0,
                "candidate_raw_margin": 100.0,
                "calibrated_probability": (
                    np.nan if model_unavailable else 0.30
                ),
                "embedding_similarity": (
                    np.nan
                    if embedding_unavailable
                    else second_similarity
                ),
                "embedding_failure_reason": failure,
                "protein_match": 1,
                "family_match": 1,
                "pack_format_match": 1,
                "candidate_pack_status": True,
                "candidate_pack_structure_status": True,
                "is_mixed_protein_offer": 0,
            },
        ]
    )


def _one(frame: pd.DataFrame, config=None):
    return evaluate_candidate_agreement(
        frame, config=config or _config()
    ).results[0]


def test_safe_agreement_routes_to_review_when_auto_influence_unapproved() -> None:
    result = _one(_candidates(probability=0.96))
    assert result.agreement_status is AgreementStatus.SAFE_AGREEMENT
    assert result.routing_decision is ReviewRoute.MANUAL_REVIEW
    assert "EMBEDDING_AUTO_INFLUENCE_DISABLED" in result.routing_reason
    assert result.same_top_candidate is True
    assert result.lightgbm_top_candidate == "SKU-A"
    assert result.embedding_top_candidate == "SKU-A"
    assert result.lightgbm_top_rank == 1
    assert result.embedding_rank == 1
    assert result.lightgbm_score_margin == pytest.approx(0.66)
    assert result.embedding_score_margin == pytest.approx(0.30)
    assert result.candidate_generation_margin == 9.0


def test_explicitly_approved_embedding_can_route_safe_agreement_to_auto() -> None:
    result = _one(
        _candidates(probability=0.96),
        replace(_config(), allow_embedding_auto_accept=True),
    )
    assert result.agreement_status is AgreementStatus.SAFE_AGREEMENT
    assert result.routing_decision is ReviewRoute.AUTO_ACCEPT


def test_exact_commercial_candidate_precedes_higher_scoring_adapted() -> None:
    frame = _candidates(probability=0.99)
    frame.loc[0, "commercial_outcome"] = "ADAPTED_MATCH"
    frame.loc[1, "commercial_outcome"] = "EXACT_MATCH"
    frame.loc[1, "calibrated_probability"] = 0.80
    frame.loc[0, "embedding_similarity"] = 0.99
    frame.loc[1, "embedding_similarity"] = 0.50
    result = _one(frame)
    assert result.lightgbm_top_candidate == "SKU-B"
    assert result.embedding_top_candidate == "SKU-B"
    assert result.routing_decision is ReviewRoute.LLM_REVIEW


def test_different_top_candidates_route_to_llm_review() -> None:
    result = _one(_candidates(same_top=False))
    assert result.agreement_status is AgreementStatus.DISAGREEMENT
    assert result.routing_decision is ReviewRoute.LLM_REVIEW
    assert result.same_top_candidate is False
    assert "DIFFERENT_TOP_CANDIDATE" in result.routing_reason


def test_same_top_below_lightgbm_threshold_routes_to_llm_review() -> None:
    result = _one(_candidates(probability=0.849999))
    assert result.agreement_status is AgreementStatus.WEAK_AGREEMENT
    assert result.routing_decision is ReviewRoute.LLM_REVIEW
    assert "LIGHTGBM_BELOW_THRESHOLD" in result.routing_reason


def test_hard_conflict_routes_to_manual_review_even_with_high_scores() -> None:
    result = _one(
        _candidates(probability=0.99, hard_conflict=True)
    )
    assert result.agreement_status is AgreementStatus.WEAK_AGREEMENT
    assert result.routing_decision is ReviewRoute.MANUAL_REVIEW
    assert result.conflict_flags["protein_conflict"] is True
    assert "HARD_CONFLICT" in result.routing_reason


def test_ml_only_is_never_reported_as_agreement() -> None:
    """One scorer must never be dressed up as two.

    The original protection this test carried - a missing second scorer may
    not become false corroboration - still holds. What changed is that the
    absence of a second scorer no longer blocks a decision.
    """
    result = _one(_candidates(embedding_unavailable=True))
    assert result.agreement_status is AgreementStatus.LIGHTGBM_ONLY
    assert result.agreement_status not in {
        AgreementStatus.SAFE_AGREEMENT,
        AgreementStatus.WEAK_AGREEMENT,
    }
    assert result.same_top_candidate is False
    assert result.embedding_top_candidate is None
    assert result.embedding_similarity is None


def test_ml_only_high_confidence_auto_accepts() -> None:
    """Above the configured bar the model decides on its own."""
    result = _one(_candidates(embedding_unavailable=True, probability=0.99))
    assert result.agreement_status is AgreementStatus.LIGHTGBM_ONLY
    assert result.routing_decision is ReviewRoute.AUTO_ACCEPT
    assert "LIGHTGBM_BELOW_THRESHOLD" not in result.routing_reason


def test_ml_only_low_confidence_routes_to_review() -> None:
    """Below the bar the candidate is escalated, never silently accepted."""
    result = _one(_candidates(embedding_unavailable=True, probability=0.10))
    assert result.agreement_status is AgreementStatus.LIGHTGBM_ONLY
    assert result.routing_decision is not ReviewRoute.AUTO_ACCEPT
    assert "LIGHTGBM_BELOW_THRESHOLD" in result.routing_reason


def test_ml_only_hard_conflict_still_overrides_a_confident_model() -> None:
    """Confidence must not buy its way past a semantic conflict."""
    result = _one(
        _candidates(
            embedding_unavailable=True, hard_conflict=True, probability=0.99
        )
    )
    assert result.routing_decision is ReviewRoute.MANUAL_REVIEW
    assert result.routing_decision is not ReviewRoute.AUTO_ACCEPT
    assert "HARD_CONFLICT" in result.routing_reason


def test_ml_only_missing_master_never_auto_accepts() -> None:
    """A candidate that is not in the master cannot be a mapping."""
    result = _one(
        _candidates(
            embedding_unavailable=True, missing_master=True, probability=0.99
        )
    )
    assert result.routing_decision is not ReviewRoute.AUTO_ACCEPT
    assert "MASTER_SKU_MISSING" in result.routing_reason


def test_lightgbm_unavailable_routes_to_safe_fallback() -> None:
    result = _one(_candidates(model_unavailable=True))
    assert result.agreement_status is AgreementStatus.MODEL_UNAVAILABLE
    assert result.routing_decision is ReviewRoute.SAFE_FALLBACK
    assert result.lightgbm_top_candidate is None
    assert "LIGHTGBM_UNAVAILABLE" in result.routing_reason


def test_missing_master_routes_to_manual_review() -> None:
    result = _one(_candidates(missing_master=True))
    assert result.agreement_status is AgreementStatus.WEAK_AGREEMENT
    assert result.routing_decision is ReviewRoute.MANUAL_REVIEW
    assert result.conflict_flags["missing_master"] is True
    assert "MASTER_SKU_MISSING" in result.routing_reason


def test_optional_embedding_margin_is_not_invented_but_is_enforced_if_set() -> None:
    config = replace(_config(), minimum_embedding_margin=0.40)
    result = _one(_candidates(), config=config)
    assert result.agreement_status is AgreementStatus.WEAK_AGREEMENT
    assert result.routing_decision is ReviewRoute.LLM_REVIEW
    assert "WEAK_EMBEDDING_MARGIN" in result.routing_reason


def test_multiple_offers_produce_one_explicit_result_each() -> None:
    first = _candidates()
    second = _candidates(same_top=False).assign(
        offer_group_id="offer-2"
    )
    candidates = pd.concat([first, second], ignore_index=True)
    before = candidates.copy(deep=True)
    evaluation = evaluate_candidate_agreement(
        candidates, config=_config()
    )
    assert len(evaluation.results) == 2
    assert evaluation.frame["offer_id"].tolist() == [
        "offer-1",
        "offer-2",
    ]
    required = {
        "offer_id",
        "lightgbm_top_candidate",
        "lightgbm_calibrated_probability",
        "lightgbm_top_rank",
        "embedding_top_candidate",
        "embedding_similarity",
        "embedding_rank",
        "same_top_candidate",
        "lightgbm_score_margin",
        "embedding_score_margin",
        "conflict_flags",
        "agreement_status",
        "routing_decision",
        "routing_reason",
    }
    assert required.issubset(evaluation.frame.columns)
    pd.testing.assert_frame_equal(candidates, before, check_exact=True)
