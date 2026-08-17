"""Full final-routing matrix for the unified assisted inference policy."""

from __future__ import annotations

import pandas as pd

from sku_mapping.constants import FinalMatchDecision
from sku_mapping.inference.pipeline import (
    finalize_unified_decisions,
    select_competitor_eligible_rows,
)


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "offer_group_id": [
                "safe",
                "llm",
                "uncertain",
                "hard",
                "none",
                "model",
                "llm-invalid",
                "master-missing",
            ],
            "Offer Name": [
                "Safe offer",
                "LLM offer",
                "Uncertain offer",
                "Conflict offer",
                "No candidate offer",
                "Model failure offer",
                "Invalid LLM offer",
                "Missing master offer",
            ],
        }
    )


def _candidate(
    offer_id: str,
    sku: str,
    *,
    probability: float,
    similarity: float | None,
    candidate_rank: int = 1,
    agreement_status: str = "SAFE_AGREEMENT",
    agreement_route: str = "AUTO_ACCEPT",
    same_top: bool = True,
    llm_status: str = "",
    llm_route: str = "",
    llm_decision: str = "",
    llm_candidate: str = "",
    llm_confidence: float | None = None,
    hard_conflict: bool = False,
) -> dict:
    return {
        "offer_group_id": offer_id,
        "candidate_rank": candidate_rank,
        "master_itemcode": sku,
        "master_item_description": f"Description {sku}",
        "calibrated_probability": probability,
        "agreement_status": agreement_status,
        "routing_decision": agreement_route,
        "same_top_candidate": same_top,
        "lightgbm_top_candidate": sku if candidate_rank == 1 else "",
        "routing_reason": agreement_status,
        "llm_review_status": llm_status,
        "llm_final_route": llm_route,
        "llm_parsed_decision": llm_decision,
        "llm_selected_candidate": llm_candidate,
        "llm_confidence": llm_confidence,
        "llm_model_id": "fake:reviewer-v1" if llm_status else "",
        "llm_routing_reason": llm_status,
        "protein_conflict": hard_conflict,
        "mixed_protein_ambiguity": False,
        "strong_family_conflict": False,
        "strong_size_weight_conflict": False,
        "strong_pack_format_conflict": False,
        "feature_generation_failure": False,
        "missing_master": False,
    }


def _candidates() -> pd.DataFrame:
    rows = [
        _candidate(
            "safe",
            "SKU-SAFE",
            probability=0.91,
            similarity=0.72,
        ),
        _candidate(
            "llm",
            "SKU-LGB",
            probability=0.79,
            similarity=0.50,
            agreement_status="DISAGREEMENT",
            agreement_route="LLM_REVIEW",
            same_top=False,
            llm_status="COMPLETED",
            llm_route="LLM_ACCEPT",
            llm_decision="ACCEPT_CANDIDATE",
            llm_candidate="SKU-LLM",
            llm_confidence=0.94,
        ),
        _candidate(
            "llm",
            "SKU-LLM",
            probability=0.75,
            similarity=0.71,
            candidate_rank=2,
            agreement_status="DISAGREEMENT",
            agreement_route="LLM_REVIEW",
            same_top=False,
            llm_status="COMPLETED",
            llm_route="LLM_ACCEPT",
            llm_decision="ACCEPT_CANDIDATE",
            llm_candidate="SKU-LLM",
            llm_confidence=0.94,
        ),
        _candidate(
            "uncertain",
            "SKU-U",
            probability=0.80,
            similarity=0.60,
            agreement_status="WEAK_AGREEMENT",
            agreement_route="LLM_REVIEW",
            llm_status="COMPLETED",
            llm_route="MANUAL_REVIEW",
            llm_decision="UNCERTAIN",
            llm_confidence=0.61,
        ),
        _candidate(
            "hard",
            "SKU-H",
            probability=0.98,
            similarity=0.90,
            agreement_status="WEAK_AGREEMENT",
            agreement_route="MANUAL_REVIEW",
            hard_conflict=True,
        ),
        _candidate(
            "model",
            "SKU-M",
            probability=0.50,
            similarity=0.50,
            agreement_status="MODEL_UNAVAILABLE",
            agreement_route="SAFE_FALLBACK",
        ),
        _candidate(
            "llm-invalid",
            "SKU-I",
            probability=0.81,
            similarity=0.61,
            agreement_status="DISAGREEMENT",
            agreement_route="LLM_REVIEW",
            llm_status="INVALID_RESPONSE",
            llm_route="MANUAL_REVIEW",
        ),
        {
            **_candidate(
                "master-missing",
                "",
                probability=0.90,
                similarity=0.70,
                agreement_status="WEAK_AGREEMENT",
                agreement_route="MANUAL_REVIEW",
            ),
            "missing_master": True,
        },
    ]
    return pd.DataFrame(rows)


def test_full_final_routing_matrix_and_provenance() -> None:
    assert {decision.value for decision in FinalMatchDecision} == {
        "AUTO_ACCEPT",
        "LLM_ACCEPT",
        "MANUAL_REVIEW",
        "NO_MATCH",
        "NO_CANDIDATE",
        "MASTER_SKU_NOT_FOUND",
        "MODEL_ERROR",
        "COMPETITOR_OFFER",
    }
    decisions = finalize_unified_decisions(
        _offers(),
        _candidates(),
        run_id="unified-test",
        model_id="model-v3",
    ).set_index("offer_id")

    assert decisions.loc["safe", "final_decision"] == (
        FinalMatchDecision.AUTO_ACCEPT.value
    )
    assert decisions.loc["safe", "matched_master_sku"] == "SKU-SAFE"
    assert decisions.loc["llm", "final_decision"] == (
        FinalMatchDecision.LLM_ACCEPT.value
    )
    assert decisions.loc["llm", "matched_master_sku"] == "SKU-LLM"
    assert decisions.loc["uncertain", "final_decision"] == (
        FinalMatchDecision.MANUAL_REVIEW.value
    )
    assert decisions.loc["uncertain", "proposed_master_sku"] == "SKU-U"
    assert (
        decisions.loc["uncertain", "proposed_master_description"]
        == "Description SKU-U"
    )
    assert decisions.loc["uncertain", "proposed_candidate_rank"] == 1
    assert decisions.loc["uncertain", "matched_master_sku"] == ""
    assert decisions.loc["hard", "final_decision"] == (
        FinalMatchDecision.MANUAL_REVIEW.value
    )
    assert decisions.loc["hard", "hard_conflict"]
    assert decisions.loc["none", "final_decision"] == (
        FinalMatchDecision.NO_CANDIDATE.value
    )
    assert decisions.loc["model", "final_decision"] == (
        FinalMatchDecision.MODEL_ERROR.value
    )
    assert decisions.loc["llm-invalid", "final_decision"] == (
        FinalMatchDecision.MANUAL_REVIEW.value
    )
    assert decisions.loc["master-missing", "final_decision"] == (
        FinalMatchDecision.MASTER_SKU_NOT_FOUND.value
    )

    required = {
        "offer_description",
        "proposed_master_sku",
        "proposed_master_description",
        "proposed_candidate_rank",
        "matched_master_sku",
        "matched_master_description",
        "final_decision",
        "decision_source",
        "lightgbm_probability",
        "llm_decision",
        "llm_confidence",
        "human_review_status",
        "model_id",
        "llm_model_id",
        "run_id",
    }
    assert required.issubset(decisions.columns)
    assert decisions["run_id"].eq("unified-test").all()


def test_adapted_commercial_outcome_cannot_auto_accept() -> None:
    offers = pd.DataFrame(
        {"offer_group_id": ["adapted"], "Offer Name": ["Bundle offer"]}
    )
    candidate = _candidate(
        "adapted", "SKU-A", probability=0.99, similarity=0.99
    )
    candidate.update(
        {
            "commercial_outcome": "ADAPTED_MATCH",
            "commercial_severity": "SEVERE",
            "commercial_measurement_match": "PROMOTION_MISMATCH",
            "commercial_reason_codes": "PROMOTIONAL_STRUCTURE_MISMATCH",
            "commercial_exact_match_eligible": False,
        }
    )
    decision = finalize_unified_decisions(
        offers,
        pd.DataFrame([candidate]),
        run_id="adapted-test",
        model_id="model-v3",
    ).iloc[0]
    assert decision["final_decision"] == "MANUAL_REVIEW"
    assert decision["mapping_outcome"] == "ADAPTED_MATCH"
    assert not bool(decision["exact_match_eligible"])
    assert (
        decision["final_decision_reason"]
        == "ADAPTED_MATCH_REQUIRES_REVIEW"
    )


def test_non_own_offer_has_one_explicit_terminal_state() -> None:
    offers = pd.DataFrame(
        [
            {
                "offer_group_id": "own",
                "Offer Name": "Own offer",
                "is_own": True,
            },
            {
                "offer_group_id": "competitor",
                "Offer Name": "Competitor offer",
                "is_own": False,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            _candidate(
                "own",
                "SKU-OWN",
                probability=0.80,
                similarity=None,
                agreement_status="EMBEDDING_UNAVAILABLE",
                agreement_route="SAFE_FALLBACK",
            )
        ]
    )

    decisions = finalize_unified_decisions(
        offers,
        candidates,
        run_id="mixed",
        model_id="model",
    ).set_index("offer_id")

    assert len(decisions) == len(offers)
    assert decisions.index.is_unique
    assert (
        decisions.loc["competitor", "final_decision"]
        == FinalMatchDecision.COMPETITOR_OFFER.value
    )
    assert decisions.loc["competitor", "final_eligible_mapping"] == False
    assert decisions.loc["own", "proposed_master_sku"] == "SKU-OWN"


def test_competitor_discovery_receives_only_explicitly_eligible_rows() -> None:
    decisions = finalize_unified_decisions(
        _offers(),
        _candidates(),
        run_id="unified-test",
        model_id="model-v3",
    )
    eligible = select_competitor_eligible_rows(decisions)
    assert set(eligible["final_decision"]) == {
        FinalMatchDecision.AUTO_ACCEPT.value,
        FinalMatchDecision.LLM_ACCEPT.value,
    }
    assert set(eligible["offer_id"]) == {"safe", "llm"}
    assert not eligible["matched_master_sku"].eq("").any()
