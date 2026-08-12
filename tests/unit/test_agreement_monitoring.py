"""Agreement status and routing counts in operational monitoring."""

import pandas as pd

from sku_mapping.shadow.monitoring import build_shadow_monitoring_report


def test_monitoring_counts_every_agreement_and_route_branch() -> None:
    statuses = [
        "SAFE_AGREEMENT",
        "WEAK_AGREEMENT",
        "DISAGREEMENT",
        "EMBEDDING_UNAVAILABLE",
        "MODEL_UNAVAILABLE",
    ]
    routes = [
        "AUTO_ACCEPT",
        "LLM_REVIEW",
        "LLM_REVIEW",
        "MANUAL_REVIEW",
        "SAFE_FALLBACK",
    ]
    predictions = pd.DataFrame(
        {
            "offer_id": [f"offer-{index}" for index in range(5)],
            "offer_group_id": [
                f"offer-{index}" for index in range(5)
            ],
            "agreement_status": statuses,
            "routing_decision": routes,
            "candidate_rank": [1] * 5,
            "calibrated_probability": [0.9] * 5,
            "raw_model_score": [1.0] * 5,
            "shadow_decision_bucket": ["SHADOW_REVIEW"] * 5,
            "human_label": [""] * 5,
        }
    )
    report = build_shadow_monitoring_report(
        predictions,
        model_id="model",
        package_sha256="a" * 64,
    )
    agreement = report["agreement_routing"]
    assert agreement == {
        "total_offers": 5,
        "safe_agreements": 1,
        "weak_agreements": 1,
        "disagreements": 1,
        "embedding_unavailable": 1,
        "model_unavailable": 1,
        "routed_to_auto_accept": 1,
        "routed_to_llm_review": 2,
        "routed_to_manual_review": 1,
        "routed_to_safe_fallback": 1,
        "llm_calls": 0,
        "accuracy_claim_permitted": False,
    }


def test_monitoring_counts_structured_llm_review_outcomes() -> None:
    predictions = pd.DataFrame(
        {
            "offer_id": [f"offer-{index}" for index in range(4)],
            "offer_group_id": [f"offer-{index}" for index in range(4)],
            "agreement_status": ["DISAGREEMENT"] * 4,
            "routing_decision": ["LLM_REVIEW"] * 4,
            "llm_review_status": [
                "COMPLETED",
                "COMPLETED",
                "TIMEOUT",
                "PROVIDER_FAILURE",
            ],
            "llm_final_route": [
                "LLM_ACCEPT",
                "NO_MATCH",
                "MANUAL_REVIEW",
                "MANUAL_REVIEW",
            ],
            "candidate_rank": [1] * 4,
            "calibrated_probability": [0.5] * 4,
            "raw_model_score": [0.0] * 4,
            "shadow_decision_bucket": ["SHADOW_REVIEW"] * 4,
            "human_label": [""] * 4,
        }
    )
    report = build_shadow_monitoring_report(
        predictions,
        model_id="model",
        package_sha256="a" * 64,
        llm_review_summary={
            "enabled": True,
            "offers_routed": 4,
            "provider_calls": 5,
            "cache_hits": 1,
        },
    )
    llm = report["llm_review"]
    assert llm["offers_routed"] == 4
    assert llm["provider_calls"] == 5
    assert llm["cache_hits"] == 1
    assert llm["completed"] == 2
    assert llm["timeouts"] == 1
    assert llm["provider_failures"] == 1
    assert llm["eligible_llm_accepts"] == 1
    assert llm["routed_to_no_match"] == 1
    assert llm["routed_to_manual_review"] == 2
    assert llm["production_decisions_modified"] is False
    assert llm["training_data_modified"] is False
