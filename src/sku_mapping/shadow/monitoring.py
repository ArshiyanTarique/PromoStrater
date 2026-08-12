"""Operational monitoring for unreviewed and reviewed shadow observations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _distribution(values: pd.Series, bins: list[float]) -> list[dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    counts, edges = np.histogram(numeric, bins=bins)
    return [
        {
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "rows": int(counts[index]),
        }
        for index in range(len(counts))
    ]


def _group_score_summary(
    frame: pd.DataFrame, group_column: str
) -> dict[str, Any]:
    if group_column not in frame:
        return {}
    result: dict[str, Any] = {}
    for key, group in frame.groupby(group_column, dropna=False, sort=True):
        probabilities = pd.to_numeric(
            group["calibrated_probability"], errors="coerce"
        )
        result[str(key) if pd.notna(key) else "<missing>"] = {
            "candidate_rows": int(len(group)),
            "offer_groups": int(group["offer_group_id"].nunique()),
            "mean_probability": (
                float(probabilities.mean()) if probabilities.notna().any() else None
            ),
            "high_score_rows": int(
                group["shadow_decision_bucket"]
                .eq("SHADOW_HIGH_SCORE")
                .sum()
            ),
        }
    return result


def build_shadow_monitoring_report(
    predictions: pd.DataFrame,
    *,
    model_id: str,
    package_sha256: str,
    failed_shadow_predictions: int = 0,
    package_validation_failures: int = 0,
    llm_review_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build diagnostics without presenting unreviewed rows as accuracy."""
    reviewed = (
        predictions["human_label"].astype("string").fillna("").str.strip().ne("")
        if "human_label" in predictions
        else pd.Series(False, index=predictions.index)
    )
    rank_counts = (
        predictions["candidate_rank"].value_counts().sort_index()
        if "candidate_rank" in predictions
        else pd.Series(dtype=int)
    )
    family_counts = (
        predictions["product_family"]
        .astype("string")
        .fillna("<missing>")
        .value_counts()
        if "product_family" in predictions
        else pd.Series(dtype=int)
    )
    missingness_counts = (
        predictions["feature_missingness_count"].value_counts().sort_index()
        if "feature_missingness_count" in predictions
        else pd.Series(dtype=int)
    )
    disagreement = (
        predictions["production_shadow_disagreement"].fillna(False).astype(bool)
        if "production_shadow_disagreement" in predictions
        else pd.Series(False, index=predictions.index)
    )
    raw = pd.to_numeric(
        predictions.get("raw_model_score", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    raw_min = float(raw.min()) if not raw.empty else -1.0
    raw_max = float(raw.max()) if not raw.empty else 1.0
    if raw_min == raw_max:
        raw_min -= 0.5
        raw_max += 0.5
    reviewed_labels = (
        predictions.loc[reviewed, "human_label"].value_counts()
        if reviewed.any()
        else pd.Series(dtype=int)
    )
    embedding_similarity = pd.to_numeric(
        predictions.get(
            "embedding_similarity", pd.Series(dtype=float)
        ),
        errors="coerce",
    )
    embedding_failures = (
        predictions.get(
            "embedding_failure_reason",
            pd.Series("", index=predictions.index),
        )
        .astype("string")
        .fillna("")
        .str.strip()
        .ne("")
    )
    embedding_status_counts = (
        predictions.get(
            "embedding_status",
            pd.Series("NOT_RECORDED", index=predictions.index),
        )
        .astype("string")
        .fillna("NOT_RECORDED")
        .value_counts()
    )
    embedding_failure_reasons = (
        predictions.get(
            "embedding_failure_reason",
            pd.Series("", index=predictions.index),
        )
        .astype("string")
        .fillna("")
        .loc[lambda values: values.str.strip().ne("")]
        .value_counts()
    )
    embedding_model_ids = (
        predictions.get(
            "embedding_model_id", pd.Series(dtype="string")
        )
        .astype("string")
        .dropna()
        .unique()
        .tolist()
    )
    embedding_versions = (
        predictions.get(
            "embedding_model_version", pd.Series(dtype="string")
        )
        .astype("string")
        .dropna()
        .unique()
        .tolist()
    )
    agreement_rows = (
        predictions.drop_duplicates("offer_id")
        if "offer_id" in predictions
        else pd.DataFrame()
    )
    agreement_status = (
        agreement_rows["agreement_status"].astype("string")
        if "agreement_status" in agreement_rows
        else pd.Series(dtype="string")
    )
    agreement_route = (
        agreement_rows["routing_decision"].astype("string")
        if "routing_decision" in agreement_rows
        else pd.Series(dtype="string")
    )
    llm_status = (
        agreement_rows["llm_review_status"].astype("string")
        if "llm_review_status" in agreement_rows
        else pd.Series(dtype="string")
    )
    llm_route = (
        agreement_rows["llm_final_route"].astype("string")
        if "llm_final_route" in agreement_rows
        else pd.Series(dtype="string")
    )
    llm_summary = llm_review_summary or {}
    return {
        "report_type": "SHADOW_OPERATIONAL_MONITORING",
        "not_production_accuracy": True,
        "model_package": {
            "model_id": model_id,
            "package_sha256": package_sha256,
        },
        "prediction_volume": {
            "candidate_rows": int(len(predictions)),
            "offer_groups": int(predictions["offer_group_id"].nunique())
            if "offer_group_id" in predictions
            else 0,
        },
        "raw_score_distribution": _distribution(
            raw, list(np.linspace(raw_min, raw_max, 11))
        ),
        "calibrated_probability_distribution": _distribution(
            predictions.get(
                "calibrated_probability", pd.Series(dtype=float)
            ),
            list(np.linspace(0.0, 1.0, 11)),
        ),
        "score_distribution_by_source": _group_score_summary(
            predictions, "source_dataset"
        ),
        "score_distribution_by_product_family": _group_score_summary(
            predictions, "product_family"
        ),
        "missingness_distribution": {
            str(key): int(value) for key, value in missingness_counts.items()
        },
        "production_shadow_disagreement": {
            "candidate_rows": int(disagreement.sum()),
            "offer_groups": int(
                predictions.loc[disagreement, "offer_group_id"].nunique()
            )
            if "offer_group_id" in predictions
            else 0,
        },
        "candidate_rank_distribution": {
            str(key): int(value) for key, value in rank_counts.items()
        },
        "high_score_volume": int(
            predictions.get(
                "shadow_decision_bucket", pd.Series(dtype="string")
            )
            .eq("SHADOW_HIGH_SCORE")
            .sum()
        ),
        "family_coverage": {
            str(key): int(value) for key, value in family_counts.items()
        },
        "pack_conflict_frequency": float(
            predictions.get("pack_conflict", pd.Series(False, index=predictions.index))
            .fillna(False)
            .astype(bool)
            .mean()
        )
        if len(predictions)
        else 0.0,
        "mixed_protein_frequency": float(
            pd.to_numeric(
                predictions.get(
                    "is_mixed_protein_offer",
                    pd.Series(0, index=predictions.index),
                ),
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .mean()
        )
        if len(predictions)
        else 0.0,
        "failed_shadow_predictions": int(failed_shadow_predictions),
        "package_validation_failures": int(package_validation_failures),
        "embedding_scoring": {
            "present": "embedding_similarity" in predictions,
            "used_for_production_decision": False,
            "status_counts": {
                str(key): int(value)
                for key, value in embedding_status_counts.items()
            },
            "failure_reason_counts": {
                str(key): int(value)
                for key, value in embedding_failure_reasons.items()
            },
            "model_ids": sorted(str(value) for value in embedding_model_ids),
            "model_versions": sorted(
                str(value) for value in embedding_versions
            ),
            "candidate_rows": int(embedding_similarity.notna().sum()),
            "failure_rows": int(embedding_failures.sum()),
            "top_candidate_rows": int(
                predictions.get(
                    "embedding_top_candidate",
                    pd.Series(False, index=predictions.index),
                )
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "similarity_distribution": _distribution(
                embedding_similarity, list(np.linspace(-1.0, 1.0, 11))
            ),
        },
        "agreement_routing": {
            "total_offers": int(len(agreement_rows)),
            "safe_agreements": int(
                agreement_status.eq("SAFE_AGREEMENT").sum()
            ),
            "weak_agreements": int(
                agreement_status.eq("WEAK_AGREEMENT").sum()
            ),
            "disagreements": int(
                agreement_status.eq("DISAGREEMENT").sum()
            ),
            "embedding_unavailable": int(
                agreement_status.eq("EMBEDDING_UNAVAILABLE").sum()
            ),
            "model_unavailable": int(
                agreement_status.eq("MODEL_UNAVAILABLE").sum()
            ),
            "routed_to_auto_accept": int(
                agreement_route.eq("AUTO_ACCEPT").sum()
            ),
            "routed_to_llm_review": int(
                agreement_route.eq("LLM_REVIEW").sum()
            ),
            "routed_to_manual_review": int(
                agreement_route.eq("MANUAL_REVIEW").sum()
            ),
            "routed_to_safe_fallback": int(
                agreement_route.eq("SAFE_FALLBACK").sum()
            ),
            "llm_calls": int(llm_summary.get("provider_calls", 0)),
            "accuracy_claim_permitted": False,
        },
        "llm_review": {
            "enabled": bool(llm_summary.get("enabled", False)),
            "offers_routed": int(llm_summary.get("offers_routed", 0)),
            "provider_calls": int(llm_summary.get("provider_calls", 0)),
            "cache_hits": int(llm_summary.get("cache_hits", 0)),
            "completed": int(llm_status.eq("COMPLETED").sum()),
            "disabled": int(llm_status.eq("DISABLED").sum()),
            "invalid_responses": int(
                llm_status.eq("INVALID_RESPONSE").sum()
            ),
            "timeouts": int(llm_status.eq("TIMEOUT").sum()),
            "provider_failures": int(
                llm_status.eq("PROVIDER_FAILURE").sum()
            ),
            "eligible_llm_accepts": int(
                llm_route.eq("LLM_ACCEPT").sum()
            ),
            "routed_to_manual_review": int(
                llm_route.eq("MANUAL_REVIEW").sum()
            ),
            "routed_to_no_match": int(
                llm_route.eq("NO_MATCH").sum()
            ),
            "production_decisions_modified": False,
            "training_data_modified": False,
            "accuracy_claim_permitted": False,
        },
        "reviewed_performance": {
            "status": (
                "HUMAN_LABELS_AVAILABLE"
                if reviewed.any()
                else "NO_HUMAN_LABELS_AVAILABLE"
            ),
            "reviewed_candidate_rows": int(reviewed.sum()),
            "human_label_distribution": {
                str(key): int(value) for key, value in reviewed_labels.items()
            },
            "accuracy_metrics": None,
        },
        "unreviewed_operational_diagnostics": {
            "candidate_rows": int((~reviewed).sum()),
            "accuracy_claim_permitted": False,
        },
    }
