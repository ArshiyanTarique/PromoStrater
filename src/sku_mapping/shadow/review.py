"""Offer-centred human-review templates for shadow observations."""

from __future__ import annotations

from typing import Any

import pandas as pd

ALLOWED_HUMAN_LABELS = frozenset(
    {
        "CORRECT_TOP_CANDIDATE",
        "CORRECT_OTHER_CANDIDATE",
        "NO_VALID_MASTER_SKU",
        "AMBIGUOUS_OFFER",
        "MULTIPLE_VALID_SKUS",
        "INSUFFICIENT_INFORMATION",
        "DATA_QUALITY_ERROR",
    }
)

REVIEW_COMPLETION_COLUMNS = (
    "human_label",
    "selected_master_itemcode",
    "reviewer_code",
    "reviewer_notes",
    "review_timestamp",
)


def create_human_review_view(
    selected_offers: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Create one reviewer-friendly row per offer with flattened candidates."""
    if selected_offers.empty:
        base_columns = [
            "shadow_run_id",
            "offer_group_id",
            "source_row_identifier",
            "source_offer_id",
            "source_offer_text",
            "entity_id",
            "entity_index",
            "entity_count",
            "entity_text",
            "entity_type",
            "conjunction_type",
            "attribute_inheritance_flags",
            "entity_parse_confidence",
            "offer_text",
            "selection_reasons",
            "existing_production_decision",
            "existing_production_tier",
            "current_production_match_itemcode",
            "product_family",
            "protein_classification",
            "offer_unit_pack_weight_g",
            "offer_number_of_units",
            "offer_total_weight_g",
            "pack_conflict",
            *REVIEW_COMPLETION_COLUMNS,
        ]
        return pd.DataFrame(columns=base_columns)

    rows: list[dict[str, Any]] = []
    # Only the selected offers are ever looked up below, so grouping and
    # sorting the whole candidate frame does the work for every unselected
    # offer as well - the dominant cost of this function on a full run.
    # Restricting to the selected ids first is behaviour-identical: the
    # discarded groups were never read.
    selected_ids = set(
        selected_offers["offer_group_id"].astype(str).tolist()
    )
    relevant = candidates[
        candidates["offer_group_id"].astype(str).isin(selected_ids)
    ]
    candidate_groups = {
        str(key): group.sort_values(
            ["candidate_rank", "master_itemcode"], kind="stable"
        )
        for key, group in relevant.groupby("offer_group_id", sort=False)
    }
    for _, selected in selected_offers.iterrows():
        offer_group_id = str(selected["offer_group_id"])
        offer_candidates = candidate_groups.get(
            offer_group_id, pd.DataFrame()
        )
        row: dict[str, Any] = {
            "shadow_run_id": selected.get("shadow_run_id", ""),
            "offer_group_id": offer_group_id,
            "source_row_identifier": selected.get(
                "source_row_identifier", ""
            ),
            "source_offer_id": selected.get("source_offer_id", offer_group_id),
            "source_offer_text": selected.get(
                "source_offer_text", selected.get("offer_text", "")
            ),
            "entity_id": selected.get("entity_id", offer_group_id),
            "entity_index": selected.get("entity_index", 1),
            "entity_count": selected.get("entity_count", 1),
            "entity_text": selected.get(
                "entity_text", selected.get("offer_text", "")
            ),
            "entity_type": selected.get("entity_type", "SINGLE_PRODUCT"),
            "conjunction_type": selected.get("conjunction_type", "SINGLE"),
            "attribute_inheritance_flags": selected.get(
                "attribute_inheritance_flags", ""
            ),
            "entity_parse_confidence": selected.get(
                "entity_parse_confidence", pd.NA
            ),
            "offer_text": selected.get("offer_text", ""),
            "selection_reasons": selected.get("selection_reasons", ""),
            "existing_production_decision": selected.get(
                "existing_production_decision", ""
            ),
            "existing_production_tier": selected.get(
                "existing_production_tier", ""
            ),
            "current_production_match_itemcode": selected.get(
                "current_production_match_itemcode", ""
            ),
            "product_family": selected.get("product_family", ""),
            "protein_classification": selected.get(
                "protein_classification", ""
            ),
            "offer_unit_pack_weight_g": selected.get(
                "unit_pack_weight_g", pd.NA
            ),
            "offer_number_of_units": selected.get(
                "number_of_units", pd.NA
            ),
            "offer_total_weight_g": selected.get(
                "total_offer_weight_g", pd.NA
            ),
            "pack_conflict": selected.get("pack_conflict", False),
            "agreement_status": selected.get("agreement_status", ""),
            "routing_decision": selected.get("routing_decision", ""),
            "routing_reason": selected.get("routing_reason", ""),
            "lightgbm_top_candidate": selected.get(
                "lightgbm_top_candidate", ""
            ),
            "embedding_top_candidate": selected.get(
                "agreement_embedding_top_candidate", ""
            ),
            "same_top_candidate": selected.get(
                "same_top_candidate", False
            ),
            "llm_review_status": selected.get(
                "llm_review_status", ""
            ),
            "llm_final_route": selected.get("llm_final_route", ""),
            "llm_selected_candidate": selected.get(
                "llm_selected_candidate", ""
            ),
            "llm_confidence": selected.get("llm_confidence", pd.NA),
            "llm_routing_reason": selected.get(
                "llm_routing_reason", ""
            ),
            "human_label": "",
            "selected_master_itemcode": "",
            "reviewer_code": "",
            "reviewer_notes": "",
            "review_timestamp": "",
        }
        for rank in range(1, top_k + 1):
            ranked = offer_candidates[
                offer_candidates["candidate_rank"] == rank
            ]
            candidate = ranked.iloc[0] if not ranked.empty else None
            prefix = f"top_candidate_{rank}"
            row[f"{prefix}_itemcode"] = (
                candidate.get("master_itemcode", "")
                if candidate is not None
                else ""
            )
            row[f"{prefix}_description"] = (
                candidate.get("master_item_description", "")
                if candidate is not None
                else ""
            )
            row[f"{prefix}_probability"] = (
                candidate.get("calibrated_probability", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_embedding_similarity"] = (
                candidate.get("embedding_similarity", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_embedding_rank"] = (
                candidate.get("embedding_rank", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_embedding_top_candidate"] = (
                candidate.get("embedding_top_candidate", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_embedding_failure_reason"] = (
                candidate.get("embedding_failure_reason", "")
                if candidate is not None
                else ""
            )
            row[f"{prefix}_text_score"] = (
                candidate.get("candidate_text_score", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_master_unit_weight_g"] = (
                candidate.get("master_unit_weight_g", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_family_match"] = (
                candidate.get("family_match", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_protein_match"] = (
                candidate.get("protein_match", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_variant_match"] = (
                candidate.get("variant_match", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_size_match"] = (
                candidate.get("size_match", pd.NA)
                if candidate is not None
                else pd.NA
            )
            row[f"{prefix}_pack_conflict"] = (
                candidate.get("pack_conflict", pd.NA)
                if candidate is not None
                else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows)
