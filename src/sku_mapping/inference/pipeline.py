"""One explicit candidate-to-final-decision assisted inference pipeline.

This module composes existing inference components. It never fits a model,
updates a package, writes training data, or treats review output as learning.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

import pandas as pd

from sku_mapping.config import PipelineConfig
from sku_mapping.constants import FinalMatchDecision, MLDeploymentMode
from sku_mapping.failure_diagnostics import capture_exception_details
from sku_mapping.features.commercial_entities import expand_offer_entities
from sku_mapping.inference.invariants import validate_terminal_decisions
from sku_mapping.shadow.pipeline import (
    InferenceProgressCallback,
    ShadowRunResult,
    run_shadow_observation_non_blocking,
    stable_offer_group_id,
)

LOGGER = logging.getLogger(__name__)

ELIGIBLE_FINAL_DECISIONS = frozenset(
    {
        FinalMatchDecision.AUTO_ACCEPT.value,
        FinalMatchDecision.LLM_ACCEPT.value,
    }
)
HARD_CONFLICT_COLUMNS = (
    "protein_conflict",
    "mixed_protein_ambiguity",
    "strong_family_conflict",
    "strong_size_weight_conflict",
    "strong_pack_format_conflict",
    "feature_generation_failure",
    "missing_master",
    "commercial_hard_conflict",
)


@dataclass(frozen=True)
class UnifiedInferenceResult:
    """Applied rows, auditable decisions, stage artifacts, and statistics."""

    status: str
    rows: pd.DataFrame
    decisions: pd.DataFrame
    candidates: pd.DataFrame
    run_id: str | None
    statistics: dict[str, Any]
    output_paths: dict[str, Path]
    shadow_result: ShadowRunResult | None
    error: str | None = None
    original_exception: dict[str, Any] | None = None
    exception: BaseException | None = None


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_int(value: object) -> int | None:
    numeric = _safe_float(value)
    return int(numeric) if numeric is not None else None


def _safe_bool(value: object) -> bool:
    return bool(pd.notna(value) and bool(value))


def _hard_conflict(row: Mapping[str, Any]) -> bool:
    return any(
        _safe_bool(row.get(column, False))
        for column in HARD_CONFLICT_COLUMNS
    )


def _base_record(
    offer: Mapping[str, Any],
    *,
    run_id: str,
    model_id: str | None,
) -> dict[str, Any]:
    return {
        "offer_id": _safe_text(offer["offer_group_id"]),
        "source_offer_id": _safe_text(
            offer.get("source_offer_id", offer["offer_group_id"])
        ),
        "source_offer_text": _safe_text(
            offer.get(
                "source_offer_text",
                offer.get("Offer Name", offer.get("offer_text", "")),
            )
        ),
        "entity_id": _safe_text(
            offer.get("entity_id", offer["offer_group_id"])
        ),
        "entity_index": _safe_int(offer.get("entity_index")) or 1,
        "entity_count": _safe_int(offer.get("entity_count")) or 1,
        "entity_text": _safe_text(
            offer.get("entity_text", offer.get("Offer Name", ""))
        ),
        "entity_type": _safe_text(
            offer.get("entity_type", "SINGLE_PRODUCT")
        ),
        "conjunction_type": _safe_text(
            offer.get("conjunction_type", "SINGLE")
        ),
        "attribute_inheritance_flags": _safe_text(
            offer.get("attribute_inheritance_flags", "")
        ),
        "entity_parse_confidence": _safe_float(
            offer.get("entity_parse_confidence")
        ),
        "offer_description": _safe_text(
            offer.get("Offer Name", offer.get("offer_text", ""))
        ),
        "source_row_count": max(
            1, _safe_int(offer.get("source_row_count")) or 1
        ),
        "is_own_brand": (
            _safe_bool(offer.get("is_own"))
            if "is_own" in offer
            else True
        ),
        "proposed_master_sku": "",
        "proposed_master_description": "",
        "proposed_candidate_rank": None,
        "matched_master_sku": "",
        "matched_master_description": "",
        "final_decision": FinalMatchDecision.MANUAL_REVIEW.value,
        "decision_source": "DETERMINISTIC_POLICY",
        "final_decision_reason": "",
        "final_eligible_mapping": False,
        "lightgbm_probability": None,
        "agreement_status": "",
        "agreement_route": "",
        "llm_decision": "",
        "llm_confidence": None,
        "human_review_status": _safe_text(
            offer.get("human_review_status", "UNREVIEWED")
        )
        or "UNREVIEWED",
        "model_id": model_id,
        "llm_model_id": "",
        "run_id": run_id,
        "hard_conflict": False,
        "mapping_outcome": "",
        "commercial_severity": "",
        "commercial_measurement_match": "",
        "commercial_reason_codes": "",
        "exact_match_eligible": False,
    }


_ORDER_COLUMN = "_expansion_source_position"


def _expand_inference_entities(
    production_rows: pd.DataFrame,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Expand own-brand multi-product rows, preserving competitor rows.

    Own-brand rows are decomposed in one batched call and competitor rows are
    filled in with vectorised assignment. The previous implementation walked
    every canonical offer with ``iterrows`` and built a throwaway one-row
    DataFrame per own-brand offer, so a large upload spent minutes here with
    nothing to report. Output rows and their order are unchanged.
    """
    if production_rows.empty:
        return production_rows.copy()

    frame = production_rows.copy()
    frame[_ORDER_COLUMN] = range(len(frame))
    frame["source_offer_id"] = frame["offer_group_id"].map(_safe_text)
    frame["source_offer_text"] = (
        frame["Offer Name"].map(_safe_text)
        if "Offer Name" in frame.columns
        else ""
    )
    own_mask = (
        frame["is_own"].map(_safe_bool)
        if "is_own" in frame.columns
        else pd.Series(True, index=frame.index)
    )

    parts: list[pd.DataFrame] = []
    own = frame.loc[own_mask]
    if not own.empty:
        expanded = expand_offer_entities(own, progress=progress)
        if expanded.empty:
            # Fail closed: silently dropping every own-brand offer would
            # produce an empty mapping export that looks like a clean run.
            raise ValueError(
                "Entity expansion returned no rows for "
                f"{len(own):,} own-brand offers"
            )
        counts = expanded["entity_count"].map(
            lambda value: _safe_int(value) or 1
        )
        single = counts.eq(1)
        expanded["offer_group_id"] = expanded["entity_id"].map(_safe_text)
        expanded.loc[single, "offer_group_id"] = expanded.loc[
            single, "source_offer_id"
        ].map(_safe_text)
        parts.append(expanded)

    others = frame.loc[~own_mask]
    if not others.empty:
        others = others.copy()
        others["entity_id"] = others["source_offer_id"].astype(str) + "_1"
        others["entity_index"] = 1
        others["entity_count"] = 1
        others["entity_text"] = others["source_offer_text"]
        others["entity_type"] = "SINGLE_PRODUCT"
        others["conjunction_type"] = "SINGLE"
        others["attribute_inheritance_flags"] = ""
        others["entity_parse_confidence"] = 1.0
        parts.append(others)

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined = combined.sort_values(
        [_ORDER_COLUMN, "entity_index"], kind="stable"
    )
    return combined.drop(columns=[_ORDER_COLUMN]).reset_index(drop=True)


def _manual_reason(group_top: Mapping[str, Any]) -> str:
    llm_status = _safe_text(group_top.get("llm_review_status"))
    llm_reason = _safe_text(group_top.get("llm_routing_reason"))
    agreement_reason = _safe_text(group_top.get("routing_reason"))
    if llm_status:
        return llm_reason or f"LLM_{llm_status}"
    return agreement_reason or "AGREEMENT_POLICY_REQUIRES_REVIEW"


def finalize_unified_decisions(
    offers: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    run_id: str,
    model_id: str | None,
) -> pd.DataFrame:
    """Apply the final deterministic routing matrix to scorer outputs."""
    if "offer_group_id" not in offers:
        raise ValueError("Unified offers require offer_group_id")
    candidate_offer_ids: set[str] = set()
    candidate_by_offer_and_sku: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    first_by_offer: dict[str, dict[str, Any]] = {}
    for candidate in candidates.to_dict(orient="records"):
        candidate_offer_id = _safe_text(
            candidate.get("offer_group_id")
        )
        candidate_sku = _safe_text(candidate.get("master_itemcode"))
        if not candidate_offer_id:
            continue
        candidate_offer_ids.add(candidate_offer_id)
        first_by_offer.setdefault(candidate_offer_id, candidate)
        if candidate_sku:
            candidate_by_offer_and_sku.setdefault(
                (candidate_offer_id, candidate_sku), candidate
            )

    top_by_offer: dict[str, dict[str, Any]] = {}
    if not candidates.empty:
        ranked_candidates = candidates.assign(
            _probability=pd.to_numeric(
                candidates["calibrated_probability"], errors="coerce"
            ),
            _commercial_priority=candidates.get(
                "commercial_outcome",
                pd.Series("UNKNOWN", index=candidates.index),
            ).map(
                {
                    "EXACT_MATCH": 0,
                    "ADAPTED_MATCH": 1,
                    "UNKNOWN": 2,
                    "UNACCEPTABLE_MATCH": 3,
                }
            ).fillna(2),
        ).sort_values(
            [
                "offer_group_id",
                "_commercial_priority",
                "_probability",
                "candidate_rank",
                "master_itemcode",
            ],
            ascending=[True, True, False, True, True],
            kind="stable",
        )
        top_by_offer = {
            _safe_text(row.get("offer_group_id")): row
            for row in ranked_candidates.drop_duplicates(
                "offer_group_id", keep="first"
            ).to_dict(orient="records")
        }
    records: list[dict[str, Any]] = []
    for offer in offers.to_dict(orient="records"):
        record = _base_record(
            offer, run_id=run_id, model_id=model_id
        )
        offer_id = record["offer_id"]
        if "is_own" in offer and not _safe_bool(offer.get("is_own")):
            record.update(
                {
                    "final_decision": (
                        FinalMatchDecision.COMPETITOR_OFFER.value
                    ),
                    "decision_source": "OFFER_CLASSIFICATION",
                    "final_decision_reason": "NON_OWN_BRAND_OFFER",
                    "human_review_status": "NOT_APPLICABLE",
                }
            )
            records.append(record)
            continue
        if offer_id not in candidate_offer_ids:
            record.update(
                {
                    "final_decision": (
                        FinalMatchDecision.NO_CANDIDATE.value
                    ),
                    "decision_source": "CANDIDATE_GENERATOR",
                    "final_decision_reason": "NO_RETAINED_CANDIDATE",
                }
            )
            records.append(record)
            continue

        top = top_by_offer.get(offer_id) or first_by_offer[offer_id]
        agreement_status = _safe_text(top.get("agreement_status"))
        agreement_route = _safe_text(top.get("routing_decision"))
        top_itemcode = _safe_text(
            top.get("lightgbm_top_candidate", top["master_itemcode"])
        )
        selected = candidate_by_offer_and_sku.get(
            (offer_id, top_itemcode), top
        )
        selected_itemcode = _safe_text(selected.get("master_itemcode"))
        selected_description = _safe_text(
            selected.get("master_item_description")
        )
        record.update(
            {
                "proposed_master_sku": selected_itemcode,
                "proposed_master_description": selected_description,
                "proposed_candidate_rank": _safe_int(
                    selected.get("candidate_rank")
                ),
                "lightgbm_probability": _safe_float(
                    selected.get("calibrated_probability")
                ),
                "agreement_status": agreement_status,
                "agreement_route": agreement_route,
                "mapping_outcome": _safe_text(
                    selected.get("commercial_outcome")
                ),
                "commercial_severity": _safe_text(
                    selected.get("commercial_severity")
                ),
                "commercial_measurement_match": _safe_text(
                    selected.get("commercial_measurement_match")
                ),
                "commercial_reason_codes": _safe_text(
                    selected.get("commercial_reason_codes")
                ),
                "exact_match_eligible": _safe_bool(
                    selected.get("commercial_exact_match_eligible", False)
                ),
                "llm_decision": _safe_text(
                    top.get("llm_parsed_decision")
                ),
                "llm_confidence": _safe_float(
                    top.get("llm_confidence")
                ),
                "llm_model_id": _safe_text(
                    top.get("llm_model_id")
                ),
            }
        )

        if agreement_status == "MODEL_UNAVAILABLE":
            record.update(
                {
                    "final_decision": (
                        FinalMatchDecision.MODEL_ERROR.value
                    ),
                    "decision_source": "LIGHTGBM",
                    "final_decision_reason": "LIGHTGBM_UNAVAILABLE",
                }
            )
            records.append(record)
            continue

        hard_conflict = _hard_conflict(top)
        record["hard_conflict"] = hard_conflict
        if _safe_bool(top.get("missing_master", False)):
            record.update(
                {
                    "final_decision": (
                        FinalMatchDecision.MASTER_SKU_NOT_FOUND.value
                    ),
                    "decision_source": "CATALOGUE_VALIDATION",
                    "final_decision_reason": "MASTER_SKU_NOT_FOUND",
                }
            )
            records.append(record)
            continue
        if hard_conflict:
            record.update(
                {
                    "final_decision": (
                        FinalMatchDecision.MANUAL_REVIEW.value
                    ),
                    "decision_source": "DETERMINISTIC_SAFETY_POLICY",
                    "final_decision_reason": "HARD_CONFLICT",
                }
            )
            records.append(record)
            continue

        commercial_outcome = _safe_text(
            selected.get("commercial_outcome")
        )
        if commercial_outcome in {"ADAPTED_MATCH", "UNKNOWN"}:
            record.update(
                {
                    "final_decision": (
                        FinalMatchDecision.MANUAL_REVIEW.value
                    ),
                    "decision_source": (
                        "DETERMINISTIC_COMMERCIAL_POLICY"
                    ),
                    "final_decision_reason": (
                        f"{commercial_outcome}_REQUIRES_REVIEW"
                    ),
                }
            )
            records.append(record)
            continue

        if agreement_route == "AUTO_ACCEPT":
            decision = FinalMatchDecision.AUTO_ACCEPT
            decision_source = "LIGHTGBM_THRESHOLD"
            decision_reason = "ABOVE_AUTO_ACCEPT_THRESHOLD"
        elif agreement_route == "LLM_REVIEW":
            llm_route = _safe_text(top.get("llm_final_route"))
            if llm_route == "LLM_ACCEPT":
                llm_itemcode = _safe_text(
                    top.get("llm_selected_candidate")
                )
                llm_candidate = candidate_by_offer_and_sku.get(
                    (offer_id, llm_itemcode)
                )
                if llm_candidate is None or _hard_conflict(llm_candidate):
                    decision = FinalMatchDecision.MANUAL_REVIEW
                    decision_source = "DETERMINISTIC_SAFETY_POLICY"
                    decision_reason = (
                        "INVALID_OR_CONFLICTING_LLM_SELECTION"
                    )
                else:
                    selected = llm_candidate
                    decision = FinalMatchDecision.LLM_ACCEPT
                    decision_source = "STRUCTURED_LLM_REVIEW"
                    decision_reason = "VALID_HIGH_CONFIDENCE_LLM_ACCEPT"
            elif llm_route == "NO_MATCH":
                decision = FinalMatchDecision.NO_MATCH
                decision_source = "STRUCTURED_LLM_REVIEW"
                decision_reason = "EXPLICIT_LLM_REJECT_ALL_POLICY"
            else:
                decision = FinalMatchDecision.MANUAL_REVIEW
                decision_source = "STRUCTURED_LLM_REVIEW"
                decision_reason = _manual_reason(top)
        else:
            decision = FinalMatchDecision.MANUAL_REVIEW
            decision_source = "AGREEMENT_POLICY"
            decision_reason = _manual_reason(top)

        selected_itemcode = _safe_text(
            selected.get("master_itemcode")
        )
        selected_description = _safe_text(
            selected.get("master_item_description")
        )
        if decision in {
            FinalMatchDecision.AUTO_ACCEPT,
            FinalMatchDecision.LLM_ACCEPT,
        } and not selected_itemcode:
            decision = FinalMatchDecision.MASTER_SKU_NOT_FOUND
            decision_source = "CATALOGUE_VALIDATION"
            decision_reason = "SELECTED_MASTER_SKU_MISSING"
        eligible = decision.value in ELIGIBLE_FINAL_DECISIONS
        record.update(
            {
                "proposed_master_sku": selected_itemcode,
                "proposed_master_description": selected_description,
                "proposed_candidate_rank": _safe_int(
                    selected.get("candidate_rank")
                ),
                "matched_master_sku": (
                    selected_itemcode if eligible else ""
                ),
                "matched_master_description": (
                    selected_description if eligible else ""
                ),
                "final_decision": decision.value,
                "decision_source": decision_source,
                "final_decision_reason": decision_reason,
                "final_eligible_mapping": eligible,
                "lightgbm_probability": _safe_float(
                    selected.get("calibrated_probability")
                ),
            }
        )
        records.append(record)
    decisions = pd.DataFrame(records)
    aggregate_status: dict[str, str] = {}
    for source_id, group in decisions.groupby(
        "source_offer_id", sort=False
    ):
        entity_total = len(group)
        eligible_total = int(
            group["final_eligible_mapping"].fillna(False).astype(bool).sum()
        )
        manual_total = int(
            group["final_decision"].eq(
                FinalMatchDecision.MANUAL_REVIEW.value
            ).sum()
        )
        ambiguous = bool(
            entity_total > 1
            and (
                group["entity_parse_confidence"]
                .fillna(0)
                .astype(float)
                .lt(0.7)
                .any()
                or group["conjunction_type"].eq(
                    "UNKNOWN_MULTI_PRODUCT"
                ).any()
            )
        )
        if ambiguous:
            status = "AMBIGUOUS_DECOMPOSITION"
        elif eligible_total == entity_total:
            status = "ALL_ENTITIES_MATCHED"
        elif eligible_total:
            status = "PARTIALLY_MATCHED"
        elif manual_total == entity_total:
            status = "ALL_ENTITIES_REVIEW"
        else:
            status = "NO_ENTITIES_MATCHED"
        aggregate_status[str(source_id)] = status
    decisions["source_offer_aggregate_status"] = decisions[
        "source_offer_id"
    ].map(aggregate_status)
    return decisions


def _attach_final_decisions(
    offers: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    result = offers.copy(deep=True)
    appended = decisions.rename(columns={"offer_id": "offer_group_id"})
    existing_append_columns = [
        column
        for column in appended.columns
        if column != "offer_group_id" and column in result.columns
    ]
    if existing_append_columns:
        result.drop(columns=existing_append_columns, inplace=True)
    result = result.merge(
        appended,
        on="offer_group_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    eligible = result["final_eligible_mapping"].fillna(False).astype(bool)
    manual = result["final_decision"].eq(
        FinalMatchDecision.MANUAL_REVIEW.value
    )
    no_match = result["final_decision"].eq(
        FinalMatchDecision.NO_MATCH.value
    )
    result["suggested_itemcode"] = result.get(
        "suggested_itemcode", pd.Series("", index=result.index)
    )
    result["suggested_itemname"] = result.get(
        "suggested_itemname", pd.Series("", index=result.index)
    )
    result.loc[
        result["proposed_master_sku"].astype(str).ne(""),
        "suggested_itemcode",
    ] = result.loc[
        result["proposed_master_sku"].astype(str).ne(""),
        "proposed_master_sku",
    ]
    result.loc[
        result["proposed_master_description"].astype(str).ne(""),
        "suggested_itemname",
    ] = result.loc[
        result["proposed_master_description"].astype(str).ne(""),
        "proposed_master_description",
    ]
    result["matched_itemcode"] = "REVIEW_REQUIRED"
    result["matched_itemname"] = "REVIEW_REQUIRED"
    result.loc[eligible, "matched_itemcode"] = result.loc[
        eligible, "matched_master_sku"
    ]
    result.loc[eligible, "matched_itemname"] = result.loc[
        eligible, "matched_master_description"
    ]
    result.loc[no_match, "matched_itemcode"] = "NO_MATCH"
    result.loc[no_match, "matched_itemname"] = "None"
    result["ml_decision"] = result["final_decision"]
    result.loc[eligible, "ml_decision"] = "AUTO_MATCH"
    result["ml_probability"] = result["lightgbm_probability"]
    result["confidence_tier"] = "medium (ml)"
    result.loc[eligible, "confidence_tier"] = "high (ml)"
    result.loc[no_match, "confidence_tier"] = "low (ml)"
    result.loc[manual, "matched_itemcode"] = "REVIEW_REQUIRED"
    result.loc[manual, "matched_itemname"] = "REVIEW_REQUIRED"

    result["assisted_mode"] = MLDeploymentMode.ASSISTED.value
    result["assisted_run_id"] = result["run_id"]
    result["assisted_model_id"] = result["model_id"]
    result["assisted_calibrated_probability"] = result[
        "lightgbm_probability"
    ]
    result["assisted_decision"] = result["final_decision"]
    result["assisted_decision_reason"] = result[
        "final_decision_reason"
    ]
    result["assisted_safety_override_applied"] = result[
        "hard_conflict"
    ]
    return result


def select_competitor_eligible_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only explicitly accepted mappings for competitor discovery."""
    if "final_decision" in rows and "final_eligible_mapping" in rows:
        eligible = (
            rows["final_eligible_mapping"].fillna(False).astype(bool)
            & rows["final_decision"].isin(ELIGIBLE_FINAL_DECISIONS)
        )
        return rows.loc[eligible].copy()
    if "confidence_tier" in rows:
        return rows.loc[
            rows["confidence_tier"].eq("high (ml)")
        ].copy()
    return rows.iloc[0:0].copy()


def _failure_decisions(
    offers: pd.DataFrame,
    *,
    run_id: str,
    model_id: str | None,
    no_candidates: bool,
    error: str | None,
) -> pd.DataFrame:
    decision = (
        FinalMatchDecision.NO_CANDIDATE
        if no_candidates
        else FinalMatchDecision.MODEL_ERROR
    )
    records = []
    for _, offer in offers.iterrows():
        record = _base_record(
            offer, run_id=run_id, model_id=model_id
        )
        if "is_own" in offer and not _safe_bool(offer.get("is_own")):
            record.update(
                {
                    "final_decision": (
                        FinalMatchDecision.COMPETITOR_OFFER.value
                    ),
                    "decision_source": "OFFER_CLASSIFICATION",
                    "final_decision_reason": "NON_OWN_BRAND_OFFER",
                    "human_review_status": "NOT_APPLICABLE",
                }
            )
            records.append(record)
            continue
        record.update(
            {
                "final_decision": decision.value,
                "decision_source": (
                    "CANDIDATE_GENERATOR"
                    if no_candidates
                    else "MODEL_PACKAGE"
                ),
                "final_decision_reason": (
                    "NO_RETAINED_CANDIDATE"
                    if no_candidates
                    else "MODEL_ERROR_SAFE_FALLBACK"
                ),
                "pipeline_error": error or "",
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _statistics(
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    offers_processed: int,
    shadow_manifest: dict[str, Any],
    total_runtime: float,
    finalization_runtime: float,
) -> dict[str, Any]:
    counts = decisions.get(
        "final_decision", pd.Series(dtype="string")
    ).value_counts()
    agreement_rows = (
        candidates.drop_duplicates("offer_group_id")
        if not candidates.empty and "offer_group_id" in candidates
        else pd.DataFrame()
    )
    same_top = (
        agreement_rows["same_top_candidate"]
        .fillna(False)
        .astype(bool)
        if "same_top_candidate" in agreement_rows
        else pd.Series(dtype=bool)
    )
    llm_routed = int(
        agreement_rows.get(
            "routing_decision", pd.Series(dtype="string")
        ).eq("LLM_REVIEW").sum()
    )
    llm_accepts = int(
        counts.get(FinalMatchDecision.LLM_ACCEPT.value, 0)
    )
    shadow_stages = shadow_manifest.get(
        "stage_runtimes_seconds", {}
    )
    stage_runtimes = {
        str(key): float(value)
        for key, value in shadow_stages.items()
        if isinstance(value, (int, float))
    }
    stage_runtimes["final_decision_policy"] = float(
        finalization_runtime
    )
    return {
        "report_type": "UNIFIED_ASSISTED_INFERENCE_STATISTICS",
        "offers_processed": int(offers_processed),
        "terminal_decision_count": int(len(decisions)),
        "decision_coverage_complete": bool(
            len(decisions) == offers_processed
            and decisions.get(
                "offer_id", pd.Series(dtype="string")
            ).nunique()
            == offers_processed
        ),
        "candidates_generated": int(len(candidates)),
        "auto_accept_count": int(
            counts.get(FinalMatchDecision.AUTO_ACCEPT.value, 0)
        ),
        "llm_accept_count": llm_accepts,
        "manual_review_count": int(
            counts.get(FinalMatchDecision.MANUAL_REVIEW.value, 0)
        ),
        "no_match_count": int(
            counts.get(FinalMatchDecision.NO_MATCH.value, 0)
        ),
        "no_candidate_count": int(
            counts.get(FinalMatchDecision.NO_CANDIDATE.value, 0)
        ),
        "master_sku_not_found_count": int(
            counts.get(
                FinalMatchDecision.MASTER_SKU_NOT_FOUND.value, 0
            )
        ),
        "model_error_count": int(
            counts.get(FinalMatchDecision.MODEL_ERROR.value, 0)
        ),
        "competitor_offer_count": int(
            counts.get(FinalMatchDecision.COMPETITOR_OFFER.value, 0)
        ),
        "llm_calls": int(
            shadow_manifest.get("llm_review", {}).get(
                "provider_calls", 0
            )
        ),
        "llm_acceptance_rate": (
            float(llm_accepts / llm_routed) if llm_routed else None
        ),
        "hard_conflict_count": int(
            decisions.get(
                "hard_conflict",
                pd.Series(False, index=decisions.index),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "total_runtime_seconds": float(total_runtime),
        "stage_runtimes_seconds": stage_runtimes,
        "training_or_online_learning_performed": False,
    }


def run_unified_inference(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    config: PipelineConfig,
    model_directory: str | Path | None = None,
    run_id: str | None = None,
    persist_records: bool = True,
    source_path: str | Path | None = None,
    progress: InferenceProgressCallback | None = None,
) -> UnifiedInferenceResult:
    """Run all configured inference stages and apply only explicit policy."""
    if config.ml.mode is MLDeploymentMode.DISABLED:
        return UnifiedInferenceResult(
            status="DISABLED",
            rows=production_rows,
            decisions=pd.DataFrame(),
            candidates=pd.DataFrame(),
            run_id=None,
            statistics={},
            output_paths={},
            shadow_result=None,
        )

    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    effective_run_id = run_id or (
        f"unified-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    base_offers = production_rows.copy(deep=True).reset_index(drop=True)
    # Identity assignment and entity decomposition both scale with every
    # canonical offer and together dominate the start of this stage, so they
    # report as one "preparation" phase instead of leaving the bar frozen.
    total_base_offers = len(base_offers)
    identity_interval = max(1, min(5_000, total_base_offers // 40 or 1))

    def _report_preparation(completed: int, detail: str) -> None:
        if progress is None:
            return
        progress("preparation", completed, max(1, total_base_offers * 2), detail)

    _report_preparation(0, f"Preparing {total_base_offers:,} offers")
    group_ids: list[str] = []
    for position, (_, row) in enumerate(base_offers.iterrows()):
        group_ids.append(stable_offer_group_id(row, position))
        if (position + 1) % identity_interval == 0:
            _report_preparation(
                position + 1,
                f"Assigned offer identities to {position + 1:,} of "
                f"{total_base_offers:,} offers",
            )
    base_offers["offer_group_id"] = group_ids

    offers = _expand_inference_entities(
        base_offers,
        progress=(
            lambda completed, total: _report_preparation(
                total_base_offers + completed,
                f"Prepared commercial entities for {completed:,} of "
                f"{total:,} own-brand offers",
            )
        )
        if progress is not None
        else None,
    )
    if offers["offer_group_id"].duplicated().any():
        raise ValueError(
            "Unified inference requires unique offer_group_id values"
        )

    shadow_config = replace(
        config.shadow_mode,
        enabled=True,
        model_id=config.ml.model_id,
        package_reference=None,
    )
    effective_config = replace(
        config,
        shadow_mode=shadow_config,
        agreement=replace(
            config.agreement,
            lightgbm_auto_accept_threshold=(
                config.ml.auto_accept_threshold
            ),
        ),
    )
    candidates = pd.DataFrame()
    shadow_manifest: dict[str, Any] = {}
    own_mask = (
        offers["is_own"].fillna(False).astype(bool)
        if "is_own" in offers
        else pd.Series(True, index=offers.index)
    )
    if not own_mask.any():
        shadow_result = None
        shadow_manifest = {
            "stage_runtimes_seconds": {},
        }
        finalization_started = time.perf_counter()
        decisions = finalize_unified_decisions(
            offers,
            candidates,
            run_id=effective_run_id,
            model_id=config.ml.model_id,
        )
        finalization_runtime = (
            time.perf_counter() - finalization_started
        )
        status = "COMPLETED_NO_OWN_BRAND_OFFERS"
        error = None
    else:
        shadow_result = run_shadow_observation_non_blocking(
            offers.copy(deep=True),
            product_master.copy(deep=True),
            config=effective_config,
            model_directory=model_directory,
            shadow_run_id=effective_run_id,
            progress=progress,
        )

    if (
        shadow_result is not None
        and shadow_result.status == "COMPLETED_OBSERVATIONAL_ONLY"
    ):
        candidates = pd.read_parquet(
            shadow_result.output_paths["shadow_predictions_parquet"]
        )
        manifest_path = shadow_result.output_paths.get("run_manifest")
        if manifest_path is not None and manifest_path.is_file():
            shadow_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        finalization_started = time.perf_counter()
        decisions = finalize_unified_decisions(
            offers,
            candidates,
            run_id=effective_run_id,
            model_id=config.ml.model_id,
        )
        finalization_runtime = (
            time.perf_counter() - finalization_started
        )
        status = (
            "COMPLETED_ASSISTED"
            if config.ml.mode is MLDeploymentMode.ASSISTED
            else "COMPLETED_SHADOW_OBSERVATIONAL_ONLY"
        )
        error = None
    elif shadow_result is not None:
        error = shadow_result.error
        no_candidates = bool(
            error and "no evaluable candidate rows" in error.lower()
        )
        finalization_started = time.perf_counter()
        decisions = _failure_decisions(
            offers,
            run_id=effective_run_id,
            model_id=config.ml.model_id,
            no_candidates=no_candidates,
            error=error,
        )
        finalization_runtime = (
            time.perf_counter() - finalization_started
        )
        status = (
            "NO_CANDIDATES_SAFE_REVIEW"
            if no_candidates
            else "MODEL_ERROR_SAFE_FALLBACK"
        )

    if progress is not None:
        progress(
            "final_decisions",
            len(decisions),
            len(offers),
            f"Finalized {len(decisions):,} offer decisions",
        )
    validate_terminal_decisions(offers, candidates, decisions)
    applied_rows = (
        _attach_final_decisions(offers, decisions)
        if config.ml.mode is MLDeploymentMode.ASSISTED
        else production_rows
    )
    total_runtime = time.perf_counter() - started
    statistics = _statistics(
        decisions,
        candidates,
        offers_processed=len(offers),
        shadow_manifest=shadow_manifest,
        total_runtime=total_runtime,
        finalization_runtime=finalization_runtime,
    )
    statistics.update(
        {
            "run_id": effective_run_id,
            "mode": config.ml.mode.value,
            "model_id": config.ml.model_id,
            "llm_review_enabled": config.llm_review.enabled,
            "shadow_status": (
                shadow_result.status
                if shadow_result is not None
                else "NOT_APPLICABLE_NO_OWN_BRAND_OFFERS"
            ),
            "production_application_enabled": (
                config.ml.mode is MLDeploymentMode.ASSISTED
            ),
        }
    )
    if (
        shadow_result is not None
        and shadow_result.original_exception is not None
    ):
        statistics["original_exception"] = (
            shadow_result.original_exception
        )

    output_paths = (
        dict(shadow_result.output_paths)
        if shadow_result is not None
        else {}
    )
    if persist_records:
        run_directory = (
            config.output.output_dir
            / "unified_inference"
            / effective_run_id
        )
        decisions_path = run_directory / "unified_decisions.csv"
        statistics_path = run_directory / "run_statistics.json"
        try:
            _atomic_csv(decisions, decisions_path)
            _atomic_json(statistics, statistics_path)
            output_paths.update(
                {
                    "unified_decisions": decisions_path,
                    "unified_statistics": statistics_path,
                }
            )
        except Exception:
            LOGGER.exception(
                "Unified audit persistence failed; production control continues"
            )

    result = UnifiedInferenceResult(
        status=status,
        rows=applied_rows,
        decisions=decisions,
        candidates=candidates,
        run_id=effective_run_id,
        statistics=statistics,
        output_paths=output_paths,
        shadow_result=shadow_result,
        error=error,
        original_exception=(
            shadow_result.original_exception
            if shadow_result is not None
            else None
        ),
        exception=(
            shadow_result.exception
            if shadow_result is not None
            else None
        ),
    )
    from sku_mapping.learning.observer import (
        observe_unified_result_non_blocking,
    )

    if persist_records:
        observe_unified_result_non_blocking(
            result, config=config, source_path=source_path
        )
    return result


def run_unified_inference_non_blocking(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    config: PipelineConfig,
    model_directory: str | Path | None = None,
    run_id: str | None = None,
    persist_records: bool = True,
    source_path: str | Path | None = None,
    progress: InferenceProgressCallback | None = None,
) -> UnifiedInferenceResult:
    """Convert an unexpected orchestration failure to explicit MODEL_ERROR."""
    wrapper_started = time.perf_counter()
    if config.ml.mode is MLDeploymentMode.DISABLED:
        return run_unified_inference(
            production_rows,
            product_master,
            config=config,
            model_directory=model_directory,
            run_id=run_id,
            persist_records=persist_records,
            source_path=source_path,
            progress=progress,
        )
    before_rows = production_rows.copy(deep=True)
    before_master = product_master.copy(deep=True)
    try:
        return run_unified_inference(
            production_rows,
            product_master,
            config=config,
            model_directory=model_directory,
            run_id=run_id,
            persist_records=persist_records,
            source_path=source_path,
            progress=progress,
        )
    except Exception as error:
        LOGGER.exception(
            "Unified assisted inference failed closed to MODEL_ERROR"
        )
        pd.testing.assert_frame_equal(
            production_rows, before_rows, check_exact=True
        )
        pd.testing.assert_frame_equal(
            product_master, before_master, check_exact=True
        )
        now = datetime.now(timezone.utc)
        original_exception = capture_exception_details(error)
        effective_run_id = run_id or (
            f"unified-failed-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        offers = production_rows.copy(deep=True).reset_index(drop=True)
        offers["offer_group_id"] = [
            stable_offer_group_id(row, position)
            for position, (_, row) in enumerate(offers.iterrows())
        ]
        if offers["offer_group_id"].duplicated().any():
            offers["offer_group_id"] = [
                f"{value}::{position}"
                for position, value in enumerate(
                    offers["offer_group_id"].astype(str)
                )
            ]
        decisions = _failure_decisions(
            offers,
            run_id=effective_run_id,
            model_id=config.ml.model_id,
            no_candidates=False,
            error=f"{type(error).__name__}: {error}",
        )
        rows = (
            _attach_final_decisions(offers, decisions)
            if config.ml.mode is MLDeploymentMode.ASSISTED
            else production_rows
        )
        statistics = _statistics(
            decisions,
            pd.DataFrame(),
            offers_processed=len(offers),
            shadow_manifest={},
            total_runtime=time.perf_counter() - wrapper_started,
            finalization_runtime=0.0,
        )
        statistics.update(
            {
                "run_id": effective_run_id,
                "mode": config.ml.mode.value,
                "model_id": config.ml.model_id,
                "llm_review_enabled": config.llm_review.enabled,
                "shadow_status": "FAILED_BEFORE_COMPLETION",
                "production_application_enabled": (
                    config.ml.mode is MLDeploymentMode.ASSISTED
                ),
                "pipeline_error": f"{type(error).__name__}: {error}",
                "original_exception": original_exception,
            }
        )
        output_paths: dict[str, Path] = {}
        if persist_records:
            run_directory = (
                config.output.output_dir
                / "unified_inference"
                / effective_run_id
            )
            decisions_path = run_directory / "unified_decisions.csv"
            statistics_path = run_directory / "run_statistics.json"
            try:
                _atomic_csv(decisions, decisions_path)
                _atomic_json(statistics, statistics_path)
                output_paths = {
                    "unified_decisions": decisions_path,
                    "unified_statistics": statistics_path,
                }
            except Exception:
                LOGGER.exception(
                    "Could not persist unified failure artifacts"
                )
        result = UnifiedInferenceResult(
            status="MODEL_ERROR_SAFE_FALLBACK",
            rows=rows,
            decisions=decisions,
            candidates=pd.DataFrame(),
            run_id=effective_run_id,
            statistics=statistics,
            output_paths=output_paths,
            shadow_result=None,
            error=f"{type(error).__name__}: {error}",
            original_exception=original_exception,
            exception=error,
        )
        from sku_mapping.learning.observer import (
            observe_unified_result_non_blocking,
        )

        if persist_records:
            observe_unified_result_non_blocking(
                result, config=config, source_path=source_path
            )
        return result
