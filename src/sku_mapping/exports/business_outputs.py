"""Business-facing SKU mapping and competitor output orchestration.

The unified inference decision table is an internal lifecycle artifact.  This
module deliberately builds the two business outputs from that lifecycle
without exposing competitor-brand decisions as SKU mappings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pandas as pd

from sku_mapping.competitors.discovery import (
    CompetitorDiscoveryResult,
    discover_competitors,
)
from sku_mapping.config import CompetitorConfig

SKU_MAPPING_COLUMNS = (
    "source_offer_id",
    "source_offer_name",
    "source_product",
    "source_brand",
    "source_variant",
    "source_pack_size",
    "source_retailer",
    "source_flyer",
    "source_offer_price",
    "source_regular_price",
    "matched_master_sku",
    "matched_master_name",
    "matched_master_description",
    "mapping_score",
    "mapping_status",
    "mapping_reason",
    "requires_human_review",
    "run_id",
    "entity_id",
    "entity_index",
    "entity_count",
    "entity_text",
    "entity_type",
    "conjunction_type",
    "attribute_inheritance_flags",
    "entity_parse_confidence",
)

_APPROVED_STATUSES = frozenset(
    {
        "AUTO_ACCEPTED",
        "LLM_ACCEPTED",
        "HUMAN_APPROVED",
        "HUMAN_CORRECTED",
    }
)
_EXPLICIT_REJECTION_STATUSES = frozenset({"HUMAN_REJECTED"})
_SENTINEL_SKUS = frozenset(
    {
        "",
        "N/A",
        "N/A - COMPETITOR BRAND ROW",
        "NAN",
        "NO_MATCH",
        "NONE",
        "REVIEW_REQUIRED",
    }
)
_STATUS_ALIASES = {
    "AUTO_ACCEPT": "AUTO_ACCEPTED",
    "AUTO_MATCH": "AUTO_ACCEPTED",
    "LLM_ACCEPT": "LLM_ACCEPTED",
}


@dataclass(frozen=True)
class BusinessOutputResult:
    """The two business outputs, internal competitor audit, and exact counts."""

    sku_mapping: pd.DataFrame
    competitor_export: pd.DataFrame
    competitor_long_format: pd.DataFrame
    competitor_long_format_path: Path | None
    diagnostics: dict[str, object]


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_present(record: dict[str, Any], *columns: str) -> object:
    for column in columns:
        if column not in record:
            continue
        value = record[column]
        if _safe_text(value):
            return value
    return ""


def _normalized_sku(value: object) -> str:
    text = _safe_text(value)
    return "" if text.upper() in _SENTINEL_SKUS else text


def _mapping_status(record: dict[str, Any]) -> str:
    raw = _safe_text(
        _first_present(
            record,
            "mapping_status",
            "final_decision",
            "ml_decision",
        )
    ).upper()
    if not raw:
        return "NO_CANDIDATE"
    return _STATUS_ALIASES.get(raw, raw)


def _mapping_reason(
    record: dict[str, Any],
    *,
    status: str,
    has_candidate: bool,
) -> str:
    reason = _safe_text(
        _first_present(
            record,
            "mapping_reason",
            "final_decision_reason",
            "assisted_decision_reason",
            "decision_reason",
        )
    )
    if reason:
        return reason
    if status == "NO_CANDIDATE":
        return "NO_RETAINED_CANDIDATE"
    if status == "MASTER_SKU_NOT_FOUND":
        return "SELECTED_MASTER_SKU_NOT_IN_PRODUCT_MASTER"
    if status == "MANUAL_REVIEW" and has_candidate:
        return "BEST_CANDIDATE_REQUIRES_HUMAN_REVIEW"
    if status in _APPROVED_STATUSES and has_candidate:
        return "SELECTED_MASTER_SKU_ACCEPTED"
    if not has_candidate:
        return "NO_VALID_MASTER_CANDIDATE"
    return "BEST_AVAILABLE_MASTER_CANDIDATE"


def _requires_human_review(status: str) -> bool:
    return (
        status not in _APPROVED_STATUSES
        and status not in _EXPLICIT_REJECTION_STATUSES
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_sku_mapping_export(
    prepared_offers: pd.DataFrame,
    product_master: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    run_id: str,
) -> pd.DataFrame:
    """Build one SKU mapping row per canonical own-brand dump offer.

    A final approved/corrected SKU takes precedence.  Otherwise the best
    retained proposal remains visible, including for manual-review rows.
    Competitor-brand lifecycle decisions never enter this output.
    """

    if not _safe_text(run_id):
        raise ValueError("run_id is required for business exports")
    _require_columns(
        prepared_offers,
        {
            "offer_group_id",
            "is_own",
            "Offer Name",
            "Product",
            "Brand Name",
            "Variant",
            "Base Packsize",
            "Retailer Name",
            "Flyer Name",
            "Offer Price",
            "Regular Price",
        },
        label="Prepared ClickFlyer offers",
    )
    _require_columns(
        product_master,
        {"Itemcode", "Itemname", "Item Description"},
        label="Product Master",
    )
    if not decisions.empty:
        _require_columns(
            decisions,
            {"offer_id"},
            label="Lifecycle decisions",
        )

    own = prepared_offers[
        prepared_offers["is_own"].fillna(False).astype(bool)
    ].drop_duplicates("offer_group_id", keep="first")
    if own["offer_group_id"].map(_safe_text).eq("").any():
        raise ValueError("Canonical own-brand offers contain an empty identity")
    if not decisions.empty and decisions["offer_id"].map(_safe_text).duplicated().any():
        raise ValueError("Lifecycle decisions contain duplicate offer_id values")

    decision_lookup = {
        _safe_text(record["offer_id"]): record
        for record in decisions.to_dict(orient="records")
    }
    master = product_master.copy()
    master["_business_itemcode"] = master["Itemcode"].map(_normalized_sku)
    master = master[
        master["_business_itemcode"].ne("")
    ].drop_duplicates("_business_itemcode", keep="first")
    master_lookup = {
        _safe_text(record["_business_itemcode"]): record
        for record in master.to_dict(orient="records")
    }

    rows: list[dict[str, object]] = []
    for offer in own.to_dict(orient="records"):
        offer_id = _safe_text(offer["offer_group_id"])
        decision = decision_lookup.get(offer_id, {})
        status = _mapping_status(decision)
        final_sku = _normalized_sku(
            _first_present(
                decision,
                "final_master_sku",
                "matched_master_sku",
            )
        )
        proposed_sku = _normalized_sku(
            _first_present(
                decision,
                "proposed_master_sku",
                "suggested_itemcode",
            )
        )
        selected_sku = final_sku or proposed_sku
        if status in _EXPLICIT_REJECTION_STATUSES:
            selected_sku = ""

        selected_master = master_lookup.get(selected_sku)
        if selected_sku and selected_master is None:
            selected_sku = ""
            status = "MASTER_SKU_NOT_FOUND"
        has_candidate = bool(selected_sku)
        reason = _mapping_reason(
            decision,
            status=status,
            has_candidate=has_candidate,
        )
        score = _safe_score(
            _first_present(
                decision,
                "mapping_score",
                "lightgbm_probability",
                "assisted_calibrated_probability",
                "ml_probability",
            )
        )

        rows.append(
            {
                "source_offer_id": _safe_text(
                    offer.get("source_offer_id", offer_id)
                ),
                "source_offer_name": _safe_text(
                    offer.get("source_offer_text", offer["Offer Name"])
                ),
                "source_product": offer["Product"],
                "source_brand": offer["Brand Name"],
                "source_variant": offer["Variant"],
                "source_pack_size": offer["Base Packsize"],
                "source_retailer": offer["Retailer Name"],
                "source_flyer": offer["Flyer Name"],
                "source_offer_price": offer["Offer Price"],
                "source_regular_price": offer["Regular Price"],
                "matched_master_sku": selected_sku,
                "matched_master_name": (
                    selected_master["Itemname"]
                    if selected_master is not None
                    else ""
                ),
                "matched_master_description": (
                    selected_master["Item Description"]
                    if selected_master is not None
                    else ""
                ),
                "mapping_score": score,
                "mapping_status": status,
                "mapping_reason": reason,
                "requires_human_review": _requires_human_review(status),
                "run_id": run_id,
                "entity_id": _safe_text(
                    offer.get("entity_id", offer_id)
                ),
                "entity_index": int(
                    offer.get("entity_index", 1) or 1
                ),
                "entity_count": int(
                    offer.get("entity_count", 1) or 1
                ),
                "entity_text": _safe_text(
                    offer.get("entity_text", offer["Offer Name"])
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
                "entity_parse_confidence": _safe_score(
                    offer.get("entity_parse_confidence")
                ),
            }
        )

    export = pd.DataFrame(rows, columns=SKU_MAPPING_COLUMNS)
    if export["entity_id"].duplicated().any():
        raise ValueError("SKU mapping export contains duplicate entity IDs")
    if len(export) != len(own):
        raise ValueError(
            "SKU mapping row count does not equal canonical own-brand count"
        )
    return export


def build_business_outputs(
    prepared_offers: pd.DataFrame,
    product_master: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    competitor_config: CompetitorConfig,
    run_id: str,
    competitor_audit_path: str | Path | None = None,
    competitor_progress: (
        Callable[[int, int, int, int, str], None] | None
    ) = None,
    stage_progress: Callable[[str, int, int, str], None] | None = None,
    competitor_offers: pd.DataFrame | None = None,
    competitor_reranker: Any | None = None,
) -> BusinessOutputResult:
    """Build the exact own-offer → master-SKU → competitor business flow.

    ``competitor_offers`` is forwarded to competitor discovery so a caller that
    has collapsed ``prepared_offers`` to one row per offer identity can still
    supply the variant-level competitor pool.  The own-brand SKU mapping is
    always built from ``prepared_offers`` and is unaffected.
    """

    sku_mapping = build_sku_mapping_export(
        prepared_offers,
        product_master,
        decisions,
        run_id=run_id,
    )
    if stage_progress is not None:
        stage_progress(
            "sku_mapping",
            len(sku_mapping),
            len(sku_mapping),
            f"Built {len(sku_mapping):,} SKU mapping rows",
        )
    competitors: CompetitorDiscoveryResult = discover_competitors(
        prepared_offers,
        product_master,
        sku_mapping,
        config=competitor_config,
        run_id=run_id,
        audit_path=competitor_audit_path,
        progress=competitor_progress,
        competitor_offers=competitor_offers,
        reranker=competitor_reranker,
    )
    canonical_offer_count = int(
        prepared_offers["offer_group_id"].nunique(dropna=True)
    )
    own_offer_count = int(
        sku_mapping["source_offer_id"].nunique(dropna=True)
    )
    populated = sku_mapping["matched_master_sku"].map(_safe_text).ne("")
    diagnostics: dict[str, object] = {
        "input_rows": int(len(prepared_offers)),
        "canonical_offer_count": canonical_offer_count,
        "own_brand_offer_count": own_offer_count,
        "competitor_offer_count": int(
            canonical_offer_count - own_offer_count
        ),
        "sku_mapping_row_count": int(len(sku_mapping)),
        "commercial_entity_count": int(len(sku_mapping)),
        "multi_entity_source_offer_count": int(
            sku_mapping.loc[
                sku_mapping["entity_count"].fillna(1).astype(int).gt(1),
                "source_offer_id",
            ].nunique()
        ),
        "sku_mapping_populated_master_sku_count": int(populated.sum()),
        "sku_mapping_manual_review_count": int(
            sku_mapping["mapping_status"].eq("MANUAL_REVIEW").sum()
        ),
        "sku_mapping_no_candidate_count": int(
            sku_mapping["mapping_status"].isin(
                {"NO_CANDIDATE", "MASTER_SKU_NOT_FOUND"}
            ).sum()
        ),
        "distinct_matched_master_sku_count": int(
            sku_mapping.loc[populated, "matched_master_sku"].nunique()
        ),
        "competitor_csv_row_count": int(len(competitors.export)),
        **competitors.diagnostics,
    }
    return BusinessOutputResult(
        sku_mapping=sku_mapping,
        competitor_export=competitors.export,
        competitor_long_format=competitors.long_format,
        competitor_long_format_path=competitors.long_format_path,
        diagnostics=diagnostics,
    )
