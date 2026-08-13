"""Turn a competitor run into a human-sized review queue.

A run admits hundreds of thousands of competitor relationships. Nobody will
review that, and a queue nobody works is not ground truth. So the queue takes
the top few per master SKU in the order the run itself ranked them - which is
the model's order when re-ranking is on and the rule order when it is not.

That bias is deliberate and worth stating: labelling the top of each list
measures precision at the top, which is what the business reads. It does not
measure recall, and a model calibrated only on these labels would inherit the
same blind spot. Sampling further down the list is the obvious next step once
the top is no longer the unknown.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def _json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def competitor_review_proposals(
    export: pd.DataFrame,
    *,
    run_id: str,
    per_target: int,
    model_id: str | None = None,
    ranking_source: str = "rules",
) -> list[dict[str, Any]]:
    """Build staging rows from the competitor aggregate.

    Reads the aggregate rather than the long-format audit because the audit is
    spooled to disk on a real run and can reach millions of rows, while the
    aggregate already carries the supported competitors per SKU in rank order.

    Per-pair model margins are NOT carried here - they live in the audit CSV.
    The queue records which ordering produced the shortlist, not the score,
    because a reviewer judging "is this a rival" should not be anchored by a
    number that is explicitly not a competitor probability.
    """
    if per_target <= 0 or export is None or export.empty:
        return []
    proposals: list[dict[str, Any]] = []
    for _, row in export.iterrows():
        master_sku = str(row.get("master_sku") or "").strip()
        if not master_sku:
            continue
        offer_ids = _json_list(row.get("competitor_offer_ids"))
        if not offer_ids:
            continue
        names = _json_list(row.get("competitor_offer_names"))
        brands = _json_list(row.get("competitor_brand_names"))
        status = str(row.get("competitor_status") or "UNKNOWN")
        reason = row.get("competitor_reason")
        for position, offer_id in enumerate(offer_ids[:per_target]):
            offer_id = str(offer_id or "").strip()
            if not offer_id:
                continue
            proposals.append(
                {
                    "run_id": run_id,
                    "master_sku": master_sku,
                    "competitor_offer_id": offer_id,
                    "competitor_offer_name": (
                        names[position] if position < len(names) else None
                    ),
                    "competitor_brand": (
                        brands[position] if position < len(brands) else None
                    ),
                    "master_name": row.get("master_name"),
                    "proposed_status": status,
                    "proposed_reason": reason,
                    "lightgbm_rank": position + 1,
                    "ranking_source": ranking_source,
                    "model_id": model_id,
                }
            )
    return proposals


def stage_competitor_review_queue(
    store: Any,
    export: pd.DataFrame,
    *,
    run_id: str,
    per_target: int,
    model_id: str | None = None,
    ranking_source: str = "rules",
) -> int:
    """Stage the queue, returning how many rows were written.

    Staging must never be able to fail a completed run: the outputs are
    already on disk and correct, and a review queue is an extra. A failure is
    logged by the caller and the run stands.
    """
    proposals = competitor_review_proposals(
        export,
        run_id=run_id,
        per_target=per_target,
        model_id=model_id,
        ranking_source=ranking_source,
    )
    if not proposals:
        return 0
    return int(store.stage_competitor_proposals(proposals))


def review_queue_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts a reviewer or a dashboard can show without re-querying."""
    summary: dict[str, int] = {"total": len(rows)}
    for row in rows:
        decision = str(row.get("decision") or "PENDING")
        summary[decision] = summary.get(decision, 0) + 1
    return summary
