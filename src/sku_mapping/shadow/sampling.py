"""Deterministic, hard-case-aware sampling for human review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DIFFICULT_FAMILY_PATTERN = r"(?:cube|frank|fries|stick|paratha|tortilla)"
SAMPLING_STRATA = (
    "highest_shadow_probabilities",
    "near_diagnostic_threshold",
    "production_shadow_disagreement",
    "mixed_protein_offers",
    "pack_format_conflicts",
    "missing_measurements",
    "underrepresented_families",
    "candidate_rank_greater_than_one",
    "difficult_families",
    "random_production_baseline",
    "likely_hard_negatives",
)


@dataclass(frozen=True)
class ReviewSample:
    """Deduplicated offer selection and auditable stratum accounting."""

    offers: pd.DataFrame
    report: dict[str, Any]


def _rank_one(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(
            ["offer_group_id", "candidate_rank", "master_itemcode"],
            kind="stable",
        )
        .drop_duplicates("offer_group_id", keep="first")
        .copy()
    )


def sample_offers_for_review(
    candidates: pd.DataFrame,
    *,
    counts_by_stratum: dict[str, int],
    random_seed: int,
    diagnostic_threshold: float,
) -> ReviewSample:
    """Select varied offers reproducibly while preserving every reason."""
    required = {
        "offer_group_id",
        "candidate_rank",
        "master_itemcode",
        "calibrated_probability",
        "shadow_decision_bucket",
        "existing_production_decision",
        "product_family",
        "is_mixed_protein_offer",
        "pack_conflict",
        "feature_missingness_count",
    }
    missing = sorted(required - set(candidates))
    if missing:
        raise ValueError(f"Shadow sampling is missing columns: {missing}")
    if candidates.empty:
        return ReviewSample(
            offers=pd.DataFrame(
                columns=["offer_group_id", "selection_reasons"]
            ),
            report={
                "random_seed": random_seed,
                "diagnostic_threshold": diagnostic_threshold,
                "strata": {},
                "unique_offers_selected": 0,
            },
        )

    frame = candidates.copy()
    frame["offer_group_id"] = frame["offer_group_id"].astype(str)
    frame["product_family"] = (
        frame["product_family"].astype("string").fillna("<missing>")
    )
    best = _rank_one(frame)
    production_accepted = best["existing_production_decision"].astype(str).isin(
        {"AUTO_MATCH", "ACCEPTED", "high (ml)"}
    )
    shadow_high = best["shadow_decision_bucket"].eq("SHADOW_HIGH_SCORE")
    best["production_shadow_disagreement"] = production_accepted.ne(shadow_high)

    family_counts = best["product_family"].value_counts()
    best["family_frequency"] = best["product_family"].map(family_counts)
    alternatives = frame[frame["candidate_rank"] > 1].copy()
    alternative_best = (
        alternatives.sort_values(
            ["calibrated_probability", "candidate_rank", "offer_group_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        .drop_duplicates("offer_group_id")
    )

    ordered: dict[str, pd.DataFrame] = {
        "highest_shadow_probabilities": best.sort_values(
            ["calibrated_probability", "offer_group_id"],
            ascending=[False, True],
            kind="stable",
        ),
        "near_diagnostic_threshold": best.assign(
            _distance=(
                best["calibrated_probability"] - diagnostic_threshold
            ).abs()
        ).sort_values(
            ["_distance", "offer_group_id"], kind="stable"
        ),
        "production_shadow_disagreement": best[
            best["production_shadow_disagreement"]
        ].sort_values(
            ["calibrated_probability", "offer_group_id"],
            ascending=[False, True],
            kind="stable",
        ),
        "mixed_protein_offers": best[
            pd.to_numeric(
                best["is_mixed_protein_offer"], errors="coerce"
            ).fillna(0)
            > 0
        ].sort_values(
            ["calibrated_probability", "offer_group_id"],
            ascending=[False, True],
            kind="stable",
        ),
        "pack_format_conflicts": best[
            best["pack_conflict"].fillna(False).astype(bool)
        ].sort_values(
            ["calibrated_probability", "offer_group_id"],
            ascending=[False, True],
            kind="stable",
        ),
        "missing_measurements": best[
            best["feature_missingness_count"] > 0
        ].sort_values(
            [
                "feature_missingness_count",
                "calibrated_probability",
                "offer_group_id",
            ],
            ascending=[False, False, True],
            kind="stable",
        ),
        "underrepresented_families": best.sort_values(
            ["family_frequency", "product_family", "offer_group_id"],
            kind="stable",
        ),
        "candidate_rank_greater_than_one": alternative_best,
        "difficult_families": best[
            best["product_family"].str.contains(
                DIFFICULT_FAMILY_PATTERN, case=False, na=False
            )
        ].sort_values(
            ["product_family", "calibrated_probability", "offer_group_id"],
            ascending=[True, False, True],
            kind="stable",
        ),
        "likely_hard_negatives": alternative_best[
            (
                alternative_best["shadow_decision_bucket"].eq(
                    "SHADOW_HIGH_SCORE"
                )
            )
            | alternative_best["pack_conflict"].fillna(False).astype(bool)
        ].sort_values(
            ["calibrated_probability", "candidate_rank", "offer_group_id"],
            ascending=[False, True, True],
            kind="stable",
        ),
    }
    baseline = best.sort_values("offer_group_id", kind="stable")
    rng = np.random.default_rng(random_seed)
    ordered["random_production_baseline"] = baseline.iloc[
        rng.permutation(len(baseline))
    ]

    reasons: dict[str, set[str]] = {}
    stratum_report: dict[str, Any] = {}
    for stratum in SAMPLING_STRATA:
        requested = int(counts_by_stratum.get(stratum, 0))
        eligible = ordered[stratum]
        selected_ids = (
            eligible["offer_group_id"].astype(str).head(requested).tolist()
            if requested > 0
            else []
        )
        for offer_group_id in selected_ids:
            reasons.setdefault(offer_group_id, set()).add(stratum)
        stratum_report[stratum] = {
            "requested": requested,
            "eligible_offers": int(eligible["offer_group_id"].nunique()),
            "selected_offers": len(selected_ids),
            "selected_offer_group_ids": selected_ids,
        }

    selected = best[best["offer_group_id"].isin(reasons)].copy()
    selected["selection_reasons"] = selected["offer_group_id"].map(
        lambda value: "|".join(sorted(reasons[str(value)]))
    )
    selected = selected.sort_values("offer_group_id", kind="stable").reset_index(
        drop=True
    )
    return ReviewSample(
        offers=selected,
        report={
            "random_seed": random_seed,
            "diagnostic_threshold": diagnostic_threshold,
            "strata": stratum_report,
            "unique_offers_selected": int(len(selected)),
            "deduplication": (
                "Offer groups are selected once; all qualifying stratum "
                "reasons are retained."
            ),
        },
    )
