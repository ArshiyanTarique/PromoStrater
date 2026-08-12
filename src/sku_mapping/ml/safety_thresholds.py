"""Pre-registered threshold evidence policy for shadow-mode calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

from sku_mapping.ml.threshold_tuning import validate_threshold_order


@dataclass(frozen=True)
class ThresholdEvidencePolicy:
    """Minimum calibration evidence required for an approved threshold."""

    target_auto_precision: float
    min_auto_match_rows: int
    max_auto_match_false_positives: int
    min_auto_precision_lower_bound: float
    precision_confidence_level: float
    min_calibration_rows: int
    min_calibration_positive_rows: int
    auto_threshold_min: float
    auto_threshold_max: float
    auto_threshold_steps: int
    manual_threshold_min: float
    manual_threshold_max: float
    manual_threshold_steps: int


@dataclass(frozen=True)
class SafetyThresholdResult:
    """Shadow candidate thresholds and their approval evidence."""

    auto_match_threshold: float
    manual_review_threshold: float
    approved_auto_match_threshold: float | None
    evidence_requirements_met: bool
    selected_metrics: dict[str, Any]
    analysis: pd.DataFrame
    policy: dict[str, Any]


def one_sided_precision_lower_bound(
    true_positives: int,
    false_positives: int,
    confidence_level: float,
) -> float:
    """Clopper-Pearson one-sided lower confidence bound for precision."""
    if true_positives <= 0:
        return 0.0
    alpha = 1.0 - confidence_level
    return float(beta.ppf(alpha, true_positives, false_positives + 1))


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    auto_threshold: float,
    manual_threshold: float,
    confidence_level: float,
) -> dict[str, Any]:
    auto = probabilities >= auto_threshold
    review = (probabilities >= manual_threshold) & ~auto
    no_match = probabilities < manual_threshold
    positives = labels == 1
    true_positives = int((auto & positives).sum())
    false_positives = int((auto & ~positives).sum())
    auto_rows = int(auto.sum())
    positive_rows = int(positives.sum())
    precision = float(true_positives / auto_rows) if auto_rows else 0.0
    return {
        "auto_match_threshold": float(auto_threshold),
        "manual_review_threshold": float(manual_threshold),
        "auto_match_rows": auto_rows,
        "auto_match_true_positives": true_positives,
        "auto_match_false_positives": false_positives,
        "auto_match_precision": precision,
        "auto_match_precision_lower_bound": one_sided_precision_lower_bound(
            true_positives, false_positives, confidence_level
        ),
        "auto_match_recall": (
            float(true_positives / positive_rows) if positive_rows else 0.0
        ),
        "accepted_coverage": float(auto_rows / len(labels)),
        "manual_review_rows": int(review.sum()),
        "manual_review_positive_rows": int((review & positives).sum()),
        "manual_review_coverage": float(review.mean()),
        "no_match_rows": int(no_match.sum()),
        "no_match_coverage": float(no_match.mean()),
    }


def tune_shadow_thresholds(
    labels: np.ndarray,
    calibrated_probabilities: np.ndarray,
    policy: ThresholdEvidencePolicy,
) -> SafetyThresholdResult:
    """Tune only on calibration and withhold approval if evidence is weak."""
    y = np.asarray(labels, dtype=int)
    probabilities = np.asarray(calibrated_probabilities, dtype=float)
    if y.ndim != 1 or probabilities.shape != y.shape or len(y) == 0:
        raise ValueError("Calibration labels and probabilities must align")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold calibration requires both binary classes")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Calibrated probabilities must be finite within [0, 1]")

    auto_values = np.linspace(
        policy.auto_threshold_min,
        policy.auto_threshold_max,
        policy.auto_threshold_steps,
    )
    manual_values = np.linspace(
        policy.manual_threshold_min,
        policy.manual_threshold_max,
        policy.manual_threshold_steps,
    )
    rows = [
        _metrics(y, probabilities, float(auto), float(manual), policy.precision_confidence_level)
        for auto in auto_values
        for manual in manual_values
        if auto > manual
    ]
    if not rows:
        raise ValueError("Configured grids contain no ordered threshold pairs")
    analysis = pd.DataFrame(rows)
    global_evidence = (
        len(y) >= policy.min_calibration_rows
        and int(y.sum()) >= policy.min_calibration_positive_rows
    )
    analysis["calibration_sample_requirement_met"] = global_evidence
    analysis["accepted_count_requirement_met"] = (
        analysis["auto_match_rows"] >= policy.min_auto_match_rows
    )
    analysis["false_positive_requirement_met"] = (
        analysis["auto_match_false_positives"]
        <= policy.max_auto_match_false_positives
    )
    analysis["precision_requirement_met"] = (
        analysis["auto_match_precision"] >= policy.target_auto_precision
    )
    analysis["precision_bound_requirement_met"] = (
        analysis["auto_match_precision_lower_bound"]
        >= policy.min_auto_precision_lower_bound
    )
    requirement_columns = [
        "calibration_sample_requirement_met",
        "accepted_count_requirement_met",
        "false_positive_requirement_met",
        "precision_requirement_met",
        "precision_bound_requirement_met",
    ]
    analysis["all_evidence_requirements_met"] = analysis[
        requirement_columns
    ].all(axis=1)

    eligible = analysis[analysis["all_evidence_requirements_met"]]
    if not eligible.empty:
        selected_pool = eligible.sort_values(
            [
                "auto_match_recall",
                "auto_match_rows",
                "manual_review_positive_rows",
                "manual_review_rows",
                "auto_match_threshold",
                "manual_review_threshold",
            ],
            ascending=[False, False, False, True, False, False],
            kind="stable",
        )
        evidence_met = True
    else:
        nonempty = analysis[analysis["auto_match_rows"] > 0]
        selected_pool = (nonempty if not nonempty.empty else analysis).sort_values(
            [
                "auto_match_false_positives",
                "auto_match_precision_lower_bound",
                "auto_match_precision",
                "auto_match_rows",
                "manual_review_positive_rows",
                "manual_review_rows",
                "auto_match_threshold",
                "manual_review_threshold",
            ],
            ascending=[True, False, False, False, False, True, False, False],
            kind="stable",
        )
        evidence_met = False
    selected = selected_pool.iloc[0]
    auto_threshold = float(selected["auto_match_threshold"])
    manual_threshold = float(selected["manual_review_threshold"])
    validate_threshold_order(auto_threshold, manual_threshold)
    return SafetyThresholdResult(
        auto_match_threshold=auto_threshold,
        manual_review_threshold=manual_threshold,
        approved_auto_match_threshold=auto_threshold if evidence_met else None,
        evidence_requirements_met=evidence_met,
        selected_metrics={
            column: (
                bool(selected[column])
                if column in requirement_columns
                or column == "all_evidence_requirements_met"
                else int(selected[column])
                if column.endswith("_rows")
                or column.endswith("_positives")
                else float(selected[column])
            )
            for column in analysis.columns
            if column not in {"calibration_sample_requirement_met"}
        }
        | {"calibration_sample_requirement_met": bool(global_evidence)},
        analysis=analysis.sort_values(
            ["auto_match_threshold", "manual_review_threshold"], kind="stable"
        ).reset_index(drop=True),
        policy={
            key: value for key, value in policy.__dict__.items()
        },
    )
