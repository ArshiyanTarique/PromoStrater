"""Business-aware tuning for AUTO_MATCH and MANUAL_REVIEW thresholds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Selected thresholds and the complete candidate analysis."""

    auto_match_threshold: float
    manual_review_threshold: float
    selected_metrics: dict[str, float | int]
    analysis: pd.DataFrame


def validate_threshold_order(
    auto_match_threshold: float,
    manual_review_threshold: float,
) -> None:
    """Fail when thresholds do not define three ordered decision regions."""
    if not (
        0.0
        <= float(manual_review_threshold)
        < float(auto_match_threshold)
        <= 1.0
    ):
        raise ValueError(
            "Thresholds must satisfy "
            "0 <= manual_review_threshold < auto_match_threshold <= 1"
        )


def _candidate_values(
    probabilities: np.ndarray,
    configured: Iterable[float] | None,
) -> list[float]:
    if configured is not None:
        values = configured
    else:
        grid = np.linspace(0.05, 0.99, 20)
        quantiles = np.quantile(probabilities, np.linspace(0.05, 0.95, 19))
        values = np.concatenate([grid, quantiles])
    return sorted(
        {
            round(float(value), 6)
            for value in values
            if 0.0 <= float(value) <= 1.0
        }
    )


def _pair_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    auto_threshold: float,
    manual_threshold: float,
) -> dict[str, float | int]:
    auto = probabilities >= auto_threshold
    manual = (probabilities >= manual_threshold) & ~auto
    no_match = probabilities < manual_threshold
    positives = y_true == 1
    auto_tp = int((auto & positives).sum())
    auto_fp = int((auto & ~positives).sum())
    auto_count = int(auto.sum())
    positive_count = int(positives.sum())
    row_count = int(y_true.size)
    return {
        "auto_match_threshold": float(auto_threshold),
        "manual_review_threshold": float(manual_threshold),
        "auto_match_precision": (
            float(auto_tp / auto_count) if auto_count else 0.0
        ),
        "auto_match_recall": (
            float(auto_tp / positive_count) if positive_count else 0.0
        ),
        "auto_match_false_positives": auto_fp,
        "auto_match_volume": auto_count,
        "accepted_coverage": float(auto_count / row_count),
        "manual_review_volume": int(manual.sum()),
        "manual_review_coverage": float(manual.sum() / row_count),
        "manual_review_positive_capture": int((manual & positives).sum()),
        "no_match_volume": int(no_match.sum()),
        "no_match_coverage": float(no_match.sum() / row_count),
    }


def tune_thresholds(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    auto_candidates: Iterable[float] | None = None,
    manual_candidates: Iterable[float] | None = None,
    target_auto_precision: float = 0.98,
) -> ThresholdTuningResult:
    """Evaluate threshold pairs and select a high-precision operating point.

    Selection first prefers candidates meeting the configured AUTO_MATCH
    precision target, then maximises AUTO_MATCH recall/coverage. If no
    candidate meets the target, the highest observed precision is preferred.
    Thresholds accepting zero rows are never selected when any non-empty
    AUTO_MATCH candidate exists.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Labels and probabilities must be equal-length vectors")
    if labels.size == 0:
        raise ValueError("Cannot tune thresholds on an empty validation set")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Threshold labels must contain only 0 and 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Threshold probabilities must be finite and within [0, 1]")
    if not 0.0 <= target_auto_precision <= 1.0:
        raise ValueError("target_auto_precision must be between 0 and 1")

    auto_values = _candidate_values(scores, auto_candidates)
    manual_values = _candidate_values(scores, manual_candidates)
    rows = [
        _pair_metrics(labels, scores, auto, manual)
        for auto in auto_values
        for manual in manual_values
        if auto > manual
    ]
    if not rows:
        raise ValueError("No valid threshold pairs satisfy auto > manual")
    analysis = pd.DataFrame(rows)
    nonempty = analysis[analysis["auto_match_volume"] > 0]
    selectable = nonempty if not nonempty.empty else analysis
    meeting_target = selectable[
        selectable["auto_match_precision"] >= target_auto_precision
    ]
    if not meeting_target.empty:
        selected_pool = meeting_target.sort_values(
            [
                "auto_match_recall",
                "accepted_coverage",
                "auto_match_false_positives",
                "manual_review_positive_capture",
                "manual_review_volume",
                "auto_match_threshold",
                "manual_review_threshold",
            ],
            ascending=[False, False, True, False, True, False, False],
            kind="stable",
        )
    else:
        selected_pool = selectable.sort_values(
            [
                "auto_match_precision",
                "auto_match_false_positives",
                "auto_match_recall",
                "accepted_coverage",
                "manual_review_positive_capture",
                "manual_review_volume",
                "auto_match_threshold",
                "manual_review_threshold",
            ],
            ascending=[False, True, False, False, False, True, False, False],
            kind="stable",
        )
    selected = selected_pool.iloc[0]
    auto_threshold = float(selected["auto_match_threshold"])
    manual_threshold = float(selected["manual_review_threshold"])
    validate_threshold_order(auto_threshold, manual_threshold)
    return ThresholdTuningResult(
        auto_match_threshold=auto_threshold,
        manual_review_threshold=manual_threshold,
        selected_metrics={
            column: (
                int(selected[column])
                if column.endswith("_volume")
                or column.endswith("_positives")
                or column.endswith("_capture")
                else float(selected[column])
            )
            for column in analysis.columns
        },
        analysis=analysis.sort_values(
            ["auto_match_threshold", "manual_review_threshold"]
        ).reset_index(drop=True),
    )
