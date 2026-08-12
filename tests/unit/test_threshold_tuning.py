"""Tests for ordered, validation-driven decision threshold selection."""

from __future__ import annotations

import pytest

from sku_mapping.ml.threshold_tuning import (
    tune_thresholds,
    validate_threshold_order,
)


def test_threshold_tuning_prioritises_high_precision_auto_match() -> None:
    labels = [1, 1, 0, 1, 0, 0]
    probabilities = [0.99, 0.95, 0.80, 0.70, 0.30, 0.10]
    result = tune_thresholds(
        labels,
        probabilities,
        auto_candidates=[0.7, 0.9],
        manual_candidates=[0.2, 0.6],
        target_auto_precision=1.0,
    )
    assert result.auto_match_threshold == 0.9
    assert result.manual_review_threshold < result.auto_match_threshold
    assert result.selected_metrics["auto_match_false_positives"] == 0
    assert set(
        [
            "auto_match_precision",
            "auto_match_recall",
            "accepted_coverage",
            "manual_review_volume",
            "no_match_volume",
        ]
    ).issubset(result.analysis.columns)


@pytest.mark.parametrize("auto,manual", [(0.7, 0.7), (0.6, 0.7), (1.1, 0.5)])
def test_threshold_order_validation_rejects_invalid_pairs(
    auto: float, manual: float
) -> None:
    with pytest.raises(ValueError, match="manual_review_threshold"):
        validate_threshold_order(auto, manual)
