"""Evidence-gated calibration threshold tests."""

from __future__ import annotations

import numpy as np

from sku_mapping.ml.safety_thresholds import (
    ThresholdEvidencePolicy,
    one_sided_precision_lower_bound,
    tune_shadow_thresholds,
)


def _policy(min_rows: int = 5) -> ThresholdEvidencePolicy:
    return ThresholdEvidencePolicy(
        target_auto_precision=0.99,
        min_auto_match_rows=min_rows,
        max_auto_match_false_positives=0,
        min_auto_precision_lower_bound=0.50,
        precision_confidence_level=0.95,
        min_calibration_rows=10,
        min_calibration_positive_rows=5,
        auto_threshold_min=0.5,
        auto_threshold_max=0.99,
        auto_threshold_steps=20,
        manual_threshold_min=0.1,
        manual_threshold_max=0.4,
        manual_threshold_steps=4,
    )


def test_threshold_evidence_can_withhold_auto_match_approval() -> None:
    labels = np.array([1] * 5 + [0] * 15)
    probabilities = np.array([0.99] * 4 + [0.8] + [0.98] + [0.1] * 14)
    result = tune_shadow_thresholds(labels, probabilities, _policy(min_rows=10))
    assert result.evidence_requirements_met is False
    assert result.approved_auto_match_threshold is None
    assert result.auto_match_threshold > result.manual_review_threshold


def test_threshold_report_includes_one_sided_precision_bound() -> None:
    labels = np.array([1] * 10 + [0] * 10)
    probabilities = np.array([0.99] * 10 + [0.01] * 10)
    result = tune_shadow_thresholds(labels, probabilities, _policy())
    assert "auto_match_precision_lower_bound" in result.analysis
    assert result.selected_metrics["auto_match_false_positives"] == 0
    assert one_sided_precision_lower_bound(100, 0, 0.95) > 0.95
