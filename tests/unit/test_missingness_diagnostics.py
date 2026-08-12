"""Missingness diagnostics retain NaNs and expose split drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.diagnostics import build_missingness_diagnostics


class _Predictor:
    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[MODEL_FEATURE_COLUMNS[0]].fillna(-10.0).to_numpy(float)
        return 1.0 / (1.0 + np.exp(-values))


def _frame(values: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            feature: values
            for feature in MODEL_FEATURE_COLUMNS
        }
    )
    frame["pair_label"] = [0, 1]
    frame["provenance_category"] = ["synthetic", "human_audited"]
    frame["product_class_offer"] = ["nugget", "burger"]
    return frame


def test_missingness_report_includes_drift_groups_and_sensitivity() -> None:
    train = _frame([np.nan, 1.0])
    validation = _frame([0.0, 1.0])
    calibration = _frame([np.nan, 2.0])
    report = build_missingness_diagnostics(
        {
            "train": train,
            "validation": validation,
            "calibration": calibration,
        },
        predictor=_Predictor(),  # type: ignore[arg-type]
        auto_threshold=0.8,
        manual_threshold=0.4,
        feature_importance_gain={feature: 0.0 for feature in MODEL_FEATURE_COLUMNS},
    )
    assert report["nan_retained"] is True
    assert report["unknown_measurements_replaced_with_zero"] is False
    assert report["feature_missingness_by_split"]["train"][
        MODEL_FEATURE_COLUMNS[0]
    ] == 0.5
    assert report["missingness_distribution_drift"][
        MODEL_FEATURE_COLUMNS[0]
    ]["max_absolute_rate_difference"] == 0.5
    assert "synthetic" in report["feature_missingness_by_provenance"]
    assert "nugget" in report["feature_missingness_by_product_family"]
    assert "decision_changes" in report[
        "calibration_decision_sensitivity_when_feature_is_forced_missing"
    ][MODEL_FEATURE_COLUMNS[0]]
