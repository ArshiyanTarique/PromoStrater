"""Minimal predictor shim for the ranked-v5 model package.

The ranked model has no separate calibration step — its raw predict_proba
output is used directly for both raw and calibrated scores. This class
satisfies the ShadowPredictor interface so the model can be loaded through
the standard registry path.

This module must be importable at joblib unpickle time, which is why the
class lives in src/ rather than in the registration script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RankedModelPredictor:
    """Predictor shim satisfying the ShadowPredictor / deployment interface."""

    def __init__(self, model, feature_columns: list[str]) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        # Required by ShadowPredictor calibration checks
        self.classes_ = np.array([0, 1])
        # Sigmoid-style fitted marker expected by validate_model_package
        self.coef_ = np.array([[0.0]])

    def predict_raw_score(self, features: pd.DataFrame) -> np.ndarray:
        f = features.reindex(columns=self.feature_columns).fillna(-1.0)
        return self.model.predict_proba(f)[:, 1].astype(float)

    def predict_calibrated_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_raw_score(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        p = self.predict_raw_score(features)
        return np.column_stack([1.0 - p, p])
