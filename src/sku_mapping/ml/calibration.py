"""Separated raw-score calibration for shadow-mode model packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from sku_mapping.constants import MODEL_FEATURE_COLUMNS


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Calculate weighted absolute calibration error."""
    bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(error)


class SigmoidScoreCalibrator:
    """Platt-style logistic calibration over raw LightGBM margins."""

    method = "sigmoid"

    def __init__(self, random_seed: int) -> None:
        self.random_seed = random_seed
        self.estimator = LogisticRegression(
            solver="lbfgs",
            random_state=random_seed,
            max_iter=2_000,
        )

    def fit(
        self, raw_scores: np.ndarray, labels: np.ndarray
    ) -> "SigmoidScoreCalibrator":
        self.estimator.fit(np.asarray(raw_scores).reshape(-1, 1), labels)
        self.classes_ = self.estimator.classes_.copy()
        self.coef_ = self.estimator.coef_.copy()
        self.intercept_ = self.estimator.intercept_.copy()
        return self

    def predict_positive_probability(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(
            np.asarray(raw_scores).reshape(-1, 1)
        )[:, 1]


class IsotonicScoreCalibrator:
    """Monotonic non-parametric calibration over raw LightGBM margins."""

    method = "isotonic"

    def __init__(self) -> None:
        self.estimator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

    def fit(
        self, raw_scores: np.ndarray, labels: np.ndarray
    ) -> "IsotonicScoreCalibrator":
        self.estimator.fit(np.asarray(raw_scores), labels)
        self.classes_ = np.array([0, 1], dtype=int)
        self.X_thresholds_ = self.estimator.X_thresholds_.copy()
        self.y_thresholds_ = self.estimator.y_thresholds_.copy()
        return self

    def predict_positive_probability(self, raw_scores: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(np.asarray(raw_scores)), dtype=float)


@dataclass
class ShadowModelPredictor:
    """Package boundary exposing both raw and calibrated predictions."""

    model: Any
    calibration_model: SigmoidScoreCalibrator | IsotonicScoreCalibrator
    feature_columns: list[str]

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if list(frame.columns) != self.feature_columns:
            raise ValueError(
                "Prediction columns must exactly match packaged feature order"
            )
        base_present = [
            column
            for column in self.feature_columns
            if column in MODEL_FEATURE_COLUMNS
        ]
        if base_present != list(MODEL_FEATURE_COLUMNS):
            raise ValueError("Packaged predictor feature schema is incompatible")
        return frame

    def predict_raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        features = self._features(frame)
        best_iteration = getattr(self.model, "best_iteration_", None)
        return np.asarray(
            self.model.predict(
                features,
                raw_score=True,
                num_iteration=best_iteration,
            ),
            dtype=float,
        )

    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw_scores = self.predict_raw_score(frame)
        return self.calibration_model.predict_positive_probability(raw_scores)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = self.predict_calibrated_proba(frame)
        return np.column_stack([1.0 - positive, positive])


def _raw_probability(model: Any, feature_frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(feature_frame)[:, 1], dtype=float)


def choose_calibration_method(
    requested_method: str,
    *,
    row_count: int,
    positive_count: int,
    isotonic_min_rows: int,
    isotonic_min_positive_rows: int,
) -> str:
    """Choose calibration using a deterministic, pre-registered rule."""
    if requested_method in {"sigmoid", "isotonic"}:
        return requested_method
    if requested_method != "auto":
        raise ValueError("Calibration method must be auto, sigmoid, or isotonic")
    negative_count = row_count - positive_count
    if (
        row_count >= isotonic_min_rows
        and positive_count >= isotonic_min_positive_rows
        and negative_count >= isotonic_min_positive_rows
    ):
        return "isotonic"
    return "sigmoid"


def fit_probability_calibrator(
    model: Any,
    calibration_frame: pd.DataFrame,
    *,
    requested_method: str,
    isotonic_min_rows: int,
    isotonic_min_positive_rows: int,
    random_seed: int,
    feature_columns: list[str] | None = None,
) -> tuple[
    SigmoidScoreCalibrator | IsotonicScoreCalibrator,
    ShadowModelPredictor,
    dict[str, Any],
    pd.DataFrame,
]:
    """Fit calibration only on the designated calibration split.

    ``feature_columns`` defaults to the 19-column base contract; ranked
    packages pass their full 41-column list so calibration, the packaged
    predictor, and inference all share one schema.
    """
    if feature_columns is None:
        feature_columns = list(MODEL_FEATURE_COLUMNS)
    features = calibration_frame.loc[:, feature_columns]
    labels = calibration_frame["pair_label"].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires both binary classes")
    raw_scores = np.asarray(
        model.predict(
            features,
            raw_score=True,
            num_iteration=getattr(model, "best_iteration_", None),
        ),
        dtype=float,
    )
    method = choose_calibration_method(
        requested_method,
        row_count=len(calibration_frame),
        positive_count=int(labels.sum()),
        isotonic_min_rows=isotonic_min_rows,
        isotonic_min_positive_rows=isotonic_min_positive_rows,
    )
    calibration_model: SigmoidScoreCalibrator | IsotonicScoreCalibrator
    if method == "isotonic":
        calibration_model = IsotonicScoreCalibrator().fit(raw_scores, labels)
    else:
        calibration_model = SigmoidScoreCalibrator(random_seed).fit(
            raw_scores, labels
        )
    predictor = ShadowModelPredictor(
        model=model,
        calibration_model=calibration_model,
        feature_columns=list(feature_columns),
    )
    raw_probabilities = _raw_probability(model, features)
    calibrated_probabilities = predictor.predict_calibrated_proba(features)
    predictions = calibration_frame.copy()
    predictions["raw_model_score"] = raw_scores
    predictions["raw_model_probability"] = raw_probabilities
    predictions["calibrated_probability"] = calibrated_probabilities
    report = {
        "split_used": "calibration",
        "method_requested": requested_method,
        "method_selected": method,
        "selection_rule": {
            "isotonic_min_rows": isotonic_min_rows,
            "isotonic_min_positive_rows_per_class": isotonic_min_positive_rows,
            "fallback": "sigmoid",
        },
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "negative_rows": int(len(labels) - labels.sum()),
        "raw_probability_metrics": {
            "brier_score": float(brier_score_loss(labels, raw_probabilities)),
            "log_loss": float(log_loss(labels, raw_probabilities, labels=[0, 1])),
            "expected_calibration_error": expected_calibration_error(
                labels, raw_probabilities
            ),
        },
        "calibrated_probability_metrics": {
            "brier_score": float(
                brier_score_loss(labels, calibrated_probabilities)
            ),
            "log_loss": float(
                log_loss(labels, calibrated_probabilities, labels=[0, 1])
            ),
            "expected_calibration_error": expected_calibration_error(
                labels, calibrated_probabilities
            ),
        },
        "not_a_final_performance_estimate": True,
    }
    return calibration_model, predictor, report, predictions
