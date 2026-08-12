"""Evaluation reports for binary SKU candidate classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(metric, y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(metric(y_true, probabilities))


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    negative_count = tn + fp
    positive_count = tp + fn
    return {
        "row_count": int(y_true.size),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "roc_auc": _safe_auc(roc_auc_score, y_true, probabilities),
        "pr_auc": _safe_auc(average_precision_score, y_true, probabilities),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "false_positive_rate": float(fp / negative_count) if negative_count else 0.0,
        "false_negative_rate": float(fn / positive_count) if positive_count else 0.0,
    }


def _calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> dict[str, Any]:
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=bins,
        strategy="uniform",
    )
    bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    expected_calibration_error = 0.0
    calibration_bins: list[dict[str, float | int]] = []
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        if not mask.any():
            continue
        mean_probability = float(probabilities[mask].mean())
        observed_rate = float(y_true[mask].mean())
        count = int(mask.sum())
        expected_calibration_error += (count / y_true.size) * abs(
            observed_rate - mean_probability
        )
        calibration_bins.append(
            {
                "bin": bin_index,
                "count": count,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
            }
        )
    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "expected_calibration_error": float(expected_calibration_error),
        "curve_fraction_positive": [float(value) for value in fraction_positive],
        "curve_mean_predicted_probability": [
            float(value) for value in mean_predicted
        ],
        "bins": calibration_bins,
    }


def _probability_distribution(probabilities: np.ndarray) -> dict[str, Any]:
    quantiles = np.quantile(
        probabilities,
        [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
    )
    counts, edges = np.histogram(probabilities, bins=np.linspace(0.0, 1.0, 11))
    return {
        "minimum": float(probabilities.min()),
        "maximum": float(probabilities.max()),
        "mean": float(probabilities.mean()),
        "standard_deviation": float(probabilities.std()),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1"),
                quantiles,
            )
        },
        "histogram": [
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(count),
            }
            for index, count in enumerate(counts)
        ],
    }


def _group_metrics(
    frame: pd.DataFrame,
    group_column: str,
    probability_column: str,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    if group_column not in frame.columns:
        return {}
    output: dict[str, dict[str, Any]] = {}
    groups = frame[group_column].astype("string").fillna("<missing>")
    for group_value, indices in groups.groupby(groups, sort=True).groups.items():
        subset = frame.loc[indices]
        output[str(group_value)] = _classification_metrics(
            subset["pair_label"].to_numpy(dtype=int),
            subset[probability_column].to_numpy(dtype=float),
            threshold,
        )
    return output


def evaluate_binary_classifier(
    frame: pd.DataFrame,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    calibration_bins: int = 10,
    product_family_columns: Sequence[str] = (
        "product_family",
        "product_class_offer",
    ),
) -> dict[str, Any]:
    """Evaluate probabilities without fitting or selecting a threshold."""
    if "pair_label" not in frame.columns:
        raise ValueError("Evaluation frame is missing required column: pair_label")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Evaluation threshold must be between 0 and 1")
    y_true = frame["pair_label"].to_numpy(dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if probability_array.shape != y_true.shape:
        raise ValueError("Probability count must equal evaluation row count")
    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must be finite")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    evaluated = frame.copy()
    evaluated["_probability"] = probability_array
    product_family_column = next(
        (column for column in product_family_columns if column in frame.columns),
        None,
    )
    return {
        **_classification_metrics(y_true, probability_array, threshold),
        "threshold": float(threshold),
        "calibration": _calibration_metrics(
            y_true, probability_array, calibration_bins
        ),
        "probability_distribution": _probability_distribution(probability_array),
        "by_source_dataset": _group_metrics(
            evaluated, "source_dataset", "_probability", threshold
        ),
        "by_label_provenance": _group_metrics(
            evaluated, "label_provenance", "_probability", threshold
        ),
        "product_family_column": product_family_column,
        "by_product_family": (
            _group_metrics(
                evaluated, product_family_column, "_probability", threshold
            )
            if product_family_column
            else {}
        ),
    }
