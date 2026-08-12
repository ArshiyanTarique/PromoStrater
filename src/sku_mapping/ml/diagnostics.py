"""Missingness, drift, and business-feature diagnostics for shadow models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml.calibration import ShadowModelPredictor


def _missing_rates(frame: pd.DataFrame) -> dict[str, float]:
    return {
        feature: float(frame[feature].isna().mean())
        for feature in MODEL_FEATURE_COLUMNS
    }


def _grouped_missingness(
    frame: pd.DataFrame,
    group_column: str,
) -> dict[str, dict[str, float]]:
    if group_column not in frame.columns:
        return {}
    groups = frame[group_column].astype("string").fillna("<missing>")
    return {
        str(group): _missing_rates(frame.loc[indices])
        for group, indices in groups.groupby(groups, sort=True).groups.items()
    }


def _decisions(
    probabilities: np.ndarray,
    auto_threshold: float,
    manual_threshold: float,
) -> np.ndarray:
    return np.where(
        probabilities >= auto_threshold,
        "AUTO_MATCH",
        np.where(probabilities >= manual_threshold, "MANUAL_REVIEW", "NO_MATCH"),
    )


def build_missingness_diagnostics(
    splits: dict[str, pd.DataFrame],
    *,
    predictor: ShadowModelPredictor,
    auto_threshold: float,
    manual_threshold: float,
    feature_importance_gain: dict[str, float],
) -> dict[str, Any]:
    """Build missingness distributions, drift, and decision-sensitivity report."""
    missing_by_split = {
        split_name: _missing_rates(frame)
        for split_name, frame in splits.items()
    }
    drift: dict[str, Any] = {}
    for feature in MODEL_FEATURE_COLUMNS:
        rates = {
            split_name: missing_by_split[split_name][feature]
            for split_name in splits
        }
        pairwise = {
            f"{left}_{right}": abs(rates[left] - rates[right])
            for index, left in enumerate(splits)
            for right in list(splits)[index + 1 :]
        }
        drift[feature] = {
            "rates": rates,
            "max_absolute_rate_difference": float(max(pairwise.values())),
            "pairwise_absolute_differences": pairwise,
        }

    combined = pd.concat(
        [frame.assign(_diagnostic_split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    calibration = splits["calibration"]
    calibration_features = calibration.loc[:, MODEL_FEATURE_COLUMNS]
    base_probabilities = predictor.predict_calibrated_proba(calibration_features)
    base_decisions = _decisions(
        base_probabilities, auto_threshold, manual_threshold
    )
    sensitivity: dict[str, Any] = {}
    for feature in MODEL_FEATURE_COLUMNS:
        stressed = calibration_features.copy()
        stressed[feature] = np.nan
        probabilities = predictor.predict_calibrated_proba(stressed)
        decisions = _decisions(probabilities, auto_threshold, manual_threshold)
        sensitivity[feature] = {
            "mean_absolute_probability_change": float(
                np.abs(probabilities - base_probabilities).mean()
            ),
            "maximum_absolute_probability_change": float(
                np.abs(probabilities - base_probabilities).max()
            ),
            "decision_changes": int((decisions != base_decisions).sum()),
            "auto_match_state_changes": int(
                (
                    (decisions == "AUTO_MATCH")
                    != (base_decisions == "AUTO_MATCH")
                ).sum()
            ),
        }

    investigated = {}
    for feature in (
        "pack_format_match",
        "bonus_weight_g",
        "is_mixed_protein_offer",
    ):
        series = combined[feature]
        investigated[feature] = {
            "gain_importance": float(feature_importance_gain.get(feature, 0.0)),
            "missing_rate": float(series.isna().mean()),
            "non_missing_unique_values": int(series.dropna().nunique()),
            "value_distribution": {
                str(key): int(value)
                for key, value in series.astype("string")
                .fillna("<missing>")
                .value_counts()
                .head(10)
                .items()
            },
            "possible_zero_importance_reasons": [
                "sparse or near-constant values",
                "redundancy with stronger similarity or measurement features",
                "limited labelled examples for the business condition",
            ],
            "automatically_removed": False,
        }

    family_column = next(
        (
            column
            for column in ("product_family", "product_class_offer")
            if column in combined.columns
        ),
        None,
    )
    return {
        "nan_retained": True,
        "unknown_measurements_replaced_with_zero": False,
        "feature_missingness_by_split": missing_by_split,
        "feature_missingness_by_label": _grouped_missingness(
            combined, "pair_label"
        ),
        "feature_missingness_by_provenance": _grouped_missingness(
            combined, "provenance_category"
        ),
        "product_family_column": family_column,
        "feature_missingness_by_product_family": (
            _grouped_missingness(combined, family_column)
            if family_column
            else {}
        ),
        "missingness_distribution_drift": drift,
        "calibration_decision_sensitivity_when_feature_is_forced_missing": sensitivity,
        "zero_importance_business_feature_investigation": investigated,
        "legacy_unresolved_business_rule": {
            "expression": "400+60 G X 20 Pkts",
            "status": "UNRESOLVED_BUSINESS_RULE_DECISION",
            "action_in_phase_5c": "documented_only_no_parser_change",
        },
    }
