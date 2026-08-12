"""Leakage-component-aware train/validation/calibration splitting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from sku_mapping.ml.leakage import SYNTHETIC_PROVENANCE_CATEGORIES

SPLIT_NAMES = ("train", "validation", "calibration")
ASSIGNMENT_COLUMNS = (
    "input_row_number",
    "record_id",
    "offer_group_id",
    "template_group_id",
    "feature_vector_hash",
    "leakage_group_id",
    "provenance_category",
    "split",
)


@dataclass(frozen=True)
class LeakageSplitConfig:
    """Pre-registered deterministic split proportions."""

    random_seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.15
    candidate_splits: int = 256

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.calibration_fraction,
        )
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("All split fractions must be between 0 and 1")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("Split fractions must sum to 1")
        if self.candidate_splits < 1:
            raise ValueError("candidate_splits must be positive")


@dataclass(frozen=True)
class LeakageSafeSplits:
    """Leakage-isolated train, validation, and calibration frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    assignments: pd.DataFrame
    assignment_sha256: str
    audit: dict[str, Any]


def _assignment_hash(assignments: pd.DataFrame) -> str:
    canonical = assignments.loc[:, ASSIGNMENT_COLUMNS].sort_values(
        "input_row_number", kind="stable"
    )
    payload = canonical.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _key_overlap(
    splits: dict[str, pd.DataFrame],
    column: str,
    *,
    relevant_templates_only: bool = False,
) -> dict[str, int]:
    values: dict[str, set[str]] = {}
    for split_name, split in splits.items():
        selected = split
        if relevant_templates_only:
            selected = selected[
                selected["provenance_category"].isin(
                    SYNTHETIC_PROVENANCE_CATEGORIES
                )
            ]
        values[split_name] = set(selected[column].astype(str))
    return {
        f"{left}_{right}": len(values[left] & values[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }


def assert_zero_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Assert zero overlap for every required leakage key."""
    checks = {
        "offer_group_id": _key_overlap(splits, "offer_group_id"),
        "template_group_id": _key_overlap(
            splits, "template_group_id", relevant_templates_only=True
        ),
        "feature_vector_hash": _key_overlap(splits, "feature_vector_hash"),
        "leakage_group_id": _key_overlap(splits, "leakage_group_id"),
    }
    failures = {
        key: pairs
        for key, pairs in checks.items()
        if any(count != 0 for count in pairs.values())
    }
    if failures:
        raise ValueError(f"Leakage-safe split overlap detected: {failures}")
    return checks


def _candidate_score(
    frames: dict[str, pd.DataFrame],
    config: LeakageSplitConfig,
    overall_positive_rate: float,
    total_rows: int,
) -> float:
    targets = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "calibration": config.calibration_fraction,
    }
    row_error = sum(
        abs(len(frames[name]) / total_rows - targets[name])
        for name in SPLIT_NAMES
    )
    class_error = sum(
        abs(float(frames[name]["pair_label"].mean()) - overall_positive_rate)
        for name in SPLIT_NAMES
    )
    return row_error * 10.0 + class_error


def create_leakage_safe_splits(
    augmented: pd.DataFrame,
    config: LeakageSplitConfig | None = None,
) -> LeakageSafeSplits:
    """Split indivisible leakage components into train/validation/calibration."""
    effective = config or LeakageSplitConfig()
    required = [
        "input_row_number",
        "offer_group_id",
        "pair_label",
        "template_group_id",
        "feature_vector_hash",
        "leakage_group_id",
        "provenance_category",
    ]
    missing = [column for column in required if column not in augmented.columns]
    if missing:
        raise ValueError(f"Leakage-safe splitting is missing columns: {missing}")
    if augmented.empty:
        raise ValueError("Cannot split an empty feature table")
    if augmented["input_row_number"].duplicated().any():
        raise ValueError("input_row_number must identify every row exactly once")
    labels = pd.to_numeric(augmented["pair_label"], errors="coerce")
    if labels.isna().any() or set(labels.astype(int).unique()) != {0, 1}:
        raise ValueError("Leakage-safe splitting requires both binary labels")
    if augmented["leakage_group_id"].nunique() < 3:
        raise ValueError("At least three leakage groups are required")

    groups = augmented["leakage_group_id"].astype(str)
    holdout_fraction = (
        effective.validation_fraction + effective.calibration_fraction
    )
    calibration_relative = effective.calibration_fraction / holdout_fraction
    first_splitter = GroupShuffleSplit(
        n_splits=effective.candidate_splits,
        test_size=holdout_fraction,
        random_state=effective.random_seed,
    )
    best: tuple[float, dict[str, pd.DataFrame]] | None = None
    overall_rate = float(labels.mean())
    for candidate_index, (train_indices, holdout_indices) in enumerate(
        first_splitter.split(augmented, labels, groups)
    ):
        holdout = augmented.iloc[holdout_indices]
        second_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=calibration_relative,
            random_state=effective.random_seed + 10_000 + candidate_index,
        )
        validation_relative_indices, calibration_relative_indices = next(
            second_splitter.split(
                holdout,
                holdout["pair_label"],
                holdout["leakage_group_id"].astype(str),
            )
        )
        frames = {
            "train": augmented.iloc[train_indices].copy(),
            "validation": holdout.iloc[validation_relative_indices].copy(),
            "calibration": holdout.iloc[calibration_relative_indices].copy(),
        }
        if any(
            frame.empty or set(frame["pair_label"].astype(int).unique()) != {0, 1}
            for frame in frames.values()
        ):
            continue
        score = _candidate_score(frames, effective, overall_rate, len(augmented))
        if best is None or score < best[0]:
            best = (score, frames)
    if best is None:
        raise ValueError(
            "Unable to create leakage-safe splits containing both classes"
        )

    frames = best[1]
    overlap_checks = assert_zero_leakage(frames)
    for frame in frames.values():
        frame.sort_values("input_row_number", inplace=True, kind="stable")
        frame.reset_index(drop=True, inplace=True)

    assignment_parts = []
    for split_name in SPLIT_NAMES:
        part = frames[split_name].copy()
        if "record_id" not in part:
            part["record_id"] = ""
        part["split"] = split_name
        assignment_parts.append(part.loc[:, ASSIGNMENT_COLUMNS])
    assignments = pd.concat(assignment_parts, ignore_index=True).sort_values(
        "input_row_number", kind="stable"
    )
    if len(assignments) != len(augmented):
        raise AssertionError("Split assignment silently dropped rows")
    if assignments["input_row_number"].nunique() != len(augmented):
        raise AssertionError("Every input row must have exactly one split")
    assignment_sha256 = _assignment_hash(assignments)

    audit = {
        "method": "deterministic_group_shuffle_search",
        "random_seed": effective.random_seed,
        "target_fractions": {
            "train": effective.train_fraction,
            "validation": effective.validation_fraction,
            "calibration": effective.calibration_fraction,
        },
        "rows_total": int(len(augmented)),
        "rows_assigned": int(len(assignments)),
        "rows_dropped": 0,
        "assignment_sha256": assignment_sha256,
        "overlap_checks": overlap_checks,
        "splits": {
            name: {
                "rows": int(len(frames[name])),
                "row_fraction": float(len(frames[name]) / len(augmented)),
                "positive_rows": int(frames[name]["pair_label"].sum()),
                "negative_rows": int(
                    len(frames[name]) - frames[name]["pair_label"].sum()
                ),
                "positive_rate": float(frames[name]["pair_label"].mean()),
                "leakage_groups": int(
                    frames[name]["leakage_group_id"].nunique()
                ),
                "offer_groups": int(frames[name]["offer_group_id"].nunique()),
            }
            for name in SPLIT_NAMES
        },
    }
    return LeakageSafeSplits(
        train=frames["train"],
        validation=frames["validation"],
        calibration=frames["calibration"],
        assignments=assignments.reset_index(drop=True),
        assignment_sha256=assignment_sha256,
        audit=audit,
    )
