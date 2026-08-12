"""Evaluate a packaged classifier against a labelled processed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sku_mapping.ml.evaluator import evaluate_binary_classifier
from sku_mapping.ml.model_package import (
    load_model_package,
    validate_feature_frame_columns,
)
from sku_mapping.paths import PROJECT_ROOT
from sku_mapping.shadow.challenge import assert_not_sealed_challenge_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test_split.parquet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_not_sealed_challenge_input(args.features)
    package = load_model_package(args.model)
    frame = pd.read_parquet(args.features)
    missing = [
        column
        for column in [*package["feature_columns"], "pair_label"]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Evaluation table is missing columns: {missing}")
    feature_frame = frame.loc[:, package["feature_columns"]]
    validate_feature_frame_columns(feature_frame.columns)
    probabilities = package["model"].predict_proba(feature_frame)[:, 1]
    metrics = evaluate_binary_classifier(
        frame,
        probabilities,
        threshold=package["auto_match_threshold"],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
