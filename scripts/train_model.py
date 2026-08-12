"""Train and package the Phase 5 LightGBM SKU candidate classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sku_mapping.config import load_config
from sku_mapping.ml.trainer import TrainingConfig, run_training_pipeline
from sku_mapping.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "training_features.parquet",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--model-registry-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "registry",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "metadata",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=PROJECT_ROOT / "reports"
    )
    parser.add_argument(
        "--target-auto-precision",
        type=float,
        default=0.99,
        help="Minimum validation precision preferred for AUTO_MATCH.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_config = load_config(args.config)
    training_config = TrainingConfig(
        random_seed=pipeline_config.runtime.random_seed,
        target_auto_precision=args.target_auto_precision,
    )
    result = run_training_pipeline(
        args.features,
        processed_dir=args.processed_dir,
        model_registry_dir=args.model_registry_dir,
        metadata_dir=args.metadata_dir,
        reports_dir=args.reports_dir,
        config=training_config,
    )
    summary = {
        "split_method": result.splits.method,
        "dataset_sizes": {
            name: len(getattr(result.splits, name))
            for name in ("train", "validation", "test")
        },
        "selected_thresholds": {
            "auto_match_threshold": result.threshold_result.auto_match_threshold,
            "manual_review_threshold": (
                result.threshold_result.manual_review_threshold
            ),
        },
        "validation_pr_auc": result.validation_metrics["pr_auc"],
        "test_pr_auc": result.test_metrics["pr_auc"],
        "false_auto_matches": result.test_metrics["confusion_matrix"][
            "false_positive"
        ],
        "output_paths": {
            key: str(path) for key, path in result.output_paths.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
