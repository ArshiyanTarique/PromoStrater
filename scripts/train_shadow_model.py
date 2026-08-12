"""Train the leakage-safe experimental v3 shadow-mode package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sku_mapping.config import load_config
from sku_mapping.ml.shadow_trainer import (
    ShadowTrainingConfig,
    run_shadow_training_pipeline,
)
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
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "models" / "model_registry.json",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=PROJECT_ROOT / "reports"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_config = load_config(args.config)
    result = run_shadow_training_pipeline(
        args.features,
        config=ShadowTrainingConfig.from_pipeline_config(pipeline_config),
        processed_dir=args.processed_dir,
        model_registry_dir=args.model_registry_dir,
        metadata_dir=args.metadata_dir,
        registry_path=args.registry,
        reports_dir=args.reports_dir,
    )
    summary = {
        "model_path": str(result.output_paths["model"]),
        "deployment_status": result.package["deployment_status"],
        "automatic_production_matching_approved": False,
        "technical_checks_passed": result.technical_checks_passed,
        "calibration_method": result.package["calibration_method"],
        "threshold_evidence_requirements_met": (
            result.threshold_result.evidence_requirements_met
        ),
        "approved_auto_match_threshold": None,
        "shadow_auto_match_threshold": (
            result.threshold_result.auto_match_threshold
        ),
        "split_counts": {
            name: len(getattr(result.splits, name))
            for name in ("train", "validation", "calibration")
        },
        "split_assignment_sha256": result.splits.assignment_sha256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
