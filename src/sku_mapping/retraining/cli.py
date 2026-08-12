"""Command-line workflow for snapshots, challengers, comparison, and activation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from sku_mapping.config import PipelineConfig, load_config
from sku_mapping.learning.store import LearningStore
from sku_mapping.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from sku_mapping.retraining.comparison import compare_models
from sku_mapping.retraining.registry import ControlledModelRegistry
from sku_mapping.retraining.snapshot import build_training_snapshot
from sku_mapping.retraining.trainer import train_challenger


def _registry(config: PipelineConfig, store: LearningStore) -> ControlledModelRegistry:
    return ControlledModelRegistry(
        registry_path=config.shadow_mode.registry_path,
        model_directory=config.shadow_mode.registry_path.parent / "registry",
        metadata_directory=config.shadow_mode.registry_path.parent / "metadata",
        store=store,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit Phase 7C offline retraining controls"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("build-training-snapshot")
    _common(snapshot)
    snapshot.add_argument(
        "--baseline",
        default=PROJECT_ROOT / "data" / "processed" / "training_features.parquet",
    )
    snapshot.add_argument(
        "--challenge-manifest", action="append", default=[]
    )
    snapshot.add_argument("--include-silver", action="store_true")
    snapshot.add_argument("--minimum-gold-override", type=int)
    snapshot.add_argument("--override-reason")

    train = subparsers.add_parser("train-challenger")
    _common(train)
    train.add_argument("--snapshot-manifest", required=True)
    train.add_argument("--champion-model-id", required=True)

    compare = subparsers.add_parser("compare-models")
    _common(compare)
    compare.add_argument("--champion-model-id", required=True)
    compare.add_argument("--challenger-package", required=True)
    compare.add_argument("--challenger-metadata", required=True)
    compare.add_argument("--snapshot-manifest", required=True)
    compare.add_argument("--evaluation-path", action="append", default=[])
    compare.add_argument(
        "--skip-regression-tests",
        action="store_true",
        help="Record missing test evidence; this forces promotion failure.",
    )
    compare.add_argument(
        "--no-register",
        action="store_true",
        help="Evaluate without registering a policy-passing challenger.",
    )

    activate = subparsers.add_parser("activate-model")
    _common(activate)
    activate.add_argument("--model-id")
    activate.add_argument("--expected-current-model-id")
    activate.add_argument("--actor", required=True)
    activate.add_argument("--reason", required=True)
    activate.add_argument("--rollback", action="store_true")
    return parser


def _regression_tests() -> bool:
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(PROJECT_ROOT / "tests" / "unit" / "test_model_package.py"),
        str(PROJECT_ROOT / "tests" / "unit" / "test_shadow_model_package.py"),
        str(
            PROJECT_ROOT
            / "tests"
            / "unit"
            / "test_unified_inference_policy.py"
        ),
        "-q",
    ]
    completed = subprocess.run(command, check=False, cwd=PROJECT_ROOT)
    return completed.returncode == 0


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    store = LearningStore(config.learning_store.database_path)
    registry = _registry(config, store)

    if args.command == "build-training-snapshot":
        result = build_training_snapshot(
            store=store,
            baseline_path=args.baseline,
            config=config,
            challenge_manifest_paths=args.challenge_manifest,
            include_silver=(
                True if args.include_silver else config.retraining.include_silver
            ),
            minimum_gold_override=args.minimum_gold_override,
            override_reason=args.override_reason,
        )
        _print(
            {
                "dataset_id": result.dataset_id,
                "snapshot_manifest": result.manifest_path,
                "training_rows": result.manifest["row_count"],
                "recent_gold_evaluation_rows": result.manifest[
                    "evaluation_row_count"
                ],
                "retraining_performed": False,
            }
        )
        return 0

    if args.command == "train-challenger":
        result = train_challenger(
            snapshot_manifest_path=args.snapshot_manifest,
            champion_model_id=args.champion_model_id,
            config=config,
            store=store,
        )
        _print(
            {
                "model_id": result.model_id,
                "package_path": result.package_path,
                "package_sha256": result.package_sha256,
                "active_model_changed": False,
                "evaluation_status": "PENDING",
            }
        )
        return 0

    if args.command == "compare-models":
        tests_passed = (
            False if args.skip_regression_tests else _regression_tests()
        )
        result = compare_models(
            champion_model_id=args.champion_model_id,
            challenger_package_path=args.challenger_package,
            challenger_metadata_path=args.challenger_metadata,
            snapshot_manifest_path=args.snapshot_manifest,
            registry=registry,
            store=store,
            config=config,
            additional_evaluation_paths=args.evaluation_path,
            regression_tests_passed=tests_passed,
            register_if_passed=not args.no_register,
        )
        _print(
            {
                "comparison_id": result.comparison_id,
                "decision": result.decision,
                "promotion_policy_passed": result.promotion_policy_passed,
                "report_path": result.report_path,
                "registered_package_path": result.registered_package_path,
                "activation_performed": False,
            }
        )
        return 0 if result.promotion_policy_passed else 2

    if args.rollback:
        target = registry.rollback(actor=args.actor, reason=args.reason)
        _print({"action": "ROLLBACK", "active_assisted_model_id": target})
        return 0
    if not args.model_id or not args.expected_current_model_id:
        raise ValueError(
            "activate-model requires --model-id and "
            "--expected-current-model-id unless --rollback is used"
        )
    previous = registry.activate(
        model_id=args.model_id,
        expected_current_model_id=args.expected_current_model_id,
        actor=args.actor,
        reason=args.reason,
    )
    _print(
        {
            "action": "ACTIVATE",
            "previous_model_id": previous,
            "active_assisted_model_id": args.model_id,
        }
    )
    return 0
