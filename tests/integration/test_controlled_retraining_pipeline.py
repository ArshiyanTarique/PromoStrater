"""Small-fixture challenger training and conservative rejection workflow."""

from __future__ import annotations

import platform
from dataclasses import replace
from pathlib import Path

import lightgbm
import pytest
import sklearn

from sku_mapping.ml.leakage_split import LeakageSplitConfig
from sku_mapping.ml.safety_thresholds import ThresholdEvidencePolicy
from sku_mapping.ml.shadow_trainer import ShadowTrainingConfig
from sku_mapping.retraining import trainer as trainer_module
from sku_mapping.retraining.comparison import PromotionPolicy, compare_models
from sku_mapping.retraining.snapshot import build_training_snapshot
from sku_mapping.retraining.trainer import train_challenger
from tests.retraining_fixtures import (
    CHAMPION_ID,
    phase7c_config,
    populated_learning_store,
    registry_with_champion,
    write_baseline,
)


def _major_minor(value: str) -> str:
    return ".".join(value.split(".")[:2])


def _small_training_config() -> ShadowTrainingConfig:
    return ShadowTrainingConfig(
        random_seed=17,
        split=LeakageSplitConfig(random_seed=17, candidate_splits=32),
        calibration_method="sigmoid",
        isotonic_min_rows=1000,
        isotonic_min_positive_rows=200,
        threshold_policy=ThresholdEvidencePolicy(
            target_auto_precision=0.8,
            min_auto_match_rows=1,
            max_auto_match_false_positives=10,
            min_auto_precision_lower_bound=0.0,
            precision_confidence_level=0.95,
            min_calibration_rows=4,
            min_calibration_positive_rows=1,
            auto_threshold_min=0.5,
            auto_threshold_max=0.99,
            auto_threshold_steps=10,
            manual_threshold_min=0.1,
            manual_threshold_max=0.4,
            manual_threshold_steps=4,
        ),
        provenance_weights={"human_audited": 1.0},
        compatibility_policy={
            "python_major_minor": _major_minor(platform.python_version()),
            "lightgbm_major_minor": _major_minor(lightgbm.__version__),
            "sklearn_major_minor": _major_minor(sklearn.__version__),
        },
        early_stopping_rounds=5,
        hyperparameter_candidates=(
            {
                "n_estimators": 30,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 2,
            },
        ),
    )


def test_challenger_training_does_not_activate_and_failed_comparison_rejects(
    tmp_path: Path,
) -> None:
    config = phase7c_config(tmp_path)
    store = populated_learning_store(tmp_path)
    snapshot = build_training_snapshot(
        store=store,
        baseline_path=write_baseline(tmp_path, groups=60),
        config=config,
        minimum_gold_override=4,
        override_reason="small integration fixture",
    )
    registry = registry_with_champion(tmp_path, store, active=True)
    active_before = registry.active_model_id()
    challenger = train_challenger(
        snapshot_manifest_path=snapshot.manifest_path,
        champion_model_id=CHAMPION_ID,
        config=config,
        store=store,
        training_config=_small_training_config(),
    )
    assert registry.active_model_id() == active_before
    comparison = compare_models(
        champion_model_id=CHAMPION_ID,
        challenger_package_path=challenger.package_path,
        challenger_metadata_path=challenger.metadata_path,
        snapshot_manifest_path=snapshot.manifest_path,
        registry=registry,
        store=store,
        config=config,
        policy=PromotionPolicy(
            minimum_evaluation_rows=2,
            minimum_subgroup_rows=1,
        ),
        regression_tests_passed=False,
    )
    assert comparison.promotion_policy_passed is False
    assert comparison.decision == "REJECTED"
    assert comparison.registered_package_path is None
    assert registry.active_model_id() == CHAMPION_ID
    model = next(
        item
        for item in store.list_model_versions()
        if item["model_id"] == challenger.model_id
    )
    assert model["status"] == "REJECTED"
    assert challenger.package_path.is_file()
    assert comparison.report_path.is_file()


def test_training_failure_leaves_current_champion_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = phase7c_config(tmp_path)
    store = populated_learning_store(tmp_path)
    snapshot = build_training_snapshot(
        store=store,
        baseline_path=write_baseline(tmp_path),
        config=config,
        minimum_gold_override=4,
        override_reason="training failure fixture",
    )
    registry = registry_with_champion(tmp_path, store, active=True)

    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate fit failure")

    monkeypatch.setattr(trainer_module, "_fit_model", fail_fit)
    with pytest.raises(RuntimeError, match="deliberate"):
        train_challenger(
            snapshot_manifest_path=snapshot.manifest_path,
            champion_model_id=CHAMPION_ID,
            config=config,
            store=store,
            training_config=_small_training_config(),
        )
    assert registry.active_model_id() == CHAMPION_ID
