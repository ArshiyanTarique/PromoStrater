"""End-to-end leakage-safe v3 shadow training isolation test."""

from __future__ import annotations

import hashlib
import json
import platform

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml import shadow_trainer
from sku_mapping.ml.leakage import build_leakage_groups
from sku_mapping.ml.leakage_split import (
    LeakageSplitConfig,
    create_leakage_safe_splits,
)
from sku_mapping.ml.safety_thresholds import ThresholdEvidencePolicy
from sku_mapping.ml.shadow_trainer import (
    ShadowTrainingConfig,
    run_shadow_training_pipeline,
)


def _major_minor(value: str) -> str:
    return ".".join(value.split(".")[:2])


def _fixture() -> pd.DataFrame:
    rows = []
    for group_index in range(60):
        for label in (0, 1):
            row = {
                "record_id": f"record-{group_index}-{label}",
                "offer_group_id": f"group-{group_index}",
                "offer_text": f"Human offer family-{group_index} label-{label}",
                "master_itemcode": f"sku-{group_index}-{label}",
                "pair_label": label,
                "source_dataset": "HUMAN_AUDIT",
                "label_provenance": "human_audited",
                "product_class_offer": "nugget",
            }
            row.update(
                {
                    feature: float(group_index * 5 + label)
                    for feature in MODEL_FEATURE_COLUMNS
                }
            )
            row["word_similarity"] = 15.0 if label == 0 else 95.0
            row["character_similarity"] = 20.0 if label == 0 else 90.0
            rows.append(row)
    return pd.DataFrame(rows)


def _config() -> ShadowTrainingConfig:
    return ShadowTrainingConfig(
        random_seed=7,
        split=LeakageSplitConfig(random_seed=7, candidate_splits=32),
        calibration_method="sigmoid",
        isotonic_min_rows=1000,
        isotonic_min_positive_rows=200,
        threshold_policy=ThresholdEvidencePolicy(
            target_auto_precision=0.95,
            min_auto_match_rows=2,
            max_auto_match_false_positives=0,
            min_auto_precision_lower_bound=0.25,
            precision_confidence_level=0.95,
            min_calibration_rows=10,
            min_calibration_positive_rows=4,
            auto_threshold_min=0.5,
            auto_threshold_max=0.99,
            auto_threshold_steps=20,
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


def test_shadow_pipeline_separates_validation_and_calibration(
    tmp_path, monkeypatch
) -> None:
    feature_path = tmp_path / "training_features.parquet"
    frame = _fixture()
    frame.to_parquet(feature_path, index=False)
    processed_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_hashes": {"gold_pairs": "b" * 64},
                "output_hashes": {
                    "training_features_parquet": processed_hash
                },
            }
        ),
        encoding="utf-8",
    )

    calibration_record_ids: set[str] = set()
    threshold_row_counts: list[int] = []
    original_calibrate = shadow_trainer.fit_probability_calibrator
    original_tune = shadow_trainer.tune_shadow_thresholds

    def recording_calibrate(model, calibration_frame, **kwargs):
        calibration_record_ids.update(calibration_frame["record_id"])
        return original_calibrate(model, calibration_frame, **kwargs)

    def recording_tune(labels, probabilities, policy):
        threshold_row_counts.append(len(labels))
        return original_tune(labels, probabilities, policy)

    monkeypatch.setattr(
        shadow_trainer, "fit_probability_calibrator", recording_calibrate
    )
    monkeypatch.setattr(shadow_trainer, "tune_shadow_thresholds", recording_tune)
    result = run_shadow_training_pipeline(
        feature_path,
        config=_config(),
        processed_dir=tmp_path / "processed",
        model_registry_dir=tmp_path / "models" / "registry",
        metadata_dir=tmp_path / "models" / "metadata",
        registry_path=tmp_path / "models" / "model_registry.json",
        reports_dir=tmp_path / "reports",
        manifest_path=manifest_path,
    )

    expected_calibration_ids = set(result.splits.calibration["record_id"])
    validation_ids = set(result.splits.validation["record_id"])
    assert calibration_record_ids == expected_calibration_ids
    assert calibration_record_ids.isdisjoint(validation_ids)
    assert threshold_row_counts == [len(result.splits.calibration)]
    assert result.package["metrics"]["model_selection"][
        "calibration_rows_used_for_model_selection"
    ] == 0
    assert result.package["deployment_status"] == "SHADOW_MODE_ONLY"
    assert result.package["automatic_production_matching_approved"] is False
    assert result.package["approved_auto_match_threshold"] is None
    assert result.output_paths["model"].is_file()
    assert result.output_paths["registry"].is_file()
    assert result.output_paths["completion_report"].is_file()


def test_shadow_model_fit_is_deterministic() -> None:
    config = _config()
    grouped = build_leakage_groups(_fixture())
    splits = create_leakage_safe_splits(grouped.frame, config.split)

    first_model, first_results, _, _ = shadow_trainer._fit_model_candidates(
        splits, config
    )
    second_model, second_results, _, _ = shadow_trainer._fit_model_candidates(
        splits, config
    )
    validation_features = splits.validation.loc[:, MODEL_FEATURE_COLUMNS]

    np.testing.assert_array_equal(
        first_model.predict_proba(validation_features),
        second_model.predict_proba(validation_features),
    )
    assert first_results == second_results
