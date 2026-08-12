"""Small-fixture end-to-end LightGBM training test."""

from __future__ import annotations

import pandas as pd

from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.ml import trainer
from sku_mapping.ml.trainer import TrainingConfig, run_training_pipeline


def _training_fixture() -> pd.DataFrame:
    rows = []
    split_by_group = {}
    for group_index in range(24):
        if group_index < 16:
            split = "train"
        elif group_index < 20:
            split = "validation"
        else:
            split = "test"
        split_by_group[f"g{group_index}"] = split
        for label in (0, 1):
            similarity = 20.0 + label * 70.0 + (group_index % 3)
            row = {
                "record_id": f"r{group_index}-{label}",
                "offer_group_id": f"g{group_index}",
                "pair_label": label,
                "recommended_split": split,
                "source_dataset": "fixture",
                "label_provenance": "human",
                "product_class_offer": "nugget",
            }
            row.update({feature: float(label) for feature in MODEL_FEATURE_COLUMNS})
            row["word_similarity"] = similarity
            row["character_similarity"] = similarity
            row["token_similarity"] = similarity
            rows.append(row)
    return pd.DataFrame(rows)


def test_successful_small_fixture_training_and_test_not_used_for_tuning(
    tmp_path, monkeypatch
) -> None:
    feature_path = tmp_path / "training_features.parquet"
    fixture = _training_fixture()
    fixture.to_parquet(feature_path, index=False)
    seen_labels: list[int] = []
    original_tune = trainer.tune_thresholds

    def recording_tune(y_true, probabilities, **kwargs):
        seen_labels.extend(list(y_true))
        assert len(y_true) == len(fixture[fixture["recommended_split"] == "validation"])
        return original_tune(y_true, probabilities, **kwargs)

    monkeypatch.setattr(trainer, "tune_thresholds", recording_tune)
    result = run_training_pipeline(
        feature_path,
        processed_dir=tmp_path / "processed",
        model_registry_dir=tmp_path / "registry",
        metadata_dir=tmp_path / "metadata",
        reports_dir=tmp_path / "reports",
        config=TrainingConfig(
            random_seed=7,
            early_stopping_rounds=5,
            hyperparameter_candidates=(
                {
                    "n_estimators": 30,
                    "learning_rate": 0.1,
                    "num_leaves": 7,
                    "min_child_samples": 2,
                },
            ),
        ),
    )
    assert seen_labels
    assert result.splits.method == "recommended_split"
    assert callable(result.model.predict_proba)
    assert result.threshold_result.auto_match_threshold > (
        result.threshold_result.manual_review_threshold
    )
    assert all(path.is_file() for path in result.output_paths.values())
    assert result.output_paths["model"].name.startswith(
        "alkabeer_sku_matcher_v2_"
    )
