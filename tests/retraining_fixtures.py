"""Shared deterministic fixtures for Phase 7C tests."""

from __future__ import annotations

import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from sku_mapping.config import PipelineConfig, load_config
from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.learning.models import HumanReviewAnswer, LabelQuality
from sku_mapping.learning.store import LearningStore
from sku_mapping.ml.calibration import (
    ShadowModelPredictor,
    SigmoidScoreCalibrator,
)
from sku_mapping.ml.model_package import save_model_package
from sku_mapping.retraining.artifacts import sha256_file
from sku_mapping.retraining.registry import ControlledModelRegistry

# The champion used by retraining tests is built here rather than copied from
# models/. Depending on a gitignored repository binary made these tests break
# the moment superseded models were pruned, and the retraining comparison path
# is defined against the 19-column feature contract, which a ranked 41-column
# package does not satisfy.
CHAMPION_ID = "fixture-champion-19-column"
CHAMPION_PACKAGE_FILENAME = "fixture_champion_19_column.joblib"
CHAMPION_DIGEST = "c" * 64


def _major_minor(value: str) -> str:
    return ".".join(str(value).split(".")[:2])


def build_champion_package(
    registry_directory: Path, metadata_directory: Path
) -> Path:
    """Write a self-contained, strictly valid 19-feature champion package."""
    generator = np.random.default_rng(42)
    features = pd.DataFrame(
        generator.normal(size=(200, len(MODEL_FEATURE_COLUMNS))),
        columns=list(MODEL_FEATURE_COLUMNS),
    )
    labels = (features["protein_match"] > 0).astype(int).to_numpy()
    model = lightgbm.LGBMClassifier(
        n_estimators=20,
        num_leaves=3,
        min_child_samples=5,
        random_state=42,
        verbose=-1,
    ).fit(features, labels)
    calibration = SigmoidScoreCalibrator(42).fit(
        np.asarray(model.predict(features, raw_score=True), dtype=float),
        labels,
    )
    predictor = ShadowModelPredictor(
        model=model,
        calibration_model=calibration,
        feature_columns=list(MODEL_FEATURE_COLUMNS),
    )
    package = {
        "package_schema_version": "3.0",
        "model": model,
        "calibration_model": calibration,
        "predictor": predictor,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "auto_match_threshold": 0.95,
        "manual_review_threshold": 0.5,
        "approved_auto_match_threshold": None,
        "auto_match_threshold_approved": False,
        "model_id": CHAMPION_ID,
        "package_version": "3.0.0+fixture",
        "model_version": "fixture-champion",
        "parent_model": None,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_dataset_hash": CHAMPION_DIGEST,
        "processed_feature_table_hash": CHAMPION_DIGEST,
        "split_assignment_hash": CHAMPION_DIGEST,
        "metrics": {},
        "threshold_evidence": {},
        "calibration_method": "sigmoid",
        "lightgbm_version": lightgbm.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "compatibility_policy": {
            "python_major_minor": _major_minor(platform.python_version()),
            "lightgbm_major_minor": _major_minor(lightgbm.__version__),
            "sklearn_major_minor": _major_minor(sklearn.__version__),
        },
        "training_config": {},
        "random_seed": 42,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
    }
    package_path, _ = save_model_package(
        package,
        registry_directory / CHAMPION_PACKAGE_FILENAME,
        metadata_directory
        / CHAMPION_PACKAGE_FILENAME.replace(".joblib", ".json"),
    )
    return package_path


def champion_registry_entry() -> dict:
    """Return the immutable registry entry describing the fixture champion."""
    return {
        "package_filename": CHAMPION_PACKAGE_FILENAME,
        "model_id": CHAMPION_ID,
        "model_version": "fixture-champion",
        "package_version": "3.0.0+fixture",
        "creation_timestamp": "2026-01-01T00:00:00+00:00",
        "training_dataset_hash": CHAMPION_DIGEST,
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
        "notes": "Self-contained retraining fixture champion.",
        "parent_model": None,
    }


def phase7c_config(tmp_path: Path) -> PipelineConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        learning_store=replace(
            config.learning_store,
            database_path=tmp_path / "learning.db",
        ),
        retraining=replace(
            config.retraining,
            minimum_new_gold_labels=4,
            recent_gold_holdout_count=2,
            minimum_evaluation_rows=2,
            minimum_subgroup_rows=1,
            snapshot_directory=tmp_path / "snapshots",
            challenger_directory=tmp_path / "challengers",
            comparison_report_directory=tmp_path / "comparisons",
        ),
        shadow_mode=replace(
            config.shadow_mode,
            registry_path=tmp_path / "models" / "model_registry.json",
        ),
    )


def baseline_frame(groups: int = 50) -> pd.DataFrame:
    rows = []
    for group in range(groups):
        for label in (0, 1):
            features = {
                column: float(group * 10 + label)
                for column in MODEL_FEATURE_COLUMNS
            }
            features.update(
                {
                    "protein_match": float(label),
                    "family_match": float(label),
                    "variant_match": float(label),
                    "size_match": float(label),
                    "pack_format_match": float(label),
                    "word_similarity": float(10 + group / 10)
                    if label == 0
                    else float(90 + group / 100),
                    "character_similarity": 15.0 if label == 0 else 92.0,
                    "token_similarity": 12.0 if label == 0 else 94.0,
                }
            )
            rows.append(
                {
                    "record_id": f"baseline-{group}-{label}",
                    "offer_group_id": f"baseline-offer-{group}",
                    "offer_text": f"Baseline family {group} label {label}",
                    "master_itemcode": f"baseline-sku-{group}-{label}",
                    "pair_label": label,
                    "source_dataset": "HUMAN_AUDIT",
                    "label_provenance": "human_audited",
                    "product_class_offer": f"family-{group % 3}",
                    **features,
                }
            )
    return pd.DataFrame(rows)


def write_baseline(tmp_path: Path, groups: int = 50) -> Path:
    path = tmp_path / "training_features.parquet"
    baseline_frame(groups).to_parquet(path, index=False)
    return path


def populated_learning_store(tmp_path: Path) -> LearningStore:
    store = LearningStore(tmp_path / "learning.db")
    run_id = "review-run"
    store.upsert_pipeline_run(
        {
            "run_id": run_id,
            "status": "COMPLETED_ASSISTED",
            "deployment_mode": "assisted",
            "source_row_count": 5,
            "unique_offer_count": 5,
        }
    )
    records = []
    for offer_index in range(5):
        for rank in (1, 2):
            candidate_id = f"review-sku-{offer_index}-{rank}"
            feature_snapshot = {
                column: float(10_000 + offer_index * 100 + rank)
                for column in MODEL_FEATURE_COLUMNS
            }
            is_first = rank == 1
            feature_snapshot.update(
                {
                    "protein_match": float(is_first),
                    "family_match": float(is_first),
                    "variant_match": float(is_first),
                    "size_match": float(is_first),
                    "pack_format_match": float(is_first),
                    "word_similarity": 95.0 if is_first else 5.0,
                    "character_similarity": 95.0 if is_first else 5.0,
                    "token_similarity": 95.0 if is_first else 5.0,
                }
            )
            records.append(
                {
                    "offer_id": f"review-offer-{offer_index}",
                    "offer_description": f"Reviewed offer {offer_index}",
                    "candidate_id": candidate_id,
                    "candidate_description": candidate_id,
                    "candidate_rank": rank,
                    "lightgbm_probability": 0.95 if is_first else 0.05,
                    "embedding_similarity": 0.9 if is_first else 0.1,
                    "agreement_status": (
                        "DISAGREEMENT"
                        if offer_index == 2
                        else "WEAK_AGREEMENT"
                    ),
                    "llm_decision": (
                        "ACCEPT_CANDIDATE" if offer_index == 3 else ""
                    ),
                    "final_decision": (
                        "AUTO_ACCEPT"
                        if offer_index == 0 and is_first
                        else (
                            "MANUAL_REVIEW"
                            if is_first
                            else "CANDIDATE_NOT_SELECTED"
                        )
                    ),
                    "decision_source": (
                        "STRUCTURED_LLM_REVIEW"
                        if offer_index == 3 and is_first
                        else "AGREEMENT_POLICY"
                    ),
                    "conflict_flags": (
                        ["pack_conflict"] if offer_index == 4 else []
                    ),
                    "feature_snapshot": feature_snapshot,
                }
            )
    prediction_ids = store.add_predictions(run_id, records)
    store.add_automated_label(
        prediction_id=prediction_ids[0],
        source="STRUCTURED_LLM_REVIEW",
        proposed_label="LLM_ACCEPT",
        selected_candidate_id="review-sku-0-1",
        confidence=0.95,
        label_quality=LabelQuality.SILVER,
        eligibility_status="POLICY_QUALIFIED_REVIEW_REQUIRED_BEFORE_TRAINING",
    )
    store.add_automated_label(
        prediction_id=prediction_ids[2],
        source="MODEL_POLICY",
        proposed_label="AUTO_ACCEPT",
        selected_candidate_id="review-sku-1-1",
        confidence=0.99,
        label_quality=LabelQuality.PSEUDO,
        eligibility_status="NOT_TRAINING_ELIGIBLE",
    )
    session_id = store.create_review_session(run_id)
    assert session_id
    for index in range(5):
        question = store.next_unanswered_question(session_id)
        assert question is not None
        if index == 4:
            store.save_answer(
                question.review_id,
                HumanReviewAnswer(
                    is_correct=False,
                    corrected_candidate_id=question.supplied_candidates[1][0],
                ),
            )
        else:
            store.save_answer(
                question.review_id,
                HumanReviewAnswer(is_correct=True),
            )
    return store


def registry_with_champion(
    tmp_path: Path,
    store: LearningStore,
    *,
    active: bool = False,
) -> ControlledModelRegistry:
    import json

    registry_directory = tmp_path / "models" / "registry"
    metadata_directory = tmp_path / "models" / "metadata"
    registry_directory.mkdir(parents=True, exist_ok=True)
    metadata_directory.mkdir(parents=True, exist_ok=True)
    destination = build_champion_package(
        registry_directory, metadata_directory
    )
    entry = champion_registry_entry()
    payload = {
        "schema_version": "2.0",
        "automatic_production_matching_enabled": False,
        "active_assisted_model_id": CHAMPION_ID if active else None,
        "activation_history": [],
        "models": [entry],
    }
    registry_path = tmp_path / "models" / "model_registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    store.register_model_version(
        model_id=CHAMPION_ID,
        model_hash=sha256_file(destination),
        status="EXISTING_CHAMPION",
        champion_status="CHAMPION_ACTIVE" if active else "PREVIOUS_CHAMPION",
    )
    return ControlledModelRegistry(
        registry_path=registry_path,
        model_directory=registry_directory,
        metadata_directory=metadata_directory,
        store=store,
    )
