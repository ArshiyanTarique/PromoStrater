"""Immutable snapshot, trust-weight, and challenge-isolation tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from sku_mapping.learning.store import LearningStoreError
from sku_mapping.retraining.snapshot import (
    InsufficientGoldLabelsError,
    build_training_snapshot,
    load_training_snapshot,
)
from sku_mapping.shadow.challenge import SealedChallengeSetError
from tests.retraining_fixtures import (
    populated_learning_store,
    phase7c_config,
    write_baseline,
)


def test_retraining_requires_minimum_gold_unless_override(
    tmp_path: Path,
) -> None:
    config = phase7c_config(tmp_path)
    store = populated_learning_store(tmp_path)
    baseline = write_baseline(tmp_path)
    with pytest.raises(InsufficientGoldLabelsError, match="requires 50"):
        build_training_snapshot(
            store=store,
            baseline_path=baseline,
            config=replace(
                config,
                retraining=replace(
                    config.retraining,
                    minimum_new_gold_labels=50,
                ),
            ),
        )
    result = build_training_snapshot(
        store=store,
        baseline_path=baseline,
        config=config,
        minimum_gold_override=4,
        override_reason="small deterministic unit fixture",
    )
    assert result.manifest["override_record"]["used"] is True


def test_snapshot_excludes_pseudo_and_weights_silver_lower_than_gold(
    tmp_path: Path,
) -> None:
    config = phase7c_config(tmp_path)
    store = populated_learning_store(tmp_path)
    result = build_training_snapshot(
        store=store,
        baseline_path=write_baseline(tmp_path),
        config=config,
        include_silver=True,
        minimum_gold_override=4,
        override_reason="test trust weights",
    )
    _, training, _ = load_training_snapshot(result.manifest_path)
    assert "PSEUDO" not in set(training["training_label_trust"])
    silver = training.loc[
        training["training_label_trust"] == "SILVER",
        "training_sample_weight",
    ]
    assert not silver.empty
    assert set(silver) == {config.retraining.silver_weight}
    assert config.retraining.silver_weight < config.retraining.gold_weight
    assert result.manifest["inclusion_policy"]["pseudo_weight"] == 0.0


def test_snapshot_is_immutable_and_content_hashed(tmp_path: Path) -> None:
    config = phase7c_config(tmp_path)
    store = populated_learning_store(tmp_path)
    first = build_training_snapshot(
        store=store,
        baseline_path=write_baseline(tmp_path),
        config=config,
        minimum_gold_override=4,
        override_reason="immutability test",
    )
    second = build_training_snapshot(
        store=store,
        baseline_path=tmp_path / "training_features.parquet",
        config=config,
        minimum_gold_override=4,
        override_reason="immutability test",
    )
    assert first.dataset_id == second.dataset_id
    original_hash = first.manifest["artifact_sha256"]
    frame = pd.read_parquet(first.snapshot_path)
    frame.loc[0, "pair_label"] = 1 - int(frame.loc[0, "pair_label"])
    frame.to_parquet(first.snapshot_path, index=False)
    with pytest.raises((ValueError, FileExistsError), match="hash|differs"):
        load_training_snapshot(first.manifest_path)
    assert original_hash != ""


def test_snapshot_rejects_sealed_challenge_overlap(tmp_path: Path) -> None:
    config = phase7c_config(tmp_path)
    store = populated_learning_store(tmp_path)
    review_id = store.governed_training_labels()["gold"][0]["review_id"]
    manifest = tmp_path / "sealed_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "SEALED_UNOPENED",
                "review_ids": [review_id],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LearningStoreError, match="sealed challenge"):
        build_training_snapshot(
            store=store,
            baseline_path=write_baseline(tmp_path),
            config=config,
            challenge_manifest_paths=[manifest],
            minimum_gold_override=4,
            override_reason="challenge test",
        )


def test_sealed_challenge_artifact_path_is_never_loaded(tmp_path: Path) -> None:
    config = phase7c_config(tmp_path)
    store = populated_learning_store(tmp_path)
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    baseline = sealed / "sealed_challenge_records.parquet"
    write_baseline(sealed).replace(baseline)
    (sealed / "challenge_manifest.json").write_text(
        json.dumps({"status": "SEALED_UNOPENED"}),
        encoding="utf-8",
    )
    with pytest.raises(SealedChallengeSetError, match="cannot be loaded"):
        build_training_snapshot(
            store=store,
            baseline_path=baseline,
            config=config,
            minimum_gold_override=4,
            override_reason="must still be rejected",
        )
