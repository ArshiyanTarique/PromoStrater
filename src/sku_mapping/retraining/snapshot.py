"""Immutable baseline-plus-reviewed-label training snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from sku_mapping.config import PipelineConfig, RetrainingConfig
from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.learning.store import LearningStore, LearningStoreError
from sku_mapping.ml.leakage import build_leakage_groups
from sku_mapping.ml.provenance import build_provenance_weights
from sku_mapping.retraining.artifacts import (
    atomic_json,
    atomic_parquet,
    canonical_json,
    sha256_file,
)
from sku_mapping.shadow.challenge import assert_not_sealed_challenge_input

SNAPSHOT_SCHEMA_VERSION = "phase-7c-v1"
SNAPSHOT_REQUIRED_COLUMNS = (
    "record_id",
    "offer_group_id",
    "offer_text",
    "master_itemcode",
    "pair_label",
    "source_dataset",
    "label_provenance",
    "product_class_offer",
    "training_label_trust",
    "training_sample_weight",
    "row_review_id",
    "row_automated_label_id",
    *MODEL_FEATURE_COLUMNS,
)


class InsufficientGoldLabelsError(ValueError):
    """Raised when the explicit retraining evidence gate is not met."""


@dataclass(frozen=True)
class SnapshotBuildResult:
    dataset_id: str
    snapshot_path: Path
    evaluation_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _feature_schema_version() -> str:
    payload = canonical_json(list(MODEL_FEATURE_COLUMNS))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{FEATURE_GENERATOR_VERSION}+{digest}"


def _validate_feature_snapshot(snapshot: Mapping[str, Any]) -> dict[str, float]:
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in snapshot]
    if missing:
        raise LearningStoreError(
            f"Reviewed prediction is missing feature snapshot columns: {missing}"
        )
    output: dict[str, float] = {}
    for column in MODEL_FEATURE_COLUMNS:
        value = snapshot[column]
        try:
            output[column] = float(value) if value is not None else np.nan
        except (TypeError, ValueError) as error:
            raise LearningStoreError(
                f"Reviewed feature {column!r} is not numeric"
            ) from error
    return output


def _review_candidate_row(
    review: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    pair_label: int,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    return {
        "record_id": (
            f"gold:{review['review_id']}:{candidate_id}:{int(pair_label)}"
        ),
        "offer_group_id": f"review:{review['run_id']}:{review['offer_id']}",
        "offer_text": str(review.get("offer_description") or ""),
        "master_itemcode": candidate_id,
        "pair_label": int(pair_label),
        "source_dataset": "HUMAN_REVIEW",
        "label_provenance": "GOLD_HUMAN_CONFIRMED",
        "product_class_offer": "<unknown>",
        "training_label_trust": "GOLD",
        "row_review_id": str(review["review_id"]),
        "row_automated_label_id": "",
        **_validate_feature_snapshot(candidate["feature_snapshot"]),
    }


def _gold_rows(
    reviews: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for review in reviews:
        candidates = {
            str(candidate["candidate_id"]): candidate
            for candidate in review["candidates"]
        }
        suggested = str(review["suggested_candidate_id"])
        if suggested not in candidates:
            raise LearningStoreError(
                f"Review {review['review_id']} lacks its suggested prediction"
            )
        if bool(review["human_answer"]):
            rows.append(
                _review_candidate_row(
                    review, candidates[suggested], pair_label=1
                )
            )
            continue
        corrected = str(review.get("corrected_candidate_id") or "")
        if corrected:
            if corrected not in candidates:
                raise LearningStoreError(
                    "Corrected GOLD candidate is outside supplied predictions"
                )
            rows.append(
                _review_candidate_row(
                    review, candidates[suggested], pair_label=0
                )
            )
            rows.append(
                _review_candidate_row(
                    review, candidates[corrected], pair_label=1
                )
            )
        elif bool(review.get("none_of_candidates")):
            rows.extend(
                _review_candidate_row(review, candidate, pair_label=0)
                for candidate in candidates.values()
            )
        else:
            raise LearningStoreError(
                "GOLD review lacks a decisive correction state"
            )
    return pd.DataFrame(rows)


def _silver_rows(labels: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in labels:
        candidate_id = str(label.get("selected_candidate_id") or "")
        if not candidate_id or candidate_id != str(label["candidate_id"]):
            raise LearningStoreError(
                "SILVER label selected candidate differs from prediction"
            )
        rows.append(
            {
                "record_id": f"silver:{label['label_id']}:{candidate_id}",
                "offer_group_id": (
                    f"review:{label['run_id']}:{label['offer_id']}"
                ),
                "offer_text": str(label.get("offer_description") or ""),
                "master_itemcode": candidate_id,
                "pair_label": 1,
                "source_dataset": "STRUCTURED_LLM_REVIEW",
                "label_provenance": "SILVER_POLICY_QUALIFIED_LLM",
                "product_class_offer": "<unknown>",
                "training_label_trust": "SILVER",
                "row_review_id": "",
                "row_automated_label_id": str(label["label_id"]),
                **_validate_feature_snapshot(label["feature_snapshot"]),
            }
        )
    return pd.DataFrame(rows)


def _challenge_proof(
    manifest_paths: Iterable[str | Path],
    included_review_ids: set[str],
) -> dict[str, Any]:
    checked: list[dict[str, str]] = []
    sealed_ids: set[str] = set()
    for raw_path in manifest_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Challenge manifest not found: {path}")
        content = path.read_bytes()
        try:
            manifest = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LearningStoreError(
                f"Challenge manifest is unreadable: {path}"
            ) from error
        if manifest.get("status") != "SEALED_UNOPENED":
            raise LearningStoreError(
                "Challenge exclusion proof requires SEALED_UNOPENED manifests"
            )
        for key in ("review_ids", "included_review_ids", "review_record_ids"):
            values = manifest.get(key, [])
            if isinstance(values, list):
                sealed_ids.update(str(value) for value in values)
        hashes = manifest.get("artifact_hashes", {}).get(
            "review_record_sha256", []
        )
        if isinstance(hashes, list):
            sealed_ids.update(str(value) for value in hashes)
        checked.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "status": "SEALED_UNOPENED",
            }
        )
    overlap = sorted(included_review_ids & sealed_ids)
    if overlap:
        raise LearningStoreError(
            f"Snapshot contains sealed challenge identities: {overlap}"
        )
    return {
        "proof_version": SNAPSHOT_SCHEMA_VERSION,
        "sealed_manifest_checks": checked,
        "sealed_review_ids_checked": sorted(sealed_ids),
        "intersection": [],
        "challenge_rows_excluded": True,
        "sealed_artifacts_opened": False,
    }


def _content_hash(frame: pd.DataFrame) -> str:
    columns = list(SNAPSHOT_REQUIRED_COLUMNS)
    canonical = frame.loc[:, columns].sort_values(
        ["record_id", "master_itemcode"], kind="stable"
    )
    payload = canonical.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_snapshot_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in SNAPSHOT_REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Training snapshot is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Training snapshot cannot be empty")
    if set(pd.to_numeric(frame["pair_label"]).astype(int).unique()) != {0, 1}:
        raise ValueError("Training snapshot must contain both binary labels")
    weights = pd.to_numeric(frame["training_sample_weight"], errors="coerce")
    if weights.isna().any() or (weights <= 0).any():
        raise ValueError("Included training rows require positive sample weights")
    if (frame["training_label_trust"] == "PSEUDO").any():
        raise ValueError("PSEUDO rows are prohibited from training snapshots")


def build_training_snapshot(
    *,
    store: LearningStore,
    baseline_path: str | Path,
    config: PipelineConfig,
    challenge_manifest_paths: Sequence[str | Path] = (),
    include_silver: bool | None = None,
    minimum_gold_override: int | None = None,
    override_reason: str | None = None,
) -> SnapshotBuildResult:
    """Materialize one immutable, leakage-audited retraining snapshot."""
    policy: RetrainingConfig = config.retraining
    baseline = Path(baseline_path)
    output_root = policy.snapshot_directory
    assert_not_sealed_challenge_input(baseline)
    assert_not_sealed_challenge_input(output_root)
    if not baseline.is_file():
        raise FileNotFoundError(f"Baseline feature table not found: {baseline}")

    include_silver_effective = (
        policy.include_silver if include_silver is None else bool(include_silver)
    )
    labels = store.governed_training_labels(
        include_silver=include_silver_effective
    )
    gold_reviews = labels["gold"]
    new_gold_count = store.count_new_gold_labels_since_last_model()
    minimum = (
        policy.minimum_new_gold_labels
        if minimum_gold_override is None
        else int(minimum_gold_override)
    )
    if minimum < 1:
        raise ValueError("minimum_gold_override must be positive")
    if minimum_gold_override is not None and not str(override_reason or "").strip():
        raise ValueError("A development GOLD-label override requires a reason")
    if new_gold_count < minimum:
        raise InsufficientGoldLabelsError(
            f"Retraining requires {minimum} new GOLD labels; "
            f"only {new_gold_count} are available"
        )
    if len(gold_reviews) < policy.recent_gold_holdout_count + 1:
        raise InsufficientGoldLabelsError(
            "Not enough GOLD labels remain after the configured recent-label "
            "evaluation holdout"
        )

    ordered_reviews = sorted(
        gold_reviews,
        key=lambda row: (str(row["answered_at"]), str(row["review_id"])),
    )
    holdout_reviews = ordered_reviews[-policy.recent_gold_holdout_count :]
    training_reviews = ordered_reviews[: -policy.recent_gold_holdout_count]
    holdout_ids = {str(row["review_id"]) for row in holdout_reviews}
    included_review_ids = {str(row["review_id"]) for row in training_reviews}
    proof = _challenge_proof(
        challenge_manifest_paths,
        included_review_ids | holdout_ids,
    )

    baseline_frame = pd.read_parquet(baseline)
    missing_baseline = [
        column
        for column in (
            "record_id",
            "offer_group_id",
            "offer_text",
            "master_itemcode",
            "pair_label",
            "source_dataset",
            "label_provenance",
            "product_class_offer",
            *MODEL_FEATURE_COLUMNS,
        )
        if column not in baseline_frame
    ]
    if missing_baseline:
        raise ValueError(
            f"Baseline feature table is missing columns: {missing_baseline}"
        )
    baseline_frame = baseline_frame.copy()
    baseline_frame["training_label_trust"] = "BASELINE"
    baseline_frame["row_review_id"] = ""
    baseline_frame["row_automated_label_id"] = ""
    baseline_weights, _, _ = build_provenance_weights(
        baseline_frame,
        config.training.provenance_weights,
    )
    baseline_frame["training_sample_weight"] = (
        baseline_weights * policy.baseline_weight_multiplier
    )

    gold_training = _gold_rows(training_reviews)
    gold_holdout = _gold_rows(holdout_reviews)
    gold_training["training_sample_weight"] = policy.gold_weight
    gold_holdout["training_sample_weight"] = policy.gold_weight
    silver = _silver_rows(labels["silver"])
    if not silver.empty:
        silver["training_sample_weight"] = policy.silver_weight

    parts = [baseline_frame, gold_training]
    if include_silver_effective and not silver.empty:
        parts.append(silver)
    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined = combined.loc[:, list(dict.fromkeys(SNAPSHOT_REQUIRED_COLUMNS))]
    evaluation = gold_holdout.loc[
        :, list(dict.fromkeys(SNAPSHOT_REQUIRED_COLUMNS))
    ].copy()

    # Remove every connected component touching the recent GOLD holdout.
    leakage_input = pd.concat([combined, evaluation], ignore_index=True)
    grouped = build_leakage_groups(leakage_input).frame
    holdout_mask = grouped["row_review_id"].astype(str).isin(holdout_ids)
    holdout_groups = set(grouped.loc[holdout_mask, "leakage_group_id"].astype(str))
    training_mask = ~grouped["leakage_group_id"].astype(str).isin(holdout_groups)
    training = grouped.loc[training_mask].copy()
    evaluation = grouped.loc[holdout_mask].copy()
    training = training.loc[:, list(dict.fromkeys(SNAPSHOT_REQUIRED_COLUMNS))]
    evaluation = evaluation.loc[:, list(dict.fromkeys(SNAPSHOT_REQUIRED_COLUMNS))]
    included_review_ids = (
        set(training["row_review_id"].astype(str)) - {""}
    )
    removed_review_ids = {
        str(row["review_id"]) for row in training_reviews
    } - included_review_ids
    excluded_review_ids = holdout_ids | removed_review_ids
    _validate_snapshot_frame(training)
    if evaluation.empty:
        raise ValueError("Recent GOLD evaluation holdout is empty")
    if set(pd.to_numeric(evaluation["pair_label"]).astype(int).unique()) != {0, 1}:
        raise ValueError(
            "Recent GOLD evaluation holdout must contain both binary labels"
        )

    content_hash = _content_hash(training)
    evaluation_content_hash = _content_hash(evaluation)
    dataset_id = f"training-dataset-{content_hash[:24]}"
    directory = output_root / dataset_id
    snapshot_path = directory / "training_snapshot.parquet"
    evaluation_path = directory / "recent_gold_evaluation.parquet"
    manifest_path = directory / "snapshot_manifest.json"
    if directory.exists():
        if not snapshot_path.is_file() or not manifest_path.is_file():
            raise FileExistsError(
                f"Refusing to replace incomplete snapshot directory: {directory}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("content_hash") != content_hash
            or sha256_file(snapshot_path)
            != existing.get("artifact_sha256")
        ):
            raise FileExistsError(
                f"Immutable snapshot differs at {directory}"
            )
        return SnapshotBuildResult(
            dataset_id=dataset_id,
            snapshot_path=snapshot_path,
            evaluation_path=evaluation_path,
            manifest_path=manifest_path,
            manifest=existing,
        )

    directory.mkdir(parents=True, exist_ok=False)
    atomic_parquet(training, snapshot_path)
    atomic_parquet(evaluation, evaluation_path)
    source_counts = {
        str(key): int(value)
        for key, value in training["training_label_trust"]
        .value_counts()
        .sort_index()
        .items()
    }
    included_silver_ids = sorted(
        set(training["row_automated_label_id"].astype(str)) - {""}
    )
    override_record = {
        "used": minimum_gold_override is not None,
        "configured_minimum": policy.minimum_new_gold_labels,
        "effective_minimum": minimum,
        "reason": str(override_reason or "") or None,
    }
    inclusion_policy = {
        "include_silver": include_silver_effective,
        "baseline_weight_multiplier": policy.baseline_weight_multiplier,
        "gold_weight": policy.gold_weight,
        "silver_weight": policy.silver_weight,
        "pseudo_weight": policy.pseudo_weight,
        "pseudo_included": False,
        "recent_gold_holdout_count": policy.recent_gold_holdout_count,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "content_hash": content_hash,
        "artifact_sha256": sha256_file(snapshot_path),
        "artifact_filename": snapshot_path.name,
        "evaluation_content_hash": evaluation_content_hash,
        "evaluation_artifact_sha256": sha256_file(evaluation_path),
        "evaluation_artifact_filename": evaluation_path.name,
        "baseline_path": str(baseline.resolve()),
        "baseline_sha256": sha256_file(baseline),
        "row_count": int(len(training)),
        "evaluation_row_count": int(len(evaluation)),
        "counts_by_trust_level": source_counts,
        "included_review_ids": sorted(included_review_ids),
        "excluded_review_ids": sorted(excluded_review_ids),
        "included_automated_label_ids": included_silver_ids,
        "feature_schema_version": _feature_schema_version(),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "challenge_set_exclusion_proof": proof,
        "inclusion_policy": inclusion_policy,
        "override_record": override_record,
        "leakage_policy": {
            "connected_components": True,
            "offer_group_id": True,
            "synthetic_template": True,
            "feature_vector_hash": True,
            "holdout_component_overlap": 0,
        },
        "retraining_performed": False,
    }
    atomic_json(manifest, manifest_path)
    store.register_training_snapshot(
        dataset_id=dataset_id,
        created_at=created_at,
        source_label_counts=source_counts,
        row_count=len(training),
        included_review_ids=sorted(included_review_ids),
        excluded_review_ids=sorted(excluded_review_ids),
        included_automated_label_ids=included_silver_ids,
        content_hash=content_hash,
        feature_schema_version=manifest["feature_schema_version"],
        artifact_path=str(snapshot_path.resolve()),
        artifact_sha256=manifest["artifact_sha256"],
        manifest_path=str(manifest_path.resolve()),
        challenge_exclusion_proof=proof,
        inclusion_policy=inclusion_policy,
        override_record=override_record,
        evaluation_artifact_path=str(evaluation_path.resolve()),
        evaluation_artifact_sha256=manifest["evaluation_artifact_sha256"],
    )
    return SnapshotBuildResult(
        dataset_id=dataset_id,
        snapshot_path=snapshot_path,
        evaluation_path=evaluation_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_training_snapshot(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Validate an immutable snapshot and return training/evaluation frames."""
    path = Path(manifest_path)
    assert_not_sealed_challenge_input(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Unsupported training snapshot schema")
    if manifest.get("feature_columns") != list(MODEL_FEATURE_COLUMNS):
        raise ValueError("Snapshot feature order is incompatible")
    snapshot_path = path.parent / str(manifest["artifact_filename"])
    evaluation_path = path.parent / str(
        manifest["evaluation_artifact_filename"]
    )
    for artifact, expected in (
        (snapshot_path, manifest["artifact_sha256"]),
        (evaluation_path, manifest["evaluation_artifact_sha256"]),
    ):
        assert_not_sealed_challenge_input(artifact)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Immutable snapshot artifact hash mismatch: {artifact}")
    training = pd.read_parquet(snapshot_path)
    evaluation = pd.read_parquet(evaluation_path)
    _validate_snapshot_frame(training)
    if _content_hash(training) != manifest["content_hash"]:
        raise ValueError("Training snapshot content hash mismatch")
    if _content_hash(evaluation) != manifest["evaluation_content_hash"]:
        raise ValueError("Evaluation snapshot content hash mismatch")
    if (training["training_label_trust"] == "PSEUDO").any():
        raise ValueError("Snapshot unexpectedly contains PSEUDO labels")
    return manifest, training, evaluation
