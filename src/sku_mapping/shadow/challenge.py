"""Build immutable sealed challenge sets without opening or evaluating them."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DECISIVE_CHALLENGE_LABELS = frozenset(
    {
        "CORRECT_TOP_CANDIDATE",
        "CORRECT_OTHER_CANDIDATE",
        "NO_VALID_MASTER_SKU",
    }
)


class SealedChallengeSetError(ValueError):
    """Raised when a sealed challenge artifact is accessed unsafely."""


@dataclass(frozen=True)
class SealedChallengeSetResult:
    """Paths identifying a newly sealed, unopened challenge set."""

    challenge_set_id: str
    directory: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def challenge_manifest_template() -> dict[str, Any]:
    """Return the schema template without creating or opening a challenge set."""
    return {
        "challenge_set_id": None,
        "status": "SEALED_UNOPENED",
        "evaluation_approval_required": True,
        "evaluation_approved": False,
        "ordinary_training_access": "PROHIBITED",
        "row_count": None,
        "class_balance": None,
        "source_date_range": None,
        "reviewer_coverage": None,
        "family_coverage": None,
        "artifact_hashes": None,
        "opened_at": None,
        "evaluated_at": None,
    }


def _load_review_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SealedChallengeSetError(
                f"Unable to read staged review record: {path}"
            ) from error
        expected = payload.get("review_record_sha256")
        immutable = {
            key: value
            for key, value in payload.items()
            if key != "review_record_sha256"
        }
        actual = _sha256_bytes(
            json.dumps(
                immutable,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        if expected != actual:
            raise SealedChallengeSetError(
                f"Staged review record hash mismatch: {path}"
            )
        records.append(payload)
    return records


def build_sealed_challenge_set(
    *,
    normalized_review_paths: Iterable[str | Path],
    shadow_predictions_path: str | Path,
    challenge_root: str | Path,
) -> SealedChallengeSetResult:
    """Create a SEALED_UNOPENED set; this function performs no evaluation."""
    records = _load_review_records(normalized_review_paths)
    prediction_path = Path(shadow_predictions_path)
    if not records:
        raise SealedChallengeSetError("At least one staged review is required")
    if not prediction_path.is_file():
        raise SealedChallengeSetError("Shadow predictions artifact is missing")
    predictions = pd.read_parquet(prediction_path)
    required = {
        "shadow_run_id",
        "offer_group_id",
        "candidate_rank",
        "master_itemcode",
        "product_family",
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise SealedChallengeSetError(
            f"Shadow predictions lack challenge columns: {missing}"
        )

    pair_rows: list[dict[str, Any]] = []
    excluded_labels: dict[str, int] = {}
    included_records: list[dict[str, Any]] = []
    for record in records:
        label = str(record.get("human_label", ""))
        if label not in DECISIVE_CHALLENGE_LABELS:
            excluded_labels[label] = excluded_labels.get(label, 0) + 1
            continue
        subset = predictions[
            predictions["shadow_run_id"].astype(str).eq(
                str(record["shadow_run_id"])
            )
            & predictions["offer_group_id"].astype(str).eq(
                str(record["offer_group_id"])
            )
        ].copy()
        if subset.empty:
            raise SealedChallengeSetError(
                "A staged review has no matching shadow candidate rows"
            )
        selected = record.get("selected_master_itemcode")
        if label == "NO_VALID_MASTER_SKU":
            subset["pair_label"] = 0
        else:
            subset["pair_label"] = subset["master_itemcode"].astype(str).eq(
                str(selected)
            ).astype(int)
            if int(subset["pair_label"].sum()) != 1:
                raise SealedChallengeSetError(
                    "Reviewed selected SKU is not unique in shadow candidates"
                )
        subset["review_record_sha256"] = record["review_record_sha256"]
        pair_rows.extend(subset.to_dict(orient="records"))
        included_records.append(record)
    if not pair_rows:
        raise SealedChallengeSetError(
            "No decisive human reviews are eligible for sealing"
        )

    sealed = pd.DataFrame(pair_rows)
    record_hashes = sorted(
        str(record["review_record_sha256"]) for record in included_records
    )
    prediction_hash = _sha256_file(prediction_path)
    identity_payload = json.dumps(
        {
            "review_record_hashes": record_hashes,
            "shadow_predictions_sha256": prediction_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    challenge_set_id = (
        "challenge-"
        + hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:24]
    )
    directory = Path(challenge_root) / challenge_set_id
    if directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite sealed challenge set: {directory}"
        )
    directory.mkdir(parents=True)
    sealed_path = directory / "sealed_challenge_records.parquet"
    _atomic_parquet(sealed, sealed_path)

    review_dates = pd.to_datetime(
        [record["review_timestamp"] for record in included_records],
        utc=True,
    )
    reviewer_counts = pd.Series(
        [str(record["reviewer_code"]) for record in included_records]
    ).value_counts()
    family_counts = (
        sealed["product_family"]
        .astype("string")
        .fillna("<missing>")
        .value_counts()
    )
    class_counts = sealed["pair_label"].astype(int).value_counts()
    manifest = {
        "challenge_set_id": challenge_set_id,
        "status": "SEALED_UNOPENED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_approval_required": True,
        "evaluation_approved": False,
        "ordinary_training_access": "PROHIBITED",
        "row_count": int(len(sealed)),
        "offer_count": int(sealed["offer_group_id"].nunique()),
        "class_balance": {
            "negative": int(class_counts.get(0, 0)),
            "positive": int(class_counts.get(1, 0)),
            "positive_rate": float(sealed["pair_label"].mean()),
        },
        "source_date_range": {
            "review_timestamp_min": review_dates.min().isoformat(),
            "review_timestamp_max": review_dates.max().isoformat(),
        },
        "reviewer_coverage": {
            str(key): int(value) for key, value in reviewer_counts.items()
        },
        "family_coverage": {
            str(key): int(value) for key, value in family_counts.items()
        },
        "review_records_included": len(included_records),
        "review_labels_excluded": excluded_labels,
        "artifact_hashes": {
            "sealed_challenge_records.parquet": _sha256_file(sealed_path),
            "source_shadow_predictions": prediction_hash,
            "review_record_sha256": record_hashes,
        },
        "labels_exposed_to_training": False,
        "opened_at": None,
        "evaluated_at": None,
    }
    manifest_path = directory / "challenge_manifest.json"
    _atomic_json(manifest, manifest_path)
    return SealedChallengeSetResult(
        challenge_set_id=challenge_set_id,
        directory=directory,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def assert_not_sealed_challenge_input(path: str | Path) -> None:
    """Block ordinary training/evaluation from unopened challenge artifacts."""
    current = Path(path).resolve()
    search_start = current if current.is_dir() else current.parent
    for directory in (search_start, *search_start.parents):
        manifest_path = directory / "challenge_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SealedChallengeSetError(
                f"Challenge manifest is unreadable: {manifest_path}"
            ) from error
        if manifest.get("status") == "SEALED_UNOPENED":
            raise SealedChallengeSetError(
                "SEALED_UNOPENED challenge sets cannot be loaded by ordinary "
                "training, threshold, or evaluation code"
            )
