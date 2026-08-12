"""Immutable intake validation for completed shadow human-review files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sku_mapping.data.validators import normalize_itemcode
from sku_mapping.shadow.review import (
    ALLOWED_HUMAN_LABELS,
    REVIEW_COMPLETION_COLUMNS,
)


class ReviewIntakeError(ValueError):
    """Raised when a submitted review file is invalid."""


class DuplicateReviewError(ReviewIntakeError):
    """Raised when raw or normalised review content already exists."""


@dataclass(frozen=True)
class ReviewIntakeResult:
    """Immutable staging paths and intake audit."""

    raw_submission_path: Path
    normalized_record_paths: tuple[Path, ...]
    audit_path: Path
    audit: dict[str, Any]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise DuplicateReviewError(
            f"Refusing to overwrite existing immutable artifact: {path}"
        ) from error


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


def _clean_scalar(value: object) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if pd.notna(value) else None
    return str(value).strip()


def _candidate_codes(row: pd.Series) -> list[str]:
    columns = sorted(
        (
            column
            for column in row.index
            if re.fullmatch(r"top_candidate_\d+_itemcode", str(column))
        ),
        key=lambda value: int(str(value).split("_")[2]),
    )
    codes: list[str] = []
    for column in columns:
        normalized = normalize_itemcode(row[column])
        if not pd.isna(normalized):
            codes.append(str(normalized))
    return codes


def _load_existing_identities(normalized_directory: Path) -> set[str]:
    identities: set[str] = set()
    if not normalized_directory.is_dir():
        return identities
    for path in normalized_directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        identity = payload.get("review_identity")
        if isinstance(identity, str):
            identities.add(identity)
    return identities


def _validate_review_rows(
    frame: pd.DataFrame,
    product_master: pd.DataFrame,
    existing_identities: set[str],
) -> list[dict[str, Any]]:
    required = {
        "shadow_run_id",
        "offer_group_id",
        "offer_text",
        *REVIEW_COMPLETION_COLUMNS,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ReviewIntakeError(
            f"Completed review file is missing columns: {missing}"
        )
    master_codes = {
        str(code)
        for value in product_master["Itemcode"]
        if not pd.isna(code := normalize_itemcode(value))
    }
    normalized_rows: list[dict[str, Any]] = []
    incoming_identities: set[str] = set()
    for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
        label = str(row["human_label"]).strip()
        if label not in ALLOWED_HUMAN_LABELS:
            raise ReviewIntakeError(
                f"Row {row_number} has invalid human_label: {label!r}"
            )
        reviewer = str(row["reviewer_code"]).strip()
        if not reviewer:
            raise ReviewIntakeError(
                f"Row {row_number} requires reviewer_code"
            )
        reviewed_at = pd.to_datetime(
            row["review_timestamp"], errors="coerce", utc=True
        )
        if pd.isna(reviewed_at):
            raise ReviewIntakeError(
                f"Row {row_number} requires a valid review_timestamp"
            )
        offer_group_id = str(row["offer_group_id"]).strip()
        shadow_run_id = str(row["shadow_run_id"]).strip()
        if not offer_group_id or not shadow_run_id:
            raise ReviewIntakeError(
                f"Row {row_number} lacks immutable run/offer identity"
            )
        identity_payload = (
            f"{shadow_run_id}|{offer_group_id}|{reviewer.lower()}"
        )
        review_identity = hashlib.sha256(
            identity_payload.encode("utf-8")
        ).hexdigest()
        if (
            review_identity in incoming_identities
            or review_identity in existing_identities
        ):
            raise DuplicateReviewError(
                f"Duplicate review detected at row {row_number}"
            )
        incoming_identities.add(review_identity)

        candidate_codes = _candidate_codes(row)
        selected_value = normalize_itemcode(row["selected_master_itemcode"])
        selected = None if pd.isna(selected_value) else str(selected_value)
        if selected is not None and selected not in master_codes:
            raise ReviewIntakeError(
                f"Row {row_number} selected unknown Product Master SKU "
                f"{selected!r}"
            )
        if label == "CORRECT_TOP_CANDIDATE":
            if not candidate_codes or selected != candidate_codes[0]:
                raise ReviewIntakeError(
                    f"Row {row_number} CORRECT_TOP_CANDIDATE selection "
                    "must equal the first listed candidate"
                )
        elif label == "CORRECT_OTHER_CANDIDATE":
            if (
                selected is None
                or selected not in candidate_codes[1:]
                or selected == (candidate_codes[0] if candidate_codes else None)
            ):
                raise ReviewIntakeError(
                    f"Row {row_number} CORRECT_OTHER_CANDIDATE selection "
                    "must be a listed non-top candidate"
                )
        elif selected is not None:
            raise ReviewIntakeError(
                f"Row {row_number} label {label} contradicts a selected SKU"
            )

        normalized = {
            str(column): _clean_scalar(value)
            for column, value in row.items()
        }
        normalized["human_label"] = label
        normalized["selected_master_itemcode"] = selected
        normalized["reviewer_code"] = reviewer
        normalized["review_timestamp"] = reviewed_at.isoformat()
        normalized["review_identity"] = review_identity
        normalized_rows.append(normalized)
    return normalized_rows


def stage_completed_review_file(
    review_path: str | Path,
    *,
    product_master: pd.DataFrame,
    staging_directory: str | Path,
) -> ReviewIntakeResult:
    """Preserve, validate, and immutably stage a completed review CSV."""
    source = Path(review_path)
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ReviewIntakeError("Completed review input must be an existing CSV")
    if "Itemcode" not in product_master:
        raise ReviewIntakeError("Product Master must contain Itemcode")
    content = source.read_bytes()
    source_hash = _sha256_bytes(content)
    staging = Path(staging_directory)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name)
    raw_path = staging / "raw" / f"{source_hash}_{safe_name}"
    _write_bytes_exclusive(raw_path, content)

    normalized_directory = staging / "normalized"
    timestamp = datetime.now(timezone.utc)
    intake_id = (
        f"review-intake-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{source_hash[:12]}"
    )
    audit_path = staging / "audits" / f"{intake_id}.json"
    try:
        frame = pd.read_csv(
            source,
            dtype="string",
            keep_default_na=False,
        )
        normalized_rows = _validate_review_rows(
            frame,
            product_master,
            _load_existing_identities(normalized_directory),
        )
        record_paths: list[Path] = []
        for normalized in normalized_rows:
            immutable = {
                **normalized,
                "source_submission_sha256": source_hash,
                "intake_id": intake_id,
            }
            record_hash = _canonical_sha256(immutable)
            immutable["review_record_sha256"] = record_hash
            destination = normalized_directory / f"{record_hash}.json"
            payload = (
                json.dumps(
                    immutable,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            _write_bytes_exclusive(destination, payload)
            record_paths.append(destination)
        audit = {
            "intake_id": intake_id,
            "status": "ACCEPTED_TO_REVIEW_STAGING",
            "source_filename": source.name,
            "source_submission_sha256": source_hash,
            "raw_submission_path": str(raw_path),
            "rows_received": int(len(frame)),
            "rows_staged": len(record_paths),
            "normalized_record_hashes": [
                path.stem for path in record_paths
            ],
            "training_data_updated": False,
            "created_at": timestamp.isoformat(),
        }
        _atomic_json(audit, audit_path)
        return ReviewIntakeResult(
            raw_submission_path=raw_path,
            normalized_record_paths=tuple(record_paths),
            audit_path=audit_path,
            audit=audit,
        )
    except Exception as error:
        rejected_audit = {
            "intake_id": intake_id,
            "status": "REJECTED",
            "source_filename": source.name,
            "source_submission_sha256": source_hash,
            "raw_submission_path": str(raw_path),
            "error_type": type(error).__name__,
            "error": str(error),
            "training_data_updated": False,
            "created_at": timestamp.isoformat(),
        }
        _atomic_json(rejected_audit, audit_path)
        raise
