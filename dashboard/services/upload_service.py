"""Secure upload validation and controlled run-scoped staging."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sku_mapping.config import DashboardConfig
from sku_mapping.data.validators import validate_clickflyer


class UploadValidationError(ValueError):
    """Raised for safe, user-displayable upload validation failures."""


@dataclass(frozen=True)
class ValidatedUpload:
    """Validated upload identity before it is staged or processed."""

    original_filename: str
    sanitized_filename: str
    extension: str
    size_bytes: int
    source_file_hash: str
    content: bytes


def sanitize_upload_filename(filename: str) -> str:
    """Remove directories and unsafe characters while preserving extension."""
    basename = Path(str(filename).replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem)
    stem = stem.strip("._-")[:100] or "upload"
    suffix = Path(basename).suffix.lower()
    return f"{stem}{suffix}"


class DashboardUploadService:
    """Validate bytes and persist them only under a controlled run directory."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config

    def validate(self, filename: str, content: bytes) -> ValidatedUpload:
        """Validate extension, size, basic signature, and non-empty bytes."""
        sanitized = sanitize_upload_filename(filename)
        extension = Path(sanitized).suffix.lower()
        if extension not in self.config.allowed_extensions:
            allowed = ", ".join(self.config.allowed_extensions)
            raise UploadValidationError(
                f"Unsupported file type. Allowed: {allowed}"
            )
        if not content:
            raise UploadValidationError("Uploaded file is empty")
        maximum = self.config.max_upload_size_mb * 1024 * 1024
        if len(content) > maximum:
            raise UploadValidationError(
                f"Upload exceeds the {self.config.max_upload_size_mb} MB limit"
            )
        if extension == ".csv" and b"\x00" in content[:8192]:
            raise UploadValidationError(
                "CSV contains binary null bytes and cannot be processed"
            )
        if extension == ".xlsx" and not content.startswith(b"PK"):
            raise UploadValidationError(
                "File extension is .xlsx but content is not an Excel workbook"
            )
        if extension == ".xls" and not content.startswith(
            bytes.fromhex("D0CF11E0A1B11AE1")
        ):
            raise UploadValidationError(
                "File extension is .xls but content is not an Excel workbook"
            )
        return ValidatedUpload(
            original_filename=Path(filename).name,
            sanitized_filename=sanitized,
            extension=extension,
            size_bytes=len(content),
            source_file_hash=hashlib.sha256(content).hexdigest(),
            content=content,
        )

    def stage(self, upload: ValidatedUpload, *, run_id: str) -> Path:
        """Atomically write upload bytes inside one controlled run directory."""
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise UploadValidationError("Generated run ID is unsafe")
        root = self.config.input_directory.resolve()
        directory = (root / run_id).resolve()
        if root not in directory.parents:
            raise UploadValidationError("Upload directory escaped configured root")
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / upload.sanitized_filename
        if destination.exists():
            raise UploadValidationError(
                "This run already contains a staged upload"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=directory,
            prefix=".upload.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(upload.content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @staticmethod
    def read_and_validate(path: Path) -> pd.DataFrame:
        """Read CSV/Excel as data only and validate the ClickFlyer schema."""
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(
                    path,
                    low_memory=False,
                    dtype={"offerid": "string"},
                )
            else:
                frame = pd.read_excel(
                    path, dtype={"offerid": "string"}
                )
        except Exception as error:
            raise UploadValidationError(
                "The uploaded file could not be read as a data table"
            ) from error
        return validate_clickflyer(frame)

