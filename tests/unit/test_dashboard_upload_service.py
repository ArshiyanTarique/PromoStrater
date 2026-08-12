"""Security and schema tests for dashboard upload handling."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from dashboard.services.upload_service import (
    DashboardUploadService,
    UploadValidationError,
    sanitize_upload_filename,
)
from sku_mapping.config import load_config


def _service(tmp_path: Path, *, maximum_mb: int = 1) -> DashboardUploadService:
    config = load_config("config/default.yaml").dashboard
    return DashboardUploadService(
        replace(
            config,
            input_directory=tmp_path / "uploads",
            max_upload_size_mb=maximum_mb,
        )
    )


def test_filename_is_sanitized_and_staged_inside_run(tmp_path: Path) -> None:
    service = _service(tmp_path)
    content = Path("tests/fixtures/clickflyer_valid.csv").read_bytes()
    upload = service.validate("../../evil report.csv", content)
    assert upload.sanitized_filename == "evil_report.csv"
    destination = service.stage(upload, run_id="run-safe")
    assert destination.parent == tmp_path / "uploads" / "run-safe"
    assert destination.read_bytes() == content


@pytest.mark.parametrize("name", ["payload.exe", "dump.csv.exe", "no_suffix"])
def test_unsupported_extensions_are_rejected(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(UploadValidationError, match="Unsupported"):
        _service(tmp_path).validate(name, b"not empty")


def test_size_and_signature_checks_are_enforced(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(UploadValidationError, match="exceeds"):
        service.validate("large.csv", b"a" * (1024 * 1024 + 1))
    with pytest.raises(UploadValidationError, match="not an Excel"):
        service.validate("fake.xlsx", b"plain text")


def test_csv_and_excel_schema_are_validated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    csv_upload = service.validate(
        "offers.csv",
        Path("tests/fixtures/clickflyer_valid.csv").read_bytes(),
    )
    csv_path = service.stage(csv_upload, run_id="csv-run")
    assert len(service.read_and_validate(csv_path)) == 1

    frame = pd.read_csv("tests/fixtures/clickflyer_valid.csv")
    buffer = BytesIO()
    frame.to_excel(buffer, index=False)
    excel_upload = service.validate("offers.xlsx", buffer.getvalue())
    excel_path = service.stage(excel_upload, run_id="excel-run")
    assert len(service.read_and_validate(excel_path)) == 1


def test_missing_required_columns_are_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    upload = service.validate("bad.csv", b"Offer Name\nOnly one column\n")
    path = service.stage(upload, run_id="bad-run")
    with pytest.raises(ValueError, match="missing required"):
        service.read_and_validate(path)


def test_filename_sanitizer_never_preserves_directories() -> None:
    assert "/" not in sanitize_upload_filename("../a/b.csv")
    assert "\\" not in sanitize_upload_filename("..\\a\\b.csv")

