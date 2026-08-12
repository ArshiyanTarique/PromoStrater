"""Run-scoped dashboard exports with explicit schema validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from sku_mapping.competitors.discovery import (
    COMPETITOR_EXPORT_COLUMNS,
    COMPETITOR_LONG_COLUMNS,
)
from sku_mapping.exports.business_outputs import SKU_MAPPING_COLUMNS


@dataclass(frozen=True)
class RunOutputBundle:
    """Validated run-scoped downloadable artifacts and their hashes."""

    paths: dict[str, Path]
    hashes: dict[str, str]


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            )
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_csv(path: Path, columns: tuple[str, ...]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Output is missing or empty: {path.name}")
    observed = tuple(
        pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    )
    if observed != columns:
        raise ValueError(
            f"{path.name} schema mismatch: expected {columns}, got {observed}"
        )


def write_run_outputs(
    *,
    run_id: str,
    sku_mapping_export: pd.DataFrame,
    competitor_export: pd.DataFrame,
    competitor_long_format: pd.DataFrame,
    competitor_long_format_path: str | Path | None = None,
    summary: Mapping[str, Any],
    output_root: str | Path,
    monitoring_path: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> RunOutputBundle:
    """Write validated business outputs plus the internal competitor audit."""
    if not run_id or any(character in run_id for character in "\\/:*?\"<>|"):
        raise ValueError("run_id is unsafe for output filenames")
    if tuple(sku_mapping_export.columns) != SKU_MAPPING_COLUMNS:
        raise ValueError("SKU mapping export schema is incompatible")
    if tuple(competitor_export.columns) != COMPETITOR_EXPORT_COLUMNS:
        raise ValueError("Competitor export schema is incompatible")
    staged_long_path = (
        Path(competitor_long_format_path)
        if competitor_long_format_path is not None
        else None
    )
    if staged_long_path is None:
        if tuple(competitor_long_format.columns) != COMPETITOR_LONG_COLUMNS:
            raise ValueError("Competitor long-form schema is incompatible")
    else:
        _validate_csv(staged_long_path, COMPETITOR_LONG_COLUMNS)
    if sku_mapping_export["entity_id"].duplicated().any():
        raise ValueError("SKU mapping export contains duplicate entity IDs")
    if competitor_export["master_sku"].duplicated().any():
        raise ValueError("Competitor export contains duplicate Master SKUs")
    target_skus = set(
        sku_mapping_export.loc[
            sku_mapping_export["matched_master_sku"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne(""),
            "matched_master_sku",
        ]
        .astype(str)
        .str.strip()
    )
    exported_skus = set(
        competitor_export["master_sku"].astype(str).str.strip()
    )
    if target_skus != exported_skus:
        raise ValueError(
            "Competitor export targets do not equal distinct mapped Master SKUs"
        )

    directory = Path(output_root) / run_id
    mapping_path = directory / f"sku_mapping_{run_id}.csv"
    competitor_path = directory / f"competitor_offers_{run_id}.csv"
    competitor_long_path = (
        directory / f"competitor_long_form_{run_id}.csv"
    )
    summary_path = directory / f"run_summary_{run_id}.json"
    _atomic_csv(sku_mapping_export, mapping_path)
    if progress is not None:
        progress(1, 4, "SKU mapping file written")
    _atomic_csv(competitor_export, competitor_path)
    if progress is not None:
        progress(2, 4, "Competitor summary file written")
    if staged_long_path is None:
        _atomic_csv(competitor_long_format, competitor_long_path)
    else:
        competitor_long_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_long_path, competitor_long_path)
    if progress is not None:
        progress(3, 4, "Competitor audit file written")
    _atomic_json(summary, summary_path)
    if progress is not None:
        progress(4, 4, "Run summary written")
    _validate_csv(mapping_path, SKU_MAPPING_COLUMNS)
    _validate_csv(competitor_path, COMPETITOR_EXPORT_COLUMNS)
    _validate_csv(competitor_long_path, COMPETITOR_LONG_COLUMNS)

    paths = {
        "sku_mapping": mapping_path,
        "competitor_offers": competitor_path,
        "competitor_long_form": competitor_long_path,
        "run_summary": summary_path,
    }
    if monitoring_path is not None and monitoring_path.is_file():
        paths["monitoring_report"] = monitoring_path
    return RunOutputBundle(
        paths=paths,
        hashes={key: _sha256(path) for key, path in paths.items()},
    )
