"""Durable run summaries and validated download access."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sku_mapping.competitors.discovery import COMPETITOR_EXPORT_COLUMNS
from sku_mapping.config import PipelineConfig
from sku_mapping.exports.run_outputs import SKU_MAPPING_COLUMNS
from sku_mapping.learning.store import LearningStore


@dataclass(frozen=True)
class DownloadArtifact:
    """Validated downloadable bytes without exposing a server path."""

    key: str
    filename: str
    media_type: str
    content: bytes


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

_DOWNLOAD_SUFFIXES = {
    "sku_mapping": "MAPPED",
    "competitor_offers": "COMPETITORS",
    "run_summary": "RUN_SUMMARY",
    "monitoring_report": "MONITORING",
}


def _short_model_label(model_id: object) -> str:
    """Compress a registry model id into a filename-safe version label."""
    text = str(model_id or "").strip()
    if not text:
        return "no-model"
    text = re.sub(r"-?\d{8}T\d{6}Z", "", text)
    text = re.sub(r"-?matcher$", "", text).strip("-_")
    return (_FILENAME_SAFE.sub("_", text)[:40].strip("_-")) or "no-model"


def download_filename(run: dict[str, object], key: str, artifact_path: Path) -> str:
    """Business-facing download name: upload stem, role, date, model version.

    The on-disk artifact keeps its run-scoped name; only the name offered to
    the browser changes, e.g. ``weekly_dump_MAPPED_20260810_ranked-v5-cal.csv``.
    """
    source = str(run.get("source_filename") or "").strip()
    if source:
        stem = Path(source).stem
    else:
        stem = str(run.get("run_id") or "run")[:12]
    stem = _FILENAME_SAFE.sub("_", stem)[:60].strip("_-") or "run"
    parts = [stem, _DOWNLOAD_SUFFIXES.get(key, key.upper())]
    date_match = re.match(
        r"(\d{4})-?(\d{2})-?(\d{2})", str(run.get("started_at") or "")
    )
    if date_match:
        parts.append("".join(date_match.groups()))
    parts.append(_short_model_label(run.get("model_id")))
    return "_".join(parts) + artifact_path.suffix.lower()


class DashboardRunService:
    """Read run state from SQLite and return only validated artifacts."""

    def __init__(
        self, config: PipelineConfig, store: LearningStore
    ) -> None:
        self.config = config
        self.store = store

    def runs(self) -> list[dict[str, object]]:
        return self.store.list_pipeline_runs(limit=200)

    def run_summary(self, run_id: str) -> dict[str, object]:
        run = self.store.get_pipeline_run(run_id)
        if run is None:
            raise ValueError("Run not found")
        output_paths = run.get("output_paths", {})
        dashboard_outputs_complete = all(
            output_paths.get(key)
            for key in ("run_summary", "sku_mapping", "competitor_offers")
        )
        summary = {
            "run_id": run_id,
            "status": run["status"],
            "input_rows": run["source_row_count"],
            "unique_offers": run["unique_offer_count"],
            "deployment_mode": run["deployment_mode"],
            "model_id": run["model_id"],
            "llm_model_id": run["llm_model_id"],
            "threshold": run["threshold"],
            "runtime": sum(
                float(value)
                for value in run.get("stage_runtimes", {}).values()
                if isinstance(value, (int, float))
            ),
            "errors": run["error_summary"],
            "dashboard_outputs_complete": dashboard_outputs_complete,
            "summary_source": "learning_store",
            **run.get("run_metadata", {}),
        }
        summary_artifacts = (
            ("run_summary", "dashboard_run_summary"),
            ("unified_statistics", "unified_inference_statistics"),
        )
        for artifact_key, source_name in summary_artifacts:
            raw_path = output_paths.get(artifact_key)
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_file() or not self._inside_allowed_root(path):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            summary.update(payload)
            summary["summary_source"] = source_name
            break
        # SQLite is the durable source of truth after a browser refresh.
        # Generated summaries may be from an older schema and must not
        # overwrite the persisted canonical identity count or its provenance.
        summary["unique_offers"] = run["unique_offer_count"]
        summary.update(run.get("run_metadata", {}))
        return summary

    def downloads(self, run_id: str) -> list[DownloadArtifact]:
        run = self.store.get_pipeline_run(run_id)
        if run is None:
            raise ValueError("Run not found")
        artifacts = []
        specs = {
            "sku_mapping": (
                "text/csv",
                SKU_MAPPING_COLUMNS,
            ),
            "competitor_offers": (
                "text/csv",
                COMPETITOR_EXPORT_COLUMNS,
            ),
            "run_summary": ("application/json", None),
            "monitoring_report": ("application/json", None),
        }
        for key, (media_type, columns) in specs.items():
            raw_path = run.get("output_paths", {}).get(key)
            if not raw_path:
                continue
            path = Path(raw_path)
            if (
                not path.is_file()
                or not self._inside_allowed_root(path)
                or path.stat().st_size == 0
            ):
                continue
            try:
                if columns is not None:
                    observed = tuple(
                        pd.read_csv(
                            path, nrows=0, encoding="utf-8-sig"
                        ).columns
                    )
                    if observed != columns:
                        continue
                else:
                    json.loads(path.read_text(encoding="utf-8"))
                content = path.read_bytes()
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            artifacts.append(
                DownloadArtifact(
                    key=key,
                    filename=download_filename(run, key, path),
                    media_type=media_type,
                    content=content,
                )
            )
        return artifacts

    def _inside_allowed_root(self, path: Path) -> bool:
        resolved = path.resolve()
        roots = (
            self.config.dashboard.output_directory.resolve(),
            self.config.output.output_dir.resolve(),
            self.config.shadow_mode.output_directory.resolve(),
        )
        return any(
            resolved == root or root in resolved.parents for root in roots
        )
