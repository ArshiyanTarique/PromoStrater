"""Repository-location portability and path-safety tests."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from sku_mapping.config import load_config
from sku_mapping.learning.store import LearningStore
from sku_mapping.paths import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    find_project_root,
    portable_repository_path,
    resolve_portable_path,
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def test_project_root_is_detected_from_source_not_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert find_project_root(Path(__file__)) == PROJECT_ROOT
    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "config" / "default.yaml"

    from dashboard.bootstrap import PROJECT_ROOT as dashboard_root

    assert dashboard_root == PROJECT_ROOT


def test_default_runtime_paths_are_inside_project_root() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    paths = (
        config.data.processed_dir,
        config.model.model_dir,
        config.model.model_path,
        config.output.output_dir,
        config.retraining.snapshot_directory,
        config.retraining.challenger_directory,
        config.retraining.comparison_report_directory,
        config.embedding.cache_path,
        config.llm_review.cache_path,
        config.learning_store.database_path,
        config.learning_store.csv_export_directory,
        config.dashboard.input_directory,
        config.dashboard.output_directory,
        config.shadow_mode.registry_path,
        config.shadow_mode.output_directory,
        config.shadow_mode.review_staging_directory,
        config.shadow_mode.challenge_set_directory,
    )
    assert all(_inside(path, PROJECT_ROOT) for path in paths)


def test_repository_paths_round_trip_portably() -> None:
    artifact = PROJECT_ROOT / "outputs" / "run" / "result.csv"
    stored = portable_repository_path(artifact)
    assert stored == "outputs/run/result.csv"
    assert resolve_portable_path(stored) == artifact


def test_external_configurable_path_remains_absolute(tmp_path: Path) -> None:
    stored = portable_repository_path(tmp_path / "external.csv")
    assert Path(stored).is_absolute()
    assert resolve_portable_path(stored) == (tmp_path / "external.csv").resolve()


def test_learning_store_persists_repository_output_paths_portably(
    tmp_path: Path,
) -> None:
    store = LearningStore(tmp_path / "learning.db")
    artifact = PROJECT_ROOT / "outputs" / "example" / "result.csv"
    store.upsert_pipeline_run(
        {
            "run_id": "portable-run",
            "output_paths": {"result": artifact},
        }
    )

    connection = sqlite3.connect(store.path)
    try:
        raw = connection.execute(
            "SELECT output_paths_json FROM pipeline_runs WHERE run_id = ?",
            ("portable-run",),
        ).fetchone()[0]
    finally:
        connection.close()

    assert json.loads(raw) == {"result": "outputs/example/result.csv"}
    assert store.get_pipeline_run("portable-run")["output_paths"] == {
        "result": str(artifact)
    }
