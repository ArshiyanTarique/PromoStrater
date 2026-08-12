"""Schema-version and persistence tests for the Phase 7A store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sku_mapping.learning.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
)
from sku_mapping.learning.store import LearningStore


def test_learning_store_creates_required_schema(tmp_path: Path) -> None:
    store = LearningStore(tmp_path / "learning.db")
    assert store.schema_version == CURRENT_SCHEMA_VERSION
    assert {
        "pipeline_runs",
        "offer_decisions",
        "predictions",
        "review_sessions",
        "human_reviews",
        "automated_labels",
        "model_versions",
        "training_datasets",
        "schema_migrations",
    }.issubset(store.table_names())
    connection = sqlite3.connect(store.path)
    try:
        training_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(training_datasets)"
            )
        }
        run_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(pipeline_runs)"
            )
        }
    finally:
        connection.close()
    assert {
        "feature_schema_version",
        "artifact_path",
        "artifact_sha256",
        "manifest_path",
        "included_automated_label_ids_json",
        "inclusion_policy_json",
        "override_record_json",
        "evaluation_artifact_path",
        "evaluation_artifact_sha256",
    }.issubset(training_columns)
    assert "run_metadata_json" in run_columns


def test_existing_version_one_database_migrates_forward(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v1.db"
    connection = sqlite3.connect(path)
    connection.executescript(MIGRATIONS[1])
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    store = LearningStore(path)
    assert store.schema_version == CURRENT_SCHEMA_VERSION
    connection = sqlite3.connect(path)
    try:
        indexes = {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'index'
                """
            )
        }
    finally:
        connection.close()
    assert "idx_reviews_session_unanswered" in indexes


def _seed_job(store: LearningStore) -> None:
    """Create one job, which also proves the schema is migrated."""
    store.upsert_pipeline_run(
        {"run_id": "run-1", "source_path": "upload.csv", "status": "RUNNING"}
    )
    store.create_processing_job(
        {
            "job_id": "job-1",
            "state": "RUNNING",
            "run_id": "run-1",
            "source_filename": "upload.csv",
            "source_file_hash": "a" * 64,
        }
    )


def test_job_status_reads_do_not_wait_on_the_write_lock(tmp_path: Path) -> None:
    """The dashboard polls job state while a bulk write holds the lock.

    Sharing one lock between readers and writers froze the live run panel -
    and its elapsed clock - for the whole of every stage. Holding the write
    lock here stands in for a bulk insert in flight: if a status read still
    acquired it, the worker thread would block until the timeout.
    """
    import threading

    store = LearningStore(tmp_path / "learning.db")
    _seed_job(store)

    observed: list[object] = []
    failures: list[BaseException] = []

    def poll() -> None:
        try:
            observed.append(store.active_processing_jobs())
            observed.append(store.get_processing_job("job-1"))
            observed.append(store.latest_processing_job())
            observed.append(store.latest_processing_job_for_run("run-1"))
            observed.append(store.active_jobs_for_source_hash("a" * 64))
        except BaseException as error:  # pragma: no cover - failure detail
            failures.append(error)

    with store._lock:
        reader = threading.Thread(target=poll, name="status-poll", daemon=True)
        reader.start()
        reader.join(timeout=10.0)
        blocked = reader.is_alive()

    assert not blocked, "a job-status read waited on the write lock"
    assert not failures, f"status read raised: {failures}"
    assert len(observed) == 5
    assert observed[0] and observed[0][0]["job_id"] == "job-1"
    assert observed[1] is not None and observed[1]["state"] == "RUNNING"
    assert observed[2] is not None and observed[2]["job_id"] == "job-1"
    assert observed[3] is not None and observed[3]["run_id"] == "run-1"
    assert observed[4] and observed[4][0]["job_id"] == "job-1"


def test_reads_before_first_migration_still_take_the_migrating_path(
    tmp_path: Path,
) -> None:
    """A brand-new store must migrate on its first read, not skip it."""
    store = LearningStore(tmp_path / "fresh.db")
    assert store._migrated is False

    assert store.active_processing_jobs() == []

    assert store._migrated is True
    assert store.schema_version == CURRENT_SCHEMA_VERSION
