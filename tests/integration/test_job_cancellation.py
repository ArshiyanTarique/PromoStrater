"""Cooperative cancellation, orphan recovery, and job-slot release."""

from __future__ import annotations

import os
import socket
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dashboard.services.job_manager import (
    ActiveJobExistsError,
    DashboardJobManager,
)
from dashboard.services.job_state import (
    CANCELLED_RUN_STATUS,
    ORPHANED_RUN_STATUS,
    CancellationToken,
    JobState,
    RunCancelled,
)
from dashboard.services.registry_service import DashboardRegistryService
from dashboard.services.run_service import DashboardRunService
from dashboard.services.processing_service import (
    DashboardProcessRequest,
    DashboardProcessingService,
    ProcessingCancelledError,
)
from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.learning.migrations import CURRENT_SCHEMA_VERSION
from sku_mapping.learning.store import LearningStore

from tests.registered_models import registered_model_id  # noqa: E402

MODEL_ID = registered_model_id()

#: Stages that must each be independently cancellable. ``exports`` and
#: ``persistence`` are deliberately absent: once the last checkpoint before
#: writing is passed, the run commits its outputs rather than leaving them
#: half-written. That commit phase is covered by its own test below.
CANCELLABLE_STAGES = (
    "input_loading",
    "canonicalisation",
    "inference",
    "sku_mapping",
    "competitor_discovery",
)


def _config(tmp_path: Path):
    base = load_config("config/default.yaml")
    return replace(
        base,
        output=replace(base.output, output_dir=tmp_path / "legacy_outputs"),
        dashboard=replace(
            base.dashboard,
            input_directory=tmp_path / "uploads",
            output_directory=tmp_path / "dashboard_outputs",
        ),
        learning_store=replace(
            base.learning_store,
            database_path=tmp_path / "learning.db",
            csv_export_directory=tmp_path / "exports",
        ),
        shadow_mode=replace(
            base.shadow_mode,
            output_directory=tmp_path / "shadow",
            review_staging_directory=tmp_path / "reviews",
            challenge_set_directory=tmp_path / "challenge",
        ),
        llm_review=replace(
            base.llm_review, cache_path=tmp_path / "llm.sqlite3"
        ),
    )


def _csv_bytes(marker: str = "a") -> bytes:
    base = pd.read_csv("tests/fixtures/clickflyer_valid.csv")
    rows = []
    for number, weight in enumerate((270, 400, 500)):
        row = base.iloc[0].copy()
        row["offerid"] = f"cancel-{marker}-{number}"
        row["Offer Name"] = f"Al Kabeer Chicken Nuggets {weight}g"
        row["Base Packsize"] = f"{weight}g"
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _request(content: bytes) -> DashboardProcessRequest:
    return DashboardProcessRequest(
        filename="clickflyer.csv",
        content=content,
        deployment_mode=MLDeploymentMode.ASSISTED,
        model_id=MODEL_ID,
        allow_duplicate=True,
    )


@pytest.mark.parametrize("stage", CANCELLABLE_STAGES)
def test_cancelling_during_each_stage_stops_cleanly(
    tmp_path: Path, stage: str
) -> None:
    """Each major stage observes the token and stops at its checkpoint."""
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    token = CancellationToken()
    observed: list[str] = []

    def progress(update) -> None:
        if update.stage_key is not None:
            observed.append(update.stage_key)
        if update.stage_key == stage:
            token.request()

    with pytest.raises(ProcessingCancelledError) as caught:
        service.process(
            _request(_csv_bytes()), progress=progress, cancel_token=token
        )

    error = caught.value
    assert error.run_id is not None
    # The run never reports completion, and the stopping point is recorded.
    run = LearningStore(config.learning_store.database_path).get_pipeline_run(
        error.run_id
    )
    assert run is not None
    assert run["status"] == CANCELLED_RUN_STATUS
    assert not str(run["status"]).startswith("COMPLETED")
    assert stage in observed


def test_cancellation_preserves_partial_logs_and_outputs(
    tmp_path: Path,
) -> None:
    """A cancelled run keeps an auditable record of where it stopped."""
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    token = CancellationToken()

    def progress(update) -> None:
        if update.stage_key == "inference":
            token.request()

    with pytest.raises(ProcessingCancelledError) as caught:
        service.process(
            _request(_csv_bytes()), progress=progress, cancel_token=token
        )

    run_id = caught.value.run_id
    assert run_id is not None
    report = config.dashboard.output_directory / run_id / (
        "cancellation_report.json"
    )
    assert report.is_file()
    import json

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "CANCELLED"
    assert payload["cancellation_type"] == "cooperative_checkpoint"
    assert payload["cancelled_during_stage"] is not None

    store = LearningStore(config.learning_store.database_path)
    run = store.get_pipeline_run(run_id)
    assert run is not None
    assert "cancellation_log" in run["output_paths"]


def test_cancel_during_commit_phase_completes_rather_than_half_writing(
    tmp_path: Path,
) -> None:
    """Past the final checkpoint the run commits its outputs intact."""
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    token = CancellationToken()

    def progress(update) -> None:
        if update.stage_key in {"exports", "persistence"}:
            token.request()

    result = service.process(
        _request(_csv_bytes()), progress=progress, cancel_token=token
    )

    assert result.status.startswith("COMPLETED")
    store = LearningStore(config.learning_store.database_path)
    run = store.get_pipeline_run(result.run_id)
    assert run is not None
    assert str(run["status"]).startswith("COMPLETED")


def test_cancelling_a_completed_job_is_rejected_and_changes_nothing(
    tmp_path: Path,
) -> None:
    """A finished job keeps its terminal state when cancel arrives late."""
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(store, DashboardProcessingService(config))

    job = manager.submit(
        _request(_csv_bytes()), source_file_hash="f" * 64, start=False
    )
    store.claim_processing_job(
        job.job_id,
        run_id="run-complete",
        worker_pid=1234,
        worker_host="test",
        worker_boot_id="boot",
    )
    store.finish_processing_job(job.job_id, state=JobState.COMPLETED.value)

    after = manager.cancel(job.job_id)

    assert after is not None
    assert after.state is JobState.COMPLETED
    assert after.cancel_requested is False


def test_cancelling_a_queued_job_never_starts_a_stage(tmp_path: Path) -> None:
    """A queued job cancels immediately and is not claimable afterwards."""
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(store, DashboardProcessingService(config))

    job = manager.submit(
        _request(_csv_bytes()), source_file_hash="a" * 64, start=False
    )
    assert job.state is JobState.QUEUED

    cancelled = manager.cancel(job.job_id)

    assert cancelled is not None
    assert cancelled.state is JobState.CANCELLED
    # A worker arriving late must not resurrect a cancelled job.
    assert not store.claim_processing_job(
        job.job_id,
        run_id="late",
        worker_pid=1,
        worker_host="test",
        worker_boot_id="boot",
    )
    assert manager.active_job() is None


def test_orphaned_running_job_from_a_dead_worker_is_recovered(
    tmp_path: Path,
) -> None:
    """A RUNNING job left by a previous process must not stay RUNNING.

    Death is proven by the stale heartbeat, not by the foreign boot id: a
    previous interpreter leaves both behind, but only the silence tells the
    two apart from an interpreter that is still running.
    """
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(
        store,
        DashboardProcessingService(config),
        heartbeat_expiry_seconds=0.0,
    )

    job = manager.submit(
        _request(_csv_bytes()), source_file_hash="b" * 64, start=False
    )
    store.upsert_pipeline_run(
        {
            "run_id": "orphan-run",
            "deployment_mode": "assisted",
            "status": "PROCESSING",
        }
    )
    # A different boot id is exactly what a previous interpreter leaves behind.
    store.claim_processing_job(
        job.job_id,
        run_id="orphan-run",
        worker_pid=999999,
        worker_host="test",
        worker_boot_id="a-previous-process",
    )
    assert manager.get(job.job_id).state is JobState.RUNNING
    time.sleep(0.01)

    recovered = manager.recover_orphaned_jobs()

    assert [item.job_id for item in recovered] == [job.job_id]
    assert manager.get(job.job_id).state is JobState.FAILED
    assert manager.active_job() is None
    run = store.get_pipeline_run("orphan-run")
    assert run["status"] == ORPHANED_RUN_STATUS


def test_orphaned_cancelling_job_is_recovered_as_cancelled(
    tmp_path: Path,
) -> None:
    """A cancel already requested is honoured when the worker disappears."""
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(
        store,
        DashboardProcessingService(config),
        heartbeat_expiry_seconds=0.0,
    )

    job = manager.submit(
        _request(_csv_bytes()), source_file_hash="c" * 64, start=False
    )
    store.claim_processing_job(
        job.job_id,
        run_id=None,
        worker_pid=999999,
        worker_host="test",
        worker_boot_id="a-previous-process",
    )
    store.request_job_cancellation(job.job_id)
    time.sleep(0.01)

    manager.recover_orphaned_jobs()

    assert manager.get(job.job_id).state is JobState.CANCELLED
    assert manager.active_job() is None


def test_a_second_process_leaves_another_live_worker_alone(
    tmp_path: Path,
) -> None:
    """A run owned by another *running* interpreter must survive.

    Opening a second dashboard, or restarting one while a run is in flight,
    used to mark that run FAILED the moment the new process swept: a foreign
    boot id counted as death on its own. The worker was untouched and kept
    heartbeating, so the failure was pure fiction - and it cost real runs
    mid-pipeline. A fresh heartbeat now protects the job.
    """
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    # Default expiry: the heartbeat written by the claim below is fresh.
    manager = DashboardJobManager(store, DashboardProcessingService(config))

    job = manager.submit(
        _request(_csv_bytes()), source_file_hash="e" * 64, start=False
    )
    store.upsert_pipeline_run(
        {
            "run_id": "live-run",
            "deployment_mode": "assisted",
            "status": "PROCESSING",
        }
    )
    # Another interpreter owns this job and is still working on it.
    store.claim_processing_job(
        job.job_id,
        run_id="live-run",
        worker_pid=999999,
        worker_host="test",
        worker_boot_id="a-different-live-process",
    )

    recovered = manager.recover_orphaned_jobs()

    assert recovered == []
    assert manager.get(job.job_id).state is JobState.RUNNING
    assert store.get_pipeline_run("live-run")["status"] == "PROCESSING"
    # The slot stays held, so this process cannot start a competing run.
    active = manager.active_job()
    assert active is not None and active.job_id == job.job_id


def test_a_worker_that_dies_mid_session_stops_holding_the_slot(
    tmp_path: Path,
) -> None:
    """Recovery must keep working after startup, not only at first use.

    A heartbeat goes stale while the process runs, so a boot-time-only sweep
    would leave the dead job occupying the submission slot forever.
    """
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(
        store,
        DashboardProcessingService(config),
        heartbeat_expiry_seconds=0.0,
    )

    job = manager.submit(
        _request(_csv_bytes()), source_file_hash="f" * 63 + "0", start=False
    )
    store.claim_processing_job(
        job.job_id,
        run_id=None,
        worker_pid=999999,
        worker_host="test",
        worker_boot_id="a-worker-that-then-died",
    )
    time.sleep(0.01)

    # No explicit recovery call: querying the slot is what the UI does.
    assert manager.active_job() is None
    assert manager.get(job.job_id).state is JobState.FAILED


def test_expired_heartbeat_marks_a_job_orphaned(tmp_path: Path) -> None:
    """Liveness comes from the heartbeat, not from a reusable pid."""
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(
        store,
        DashboardProcessingService(config),
        heartbeat_expiry_seconds=0.0,
    )
    job = manager.submit(
        _request(_csv_bytes()), source_file_hash="d" * 64, start=False
    )
    from dashboard.services import job_manager as job_manager_module

    store.claim_processing_job(
        job.job_id,
        run_id=None,
        worker_pid=1,
        worker_host="test",
        worker_boot_id=job_manager_module.WORKER_BOOT_ID,
    )
    time.sleep(0.01)

    manager.recover_orphaned_jobs()

    assert manager.get(job.job_id).state is JobState.FAILED


def test_new_run_can_start_immediately_after_cancellation(
    tmp_path: Path,
) -> None:
    """Cancelling frees the slot at once; no cooldown, no stale block."""
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(store, DashboardProcessingService(config))

    first = manager.submit(
        _request(_csv_bytes()), source_file_hash="e" * 64, start=False
    )
    with pytest.raises(ActiveJobExistsError):
        manager.submit(
            _request(_csv_bytes()), source_file_hash="e" * 64, start=False
        )

    manager.cancel(first.job_id)

    second = manager.submit(
        _request(_csv_bytes()), source_file_hash="e" * 64, start=False
    )
    assert second.job_id != first.job_id
    assert second.state is JobState.QUEUED
    assert manager.get(first.job_id).state is JobState.CANCELLED


def test_duplicate_prevention_only_blocks_a_genuine_active_job(
    tmp_path: Path,
) -> None:
    """Terminal jobs never hold the slot for the same uploaded bytes."""
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    manager = DashboardJobManager(store, DashboardProcessingService(config))
    source_hash = "1" * 64

    job = manager.submit(
        _request(_csv_bytes()), source_file_hash=source_hash, start=False
    )
    assert len(store.active_jobs_for_source_hash(source_hash)) == 1

    manager.cancel(job.job_id)

    assert store.active_jobs_for_source_hash(source_hash) == []
    assert manager.active_job() is None


def test_cancellation_leaves_the_database_uncorrupted(tmp_path: Path) -> None:
    """Cancelling mid-run must not damage schema or referential integrity."""
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    token = CancellationToken()

    def progress(update) -> None:
        if update.stage_key == "competitor_discovery":
            token.request()

    with pytest.raises(ProcessingCancelledError):
        service.process(
            _request(_csv_bytes()), progress=progress, cancel_token=token
        )

    database = config.learning_store.database_path
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
        assert connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall() == []
    finally:
        connection.close()
    store = LearningStore(database)
    assert "processing_jobs" in store.table_names()
    assert store.schema_version == CURRENT_SCHEMA_VERSION


def test_background_worker_cancels_gracefully_without_being_killed(
    tmp_path: Path,
) -> None:
    """The worker thread observes the flag and exits on its own terms."""
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    reached_competitors = threading.Event()
    service = DashboardProcessingService(config)
    original = service.process

    def instrumented(request, *, progress=None, cancel_token=None):
        def watched(update) -> None:
            if update.stage_key == "inference":
                reached_competitors.set()
            if progress is not None:
                progress(update)

        return original(
            request, progress=watched, cancel_token=cancel_token
        )

    service.process = instrumented  # type: ignore[method-assign]
    manager = DashboardJobManager(store, service)

    job = manager.submit(_request(_csv_bytes()), source_file_hash="9" * 64)
    assert reached_competitors.wait(timeout=120)
    manager.cancel(job.job_id)

    thread = manager._threads.get(job.job_id)
    deadline = time.time() + 120
    while time.time() < deadline:
        current = manager.get(job.job_id)
        if current is not None and current.is_terminal:
            break
        time.sleep(0.2)

    final = manager.get(job.job_id)
    assert final is not None
    assert final.state in {JobState.CANCELLED, JobState.COMPLETED}
    if thread is not None:
        # Graceful means the thread returned; it was never killed.
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert manager.active_job() is None


def test_upload_page_renders_cancel_and_writes_the_request_to_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real page must render Cancel and persist the click durably.

    This drives the actual page script, so it also guards against the page
    blocking on a polling loop: a page that never returns fails here.
    """
    from streamlit.testing.v1 import AppTest

    import dashboard.bootstrap as bootstrap
    from dashboard.services.job_manager import WORKER_BOOT_ID

    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    service = DashboardProcessingService(config)
    manager = DashboardJobManager(store, service)
    context = (
        config,
        store,
        service,
        DashboardRegistryService(config),
        None,
        DashboardRunService(config, store),
        manager,
    )
    monkeypatch.setattr(bootstrap, "load_dashboard_context", lambda: context)

    job_id = "ui-cancel-job"
    run_id = "ui-cancel-run"
    store.create_processing_job(
        {
            "job_id": job_id,
            "run_id": run_id,
            "state": "QUEUED",
            "source_filename": "probe.csv",
            "source_file_hash": "p" * 64,
            "deployment_mode": "assisted",
            "model_id": MODEL_ID,
        }
    )
    # Claimed by this interpreter so the job is genuinely owned rather than
    # being recovered as an orphan.
    store.claim_processing_job(
        job_id,
        run_id=run_id,
        worker_pid=os.getpid(),
        worker_host=socket.gethostname(),
        worker_boot_id=WORKER_BOOT_ID,
    )
    store.record_job_heartbeat(
        job_id,
        run_id=run_id,
        stage_key="competitor_discovery",
        stage_label="Competitor discovery",
        overall_percent=61.0,
        elapsed_seconds=42.0,
    )

    app = AppTest.from_file(
        Path("dashboard/pages/1_Upload_and_Process.py")
    ).run(timeout=30)
    assert not app.exception
    assert any(button.key == "cancel_run" for button in app.button)

    app.button(key="cancel_run").click().run(timeout=30)

    # Re-read through a separate store handle: the request must live in
    # SQLite, not in session state.
    persisted = LearningStore(
        config.learning_store.database_path
    ).get_processing_job(job_id)
    assert persisted["state"] == JobState.CANCELLING.value
    assert persisted["cancel_requested"] is True
