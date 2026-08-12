"""Cancellation token semantics and durable job state across refreshes."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from dashboard.services.job_state import (
    ACTIVE_JOB_STATES,
    TERMINAL_JOB_STATES,
    CancellationToken,
    JobState,
    RunCancelled,
)
from dashboard.services.progress import (
    ProcessingState,
    ProgressTracker,
)
from sku_mapping.learning.store import LearningStore


def test_liveness_heartbeats_keep_a_silent_stage_from_looking_orphaned(
    tmp_path: Path,
) -> None:
    """A stage that reports no progress must still prove the worker is alive.

    Regression: entity preparation ran for minutes without a progress
    callback, the heartbeat went stale, and a healthy run was marked
    "The processing worker stopped unexpectedly".
    """
    import dashboard.services.job_manager as job_manager

    store = LearningStore(tmp_path / "learning.db")
    job_id = store.create_processing_job(
        {"job_id": "job-silent-stage", "source_filename": "silent.csv"}
    )
    store.claim_processing_job(
        job_id, run_id=None, worker_pid=1, worker_host="test", worker_boot_id="b"
    )
    before = store.get_processing_job(job_id)["heartbeat_at"]

    manager = object.__new__(job_manager.DashboardJobManager)
    manager.store = store
    stopped = threading.Event()
    monkeypatched = 0.05
    original = job_manager.LIVENESS_HEARTBEAT_SECONDS
    job_manager.LIVENESS_HEARTBEAT_SECONDS = monkeypatched
    try:
        worker = threading.Thread(
            target=manager._write_liveness_heartbeats,
            args=(job_id, stopped),
            daemon=True,
        )
        worker.start()
        # No progress is ever reported here; only liveness should advance.
        threading.Event().wait(0.4)
        stopped.set()
        worker.join(timeout=5.0)
    finally:
        job_manager.LIVENESS_HEARTBEAT_SECONDS = original

    after = store.get_processing_job(job_id)["heartbeat_at"]
    assert after is not None
    assert after != before


def test_token_request_is_monotonic_and_records_a_reason() -> None:
    token = CancellationToken()

    assert not token.cancelled
    token.request("User clicked Cancel Run")
    token.request("A later, ignored reason")

    assert token.cancelled
    assert token.reason == "User clicked Cancel Run"
    assert token.requested_at is not None


def test_token_raises_only_after_a_request() -> None:
    token = CancellationToken()

    token.raise_if_cancelled("inference")
    token.request()

    with pytest.raises(RunCancelled) as caught:
        token.raise_if_cancelled("inference")
    assert caught.value.stage_key == "inference"


def test_token_is_observable_from_another_thread() -> None:
    token = CancellationToken()
    seen: list[bool] = []
    ready = threading.Event()

    def worker() -> None:
        ready.set()
        token.wait(timeout=5)
        seen.append(token.cancelled)

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(timeout=5)
    token.request()
    thread.join(timeout=5)

    assert seen == [True]


def test_job_state_sets_partition_the_lifecycle() -> None:
    assert ACTIVE_JOB_STATES | TERMINAL_JOB_STATES == set(JobState)
    assert not ACTIVE_JOB_STATES & TERMINAL_JOB_STATES


def test_tracker_reports_cancelling_then_cancelled_without_reaching_100(
) -> None:
    updates = []
    tracker = ProgressTracker(updates.append)
    tracker.start()
    tracker.update("inference", completed=500, total=1000, detail="Scoring")

    tracker.cancelling()
    # A stage mid-flight keeps reporting real work while cancelling.
    tracker.update("inference", completed=600, total=1000, detail="Scoring")
    tracker.cancelled(run_id="run-1")

    states = [item.state for item in updates]
    assert ProcessingState.CANCELLING in states
    assert states[-1] is ProcessingState.CANCELLED
    assert updates[-1].overall_percent < 100.0
    assert updates[-1].pipeline_status == "CANCELLED_DASHBOARD"


def test_cancelled_state_rejects_further_progress_updates() -> None:
    tracker = ProgressTracker()
    tracker.start()
    tracker.cancelled(run_id="run-1")

    with pytest.raises(RuntimeError):
        tracker.update("inference", completed=1, total=2)


def test_job_state_survives_a_simulated_browser_refresh(
    tmp_path: Path,
) -> None:
    """A second store handle reads the same durable state, as a reload does."""
    database = tmp_path / "learning.db"
    writer = LearningStore(database)
    writer.create_processing_job(
        {
            "job_id": "job-1",
            "state": JobState.QUEUED.value,
            "source_file_hash": "0" * 64,
            "deployment_mode": "assisted",
        }
    )
    writer.claim_processing_job(
        "job-1",
        run_id="run-1",
        worker_pid=42,
        worker_host="host",
        worker_boot_id="boot",
    )
    writer.record_job_heartbeat(
        "job-1",
        stage_key="inference",
        stage_label="Mapping own-brand SKUs",
        overall_percent=41.5,
        elapsed_seconds=12.0,
    )
    writer.request_job_cancellation("job-1")

    reader = LearningStore(database)
    record = reader.get_processing_job("job-1")

    assert record["state"] == JobState.CANCELLING.value
    assert record["cancel_requested"] is True
    assert record["stage_label"] == "Mapping own-brand SKUs"
    assert record["overall_percent"] == pytest.approx(41.5)
    assert reader.active_jobs_for_source_hash("0" * 64) != []


def test_heartbeat_never_overwrites_a_pending_cancel_request(
    tmp_path: Path,
) -> None:
    """An in-flight heartbeat must not resurrect a cancelling job."""
    store = LearningStore(tmp_path / "learning.db")
    store.create_processing_job(
        {"job_id": "job-2", "state": JobState.QUEUED.value}
    )
    store.claim_processing_job(
        "job-2",
        run_id=None,
        worker_pid=1,
        worker_host="host",
        worker_boot_id="boot",
    )
    store.request_job_cancellation("job-2")

    store.record_job_heartbeat("job-2", stage_key="competitor_discovery")

    assert store.get_processing_job("job-2")["state"] == (
        JobState.CANCELLING.value
    )


def test_latest_job_is_readable_without_any_session_state(
    tmp_path: Path,
) -> None:
    """A refreshed page must still find the previous run's outcome.

    Regression: the outcome panel used to depend on a session-state job id,
    so refreshing the browser silently hid the completion or cancellation
    notice for the run that had just finished.
    """
    store = LearningStore(tmp_path / "learning.db")
    for index in range(3):
        store.create_processing_job(
            {"job_id": f"job-{index}", "state": JobState.QUEUED.value}
        )
    store.finish_processing_job("job-2", state=JobState.CANCELLED.value)

    latest = store.latest_processing_job()

    assert latest is not None
    assert latest["job_id"] == "job-2"
    assert latest["state"] == JobState.CANCELLED.value


def test_latest_job_is_none_on_a_fresh_database(tmp_path: Path) -> None:
    assert LearningStore(tmp_path / "learning.db").latest_processing_job() is (
        None
    )
