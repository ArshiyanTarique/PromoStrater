"""One-snapshot status resolution shared by every dashboard surface.

These replace the old ``test_run_state`` suite. That module resolved status
from ``pipeline_runs`` rows while the upload page resolved it from the durable
job record, so the sidebar and the page could describe the same run
differently. The cases that mattered there are kept below, now asserted against
the single resolver both surfaces read.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard.services.job_manager import JobSnapshot
from dashboard.services.job_state import JobState
from dashboard.services.pipeline_status import (
    WATCHED_JOB_SESSION_KEY,
    PipelinePhase,
    PipelineStatus,
    resolve_pipeline_status,
)
from sku_mapping.learning.store import LearningStore


class FakeJobs:
    """Stand-in for the job manager, including its orphan sweep on read."""

    def __init__(
        self,
        *,
        active: JobSnapshot | None = None,
        latest: JobSnapshot | None = None,
        by_id: dict[str, JobSnapshot] | None = None,
    ) -> None:
        self._active = active
        self._latest = latest
        self._by_id = by_id or {}

    def active_job(self) -> JobSnapshot | None:
        return self._active

    def get(self, job_id: str) -> JobSnapshot | None:
        return self._by_id.get(job_id)

    def latest_job(self) -> JobSnapshot | None:
        return self._latest


def make_job(
    *,
    state: JobState,
    job_id: str = "job-1",
    run_id: str | None = "run-1",
    stage_key: str | None = "inference",
    stage_label: str | None = "Mapping own-brand SKUs",
    overall_percent: float = 42.6,
    elapsed_seconds: float = 12.0,
    created_at: str | None = None,
    finished_at: str | None = None,
    error_summary: str | None = None,
) -> JobSnapshot:
    return JobSnapshot(
        job_id=job_id,
        state=state,
        run_id=run_id,
        cancel_requested=state is JobState.CANCELLING,
        stage_key=stage_key,
        stage_label=stage_label,
        stage_detail="Working",
        overall_percent=overall_percent,
        elapsed_seconds=elapsed_seconds,
        source_filename="offers.csv",
        error_summary=error_summary,
        partial_artifacts=(),
        heartbeat_at=None,
        finished_at=finished_at,
        created_at=created_at,
    )


@pytest.fixture()
def store(tmp_path: Path) -> LearningStore:
    return LearningStore(tmp_path / "learning.db")


def test_idle_on_empty_store(store: LearningStore) -> None:
    status = resolve_pipeline_status(FakeJobs(), store)
    assert status.phase is PipelinePhase.IDLE
    assert not status.is_active
    assert not status.has_run_detail
    assert status.percent == 0.0


def test_active_job_is_reported_with_its_live_stage(
    store: LearningStore,
) -> None:
    job = make_job(state=JobState.RUNNING)
    status = resolve_pipeline_status(FakeJobs(active=job), store)

    assert status.phase is PipelinePhase.RUNNING
    assert status.is_active
    assert status.stage_text == "Mapping own-brand SKUs"
    assert status.headline == "Mapping own-brand SKUs"
    assert status.run_id == "run-1"
    assert status.filename == "offers.csv"


def test_abandoned_processing_run_is_not_reported_as_active(
    store: LearningStore,
) -> None:
    """A killed worker leaves PROCESSING behind; that is not a live run.

    Without this the sidebar would advertise a live run forever while the
    upload page, which sweeps orphaned jobs, showed the run as finished.
    """
    store.upsert_pipeline_run(
        {
            "run_id": "run-abandoned",
            "status": "PROCESSING",
            "source_filename": "abandoned.csv",
        }
    )

    status = resolve_pipeline_status(FakeJobs(), store)
    assert status.phase is PipelinePhase.IDLE
    assert not status.is_active


def test_completed_and_failed_runs_without_a_job_still_resolve(
    store: LearningStore,
) -> None:
    """A run created outside the dashboard still has a status to report."""
    store.upsert_pipeline_run(
        {
            "run_id": "run-failed",
            "status": "FAILED_DASHBOARD",
            "source_filename": "fail.csv",
            "started_at": "2026-07-31T10:00:00Z",
        }
    )
    failed = resolve_pipeline_status(FakeJobs(), store)
    assert failed.phase is PipelinePhase.FAILED
    assert failed.run_id == "run-failed"

    store.upsert_pipeline_run(
        {
            "run_id": "run-completed",
            "status": "COMPLETED_DASHBOARD_ASSISTED",
            "source_filename": "success.csv",
            "completed_at": "2026-07-31T10:05:00Z",
        }
    )
    completed = resolve_pipeline_status(FakeJobs(), store)
    assert completed.phase is PipelinePhase.COMPLETED
    assert completed.run_id == "run-completed"
    assert completed.percent == 100.0


def test_orphaned_run_status_reads_as_failed(store: LearningStore) -> None:
    store.upsert_pipeline_run(
        {
            "run_id": "run-orphaned",
            "status": "FAILED_DASHBOARD_ORPHANED",
            "source_filename": "orphan.csv",
        }
    )
    assert (
        resolve_pipeline_status(FakeJobs(), store).phase
        is PipelinePhase.FAILED
    )


def test_cancelled_run_is_never_reported_as_idle(store: LearningStore) -> None:
    """The sidebar used to read a cancelled run as idle while the page did not."""
    job = make_job(state=JobState.CANCELLED, run_id=None)
    status = resolve_pipeline_status(FakeJobs(latest=job), store)

    assert status.phase is PipelinePhase.CANCELLED
    assert status.has_run_detail
    assert status.stage_text == "Cancelled"


def test_job_record_outranks_a_stale_run_row(store: LearningStore) -> None:
    """A failed job with no run row must not surface an older happy run."""
    store.upsert_pipeline_run(
        {
            "run_id": "run-old",
            "status": "COMPLETED_DASHBOARD_ASSISTED",
            "source_filename": "old.csv",
        }
    )
    job = make_job(
        state=JobState.FAILED,
        run_id=None,
        error_summary="Upload rejected",
    )

    status = resolve_pipeline_status(FakeJobs(latest=job), store)
    assert status.phase is PipelinePhase.FAILED
    assert status.headline == "Upload rejected"


def test_watched_job_is_preferred_over_the_latest_job(
    store: LearningStore,
) -> None:
    watched = make_job(state=JobState.COMPLETED, job_id="job-mine")
    other = make_job(state=JobState.FAILED, job_id="job-theirs")
    jobs = FakeJobs(latest=other, by_id={"job-mine": watched})

    status = resolve_pipeline_status(
        jobs, store, {WATCHED_JOB_SESSION_KEY: "job-mine"}
    )
    assert status.job_id == "job-mine"
    assert status.phase is PipelinePhase.COMPLETED


def test_active_job_outranks_the_watched_job(store: LearningStore) -> None:
    """Whatever is running now is what every surface must describe."""
    active = make_job(state=JobState.RUNNING, job_id="job-live")
    watched = make_job(state=JobState.COMPLETED, job_id="job-mine")
    jobs = FakeJobs(active=active, by_id={"job-mine": watched})

    status = resolve_pipeline_status(
        jobs, store, {WATCHED_JOB_SESSION_KEY: "job-mine"}
    )
    assert status.job_id == "job-live"


def test_unfinished_work_never_renders_as_complete(
    store: LearningStore,
) -> None:
    for state in (JobState.RUNNING, JobState.CANCELLING, JobState.FAILED):
        job = make_job(state=state, overall_percent=99.98)
        status = resolve_pipeline_status(FakeJobs(latest=job), store)
        assert status.percent == 99.0
        assert status.percent_text == "99%"


def test_completed_job_always_reads_as_one_hundred(
    store: LearningStore,
) -> None:
    job = make_job(state=JobState.COMPLETED, overall_percent=97.2)
    status = resolve_pipeline_status(FakeJobs(latest=job), store)
    assert status.percent == 100.0
    assert status.progress_fraction == 1.0


def test_bar_and_text_agree_on_the_same_percentage() -> None:
    """The bar truncated while the caption rounded, so 45.6% drew 45 beside 46%."""
    status = PipelineStatus(phase=PipelinePhase.RUNNING, percent=45.6)
    assert status.percent_text == "46%"
    assert status.progress_fraction == 0.46


def test_queued_job_reports_queued_rather_than_a_stage(
    store: LearningStore,
) -> None:
    job = make_job(state=JobState.QUEUED, stage_key=None, stage_label=None)
    status = resolve_pipeline_status(FakeJobs(active=job), store)

    assert status.is_active
    assert status.stage_text == "Queued"
    assert "worker" in status.headline


def test_running_job_without_a_stage_label_still_names_its_position(
    store: LearningStore,
) -> None:
    job = make_job(state=JobState.RUNNING, stage_label=None)
    status = resolve_pipeline_status(FakeJobs(active=job), store)
    assert status.stage_text == "Starting"


def test_active_elapsed_follows_the_wall_clock(store: LearningStore) -> None:
    """A long stage writes no progress, so the recorded value stands still."""
    created = datetime.now(timezone.utc) - timedelta(minutes=30)
    job = make_job(
        state=JobState.RUNNING,
        elapsed_seconds=56.0,
        created_at=created.isoformat(),
    )
    status = resolve_pipeline_status(FakeJobs(active=job), store)
    assert status.elapsed_seconds > 1700


def test_finished_elapsed_uses_the_real_span(store: LearningStore) -> None:
    job = make_job(
        state=JobState.FAILED,
        elapsed_seconds=56.0,
        created_at="2026-07-31T10:00:00+00:00",
        finished_at="2026-07-31T10:04:00+00:00",
    )
    status = resolve_pipeline_status(FakeJobs(latest=job), store)
    assert status.elapsed_seconds == 240.0


def test_dismissed_failure_reads_as_idle(store: LearningStore) -> None:
    job = make_job(state=JobState.FAILED, error_summary="Boom")
    status = resolve_pipeline_status(FakeJobs(latest=job), store).dismissed()

    assert status.phase is PipelinePhase.IDLE
    assert not status.has_run_detail
    assert status.error_summary is None
    assert status.percent == 0.0


#: Every surface that reports processing state to the user.
_STATUS_SURFACES = (
    Path("dashboard/Dashboard.py"),
    Path("dashboard/pages/1_Upload_and_Process.py"),
    Path("dashboard/components/sidebar_status.py"),
)


def test_no_surface_resolves_processing_state_on_its_own() -> None:
    """The bug was two resolvers, not two renderers.

    A surface that reads job or run rows directly is free to disagree with
    every other surface, which is exactly how the sidebar came to report a
    stage and percentage the upload page did not share.
    """
    forbidden = (
        "active_processing_jobs",
        "list_pipeline_runs",
        "latest_job(",
        ".active_job(",
        "get_global_status",
    )
    for path in _STATUS_SURFACES:
        source = path.read_text(encoding="utf-8")
        assert "current_status" in source, f"{path} must use the snapshot"
        for token in forbidden:
            assert token not in source, f"{path} resolves state itself: {token}"


def test_every_page_gives_the_sidebar_the_job_manager() -> None:
    """A sidebar without the job manager cannot see orphan recovery."""
    for path in sorted(Path("dashboard").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for call in re.findall(r"render_sidebar_status\(([^)]*)\)", source):
            if "store: LearningStore" in call or not call.strip():
                continue  # the definition itself
            assert "jobs" in call, f"{path} calls the sidebar without jobs"


def test_every_job_state_maps_onto_a_phase() -> None:
    """A new job state must not leave a surface with nothing to show."""
    for state in JobState:
        status = PipelineStatus(phase=PipelinePhase(state.value))
        assert status.stage_text
        assert status.headline
        assert status.badge_label == state.value
