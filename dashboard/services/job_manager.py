"""Background processing jobs owned by SQLite rather than by a browser session.

The worker thread executes the existing processing pipeline unchanged and
writes every state transition to SQLite. The UI never owns a job: it polls
this manager, which reads durable state. A cancel request is a flag write,
never a thread kill.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from dashboard.services.job_state import (
    ACTIVE_JOB_STATES,
    CANCELLED_RUN_STATUS,
    ORPHANED_RUN_STATUS,
    CancellationToken,
    JobState,
)
from dashboard.services.processing_service import (
    DashboardProcessRequest,
    DashboardProcessingError,
    DashboardProcessingService,
    DuplicateProcessingError,
    ProcessingCancelledError,
)
from dashboard.services.progress import ProgressUpdate
from sku_mapping.learning.store import LearningStore

LOGGER = logging.getLogger(__name__)

#: A worker writes a heartbeat at every progress callback. A job whose
#: heartbeat is older than this is treated as orphaned on next startup.
HEARTBEAT_EXPIRY_SECONDS = 120.0
#: How often the worker proves liveness independently of pipeline progress.
#: Comfortably below the expiry so a slow stage cannot look abandoned.
LIVENESS_HEARTBEAT_SECONDS = 20.0
#: Identifies this interpreter. A recorded pid from a previous process can be
#: reused by the OS, so pid alone is never treated as proof of liveness.
WORKER_BOOT_ID = uuid.uuid4().hex


@dataclass(frozen=True)
class JobSnapshot:
    """Presentation-safe view of one durable job, read from SQLite."""

    job_id: str
    state: JobState
    run_id: str | None
    cancel_requested: bool
    stage_key: str | None
    stage_label: str | None
    stage_detail: str | None
    overall_percent: float
    elapsed_seconds: float
    source_filename: str | None
    error_summary: str | None
    partial_artifacts: tuple[str, ...]
    heartbeat_at: str | None
    finished_at: str | None
    created_at: str | None = None

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_JOB_STATES

    @property
    def live_elapsed_seconds(self) -> float:
        """Wall-clock time so far, rather than the last value the worker wrote.

        ``elapsed_seconds`` is only refreshed when the pipeline emits a
        progress update, so it stands still for the whole of a long stage -
        one job sat at a recorded 56s while actually running for hours.
        Active jobs are therefore measured from ``created_at``; finished jobs
        keep the total the worker recorded.
        """
        if not self.is_active or not self.created_at:
            return self.elapsed_seconds
        try:
            started = datetime.fromisoformat(str(self.created_at))
        except ValueError:
            return self.elapsed_seconds
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        live = (datetime.now(timezone.utc) - started).total_seconds()
        # Never let the display go backwards if the clocks disagree.
        return max(self.elapsed_seconds, live)

    @property
    def is_cancelling(self) -> bool:
        return self.state is JobState.CANCELLING

    @property
    def is_terminal(self) -> bool:
        return not self.is_active


def _snapshot(record: Mapping[str, Any]) -> JobSnapshot:
    return JobSnapshot(
        job_id=str(record["job_id"]),
        state=JobState(str(record["state"])),
        run_id=(
            str(record["run_id"]) if record.get("run_id") else None
        ),
        cancel_requested=bool(record.get("cancel_requested")),
        stage_key=record.get("stage_key"),
        stage_label=record.get("stage_label"),
        stage_detail=record.get("stage_detail"),
        overall_percent=float(record.get("overall_percent") or 0.0),
        elapsed_seconds=float(record.get("elapsed_seconds") or 0.0),
        source_filename=record.get("source_filename"),
        error_summary=record.get("error_summary"),
        partial_artifacts=tuple(record.get("partial_artifacts") or ()),
        heartbeat_at=record.get("heartbeat_at"),
        finished_at=record.get("finished_at"),
        created_at=record.get("created_at"),
    )


_MANAGERS: dict[str, "DashboardJobManager"] = {}
_MANAGER_LOCK = threading.Lock()
_RECOVERED: set[str] = set()


def get_job_manager(
    store: LearningStore,
    processing: DashboardProcessingService,
) -> "DashboardJobManager":
    """Return the process-wide manager for one learning-store database.

    Streamlit re-executes a page script on every interaction. A per-rerun
    manager would lose the live worker's cancellation token, so cancelling
    would write the durable flag but never reach the running stage. One
    manager per database keeps worker ownership stable for the process, and
    orphan recovery runs once on first use.
    """
    key = str(store.path)
    with _MANAGER_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = DashboardJobManager(store, processing)
            _MANAGERS[key] = manager
        if key not in _RECOVERED:
            _RECOVERED.add(key)
            try:
                manager.recover_orphaned_jobs()
            except Exception:
                LOGGER.exception(
                    "Startup orphaned-job recovery failed for %s", key
                )
    return manager


class ActiveJobExistsError(RuntimeError):
    """Raised when a genuine active job already holds the processing slot."""

    def __init__(self, snapshot: JobSnapshot) -> None:
        self.snapshot = snapshot
        super().__init__(
            f"Job {snapshot.job_id} is already {snapshot.state.value}"
        )


class DashboardJobManager:
    """Own worker threads and durable job state for one dashboard process."""

    def __init__(
        self,
        store: LearningStore,
        processing: DashboardProcessingService,
        *,
        heartbeat_expiry_seconds: float = HEARTBEAT_EXPIRY_SECONDS,
    ) -> None:
        self.store = store
        self.processing = processing
        self.heartbeat_expiry_seconds = float(heartbeat_expiry_seconds)
        self._lock = threading.RLock()
        # Tokens and threads are per-process. They are an optimisation for
        # the worker that lives here; SQLite remains the source of truth.
        self._tokens: dict[str, CancellationToken] = {}
        self._threads: dict[str, threading.Thread] = {}

    # -- queries -------------------------------------------------------

    def get(self, job_id: str) -> JobSnapshot | None:
        record = self.store.get_processing_job(job_id)
        return _snapshot(record) if record is not None else None

    def active_job(self) -> JobSnapshot | None:
        """Return the single genuine active job, if one exists.

        The sweep runs here, not only at startup. Liveness is now proven by
        heartbeat, and a heartbeat goes stale *while this process runs* -
        typically minutes after the worker actually died. A boot-time-only
        sweep would miss that window and leave a dead job holding the
        submission slot for the life of the process, refusing every new run.
        """
        self.recover_orphaned_jobs()
        records = self.store.active_processing_jobs()
        return _snapshot(records[0]) if records else None

    def latest_job(self) -> JobSnapshot | None:
        """Return the most recent job so a refresh still shows an outcome."""
        record = self.store.latest_processing_job()
        return _snapshot(record) if record is not None else None

    def job_for_run(self, run_id: str) -> JobSnapshot | None:
        record = self.store.latest_processing_job_for_run(run_id)
        return _snapshot(record) if record is not None else None

    # -- lifecycle -----------------------------------------------------

    def submit(
        self,
        request: DashboardProcessRequest,
        *,
        source_file_hash: str,
        start: bool = True,
    ) -> JobSnapshot:
        """Queue one job and hand it to a worker thread.

        Duplicate prevention is deliberately narrow: only a genuinely active
        job blocks a new submission, so a cancelled run frees the slot at once.
        """
        with self._lock:
            active = self.active_job()
            if active is not None:
                raise ActiveJobExistsError(active)
            job_id = (
                f"job-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
                f"-{uuid.uuid4().hex[:8]}"
            )
            self.store.create_processing_job(
                {
                    "job_id": job_id,
                    "state": JobState.QUEUED.value,
                    "source_filename": request.filename,
                    "source_file_hash": source_file_hash,
                    "deployment_mode": request.deployment_mode.value,
                    "model_id": request.model_id,
                }
            )
            token = CancellationToken()
            self._tokens[job_id] = token
            if start:
                thread = threading.Thread(
                    target=self._run_job,
                    args=(job_id, request, token),
                    name=f"dashboard-processing-{job_id}",
                    daemon=True,
                )
                self._threads[job_id] = thread
                thread.start()
        snapshot = self.get(job_id)
        assert snapshot is not None
        return snapshot

    def cancel(self, job_id: str) -> JobSnapshot | None:
        """Request cooperative cancellation; never kill the worker.

        The durable flag is written first so a browser refresh, or a restart
        of this process, still observes the request.
        """
        state = self.store.request_job_cancellation(job_id)
        if state is None:
            return None
        token = self._tokens.get(job_id)
        if token is not None:
            token.request()
        if state == JobState.CANCELLED.value:
            # A queued job never entered a stage, so there is nothing to
            # unwind and no run row to reconcile.
            LOGGER.info("Queued job %s cancelled before starting", job_id)
        return self.get(job_id)

    # -- orphan recovery -----------------------------------------------

    def _is_orphaned(self, record: Mapping[str, Any]) -> bool:
        """Decide whether no live worker can still own this job.

        The heartbeat is the only evidence of liveness that crosses process
        boundaries, so it is what decides. A foreign ``worker_boot_id`` used
        to mean "orphaned" outright, on the reasoning that another
        interpreter's threads are gone - true after a restart, but false
        while that interpreter is still running. Any second dashboard
        process therefore declared healthy runs dead: a run was marked
        FAILED while its worker kept writing heartbeats for three more
        minutes and finished the stage it was on.

        A worker proves itself every ``LIVENESS_HEARTBEAT_SECONDS`` from a
        thread of its own, independent of pipeline progress, so a fresh
        heartbeat means *some* live worker owns this job - whichever process
        that worker lives in. A pid is still never consulted, since the OS
        reuses them.
        """
        boot_id = str(record.get("worker_boot_id") or "")
        if not boot_id:
            # QUEUED and not yet claimed: no worker has owned it, so there is
            # no heartbeat to read and its absence proves nothing. Age is the
            # only signal - a job is abandoned once it has sat unclaimed for
            # longer than a worker would ever take to pick it up.
            return self._heartbeat_expired(record.get("created_at"))
        if boot_id == WORKER_BOOT_ID:
            # Our own worker: the thread object is stronger evidence than any
            # timestamp, and it costs nothing to ask.
            thread = self._threads.get(str(record["job_id"]))
            if thread is not None and thread.is_alive():
                return False
        return self._heartbeat_expired(record.get("heartbeat_at"))

    def _heartbeat_expired(self, heartbeat_at: object) -> bool:
        if not heartbeat_at:
            return True
        try:
            beat = datetime.fromisoformat(str(heartbeat_at))
        except ValueError:
            return True
        if beat.tzinfo is None:
            beat = beat.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - beat).total_seconds()
        return age > self.heartbeat_expiry_seconds

    def recover_orphaned_jobs(self) -> list[JobSnapshot]:
        """Resolve jobs whose worker no longer exists. Safe to call anytime.

        A job already asked to cancel is honoured as CANCELLED; anything else
        becomes FAILED. Either way the active slot is released, so a restart
        never leaves a permanently RUNNING job.
        """
        recovered: list[JobSnapshot] = []
        with self._lock:
            for record in self.store.active_processing_jobs():
                if not self._is_orphaned(record):
                    continue
                job_id = str(record["job_id"])
                was_cancelling = (
                    bool(record.get("cancel_requested"))
                    or str(record.get("state")) == JobState.CANCELLING.value
                )
                state = (
                    JobState.CANCELLED if was_cancelling else JobState.FAILED
                )
                reason = (
                    "Cancelled; the worker stopped before confirming"
                    if was_cancelling
                    else "The processing worker stopped unexpectedly"
                )
                self.store.finish_processing_job(
                    job_id,
                    state=state.value,
                    error_summary=reason,
                    stage_detail=reason,
                )
                self._reconcile_orphaned_run(record, state)
                snapshot = self.get(job_id)
                if snapshot is not None:
                    recovered.append(snapshot)
                LOGGER.warning(
                    "Recovered orphaned job %s as %s", job_id, state.value
                )
        return recovered

    def _reconcile_orphaned_run(
        self, record: Mapping[str, Any], state: JobState
    ) -> None:
        """Stop an abandoned run row from looking permanently in-flight."""
        run_id = record.get("run_id")
        if not run_id:
            return
        run = self.store.get_pipeline_run(str(run_id))
        if run is None or str(run["status"]).startswith("COMPLETED"):
            return
        self.store.update_run_outputs(
            str(run_id),
            {},
            status=(
                CANCELLED_RUN_STATUS
                if state is JobState.CANCELLED
                else ORPHANED_RUN_STATUS
            ),
            error_summary=(
                "The processing worker stopped before this run finished"
            ),
        )

    # -- worker --------------------------------------------------------

    def _progress_writer(self, job_id: str):
        """Persist each progress update so the UI can stay passive."""

        def write(update: ProgressUpdate) -> None:
            try:
                self.store.record_job_heartbeat(
                    job_id,
                    run_id=update.run_id,
                    stage_key=update.stage_key,
                    stage_label=update.stage_label,
                    stage_detail=update.detail,
                    overall_percent=update.overall_percent,
                    completed_stage_keys=update.completed_stage_keys,
                    elapsed_seconds=update.elapsed_seconds,
                )
            except Exception:
                # Progress persistence must never abort real processing.
                LOGGER.exception(
                    "Could not persist progress for job %s", job_id
                )

        return write

    def _write_liveness_heartbeats(
        self, job_id: str, stopped: threading.Event
    ) -> None:
        """Prove the worker is alive even while a stage reports no progress.

        Heartbeats used to be written only from the progress callback, so any
        stage that ran longer than the expiry without emitting one had its
        healthy job declared orphaned. Liveness is a property of the worker,
        not of how chatty the current stage happens to be.
        """
        while not stopped.wait(LIVENESS_HEARTBEAT_SECONDS):
            try:
                self.store.record_job_heartbeat(job_id)
            except Exception:
                # Liveness reporting must never abort real processing.
                LOGGER.exception(
                    "Could not write liveness heartbeat for job %s", job_id
                )

    def _run_job(
        self,
        job_id: str,
        request: DashboardProcessRequest,
        token: CancellationToken,
    ) -> None:
        """Execute the existing pipeline and record only durable outcomes."""
        claimed = self.store.claim_processing_job(
            job_id,
            run_id=None,
            worker_pid=os.getpid(),
            worker_host=socket.gethostname(),
            worker_boot_id=WORKER_BOOT_ID,
        )
        if not claimed:
            # The job was cancelled while still QUEUED; honour that decision.
            LOGGER.info("Job %s was not claimable; leaving state intact", job_id)
            return
        liveness_stopped = threading.Event()
        liveness = threading.Thread(
            target=self._write_liveness_heartbeats,
            args=(job_id, liveness_stopped),
            name=f"job-heartbeat-{job_id}",
            daemon=True,
        )
        liveness.start()
        try:
            result = self.processing.process(
                request,
                progress=self._progress_writer(job_id),
                cancel_token=token,
            )
        except ProcessingCancelledError as cancellation:
            self.store.finish_processing_job(
                job_id,
                state=JobState.CANCELLED.value,
                run_id=cancellation.run_id,
                error_summary=(
                    "Cancelled during "
                    f"{cancellation.cancelled_stage or 'processing'}"
                ),
                partial_artifacts=cancellation.partial_artifacts,
                stage_detail="Cancelled before completion",
            )
        except DuplicateProcessingError as duplicate:
            self.store.finish_processing_job(
                job_id,
                state=JobState.FAILED.value,
                error_summary=(
                    "Duplicate of run(s): " + ", ".join(duplicate.run_ids)
                ),
                stage_detail="Blocked as a duplicate upload",
            )
        except DashboardProcessingError as error:
            self.store.finish_processing_job(
                job_id,
                state=JobState.FAILED.value,
                run_id=error.run_id,
                error_summary=str(error),
                partial_artifacts=error.partial_artifacts,
                stage_detail="Processing failed",
            )
        except BaseException as error:  # noqa: BLE001 - worker must not vanish
            LOGGER.exception("Job %s ended unexpectedly", job_id)
            self.store.finish_processing_job(
                job_id,
                state=JobState.FAILED.value,
                error_summary=f"{type(error).__name__}: {error}",
                stage_detail="Processing stopped unexpectedly",
            )
        else:
            self.store.finish_processing_job(
                job_id,
                state=JobState.COMPLETED.value,
                run_id=result.run_id,
                overall_percent=100.0,
                stage_detail="Processing complete",
            )
        finally:
            liveness_stopped.set()
            liveness.join(timeout=5.0)
            with self._lock:
                self._tokens.pop(job_id, None)
                self._threads.pop(job_id, None)
