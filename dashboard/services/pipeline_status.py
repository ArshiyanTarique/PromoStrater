"""One resolved answer to "what is the pipeline doing right now?".

Every surface that reports processing state - the sidebar card, the upload
page's run panel, the home KPI - renders from the single snapshot this module
produces. They used to resolve it independently: the sidebar read
``pipeline_runs`` rows while the upload page read the durable *job* record, and
the two disagreed constantly. A cancelled run showed as idle in the sidebar
while the page reported it cancelled; a job whose worker died kept the sidebar
on ACTIVE at a frozen percentage because only the page's path ran the orphan
sweep; the same percentage rendered truncated in one place and rounded in the
other.

The job record is the authority here, exactly as it is for the upload page,
because it is what a live worker writes to and what orphan recovery resolves. A
``pipeline_runs`` row is consulted only to enrich a job (row counts) or as a
last resort when no dashboard job exists at all, which is how a run created by
the CLI pipeline still appears.

Presentation strings live on the snapshot rather than at the call sites, so two
surfaces cannot word or round the same fact differently. The one exception is
elapsed time: the snapshot carries seconds and both surfaces pass them through
:func:`dashboard.components.formatters.format_elapsed`, keeping Streamlit out of
this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping, Protocol

if TYPE_CHECKING:
    from dashboard.services.job_manager import JobSnapshot
    from sku_mapping.learning.store import LearningStore

#: Session key holding the job this browser session started.
WATCHED_JOB_SESSION_KEY = "watched_job_id"

#: Nothing unfinished may render as complete, however far it reached. The
#: worker's own tracker applies the same ceiling, but a job that died mid-stage
#: never gets a final update, so the ceiling is re-applied on read.
MAX_INCOMPLETE_PERCENT = 99.0


class PipelinePhase(StrEnum):
    """The one status vocabulary every surface displays.

    Values mirror :class:`dashboard.services.job_state.JobState` so a job maps
    across without translation, plus ``IDLE`` for "nothing to report".
    """

    IDLE = "IDLE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def is_active(self) -> bool:
        return self in _ACTIVE_PHASES


_ACTIVE_PHASES = frozenset(
    {PipelinePhase.QUEUED, PipelinePhase.RUNNING, PipelinePhase.CANCELLING}
)

#: Shown wherever a run's position is named, including when the worker has not
#: reported a stage yet. A phase always resolves to a word, so no surface has
#: to invent its own placeholder.
_PHASE_STAGE_TEXT = {
    PipelinePhase.IDLE: "Not running",
    PipelinePhase.QUEUED: "Queued",
    PipelinePhase.RUNNING: "Starting",
    PipelinePhase.CANCELLING: "Stopping",
    PipelinePhase.CANCELLED: "Cancelled",
    PipelinePhase.COMPLETED: "Completed",
    PipelinePhase.FAILED: "Stopped",
}

#: The single sentence describing the phase. ``RUNNING`` is absent on purpose:
#: a running job describes itself with its live stage.
_PHASE_HEADLINE = {
    PipelinePhase.IDLE: "No active background processing.",
    PipelinePhase.QUEUED: (
        "Queued - waiting for the worker to pick this job up."
    ),
    PipelinePhase.CANCELLING: (
        "Cancelling - the current stage finishes before the run stops."
    ),
    PipelinePhase.CANCELLED: "This run was cancelled and did not complete.",
    PipelinePhase.COMPLETED: "Processing completed successfully.",
    PipelinePhase.FAILED: "The last processing job failed.",
}


class JobReader(Protocol):
    """The job-manager surface this resolver depends on."""

    def active_job(self) -> "JobSnapshot | None": ...

    def get(self, job_id: str) -> "JobSnapshot | None": ...

    def latest_job(self) -> "JobSnapshot | None": ...


@dataclass(frozen=True)
class PipelineStatus:
    """One presentation-ready view of processing state, shared by all surfaces."""

    phase: PipelinePhase
    percent: float = 0.0
    elapsed_seconds: float = 0.0
    job_id: str | None = None
    run_id: str | None = None
    filename: str | None = None
    stage_key: str | None = None
    stage_label: str | None = None
    stage_detail: str | None = None
    error_summary: str | None = None
    partial_artifacts: tuple[str, ...] = ()
    source_row_count: int = 0

    # -- lifecycle -----------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.phase.is_active

    @property
    def is_cancelling(self) -> bool:
        return self.phase is PipelinePhase.CANCELLING

    @property
    def has_run_detail(self) -> bool:
        """Whether there is a run to describe at all."""
        return self.phase is not PipelinePhase.IDLE

    # -- display -------------------------------------------------------

    @property
    def badge_label(self) -> str:
        return self.phase.value

    @property
    def file_text(self) -> str:
        return self.filename or "Unnamed upload"

    @property
    def stage_text(self) -> str:
        """Where the run is, in the same words on every surface."""
        # Only a job inside a stage has a live label worth showing. A queued
        # job has not started one, and a finished job's last stage is not where
        # it *is* - it is how it ended.
        if self.phase in {PipelinePhase.RUNNING, PipelinePhase.CANCELLING}:
            return self.stage_label or _PHASE_STAGE_TEXT[self.phase]
        return _PHASE_STAGE_TEXT[self.phase]

    @property
    def headline(self) -> str:
        if self.phase is PipelinePhase.RUNNING:
            return self.stage_text
        if self.phase is PipelinePhase.FAILED and self.error_summary:
            return self.error_summary
        return _PHASE_HEADLINE[self.phase]

    @property
    def percent_text(self) -> str:
        return f"{self.percent:.0f}%"

    @property
    def progress_fraction(self) -> float:
        """Bar fill that always matches :attr:`percent_text`.

        The sidebar bar used to truncate while its own caption rounded, so a
        run at 45.6% drew a 45% bar beside the text "46%".
        """
        return min(1.0, max(0.0, round(self.percent) / 100.0))

    # -- derivation ----------------------------------------------------

    def dismissed(self) -> "PipelineStatus":
        """Recast a dismissed failure as idle, leaving SQLite untouched."""
        return replace(
            self,
            phase=PipelinePhase.IDLE,
            percent=0.0,
            stage_key=None,
            stage_label=None,
            stage_detail=None,
            error_summary=None,
            partial_artifacts=(),
        )


def resolve_pipeline_status(
    jobs: JobReader,
    store: "LearningStore",
    session_state: Mapping[str, Any] | None = None,
) -> PipelineStatus:
    """Resolve the one status snapshot every surface renders.

    ``session_state`` is read only for the job this browser session started, so
    a surface that omits it still resolves the same active job.
    """
    job = _select_job(jobs, session_state)
    if job is not None:
        return _status_from_job(job, store)
    return _status_from_latest_run(store)


def _select_job(
    jobs: JobReader, session_state: Mapping[str, Any] | None
) -> "JobSnapshot | None":
    """Pick the job every surface should be describing.

    ``active_job`` runs orphan recovery, so a job whose worker died is resolved
    to its terminal state here rather than being reported as live.
    """
    active = jobs.active_job()
    if active is not None:
        return active
    watched = (session_state or {}).get(WATCHED_JOB_SESSION_KEY)
    if watched:
        job = jobs.get(str(watched))
        if job is not None:
            return job
    return jobs.latest_job()


def _status_from_job(
    job: "JobSnapshot", store: "LearningStore"
) -> PipelineStatus:
    phase = PipelinePhase(job.state.value)
    run = store.get_pipeline_run(job.run_id) if job.run_id else None
    return PipelineStatus(
        phase=phase,
        percent=_percent(phase, job.overall_percent),
        elapsed_seconds=_job_elapsed_seconds(job),
        job_id=job.job_id,
        run_id=job.run_id,
        filename=job.source_filename or _text(run, "source_filename"),
        stage_key=job.stage_key,
        stage_label=job.stage_label,
        stage_detail=job.stage_detail,
        error_summary=job.error_summary,
        partial_artifacts=job.partial_artifacts,
        source_row_count=_row_count(run),
    )


def _status_from_latest_run(store: "LearningStore") -> PipelineStatus:
    """Describe the most recent run when no dashboard job exists.

    A run left ``PROCESSING`` by a killed worker deliberately resolves to
    ``IDLE``: without a job record there is no evidence anything owns it, and
    claiming otherwise is how the sidebar used to advertise a dead run forever.
    """
    runs = store.list_pipeline_runs(limit=1)
    if not runs:
        return PipelineStatus(phase=PipelinePhase.IDLE)
    run = runs[0]
    phase = _phase_from_run_status(run.get("status"))
    if phase is PipelinePhase.IDLE:
        return PipelineStatus(phase=PipelinePhase.IDLE)
    return PipelineStatus(
        phase=phase,
        percent=_percent(phase, 0.0),
        elapsed_seconds=_span_seconds(
            run.get("started_at"), run.get("completed_at")
        )
        or 0.0,
        run_id=_text(run, "run_id"),
        filename=_text(run, "source_filename"),
        error_summary=_text(run, "error_summary"),
        source_row_count=_row_count(run),
    )


def _phase_from_run_status(status: object) -> PipelinePhase:
    text = str(status or "").upper()
    if text.startswith("COMPLETED"):
        return PipelinePhase.COMPLETED
    if text.startswith("CANCELLED"):
        return PipelinePhase.CANCELLED
    if text.startswith("FAILED"):
        return PipelinePhase.FAILED
    return PipelinePhase.IDLE


def _percent(phase: PipelinePhase, recorded: float | None) -> float:
    if phase is PipelinePhase.COMPLETED:
        return 100.0
    if phase is PipelinePhase.IDLE:
        return 0.0
    return min(MAX_INCOMPLETE_PERCENT, max(0.0, float(recorded or 0.0)))


def _job_elapsed_seconds(job: "JobSnapshot") -> float:
    """Wall-clock time for a live job; the real span for a finished one.

    ``elapsed_seconds`` only advances when the pipeline emits progress, so a
    finished job that died mid-stage would otherwise report the age of its last
    update rather than how long it actually ran.
    """
    if job.is_active:
        return max(0.0, job.live_elapsed_seconds)
    span = _span_seconds(job.created_at, job.finished_at)
    return span if span is not None else max(0.0, job.elapsed_seconds)


def _span_seconds(start: object, end: object) -> float | None:
    started, finished = _parse_timestamp(start), _parse_timestamp(end)
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _text(record: Mapping[str, Any] | None, key: str) -> str | None:
    value = (record or {}).get(key)
    return str(value) if value else None


def _row_count(record: Mapping[str, Any] | None) -> int:
    try:
        return int((record or {}).get("source_row_count") or 0)
    except (TypeError, ValueError):
        return 0
