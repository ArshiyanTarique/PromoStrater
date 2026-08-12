"""Durable job lifecycle vocabulary and the cooperative cancellation token.

Cancellation is cooperative by design: nothing here kills a thread or a
process. A token is set to *requested*, and the processing stages that already
report progress observe it at their existing checkpoints and stop cleanly.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import StrEnum


class JobState(StrEnum):
    """Explicit lifecycle for one durable dashboard processing job."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


#: States where a worker is expected to be alive and holding the run.
ACTIVE_JOB_STATES = frozenset(
    {JobState.QUEUED, JobState.RUNNING, JobState.CANCELLING}
)
#: States that are final; a new job may always start after one of these.
TERMINAL_JOB_STATES = frozenset(
    {JobState.CANCELLED, JobState.COMPLETED, JobState.FAILED}
)

#: Persisted ``pipeline_runs.status`` written when a run is cancelled. It
#: deliberately does not start with ``COMPLETED`` so existing duplicate and
#: completion queries keep treating a cancelled run as *not* completed.
CANCELLED_RUN_STATUS = "CANCELLED_DASHBOARD"
#: ``pipeline_runs.status`` for a job whose worker disappeared.
ORPHANED_RUN_STATUS = "FAILED_DASHBOARD_ORPHANED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunCancelled(Exception):
    """Raised inside a stage checkpoint once cancellation was requested."""

    def __init__(self, stage_key: str | None = None) -> None:
        self.stage_key = stage_key
        super().__init__(
            "Processing was cancelled"
            + (f" during stage {stage_key}" if stage_key else "")
        )


class CancellationToken:
    """Thread-safe, monotonic cancel flag shared with a running job.

    The token is intentionally one-way: once requested it never clears, so a
    stage that checks late still observes the request and stops.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._requested_at: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def requested_at(self) -> str | None:
        with self._lock:
            return self._requested_at

    def request(self, reason: str = "User requested cancellation") -> None:
        """Record a cancellation request without interrupting any stage."""
        with self._lock:
            if self._reason is None:
                self._reason = reason
                self._requested_at = utc_now_iso()
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancellation is requested; used only by tests/waiters."""
        return self._event.wait(timeout)

    def raise_if_cancelled(self, stage_key: str | None = None) -> None:
        """Stop the current stage cleanly at an existing checkpoint."""
        if self._event.is_set():
            raise RunCancelled(stage_key)
