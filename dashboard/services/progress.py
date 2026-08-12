"""Central processing stages and monotonic progress-state transitions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, MutableMapping


class ProcessingState(StrEnum):
    """Explicit lifecycle for one dashboard processing attempt."""

    IDLE = "IDLE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StageDefinition:
    key: str
    label: str
    weight: float


# One maintained source for overall-progress weighting. The weights reflect
# the observed production workload while preserving room for final exports.
PROCESSING_STAGES = (
    StageDefinition("validation", "Validating upload", 0.04),
    StageDefinition("input_loading", "Loading input", 0.08),
    StageDefinition("canonicalisation", "Preparing offers", 0.10),
    StageDefinition("inference", "Mapping own-brand SKUs", 0.43),
    StageDefinition("sku_mapping", "Building SKU mapping output", 0.05),
    StageDefinition(
        "competitor_discovery", "Discovering competitor offers", 0.22
    ),
    StageDefinition("exports", "Writing output files", 0.05),
    StageDefinition("persistence", "Saving run results", 0.03),
)

# Measurable sub-work within the inference stage, expressed as 1,000 stable
# units so callers never scatter arbitrary overall percentages.
INFERENCE_PHASE_RANGES = {
    # Assigning offer identities and decomposing commercial entities runs
    # before any candidate is generated and scales with every canonical
    # offer, not just the own-brand ones. It used to report nothing, so a
    # large upload sat at the stage's starting percentage for minutes.
    "preparation": (0, 60),
    "candidate_generation": (60, 200),
    "feature_generation": (200, 500),
    "lightgbm": (500, 800),
    "embedding": (800, 850),
    "agreement": (850, 920),
    "llm_review": (920, 960),
    "final_decisions": (960, 1000),
}
INFERENCE_PROGRESS_TOTAL = 1000


def inference_progress_units(
    phase: str,
    completed: int | None,
    total: int | None,
) -> tuple[int, int]:
    """Translate real phase work into the centralized inference allocation."""
    if phase not in INFERENCE_PHASE_RANGES:
        raise ValueError(f"Unknown inference phase: {phase}")
    start, end = INFERENCE_PHASE_RANGES[phase]
    if total is None:
        return start, INFERENCE_PROGRESS_TOTAL
    if total <= 0:
        return end, INFERENCE_PROGRESS_TOTAL
    fraction = max(0.0, min(float(completed or 0) / total, 1.0))
    units = start + round((end - start) * fraction)
    return units, INFERENCE_PROGRESS_TOTAL


@dataclass(frozen=True)
class ProgressUpdate:
    """One presentation-safe snapshot of measurable processing work."""

    state: ProcessingState
    stage_key: str | None
    stage_label: str | None
    overall_percent: float
    completed: int | None = None
    total: int | None = None
    detail: str | None = None
    last_completed_stage: str | None = None
    run_id: str | None = None
    error_summary: str | None = None
    technical_details: str | None = None
    partial_artifacts: tuple[str, ...] = ()
    pipeline_status: str | None = None
    original_exception: dict[str, Any] | None = None
    completed_stage_keys: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    @property
    def has_known_total(self) -> bool:
        return self.total is not None


ProgressCallback = Callable[[ProgressUpdate], None]
PROCESSING_SESSION_KEY = "processing_progress"


def idle_progress_update() -> ProgressUpdate:
    return ProgressUpdate(
        state=ProcessingState.IDLE,
        stage_key=None,
        stage_label=None,
        overall_percent=0.0,
    )


def reset_session_progress(
    session_state: MutableMapping[str, Any],
) -> ProgressUpdate:
    """Clear every prior attempt before a new run starts."""
    update = idle_progress_update()
    session_state[PROCESSING_SESSION_KEY] = update
    return update


def apply_session_progress(
    session_state: MutableMapping[str, Any],
    update: ProgressUpdate,
) -> None:
    session_state[PROCESSING_SESSION_KEY] = update


class ProgressTracker:
    """Calculate weighted, monotonic progress without invented denominators."""

    def __init__(
        self,
        callback: ProgressCallback | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.callback = callback
        self.clock = clock
        self.started_at = clock()
        self.state = ProcessingState.IDLE
        self.current_stage_key: str | None = None
        self.last_completed_stage: str | None = None
        self._fractions = {stage.key: 0.0 for stage in PROCESSING_STAGES}
        self._maximum_percent = 0.0
        self.latest = ProgressUpdate(
            state=ProcessingState.IDLE,
            stage_key=None,
            stage_label=None,
            overall_percent=0.0,
        )

    @staticmethod
    def _definition(stage_key: str) -> StageDefinition:
        for stage in PROCESSING_STAGES:
            if stage.key == stage_key:
                return stage
        raise ValueError(f"Unknown processing stage: {stage_key}")

    def _overall_percent(self) -> float:
        weighted = sum(
            stage.weight * self._fractions[stage.key]
            for stage in PROCESSING_STAGES
        )
        return min(100.0, max(self._maximum_percent, weighted * 100.0))

    def _emit(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
        run_id: str | None = None,
        error_summary: str | None = None,
        technical_details: str | None = None,
        partial_artifacts: tuple[str, ...] = (),
        pipeline_status: str | None = None,
        original_exception: dict[str, Any] | None = None,
    ) -> ProgressUpdate:
        definition = (
            self._definition(self.current_stage_key)
            if self.current_stage_key is not None
            else None
        )
        percent = self._overall_percent()
        if self.state is ProcessingState.SUCCEEDED:
            percent = 100.0
        elif self.state in {
            ProcessingState.FAILED,
            ProcessingState.CANCELLING,
            ProcessingState.CANCELLED,
        }:
            # A stopped run never claims completion, however far it reached.
            percent = min(percent, 99.99)
        self._maximum_percent = max(self._maximum_percent, percent)
        self.latest = ProgressUpdate(
            state=self.state,
            stage_key=self.current_stage_key,
            stage_label=definition.label if definition else None,
            overall_percent=percent,
            completed=completed,
            total=total,
            detail=detail,
            last_completed_stage=self.last_completed_stage,
            run_id=run_id,
            error_summary=error_summary,
            technical_details=technical_details,
            partial_artifacts=partial_artifacts,
            pipeline_status=pipeline_status,
            original_exception=original_exception,
            completed_stage_keys=tuple(
                stage.key
                for stage in PROCESSING_STAGES
                if self._fractions[stage.key] >= 1.0
                and stage.key != self.current_stage_key
            ),
            elapsed_seconds=max(0.0, self.clock() - self.started_at),
        )
        if self.callback is not None:
            self.callback(self.latest)
        return self.latest

    def start(self) -> ProgressUpdate:
        """Start a clean attempt with no stale stage or percentage."""
        self.started_at = self.clock()
        self.state = ProcessingState.RUNNING
        self.current_stage_key = PROCESSING_STAGES[0].key
        self.last_completed_stage = None
        self._fractions = {stage.key: 0.0 for stage in PROCESSING_STAGES}
        self._maximum_percent = 0.0
        return self._emit(detail="Starting processing")

    def update(
        self,
        stage_key: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
        run_id: str | None = None,
    ) -> ProgressUpdate:
        """Record real work for a stage and calculate weighted progress."""
        # CANCELLING is still live work: the in-flight stage keeps reporting
        # real progress until it reaches its next cancellation checkpoint.
        if self.state not in {
            ProcessingState.RUNNING,
            ProcessingState.CANCELLING,
        }:
            raise RuntimeError(
                "Progress can only update while RUNNING or CANCELLING"
            )
        definition = self._definition(stage_key)
        stage_keys = [stage.key for stage in PROCESSING_STAGES]
        target_index = stage_keys.index(stage_key)
        for earlier in PROCESSING_STAGES[:target_index]:
            if self._fractions[earlier.key] < 1.0:
                self._fractions[earlier.key] = 1.0
        if target_index:
            self.last_completed_stage = PROCESSING_STAGES[
                target_index - 1
            ].label
        self.current_stage_key = stage_key

        if total is not None:
            if total <= 0:
                fraction = 1.0
                completed = 0
                total = 0
            else:
                normalized_completed = max(0, min(int(completed or 0), total))
                completed = normalized_completed
                fraction = normalized_completed / total
            self._fractions[stage_key] = max(
                self._fractions[stage_key], fraction
            )
        return self._emit(
            completed=completed,
            total=total,
            detail=detail,
            run_id=run_id,
        )

    def succeed(
        self, *, run_id: str, detail: str | None = None
    ) -> ProgressUpdate:
        for stage in PROCESSING_STAGES:
            self._fractions[stage.key] = 1.0
        self.state = ProcessingState.SUCCEEDED
        self.current_stage_key = PROCESSING_STAGES[-1].key
        self.last_completed_stage = PROCESSING_STAGES[-1].label
        return self._emit(run_id=run_id, detail=detail or "Processing complete")

    def cancelling(
        self, *, run_id: str | None = None, detail: str | None = None
    ) -> ProgressUpdate:
        """Announce an observed cancel request without ending the stage.

        The current stage keeps running to its next checkpoint, so progress
        stays truthful about where the run actually is.
        """
        self.state = ProcessingState.CANCELLING
        return self._emit(
            run_id=run_id,
            detail=(
                detail
                or "Cancelling... waiting for current stage to finish"
            ),
        )

    def cancelled(
        self,
        *,
        run_id: str | None,
        detail: str | None = None,
        partial_artifacts: tuple[str, ...] = (),
    ) -> ProgressUpdate:
        """Record a clean cooperative stop at a stage boundary."""
        self.state = ProcessingState.CANCELLED
        return self._emit(
            run_id=run_id,
            partial_artifacts=partial_artifacts,
            pipeline_status="CANCELLED_DASHBOARD",
            detail=detail or "Processing cancelled",
        )

    def fail(
        self,
        *,
        run_id: str | None,
        error_summary: str,
        technical_details: str,
        partial_artifacts: tuple[str, ...] = (),
        pipeline_status: str | None = None,
        original_exception: dict[str, Any] | None = None,
    ) -> ProgressUpdate:
        self.state = ProcessingState.FAILED
        return self._emit(
            run_id=run_id,
            error_summary=error_summary,
            technical_details=technical_details,
            partial_artifacts=partial_artifacts,
            pipeline_status=pipeline_status,
            original_exception=original_exception,
            detail="Processing stopped",
        )
