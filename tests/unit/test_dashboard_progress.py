"""Progress lifecycle and weighted-work regression tests."""

from __future__ import annotations

from dashboard.services.progress import (
    PROCESSING_SESSION_KEY,
    ProcessingState,
    ProgressTracker,
    inference_progress_units,
    reset_session_progress,
)


def test_progress_is_monotonic_and_moves_inside_competitor_stage() -> None:
    updates = []
    tracker = ProgressTracker(updates.append)
    tracker.start()
    tracker.update("validation", completed=1, total=1)
    first = tracker.update(
        "competitor_discovery", completed=10, total=100
    )
    second = tracker.update(
        "competitor_discovery", completed=60, total=100
    )

    assert second.overall_percent > first.overall_percent
    assert all(
        current.overall_percent <= following.overall_percent
        for current, following in zip(updates, updates[1:], strict=False)
    )


def test_zero_and_unknown_work_do_not_create_false_percentages() -> None:
    tracker = ProgressTracker()
    tracker.start()
    unknown = tracker.update(
        "input_loading", detail="Waiting for a row count"
    )
    empty = tracker.update(
        "canonicalisation", completed=0, total=0
    )

    assert unknown.completed is None
    assert unknown.total is None
    assert unknown.overall_percent == 4.0
    assert empty.completed == 0
    assert empty.total == 0
    assert empty.overall_percent == 22.0


def test_success_reaches_100_and_failure_never_does() -> None:
    failed = ProgressTracker()
    failed.start()
    failed.update("inference", completed=500, total=1000)
    failure = failed.fail(
        run_id="failed-run",
        error_summary="builtins.ValueError: failure",
        technical_details="traceback",
    )

    succeeded = ProgressTracker()
    succeeded.start()
    success = succeeded.succeed(run_id="successful-run")

    assert failure.state is ProcessingState.FAILED
    assert failure.overall_percent < 100
    assert success.state is ProcessingState.SUCCEEDED
    assert success.overall_percent == 100


def test_reset_clears_stale_failure_before_another_run() -> None:
    state = {
        PROCESSING_SESSION_KEY: ProgressTracker().latest,
        "unrelated": "preserved",
    }
    stale = ProgressTracker()
    stale.start()
    state[PROCESSING_SESSION_KEY] = stale.fail(
        run_id="old-run",
        error_summary="old error",
        technical_details="old traceback",
    )

    reset = reset_session_progress(state)

    assert reset.state is ProcessingState.IDLE
    assert reset.overall_percent == 0
    assert reset.run_id is None
    assert reset.error_summary is None
    assert state["unrelated"] == "preserved"


def test_inference_phase_units_use_one_central_allocation() -> None:
    generated, total = inference_progress_units(
        "candidate_generation", 50, 100
    )
    featured, _ = inference_progress_units(
        "feature_generation", 50, 100
    )
    empty_llm, _ = inference_progress_units("llm_review", 0, 0)

    assert total == 1000
    assert generated == 130
    assert featured == 350
    assert empty_llm == 960


def test_preparation_phase_reports_before_candidate_generation() -> None:
    start, total = inference_progress_units("preparation", 0, 100)
    middle, _ = inference_progress_units("preparation", 50, 100)
    end, _ = inference_progress_units("preparation", 100, 100)

    assert total == 1000
    # The bar has to move during entity preparation, not only after it.
    assert start == 0
    assert middle == 30
    assert end == 60


def test_failed_stage_is_never_also_marked_completed() -> None:
    tracker = ProgressTracker()
    tracker.start()
    tracker.update("validation", completed=1, total=1)
    tracker.update("input_loading", completed=1, total=1)
    tracker.update("canonicalisation", completed=1, total=1)
    tracker.update("inference", completed=1000, total=1000)

    failure = tracker.fail(
        run_id="failed-inference",
        error_summary="builtins.ValueError: failure",
        technical_details="traceback",
        pipeline_status="MODEL_ERROR_SAFE_FALLBACK",
    )

    assert failure.stage_key == "inference"
    assert "inference" not in failure.completed_stage_keys
    assert failure.last_completed_stage == "Preparing offers"
    assert set(failure.completed_stage_keys) == {
        "validation",
        "input_loading",
        "canonicalisation",
    }
