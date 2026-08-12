"""Structured exception-chain fidelity tests."""


from __future__ import annotations

from pathlib import Path

from sku_mapping.failure_diagnostics import capture_exception_details


def _real_failing_application_function() -> None:
    int("not-a-number")


def _wrapped_application_failure() -> None:
    try:
        _real_failing_application_function()
    except ValueError as cause:
        raise RuntimeError("outer application context") from cause


def test_capture_preserves_root_origin_traceback_and_exception_chain() -> None:
    try:
        _wrapped_application_failure()
    except RuntimeError as error:
        details = capture_exception_details(error)

    assert details["type"] == "ValueError"
    assert "invalid literal" in details["message"]
    assert details["source_filename"] == Path(__file__).name
    assert details["file"] == __file__
    assert details["function"] == "_real_failing_application_function"
    assert isinstance(details["line"], int)
    assert "_wrapped_application_failure" in details["traceback"]
    assert "_real_failing_application_function" in details["traceback"]
    assert "direct cause" in details["traceback"]
    assert [
        item["type"] for item in details["exception_chain"]
    ] == ["ValueError", "RuntimeError"]
