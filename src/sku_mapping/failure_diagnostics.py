"""Structured exception evidence for safe-fallback pipeline boundaries."""

from __future__ import annotations

import traceback
from pathlib import Path
from types import TracebackType
from typing import Any


def _origin(traceback_object: TracebackType | None) -> dict[str, object]:
    frames = traceback.extract_tb(traceback_object)
    if not frames:
        return {
            "file": None,
            "function": None,
            "line": None,
        }
    frame = frames[-1]
    return {
        "file": frame.filename,
        "function": frame.name,
        "line": frame.lineno,
    }


def _next_exception(error: BaseException) -> BaseException | None:
    if error.__cause__ is not None:
        return error.__cause__
    if error.__context__ is not None and not error.__suppress_context__:
        return error.__context__
    return None


def capture_exception_details(error: BaseException) -> dict[str, Any]:
    """Capture the root cause, complete chain, origin, and full traceback."""
    outer_to_root: list[tuple[BaseException, str | None]] = []
    current: BaseException | None = error
    seen: set[int] = set()
    relation: str | None = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        outer_to_root.append((current, relation))
        relation = "cause" if current.__cause__ is not None else "context"
        current = _next_exception(current)

    root_error = outer_to_root[-1][0]
    root_origin = _origin(root_error.__traceback__)
    exception_chain: list[dict[str, Any]] = []
    for chain_error, outer_relation in reversed(outer_to_root):
        origin = _origin(chain_error.__traceback__)
        exception_chain.append(
            {
                "type": type(chain_error).__name__,
                "qualified_type": (
                    f"{type(chain_error).__module__}."
                    f"{type(chain_error).__qualname__}"
                ),
                "message": str(chain_error),
                **origin,
                "relation_to_outer": outer_relation,
            }
        )

    return {
        "type": type(root_error).__name__,
        "qualified_type": (
            f"{type(root_error).__module__}.{type(root_error).__qualname__}"
        ),
        "message": str(root_error),
        **root_origin,
        "source_filename": (
            Path(str(root_origin["file"])).name
            if root_origin["file"]
            else None
        ),
        "traceback": "".join(
            traceback.TracebackException.from_exception(
                error,
                capture_locals=False,
            ).format(chain=True)
        ),
        "exception_chain": exception_chain,
    }


def exception_summary(details: dict[str, Any]) -> str:
    """Return a concise root-cause summary from captured details."""
    qualified_type = str(
        details.get("qualified_type") or details.get("type") or "Exception"
    )
    message = str(details.get("message") or "")
    return f"{qualified_type}: {message}".rstrip(": ")
