"""Session-scoped dismissal of stale failure notices.

A failed run stays in SQLite exactly as it was recorded. Dismissing only hides
its banner for the current browser session, so one old failure stops following
the user across every page. Nothing here writes to the learning store, and a
refresh brings the notice back.
"""

from __future__ import annotations

from typing import Iterable

import streamlit as st

_STATE_KEY = "dismissed_failure_ids"


def _dismissed_ids() -> set[str]:
    """Return this session's mutable set of dismissed identifiers."""
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = set()
    return st.session_state[_STATE_KEY]


def _clean(identifiers: Iterable[object]) -> tuple[str, ...]:
    """Drop empty identifiers and normalise the rest to strings."""
    return tuple(str(value) for value in identifiers if value)


def is_failure_dismissed(*identifiers: object) -> bool:
    """Report whether any identifier for this failure has been dismissed."""
    dismissed = _dismissed_ids()
    return any(value in dismissed for value in _clean(identifiers))


def dismiss_failure(*identifiers: object) -> None:
    """Hide a failure notice for the rest of this session.

    Every identifier is recorded together. The sidebar knows a failure by its
    run id while the upload page knows it by job id, so storing both lets a
    single click clear the notice on both surfaces.
    """
    _dismissed_ids().update(_clean(identifiers))


def dismiss_button(
    *identifiers: object,
    key: str,
    label: str = "Dismiss",
) -> bool:
    """Render the quiet dismiss control. Returns True when it was clicked."""
    clicked = st.button(
        label,
        key=key,
        icon=":material/close:",
        type="tertiary",
        help=(
            "Hide this notice for the rest of this session. "
            "The run itself stays saved."
        ),
    )
    if clicked:
        dismiss_failure(*identifiers)
    return clicked
