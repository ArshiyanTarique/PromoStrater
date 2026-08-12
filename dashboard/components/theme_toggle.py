"""Runtime palette switch.

Changing the selection writes :data:`dashboard.theme.PALETTE_STATE_KEY`, which
``inject_theme`` reads on the next run. Streamlit reruns the script whenever a
widget changes, so the new stylesheet is injected without an explicit rerun.

What this *cannot* move: Streamlit reads ``.streamlit/config.toml`` once at
startup, so native widget accents, dataframe chrome, and chart colours stay on
whichever palette the server booted with. Everything driven by ``--ps-*``
switches immediately. Swapping ``config.toml`` and restarting closes the gap.
"""

from __future__ import annotations

import streamlit as st

from dashboard.palettes import PALETTES
from dashboard.theme import PALETTE_STATE_KEY, selected_palette_name

_WIDGET_KEY = "ps_palette_toggle"

_LABELS = {"slate": "Slate", "aurora": "Aurora"}


def _persist() -> None:
    """Copy the widget's value into the durable, non-widget session key."""
    choice = st.session_state.get(_WIDGET_KEY)
    if choice in PALETTES:
        st.session_state[PALETTE_STATE_KEY] = choice


def render_palette_toggle(*, label: str = "Theme") -> None:
    """Draw the two-option palette switch."""
    options = list(PALETTES)
    current = selected_palette_name()
    st.segmented_control(
        label,
        options=options,
        format_func=lambda value: _LABELS.get(value, value.title()),
        default=current,
        key=_WIDGET_KEY,
        on_change=_persist,
        label_visibility="collapsed",
        help=(
            "Switches every themed surface immediately. Native widget and "
            "chart colours come from .streamlit/config.toml and need a "
            "restart with the matching config to follow."
        ),
    )
