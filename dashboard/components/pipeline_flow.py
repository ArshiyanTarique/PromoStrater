"""Animated pipeline flow for a live processing run.

Replaces the flat percentage bar with the run's actual shape: one cell per
entry in :data:`dashboard.services.progress.PROCESSING_STAGES`, a filled bar
on every stage already finished, and a spark sweeping the stage doing work
right now.

The cells sit in a grid that flows across the panel rather than stacking in a
single column, which on a wide screen left everything right of the labels
empty. Reading order carries the sequence, so each unfinished cell shows its
position - there is no drawn wire between them to follow once the grid wraps.
Placement stays ordered rather than scattered on purpose: the live panel is a
1s fragment, and positions that moved on each rerun would churn on screen.

Stage completion is derived from position, not from a second source of truth.
``ProgressTracker.update`` forces every earlier stage's fraction to 1.0 when it
advances, so "index below the active stage" *is* "finished" by construction.

The live panel on ``pages/1_Upload_and_Process.py`` is a 1s ``st.fragment``.
Every rerun replaces this markup wholesale, which would restart the spark's
keyframes and read as a stutter. ``_travel_offset`` phase-locks the animation
to the wall clock with a negative ``animation-delay`` so a replaced node picks
the motion up where the old one left it.

Reduced-motion needs no handling here: the global guard closing ``theme.py``
already flattens every animation on the page.
"""

from __future__ import annotations

import html
import time
from typing import Literal

import streamlit as st

from dashboard.services.progress import PROCESSING_STAGES

#: Seconds for one spark traversal. Also the phase-lock period.
TRAVEL_SECONDS = 1.6

Outcome = Literal["running", "cancelling", "succeeded", "stopped"]

_FLOW_CSS = """
<style>
.ps-flow {
    display: flex;
    flex-direction: column;
    margin: 0.35rem 0 0.15rem;
}
.ps-flow-head, .ps-flow-tail {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ps-text-secondary);
}
.ps-flow-cap {
    width: 18px;
    display: flex;
    justify-content: center;
    flex: none;
}
.ps-flow-cap span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--ps-text-secondary);
    opacity: 0.55;
}

/* ---- grid ----
 * The stages used to stack in one column, which left the whole right-hand
 * side of a wide panel empty. auto-fit lays them across the available width
 * instead - four across on a wide screen, folding to three, two, then one as
 * it narrows. Order still reads left-to-right, top-to-bottom, which matters
 * more here than in the old rail: without a drawn wire between nodes, the
 * numbering and the reading order are what carry the sequence. */
.ps-flow-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.5rem;
    margin: 0.45rem 0;
    /* Clips the link trailing the last cell in each row. The column count is
       whatever auto-fit decides, so no selector can know which cell ends a
       row - but that cell's link is the only one that reaches past the grid's
       right edge, and clipping is exactly what should happen to it. */
    overflow: hidden;
}

/* The cell clips its own overflow so the spark cannot escape it, which means
   the link has to live outside the cell. Each stage is therefore a slot: the
   cell, plus a connector reaching across the gap to the next slot. The state
   class rides the slot so both parts read from it. */
.ps-flow-slot { position: relative; min-width: 0; }
.ps-flow-link {
    position: absolute;
    top: 50%;
    right: -0.5rem;
    width: 0.5rem;
    height: 2px;
    margin-top: -1px;
    background: var(--ps-border);
    border-radius: 2px;
    transition: background-color 0.4s var(--ps-ease);
}
.is-done .ps-flow-link { background: var(--ps-primary); }

.ps-flow-cell {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    min-width: 0;
    padding: 0.6rem 0.75rem;
    border: 1px solid var(--ps-border);
    border-radius: 12px;
    background: var(--ps-glass);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    transition: border-color 0.3s var(--ps-ease),
                background-color 0.3s var(--ps-ease);
}
/* The rail became a bar across the top of each cell: the same "filled behind
   you, empty ahead of you" reading, one cell at a time. */
.ps-flow-cell::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: var(--ps-border);
    transition: background-color 0.4s var(--ps-ease);
}
.is-done .ps-flow-cell::before { background: var(--ps-primary); }
.is-active .ps-flow-cell::before {
    background: var(--ps-gradient);
    background-size: 200% 100%;
}
.is-stopped .ps-flow-cell::before { background: var(--ps-error); }
.is-held .ps-flow-cell::before { background: var(--ps-warning); }
.is-active .ps-flow-cell,
.is-stopped .ps-flow-cell,
.is-held .ps-flow-cell {
    border-color: rgba(var(--ps-primary-rgb), 0.42);
}

/* ---- node ---- */
.ps-flow-node {
    width: 18px; height: 18px;
    border-radius: 50%;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 700;
    line-height: 1;
    color: var(--ps-on-accent);
    border: 2px solid var(--ps-border);
    background: var(--ps-bg-card);
    transition: background-color 0.4s var(--ps-ease),
                border-color 0.4s var(--ps-ease);
}
/* A pending node carries its number, not the accent fill, so the sequence is
   still readable once the connecting rail is gone. */
.is-pending .ps-flow-node { color: var(--ps-text-secondary); }
.is-done .ps-flow-node {
    background: var(--ps-primary);
    border-color: var(--ps-primary);
}
.is-active .ps-flow-node {
    background: var(--ps-primary);
    border-color: var(--ps-primary);
    animation: ps-flow-pulse 1.9s var(--ps-ease) infinite;
}
.is-stopped .ps-flow-node {
    background: var(--ps-error);
    border-color: var(--ps-error);
}
.is-held .ps-flow-node {
    background: var(--ps-warning);
    border-color: var(--ps-warning);
}

@keyframes ps-flow-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(var(--ps-primary-rgb), 0.42); }
    70%  { box-shadow: 0 0 0 9px rgba(var(--ps-primary-rgb), 0); }
    100% { box-shadow: 0 0 0 0 rgba(var(--ps-primary-rgb), 0); }
}

/* ---- spark ----
 * It used to travel the vertical link feeding the working stage. There is no
 * link now, so it sweeps that stage's own bar instead - same signal, same
 * phase-lock, no wire required. */
.is-active .ps-flow-cell::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 38%;
    height: 3px;
    background: linear-gradient(90deg, transparent,
                rgba(var(--ps-card-rgb), 0.92), transparent);
    animation: ps-flow-travel __TRAVEL__s linear infinite;
    animation-delay: __OFFSET__s;
}
@keyframes ps-flow-travel {
    0%   { transform: translateX(-110%); opacity: 0; }
    18%  { opacity: 1; }
    82%  { opacity: 1; }
    100% { transform: translateX(370%);  opacity: 0; }
}

/* ---- body ---- */
.ps-flow-body { min-width: 0; }
.ps-flow-label {
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1.2;
    overflow-wrap: break-word;
    color: var(--ps-text-secondary);
    opacity: 0.62;
}
.is-done .ps-flow-label { color: var(--ps-text-primary); opacity: 1; }
.is-active .ps-flow-label,
.is-stopped .ps-flow-label,
.is-held .ps-flow-label {
    color: var(--ps-text-primary);
    opacity: 1;
    font-weight: 700;
}
/* The live detail is a full-width line under the grid rather than text inside
   one cell: strings like "Generated candidates for 7,688 of 18,046 own-brand
   offers" would otherwise stretch a single cell and leave the row ragged. */
.ps-flow-detail {
    font-size: 0.72rem;
    color: var(--ps-text-secondary);
    line-height: 1.3;
    padding-left: 0.2rem;
}
.ps-flow-pct {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    color: var(--ps-primary);
}
</style>
"""


def _travel_offset() -> float:
    """Negative delay that phase-locks the spark across fragment reruns."""
    return -(time.monotonic() % TRAVEL_SECONDS)


def _cell(*, css_class: str, glyph: str, label: str, is_last: bool) -> str:
    # The final stage leads nowhere, so it carries no link. Every other link
    # is drawn; the grid clips whichever ones end a row.
    link = "" if is_last else '<span class="ps-flow-link"></span>'
    return (
        f'<div class="ps-flow-slot {css_class}">'
        f'<div class="ps-flow-cell">'
        f'<span class="ps-flow-node">{glyph}</span>'
        f'<div class="ps-flow-body">'
        f'<div class="ps-flow-label">{html.escape(label)}</div>'
        f"</div>"
        f"</div>"
        f"{link}"
        f"</div>"
    )


def render_pipeline_flow(
    *,
    stage_key: str | None,
    overall_percent: float,
    detail: str | None = None,
    outcome: Outcome = "running",
) -> None:
    """Draw the pipeline as nodes on an animated rail.

    ``stage_key`` is the worker's real position, straight off the durable job
    record. An unrecognised or missing key parks the run at the first stage
    rather than guessing a position it cannot support.
    """
    stage_keys = [stage.key for stage in PROCESSING_STAGES]
    if outcome == "succeeded":
        active_index = len(PROCESSING_STAGES)
    elif stage_key in stage_keys:
        active_index = stage_keys.index(stage_key)
    else:
        active_index = 0

    if outcome == "stopped":
        active_class, active_glyph = "is-stopped", "!"
    elif outcome == "cancelling":
        active_class, active_glyph = "is-held", ""
    else:
        active_class, active_glyph = "is-active", ""

    cells: list[str] = []
    last = len(PROCESSING_STAGES) - 1
    for index, stage in enumerate(PROCESSING_STAGES):
        # Every cell that is not finished carries its position, so the
        # sequence survives the wrap onto a second row.
        number = str(index + 1)
        if index < active_index:
            css_class, glyph = "is-done", "&#10003;"
        elif index == active_index:
            css_class = active_class
            glyph = active_glyph or number
        else:
            css_class, glyph = "is-pending", number
        cells.append(
            _cell(
                css_class=css_class,
                glyph=glyph,
                label=stage.label,
                is_last=index == last,
            )
        )

    tail_done = " is-done" if outcome == "succeeded" else ""
    css = _FLOW_CSS.replace("__TRAVEL__", f"{TRAVEL_SECONDS:g}").replace(
        "__OFFSET__", f"{_travel_offset():.3f}"
    )

    # Streamlit wraps each st.markdown call in its own container, so a partial
    # tree emitted alone would be auto-closed and the rows would land as
    # siblings. The whole flow goes out in one call.
    # Live detail belongs to the working stage, but is drawn once beneath the
    # grid so no single cell has to carry a long string.
    detail_html = (
        f'<div class="ps-flow-detail">{html.escape(detail)}</div>'
        if detail
        else ""
    )

    st.markdown(
        css
        + '<div class="ps-flow">'
        + '<div class="ps-flow-head">'
        + '<div class="ps-flow-cap"><span></span></div>'
        + "<div>Your data &middot; "
        + f'<span class="ps-flow-pct">{overall_percent:.0f}%</span></div>'
        + "</div>"
        + '<div class="ps-flow-grid">'
        + "".join(cells)
        + "</div>"
        + detail_html
        + f'<div class="ps-flow-tail{tail_done}">'
        + '<div class="ps-flow-cap"><span></span></div>'
        + "<div>Results</div>"
        + "</div>"
        + "</div>",
        unsafe_allow_html=True,
    )
