"""A live elapsed clock that advances every second without a server rerun.

The elapsed reading used to be a string the server rendered on each fragment
poll, so it only advanced when a poll happened: the number skipped, repeated,
and - because the sidebar and the run panel poll on independent timers - the
two could sit a second apart.

The clock is therefore driven by CSS. Two registered integer custom properties
are stepped by ``steps()`` animations - one per second, one per minute - and
printed through CSS counters. A negative ``animation-delay`` carrying the run's
elapsed seconds phase-locks that animation to where the run actually is, the
same trick ``pipeline_flow`` uses to keep its spark continuous across reruns.

**One animation drives every clock on the page.** It runs on ``.stApp`` and the
properties are declared ``inherits: true``, so each clock only *prints* what
that single animation holds. An earlier version ran an animation per clock,
anchored per element: each was correct at its own paint, but the sidebar and
the run panel paint on separate fragment cycles, so their phases differed by
however long each took to reach the screen and the two readings straddled a
second boundary - 2m 17s beside 2m 16s. With one animation there is one phase
and nothing left to drift; the surfaces cannot disagree because there is only
one clock.

Elapsed is measured from the job's ``created_at`` on the server, so the reading
is the run's real age; the browser clock is never consulted. A fragment rerun
only re-anchors the phase, and whichever surface writes the anchor last sets it
for every clock at once.
"""

from __future__ import annotations

import html

#: Steps in each animation. Seconds wrap every minute; minutes run for a week,
#: which is far longer than any run and keeps the counter from wrapping to 0.
_SECONDS_CYCLE = 60
_MINUTES_CYCLE = 7 * 24 * 60
_MINUTES_DURATION = _MINUTES_CYCLE * 60

#: Key on the container wrapping the run panel's Elapsed metric. Streamlit
#: renders it as an ``st-key-*`` class, which is the only handle the metric
#: gives for styling one card and not its siblings.
ELAPSED_CLOCK_KEY = "ps-elapsed-clock"

#: Nudges the animation off the exact second boundary the server measured.
#: Two things want this. The reading is taken on the server and the animation
#: is re-anchored when the browser paints, so the clock is already this much
#: behind by then; and a phase parked exactly on a ``steps()`` edge can round to
#: the step below, which showed 2m 10s beside a static 2m 11s.
_PAINT_BIAS_SECONDS = 0.05

#: Printing the shared counters. Every clock, whatever its shape, runs this and
#: nothing else - no animation of its own, so none of them can drift apart.
_PRINT_RULES = """
    content: counter(ps-clock-m) "m " counter(ps-clock-s) "s";
    counter-reset: ps-clock-m var(--ps-clock-min)
                   ps-clock-s var(--ps-clock-sec);
"""

CLOCK_CSS = f"""
<style>
@property --ps-clock-sec {{
    syntax: "<integer>";
    initial-value: 0;
    inherits: true;
}}
@property --ps-clock-min {{
    syntax: "<integer>";
    initial-value: 0;
    inherits: true;
}}

@keyframes ps-clock-sec {{
    from {{ --ps-clock-sec: 0; }}
    to   {{ --ps-clock-sec: {_SECONDS_CYCLE}; }}
}}
@keyframes ps-clock-min {{
    from {{ --ps-clock-min: 0; }}
    to   {{ --ps-clock-min: {_MINUTES_CYCLE}; }}
}}

/* The one clock. It lives on the app root - the only ancestor the sidebar and
   the main panel share - and every reading inherits from it. --ps-clock-delay
   is written per render by whichever surface is drawing; until a run is live
   it is unset and the animation sits at zero, unread by anybody. */
.stApp {{
    animation: ps-clock-sec {_SECONDS_CYCLE}s
                   steps({_SECONDS_CYCLE}, end) infinite,
               ps-clock-min {_MINUTES_DURATION}s
                   steps({_MINUTES_CYCLE}, end) infinite;
    animation-delay: var(--ps-clock-delay, 0s), var(--ps-clock-delay, 0s);
    /* Deliberately exempt from the reduced-motion flattening at the end of
       theme.py. That rule is right for decoration, but a clock frozen at the
       second it was painted reports the wrong time - the digits are
       information, and they carry no motion to be sensitive to. Class
       specificity plus !important is what outranks its `*` rule. */
    animation-duration: {_SECONDS_CYCLE}s, {_MINUTES_DURATION}s !important;
    animation-iteration-count: infinite !important;
}}

/* st.caption keeps only the text when handed inline HTML, so a line carrying
   the clock has to be markdown. It wears the caption's test id for colour, and
   restates the metrics the emotion classes would have supplied, so it sits in
   a stack of real captions without a visible seam. */
.ps-clock-line {{ margin: 0 0 -1rem; }}
.ps-clock-line p {{
    font-size: 0.875rem;
    line-height: 1.6;
    margin: 0 0 1rem;
}}

/* The counter replaces the server-rendered text only where the mechanism
   actually exists. A browser without registered properties keeps the static
   reading, which is correct but only advances when the panel polls - a
   degraded clock rather than a wrong one. */
@supports at-rule(@property) {{
    .ps-clock-live > .ps-clock-static {{ display: none; }}
    .ps-clock-live::after {{{_PRINT_RULES}    }}
}}
</style>
"""


def clock_anchor_css(elapsed_seconds: float, *, live: bool) -> str:
    """Point the page's single clock at where this run actually is.

    Emitted per render and only while a run is live. Both surfaces derive the
    anchor from the same snapshot, so it does not matter which of them writes
    it last - the value is the same and there is only one clock to write to.
    """
    if not live:
        return ""
    anchor = max(0.0, float(elapsed_seconds)) + _PAINT_BIAS_SECONDS
    return f"<style>.stApp {{ --ps-clock-delay: -{anchor:.2f}s; }}</style>"


def inline_clock_html(static_text: str, *, live: bool) -> str:
    """A clock sized for running text, e.g. the sidebar's progress line."""
    classes = "ps-clock ps-clock-live" if live else "ps-clock"
    return (
        f'<span class="{classes}">'
        f'<span class="ps-clock-static">{html.escape(static_text)}</span>'
        f"</span>"
    )


def caption_clock_html(prefix: str, static_text: str, *, live: bool) -> str:
    """A caption-shaped line ending in the clock."""
    return (
        '<div data-testid="stCaptionContainer" class="ps-clock-line">'
        f"<p>{html.escape(prefix)}"
        f"{inline_clock_html(static_text, live=live)}</p>"
        "</div>"
    )


def caption_line_html(label: str, value: str) -> str:
    """A caption-shaped line, for stacking beside one that carries a clock."""
    return (
        '<div data-testid="stCaptionContainer" class="ps-clock-line">'
        f"<p><strong>{html.escape(label)}</strong> {html.escape(value)}</p>"
        "</div>"
    )


def metric_clock_css(*, live: bool) -> str:
    """Turn the keyed ``st.metric``'s value into the live counter.

    A real metric is overlaid rather than imitated: a hand-built lookalike sat
    7px shorter than the three cards beside it and would drift again the next
    time the metric styling moved. While the run is not live no rule is emitted
    at all, so the card simply shows the value Streamlit rendered.
    """
    if not live:
        return ""
    value = f'.st-key-{ELAPSED_CLOCK_KEY} [data-testid="stMetricValue"]'
    return (
        "<style>"
        "@supports at-rule(@property) {"
        f"{value} > * {{ display: none; }}"
        f"{value}::after {{{_PRINT_RULES}}}"
        "}"
        "</style>"
    )
