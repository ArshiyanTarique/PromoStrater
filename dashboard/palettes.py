"""Named colour palettes for the dashboard.

``theme.py`` owns layout, motion, and state; this module owns nothing but
colour. The CSS there references ``var(--ps-*)`` throughout, so swapping a
palette rewrites one ``:root`` block rather than 600 lines of rules.

Two palettes ship:

``aurora``  the original sunset scheme — deep violet leads, magenta bridges,
            amber closes, light sidebar. Preserved verbatim so the dashboard
            can be moved back to it.
``slate``   a light interface with dark charcoal sidebar sections, a single
            indigo accent instead of a rainbow gradient, and Tailwind-style
            semantic colours.

Select with the ``PROMOSTRATER_THEME`` environment variable; ``slate`` is the
default. Streamlit's *native* widget colours live in ``.streamlit/config.toml``
and are read once at startup, so a full switch means swapping that file too —
see the matching ``config.aurora.toml`` / ``config.slate.toml``.
"""

from __future__ import annotations

import os

#: Colours used by every rule in ``theme.py``. ``*_rgb`` entries are bare
#: triples so rules can compose their own alpha via ``rgba(var(--token), a)``.
AURORA: dict[str, str] = {
    # Surfaces
    "page-bg": "#ffffff",
    "card": "#ffffff",
    "card-rgb": "255, 255, 255",
    "elevated": "#f7f4fe",
    "tint": "#fbf9ff",
    "border": "#e9e2f8",
    "border-light": "#f1ebfd",
    # Text
    "text": "#2a2340",
    "text-2": "#6b6383",
    "on-accent": "#ffffff",
    # Primary
    "primary": "#7c3aed",
    "primary-hover": "#6428d4",
    "primary-rgb": "124, 58, 237",
    "primary-ink": "#6428d4",
    # Gradient partners
    "accent": "#f7941e",
    "accent-rgb": "247, 148, 30",
    "accent-2": "#e8558c",
    "accent-2-rgb": "232, 85, 140",
    "deep-1": "#3f1c7d",
    "deep-2": "#c2409b",
    # Semantic
    "success": "#0ca678",
    "success-rgb": "12, 166, 120",
    "success-ink": "#0b8f68",
    "warning": "#f0a202",
    "warning-rgb": "240, 162, 2",
    "warning-ink": "#a86f00",
    "error": "#e03131",
    "error-rgb": "224, 49, 49",
    "error-ink": "#c92a2a",
    "info": "#1098ad",
    "info-rgb": "16, 152, 173",
    "info-ink": "#0b7285",
    "neutral-rgb": "107, 99, 131",
    "neutral-ink": "#5b5474",
    # Depth + disabled
    "shadow-rgb": "76, 42, 133",
    "disabled-bg": "#ded6f3",
    "disabled-ink": "#9990b5",
    # Sidebar. Aurora's is light, so these mirror the main surfaces.
    "side-bg": "#fbf9ff",
    "side-card": "#ffffff",
    "side-card-rgb": "255, 255, 255",
    "side-border": "#e9e2f8",
    "side-text": "#2a2340",
    "side-text-2": "#6b6383",
    "side-hover": "#f7f4fe",
    "side-accent-rgb": "124, 58, 237",
}

SLATE: dict[str, str] = {
    # Surfaces — light page, translucent white cards.
    "page-bg": "#f7f8fa",
    "card": "#ffffff",
    "card-rgb": "255, 255, 255",
    "elevated": "#eef0f4",
    "tint": "#fafbfc",
    "border": "#e5e7eb",
    "border-light": "#eef0f4",
    # Text
    "text": "#111827",
    "text-2": "#6b7280",
    "on-accent": "#ffffff",
    # Primary
    "primary": "#4f46e5",
    "primary-hover": "#4338ca",
    "primary-rgb": "79, 70, 229",
    "primary-ink": "#4338ca",
    # One indigo family rather than a rainbow: the gradient reads as a single
    # accent shifting in value, not as three competing hues.
    # Ordered light -> mid -> primary so the ramp reads as one hue changing
    # value, which is what --ps-gradient composes.
    "accent": "#818cf8",
    "accent-rgb": "129, 140, 248",
    "accent-2": "#6366f1",
    "accent-2-rgb": "99, 102, 241",
    "deep-1": "#111827",
    "deep-2": "#312e81",
    # Semantic
    "success": "#16a34a",
    "success-rgb": "22, 163, 74",
    "success-ink": "#15803d",
    "warning": "#f59e0b",
    "warning-rgb": "245, 158, 11",
    "warning-ink": "#b45309",
    "error": "#dc2626",
    "error-rgb": "220, 38, 38",
    "error-ink": "#b91c1c",
    "info": "#0891b2",
    "info-rgb": "8, 145, 178",
    "info-ink": "#0e7490",
    "neutral-rgb": "107, 114, 128",
    "neutral-ink": "#4b5563",
    # Depth + disabled
    "shadow-rgb": "17, 24, 39",
    "disabled-bg": "#e5e7eb",
    "disabled-ink": "#9ca3af",
    # Sidebar — the dark charcoal section. Text flips to light here, so the
    # sidebar carries its own text tokens rather than inheriting the page's.
    "side-bg": "#111827",
    "side-card": "#1f2937",
    "side-card-rgb": "31, 41, 55",
    "side-border": "#374151",
    "side-text": "#f9fafb",
    "side-text-2": "#9ca3af",
    "side-hover": "#1f2937",
    # Lifted off the primary so the rail reads against charcoal.
    "side-accent-rgb": "129, 140, 248",
}

PALETTES: dict[str, dict[str, str]] = {"aurora": AURORA, "slate": SLATE}

DEFAULT_PALETTE = "slate"


def active_palette_name() -> str:
    """Palette selected by ``PROMOSTRATER_THEME``, falling back to the default.

    An unknown name falls back rather than raising: a typo in an environment
    variable should not take the dashboard down.
    """
    requested = os.environ.get("PROMOSTRATER_THEME", DEFAULT_PALETTE).strip().lower()
    return requested if requested in PALETTES else DEFAULT_PALETTE


def active_palette() -> dict[str, str]:
    return PALETTES[active_palette_name()]


def root_variables(palette: dict[str, str]) -> str:
    """Render a palette as the body of a CSS ``:root`` block."""
    return "\n".join(f"    --ps-{token}: {value};" for token, value in palette.items())
