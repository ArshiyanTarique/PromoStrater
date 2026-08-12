"""Centralized visual theme system and CSS injection for PromoStrater.

Layout, motion, and state live here; colour lives in :mod:`dashboard.palettes`
and is rendered into the ``:root`` block by :func:`build_theme_css`. Every rule
below references ``var(--ps-*)``, so switching palettes never edits a rule.
Typography and radii are configured natively in ``.streamlit/config.toml``;
this module adds what config cannot express — motion, gradient surfaces, and
hover states.

The ``--ps-*`` custom properties and ``.ps-*`` class names are a contract with
``Dashboard.py``, ``components/formatters.py``, ``components/sidebar_status.py``
and the page modules. Names are preserved here; only their values changed.
"""

from __future__ import annotations

import streamlit as st

from dashboard.palettes import PALETTES, active_palette_name, root_variables

#: Plain session key, deliberately not a widget key. Streamlit garbage-collects
#: widget state for widgets that are not rendered, and the toggle only lives on
#: the home page - storing the choice here keeps it alive across page switches.
PALETTE_STATE_KEY = "_ps_palette"


def selected_palette_name() -> str:
    """Palette for this session: the toggle's choice, else the env default."""
    chosen = st.session_state.get(PALETTE_STATE_KEY)
    if isinstance(chosen, str) and chosen in PALETTES:
        return chosen
    return active_palette_name()


def selected_palette() -> dict[str, str]:
    return PALETTES[selected_palette_name()]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Global Root Colors - generated from dashboard/palettes.py */
:root {
__PALETTE__

    /* Aliases for the --ps-* names the page modules already reference. */
    --ps-bg-dark: var(--ps-page-bg);
    --ps-bg-card: var(--ps-card);
    --ps-bg-elevated: var(--ps-elevated);
    --ps-text-primary: var(--ps-text);
    --ps-text-secondary: var(--ps-text-2);

    /* Card surfaces are translucent rather than solid. A white rectangle on
       the ambient wash stops the eye dead at its edge; letting the wash read
       through at low strength keeps the card distinct without the hard cut.
       The blur diffuses the particle field behind it, so what shows through
       is a soft tint instead of individual dots. --ps-card-rgb is rebound
       inside the sidebar, so both surfaces stay on their own base colour. */
    --ps-glass: rgba(var(--ps-card-rgb), 0.62);
    --ps-glass-strong: rgba(var(--ps-card-rgb), 0.78);
    --ps-blur: blur(14px) saturate(1.35);
    /* The KPI cards' top-to-bottom fall, restated in alpha. ambient.py has to
       repeat this gradient to layer its sheen above it - keep the two in step. */
    --ps-glass-grad: linear-gradient(180deg, rgba(var(--ps-card-rgb), 0.70) 0%,
                                     rgba(var(--ps-card-rgb), 0.44) 100%);

    /* Accent ramp shared by every gradient surface in the app. */
    --ps-gradient: linear-gradient(95deg, var(--ps-accent) 0%,
                                  var(--ps-accent-2) 48%, var(--ps-primary) 100%);
    --ps-gradient-deep: linear-gradient(118deg, var(--ps-deep-1) 0%,
                                        var(--ps-primary) 38%, var(--ps-deep-2) 72%,
                                        var(--ps-accent) 108%);
    --ps-ease: cubic-bezier(0.22, 1, 0.36, 1);
    --ps-shadow: 0 10px 30px -12px rgba(var(--ps-shadow-rgb), 0.28);
    --ps-shadow-lift: 0 18px 40px -14px rgba(var(--ps-shadow-rgb), 0.42);
}

/* ------------------------------------------------------------------ *
 * Palette-driven surfaces
 *
 * config.toml is read once at startup, so anything left to it freezes at
 * whichever palette booted. These rules restate the surfaces the runtime
 * toggle has to be able to move - page background, and every piece of
 * sidebar text. Without the sidebar block, switching to a light sidebar
 * would leave config.toml's near-white text unreadable on it.
 * ------------------------------------------------------------------ */
body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--ps-page-bg);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebarNavLink"] span {
    color: var(--ps-side-text) !important;
}
/* Vega paints an opaque background onto the <svg> from config.toml's
   backgroundColor, which reads as a grey slab floating on the page surface.
   Clearing it lets the card - and the ambient wash - show through. */
[data-testid="stVegaLiteChart"] svg,
[data-testid="stArrowVegaLiteChart"] svg {
    background: transparent !important;
}

[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--ps-side-text-2) !important;
}

/* ------------------------------------------------------------------ *
 * Top bar
 *
 * Streamlit paints the header as a full-width opaque slab in config.toml's
 * backgroundColor. On the home page ambient.py clears .stApp so the canvas
 * can show through, which leaves that slab as the one flat surface up there
 * - a grey band with a hard seam under it, directly above the gradient page
 * title - while on every other page the same bar blends away. Clearing it
 * lets one surface run unbroken from the top edge on all of them.
 *
 * The toolbar row occupies y 15-41 of the 56px bar, so the mask starts its
 * fade at 78% (~44px): whatever scrolls underneath dissolves instead of
 * colliding with the bar, and the menu itself stays crisp.
 * ------------------------------------------------------------------ */
[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: blur(12px) saturate(1.06);
    -webkit-backdrop-filter: blur(12px) saturate(1.06);
    -webkit-mask-image: linear-gradient(to bottom, #000 78%, transparent 100%);
    mask-image: linear-gradient(to bottom, #000 78%, transparent 100%);
}

/* ------------------------------------------------------------------ *
 * Motion primitives
 * ------------------------------------------------------------------ */
@keyframes ps-rise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
}
@keyframes ps-drift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes ps-float {
    0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
    50%      { transform: translate3d(0, -16px, 0) scale(1.05); }
}
@keyframes ps-sheen {
    from { transform: translateX(-130%) skewX(-18deg); }
    to   { transform: translateX(330%) skewX(-18deg); }
}
@keyframes ps-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.45; transform: scale(0.82); }
}
@keyframes ps-stripes {
    from { background-position: 0 0; }
    to   { background-position: 42px 0; }
}

/* A soft aurora wash behind the whole app. */
.stApp {
    background-image:
        radial-gradient(58rem 30rem at 12% -8%, rgba(var(--ps-primary-rgb), 0.10), transparent 62%),
        radial-gradient(46rem 26rem at 92% 4%, rgba(var(--ps-accent-rgb), 0.10), transparent 60%),
        radial-gradient(52rem 32rem at 70% 102%, rgba(var(--ps-accent-2-rgb), 0.08), transparent 62%);
    background-attachment: fixed;
}

/* Staggered page entrance. Direct children only, so nothing animates twice. */
.stMainBlockContainer > .stVerticalBlock > * {
    animation: ps-rise 0.5s var(--ps-ease) both;
}
.stMainBlockContainer > .stVerticalBlock > *:nth-child(1) { animation-delay: 0.02s; }
.stMainBlockContainer > .stVerticalBlock > *:nth-child(2) { animation-delay: 0.06s; }
.stMainBlockContainer > .stVerticalBlock > *:nth-child(3) { animation-delay: 0.10s; }
.stMainBlockContainer > .stVerticalBlock > *:nth-child(4) { animation-delay: 0.14s; }
.stMainBlockContainer > .stVerticalBlock > *:nth-child(5) { animation-delay: 0.18s; }
.stMainBlockContainer > .stVerticalBlock > *:nth-child(6) { animation-delay: 0.22s; }
.stMainBlockContainer > .stVerticalBlock > *:nth-child(n+7) { animation-delay: 0.26s; }

/* The page title carries the sunset ramp. */
.stMainBlockContainer [data-testid="stHeading"] h1 {
    font-family: 'Poppins', 'Inter', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: var(--ps-gradient);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: ps-drift 12s ease-in-out infinite;
}

/* App Header Fix - Rename sidebar title */
[data-testid="stSidebarNav"]::before {
    content: "PromoStrater";
    display: block;
    padding: 1.25rem 1.5rem 0.5rem;
    font-family: 'Poppins', 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: var(--ps-gradient);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: ps-drift 12s ease-in-out infinite;
}

/* Hide raw app label if present in sidebar header */
[data-testid="stSidebarNav"] > ul {
    padding-top: 0.5rem;
}

/* Card Container Styles */
.ps-card {
    background-color: var(--ps-glass);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    border: 1px solid var(--ps-border);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--ps-shadow);
    transition: transform 0.28s var(--ps-ease), box-shadow 0.28s var(--ps-ease),
                border-color 0.28s var(--ps-ease);
}

.ps-card:hover {
    transform: translateY(-4px);
    border-color: rgba(var(--ps-primary-rgb), 0.42);
    box-shadow: var(--ps-shadow-lift);
}

.ps-card-header {
    font-family: 'Poppins', 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--ps-text-primary);
    margin-bottom: 0.75rem;
}

/* KPI Card Grid */
.ps-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.ps-kpi-card {
    position: relative;
    overflow: hidden;
    background: var(--ps-glass-grad);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    border: 1px solid var(--ps-border);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    box-shadow: var(--ps-shadow);
    transition: border-color 0.28s var(--ps-ease), transform 0.28s var(--ps-ease),
                box-shadow 0.28s var(--ps-ease);
}

/* Gradient stroke that draws itself in on hover. */
.ps-kpi-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 4px;
    background: var(--ps-gradient);
    background-size: 220% 100%;
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.42s var(--ps-ease);
}

/* Light sweep across the card on hover. */
.ps-kpi-card::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 38%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(var(--ps-primary-rgb), 0.10), transparent);
    opacity: 0;
    pointer-events: none;
}

.ps-kpi-card:hover {
    border-color: rgba(var(--ps-primary-rgb), 0.42);
    transform: translateY(-6px);
    box-shadow: var(--ps-shadow-lift);
}

.ps-kpi-card:hover::before { transform: scaleX(1); }

.ps-kpi-card:hover::after {
    opacity: 1;
    animation: ps-sheen 0.9s var(--ps-ease);
}

.ps-kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ps-text-secondary);
    margin-bottom: 0.3rem;
}

.ps-kpi-value {
    font-family: 'Poppins', 'Inter', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.15;
    background: var(--ps-gradient);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: ps-drift 11s ease-in-out infinite;
}

.ps-kpi-subtext {
    font-size: 0.75rem;
    color: var(--ps-text-secondary);
    margin-top: 0.3rem;
}

/* Badges */
.ps-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: 0.02em;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    white-space: nowrap;
    transition: transform 0.2s var(--ps-ease), box-shadow 0.2s var(--ps-ease);
}

.ps-badge:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 14px -8px rgba(var(--ps-shadow-rgb), 0.5);
}

.ps-badge-primary { background: rgba(var(--ps-primary-rgb), 0.12); color: var(--ps-primary-ink); border: 1px solid rgba(var(--ps-primary-rgb), 0.30); }
.ps-badge-success { background: rgba(var(--ps-success-rgb), 0.12); color: var(--ps-success-ink); border: 1px solid rgba(var(--ps-success-rgb), 0.30); }
.ps-badge-warning { background: rgba(var(--ps-warning-rgb), 0.14);  color: var(--ps-warning-ink); border: 1px solid rgba(var(--ps-warning-rgb), 0.32); }
.ps-badge-error   { background: rgba(var(--ps-error-rgb), 0.11);  color: var(--ps-error-ink); border: 1px solid rgba(var(--ps-error-rgb), 0.28); }
.ps-badge-info    { background: rgba(var(--ps-info-rgb), 0.12); color: var(--ps-info-ink); border: 1px solid rgba(var(--ps-info-rgb), 0.30); }
.ps-badge-neutral { background: rgba(var(--ps-neutral-rgb), 0.10); color: var(--ps-neutral-ink); border: 1px solid rgba(var(--ps-neutral-rgb), 0.26); }

/* Workflow Stepper */
.ps-stepper {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin: 1.25rem 0;
}

.ps-step-card {
    position: relative;
    overflow: hidden;
    background-color: var(--ps-glass);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    border: 1px solid var(--ps-border);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-decoration: none !important;
    transition: transform 0.26s var(--ps-ease), border-color 0.26s var(--ps-ease),
                box-shadow 0.26s var(--ps-ease), background-color 0.26s var(--ps-ease);
}

.ps-step-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--ps-gradient);
    transform: scaleY(0);
    transform-origin: bottom;
    transition: transform 0.3s var(--ps-ease);
}

.ps-step-card:hover {
    border-color: rgba(var(--ps-primary-rgb), 0.42);
    background-color: var(--ps-bg-elevated);
    transform: translateY(-4px);
    box-shadow: var(--ps-shadow-lift);
}

.ps-step-card:hover::before { transform: scaleY(1); }

.ps-step-number {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--ps-gradient);
    background-size: 200% 100%;
    color: var(--ps-on-accent);
    font-weight: 700;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    animation: ps-drift 10s ease-in-out infinite;
}

.ps-step-info {
    display: flex;
    flex-direction: column;
}

.ps-step-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--ps-text-primary);
}

.ps-step-desc {
    font-size: 0.75rem;
    color: var(--ps-text-secondary);
}

/* Sidebar Styling Adjustments */
[data-testid="stSidebar"] {
    /* Rebind the surface tokens inside the sidebar so every descendant rule
       follows the section's own contrast. Under a light palette these alias
       straight back to the page values and nothing changes; under a dark
       sidebar they flip the whole subtree in one place. */
    --ps-text-primary: var(--ps-side-text);
    --ps-text-secondary: var(--ps-side-text-2);
    --ps-text: var(--ps-side-text);
    --ps-text-2: var(--ps-side-text-2);
    --ps-border: var(--ps-side-border);
    --ps-border-light: var(--ps-side-border);
    --ps-bg-card: var(--ps-side-card);
    --ps-bg-elevated: var(--ps-side-hover);
    /* Keeps every glass surface in here mixing from charcoal, not white.
       The --ps-glass* values have to be restated, not just their input:
       :root already substituted --ps-card-rgb into them, and what inherits
       down is that resolved colour. Rebinding the input alone would leave
       the sidebar painting white glass over its own charcoal. */
    --ps-card-rgb: var(--ps-side-card-rgb);
    --ps-glass: rgba(var(--ps-side-card-rgb), 0.62);
    --ps-glass-strong: rgba(var(--ps-side-card-rgb), 0.78);
    --ps-glass-grad: linear-gradient(180deg, rgba(var(--ps-side-card-rgb), 0.70) 0%,
                                     rgba(var(--ps-side-card-rgb), 0.44) 100%);

    background-color: var(--ps-side-bg);
    border-right: 1px solid var(--ps-side-border);
}

.ps-sidebar-card {
    background: var(--ps-side-card);
    border: 1px solid var(--ps-side-border);
    border-radius: 14px;
    padding: 0.85rem;
    margin-top: 1rem;
    box-shadow: var(--ps-shadow);
}

.ps-sidebar-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ps-text-secondary);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Sidebar navigation gets a sliding gradient marker. */
[data-testid="stSidebarNavLink"] {
    position: relative;
    border-radius: 10px;
    transition: background 0.24s var(--ps-ease), transform 0.24s var(--ps-ease);
}
[data-testid="stSidebarNavLink"]::before {
    content: "";
    position: absolute;
    left: 0;
    top: 12%;
    bottom: 12%;
    width: 3px;
    border-radius: 999px;
    background: var(--ps-gradient);
    transform: scaleY(0);
    transition: transform 0.26s var(--ps-ease);
}
[data-testid="stSidebarNavLink"]:hover {
    background: var(--ps-bg-elevated);
    transform: translateX(4px);
}
[data-testid="stSidebarNavLink"]:hover::before,
[data-testid="stSidebarNavLink"][aria-current="page"]::before { transform: scaleY(1); }
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(90deg, rgba(var(--ps-side-accent-rgb), 0.22),
                                rgba(var(--ps-side-accent-rgb), 0.06));
}

/* ------------------------------------------------------------------ *
 * Native widget polish
 * ------------------------------------------------------------------ */

/* Buttons: lift, sheen, and a gradient fill on primary. */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    position: relative;
    overflow: hidden;
    font-weight: 600;
    transition: transform 0.22s var(--ps-ease), box-shadow 0.22s var(--ps-ease),
                filter 0.22s var(--ps-ease);
}
.stButton button::after,
.stDownloadButton button::after,
.stFormSubmitButton button::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 42%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.55), transparent);
    opacity: 0;
    pointer-events: none;
}
.stButton button:hover:not(:disabled)::after,
.stDownloadButton button:hover:not(:disabled)::after,
.stFormSubmitButton button:hover:not(:disabled)::after {
    opacity: 1;
    animation: ps-sheen 0.85s var(--ps-ease);
}
.stButton button:hover:not(:disabled),
.stDownloadButton button:hover:not(:disabled),
.stFormSubmitButton button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: var(--ps-shadow);
}
.stButton button:active:not(:disabled),
.stDownloadButton button:active:not(:disabled),
.stFormSubmitButton button:active:not(:disabled) {
    transform: translateY(0) scale(0.985);
}

[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
    background: var(--ps-gradient) !important;
    background-size: 200% 100% !important;
    border: 0 !important;
    color: var(--ps-on-accent) !important;
    box-shadow: 0 8px 22px -10px rgba(var(--ps-primary-rgb), 0.75);
    animation: ps-drift 10s ease-in-out infinite;
}
[data-testid="stBaseButton-primary"]:hover:not(:disabled),
[data-testid="stBaseButton-primaryFormSubmit"]:hover:not(:disabled) {
    filter: saturate(1.15) brightness(1.04);
}
[data-testid="stBaseButton-primary"]:disabled,
[data-testid="stBaseButton-primaryFormSubmit"]:disabled {
    background: var(--ps-disabled-bg) !important;
    animation: none;
    box-shadow: none;
    color: var(--ps-disabled-ink) !important;
}
[data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
    border-color: var(--ps-primary) !important;
    color: var(--ps-primary) !important;
}
.stDownloadButton button {
    border-color: rgba(var(--ps-primary-rgb), 0.35) !important;
}
.stDownloadButton button:hover:not(:disabled) {
    background: var(--ps-bg-elevated) !important;
    border-color: var(--ps-primary) !important;
}

/* Streamlit dims elements it considers stale while a rerun is in flight.
   The live run panel and the sidebar status both refresh every 2s, so there
   the dimming read as a constant flicker rather than as feedback. Keep them
   at full strength; everything else still dims, just less harshly. */
[class*="st-key-ps-live-run"] [data-stale="true"],
[class*="st-key-ps-live-run"][data-stale="true"],
[data-testid="stSidebar"] [data-stale="true"] {
    opacity: 1 !important;
    transition: none !important;
}
[data-stale="true"] { opacity: 0.72; }

/* Dismiss controls stay quiet until hovered, then turn red. */
[class*="st-key-dismiss-failure"] button {
    color: var(--ps-text-secondary) !important;
    opacity: 0.7;
    font-size: 0.8rem;
    transition: opacity 0.2s var(--ps-ease), color 0.2s var(--ps-ease),
                transform 0.2s var(--ps-ease);
}
[class*="st-key-dismiss-failure"] button:hover {
    opacity: 1;
    color: var(--ps-error) !important;
    transform: translateY(-1px);
}
[class*="st-key-dismiss-failure"] button::after { display: none; }

/* Page links read as tiles that slide on hover. */
.stPageLink a {
    position: relative;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1rem 0.7rem 1.15rem;
    border-radius: 14px;
    font-weight: 600;
    border: 1px solid var(--ps-border);
    background: var(--ps-glass);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    overflow: hidden;
    transition: transform 0.26s var(--ps-ease), box-shadow 0.26s var(--ps-ease),
                border-color 0.26s var(--ps-ease), padding-left 0.26s var(--ps-ease);
}
.stPageLink a::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--ps-gradient);
    transform: scaleY(0);
    transform-origin: bottom;
    transition: transform 0.3s var(--ps-ease);
}
.stPageLink a:hover {
    transform: translateX(5px);
    padding-left: 1.35rem;
    border-color: rgba(var(--ps-primary-rgb), 0.42);
    box-shadow: var(--ps-shadow);
}
.stPageLink a:hover::before { transform: scaleY(1); }

/* Sidebar page links stay compact. */
[data-testid="stSidebar"] .stPageLink a {
    padding: 0.45rem 0.7rem;
    font-size: 0.85rem;
}

/* Metrics behave like cards. Streamlit only pads st.metric when its own
   border=True chrome is on; this surface is painted from CSS instead, so the
   card must bring its own padding or the label lands flush on the top edge.
   Padding and border match .ps-kpi-card so both card families read as one. */
.stMetric {
    border-radius: 16px;
    border: 1px solid var(--ps-border);
    padding: 1.1rem 1.3rem;
    background: var(--ps-glass);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    transition: transform 0.28s var(--ps-ease), box-shadow 0.28s var(--ps-ease),
                border-color 0.28s var(--ps-ease), background-color 0.28s var(--ps-ease);
}
.stMetric:hover { background-color: var(--ps-glass-strong); }
.stMetric:hover {
    transform: translateY(-5px);
    border-color: rgba(var(--ps-primary-rgb), 0.42);
    box-shadow: var(--ps-shadow-lift);
}
/* Streamlit sizes this as display type (~2.25rem) against a ~0.9rem label.
   That ratio suits a lone headline number, but these cards mostly carry
   short strings - "RUNNING", "Preparing offers" - where it reads as
   shouting and wraps mid-phrase. 1.5rem holds a clear step over the label
   while staying in proportion with the card. */
.stMetric [data-testid="stMetricValue"] {
    font-size: 1.5rem;
    line-height: 1.3;
    letter-spacing: -0.01em;
    overflow-wrap: break-word;
    font-family: 'Poppins', 'Inter', sans-serif;
    font-weight: 700;
    background: var(--ps-gradient);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: ps-drift 11s ease-in-out infinite;
}
.stMetric [data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--ps-text-secondary);
    /* Streamlit pins this label's margin to 0; restate the same label-to-value
       step .ps-kpi-label uses so the two card families rhyme. */
    margin-bottom: 0.3rem;
}

/* Cache spinner ("Running load_insights(...)"). Streamlit masks the
   re-running block with an opaque gradient of config.toml's boot-time
   backgroundColor, which lands as a white slab on the ambient wash and
   never follows the runtime palette toggle. Restate the wait state as a
   glass card; the blur keeps the text legible over whatever sits behind.
   Scoped to the cache variant - plain st.spinner draws no surface. */
.stSpinner.stCacheSpinner {
    background: var(--ps-glass);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    border: 1px solid var(--ps-border);
    border-radius: 14px;
    padding: 0.6rem 0.9rem;
}
/* The arc inherits config.toml's frozen text colour; bind it to the
   palette so the motion carries the brand instead of a grey ghost. */
.stSpinner [data-testid="stSpinnerIcon"] {
    border-color: var(--ps-border);
    border-top-color: var(--ps-primary);
}

/* Progress: animated stripes over the sunset ramp.
   BaseWeb drives the fill with translateX, not width. */
[data-testid="stProgressBarTrack"] > div {
    background-image:
        repeating-linear-gradient(115deg, rgba(255, 255, 255, 0.26) 0 10px, transparent 10px 21px),
        var(--ps-gradient);
    background-size: 42px 100%, 200% 100%;
    transition: transform 0.4s var(--ps-ease);
    animation: ps-stripes 0.9s linear infinite, ps-drift 8s ease-in-out infinite;
}

/* File uploader dropzone. */
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed rgba(var(--ps-primary-rgb), 0.42) !important;
    background: linear-gradient(135deg, rgba(var(--ps-primary-rgb), 0.05), rgba(var(--ps-accent-rgb), 0.05)) !important;
    transition: border-color 0.26s var(--ps-ease), background 0.26s var(--ps-ease),
                transform 0.26s var(--ps-ease);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--ps-primary) !important;
    transform: translateY(-2px);
    background: linear-gradient(135deg, rgba(var(--ps-primary-rgb), 0.10), rgba(var(--ps-accent-rgb), 0.10)) !important;
}

/* Alerts, expanders, dataframes. */
[data-testid="stAlert"] {
    border-radius: 14px;
    transition: transform 0.24s var(--ps-ease), box-shadow 0.24s var(--ps-ease);
}
[data-testid="stAlert"]:hover {
    transform: translateX(3px);
    box-shadow: var(--ps-shadow);
}
[data-testid="stExpander"] details {
    border-radius: 14px;
    overflow: hidden;
    background: var(--ps-glass);
    backdrop-filter: var(--ps-blur);
    -webkit-backdrop-filter: var(--ps-blur);
    transition: box-shadow 0.26s var(--ps-ease), border-color 0.26s var(--ps-ease);
}
[data-testid="stExpander"] details:hover {
    border-color: rgba(var(--ps-primary-rgb), 0.38);
    box-shadow: var(--ps-shadow);
}
[data-testid="stExpander"] summary:hover { color: var(--ps-primary); }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    animation: ps-rise 0.34s var(--ps-ease) both;
}
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    transition: box-shadow 0.26s var(--ps-ease);
}
[data-testid="stDataFrame"]:hover { box-shadow: var(--ps-shadow); }

/* Inputs pick up a violet focus ring. */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    transition: border-color 0.22s var(--ps-ease), box-shadow 0.22s var(--ps-ease);
}
.stSelectbox [data-baseweb="select"] > div:hover,
.stTextInput input:hover,
.stTextArea textarea:hover,
.stNumberInput input:hover {
    border-color: rgba(var(--ps-primary-rgb), 0.55) !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    box-shadow: 0 0 0 3px rgba(var(--ps-primary-rgb), 0.18) !important;
}

/* Respect reduced-motion preferences. */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
    }
}
</style>
"""

def chart_accent() -> str:
    """Bar colour for single-series charts.

    ``st.bar_chart`` otherwise takes its colour from ``config.toml``, which is
    fixed at startup - so a chart would stay on the booting palette's accent
    while the rest of the page followed the toggle.
    """
    return selected_palette()["primary"]


def build_theme_css(palette: dict[str, str] | None = None) -> str:
    """Render the stylesheet with one palette's tokens in its ``:root`` block."""
    return CUSTOM_CSS.replace(
        "__PALETTE__", root_variables(palette or selected_palette())
    )


def inject_theme() -> None:
    """Inject custom PromoStrater theme CSS into Streamlit page."""
    st.markdown(build_theme_css(), unsafe_allow_html=True)
