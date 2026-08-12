"""Al Kabeer SKUs — master product browser with live offer and competitor mappings.

Read-only page. The Product Master is the permanent left-hand anchor; the
mapped offers and competitor offers come from the latest completed pipeline run
and refresh automatically whenever a new run finishes. No pipeline code is
invoked here — all data is read from persisted CSV artifacts.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.bootstrap import load_dashboard_context
from dashboard.components.formatters import (
    format_enum_label,
    get_status_color,
    render_badge_html,
)
from dashboard.components.sidebar_status import render_sidebar_status
from dashboard.theme import inject_theme

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Al Kabeer SKUs",
    page_icon="🏷️",
    layout="wide",
)

inject_theme()
config, store, _, _, _, _, jobs = load_dashboard_context()
render_sidebar_status(store, jobs)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MASTER_COLS = {
    "Itemcode": "SKU",
    "Itemname": "Name",
    "Core-Non Core": "Category",
    "Item-Cat-2": "Type",
    "Item-Cat-3": "Segment",
    "Item-Cat-4": "Sub-category",
    "Unit-Per Ctn": "Units/Ctn",
    "Item-Spec": "Pack Spec",
}

#: Height of the master table and of each detail list, in pixels. Both are
#: capped so the page length stays constant whether a SKU has 2 mapped offers
#: or 237 - the lists scroll internally instead of stretching the page.
TABLE_HEIGHT = 420
LIST_HEIGHT = 460

_PAGE_CSS = """
<style>
.ak-head {
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}
.ak-list { padding-right: 0.25rem; }
.ak-row {
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--ps-border-light);
}
.ak-row:last-child { border-bottom: 0; }
.ak-name {
    font-weight: 500;
    font-size: 0.88rem;
    color: var(--ps-text-primary);
    overflow-wrap: anywhere;
}
.ak-meta {
    font-size: 0.78rem;
    color: var(--ps-text-secondary);
    margin-top: 0.15rem;
    overflow-wrap: anywhere;
}
</style>
"""

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_product_master(master_path: str) -> pd.DataFrame:
    """Load and normalise the Product Master.

    Cached without expiry: the master catalogue is a fixed list that only
    changes when the file itself is replaced, so re-reading the workbook on a
    timer is pure overhead. Restart the app after editing the file.
    """
    path = Path(master_path)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl")
    # Rename to friendly display names; keep only columns present in the file
    rename = {k: v for k, v in _MASTER_COLS.items() if k in df.columns}
    df = df.rename(columns=rename)
    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].astype(str).str.strip()
    return df


@st.cache_data(ttl=60, show_spinner=False)
def _load_latest_completed_run_id(store_path: str) -> str | None:
    """Return the run_id of the most recent completed run, or None.

    This is the only lookup that needs a TTL: a newly completed run is exactly
    what should invalidate the page. Everything downstream is keyed on the id
    returned here, so a new run swaps the whole view over at once rather than
    mixing artifacts from two runs.
    """
    from sku_mapping.learning.store import LearningStore
    s = LearningStore(store_path)
    runs = s.list_pipeline_runs(completed_only=True, limit=1)
    return str(runs[0]["run_id"]) if runs else None


@st.cache_data(max_entries=3, show_spinner=False)
def _load_run_artifacts(store_path: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (sku_mapping_df, competitor_df, run_meta) for a given run.

    Keyed on ``run_id`` and cached without expiry, because a completed run's
    artifacts are written once and never rewritten. ``max_entries`` bounds the
    memory held when several runs are viewed in one session - each run's
    mapping export is a few megabytes.
    """
    from sku_mapping.learning.store import LearningStore
    s = LearningStore(store_path)
    run = s.get_pipeline_run(run_id)
    if run is None:
        return pd.DataFrame(), pd.DataFrame(), {}

    output_paths: dict = run.get("output_paths", {})
    meta = {
        "run_id": run_id,
        "source_filename": run.get("source_filename", ""),
        "completed_at": run.get("completed_at", ""),
        "status": run.get("status", ""),
        "unique_offer_count": run.get("unique_offer_count", 0),
    }

    sku_df = pd.DataFrame()
    raw_sku = output_paths.get("sku_mapping")
    if raw_sku:
        p = Path(raw_sku)
        if p.is_file() and p.stat().st_size > 0:
            try:
                sku_df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
            except Exception:
                pass

    comp_df = pd.DataFrame()
    raw_comp = output_paths.get("competitor_offers")
    if raw_comp:
        p = Path(raw_comp)
        if p.is_file() and p.stat().st_size > 0:
            try:
                comp_df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
            except Exception:
                pass

    return sku_df, comp_df, meta


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _parse_json_list(value: object) -> list[str]:
    """Safely parse a stringified JSON list to a Python list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    s = str(value).strip()
    if not s or s in ("[]", "nan", "None"):
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x]
    except (json.JSONDecodeError, ValueError):
        pass
    # Single non-JSON value
    return [s]


def _clean(value: object) -> str:
    """Trim a CSV cell to display text, collapsing pandas' null spellings."""
    text = str(value).strip()
    return "" if text in ("nan", "None", "<NA>", "") else text


def _at(values: list[str], index: int) -> str:
    """Value at ``index``, or empty. The competitor columns are parallel JSON
    lists that are not guaranteed to be the same length."""
    return _clean(values[index]) if index < len(values) else ""


def _detail_row(title_html: str, meta_parts: list[str]) -> str:
    """One entry in a detail list.

    ``title_html`` may carry markup (a status badge); everything in
    ``meta_parts`` is escaped by the caller.
    """
    meta = (
        f'<div class="ak-meta">{"  ·  ".join(meta_parts)}</div>'
        if meta_parts
        else ""
    )
    return f'<div class="ak-row">{title_html}{meta}</div>'


def _render_detail_list(rows: list[str]) -> None:
    """Emit a scrollable list of pre-built rows.

    Both the height cap and the single ``st.markdown`` matter. A SKU can carry
    hundreds of mapped offers; rendering one Streamlit element per offer pushed
    the page to several thousand elements and made every rerun crawl, on top of
    the unbounded scroll length. One markdown block inside a fixed-height
    container keeps the cost flat no matter how many offers a SKU has.
    """
    with st.container(height=LIST_HEIGHT, border=False):
        st.markdown(
            '<div class="ak-list">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(_PAGE_CSS, unsafe_allow_html=True)
st.title("🏷️ Al Kabeer SKUs")
st.caption(
    "Permanent product catalogue with live offer mappings and competitor "
    "data from the most recent completed pipeline run. "
    "The master SKU list never changes — only the mapped and competitor "
    "offers update when a new run completes."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
store_path = str(config.learning_store.database_path)
master_path = str(config.data.master_path)

with st.spinner("Loading product master…"):
    master_df = _load_product_master(master_path)

if master_df.empty:
    st.error(
        f"Product Master not found or is empty at: `{master_path}`. "
        "Ensure the file exists and the path in `config/default.yaml` is correct."
    )
    st.stop()

# Latest completed run
with st.spinner("Checking for the latest completed run…"):
    latest_run_id = _load_latest_completed_run_id(store_path)

has_run = latest_run_id is not None
sku_df = pd.DataFrame()
comp_df = pd.DataFrame()
run_meta: dict = {}

if has_run:
    with st.spinner("Loading pipeline run artifacts…"):
        sku_df, comp_df, run_meta = _load_run_artifacts(store_path, latest_run_id)

# ---------------------------------------------------------------------------
# Run info banner
# ---------------------------------------------------------------------------
if has_run:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.caption(f"**Latest Run:** `{run_meta.get('run_id', 'N/A')}`")
        c2.caption(f"**Source File:** {run_meta.get('source_filename', 'N/A')}")
        completed_at = run_meta.get("completed_at", "")
        c3.caption(f"**Completed:** {completed_at[:19].replace('T', ' ') if completed_at else 'N/A'}")
else:
    st.info(
        "No completed pipeline runs found yet. "
        "Run a pipeline from **Upload & Process** to see mapped offers and competitors here.",
        icon="ℹ️",
    )

# ---------------------------------------------------------------------------
# Statistics banner
# ---------------------------------------------------------------------------
total_skus = len(master_df)

# Build lookup: master_sku -> list of mapped offer names
_sku_col_map = "matched_master_sku"
_offer_col_map = "source_offer_name"
_sku_col_comp = "master_sku"
_comp_names_col = "competitor_offer_names"
_comp_status_col = "competitor_status"

# Index the mapping CSV by master SKU
mapped_skus: set[str] = set()
competitor_skus: set[str] = set()

if not sku_df.empty and _sku_col_map in sku_df.columns:
    mapped_skus = set(
        sku_df[sku_df[_sku_col_map].notna()][_sku_col_map]
        .astype(str).str.strip().unique()
    )

if not comp_df.empty and _sku_col_comp in comp_df.columns and _comp_status_col in comp_df.columns:
    competitor_skus = set(
        comp_df[
            (comp_df[_comp_status_col].astype(str).str.upper() == "COMPETITORS_FOUND")
            & comp_df[_sku_col_comp].notna()
        ][_sku_col_comp].astype(str).str.strip().unique()
    )

num_mapped = len(mapped_skus & set(master_df["SKU"].astype(str)))
num_with_competitors = len(competitor_skus & set(master_df["SKU"].astype(str)))

with st.container(border=True):
    st.subheader("Catalogue Overview")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Master SKUs", f"{total_skus:,}")
    s2.metric(
        "SKUs with Mapped Offers",
        f"{num_mapped:,}" if has_run else "—",
        help="SKUs that appear in the latest run's mapping output.",
    )
    s3.metric(
        "SKUs with Competitors",
        f"{num_with_competitors:,}" if has_run else "—",
        help="SKUs that have at least one competitor found in the latest run.",
    )
    s4.metric(
        "Unmapped SKUs",
        f"{total_skus - num_mapped:,}" if has_run else "—",
        help="Master SKUs not matched to any offer in the latest run.",
    )

# ---------------------------------------------------------------------------
# Search & filter controls
# ---------------------------------------------------------------------------
st.subheader("Search & Filter")

fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 2])

with fc1:
    search_query = st.text_input(
        "Search SKUs",
        placeholder="Type SKU code or product name…",
        key="sku_search",
        label_visibility="collapsed",
    )

category_options = ["All Categories"]
if "Category" in master_df.columns:
    category_options += sorted(master_df["Category"].dropna().unique().tolist())

with fc2:
    selected_category = st.selectbox(
        "Category",
        category_options,
        key="sku_cat_filter",
    )

type_options = ["All Types"]
if "Type" in master_df.columns:
    type_options += sorted(master_df["Type"].dropna().unique().tolist())

with fc3:
    selected_type = st.selectbox(
        "Product Type",
        type_options,
        key="sku_type_filter",
    )

with fc4:
    mapping_filter = st.selectbox(
        "Mapping Status",
        ["All", "Mapped", "Unmapped", "Has Competitors"],
        key="sku_mapping_filter",
    )

# ---------------------------------------------------------------------------
# Apply filters to master dataframe
# ---------------------------------------------------------------------------
filtered_df = master_df.copy()

if search_query:
    q = search_query.strip().lower()
    mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    for col in ("SKU", "Name"):
        if col in filtered_df.columns:
            mask |= filtered_df[col].astype(str).str.lower().str.contains(q, na=False)
    filtered_df = filtered_df[mask]

if selected_category != "All Categories" and "Category" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

if selected_type != "All Types" and "Type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Type"] == selected_type]

if mapping_filter == "Mapped" and has_run:
    filtered_df = filtered_df[filtered_df["SKU"].astype(str).isin(mapped_skus)]
elif mapping_filter == "Unmapped" and has_run:
    filtered_df = filtered_df[~filtered_df["SKU"].astype(str).isin(mapped_skus)]
elif mapping_filter == "Has Competitors" and has_run:
    filtered_df = filtered_df[filtered_df["SKU"].astype(str).isin(competitor_skus)]

st.caption(f"Showing **{len(filtered_df):,}** of **{total_skus:,}** master SKUs")

# ---------------------------------------------------------------------------
# Master SKU selector table
# ---------------------------------------------------------------------------
# Pre-build mapping and competitor counts per SKU for the summary table
if has_run and not sku_df.empty and _sku_col_map in sku_df.columns:
    offer_counts = (
        sku_df[sku_df[_sku_col_map].notna()]
        .groupby(sku_df[_sku_col_map].astype(str).str.strip())
        .size()
        .rename("Mapped Offers")
    )
else:
    offer_counts = pd.Series(dtype=int, name="Mapped Offers")

if (
    has_run
    and not comp_df.empty
    # The aggregation below reads both of these; the status filter above only
    # guaranteed the SKU column, so a run missing either would KeyError here.
    and {_sku_col_comp, _comp_status_col, "competitor_count"} <= set(comp_df.columns)
):
    comp_counts = (
        comp_df[comp_df[_comp_status_col].astype(str).str.upper() == "COMPETITORS_FOUND"]
        .groupby(comp_df[_sku_col_comp].astype(str).str.strip())
        .agg(competitor_count=("competitor_count", "max"))
        ["competitor_count"]
        .rename("Competitors")
    )
else:
    comp_counts = pd.Series(dtype=int, name="Competitors")

# Build the display table for the master list
display_cols = [c for c in ("SKU", "Name", "Category", "Type", "Segment", "Sub-category", "Pack Spec") if c in filtered_df.columns]
table_df = filtered_df[display_cols].copy()
table_df["SKU"] = table_df["SKU"].astype(str).str.strip()

if has_run:
    table_df = table_df.join(offer_counts, on="SKU", how="left")
    table_df = table_df.join(comp_counts, on="SKU", how="left")
    table_df["Mapped Offers"] = table_df["Mapped Offers"].fillna(0).astype(int)
    table_df["Competitors"] = table_df["Competitors"].fillna(0).astype(int)

st.subheader("Master SKU List")
st.caption("Select a SKU to see its mapped offers and competitors below.")

# Render the table with row selection
_event = st.dataframe(
    table_df.reset_index(drop=True),
    width="stretch",
    height=TABLE_HEIGHT,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="master_sku_table",
    column_config={
        "SKU": st.column_config.TextColumn("SKU", width="small"),
        "Name": st.column_config.TextColumn("Product Name", width="large"),
        "Category": st.column_config.TextColumn("Category", width="medium"),
        "Type": st.column_config.TextColumn("Type", width="medium"),
        "Mapped Offers": st.column_config.NumberColumn(
            "Mapped Offers", help="Offers mapped to this SKU in the latest run", width="small"
        ) if has_run else None,
        "Competitors": st.column_config.NumberColumn(
            "Competitors", help="Competitors found for this SKU in the latest run", width="small"
        ) if has_run else None,
    },
)

# ---------------------------------------------------------------------------
# Resolve the selected SKU
#
# Two ways in: click a row in the table above, or use the picker below. The
# choice is mirrored into session state because st.dataframe clears its own row
# selection on every rerun — so searching, filtering or opening an expander
# would otherwise make the detail panel disappear mid-investigation.
# ---------------------------------------------------------------------------
_PLACEHOLDER = "— choose a SKU —"
_labels = {
    str(r["SKU"]): f"{r['SKU']} — {r.get('Name', '')}"
    for _, r in table_df.iterrows()
}

selected_rows = (_event.selection.rows if _event and hasattr(_event, "selection") else [])
if selected_rows:
    idx = selected_rows[0]
    if idx < len(table_df):
        clicked = str(table_df.iloc[idx]["SKU"])
        # Safe to assign: the picker widget below is not instantiated until
        # after this line, and Streamlit only forbids writing to a widget key
        # once that widget exists this run.
        st.session_state["ak_sku_picker"] = _labels.get(clicked, _PLACEHOLDER)

options = [_PLACEHOLDER] + [_labels[s] for s in table_df["SKU"].astype(str)]
# The remembered SKU can fall outside the current filter; reset rather than crash.
if st.session_state.get("ak_sku_picker") not in options:
    st.session_state["ak_sku_picker"] = _PLACEHOLDER

st.divider()
choice = st.selectbox(
    "Selected SKU",
    options,
    key="ak_sku_picker",
    help="Click any row in the table above, or search for a SKU directly here.",
)

selected_sku: str | None = None
selected_row_data: dict = {}
if choice != _PLACEHOLDER:
    selected_sku = choice.split(" — ")[0].strip()
    _match = table_df[table_df["SKU"].astype(str) == selected_sku]
    if not _match.empty:
        selected_row_data = _match.iloc[0].to_dict()

if selected_sku is None:
    st.info(
        "Click a row in the table above — or pick a SKU from the dropdown — to "
        "see the offers mapped to it.",
        icon="👆",
    )
    st.stop()

# Build the three-column detail view
st.subheader(f"Detail: {selected_row_data.get('Name', selected_sku)}")

col_master, col_offers, col_competitors = st.columns([1, 1, 1], gap="medium")

# ── Column 1: Al Kabeer SKU ──────────────────────────────────────────────
with col_master:
    with st.container(border=True):
        st.markdown(
            '<div class="ak-head" style="color:var(--ps-primary);">'
            '📦 Al Kabeer SKU</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**SKU Code:** `{selected_sku}`")
        st.markdown(f"**Product Name:** {selected_row_data.get('Name', '—')}")

        detail_fields = [
            ("Category", "Category"),
            ("Type", "Type"),
            ("Segment", "Segment"),
            ("Sub-category", "Sub-category"),
            ("Pack Spec", "Pack Spec"),
            ("Units/Ctn", "Units/Ctn"),
        ]
        for display_name, col_name in detail_fields:
            val = selected_row_data.get(col_name)
            if val and str(val) not in ("nan", "None", ""):
                st.caption(f"**{display_name}:** {val}")

# ── Column 2: Mapped Offer(s) ────────────────────────────────────────────
with col_offers:
    with st.container(border=True):
        st.markdown(
            '<div class="ak-head" style="color:var(--ps-success);">'
            '🏷️ Mapped Offer(s)</div>',
            unsafe_allow_html=True,
        )

        if not has_run:
            st.info("No completed run available.", icon="ℹ️")
        elif sku_df.empty or _sku_col_map not in sku_df.columns:
            st.info("No mapping data available for this run.", icon="ℹ️")
        else:
            sku_offers = sku_df[
                sku_df[_sku_col_map].astype(str).str.strip() == selected_sku
            ].copy()

            if sku_offers.empty:
                st.info("No offers mapped to this SKU in the latest run.", icon="ℹ️")
            else:
                st.caption(f"**{len(sku_offers):,}** offer row(s) mapped")

                rows: list[str] = []
                for _, offer_row in sku_offers.iterrows():
                    offer_name = _clean(offer_row.get(_offer_col_map))
                    offer_id = _clean(offer_row.get("source_offer_id"))
                    status_raw = _clean(offer_row.get("mapping_status"))
                    score = offer_row.get("mapping_score")
                    retailer = _clean(offer_row.get("source_retailer"))
                    flyer = _clean(offer_row.get("source_flyer"))
                    price = offer_row.get("source_offer_price")

                    # Offer names come straight from a supplier CSV, so they are
                    # escaped before going anywhere near unsafe_allow_html.
                    name_display = escape(offer_name or offer_id or "—")
                    badge_html = render_badge_html(
                        format_enum_label(status_raw),
                        get_status_color(
                            status_raw.replace("AUTO_ACCEPTED", "AUTO_ACCEPT")
                        ),
                    ) if status_raw else ""

                    meta: list[str] = []
                    try:
                        if score is not None and not pd.isna(score):
                            meta.append(f"Score: {float(score):.1%}")
                    except (ValueError, TypeError):
                        pass
                    if retailer:
                        meta.append(f"📍 {escape(retailer)}")
                    if flyer:
                        meta.append(f"📄 {escape(flyer)}")
                    try:
                        if price is not None and not pd.isna(price):
                            meta.append(f"💰 {float(price):.2f}")
                    except (ValueError, TypeError):
                        pass

                    rows.append(
                        _detail_row(
                            f'<div class="ak-name">{name_display}</div>{badge_html}',
                            meta,
                        )
                    )

                _render_detail_list(rows)

# ── Column 3: Competitor Offer(s) ───────────────────────────────────────
with col_competitors:
    with st.container(border=True):
        st.markdown(
            '<div class="ak-head" style="color:var(--ps-warning);">'
            '🔍 Competitor Offer(s)</div>',
            unsafe_allow_html=True,
        )

        if not has_run:
            st.info("No completed run available.", icon="ℹ️")
        elif comp_df.empty or _sku_col_comp not in comp_df.columns:
            st.info("No competitor data available for this run.", icon="ℹ️")
        else:
            sku_comps = comp_df[
                comp_df[_sku_col_comp].astype(str).str.strip() == selected_sku
            ].copy()

            if sku_comps.empty:
                st.info("No competitor data found for this SKU.", icon="ℹ️")
            else:
                # One row per master SKU in the wide format
                first_row = sku_comps.iloc[0]
                comp_status = str(first_row.get(_comp_status_col, "")).upper()

                if comp_status != "COMPETITORS_FOUND":
                    reason = str(first_row.get("competitor_reason", ""))
                    st.info(
                        reason if reason not in ("nan", "") else "No competitors found for this SKU.",
                        icon="ℹ️",
                    )
                else:
                    comp_names = _parse_json_list(first_row.get("competitor_offer_names"))
                    comp_brands = _parse_json_list(first_row.get("competitor_brand_names"))
                    comp_retailers = _parse_json_list(first_row.get("competitor_retailers"))
                    comp_prices = _parse_json_list(first_row.get("competitor_offer_prices"))
                    comp_pack_sizes = _parse_json_list(first_row.get("competitor_pack_sizes"))
                    comp_flyers = _parse_json_list(first_row.get("competitor_flyers"))
                    comp_count = len(comp_names) or int(first_row.get("competitor_count", 0) or 0)

                    st.caption(f"**{comp_count}** competitor offer(s) found")

                    comp_rows: list[str] = []
                    for i, name in enumerate(comp_names):
                        meta = []
                        for icon, value in (
                            ("🏢", _at(comp_brands, i)),
                            ("📍", _at(comp_retailers, i)),
                            ("💰", _at(comp_prices, i)),
                            ("📦", _at(comp_pack_sizes, i)),
                            ("📄", _at(comp_flyers, i)),
                        ):
                            if value:
                                meta.append(f"{icon} {escape(value)}")

                        comp_rows.append(
                            _detail_row(
                                f'<div class="ak-name">'
                                f'{escape(_clean(name) or "—")}</div>',
                                meta,
                            )
                        )

                    _render_detail_list(comp_rows)

# ---------------------------------------------------------------------------
# Expander: raw offer rows for the selected SKU
# ---------------------------------------------------------------------------
if selected_sku and has_run and not sku_df.empty and _sku_col_map in sku_df.columns:
    sku_offers_raw = sku_df[
        sku_df[_sku_col_map].astype(str).str.strip() == selected_sku
    ].copy()

    if not sku_offers_raw.empty:
        with st.expander(f"Raw mapping rows for {selected_sku} ({len(sku_offers_raw)} rows)", expanded=False):
            display_offer_cols = [
                c for c in (
                    "source_offer_id", "source_offer_name", "source_product",
                    "source_brand", "source_variant", "source_pack_size",
                    "source_retailer", "source_flyer",
                    "source_offer_price", "source_regular_price",
                    "matched_master_sku", "mapping_score", "mapping_status",
                    "mapping_reason", "requires_human_review",
                )
                if c in sku_offers_raw.columns
            ]
            st.dataframe(
                sku_offers_raw[display_offer_cols].reset_index(drop=True),
                width="stretch",
                hide_index=True,
            )

if selected_sku and has_run and not comp_df.empty and _sku_col_comp in comp_df.columns:
    comp_raw = comp_df[
        comp_df[_sku_col_comp].astype(str).str.strip() == selected_sku
    ].copy()

    if not comp_raw.empty:
        with st.expander(f"Raw competitor data for {selected_sku}", expanded=False):
            display_comp_cols = [
                c for c in (
                    "master_sku", "master_name",
                    "competitor_count", "competitor_brand_names",
                    "competitor_offer_names", "competitor_offer_prices",
                    "competitor_pack_sizes", "competitor_retailers",
                    "competitor_flyers", "competitor_status", "competitor_reason",
                )
                if c in comp_raw.columns
            ]
            st.dataframe(
                comp_raw[display_comp_cols].reset_index(drop=True),
                width="stretch",
                hide_index=True,
            )
