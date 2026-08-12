"""Generate the next batch of offers to review, in the sheet format you fill in.

Excludes every offer already decided or propagated, so batches never overlap.

Sampling mode matters more than batch size:

    random     the long tail - messy, low-frequency, compound offers. This is
               what moved held-out accuracy (+10 points for 70 rows). Use this.
    frequency  the most-repeated offers. Each row covers more of the dump, but
               they are the tidy easy ones and the model already handles them.

Fill Master_SKU1_Description .. SKU4, copying from gold_data_PERFECT/
sku_reference.xlsx. Write NONE if no SKU fits. An offer may need several SKUs.

Usage:
    .venv\\Scripts\\python.exe scripts\\make_next_review_batch.py
    .venv\\Scripts\\python.exe scripts\\make_next_review_batch.py --n 70 --mode random
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "gold_data_PERFECT")


def norm(s: object) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def already_reviewed() -> set[str]:
    """Every offer text a human has decided, or that inherited a decision."""
    seen: set[str] = set()
    for path in glob.glob(os.path.join(GOLD, "**", "*.xlsx"), recursive=True):
        if "~$" in path or "sku_reference" in path:
            continue
        try:
            book = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in book.sheet_names:
            d = book.parse(sheet)
            col = next((c for c in d.columns
                        if str(c).lower() in ("dump_offer", "offer_text")), None)
            if col is not None:
                seen |= set(d[col].dropna().map(norm))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--mode", choices=("random", "frequency"), default="random")
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a different sample each run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pool_path = os.path.join(ROOT, "audit_output", "alkabeer_offers_to_review.xlsx")
    if not os.path.isfile(pool_path):
        raise SystemExit(f"offer pool not found: {pool_path}")
    o = pd.read_excel(pool_path)
    o["_n"] = o["Dump_Offer"].map(norm)

    seen = already_reviewed()
    rest = o[~o["_n"].isin(seen)].copy()
    print(f"offer pool          : {len(o):,} texts ({int(o.occurrences.sum()):,} rows)")
    print(f"already reviewed    : {len(o) - len(rest):,} texts")
    print(f"available to sample : {len(rest):,} texts "
          f"({int(rest.occurrences.sum()):,} rows)")
    if rest.empty:
        raise SystemExit("nothing left to review")

    n = min(args.n, len(rest))
    if args.mode == "frequency":
        samp = rest.nlargest(n, "occurrences")
    else:
        samp = rest.sample(n=n, random_state=args.seed)
    samp = samp.sort_values("occurrences", ascending=False)

    sheet = pd.DataFrame({
        "Dump_Offer": samp["Dump_Offer"],
        "Master_SKU1_Description": "", "Master_SKU2_Description": "",
        "Master_SKU3_Description": "", "Master_SKU4_Description": "",
        "occurrences": samp["occurrences"],
        "pipeline_guess": samp.get("current_sku", pd.Series(dtype=str)).fillna(""),
    })

    out = args.out or os.path.join(
        GOLD, f"Human_Review_{n}Rows_{args.mode.upper()}_NEW.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        sheet.to_excel(xl, sheet_name="Sheet1", index=False)
        ws = xl.sheets["Sheet1"]
        for col, w in {"A": 62, "B": 46, "C": 46, "D": 46,
                       "E": 46, "F": 12, "G": 16}.items():
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "B2"

    compound = int(samp["Dump_Offer"].str.contains(r"[/+]", regex=True).sum())
    print(f"\nsampled {n} offers ({args.mode})")
    print(f"  median occurrences : {samp.occurrences.median():.0f}")
    print(f"  multi-product looking: {compound} of {n}")
    print(f"\nwrote {out}")
    print("\nnext: fill the SKU description columns, then")
    print("  .venv\\Scripts\\python.exe scripts\\resolve_descriptions.py "
          f'--sheet "{os.path.relpath(out, ROOT)}" --tab Sheet1')


if __name__ == "__main__":
    main()
