"""Generate training offers for SKUs that have no human-reviewed example.

138 of the 237 catalogue SKUs appear in no human review, so the model has never
seen a single example of them and cannot rank them well. This fills that gap.

WHAT MAKES THIS SAFE

The offer text is derived FROM the SKU's own name and pack spec, so the label is
correct by construction - no judgement about which SKU an offer means is being
invented. That is the opposite of the auto-labelling that produced the ~34%
wrong labels, where a real offer was guessed at.

The variation applied is the noise really present in the dump: brand spelling,
unit format, casing, word order, and typos observed in actual flyer text.

WHAT THIS CANNOT DO

It cannot teach the model how customers phrase things in ways unrelated to the
SKU name, and it cannot invent the commercial knowledge behind a mapping. Treat
it as coverage of unseen SKUs, nothing more, and keep it out of measurement.

Usage:
    .venv\\Scripts\\python.exe scripts\\synthesize_uncovered_skus.py
    .venv\\Scripts\\python.exe scripts\\synthesize_uncovered_skus.py --per-sku 3
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "gold_data_PERFECT")

# Brand spellings and typos taken from real offer texts in the dump.
BRANDS = ["Al Kabeer", "AL KABEER", "Alkabeer", "ALKABEER", "Al-Kabeer",
          "Al Kabeer's", "Alkabheer", "AL KABBER", "Al kbeer", "ALKABEEER"]
TYPOS = {"chicken": ["Chicken", "CHICKEN", "Chckn", "Chiken", "Chkn"],
         "vegetable": ["Vegetable", "VEGETABLE", "Veg", "Vegetables"],
         "cheese": ["Cheese", "CHEESE", "Chees"],
         "samosa": ["Samosa", "SAMOSA", "Sambosa"],
         "paratha": ["Paratha", "PARATHA", "Parata"],
         "fillet": ["Fillet", "FILLET", "Filet"],
         "prawns": ["Prawns", "PRAWNS", "Prawn"],
         "shrimps": ["Shrimps", "SHRIMPS", "Shrimp"]}


def norm(s: object) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def covered_skus() -> set[str]:
    """SKU codes that already appear in a human review."""
    out: set[str] = set()
    for path in glob.glob(os.path.join(GOLD, "**", "*.xlsx"), recursive=True):
        if "~$" in path or "sku_reference" in path or "synthesized" in path:
            continue
        try:
            book = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in book.sheet_names:
            d = book.parse(sheet)
            for c in d.columns:
                if re.fullmatch(r"SKU\d+_code", str(c)):
                    out |= {str(v).strip() for v in d[c].dropna()
                            if str(v).strip()}
    return out


def weight_phrases(spec: str, rng) -> list[str]:
    """Render the pack spec the way a flyer would write it."""
    t = str(spec).lower().replace(",", "")
    hit = re.search(r"(\d+(?:\.\d+)?)\s*(kgs?|gms?|g|grms?|grams?)\b", t)
    if not hit:
        return [""]
    val, unit = float(hit.group(1)), hit.group(2)
    grams = val * 1000 if unit.startswith("kg") else val
    forms = []
    if grams >= 1000 and grams % 1000 == 0:
        kg = int(grams // 1000)
        forms += [f"{kg} kg", f"{kg}kg", f"{int(grams)} gm", f"{int(grams)}gm"]
    else:
        g = int(grams) if grams == int(grams) else grams
        forms += [f"{g} gm", f"{g}gm", f"{g} g", f"{g} GM", f"{g} grams"]
    return forms


def vary(name: str, rng) -> str:
    """Apply the spelling noise seen in real flyer text."""
    words = str(name).split()
    out = []
    for w in words:
        key = re.sub(r"[^a-z]", "", w.lower())
        if key in TYPOS and rng.random() < 0.45:
            out.append(str(rng.choice(TYPOS[key])))
        else:
            out.append(w if rng.random() < 0.5 else w.title())
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sku", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(
        GOLD, "synthesized", "synthetic_uncovered_skus.xlsx"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    m = pd.read_excel(os.path.join(ROOT, "Product_Master.xlsx"))
    m["Itemcode"] = (m["Itemcode"].fillna("").astype(str)
                     .str.replace(r"\.0$", "", regex=True).str.strip())
    m = m[m.Itemcode != ""].drop_duplicates("Itemcode").reset_index(drop=True)
    for c in ("Itemname", "Item-Spec"):
        m[c] = m[c].fillna("").astype(str).str.strip()

    have = covered_skus()
    missing = m[~m.Itemcode.isin(have)]
    print(f"catalogue SKUs        : {len(m)}")
    print(f"covered by human review: {len(have & set(m.Itemcode))}")
    print(f"UNCOVERED             : {len(missing)}   <- generating for these")

    rows = []
    for _, r in missing.iterrows():
        name, spec, code = r["Itemname"], r["Item-Spec"], r["Itemcode"]
        # drop a size already embedded in the name so it is not stated twice
        base = re.sub(r"\b\d+(?:\.\d+)?\s*(?:gms?|g|kgs?|grms?)\b", "", name,
                      flags=re.IGNORECASE)
        base = re.sub(r"\s+", " ", base).strip(" -,")
        sizes = weight_phrases(spec, rng)
        seen_local = set()
        for _ in range(args.per_sku):
            text = f"{rng.choice(BRANDS)} {vary(base, rng)}"
            size = str(rng.choice(sizes))
            if size:
                text = f"{text} {size}"
            text = re.sub(r"\s+", " ", text).strip()
            if norm(text) in seen_local:
                continue
            seen_local.add(norm(text))
            rows.append({"Dump_Offer": text, "skus": code, "source": "synthetic",
                         "matched_gold_offer": f"{name} [{spec}]",
                         "text_score": 100.0, "occurrences": 1,
                         "status": "SYNTHETIC_FROM_SKU"})

    syn = pd.DataFrame(rows)

    # never collide with a real offer text already carrying a human decision
    real = set()
    for p in glob.glob(os.path.join(GOLD, "**", "*.xlsx"), recursive=True):
        if "~$" in p or "sku_reference" in p:
            continue
        try:
            bk = pd.ExcelFile(p)
        except Exception:
            continue
        for sh in bk.sheet_names:
            d = bk.parse(sh)
            col = next((c for c in d.columns
                        if str(c).lower() in ("dump_offer", "offer_text")), None)
            if col is not None:
                real |= set(d[col].dropna().map(norm))
    before = len(syn)
    syn = syn[~syn.Dump_Offer.map(norm).isin(real)].reset_index(drop=True)
    print(f"generated {before} rows, {before - len(syn)} dropped as duplicates "
          f"of real offers")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        syn.to_excel(xl, sheet_name="labels", index=False)
        ws = xl.sheets["labels"]
        for c, w in {"A": 56, "B": 14, "C": 14, "D": 48}.items():
            ws.column_dimensions[c].width = w

    print(f"\nsynthetic rows: {len(syn)} covering {syn.skus.nunique()} SKUs")
    print("\nsamples:")
    print(syn.sample(min(10, len(syn)), random_state=1)[
        ["Dump_Offer", "skus", "matched_gold_offer"]].to_string(index=False))
    print(f"\nwrote {args.out}")
    print("\nNOT measurement data - source=synthetic. Train only.")


if __name__ == "__main__":
    main()
