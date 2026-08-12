"""Propagate human SKU decisions to near-identical offer texts.

A reviewer who decides "Al Kabeer Chicken Nuggets 750 gm -> CKNHSD" has also
decided it for "750gm", "750 GM" and "Chicken Nuggets 750 gm". This script
carries that one judgement across every offer text that is unambiguously the
same, and leaves everything else alone.

Guards, because text similarity alone is dangerous here:

  SIZE GUARD   "GREEN PEAS 400 gm" and "GREEN PEAS 900 gm" are >95% similar and
               are DIFFERENT SKUs. Any weight/count mismatch blocks propagation
               outright, whatever the text score says. This is the same trap
               that made the earlier audit blind to size errors.

  VARIANT GUARD  spicy / non-spicy / plain must agree.

  CONFLICT GUARD  if a target text is close to two gold offers that resolved to
               different SKUs, it is left for a human.

Output is never merged into the gold set: propagated rows are clearly marked
`source=propagated` so measurement can always be restricted to human rows.

Usage:
    .venv\\Scripts\\python.exe scripts\\propagate_gold_labels.py
    .venv\\Scripts\\python.exe scripts\\propagate_gold_labels.py --threshold 92
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE_TOL = 0.02          # weights must agree within 2% to count as the same pack
VARIANTS = ({"spicy"}, {"nonspicy"}, {"plain"})


def norm(s: object) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def weights(text: str) -> frozenset[float]:
    """Every weight in the text, in grams."""
    t = str(text).lower().replace(",", "")
    out = set()
    for v, u in re.findall(r"(\d+(?:\.\d+)?)\s*(kgs?|kilograms?|gms?|g|grm?s?|grams?)\b", t):
        f = float(v)
        out.add(f * 1000 if u.startswith(("kg", "kilo")) else f)
    return frozenset(out)


def counts(text: str) -> frozenset[int]:
    """Bare integers - pack counts like '24 burgers', '8 pcs'."""
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(kgs?|kilograms?|gms?|g|grm?s?|grams?)\b", " ",
               str(text).lower())
    return frozenset(int(x) for x in re.findall(r"\b(\d{1,3})\b", t))


def variant(text: str) -> frozenset[str]:
    t = str(text).lower()
    t = re.sub(r"\b(?:non|not|no)[\s\-]*spicy\b", "nonspicy", t)
    t = re.sub(r"\bhot\s*[n&']*\s*spicy\b", "spicy", t)
    found = set()
    for grp in VARIANTS:
        for w in grp:
            if re.search(rf"\b{w}\b", t):
                found.add(w)
    return frozenset(found)


def sizes_agree(a: str, b: str) -> bool:
    wa, wb = weights(a), weights(b)
    if wa and wb:
        for x in wa:
            if not any(abs(x - y) / max(x, y) <= SIZE_TOL for y in wb):
                return False
        for y in wb:
            if not any(abs(x - y) / max(x, y) <= SIZE_TOL for x in wa):
                return False
    elif wa or wb:
        return False                      # one states a weight, the other does not
    ca, cb = counts(a), counts(b)
    if ca != cb:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=95.0)
    ap.add_argument("--gold", default=os.path.join(ROOT, "audit_output",
                                                   "review_resolved.xlsx"))
    ap.add_argument("--offers", default=os.path.join(ROOT, "audit_output",
                                                     "alkabeer_offers_to_review.xlsx"))
    ap.add_argument("--out", default=os.path.join(ROOT, "audit_output",
                                                  "propagated_labels.xlsx"))
    args = ap.parse_args()

    g = pd.read_excel(args.gold, sheet_name="resolved")
    oc = next(c for c in g.columns if c.lower() in ("dump_offer", "offer_text"))
    code_cols = [c for c in g.columns if re.fullmatch(r"SKU\d+_code", c)]
    gold = []
    for _, r in g.iterrows():
        skus = tuple(sorted({str(r[c]).strip() for c in code_cols
                             if pd.notna(r[c]) and str(r[c]).strip()}))
        if skus:
            gold.append({"offer": str(r[oc]), "n": norm(r[oc]), "skus": skus})
    gdf = pd.DataFrame(gold)
    print(f"human-decided offers: {len(gdf)}")

    o = pd.read_excel(args.offers)
    o["n"] = o["Dump_Offer"].map(norm)
    gold_norm = set(gdf.n)
    total_rows = int(o["occurrences"].sum())
    print(f"offer texts in dump : {len(o):,}  ({total_rows:,} rows)")

    sim = process.cdist(o["n"].tolist(), gdf["n"].tolist(),
                        scorer=fuzz.ratio, workers=-1)

    recs, conflicts = [], 0
    for i, row in o.iterrows():
        text, nrm = row["Dump_Offer"], row["n"]
        if nrm in gold_norm:
            hit = gdf[gdf.n == nrm].iloc[0]
            recs.append({"Dump_Offer": text, "occurrences": row["occurrences"],
                         "skus": "|".join(hit["skus"]), "source": "human",
                         "matched_gold_offer": hit["offer"], "text_score": 100.0,
                         "status": "HUMAN"})
            continue
        order = np.argsort(-sim[i])
        best = [k for k in order[:8] if sim[i][k] >= args.threshold]
        ok = [k for k in best if sizes_agree(text, gdf.n.iloc[k])
              and variant(text) == variant(gdf.offer.iloc[k])]
        if not ok:
            continue
        skusets = {gdf.skus.iloc[k] for k in ok}
        if len(skusets) > 1:
            conflicts += 1
            recs.append({"Dump_Offer": text, "occurrences": row["occurrences"],
                         "skus": "", "source": "conflict",
                         "matched_gold_offer": gdf.offer.iloc[ok[0]],
                         "text_score": float(sim[i][ok[0]]),
                         "status": "CONFLICT_NEEDS_HUMAN"})
            continue
        k = ok[0]
        recs.append({"Dump_Offer": text, "occurrences": row["occurrences"],
                     "skus": "|".join(gdf.skus.iloc[k]), "source": "propagated",
                     "matched_gold_offer": gdf.offer.iloc[k],
                     "text_score": float(sim[i][k]), "status": "PROPAGATED"})

    out = pd.DataFrame(recs).sort_values("occurrences", ascending=False)
    human = out[out.source == "human"]
    prop = out[out.source == "propagated"]
    conf = out[out.source == "conflict"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        out.to_excel(xl, sheet_name="labels", index=False)
        (conf if len(conf) else pd.DataFrame({"note": ["no conflicts"]})
         ).to_excel(xl, sheet_name="CONFLICTS", index=False)
        ws = xl.sheets["labels"]
        ws.freeze_panes = "B2"
        for c, w in {"A": 58, "B": 12, "C": 26, "D": 12, "E": 52, "F": 11, "G": 20}.items():
            ws.column_dimensions[c].width = w

    hr, pr = int(human.occurrences.sum()), int(prop.occurrences.sum())
    print(f"\n{'':22s}{'texts':>8s}{'rows':>10s}{'% of dump':>12s}")
    print(f"{'human decisions':22s}{len(human):>8d}{hr:>10,}{hr/total_rows:>11.1%}")
    print(f"{'propagated':22s}{len(prop):>8d}{pr:>10,}{pr/total_rows:>11.1%}")
    print(f"{'TOTAL labelled':22s}{len(human)+len(prop):>8d}{hr+pr:>10,}"
          f"{(hr+pr)/total_rows:>11.1%}")
    print(f"{'conflicts (for you)':22s}{len(conf):>8d}")
    print(f"\nmultiplier: {(hr+pr)/max(hr,1):.1f}x more rows than the exact matches alone")

    if len(prop):
        print("\nsample propagations:")
        print(prop[["Dump_Offer", "matched_gold_offer", "skus", "text_score"]]
              .head(8).to_string(index=False))
    if len(conf):
        print(f"\n{len(conf)} conflicts need a human - see the CONFLICTS sheet")
    print(f"\nwrote {args.out}")
    print("  'source' column separates human from propagated - keep measurement "
          "restricted to source=human")


if __name__ == "__main__":
    main()
