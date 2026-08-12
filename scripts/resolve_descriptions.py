"""Resolve human-written SKU descriptions to Product Master item codes.

Reads a review sheet whose SKU columns hold free text and writes a copy with an
item code beside each one, plus a flag on anything that is not a certain match.

Accepted spellings for a SKU (all are matched):
    CKPC                                    the code itself
    CKPC: CHICKEN POP-CORN                  code-prefixed
    CHICKEN POP-CORN                        Itemname
    CHICKEN POP-CORN [400 Gms x 12 Pkts]    name + spec
    CHICKEN POP-CORN400 Gms x 12 Pkts       the master's own Item Description
    NONE / UNSURE / blank                   no SKU

Anything that is not an exact match is reported with its score and marked
NEEDS_CHECK rather than silently accepted - an earlier fuzzy pass turned
"None(shrimps and prawns are different)" into a real SKU at 86% confidence.

Usage:
    .venv\\Scripts\\python.exe scripts\\resolve_descriptions.py
    .venv\\Scripts\\python.exe scripts\\resolve_descriptions.py --sheet my.xlsx --tab review
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO_ACCEPT = 95          # fuzzy score at or above this is taken as certain
NO_SKU = {"none", "unsure", "n/a", "na", "-", "nan", ""}


def norm(s: object) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_lookup(master: pd.DataFrame) -> tuple[dict, dict, list]:
    exact, code_of = {}, {}
    for _, r in master.iterrows():
        code = r["Itemcode"]
        code_of[code.lower()] = code
        for form in (
            r["Itemname"],
            r["Item Description"],
            f"{r['Itemname']} [{r['Item-Spec']}]",
            f"{r['Itemname']} {r['Item-Spec']}",
            code,
        ):
            k = norm(form)
            if k:
                exact.setdefault(k, code)
    return exact, code_of, list(exact)


def resolve(value: object, exact: dict, code_of: dict, keys: list):
    """Return (itemcode, status, score, matched_on)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "", "BLANK", None, ""
    raw = str(value).strip()
    if norm(raw) in NO_SKU or raw.strip().lower() in NO_SKU:
        return "", "NO_SKU", None, raw

    # "CKPC: CHICKEN POP-CORN" -> trust the code before the colon
    if ":" in raw:
        head = raw.split(":", 1)[0].strip()
        if head.lower() in code_of:
            return code_of[head.lower()], "OK_CODE_PREFIX", 100.0, head

    k = norm(raw)
    if k in exact:
        return exact[k], "OK_EXACT", 100.0, raw

    # A leading "none..." is a rejection with commentary, never a SKU.
    if k.startswith("none"):
        return "", "NO_SKU", None, raw

    best = process.extractOne(k, keys, scorer=fuzz.WRatio)
    if not best:
        return "", "NEEDS_CHECK", 0.0, raw
    code, score = exact[best[0]], float(best[1])
    return (code, "OK_FUZZY" if score >= AUTO_ACCEPT else "NEEDS_CHECK",
            score, best[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=os.path.join(ROOT, "review_sheet.xlsx"))
    ap.add_argument("--tab", default="INSTRUCTIONS")
    ap.add_argument("--out", default=os.path.join(ROOT, "audit_output",
                                                  "review_resolved.xlsx"))
    args = ap.parse_args()

    master = pd.read_excel(os.path.join(ROOT, "Product_Master.xlsx"))
    master["Itemcode"] = (master["Itemcode"].fillna("").astype(str)
                          .str.replace(r"\.0$", "", regex=True).str.strip())
    master = master[master.Itemcode != ""].drop_duplicates("Itemcode")
    for c in ("Itemname", "Item-Spec", "Item Description"):
        master[c] = master[c].fillna("").astype(str).str.strip()
    exact, code_of, keys = build_lookup(master)
    name_of = dict(zip(master.Itemcode, master.Itemname))
    spec_of = dict(zip(master.Itemcode, master["Item-Spec"]))
    print(f"catalogue: {len(master)} SKUs, {len(exact)} recognised spellings")

    df = pd.read_excel(args.sheet, sheet_name=args.tab)
    offer_col = next((c for c in df.columns
                      if c.lower() in ("dump_offer", "offer_text", "offer")), None)
    if offer_col is None:
        raise SystemExit(f"no offer column found in: {list(df.columns)}")
    sku_cols = [c for c in df.columns if re.search(r"SKU\d*_?Desc", c, re.I)]
    if not sku_cols:
        raise SystemExit(f"no SKU description columns found in: {list(df.columns)}")
    df = df.dropna(subset=[offer_col]).reset_index(drop=True)
    print(f"rows: {len(df)}   SKU columns: {sku_cols}")

    out = pd.DataFrame({offer_col: df[offer_col]})
    flags = []
    for col in sku_cols:
        res = [resolve(v, exact, code_of, keys) for v in df[col]]
        n = re.search(r"\d+", col)
        tag = n.group(0) if n else col
        out[col] = df[col]
        out[f"SKU{tag}_code"] = [r[0] for r in res]
        out[f"SKU{tag}_name"] = [name_of.get(r[0], "") for r in res]
        out[f"SKU{tag}_spec"] = [spec_of.get(r[0], "") for r in res]
        out[f"SKU{tag}_status"] = [r[1] for r in res]
        out[f"SKU{tag}_score"] = [r[2] for r in res]
        flags.append(pd.Series([r[1] == "NEEDS_CHECK" for r in res]))

    out["NEEDS_CHECK"] = pd.concat(flags, axis=1).any(axis=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        out.to_excel(xl, sheet_name="resolved", index=False)
        bad = out[out.NEEDS_CHECK]
        (bad if len(bad) else pd.DataFrame({"note": ["nothing needs checking"]})
         ).to_excel(xl, sheet_name="NEEDS_CHECK", index=False)
        ws = xl.sheets["resolved"]
        ws.freeze_panes = "B2"
        ws.column_dimensions["A"].width = 50

    tot = sum(1 for c in sku_cols for v in df[c] if pd.notna(v))
    stat = pd.concat([out[f"SKU{re.search(r'[0-9]+', c).group(0)}_status"]
                      for c in sku_cols if re.search(r"[0-9]+", c)])
    print("\nresolution status across all SKU cells:")
    print(stat.value_counts().to_string())
    n_check = int(out.NEEDS_CHECK.sum())
    print(f"\nnon-blank descriptions: {tot}")
    print(f"rows needing a human check: {n_check}")
    if n_check:
        cols = [offer_col] + [c for c in out.columns if c.endswith(("_code", "_status"))]
        print(out[out.NEEDS_CHECK][cols].head(12).to_string(index=False))
    print(f"\nwrote {args.out}")
    print("  sheet 'resolved'    - every row with its item code")
    print("  sheet 'NEEDS_CHECK' - only the rows to eyeball")


if __name__ == "__main__":
    main()
