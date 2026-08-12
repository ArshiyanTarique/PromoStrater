"""Evaluate retrieval strategies against the human-reviewed ground truth.

Answers: should the embedding sit beside RapidFuzz (retrieval) or beside
LightGBM (scoring)? Uses the human answer key, not the auto-generated labels.

Pipeline under test, run three times with only the retriever swapped:

    offer -> RETRIEVER picks top-K of 237 SKUs
          -> 19 runtime features per candidate (imported from src/)
          -> LightGBM scores them
          -> ranked shortlist

Metrics, because an offer may legitimately map to several SKUs:

    Recall@K      retrieval only - what fraction of the offer's true SKUs
                  reach the shortlist at all. Hard ceiling on everything after.
    Hit@1         the ML model's top pick is one of the true SKUs.
    Coverage@4    of the offer's true SKUs, how many land in the ML top 4.
    FullCover@4   offers where EVERY true SKU landed in the ML top 4. This is
                  the one that exposes bundle offers half-handled.

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate_retrievers.py
    .venv\\Scripts\\python.exe scripts\\evaluate_retrievers.py --k 20 --topn 4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rapidfuzz import fuzz, process                                   # noqa: E402
from sku_mapping.features.feature_generator import build_feature_vector  # noqa: E402
from sku_mapping.features.measurement_features import (                # noqa: E402
    extract_flyer_measures, extract_master_measures)
from sku_mapping.features.text_features import clean_offer_text        # noqa: E402

EMB_MODEL = "BAAI/bge-small-en-v1.5"


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="candidates the retriever passes on")
    ap.add_argument("--topn", type=int, default=4, help="shortlist depth for coverage")
    ap.add_argument("--truth", default=os.path.join(ROOT, "audit_output",
                                                    "review_resolved.xlsx"))
    ap.add_argument("--no-embedding", action="store_true")
    args = ap.parse_args()

    # ---------------------------------------------------------------- catalogue
    m = pd.read_excel(os.path.join(ROOT, "Product_Master.xlsx"))
    m["Itemcode"] = (m["Itemcode"].fillna("").astype(str)
                     .str.replace(r"\.0$", "", regex=True).str.strip())
    m = m[m.Itemcode != ""].drop_duplicates("Itemcode").reset_index(drop=True)
    for c in ("Itemname", "Item-Cat-4", "Item Description", "Item-Spec"):
        m[c] = m[c].fillna("").astype(str)
    m["match_text"] = collapse(" ").join([""]) or ""
    m["match_text"] = [
        collapse(f"{r['Itemname']} {r['Item-Cat-4']} {r['Item Description']}".lower())
        for _, r in m.iterrows()
    ]
    m["measures"] = m["Item-Spec"].apply(extract_master_measures)
    codes = m["Itemcode"].to_numpy()
    row_of = {c: i for i, c in enumerate(codes)}
    n_sku = len(m)

    # ---------------------------------------------------------------- truth
    t = pd.read_excel(args.truth, sheet_name="resolved")
    offer_col = next(c for c in t.columns if c.lower() in ("dump_offer", "offer_text"))
    code_cols = [c for c in t.columns if re.fullmatch(r"SKU\d+_code", c)]
    gold, offers = [], []
    for _, r in t.iterrows():
        s = {str(r[c]).strip() for c in code_cols
             if pd.notna(r[c]) and str(r[c]).strip() and str(r[c]).strip() in row_of}
        if s:
            offers.append(str(r[offer_col]))
            gold.append({row_of[c] for c in s})
    n = len(offers)
    sizes = [len(g) for g in gold]
    print(f"catalogue           : {n_sku} SKUs")
    print(f"evaluation offers   : {n} (with at least one confirmed SKU)")
    print(f"true SKUs per offer : 1x{sizes.count(1)}  2x{sizes.count(2)}  "
          f"3x{sizes.count(3)}  4x{sizes.count(4)}   total {sum(sizes)}")

    # ---------------------------------------------------------------- retrievers
    off_match = [collapse(clean_offer_text(o)) for o in offers]
    cat_match = m["match_text"].tolist()
    S = {}
    S["fuzzy"] = ((process.cdist(off_match, cat_match, scorer=fuzz.token_sort_ratio, workers=-1)
                   + process.cdist(off_match, cat_match, scorer=fuzz.token_set_ratio, workers=-1))
                  / 2.0).astype(np.float32)
    print("\nfuzzy scores done")

    if not args.no_embedding:
        from sentence_transformers import SentenceTransformer
        print(f"loading {EMB_MODEL} (downloads ~130MB on first run) ...")
        enc = SentenceTransformer(EMB_MODEL, device="cpu")

        def l2(a):
            return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)

        vo = l2(enc.encode(offers, batch_size=32, convert_to_numpy=True,
                           show_progress_bar=True))
        vc = l2(enc.encode(cat_match, batch_size=32, convert_to_numpy=True,
                           show_progress_bar=True))
        S["embedding"] = (vo @ vc.T).astype(np.float32)
        print("embedding scores done")

    def topk(mat, k):
        return np.argsort(-mat, axis=1)[:, :k]

    strategies = {}
    for name, mat in S.items():
        strategies[name] = [set(r) for r in topk(mat, args.k)]
    if len(S) > 1:
        strategies["union"] = [a | b for a, b in
                               zip(strategies["fuzzy"], strategies["embedding"])]

    # ---------------------------------------------------------------- ML stage
    model = None
    bundle_path = os.path.join(ROOT, "models", "alkabeer_sku_matcher_v1.joblib")
    if os.path.isfile(bundle_path):
        import joblib
        b = joblib.load(bundle_path)
        model, feats = b["model"], list(b["feature_columns"])
        print(f"\nLightGBM loaded ({len(feats)} features)")
    else:
        print("\nNO MODEL FOUND - reporting retrieval metrics only")

    master_rows = [{"Itemname": r["Itemname"], "Item-Cat-4": r["Item-Cat-4"],
                    "Item Description": r["Item Description"],
                    "Item-Spec": r["Item-Spec"],
                    "master_measures_detailed": r["measures"]}
                   for _, r in m.iterrows()]
    offer_rows = [{"Offer Name": o, "Product": "", "Variant": "", "Base Packsize": "",
                   "offer_measures_detailed": extract_flyer_measures(" " + o)}
                  for o in offers]

    def ml_rank(cands_per_offer):
        """Return each offer's candidates ordered by LightGBM score."""
        rows, owner = [], []
        for i, cands in enumerate(cands_per_offer):
            for j in sorted(cands):
                rows.append(build_feature_vector(offer_rows[i], master_rows[j]))
                owner.append((i, j))
        F = pd.DataFrame(rows)[feats].astype(float).fillna(-1)
        p = (model.predict_proba(F)[:, 1] if hasattr(model, "predict_proba")
             else np.asarray(model.predict(F), float))
        per = [[] for _ in range(len(cands_per_offer))]
        for (i, j), s in zip(owner, p):
            per[i].append((float(s), j))
        return [[j for _, j in sorted(v, key=lambda x: -x[0])] for v in per]

    # ---------------------------------------------------------------- metrics
    out = []
    for name, cands in strategies.items():
        rec = float(np.mean([len(g & c) / len(g) for g, c in zip(gold, cands)]))
        full_ret = float(np.mean([g <= c for g, c in zip(gold, cands)]))
        row = {"strategy": name, f"Recall@{args.k}": rec,
               f"FullRecall@{args.k}": full_ret,
               "avg_candidates": float(np.mean([len(c) for c in cands]))}
        if model is not None:
            ranked = ml_rank(cands)
            hit1 = float(np.mean([bool(r and r[0] in g) for r, g in zip(ranked, gold)]))
            cov = float(np.mean([len(g & set(r[:args.topn])) / len(g)
                                 for r, g in zip(ranked, gold)]))
            full = float(np.mean([g <= set(r[:args.topn]) for r, g in zip(ranked, gold)]))
            multi = [k for k, g in enumerate(gold) if len(g) > 1]
            cov_m = (float(np.mean([len(gold[k] & set(ranked[k][:args.topn])) / len(gold[k])
                                    for k in multi])) if multi else float("nan"))
            row.update({"Hit@1": hit1, f"Coverage@{args.topn}": cov,
                        f"FullCover@{args.topn}": full,
                        f"Coverage@{args.topn}_multiSKU": cov_m})
        out.append(row)

    res = pd.DataFrame(out)
    print("\n" + "=" * 78)
    print(f"RESULTS  (retriever passes {args.k} candidates, shortlist depth {args.topn})")
    print("=" * 78)
    print(res.round(3).to_string(index=False))

    print("\nhow to read this")
    print(f"  Recall@{args.k:<3d}      retrieval ceiling - true SKUs reaching the shortlist")
    print("  Hit@1            ML's top pick is one of the offer's true SKUs")
    print(f"  Coverage@{args.topn}      of an offer's true SKUs, share landing in the top {args.topn}")
    print(f"  FullCover@{args.topn}     offers where EVERY true SKU made the top {args.topn}")
    if len(res) > 1 and "Hit@1" in res:
        best = res.sort_values(f"FullCover@{args.topn}", ascending=False).iloc[0]
        print(f"\nbest by FullCover@{args.topn}: {best.strategy}")
        f = res[res.strategy == "fuzzy"].iloc[0]
        for s in res.strategy:
            if s == "fuzzy":
                continue
            r = res[res.strategy == s].iloc[0]
            print(f"  {s:10s} vs fuzzy: Recall {r[f'Recall@{args.k}']-f[f'Recall@{args.k}']:+.3f} "
                  f"| Hit@1 {r['Hit@1']-f['Hit@1']:+.3f} "
                  f"| FullCover {r[f'FullCover@{args.topn}']-f[f'FullCover@{args.topn}']:+.3f}")

    dest = os.path.join(ROOT, "audit_output", "retriever_evaluation.csv")
    res.to_csv(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
