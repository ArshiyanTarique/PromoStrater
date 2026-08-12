"""Score the candidate models end to end: retrieval -> 19 features -> LightGBM.

Compares every saved model on the human-reviewed datasets, so a retrain can be
judged before it is promoted.

    1  models/alkabeer_sku_matcher_v1.joblib     original, auto-generated labels
    2  models/matcher_human_labels.joblib        first 100 human reviews
    3  models/matcher_human_labels_v2.joblib     160 human reviews + propagation

The held-out tail set is the one that matters: those 30 offers were sampled at
random and never trained on, so they are the only unbiased read available.

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate_full_pipeline.py
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import joblib                                                          # noqa: E402
from rapidfuzz import fuzz, process                                    # noqa: E402
from sklearn.metrics import average_precision_score                    # noqa: E402
from sku_mapping.features.feature_generator import build_feature_vector  # noqa: E402
from sku_mapping.features.measurement_features import (                # noqa: E402
    extract_flyer_measures, extract_master_measures)
from sku_mapping.features.text_features import clean_offer_text        # noqa: E402

K, TOPN = 20, 4


def cs(s):
    return re.sub(r"\s+", " ", str(s)).strip()


m = pd.read_excel(os.path.join(ROOT, "Product_Master.xlsx"))
m["Itemcode"] = (m["Itemcode"].fillna("").astype(str)
                 .str.replace(r"\.0$", "", regex=True).str.strip())
m = m[m.Itemcode != ""].drop_duplicates("Itemcode").reset_index(drop=True)
for c in ("Itemname", "Item-Cat-4", "Item Description", "Item-Spec"):
    m[c] = m[c].fillna("").astype(str)
cat = [cs(f"{r['Itemname']} {r['Item-Cat-4']} {r['Item Description']}".lower())
       for _, r in m.iterrows()]
m["ms"] = m["Item-Spec"].apply(extract_master_measures)
row_of = {c: i for i, c in enumerate(m["Itemcode"])}
mr = [{"Itemname": r["Itemname"], "Item-Cat-4": r["Item-Cat-4"],
       "Item Description": r["Item Description"], "Item-Spec": r["Item-Spec"],
       "master_measures_detailed": r["ms"]} for _, r in m.iterrows()]

_look = {}
for _, r in m.iterrows():
    for f in (r["Itemname"], r["Item Description"], f"{r['Itemname']} {r['Item-Spec']}"):
        k = cs(re.sub(r"[^a-z0-9]+", " ", str(f).lower()))
        if k:
            _look.setdefault(k, r["Itemcode"])


def load(path, sheet=0):
    """Read a review sheet holding either SKU codes or free-text descriptions."""
    if not os.path.isfile(path):
        return []
    d = pd.read_excel(path, sheet_name=sheet)
    oc = next(c for c in d.columns if c.lower() in ("dump_offer", "offer_text"))
    cc = [c for c in d.columns if re.fullmatch(r"SKU\d+_code", c)]
    out = []
    if cc:
        for _, r in d.iterrows():
            s = {row_of[str(r[c]).strip()] for c in cc
                 if pd.notna(r[c]) and str(r[c]).strip() in row_of}
            if s:
                out.append((str(r[oc]), s))
    else:
        sk = [c for c in d.columns if re.search(r"SKU\d+_Desc", c, re.I)]
        for _, r in d.iterrows():
            s = set()
            for c in sk:
                v = r[c]
                if pd.isna(v):
                    continue
                k = cs(re.sub(r"[^a-z0-9]+", " ", str(v).lower()))
                if k and not k.startswith("none") and k in _look:
                    s.add(row_of[_look[k]])
            if s:
                out.append((str(r[oc]), s))
    return out


def retrieve(texts, top_k=K):
    q = [cs(clean_offer_text(t)) for t in texts]
    S = ((process.cdist(q, cat, scorer=fuzz.token_sort_ratio, workers=-1)
          + process.cdist(q, cat, scorer=fuzz.token_set_ratio, workers=-1)) / 2.0)
    return np.argsort(-S, axis=1)[:, :top_k]


MODELS = []
for name, p in (("1 original", "models/alkabeer_sku_matcher_v1.joblib"),
                ("2 human-100", "models/matcher_human_labels.joblib"),
                ("3 human-160", "models/matcher_human_labels_v2.joblib")):
    f = os.path.join(ROOT, p)
    if os.path.isfile(f):
        b = joblib.load(f)
        MODELS.append((name, b["model"], list(b["feature_columns"])))
print("models  :", [n for n, _, _ in MODELS])

DATASETS = [
    ("held-out tail", load(os.path.join(
        ROOT, "gold_data_PERFECT", "Human_Reviewed_30Rows.xlsx"))),
    ("all gold", load(os.path.join(
        ROOT, "gold_data_PERFECT", "Human_Reviewed_160_MERGED.xlsx"), "resolved")),
]
DATASETS = [(n, d) for n, d in DATASETS if d]
print("datasets:", [(n, len(d)) for n, d in DATASETS])

rows = []
for dname, ds in DATASETS:
    order = retrieve([o for o, _ in ds])
    feats, oid, cj, ys = [], [], [], []
    for i, (o, g) in enumerate(ds):
        orow = {"Offer Name": o, "Product": "", "Variant": "", "Base Packsize": "",
                "offer_measures_detailed": extract_flyer_measures(" " + o)}
        for j in sorted(set(order[i].tolist()) | g):
            feats.append(build_feature_vector(orow, mr[j]))
            oid.append(i); cj.append(j); ys.append(1 if j in g else 0)
    X = pd.DataFrame(feats).astype(float).fillna(-1)
    oid, cj, ys = np.array(oid), np.array(cj), np.array(ys)
    for name, mdl, cols in MODELS:
        p = mdl.predict_proba(X[cols])[:, 1]
        h = c = f = 0
        for i in np.unique(oid):
            sel = np.where(oid == i)[0]
            rank = [cj[t] for t in sel[np.argsort(-p[sel])]]
            g = ds[i][1]
            h += rank[0] in g
            c += len(g & set(rank[:TOPN])) / len(g)
            f += g <= set(rank[:TOPN])
        k = len(ds)
        rows.append({"dataset": dname, "n": k, "model": name,
                     "PR_AUC": average_precision_score(ys, p),
                     "Hit@1": h / k, f"Coverage@{TOPN}": c / k,
                     f"FullCover@{TOPN}": f / k})

res = pd.DataFrame(rows)
print("\n" + "=" * 76)
print("FULL PIPELINE: retrieval -> 19 runtime features -> LightGBM")
print("=" * 76)
for dname, ds in DATASETS:
    print(f"\n{dname}  (n={len(ds)})")
    print(res[res.dataset == dname][
        ["model", "PR_AUC", "Hit@1", f"Coverage@{TOPN}", f"FullCover@{TOPN}"]
    ].round(3).to_string(index=False))
res.to_csv(os.path.join(ROOT, "audit_output", "full_pipeline_results.csv"), index=False)
print("\nwrote audit_output/full_pipeline_results.csv")
