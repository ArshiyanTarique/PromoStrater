"""Retrain with the discriminative features added, and compare honestly.

Same data and same split as train_final_model.py - the only change is six extra
columns aimed at size, count, flavour, product line and bulk packaging. Anything
the comparison shows is therefore attributable to the features.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_with_extra_features.py
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import joblib                                                          # noqa: E402
import lightgbm as lgb                                                 # noqa: E402
from rapidfuzz import fuzz, process                                    # noqa: E402
from sklearn.metrics import average_precision_score                    # noqa: E402
from sku_mapping.features.discriminative_features import (             # noqa: E402
    EXTRA_FEATURE_COLUMNS, build_extra_features)
from sku_mapping.features.feature_generator import build_feature_vector  # noqa: E402
from sku_mapping.features.measurement_features import (                # noqa: E402
    extract_flyer_measures, extract_master_measures)
from sku_mapping.features.text_features import clean_offer_text        # noqa: E402

GOLD = os.path.join(ROOT, "gold_data_PERFECT")
K, SEED, TOPN = 20, 42, 4
OUT = os.path.join(ROOT, "models", "matcher_extra_features_v4.joblib")


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
# text the extra features read: name plus pack spec, so size and count are visible
mtext = [f"{r['Itemname']} {r['Item-Spec']}" for _, r in m.iterrows()]

_look = {}
for _, r in m.iterrows():
    for f in (r["Itemname"], r["Item Description"], f"{r['Itemname']} {r['Item-Spec']}"):
        k = cs(re.sub(r"[^a-z0-9]+", " ", str(f).lower()))
        if k:
            _look.setdefault(k, r["Itemcode"])


def load(path, sheet=0):
    d = pd.read_excel(path, sheet_name=sheet)
    oc = next(c for c in d.columns if str(c).lower() in ("dump_offer", "offer_text"))
    cc = [c for c in d.columns if re.fullmatch(r"SKU\d+_code", str(c))]
    out = []
    if cc:
        for _, r in d.iterrows():
            s = {row_of[str(r[c]).strip()] for c in cc
                 if pd.notna(r[c]) and str(r[c]).strip() in row_of}
            if s:
                out.append((str(r[oc]), s))
    else:
        sk = [c for c in d.columns if re.search(r"SKU\d+_Desc", str(c), re.I)]
        for _, r in d.iterrows():
            s = set()
            for c in sk:
                if pd.isna(r[c]):
                    continue
                k = cs(re.sub(r"[^a-z0-9]+", " ", str(r[c]).lower()))
                if k and not k.startswith("none") and k in _look:
                    s.add(row_of[_look[k]])
            if s:
                out.append((str(r[oc]), s))
    return out


def build(rows):
    """Feature matrix for offer/candidate pairs, base 19 plus the extras."""
    feats, y, oid, cj = [], [], [], []
    for i, (o, g) in enumerate(rows):
        q = cs(clean_offer_text(o))
        S = (process.cdist([q], cat, scorer=fuzz.token_sort_ratio)
             + process.cdist([q], cat, scorer=fuzz.token_set_ratio))[0] / 2.0
        orow = {"Offer Name": o, "Product": "", "Variant": "", "Base Packsize": "",
                "offer_measures_detailed": extract_flyer_measures(" " + o)}
        for j in sorted(set(np.argsort(-S)[:K].tolist()) | g):
            row = build_feature_vector(orow, mr[j])
            row.update(build_extra_features(o, mtext[j]))
            feats.append(row); y.append(1 if j in g else 0)
            oid.append(i); cj.append(j)
    X = pd.DataFrame(feats)
    return X, np.array(y), np.array(oid), np.array(cj)


lab = pd.read_excel(os.path.join(GOLD, "synthesized",
                                 "propagated_from_160rows.xlsx"), sheet_name="labels")
lab = lab[lab.skus.notna()]
lab["family"] = np.where(lab.source == "human", lab.Dump_Offer, lab.matched_gold_offer)
rows, fam = [], []
for _, r in lab.iterrows():
    s = {row_of[c] for c in str(r["skus"]).split("|") if c in row_of}
    if s:
        rows.append((str(r["Dump_Offer"]), s)); fam.append(str(r["family"]))
fam = np.array(fam)
print(f"training offers {len(rows):,}   families {len(set(fam))}")

rng = np.random.default_rng(SEED)
fams = np.array(sorted(set(fam))); rng.shuffle(fams)
val = set(fams[:max(1, int(len(fams) * .25))].tolist())
is_val = np.array([f in val for f in fam])

X, y, oid, cj = build(rows)
BASE = [c for c in X.columns if c not in EXTRA_FEATURE_COLUMNS]
ALL = BASE + list(EXTRA_FEATURE_COLUMNS)
X = X.astype(float).fillna(-1)
rv = is_val[oid]
print(f"pairs {len(X):,}   base features {len(BASE)}   extra {len(EXTRA_FEATURE_COLUMNS)}")


def fit(cols):
    c = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, subsample=.9,
                           colsample_bytree=.9, random_state=SEED, verbose=-1)
    c.fit(X[~rv][cols], y[~rv])
    return c


base_model, ext_model = fit(BASE), fit(ALL)
print("trained both.")

pv = ext_model.predict_proba(X[rv][ALL])[:, 1]
yv = y[rv]
auto_t = next((float(t) for t in np.linspace(.95, .05, 91)
               if (pv >= t).sum() >= 20 and yv[pv >= t].mean() >= .95), 0.5)
joblib.dump({"model": ext_model, "feature_columns": ALL,
             "auto_match_threshold": auto_t, "manual_review_threshold": auto_t,
             "model_version": "extra-features-v4",
             "trained_on": f"{len(rows)} offers, {len(set(fam))} families",
             "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            OUT)
print(f"saved {OUT}   auto-accept threshold {auto_t:.2f}")

tail = load(os.path.join(GOLD, "Human_Reviewed_30Rows.xlsx"))
gold = load(os.path.join(GOLD, "Human_Reviewed_160_MERGED.xlsx"), "resolved")

out = []
for dname, ds in (("held-out tail", tail), ("all gold", gold)):
    Xt, yt, ot, ct = build(ds)
    Xt = Xt.astype(float).fillna(-1)
    for nm, mdl, cols in (("base 19 features", base_model, BASE),
                          ("+ 6 new features", ext_model, ALL)):
        p = mdl.predict_proba(Xt[cols])[:, 1]
        h = c = f = 0
        for i in np.unique(ot):
            sel = np.where(ot == i)[0]
            rank = [ct[t] for t in sel[np.argsort(-p[sel])]]
            g = ds[i][1]
            h += rank[0] in g
            c += len(g & set(rank[:TOPN])) / len(g)
            f += g <= set(rank[:TOPN])
        k = len(ds)
        out.append({"dataset": dname, "n": k, "model": nm,
                    "PR_AUC": average_precision_score(yt, p), "Hit@1": h / k,
                    f"Coverage@{TOPN}": c / k, f"FullCover@{TOPN}": f / k})

res = pd.DataFrame(out)
print("\n" + "=" * 74)
print("SAME DATA, SAME SPLIT - only the features differ")
print("=" * 74)
for d in res.dataset.unique():
    sub = res[res.dataset == d]
    print(f"\n{d}  (n={int(sub.n.iloc[0])})")
    print(sub[["model", "PR_AUC", "Hit@1", f"Coverage@{TOPN}",
               f"FullCover@{TOPN}"]].round(3).to_string(index=False))
    a, b = sub.iloc[0], sub.iloc[1]
    print(f"   delta  PR_AUC {b.PR_AUC-a.PR_AUC:+.3f}   Hit@1 {b['Hit@1']-a['Hit@1']:+.3f}")

imp = pd.Series(ext_model.feature_importances_, index=ALL).sort_values(ascending=False)
print("\nwhere the new features rank (of %d):" % len(ALL))
for f in EXTRA_FEATURE_COLUMNS:
    print(f"  {f:28s} importance {imp[f]:5.0f}   rank {list(imp.index).index(f)+1}")
res.to_csv(os.path.join(ROOT, "audit_output", "extra_features_comparison.csv"), index=False)
print("\nwrote audit_output/extra_features_comparison.csv")
