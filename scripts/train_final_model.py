"""Train and SAVE the production candidate model from human-reviewed labels.

Model lineage, three artifacts only:

  1  models/alkabeer_sku_matcher_v1.joblib   original, auto-generated labels
  2  models/matcher_human_labels.joblib      first 100 human reviews
  3  models/matcher_human_labels_v2.joblib   this script - 160 human reviews
                                             (100 + 70 random) plus propagation

Held out throughout: the 30 randomly-sampled long-tail offers. They are the only
unbiased measure available and are never trained or threshold-tuned on.

Splitting is deliberately NOT applied during training. The splitter is an
inference-time step; training on split text would bake it in irreversibly.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_final_model.py
"""

from __future__ import annotations

import json
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
from sku_mapping.features.feature_generator import build_feature_vector  # noqa: E402
from sku_mapping.features.measurement_features import (                # noqa: E402
    extract_flyer_measures, extract_master_measures)
from sku_mapping.features.text_features import clean_offer_text        # noqa: E402

K, SEED, TOPN = 20, 42, 4
OUT_MODEL = os.path.join(ROOT, "models", "matcher_human_labels_v2.joblib")


def cs(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# ----------------------------------------------------------------- catalogue
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

# ----------------------------------------------------------------- labels
lab = pd.read_excel(os.path.join(ROOT, "gold_data_PERFECT", "synthesized", "propagated_from_160rows.xlsx"), sheet_name="labels")
lab = lab[lab.skus.notna() & (lab.skus.astype(str).str.strip() != "")]
lab["family"] = np.where(lab.source == "human", lab.Dump_Offer, lab.matched_gold_offer)
offers, gold, fam, src = [], [], [], []
for _, r in lab.iterrows():
    s = {row_of[c] for c in str(r["skus"]).split("|") if c in row_of}
    if s:
        offers.append(str(r["Dump_Offer"])); gold.append(s)
        fam.append(str(r["family"])); src.append(str(r["source"]))
fam, src = np.array(fam), np.array(src)
n = len(offers)
n_human = int((src == "human").sum())
print(f"labelled offers {n:,}  (human {n_human}, propagated {n - n_human})")
print(f"families        {len(set(fam))}")

# hold out families for honest threshold tuning; the 30-row tail file is
# excluded from this dataset entirely and scored separately.
rng = np.random.default_rng(SEED)
fams = np.array(sorted(set(fam))); rng.shuffle(fams)
val_fams = set(fams[:max(1, int(len(fams) * .25))].tolist())
is_val = np.array([f in val_fams for f in fam])
print(f"train {int((~is_val).sum()):,} offers / {len(fams)-len(val_fams)} families | "
      f"val {int(is_val.sum()):,} / {len(val_fams)}")

# ----------------------------------------------------------------- pairs
q = [cs(clean_offer_text(o)) for o in offers]
S = ((process.cdist(q, cat, scorer=fuzz.token_sort_ratio, workers=-1)
      + process.cdist(q, cat, scorer=fuzz.token_set_ratio, workers=-1)) / 2.0)
order = np.argsort(-S, axis=1)[:, :K]
orows = [{"Offer Name": o, "Product": "", "Variant": "", "Base Packsize": "",
          "offer_measures_detailed": extract_flyer_measures(" " + o)} for o in offers]
print(f"building features for ~{n*K:,} pairs ...")
feats, y, oid, cj = [], [], [], []
for i in range(n):
    for j in sorted(set(order[i].tolist()) | gold[i]):
        feats.append(build_feature_vector(orows[i], mr[j]))
        y.append(1 if j in gold[i] else 0); oid.append(i); cj.append(j)
X = pd.DataFrame(feats); FEATS = list(X.columns); X = X.astype(float).fillna(-1)
y, oid, cj = np.array(y), np.array(oid), np.array(cj)
rv = is_val[oid]
print(f"pairs {len(X):,}  positives {int(y.sum()):,} ({y.mean():.1%})")

clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         min_child_samples=20, subsample=0.9,
                         colsample_bytree=0.9, random_state=SEED, verbose=-1)
clf.fit(X[~rv], y[~rv])
print("trained.")

# --------------------------------------------- threshold on held-out families
pv = clf.predict_proba(X[rv][FEATS])[:, 1]
yv = y[rv]
best = (0.5, 0.0)
for t in np.linspace(0.05, 0.95, 91):
    pred = pv >= t
    if pred.sum() == 0:
        continue
    prec = yv[pred].mean()
    rec = pred[yv == 1].mean()
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    if f1 > best[1]:
        best = (float(t), f1)
auto_t = next((float(t) for t in np.linspace(0.95, 0.05, 91)
               if (pv >= t).sum() >= 20 and yv[pv >= t].mean() >= 0.95), best[0])
print(f"threshold: best-F1 {best[0]:.2f} (F1 {best[1]:.3f}) | "
      f"auto-accept @95% precision {auto_t:.2f}")

joblib.dump({
    "model": clf, "feature_columns": FEATS,
    "auto_match_threshold": auto_t, "manual_review_threshold": best[0],
    "model_version": "human-labels-v2",
    "trained_on": f"{n_human} human-reviewed offers (100+70) + "
                  f"{n-n_human} propagated, {len(fams)-len(val_fams)} families",
    "held_out": "30 random long-tail offers (Human_Reviewed_30Rows.xlsx)",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}, OUT_MODEL)
print(f"\nsaved {OUT_MODEL}")

# ----------------------------------------------------------------- compare
def load_tail():
    p = os.path.join(ROOT, "gold_data_PERFECT", "Human_Reviewed_30Rows.xlsx")
    d = pd.read_excel(p, sheet_name=0)
    look = {}
    for _, r in m.iterrows():
        for f in (r["Itemname"], r["Item Description"], f"{r['Itemname']} {r['Item-Spec']}"):
            k = cs(re.sub(r"[^a-z0-9]+", " ", str(f).lower()))
            if k:
                look.setdefault(k, r["Itemcode"])
    sk = [c for c in d.columns if re.search(r"SKU\d+_Desc", c, re.I)]
    out = []
    for _, r in d.iterrows():
        s = set()
        for c in sk:
            v = r[c]
            if pd.isna(v):
                continue
            k = cs(re.sub(r"[^a-z0-9]+", " ", str(v).lower()))
            if k and not k.startswith("none") and k in look:
                s.add(row_of[look[k]])
        if s:
            out.append((str(r["Dump_Offer"]), s))
    return out


tail = load_tail()
to = [o for o, _ in tail]
tq = [cs(clean_offer_text(t)) for t in to]
TS = ((process.cdist(tq, cat, scorer=fuzz.token_sort_ratio, workers=-1)
       + process.cdist(tq, cat, scorer=fuzz.token_set_ratio, workers=-1)) / 2.0)
torder = np.argsort(-TS, axis=1)[:, :K]
trows = [{"Offer Name": o, "Product": "", "Variant": "", "Base Packsize": "",
          "offer_measures_detailed": extract_flyer_measures(" " + o)} for o in to]
tf, ty, ti, tc = [], [], [], []
for i, (o, g) in enumerate(tail):
    for j in sorted(set(torder[i].tolist()) | g):
        tf.append(build_feature_vector(trows[i], mr[j]))
        ty.append(1 if j in g else 0); ti.append(i); tc.append(j)
TX = pd.DataFrame(tf).astype(float).fillna(-1)
ty, ti, tc = np.array(ty), np.array(ti), np.array(tc)

rows = []
for name, path in (("1 original (auto labels)", "models/alkabeer_sku_matcher_v1.joblib"),
                   ("2 human 100 rows", "models/matcher_human_labels.joblib"),
                   ("3 human 160 rows (NEW)", OUT_MODEL)):
    full = os.path.join(ROOT, path) if not os.path.isabs(path) else path
    if not os.path.isfile(full):
        continue
    bb = joblib.load(full)
    p = bb["model"].predict_proba(TX[list(bb["feature_columns"])])[:, 1]
    h = c = f = 0
    for i in np.unique(ti):
        sel = np.where(ti == i)[0]
        rank = [tc[t] for t in sel[np.argsort(-p[sel])]]
        g = tail[i][1]
        h += rank[0] in g
        c += len(g & set(rank[:TOPN])) / len(g)
        f += g <= set(rank[:TOPN])
    k = len(tail)
    rows.append({"model": name, "PR_AUC": average_precision_score(ty, p),
                 "Hit@1": h / k, f"Coverage@{TOPN}": c / k, f"FullCover@{TOPN}": f / k})

res = pd.DataFrame(rows)
print("\n" + "=" * 74)
print(f"ALL THREE MODELS on the held-out 30 long-tail offers (n={len(tail)})")
print("=" * 74)
print(res.round(3).to_string(index=False))
res.to_csv(os.path.join(ROOT, "audit_output", "three_model_comparison.csv"), index=False)
print("\nwrote audit_output/three_model_comparison.csv")
