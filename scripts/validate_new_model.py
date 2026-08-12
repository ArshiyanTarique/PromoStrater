"""Stress-test the model trained on human labels.

Three checks the headline number does not survive on its own:

  C1  Human-only test rows. Propagated rows are near-duplicates of a gold row;
      even held out by family they are easier than real unseen offers. Scoring
      only the HUMAN rows in held-out families removes that advantage.

  C2  Singleton families - gold reviews that produced no propagated children.
      Nothing resembling them exists anywhere in training. The cleanest test
      available.

  C3  Does propagation actually help, or just inflate? Trains a second model on
      HUMAN rows only and compares. If human-only matches the full model, the
      879 propagated rows are adding volume rather than information.

Also writes a random long-tail sample for human labelling, which is the one
check that cannot be automated.

Usage:
    .venv\\Scripts\\python.exe scripts\\validate_new_model.py
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rapidfuzz import fuzz, process                                      # noqa: E402
from sku_mapping.features.feature_generator import build_feature_vector  # noqa: E402
from sku_mapping.features.measurement_features import (                  # noqa: E402
    extract_flyer_measures, extract_master_measures)
from sku_mapping.features.text_features import clean_offer_text          # noqa: E402

K, SEED, TOPN = 20, 42, 4


def collapse(s):
    return re.sub(r"\s+", " ", str(s)).strip()


m = pd.read_excel(os.path.join(ROOT, "Product_Master.xlsx"))
m["Itemcode"] = (m["Itemcode"].fillna("").astype(str)
                 .str.replace(r"\.0$", "", regex=True).str.strip())
m = m[m.Itemcode != ""].drop_duplicates("Itemcode").reset_index(drop=True)
for c in ("Itemname", "Item-Cat-4", "Item Description", "Item-Spec"):
    m[c] = m[c].fillna("").astype(str)
m["match_text"] = [collapse(f"{r['Itemname']} {r['Item-Cat-4']} {r['Item Description']}".lower())
                   for _, r in m.iterrows()]
m["measures"] = m["Item-Spec"].apply(extract_master_measures)
row_of = {c: i for i, c in enumerate(m["Itemcode"])}
master_rows = [{"Itemname": r["Itemname"], "Item-Cat-4": r["Item-Cat-4"],
                "Item Description": r["Item Description"], "Item-Spec": r["Item-Spec"],
                "master_measures_detailed": r["measures"]} for _, r in m.iterrows()]

def _find(*patterns):
    """Locate a file by glob so renaming it does not break the script."""
    import glob
    for pat in patterns:
        hits = [h for h in glob.glob(os.path.join(ROOT, pat), recursive=True)
                if "~$" not in h]
        if hits:
            return max(hits, key=os.path.getmtime)
    raise SystemExit(f"none of these matched: {patterns}")


LAB_PATH = _find("gold_data_PERFECT/**/propagated*.xlsx",
                 "gold_data_PERFECT/**/*ropagated*.xlsx")
print(f"labels : {os.path.relpath(LAB_PATH, ROOT)}")
lab = pd.read_excel(LAB_PATH, sheet_name="labels")
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
fam_size = pd.Series(fam).value_counts()
singleton = np.array([fam_size[f] == 1 for f in fam])
print(f"offers {n}  families {fam_size.size}  singleton families {int(singleton.sum())}")

rng = np.random.default_rng(SEED)
fams = np.array(sorted(set(fam))); rng.shuffle(fams)
test_fams = set(fams[:max(1, int(len(fams) * .30))].tolist())
is_test = np.array([f in test_fams for f in fam])

off_match = [collapse(clean_offer_text(o)) for o in offers]
S = ((process.cdist(off_match, m["match_text"].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
      + process.cdist(off_match, m["match_text"].tolist(), scorer=fuzz.token_set_ratio, workers=-1))
     / 2.0).astype(np.float32)
order = np.argsort(-S, axis=1)[:, :K]
offer_rows = [{"Offer Name": o, "Product": "", "Variant": "", "Base Packsize": "",
               "offer_measures_detailed": extract_flyer_measures(" " + o)} for o in offers]

print("building features ...")
feats, y, oid, cand_j = [], [], [], []
for i in range(n):
    for j in sorted(set(order[i].tolist()) | gold[i]):
        feats.append(build_feature_vector(offer_rows[i], master_rows[j]))
        y.append(1 if j in gold[i] else 0); oid.append(i); cand_j.append(j)
X = pd.DataFrame(feats); FEATS = list(X.columns); X = X.astype(float).fillna(-1)
y, oid, cand_j = np.array(y), np.array(oid), np.array(cand_j)
row_test = is_test[oid]
row_human = src[oid] == "human"
row_single = singleton[oid]

import lightgbm as lgb
import joblib
from sklearn.metrics import average_precision_score


def fit(mask):
    c = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, subsample=0.9,
                           colsample_bytree=0.9, random_state=SEED, verbose=-1)
    c.fit(X[mask], y[mask]); return c


full = fit(~row_test)
human_only = fit((~row_test) & row_human)
old = joblib.load(os.path.join(ROOT, "models", "alkabeer_sku_matcher_v1.joblib"))
print(f"trained: full={int((~row_test).sum())} rows, "
      f"human-only={int(((~row_test)&row_human).sum())} rows")


def evaluate(model, cols, mask):
    if mask.sum() == 0:
        return None
    p = model.predict_proba(X[mask][cols])[:, 1]
    idx = np.where(mask)[0]
    hits = cov = fullc = 0; k = 0
    for i in np.unique(oid[idx]):
        sel = idx[oid[idx] == i]
        sc = p[np.searchsorted(idx, sel)]
        rank = [cand_j[t] for t in sel[np.argsort(-sc)]]
        g = gold[i]
        hits += rank[0] in g
        cov += len(g & set(rank[:TOPN])) / len(g)
        fullc += g <= set(rank[:TOPN]); k += 1
    return {"offers": k, "PR_AUC": average_precision_score(y[mask], p),
            "Hit@1": hits / k, f"Coverage@{TOPN}": cov / k, f"FullCover@{TOPN}": fullc / k}


checks = [
    ("C0  all held-out rows", row_test),
    ("C1  held-out HUMAN rows only", row_test & row_human),
    ("C2  singleton families (no children)", row_test & row_single),
]
models = [("NEW full", full, FEATS),
          ("NEW human-only", human_only, FEATS),
          ("OLD shipped", old["model"], list(old["feature_columns"]))]

rows = []
for label, mask in checks:
    for name, mdl, cols in models:
        r = evaluate(mdl, cols, mask)
        if r:
            rows.append({"check": label, "model": name, **r})
res = pd.DataFrame(rows)
print("\n" + "=" * 92)
print("VALIDATION")
print("=" * 92)
for label, _ in checks:
    sub = res[res.check == label]
    if len(sub):
        print(f"\n{label}   (n={int(sub.offers.iloc[0])} offers)")
        print(sub[["model", "PR_AUC", "Hit@1", f"Coverage@{TOPN}",
                   f"FullCover@{TOPN}"]].round(3).to_string(index=False))
res.to_csv(os.path.join(ROOT, "audit_output", "model_validation.csv"), index=False)

# ---- C3: long-tail generalisation ------------------------------------------
# If the sheet is already labelled, score it. Otherwise create it and stop.
try:
    dest = _find("gold_data_PERFECT/**/*30*ow*.xlsx",
                 "gold_data_PERFECT/**/tail_sample*.xlsx")
    print(f"tail   : {os.path.relpath(dest, ROOT)}")
except SystemExit:
    dest = os.path.join(ROOT, "audit_output", "tail_sample_to_label.xlsx")


def _norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def _lookup():
    d = {}
    for _, r in m.iterrows():
        for f in (r["Itemname"], r["Item Description"],
                  f"{r['Itemname']} [{r['Item-Spec']}]",
                  f"{r['Itemname']} {r['Item-Spec']}", r["Itemcode"]):
            k = _norm(f)
            if k:
                d.setdefault(k, r["Itemcode"])
    return d


labelled = False
if os.path.isfile(dest):
    t = pd.read_excel(dest, sheet_name=0)
    sku_cols = [c for c in t.columns if re.search(r"SKU\d+_Desc", c, re.I)]
    filled = t[sku_cols].notna().any(axis=1).sum() if sku_cols else 0
    labelled = filled > 0

if labelled:
    look = _lookup()
    t_off, t_gold, unres = [], [], []
    for _, r in t.iterrows():
        s = set()
        for c in sku_cols:
            v = r[c]
            if pd.isna(v) or not str(v).strip():
                continue
            k = _norm(v)
            if k.startswith("none") or k in ("none", "unsure"):
                continue
            code = look.get(k)
            if code:
                s.add(row_of[code])
            else:
                unres.append(str(v))
        if s:
            t_off.append(str(r["Dump_Offer"])); t_gold.append(s)
    print(f"\n\nC3  long-tail sample: {len(t_off)} labelled offers"
          + (f"   ({len(unres)} descriptions unresolved)" if unres else ""))
    for u in unres[:5]:
        print(f"     unresolved: {u!r}")

    if t_off:
        tm = [collapse(clean_offer_text(o_)) for o_ in t_off]
        TS = ((process.cdist(tm, m["match_text"].tolist(),
                             scorer=fuzz.token_sort_ratio, workers=-1)
               + process.cdist(tm, m["match_text"].tolist(),
                               scorer=fuzz.token_set_ratio, workers=-1)) / 2.0)
        torder = np.argsort(-TS, axis=1)[:, :K]
        trows = [{"Offer Name": o_, "Product": "", "Variant": "",
                  "Base Packsize": "",
                  "offer_measures_detailed": extract_flyer_measures(" " + o_)}
                 for o_ in t_off]
        tf, ty, toid, tcj = [], [], [], []
        for i in range(len(t_off)):
            for j in sorted(set(torder[i].tolist()) | t_gold[i]):
                tf.append(build_feature_vector(trows[i], master_rows[j]))
                ty.append(1 if j in t_gold[i] else 0); toid.append(i); tcj.append(j)
        TX = pd.DataFrame(tf).astype(float).fillna(-1)
        ty, toid, tcj = np.array(ty), np.array(toid), np.array(tcj)

        out = []
        for name, mdl, cols in models:
            p = mdl.predict_proba(TX[cols])[:, 1]
            h = c = f = 0
            for i in np.unique(toid):
                sel = np.where(toid == i)[0]
                rank = [tcj[x] for x in sel[np.argsort(-p[sel])]]
                g = t_gold[i]
                h += rank[0] in g
                c += len(g & set(rank[:TOPN])) / len(g)
                f += g <= set(rank[:TOPN])
            k = len(np.unique(toid))
            out.append({"check": "C3  long tail", "model": name, "offers": k,
                        "PR_AUC": average_precision_score(ty, p),
                        "Hit@1": h / k, f"Coverage@{TOPN}": c / k,
                        f"FullCover@{TOPN}": f / k})
        tail_res = pd.DataFrame(out)
        print(tail_res[["model", "PR_AUC", "Hit@1", f"Coverage@{TOPN}",
                        f"FullCover@{TOPN}"]].round(3).to_string(index=False))
        se = (0.5 / max(len(t_off), 1)) ** 0.5 * 1.96
        print(f"  n={len(t_off)} -> Hit@1 is only accurate to about +/-{se:.0%}; "
              f"this detects a cliff, not a small difference.")
        res = pd.concat([res, tail_res], ignore_index=True)
        res.to_csv(os.path.join(ROOT, "audit_output", "model_validation.csv"),
                   index=False)
else:
    o = pd.read_excel(os.path.join(ROOT, "audit_output",
                                   "alkabeer_offers_to_review.xlsx"))
    done = set(lab.Dump_Offer.astype(str))
    tail = o[~o.Dump_Offer.astype(str).isin(done)]
    samp = tail.sample(n=min(30, len(tail)), random_state=SEED)
    pd.DataFrame({"Dump_Offer": samp.Dump_Offer,
                  "Master_SKU1_Description": "", "Master_SKU2_Description": "",
                  "Master_SKU3_Description": "", "Master_SKU4_Description": "",
                  "occurrences": samp.occurrences,
                  "pipeline_guess": samp.current_sku.fillna("")}).to_excel(dest, index=False)
    print(f"\n\nC3 needs YOU: {len(samp)} random long-tail offers written to")
    print(f"   {dest}")
    print("   fill the SKU description columns, then re-run this script -")
    print("   it will detect the labels and score all three models on them.")

print("\nwrote audit_output/model_validation.csv")
