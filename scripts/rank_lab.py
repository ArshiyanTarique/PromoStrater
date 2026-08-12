"""Shared harness for ranking experiments.

Everything that follows needs the same data, the same folds and the same
metrics, or the comparisons mean nothing. This module owns all three.

Why cross-validation and not the held-out tail: the tail is 29 offers, so its
error bar is about +/-16 points and it cannot separate a real 5-point gain from
luck. Family-grouped 5-fold CV over 1,730 offers gives a mean and a spread, and
the tail is then used once at the end as an unbiased sanity check.

Grouping is by FAMILY (a human review plus every row propagated from it) so
near-duplicate texts never straddle a fold boundary.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from rapidfuzz import fuzz, process                                    # noqa: E402
from sklearn.metrics import average_precision_score                    # noqa: E402
from sku_mapping.features.discriminative_features import (             # noqa: E402
    EXTRA_FEATURE_COLUMNS, build_extra_features)
from sku_mapping.features.feature_generator import build_feature_vector  # noqa: E402
from sku_mapping.features.measurement_features import (                # noqa: E402
    extract_flyer_measures, extract_master_measures)
from sku_mapping.features.rank_features import add_rank_features       # noqa: E402
from sku_mapping.features.text_features import clean_offer_text        # noqa: E402

GOLD = os.path.join(ROOT, "gold_data_PERFECT")
K = 20
TOPN = 4
SEED = 42


def cs(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# --------------------------------------------------------------- catalogue
def load_master():
    m = pd.read_excel(os.path.join(ROOT, "Product_Master.xlsx"))
    m["Itemcode"] = (m["Itemcode"].fillna("").astype(str)
                     .str.replace(r"\.0$", "", regex=True).str.strip())
    m = m[m.Itemcode != ""].drop_duplicates("Itemcode").reset_index(drop=True)
    for c in ("Itemname", "Item-Cat-4", "Item Description", "Item-Spec"):
        m[c] = m[c].fillna("").astype(str)
    m["cat_text"] = [cs(f"{r['Itemname']} {r['Item-Cat-4']} "
                        f"{r['Item Description']}".lower())
                     for _, r in m.iterrows()]
    m["ms"] = m["Item-Spec"].apply(extract_master_measures)
    return m


MASTER = load_master()
CAT = MASTER["cat_text"].tolist()
ROW_OF = {c: i for i, c in enumerate(MASTER["Itemcode"])}
MR = [{"Itemname": r["Itemname"], "Item-Cat-4": r["Item-Cat-4"],
       "Item Description": r["Item Description"], "Item-Spec": r["Item-Spec"],
       "master_measures_detailed": r["ms"]} for _, r in MASTER.iterrows()]
MTEXT = [f"{r['Itemname']} {r['Item-Spec']}" for _, r in MASTER.iterrows()]

_LOOK = {}
for _, _r in MASTER.iterrows():
    for _f in (_r["Itemname"], _r["Item Description"],
               f"{_r['Itemname']} {_r['Item-Spec']}"):
        _k = cs(re.sub(r"[^a-z0-9]+", " ", str(_f).lower()))
        if _k:
            _LOOK.setdefault(_k, _r["Itemcode"])


def load_sheet(path, sheet=0):
    """Read a review sheet holding SKU codes or free-text descriptions."""
    if not os.path.isfile(path):
        return []
    d = pd.read_excel(path, sheet_name=sheet)
    oc = next(c for c in d.columns if str(c).lower() in ("dump_offer", "offer_text"))
    cc = [c for c in d.columns if re.fullmatch(r"SKU\d+_code", str(c))]
    out = []
    if cc:
        for _, r in d.iterrows():
            s = {ROW_OF[str(r[c]).strip()] for c in cc
                 if pd.notna(r[c]) and str(r[c]).strip() in ROW_OF}
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
                if k and not k.startswith("none") and k in _LOOK:
                    s.add(ROW_OF[_LOOK[k]])
            if s:
                out.append((str(r[oc]), s))
    return out


def load_training(include_synthetic: bool = False):
    """Return (rows, families). rows = [(offer_text, {master_idx, ...})]."""
    lab = pd.read_excel(os.path.join(GOLD, "synthesized",
                                     "propagated_from_160rows.xlsx"),
                        sheet_name="labels")
    lab = lab[lab.skus.notna()]
    lab["family"] = np.where(lab.source == "human", lab.Dump_Offer,
                             lab.matched_gold_offer)
    rows, fam = [], []
    for _, r in lab.iterrows():
        s = {ROW_OF[c] for c in str(r["skus"]).split("|") if c in ROW_OF}
        if s:
            rows.append((str(r["Dump_Offer"]), s))
            fam.append(str(r["family"]))
    if include_synthetic:
        p = os.path.join(GOLD, "synthesized", "synthetic_uncovered_skus.xlsx")
        if os.path.isfile(p):
            syn = pd.read_excel(p, sheet_name="labels")
            for _, r in syn.iterrows():
                code = str(r["skus"]).strip()
                if code in ROW_OF:
                    rows.append((str(r["Dump_Offer"]), {ROW_OF[code]}))
                    # own family per SKU so synthetic never leaks across folds
                    fam.append(f"SYN::{code}")
    return rows, np.array(fam)


def load_tail():
    return load_sheet(os.path.join(GOLD, "Human_Reviewed_30Rows.xlsx"))


# --------------------------------------------------------------- features
def _rank_features(block: pd.DataFrame) -> pd.DataFrame:
    """Return only the newly-generated rank columns for *block*.

    Delegates to the canonical src implementation so training and inference
    are always identical.
    """
    before = set(block.columns)
    enriched = add_rank_features(block.copy())
    new_cols = [c for c in enriched.columns if c not in before]
    return enriched[new_cols]


def build_matrix(rows, extra: bool = True, rank_feats: bool = False,
                 k: int = K, inject_gold: bool = True):
    """Feature matrix for offer/candidate pairs.

    ``inject_gold`` adds the true SKU even when retrieval missed it. Correct for
    TRAINING (otherwise an offer contributes no positive at all) and wrong for
    EVALUATION (it hands the model the answer). Callers must set it explicitly.
    """
    feats, y, oid, cj = [], [], [], []
    for i, (o, g) in enumerate(rows):
        q = cs(clean_offer_text(o))
        S = (process.cdist([q], CAT, scorer=fuzz.token_sort_ratio)
             + process.cdist([q], CAT, scorer=fuzz.token_set_ratio))[0] / 2.0
        orow = {"Offer Name": o, "Product": "", "Variant": "",
                "Base Packsize": "",
                "offer_measures_detailed": extract_flyer_measures(" " + o)}
        cand = set(np.argsort(-S)[:k].tolist())
        if inject_gold:
            cand |= g
        for j in sorted(cand):
            row = build_feature_vector(orow, MR[j])
            if extra:
                row.update(build_extra_features(o, MTEXT[j]))
            feats.append(row); y.append(1 if j in g else 0)
            oid.append(i); cj.append(j)
    X = pd.DataFrame(feats)
    oid = np.array(oid)
    if rank_feats:
        parts = [_rank_features(X.iloc[np.where(oid == i)[0]])
                 for i in np.unique(oid)]
        X = pd.concat([X, pd.concat(parts).sort_index()], axis=1)
    return X.astype(float).fillna(-1), np.array(y), oid, np.array(cj)


# --------------------------------------------------------------- metrics
def offer_metrics(scores, y, oid, cj, rows, topn: int = TOPN) -> dict:
    hit = cov = full = 0
    for i in np.unique(oid):
        sel = np.where(oid == i)[0]
        rank = [cj[t] for t in sel[np.argsort(-scores[sel])]]
        g = rows[i][1]
        hit += bool(rank) and rank[0] in g
        cov += len(g & set(rank[:topn])) / len(g)
        full += g <= set(rank[:topn])
    n = len(np.unique(oid))
    return {"Hit@1": hit / n,
            "PR_AUC": average_precision_score(y, scores) if y.sum() else float("nan"),
            f"Coverage@{topn}": cov / n, f"FullCover@{topn}": full / n}


def cv_evaluate(fit_predict, rows, fam, X, y, oid, cj, n_splits: int = 5,
                seed: int = SEED) -> pd.DataFrame:
    """Family-grouped K-fold. ``fit_predict(Xtr, ytr, gtr, Xte)`` -> scores."""
    fams = np.array(sorted(set(fam)))
    rng = np.random.default_rng(seed)
    rng.shuffle(fams)
    folds = np.array_split(fams, n_splits)
    out = []
    for f, hold in enumerate(folds):
        hold = set(hold.tolist())
        te_off = np.array([i for i, o in enumerate(rows) if fam[i] in hold])
        if not len(te_off):
            continue
        te = np.isin(oid, te_off)
        tr = ~te
        scores = fit_predict(X[tr], y[tr], oid[tr], X[te])
        sub_rows = {i: rows[i] for i in te_off}
        m = offer_metrics(scores, y[te], oid[te], cj[te],
                          [sub_rows.get(i, ("", set())) for i in range(len(rows))])
        m["fold"] = f
        out.append(m)
    return pd.DataFrame(out)


def summarise(df: pd.DataFrame, label: str) -> dict:
    cols = ["Hit@1", "PR_AUC", f"Coverage@{TOPN}", f"FullCover@{TOPN}"]
    row = {"experiment": label}
    for c in cols:
        row[c] = df[c].mean()
        row[f"{c}_sd"] = df[c].std()
    return row
