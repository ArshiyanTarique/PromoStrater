"""Train and save the production matcher with group-relative rank features.

The winning change from the ranking experiments: describe each candidate
RELATIVE to the others retrieved for the same offer, not just in absolute terms.
A similarity of 62 means something different when the next candidate scores 40
than when it scores 61, and nothing in the previous 25 features expressed that.

Verified over 15 runs (5 model seeds x 3 fold seeds), family-grouped 5-fold CV:

    baseline 25 features   Hit@1 0.8933 (sd 0.017)   PR_AUC 0.9266 (sd 0.004)
    + rank features (41)   Hit@1 0.9179 (sd 0.013)   PR_AUC 0.9686 (sd 0.001)

Approaches that did NOT help, all measured: lambdarank objective, hard-negative
weighting, two-stage reranking, 9-model ensembling, hyperparameter search,
synthetic SKU coverage, offer splitting, embedding retrieval.

IMPORTANT FOR INFERENCE
Rank features are computed ACROSS an offer's candidate set, so scoring one
(offer, SKU) pair in isolation is not possible - the whole candidate list must
be featurised together. Use score_offer() below rather than calling the model
directly.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_ranked_model.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import joblib                                                          # noqa: E402
import lightgbm as lgb                                                 # noqa: E402
import rank_lab as L                                                   # noqa: E402
from sklearn.metrics import average_precision_score                    # noqa: E402

OUT = os.path.join(ROOT, "models", "matcher_ranked_v5.joblib")
PARAMS = dict(n_estimators=400, learning_rate=.05, num_leaves=31,
              min_child_samples=20, subsample=.9, colsample_bytree=.9,
              random_state=42, verbose=-1)


def score_offer(bundle, offer_text: str, k: int = L.K):
    """Rank the catalogue for one offer. Returns [(itemcode, score), ...].

    Featurises the whole candidate list at once because the rank features are
    relative to it.
    """
    rows = [(offer_text, set())]
    X, _, _, cj = L.build_matrix(rows, extra=True, rank_feats=True, k=k,
                                 inject_gold=False)
    p = bundle["model"].predict_proba(X[bundle["feature_columns"]])[:, 1]
    codes = L.MASTER["Itemcode"].tolist()
    return [(codes[cj[t]], float(p[t])) for t in np.argsort(-p)]


def main() -> None:
    rows, fam = L.load_training(include_synthetic=False)
    X, y, oid, cj = L.build_matrix(rows, extra=True, rank_feats=True,
                                   inject_gold=True)
    print(f"offers {len(rows):,}  families {len(set(fam))}  "
          f"pairs {len(X):,}  features {X.shape[1]}")

    # hold out families purely to pick the threshold; never used for fitting
    rng = np.random.default_rng(L.SEED)
    fams = np.array(sorted(set(fam))); rng.shuffle(fams)
    val = set(fams[:max(1, int(len(fams) * .25))].tolist())
    is_val = np.array([f in val for f in fam])
    rv = is_val[oid]

    tuner = lgb.LGBMClassifier(**PARAMS)
    tuner.fit(X[~rv], y[~rv])
    pv, yv = tuner.predict_proba(X[rv])[:, 1], y[rv]

    grid = np.round(np.linspace(.05, .95, 91), 2)

    def clears(t: float, min_precision: float) -> bool:
        selected = pv >= t
        return bool(selected.sum() >= 20
                    and yv[selected].mean() >= min_precision)

    # Auto-accept takes the STRICTEST cutoff clearing 95% - it gates automatic
    # matching, so it errs high and stays aligned with config/default.yaml.
    auto_t = next((float(t) for t in grid[::-1] if clears(t, .95)), 0.5)
    # Review asks the opposite question: the most permissive cutoff still
    # clearing 80%, so the band beneath auto-accept is as wide as the data
    # supports. It must be searched ASCENDING. Scanning down from 0.95 (as this
    # once did) returns the strictest passing cutoff instead, and since anything
    # clearing 95% also clears 80%, it lands on auto_t itself - leaving a review
    # band of zero width and no middle tier for a human to look at.
    review_t = next(
        (float(t) for t in grid if t < auto_t and clears(t, .80)), 0.3)
    if not 0.0 <= review_t < auto_t <= 1.0:
        raise SystemExit(
            f"threshold search produced no usable review band "
            f"(review {review_t:.2f}, auto {auto_t:.2f})")
    print(f"thresholds on held-out families: auto-accept {auto_t:.2f} "
          f"(>=95% precision), review {review_t:.2f} (>=80%)")

    final = lgb.LGBMClassifier(**PARAMS)
    final.fit(X, y)
    joblib.dump({
        "model": final, "feature_columns": list(X.columns),
        "auto_match_threshold": auto_t, "manual_review_threshold": review_t,
        "model_version": "ranked-v5",
        "requires_group_features": True,
        "retrieval_k": L.K,
        "trained_on": f"{len(rows)} offers / {len(set(fam))} families "
                      f"from 160 human reviews + propagation",
        "cv_hit1": 0.9179, "cv_pr_auc": 0.9686,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inference_note": "rank features are relative to the candidate set - "
                          "featurise the whole shortlist together, see "
                          "scripts/train_ranked_model.py:score_offer",
    }, OUT)
    print(f"saved {OUT}")

    # ---- unbiased check on the held-out tail, all models, no gold injected
    tail = L.load_tail()
    Xt, yt, ot, ct = L.build_matrix(tail, extra=True, rank_feats=True,
                                    inject_gold=False)
    Xb, yb, ob, cb = L.build_matrix(tail, extra=True, rank_feats=False,
                                    inject_gold=False)
    out = []
    for nm, path in (("1 original", "models/alkabeer_sku_matcher_v1.joblib"),
                     ("3 human-160", "models/matcher_human_labels_v2.joblib"),
                     ("4 +synthetic", "models/matcher_synthetic_v3.joblib"),
                     ("5 +features", "models/matcher_extra_features_v4.joblib"),
                     ("6 +rank feats (NEW)", OUT)):
        p = os.path.join(ROOT, path) if not os.path.isabs(path) else path
        if not os.path.isfile(p):
            continue
        b = joblib.load(p)
        cols = list(b["feature_columns"])
        use_rank = any("__" in c for c in cols)
        XX, yy, oo, cc2 = (Xt, yt, ot, ct) if use_rank else (Xb, yb, ob, cb)
        sc = b["model"].predict_proba(XX[cols])[:, 1]
        m = L.offer_metrics(sc, yy, oo, cc2, tail)
        out.append({"model": nm, **m})
    res = pd.DataFrame(out)
    print("\n" + "=" * 70)
    print(f"HELD-OUT TAIL - never trained on (n={len(tail)})")
    print("=" * 70)
    print(res.round(3).to_string(index=False))
    res.to_csv(os.path.join(ROOT, "audit_output", "ranked_model_results.csv"),
               index=False)
    print("\nwrote audit_output/ranked_model_results.csv")


if __name__ == "__main__":
    main()
