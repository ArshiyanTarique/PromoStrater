"""Train, calibrate, evaluate, and register the ranked SKU matcher properly.

Replaces the train_ranked_model.py + register_ranked_v5.py two-step flow and
closes every gap the bypass registration left open:

- the model is fit on TRAIN families only; calibration is fit on a held-out
  family split (before: refit on all data, no calibration at all);
- thresholds are tuned on CALIBRATED probabilities with explicit precision
  floors, so the packaged thresholds live on the same scale inference uses;
- CV metrics are computed at training time (before: hardcoded literals);
- provenance hashes are real SHA-256 digests of the actual training rows,
  feature table, and split assignment (before: _fake_sha256 placeholders);
- the package is strict schema 3.0 and passes validate_model_package with a
  real fitted calibrator and a ShadowModelPredictor (before: "legacy" schema
  chosen specifically to skip the strict check, with a shim predictor).

Usage:
    .venv\\Scripts\\python.exe scripts\\train_ranked_v5_calibrated.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import lightgbm  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rank_lab as L  # noqa: E402
import sklearn  # noqa: E402

from sku_mapping.constants import FEATURE_GENERATOR_VERSION  # noqa: E402
from sku_mapping.ml.calibration import fit_probability_calibrator  # noqa: E402
from sku_mapping.ml.model_package import (  # noqa: E402
    save_model_package,
    update_model_registry,
)

PARAMS = dict(n_estimators=400, learning_rate=.05, num_leaves=31,
              min_child_samples=20, subsample=.9, colsample_bytree=.9,
              random_state=42, verbose=-1)
CALIBRATION_FAMILY_FRACTION = 0.25
MODEL_VERSION = "ranked-v5-calibrated"
FILENAME = "matcher_ranked_v5_calibrated.joblib"
REGISTRY_DIR = os.path.join(ROOT, "models", "registry")
METADATA_DIR = os.path.join(ROOT, "models", "metadata")
REGISTRY_PATH = os.path.join(ROOT, "models", "model_registry.json")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pick_threshold(probabilities: np.ndarray, labels: np.ndarray,
                    precision_floor: float, min_rows: int) -> float | None:
    for candidate in np.linspace(0.99, 0.05, 95):
        selected = probabilities >= candidate
        if selected.sum() >= min_rows and labels[selected].mean() >= precision_floor:
            return float(candidate)
    return None


def _tail_metrics(predictor, columns, tail_rows) -> dict:
    X, y, oid, cj = L.build_matrix(
        tail_rows, extra=True, rank_feats=True, inject_gold=False
    )
    frame = X.reindex(columns=columns).fillna(-1.0)
    calibrated = np.asarray(predictor.predict_calibrated_proba(frame), dtype=float)
    raw = np.asarray(predictor.model.predict_proba(frame)[:, 1], dtype=float)
    metrics = L.offer_metrics(calibrated, y, oid, cj, tail_rows)
    recall = full_recall = 0.0
    for i, (_, gold) in enumerate(tail_rows):
        candidates = {cj[t] for t in np.where(oid == i)[0]}
        recall += len(gold & candidates) / len(gold)
        full_recall += gold <= candidates
    metrics["Recall@20"] = recall / len(tail_rows)
    metrics["FullRecall@20"] = full_recall / len(tail_rows)
    metrics["offers"] = len(tail_rows)
    metrics["brier_calibrated"] = float(np.mean((calibrated - y) ** 2))
    metrics["brier_raw"] = float(np.mean((raw - y) ** 2))
    metrics["brier_calibrated_positives_only"] = float(
        np.mean((calibrated[y == 1] - 1.0) ** 2)
    )
    metrics["brier_raw_positives_only"] = float(
        np.mean((raw[y == 1] - 1.0) ** 2)
    )
    return metrics


def main() -> None:
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    model_id = f"ranked-v5-cal-{stamp}-matcher"
    package_version = f"ranked-v5-calibrated+{stamp.lower()}"

    rows, fam = L.load_training(include_synthetic=False)
    print(f"training offers {len(rows):,}  families {len(set(fam))}")
    X, y, oid, cj = L.build_matrix(rows, extra=True, rank_feats=True,
                                   inject_gold=True)
    columns = list(X.columns)
    print(f"pairs {len(X):,}  features {len(columns)}")

    training_dataset_hash = _sha256_text(_canonical_json([
        [offer, sorted(int(j) for j in gold), str(fam[i])]
        for i, (offer, gold) in enumerate(rows)
    ]))
    table = X.copy()
    table["pair_label"] = y
    processed_feature_table_hash = _sha256_text(table.to_csv(index=False))

    # ---- family-grouped calibration split (never trains the model)
    rng = np.random.default_rng(L.SEED)
    families = np.array(sorted(set(fam)))
    rng.shuffle(families)
    n_calibration = max(1, int(len(families) * CALIBRATION_FAMILY_FRACTION))
    calibration_families = set(families[:n_calibration].tolist())
    assignment = {
        family: ("calibration" if family in calibration_families else "train")
        for family in families
    }
    split_assignment_hash = _sha256_text(_canonical_json(assignment))
    offer_in_calibration = np.array(
        [f in calibration_families for f in fam]
    )
    pair_in_calibration = offer_in_calibration[oid]
    print(f"split: train {int((~pair_in_calibration).sum()):,} pairs / "
          f"calibration {int(pair_in_calibration.sum()):,} pairs")

    # ---- CV metrics computed NOW, not pasted in later
    def fit_predict(Xtr, ytr, _gtr, Xte):
        member = lgb.LGBMClassifier(**PARAMS)
        member.fit(Xtr, ytr)
        return member.predict_proba(Xte)[:, 1]

    cv = L.cv_evaluate(fit_predict, rows, fam, X, y, oid, cj)
    cv_summary = {
        "cv_hit1_mean": float(cv["Hit@1"].mean()),
        "cv_hit1_sd": float(cv["Hit@1"].std()),
        "cv_pr_auc_mean": float(cv["PR_AUC"].mean()),
        "cv_pr_auc_sd": float(cv["PR_AUC"].std()),
        "cv_folds": int(len(cv)),
    }
    print(f"CV Hit@1 {cv_summary['cv_hit1_mean']:.4f} "
          f"PR_AUC {cv_summary['cv_pr_auc_mean']:.4f}")

    # ---- final model: train families only (calibration stays unseen)
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X[~pair_in_calibration], y[~pair_in_calibration])

    calibration_frame = X[pair_in_calibration].copy()
    calibration_frame["pair_label"] = y[pair_in_calibration]
    calibration_model, predictor, calibration_report, _ = (
        fit_probability_calibrator(
            model,
            calibration_frame,
            requested_method="auto",
            isotonic_min_rows=2000,
            isotonic_min_positive_rows=400,
            random_seed=L.SEED,
            feature_columns=columns,
        )
    )
    print(f"calibration method: {calibration_report['method_selected']}  "
          f"brier raw {calibration_report['raw_probability_metrics']['brier_score']:.4f}"
          f" -> calibrated "
          f"{calibration_report['calibrated_probability_metrics']['brier_score']:.4f}")

    # ---- thresholds tuned on CALIBRATED probabilities
    calibrated = predictor.predict_calibrated_proba(
        calibration_frame.loc[:, columns]
    )
    labels = calibration_frame["pair_label"].to_numpy(dtype=int)
    auto_threshold = _pick_threshold(calibrated, labels, 0.95, 20) or 0.95
    review_threshold = _pick_threshold(calibrated, labels, 0.80, 20) or 0.50
    if review_threshold >= auto_threshold:
        review_threshold = round(max(0.05, auto_threshold - 0.05), 4)
    auto_selected = calibrated >= auto_threshold
    review_selected = (calibrated >= review_threshold) & ~auto_selected
    true_positives = int((auto_selected & (labels == 1)).sum())
    false_positives = int((auto_selected & (labels == 0)).sum())
    false_negatives = int(((~auto_selected) & (labels == 1)).sum())
    selected_metrics = {
        "auto_match_precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        ),
        "auto_match_recall": (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        ),
        "auto_match_true_positives": true_positives,
        "auto_match_false_positives": false_positives,
        "auto_match_false_negatives": false_negatives,
        "manual_review_rows": int(review_selected.sum()),
        "no_match_rows": int((calibrated < review_threshold).sum()),
        "manual_review_coverage": float(review_selected.mean()),
    }
    threshold_evidence = {
        "split_used": "calibration",
        "scale": "calibrated_probability",
        "auto_rule": ">=95% precision with >=20 rows selected",
        "review_rule": ">=80% precision with >=20 rows selected",
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "auto_threshold": float(auto_threshold),
        "manual_review_threshold": float(review_threshold),
        # Read by the dashboard model-insights page so the champion shows
        # plain-language precision/recall evidence, not "unavailable".
        "selected_metrics": selected_metrics,
    }
    print(f"thresholds: auto {auto_threshold:.2f}  review {review_threshold:.2f}")
    print(f"at auto threshold: precision "
          f"{selected_metrics['auto_match_precision']:.4f}  recall "
          f"{selected_metrics['auto_match_recall']:.4f}  "
          f"TP={true_positives} FP={false_positives}")

    tail = _tail_metrics(predictor, columns, L.load_tail())
    print(f"held-out tail: Hit@1 {tail['Hit@1']:.4f}  "
          f"brier {tail['brier_calibrated']:.4f} "
          f"(raw {tail['brier_raw']:.4f})  "
          f"positives brier {tail['brier_calibrated_positives_only']:.4f} "
          f"(raw {tail['brier_raw_positives_only']:.4f})")

    package = {
        "package_schema_version": "3.0",
        "model": model,
        "calibration_model": calibration_model,
        "predictor": predictor,
        "feature_columns": columns,
        "feature_count": len(columns),
        "requires_group_features": True,
        "retrieval_k": L.K,
        "auto_match_threshold": float(auto_threshold),
        "manual_review_threshold": float(review_threshold),
        "approved_auto_match_threshold": None,
        "auto_match_threshold_approved": False,
        "model_id": model_id,
        "package_version": package_version,
        "model_version": MODEL_VERSION,
        "parent_model": "ranked-v5-20260806T103601Z-matcher",
        "training_timestamp": timestamp.isoformat(timespec="seconds"),
        "training_dataset_hash": training_dataset_hash,
        "processed_feature_table_hash": processed_feature_table_hash,
        "split_assignment_hash": split_assignment_hash,
        "metrics": {
            "cv": cv_summary,
            # Flat aliases kept for the ranked-model fallback in the
            # dashboard insights service.
            "cv_hit1": cv_summary["cv_hit1_mean"],
            "cv_pr_auc": cv_summary["cv_pr_auc_mean"],
            "calibration": calibration_report,
            "held_out_tail": tail,
            "trained_on": (
                f"{len(rows)} offers / {len(set(fam))} families from 160 "
                f"human reviews + propagation; model fit on train families "
                f"only, calibration fit on {n_calibration} held-out families"
            ),
        },
        "threshold_evidence": threshold_evidence,
        "calibration_method": calibration_report["method_selected"],
        "lightgbm_version": lightgbm.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "compatibility_policy": {
            "python_major_minor": ".".join(
                platform.python_version().split(".")[:2]
            ),
            "lightgbm_major_minor": ".".join(
                lightgbm.__version__.split(".")[:2]
            ),
            "sklearn_major_minor": ".".join(
                sklearn.__version__.split(".")[:2]
            ),
        },
        "training_config": {
            "params": {k: v for k, v in PARAMS.items()},
            "calibration_family_fraction": CALIBRATION_FAMILY_FRACTION,
            "retrieval_k": L.K,
            "inject_gold_during_training": True,
            "notes": (
                "Group-relative rank features over top-20 RapidFuzz "
                "candidates; featurise the whole shortlist together at "
                "inference (requires_group_features)."
            ),
        },
        "random_seed": L.SEED,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
    }

    model_path, metadata_path = save_model_package(
        package,
        os.path.join(REGISTRY_DIR, FILENAME),
        os.path.join(METADATA_DIR, FILENAME.replace(".joblib", ".json")),
    )
    print(f"saved {model_path}")
    print(f"saved {metadata_path}")

    update_model_registry(REGISTRY_PATH, [{
        "package_filename": FILENAME,
        "model_id": model_id,
        "model_version": MODEL_VERSION,
        "package_version": package_version,
        "creation_timestamp": timestamp.isoformat(timespec="seconds"),
        "training_dataset_hash": training_dataset_hash,
        "feature_generator_version": FEATURE_GENERATOR_VERSION,
        "deployment_status": "SHADOW_MODE_ONLY",
        "approval_status": "NOT_APPROVED_FOR_AUTOMATIC_MATCHING",
        "automatic_production_matching_approved": False,
        "notes": (
            f"Ranked v5, properly calibrated "
            f"({calibration_report['method_selected']}). CV Hit@1 "
            f"{cv_summary['cv_hit1_mean']:.4f}, tail Hit@1 "
            f"{tail['Hit@1']:.4f}, tail Brier "
            f"{tail['brier_calibrated']:.4f}. requires_group_features=True "
            f"retrieval_k={L.K}."
        ),
        "parent_model": "ranked-v5-20260806T103601Z-matcher",
    }])
    print(f"registered {model_id} in {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
