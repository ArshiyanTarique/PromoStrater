"""Run bounded assisted inference with an explicit registered model.

This script never edits configuration, model packages, registry metadata, or
training data. It is an operational dry run, not a model evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode, MatchDecision
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.matching.candidate_generator import CandidateGenerator
from sku_mapping.ml.deployment import (
    apply_assisted_decisions,
    run_assisted_monitoring_non_blocking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-offers", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/default.yaml"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_offers < 1:
        raise ValueError("--max-offers must be positive")
    config = load_config(args.config)
    config = replace(
        config,
        ml=replace(
            config.ml,
            mode=MLDeploymentMode.ASSISTED,
            model_id=args.model_id,
            auto_accept_threshold=args.threshold,
        ),
    )
    offers = preprocess_clickflyer(
        pd.read_csv(config.data.flyer_path, low_memory=False)
    )
    offers = offers[offers["is_own"]].head(args.max_offers).copy()
    master = preprocess_product_master(pd.read_excel(config.data.master_path))
    candidates = CandidateGenerator(master).generate_candidates_batch(
        offers, top_k=1
    )
    offers["suggested_itemcode"] = [
        ranked[0].itemcode if ranked else "NO_MATCH"
        for ranked in candidates
    ]
    offers["suggested_itemname"] = [
        ranked[0].itemname if ranked else "None"
        for ranked in candidates
    ]
    offers["ml_decision"] = "MANUAL_REVIEW"
    offers["ml_probability"] = pd.NA
    offers["confidence_tier"] = "medium (ml)"
    offers["matched_itemcode"] = "REVIEW_REQUIRED"

    assisted = apply_assisted_decisions(
        offers,
        master,
        config=config,
    )
    accepted = assisted.rows["assisted_decision"].eq(
        MatchDecision.AUTO_ACCEPT.value
    )
    assisted.rows.loc[accepted, "matched_itemcode"] = assisted.rows.loc[
        accepted, "suggested_itemcode"
    ]
    assisted.rows.loc[accepted, "confidence_tier"] = "high (ml)"
    monitoring = run_assisted_monitoring_non_blocking(
        assisted.rows,
        master,
        config=config,
    )
    summary = {
        "status": assisted.status,
        "run_id": assisted.run_id,
        "model_id": assisted.model_id,
        "model_package_sha256": assisted.model_package_sha256,
        "offers": int(len(assisted.rows)),
        "decision_counts": {
            str(key): int(value)
            for key, value in assisted.decisions["decision"]
            .value_counts()
            .items()
        },
        "auto_accept_threshold": args.threshold,
        "threshold_source": "user_configured",
        "production_threshold_approved": False,
        "monitoring_status": monitoring.status,
        "monitoring_rows": monitoring.prediction_rows,
        "assisted_artifacts": {
            key: str(path) for key, path in assisted.output_paths.items()
        },
        "monitoring_artifacts": {
            key: str(path) for key, path in monitoring.output_paths.items()
        },
        "training_or_online_learning_performed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

