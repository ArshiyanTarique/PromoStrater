"""Run a bounded representative v3 shadow observation without production writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.data.loaders import load_product_master
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.features.feature_generator import build_feature_vector
from sku_mapping.matching.matcher import match_preprocessed_offers
from sku_mapping.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from sku_mapping.shadow.pipeline import (
    enable_shadow_for_explicit_model,
    run_shadow_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-scan-rows", type=int, default=8_000)
    parser.add_argument("--offer-sample-size", type=int, default=40)
    parser.add_argument(
        "--production-model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models"
            / "registry"
            / "matcher_ranked_v5_calibrated.joblib"
        ),
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def _production_snapshot(
    offers: pd.DataFrame,
    master: pd.DataFrame,
    production_package: dict,
) -> pd.DataFrame:
    """Build a bounded in-memory snapshot using the current v1 decision rule."""
    snapshot = offers.copy(deep=True)
    matches = match_preprocessed_offers(snapshot, master)
    master_lookup = {
        str(row["Itemcode"]): row for _, row in master.iterrows()
    }
    decisions: list[str] = []
    probabilities: list[float] = []
    suggested_codes: list[str] = []
    suggested_names: list[str] = []
    matched_codes: list[str] = []
    matched_names: list[str] = []
    tiers: list[str] = []
    for offer_position, match in enumerate(matches):
        suggested_codes.append(match.itemcode)
        suggested_names.append(match.itemname)
        if match.itemcode not in master_lookup:
            probability = np.nan
            decision = "NO_CANDIDATE"
        else:
            feature_values = build_feature_vector(
                snapshot.iloc[offer_position], master_lookup[match.itemcode]
            )
            feature_frame = pd.DataFrame(
                [[feature_values[name] for name in MODEL_FEATURE_COLUMNS]],
                columns=MODEL_FEATURE_COLUMNS,
            )
            probability = float(
                production_package["model"].predict_proba(feature_frame)[0, 1]
            )
            if probability >= float(
                production_package["auto_match_threshold"]
            ):
                decision = "AUTO_MATCH"
            elif probability >= float(
                production_package["manual_review_threshold"]
            ):
                decision = "MANUAL_REVIEW"
            else:
                decision = "NO_MATCH"
        probabilities.append(probability)
        decisions.append(decision)
        if decision == "AUTO_MATCH":
            matched_codes.append(match.itemcode)
            matched_names.append(match.itemname)
            tiers.append("high (ml)")
        else:
            matched_codes.append("REVIEW_REQUIRED")
            matched_names.append("REVIEW_REQUIRED")
            tiers.append(
                "medium (ml)" if decision == "MANUAL_REVIEW" else "low (ml)"
            )
    snapshot["suggested_itemcode"] = suggested_codes
    snapshot["suggested_itemname"] = suggested_names
    snapshot["matched_itemcode"] = matched_codes
    snapshot["matched_itemname"] = matched_names
    snapshot["ml_probability"] = probabilities
    snapshot["ml_decision"] = decisions
    snapshot["confidence_tier"] = tiers
    snapshot["source_dataset"] = "CLICKFLYER_REPRESENTATIVE_DRY_RUN"
    return snapshot


def main() -> int:
    args = parse_args()
    if args.source_scan_rows < 1 or args.offer_sample_size < 1:
        raise ValueError("Dry-run row limits must be positive")
    config = load_config(args.config)
    raw = pd.read_csv(
        config.data.flyer_path,
        nrows=args.source_scan_rows,
        dtype={"offerid": "string"},
        low_memory=False,
    )
    prepared = preprocess_clickflyer(raw)
    own = prepared[prepared["is_own"]].copy()
    if own.empty:
        raise ValueError("Bounded source scan contains no own-brand rows")
    own = own.sample(
        n=min(args.offer_sample_size, len(own)),
        random_state=config.runtime.random_seed,
    ).sort_index().reset_index(drop=True)
    master = preprocess_product_master(
        load_product_master(config.data.master_path)
    )
    production_package = joblib.load(args.production_model)
    if list(production_package["feature_columns"]) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Production snapshot model feature order is incompatible")
    production = _production_snapshot(own, master, production_package)
    dry_config = enable_shadow_for_explicit_model(
        config,
        model_id=args.model_id,
        output_directory=args.output_directory,
    )
    result = run_shadow_observation(
        production,
        master,
        config=dry_config,
        model_directory=dry_config.shadow_mode.registry_path.parent / "registry",
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "shadow_run_id": result.shadow_run_id,
                "source_scan_rows": min(args.source_scan_rows, len(raw)),
                "production_offer_rows": len(production),
                "shadow_candidate_rows": result.prediction_rows,
                "offer_groups": result.offer_groups,
                "failed_shadow_predictions": result.failed_shadow_predictions,
                "output_paths": {
                    key: str(path) for key, path in result.output_paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
