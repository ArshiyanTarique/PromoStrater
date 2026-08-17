"""Bounded second-opinion scoring on the shared shadow candidate set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.shadow.pipeline import (
    enable_shadow_for_explicit_model,
    run_shadow_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-offers", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-backend", default="local_hashing"
    )
    parser.add_argument(
        "--embedding-model-name", default="sku-hashing-384"
    )
    parser.add_argument(
        "--embedding-model-version", default="sku-hashing-384-v1"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/default.yaml"
    )
    return parser.parse_args()


def _atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _top_candidates(
    predictions: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    available = predictions[
        pd.to_numeric(predictions[score_column], errors="coerce").notna()
    ].copy()
    available[score_column] = pd.to_numeric(
        available[score_column], errors="coerce"
    )
    return (
        available.sort_values(
            [
                "offer_group_id",
                score_column,
                "candidate_rank",
                "master_itemcode",
            ],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .drop_duplicates("offer_group_id")
        [["offer_group_id", "master_itemcode", score_column]]
    )


def main() -> int:
    args = parse_args()
    if args.max_offers < 1 or args.top_k < 1:
        raise ValueError("--max-offers and --top-k must be positive")
    started = time.perf_counter()
    config = load_config(args.config)
    timestamp = datetime.now(timezone.utc)
    run_id = (
        "embedding-dry-run-"
        + timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    )
    config = enable_shadow_for_explicit_model(
        config,
        model_id=args.model_id,
        output_directory=config.output.output_dir
        / "embedding_dry_runs",
    )
    config = replace(
        config,
        shadow_mode=replace(
            config.shadow_mode,
            top_k=args.top_k,
            retain_all_candidates=True,
        ),
        embedding=replace(
            config.embedding,
            enabled=True,
            backend=args.embedding_backend,
            model_name=args.embedding_model_name,
            model_version=args.embedding_model_version,
            cache_path=config.data.processed_dir
            / "embedding_cache.sqlite3",
        ),
    )
    offers = preprocess_clickflyer(
        pd.read_csv(config.data.flyer_path, low_memory=False)
    )
    offers = offers[offers["is_own"]].head(args.max_offers).copy()
    offers["ml_decision"] = "OBSERVATIONAL_INPUT"
    offers["confidence_tier"] = "observational"
    offers["matched_itemcode"] = ""
    offers["suggested_itemcode"] = ""
    master = preprocess_product_master(pd.read_excel(config.data.master_path))

    result = run_shadow_observation(
        offers,
        master,
        config=config,
        shadow_run_id=run_id,
    )
    predictions = pd.read_parquet(
        result.output_paths["shadow_predictions_parquet"]
    )
    lightgbm_top = _top_candidates(
        predictions, "calibrated_probability"
    ).rename(
        columns={
            "master_itemcode": "lightgbm_top_candidate",
            "calibrated_probability": "lightgbm_top_probability",
        }
    )
    embedding_top = _top_candidates(
        predictions, "embedding_similarity"
    ).rename(
        columns={
            "master_itemcode": "embedding_top_candidate",
            "embedding_similarity": "embedding_top_similarity",
        }
    )
    comparison = lightgbm_top.merge(
        embedding_top, on="offer_group_id", how="outer"
    )
    comparison["top_candidate_agreement"] = (
        comparison["lightgbm_top_candidate"]
        .eq(comparison["embedding_top_candidate"])
        .fillna(False)
    )
    similarity = pd.to_numeric(
        predictions["embedding_similarity"], errors="coerce"
    ).dropna()
    failure_rows = (
        predictions["embedding_failure_reason"]
        .astype("string")
        .fillna("")
        .str.strip()
        .ne("")
    )
    summary = {
        "report_type": "PHASE_6B_BOUNDED_EMBEDDING_DRY_RUN",
        "timestamp": timestamp.isoformat(),
        "run_id": run_id,
        "offers_processed": int(predictions["offer_group_id"].nunique()),
        "candidates_retained": int(len(predictions)),
        "candidates_scored": int(similarity.notna().sum()),
        "average_candidates_per_offer": float(
            len(predictions)
            / max(1, predictions["offer_group_id"].nunique())
        ),
        "embedding_failures": int(failure_rows.sum()),
        "embedding_model_id": (
            str(predictions["embedding_model_id"].iloc[0])
            if len(predictions)
            else None
        ),
        "embedding_model_version": (
            str(predictions["embedding_model_version"].iloc[0])
            if len(predictions)
            else None
        ),
        "lightgbm_model_id": args.model_id,
        "lightgbm_top_candidates": {
            str(row["offer_group_id"]): str(
                row["lightgbm_top_candidate"]
            )
            for _, row in comparison.dropna(
                subset=["lightgbm_top_candidate"]
            ).iterrows()
        },
        "embedding_top_candidates": {
            str(row["offer_group_id"]): str(
                row["embedding_top_candidate"]
            )
            for _, row in comparison.dropna(
                subset=["embedding_top_candidate"]
            ).iterrows()
        },
        "top_candidate_agreement_count": int(
            comparison["top_candidate_agreement"].sum()
        ),
        "top_candidate_comparisons": int(len(comparison)),
        "similarity_distribution_summary": {
            "minimum": float(similarity.min()) if len(similarity) else None,
            "p25": float(similarity.quantile(0.25))
            if len(similarity)
            else None,
            "median": float(similarity.median())
            if len(similarity)
            else None,
            "mean": float(similarity.mean()) if len(similarity) else None,
            "p75": float(similarity.quantile(0.75))
            if len(similarity)
            else None,
            "maximum": float(similarity.max()) if len(similarity) else None,
            "standard_deviation": float(similarity.std(ddof=0))
            if len(similarity)
            else None,
        },
        "runtime_seconds": time.perf_counter() - started,
        "shadow_artifacts": {
            key: str(path) for key, path in result.output_paths.items()
        },
        "embedding_used_for_decisions": False,
        "agreement_policy_implemented": True,
        "agreement_routes_are_diagnostic": True,
        "llm_reviewer_implemented": True,
        "llm_review_enabled": config.llm_review.enabled,
        "llm_review_routes_are_diagnostic": True,
    }
    report_path = (
        config.output.output_dir
        / "embedding_dry_runs"
        / run_id
        / "embedding_dry_run_report.json"
    )
    _atomic_json(summary, report_path)
    summary["dry_run_report"] = str(report_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
