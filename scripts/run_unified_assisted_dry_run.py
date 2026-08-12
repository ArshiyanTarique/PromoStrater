"""Run bounded Phase 6E assisted inference without retraining or uploads."""

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
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.inference.pipeline import (
    run_unified_inference_non_blocking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded unified assisted-inference dry run"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-offers", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--embedding-backend", default="local_hashing"
    )
    parser.add_argument(
        "--embedding-model", default="sku-hashing-384"
    )
    parser.add_argument(
        "--embedding-version", default="sku-hashing-384-v1"
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Explicitly call the configured LLM provider",
    )
    parser.add_argument("--llm-provider", default="ollama")
    parser.add_argument("--llm-model", default="llama3.1:8b")
    parser.add_argument(
        "--llm-endpoint", default="http://localhost:11434"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/default.yaml",
    )
    parser.add_argument(
        "--learning-database",
        type=Path,
        help="Optional isolated learning-store path for this dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_offers < 1 or args.top_k < 1:
        raise ValueError("--max-offers and --top-k must be positive")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be within [0, 1]")

    config = load_config(args.config)
    config = replace(
        config,
        ml=replace(
            config.ml,
            mode=MLDeploymentMode.ASSISTED,
            model_id=args.model_id,
            auto_accept_threshold=args.threshold,
        ),
        shadow_mode=replace(
            config.shadow_mode,
            top_k=args.top_k,
            retain_all_candidates=True,
        ),
        embedding=replace(
            config.embedding,
            enabled=True,
            backend=args.embedding_backend,
            model_name=args.embedding_model,
            model_version=args.embedding_version,
            cache_path=config.data.processed_dir
            / "embedding_cache.sqlite3",
        ),
        agreement=replace(
            config.agreement,
            lightgbm_auto_accept_threshold=args.threshold,
        ),
        llm_review=replace(
            config.llm_review,
            enabled=args.enable_llm,
            provider=args.llm_provider,
            model=args.llm_model,
            endpoint=args.llm_endpoint,
            cache_path=config.data.processed_dir
            / "llm_review_cache.sqlite3",
        ),
        learning_store=replace(
            config.learning_store,
            database_path=(
                args.learning_database.resolve()
                if args.learning_database
                else config.learning_store.database_path
            ),
        ),
    )

    offers = preprocess_clickflyer(
        pd.read_csv(config.data.flyer_path, low_memory=False)
    )
    offers = offers[offers["is_own"]].head(args.max_offers).copy()
    master = preprocess_product_master(
        pd.read_excel(config.data.master_path)
    )
    result = run_unified_inference_non_blocking(
        offers,
        master,
        config=config,
        source_path=config.data.flyer_path,
    )
    summary = {
        **result.statistics,
        "status": result.status,
        "output_paths": {
            key: str(path) for key, path in result.output_paths.items()
        },
        "llm_explicitly_enabled": args.enable_llm,
        "production_files_modified": False,
        "product_master_modified": False,
        "training_data_modified": False,
        "self_learning": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
