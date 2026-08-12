"""Validate and immutably stage a completed shadow human-review CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sku_mapping.config import load_config
from sku_mapping.data.loaders import load_product_master
from sku_mapping.paths import DEFAULT_CONFIG_PATH
from sku_mapping.shadow.intake import stage_completed_review_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = stage_completed_review_file(
        args.review_csv,
        product_master=load_product_master(config.data.master_path),
        staging_directory=config.shadow_mode.review_staging_directory,
    )
    print(json.dumps(result.audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
