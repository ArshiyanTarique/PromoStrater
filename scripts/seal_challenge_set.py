"""Seal staged human reviews without opening or evaluating challenge labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sku_mapping.config import load_config
from sku_mapping.paths import DEFAULT_CONFIG_PATH
from sku_mapping.shadow.challenge import build_sealed_challenge_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-record", type=Path, action="append", required=True)
    parser.add_argument("--shadow-predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = build_sealed_challenge_set(
        normalized_review_paths=args.review_record,
        shadow_predictions_path=args.shadow_predictions,
        challenge_root=config.shadow_mode.challenge_set_directory,
    )
    print(
        json.dumps(
            {
                "challenge_set_id": result.challenge_set_id,
                "status": result.manifest["status"],
                "manifest_path": str(result.manifest_path),
                "evaluation_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
