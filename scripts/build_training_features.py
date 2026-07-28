"""Build the Phase 4 ML training-feature dataset; this script does not train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sku_mapping.training import build_training_features_from_paths


def parse_args() -> argparse.Namespace:
    """Parse file-oriented Phase 4 command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("GOLD_TRAINING_PAIRS_v5_FINAL.csv"),
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("Product_Master.xlsx"),
    )
    parser.add_argument(
        "--clickflyer",
        type=Path,
        default=Path("Alkabeer_Export_Data_Clickflyer.csv"),
        help="Optional exact-match enrichment source.",
    )
    parser.add_argument(
        "--without-clickflyer",
        action="store_true",
        help="Build every offer through the text fallback path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--output-encoding", default="utf-8-sig")
    return parser.parse_args()


def main() -> int:
    """Run feature generation and print machine-readable summary counts."""
    args = parse_args()
    result = build_training_features_from_paths(
        args.gold,
        args.master,
        clickflyer_path=None if args.without_clickflyer else args.clickflyer,
        output_dir=args.output_dir,
        output_encoding=args.output_encoding,
    )
    summary = {
        "total_rows": result.manifest["total_rows"],
        "eligible_binary_rows": result.eligible_binary_rows,
        "accepted_rows": len(result.accepted),
        "rejected_rows": len(result.rejected),
        "class_distribution": result.manifest["class_distribution"],
        "output_paths": {
            key: str(path) for key, path in result.output_paths.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
