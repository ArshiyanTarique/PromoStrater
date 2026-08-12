"""Export reviewed labels or one learning-store table to transparent CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from sku_mapping.config import load_config
from sku_mapping.learning.store import LearningStore
from sku_mapping.paths import DEFAULT_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="Override learning_store.database_path.",
    )
    parser.add_argument(
        "--table",
        help="Export a specific allow-listed table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit CSV destination.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    store = LearningStore(
        args.database or config.learning_store.database_path
    )
    stem = args.table or "reviewed_labels"
    destination = (
        args.output
        or config.learning_store.csv_export_directory / f"{stem}.csv"
    )
    if args.table:
        path = store.export_table(args.table, destination)
    else:
        path = store.export_reviewed_labels(destination)
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
