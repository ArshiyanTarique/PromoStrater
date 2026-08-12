"""Inspect schema and non-sensitive counts in the persistent learning store."""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    store = LearningStore(
        args.database or config.learning_store.database_path
    )
    print(json.dumps(store.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
