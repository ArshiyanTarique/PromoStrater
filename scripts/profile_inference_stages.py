"""Profile where wall-clock time goes inside a chunked inference run.

Run from the repository root::

    .venv/Scripts/python.exe scripts/profile_inference_stages.py --offers 1500

Reports three views of the same run:

1. The pipeline's own ``stage_runtimes_seconds``, which are real wall-clock
   spans around each named stage.
2. cProfile's hottest functions by ``tottime`` (time in the function itself,
   excluding sub-calls) - this is what identifies the actual hot loop.
3. cProfile's hottest by ``cumtime`` (time including sub-calls) - this shows
   which stage owns that cost.

Attribution to the named stages is done by matching the profiler's function
keys against the modules that implement each stage, so the mapping is derived
from measurement rather than assumed.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.shadow import pipeline as shadow_pipeline
from sku_mapping.shadow.pipeline import run_shadow_observation
from sku_mapping.shadow.predictor import RegisteredShadowPackage

#: Each named stage maps to the module path fragments that implement it. A
#: profiled function is attributed to the first stage whose fragments match.
_STAGE_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("candidate generation", ("matching\\candidate_generator", "rapidfuzz")),
    (
        "commercial attribute comparison",
        ("features\\commercial_attributes", "features\\commercial_entities"),
    ),
    (
        "feature engineering",
        (
            "features\\feature_generator",
            "features\\measurement_features",
            "features\\semantic_features",
        ),
    ),
    ("embedding retrieval", ("embedding\\",)),
    ("LightGBM prediction", ("shadow\\predictor", "lightgbm", "sklearn")),
    ("agreement policy", ("agreement\\",)),
    ("LLM review", ("llm_review\\",)),
    ("serialization", ("parquet", "pyarrow", "to_csv", "json", "pickle")),
    ("DataFrame operations", ("pandas\\", "numpy\\")),
)

_PRODUCTS = [
    ("Chicken Nuggets-Frozen", "Original", "400g"),
    ("Chicken Strips-Frozen", "Spicy", "400g"),
    ("Beef Burger-Frozen", "Classic", "500g"),
    ("Chicken Popcorn-Frozen", "Hot", "250g"),
    ("Beef Kofta-Frozen", "Traditional", "750g"),
]


class _CheapPredictor:
    """Constant-cost scorer so profiling reflects pipeline shape, not the model."""

    def predict_raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.linspace(-2.0, 2.0, len(frame))

    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.linspace(0.01, 0.99, len(frame))


def _build_offers(count: int) -> pd.DataFrame:
    records = []
    for index in range(count):
        product, variant, size = _PRODUCTS[index % len(_PRODUCTS)]
        records.append(
            {
                "offerid": f"prof-{index:07d}",
                "Offer Name": (
                    f"Al Kabeer {product.split('-')[0]} {size} #{index}"
                ),
                "Product": product,
                "Brand Name": "Al Kabeer",
                "Variant": variant,
                "Base Packsize": size,
                "Retailer Name": f"Retailer {index % 7}",
            }
        )
    offers = preprocess_clickflyer(pd.DataFrame(records))
    offers["ml_decision"] = "MANUAL_REVIEW"
    offers["confidence_tier"] = "medium (ml)"
    offers["matched_itemcode"] = "REVIEW_REQUIRED"
    offers["suggested_itemcode"] = "001"
    return offers


def _build_master(count: int) -> pd.DataFrame:
    rows = []
    for index in range(count):
        product, variant, size = _PRODUCTS[index % len(_PRODUCTS)]
        family = product.split("-")[0]
        rows.append(
            {
                "Itemcode": f"{index + 1:03d}",
                "Itemname": f"{family.upper()} {index}",
                "Item-Cat-2": family.split()[0],
                "Item-Cat-4": family.split()[-1],
                "Item Description": f"{variant} {family.lower()} {index}",
                "Item-Spec": size,
            }
        )
    return preprocess_product_master(pd.DataFrame(rows))


def _registered(directory: Path) -> RegisteredShadowPackage:
    path = directory / "profile.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"profile-package")
    return RegisteredShadowPackage(
        package={
            "model_id": "profile-model",
            "predictor": _CheapPredictor(),
            "feature_columns": list(shadow_pipeline.MODEL_FEATURE_COLUMNS),
            "auto_match_threshold": 0.9,
            "manual_review_threshold": 0.1,
        },
        registry_entry={"model_id": "profile-model"},
        package_path=path,
        package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _config(workspace: Path, chunk_size: int):
    config = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))
    shadow = replace(
        config.shadow_mode,
        enabled=True,
        model_id="profile-model",
        package_reference=None,
        output_directory=workspace / "shadow",
        challenge_set_directory=workspace / "challenge_sets",
        top_k=5,
    )
    object.__setattr__(shadow, "chunk_size", chunk_size)
    return replace(
        config,
        shadow_mode=shadow,
        embedding=replace(
            config.embedding,
            backend="local_hashing",
            cache_path=workspace / "embedding.sqlite3",
        ),
        llm_review=replace(
            config.llm_review, cache_path=workspace / "llm.sqlite3"
        ),
    )


def _classify(function_key: tuple[str, int, str]) -> str:
    filename, _, name = function_key
    haystack = f"{filename}|{name}".lower()
    for stage, fragments in _STAGE_MATCHERS:
        for fragment in fragments:
            if fragment.lower().replace("\\", "/") in haystack.replace(
                "\\", "/"
            ):
                return stage
    return "other / pipeline glue"


def _format_key(function_key: tuple[str, int, str]) -> str:
    filename, line, name = function_key
    short = Path(filename).name if filename != "~" else "<builtin>"
    return f"{short}:{line}({name})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offers", type=int, default=1500)
    parser.add_argument("--master", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--dump", default=None, help="Optional .prof output path"
    )
    arguments = parser.parse_args()

    workspace = PROJECT_ROOT / "_profile_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    print(
        f"Building {arguments.offers:,} offers against "
        f"{arguments.master:,} master SKUs "
        f"(chunk_size={arguments.chunk_size:,})..."
    )
    offers = _build_offers(arguments.offers)
    master = _build_master(arguments.master)

    shadow_pipeline.load_registered_shadow_package = (
        lambda **_: _registered(workspace)
    )

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    result = run_shadow_observation(
        offers,
        master,
        config=_config(workspace, arguments.chunk_size),
        shadow_run_id="profile-run",
    )
    profiler.disable()
    wall = time.perf_counter() - started

    manifest_path = result.output_paths["run_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_runtimes = manifest.get("stage_runtimes_seconds", {})

    print(
        f"\nRun complete: {result.prediction_rows:,} candidate rows from "
        f"{result.offer_groups:,} offers in {wall:.2f}s "
        f"({wall / max(arguments.offers, 1) * 1000:.2f} ms/offer)"
    )

    print("\n=== Pipeline stage wall-clock (from run manifest) ===")
    total_stage = sum(float(v) for v in stage_runtimes.values())
    for stage, seconds in sorted(
        stage_runtimes.items(), key=lambda item: -float(item[1])
    ):
        share = float(seconds) / wall * 100 if wall else 0.0
        print(f"  {stage:<38} {float(seconds):8.2f}s  {share:5.1f}% of wall")
    print(
        f"  {'(sum of instrumented stages)':<38} {total_stage:8.2f}s  "
        f"{total_stage / wall * 100 if wall else 0:5.1f}% of wall"
    )

    stats = pstats.Stats(profiler)
    if arguments.dump:
        stats.dump_stats(arguments.dump)
        print(f"\nRaw profile written to {arguments.dump}")

    entries = []
    for key, (calls, primitive, tottime, cumtime, _) in stats.stats.items():
        entries.append(
            {
                "key": key,
                "calls": calls,
                "primitive": primitive,
                "tottime": tottime,
                "cumtime": cumtime,
                "stage": _classify(key),
            }
        )

    print(
        f"\n=== Top {arguments.top} by tottime "
        f"(self time, excludes sub-calls) ==="
    )
    header = (
        f"{'#':>2} {'tottime':>9} {'percall':>10} {'calls':>12}  "
        f"{'stage':<32} function"
    )
    print(header)
    print("-" * (len(header) + 24))
    for position, entry in enumerate(
        sorted(entries, key=lambda item: -item["tottime"])[: arguments.top],
        start=1,
    ):
        percall = (
            entry["tottime"] / entry["calls"] * 1e6 if entry["calls"] else 0
        )
        print(
            f"{position:>2} {entry['tottime']:>9.2f} {percall:>9.1f}us "
            f"{entry['calls']:>12,}  {entry['stage']:<32} "
            f"{_format_key(entry['key'])}"
        )

    print(f"\n=== Top {arguments.top} by cumtime (includes sub-calls) ===")
    print(header)
    print("-" * (len(header) + 24))
    for position, entry in enumerate(
        sorted(entries, key=lambda item: -item["cumtime"])[: arguments.top],
        start=1,
    ):
        percall = (
            entry["cumtime"] / entry["calls"] * 1e6 if entry["calls"] else 0
        )
        print(
            f"{position:>2} {entry['cumtime']:>9.2f} {percall:>9.1f}us "
            f"{entry['calls']:>12,}  {entry['stage']:<32} "
            f"{_format_key(entry['key'])}"
        )

    print("\n=== Self time grouped by stage ===")
    by_stage: dict[str, dict[str, float]] = {}
    for entry in entries:
        bucket = by_stage.setdefault(
            entry["stage"], {"tottime": 0.0, "calls": 0}
        )
        bucket["tottime"] += entry["tottime"]
        bucket["calls"] += entry["calls"]
    total_self = sum(item["tottime"] for item in by_stage.values()) or 1.0
    print(f"  {'stage':<34} {'self s':>9} {'share':>7} {'calls':>14}")
    print("  " + "-" * 66)
    for stage, bucket in sorted(
        by_stage.items(), key=lambda item: -item[1]["tottime"]
    ):
        print(
            f"  {stage:<34} {bucket['tottime']:>9.2f} "
            f"{bucket['tottime'] / total_self * 100:>6.1f}% "
            f"{int(bucket['calls']):>14,}"
        )

    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(
        25
    )
    print("\n=== pstats cumulative (top 25, raw) ===")
    print(buffer.getvalue().split("Ordered by:")[-1][:4000])

    shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
