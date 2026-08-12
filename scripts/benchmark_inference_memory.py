"""Benchmark peak memory and wall time for single-pass vs chunked inference.

Run from the repository root::

    .venv/Scripts/python.exe scripts/benchmark_inference_memory.py --offers 40000
    .venv/Scripts/python.exe scripts/benchmark_inference_memory.py --offers 40000 --chunk-sizes 0,10000

``tracemalloc`` is used rather than ``psutil`` because the latter is not a
dependency of this project. It measures Python allocations, which is what the
chunking change targets: the candidate objects, per-row dicts, and intermediate
frames. Native allocations inside RapidFuzz and LightGBM are not counted, so
treat the numbers as a comparison between the two paths rather than as a total
resident-set figure.

Peak memory is reported to stdout and the logger only. It is deliberately never
written into the shadow run manifest, so artifacts stay comparable between runs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import shutil
import sys
import time
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

LOGGER = logging.getLogger("benchmark_inference_memory")

_PRODUCTS = [
    ("Chicken Nuggets-Frozen", "Original", "400g"),
    ("Chicken Strips-Frozen", "Spicy", "400g"),
    ("Beef Burger-Frozen", "Classic", "500g"),
    ("Chicken Popcorn-Frozen", "Hot", "250g"),
    ("Beef Kofta-Frozen", "Traditional", "750g"),
]


class _BenchmarkPredictor:
    """Cheap deterministic scorer, so the benchmark measures execution shape."""

    def predict_raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.linspace(-2.0, 2.0, len(frame))

    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.linspace(0.01, 0.99, len(frame))


@dataclass
class _Measurement:
    label: str
    chunk_size: int
    offers: int
    peak_mib: float
    current_mib: float
    seconds: float
    candidate_rows: int
    status: str


def _build_offers(count: int) -> pd.DataFrame:
    records = []
    for index in range(count):
        product, variant, size = _PRODUCTS[index % len(_PRODUCTS)]
        records.append(
            {
                "offerid": f"bench-{index:07d}",
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


def _build_master() -> pd.DataFrame:
    rows = []
    for index, (product, variant, size) in enumerate(_PRODUCTS, start=1):
        family = product.split("-")[0]
        rows.append(
            {
                "Itemcode": f"{index:03d}",
                "Itemname": family.upper(),
                "Item-Cat-2": family.split()[0],
                "Item-Cat-4": family.split()[-1],
                "Item Description": f"{variant} {family.lower()}",
                "Item-Spec": size,
            }
        )
    return preprocess_product_master(pd.DataFrame(rows))


def _registered(directory: Path) -> RegisteredShadowPackage:
    path = directory / "benchmark.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"benchmark-package")
    return RegisteredShadowPackage(
        package={
            "model_id": "benchmark-model",
            "predictor": _BenchmarkPredictor(),
            "feature_columns": list(
                shadow_pipeline.MODEL_FEATURE_COLUMNS
            ),
            "auto_match_threshold": 0.9,
            "manual_review_threshold": 0.1,
        },
        registry_entry={"model_id": "benchmark-model"},
        package_path=path,
        package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _config(workspace: Path, chunk_size: int):
    config = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))
    shadow = replace(
        config.shadow_mode,
        enabled=True,
        model_id="benchmark-model",
        package_reference=None,
        output_directory=workspace / "shadow",
        challenge_set_directory=workspace / "challenge_sets",
        top_k=5,
    )
    # Written directly so the benchmark can sweep below the configured
    # 10k-25k production bound when comparing shapes.
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


def _measure(
    workspace: Path, offers: pd.DataFrame, master: pd.DataFrame, chunk_size: int
) -> _Measurement:
    label = "single-pass" if chunk_size == 0 else f"chunked {chunk_size:,}"
    run_workspace = workspace / f"run_{chunk_size}"
    if run_workspace.exists():
        shutil.rmtree(run_workspace)
    run_workspace.mkdir(parents=True, exist_ok=True)

    original_loader = shadow_pipeline.load_registered_shadow_package
    shadow_pipeline.load_registered_shadow_package = (
        lambda **_: _registered(run_workspace)
    )
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    status = "OK"
    rows = 0
    try:
        result = run_shadow_observation(
            offers,
            master,
            config=_config(run_workspace, chunk_size),
            shadow_run_id=f"benchmark-{chunk_size}",
        )
        rows = result.prediction_rows
    except MemoryError:
        status = "MemoryError"
    except Exception as error:  # noqa: BLE001 - benchmark reports, never hides
        status = f"{type(error).__name__}: {error}"[:60]
    finally:
        elapsed = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        shadow_pipeline.load_registered_shadow_package = original_loader
        shutil.rmtree(run_workspace, ignore_errors=True)
        gc.collect()

    measurement = _Measurement(
        label=label,
        chunk_size=chunk_size,
        offers=len(offers),
        peak_mib=peak / 1048576,
        current_mib=current / 1048576,
        seconds=elapsed,
        candidate_rows=rows,
        status=status,
    )
    LOGGER.info(
        "%s: peak %.1f MiB, %.1fs, %d candidate rows (%s)",
        measurement.label,
        measurement.peak_mib,
        measurement.seconds,
        measurement.candidate_rows,
        measurement.status,
    )
    return measurement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offers",
        type=int,
        default=20_000,
        help="Synthetic own-brand offers to generate (default: 20000)",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="0,10000,20000,25000",
        help=(
            "Comma-separated chunk sizes to measure. 0 is the single-pass "
            "path (default: 0,10000,20000,25000)"
        ),
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Directory for temporary run output (default: a temp dir)",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )

    chunk_sizes = [
        int(value)
        for value in str(arguments.chunk_sizes).split(",")
        if value.strip()
    ]
    workspace = (
        Path(arguments.workspace)
        if arguments.workspace
        else PROJECT_ROOT / "_benchmark_workspace"
    )
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"Generating {arguments.offers:,} synthetic offers...")
    offers = _build_offers(arguments.offers)
    master = _build_master()
    print(
        f"Offers: {len(offers):,} rows x {len(offers.columns)} columns; "
        f"master: {len(master):,} rows\n"
    )

    measurements = [
        _measure(workspace, offers, master, chunk_size)
        for chunk_size in chunk_sizes
    ]

    header = (
        f"{'path':<18} {'peak MiB':>10} {'retained MiB':>13} "
        f"{'seconds':>9} {'cand rows':>11}  status"
    )
    print("\n" + header)
    print("-" * len(header))
    for measurement in measurements:
        print(
            f"{measurement.label:<18} {measurement.peak_mib:>10.1f} "
            f"{measurement.current_mib:>13.1f} "
            f"{measurement.seconds:>9.1f} "
            f"{measurement.candidate_rows:>11,}  {measurement.status}"
        )

    baseline = next(
        (item for item in measurements if item.chunk_size == 0), None
    )
    if baseline is not None and baseline.status == "OK":
        print()
        for measurement in measurements:
            if measurement.chunk_size == 0 or measurement.status != "OK":
                continue
            reduction = (
                1 - measurement.peak_mib / baseline.peak_mib
            ) * 100
            print(
                f"  {measurement.label:<18} peak memory "
                f"{reduction:+.1f}% vs single-pass"
            )

    shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
