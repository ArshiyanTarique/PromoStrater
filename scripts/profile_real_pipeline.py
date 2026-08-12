"""Profile chunked inference against the real Product Master and ClickFlyer data.

Run from the repository root::

    .venv/Scripts/python.exe scripts/profile_real_pipeline.py --rows 20000

Unlike ``profile_inference_stages.py`` (synthetic, tiny master, stub model),
this uses the production artifacts:

* ``Product_Master.xlsx`` - all 237 SKUs, so candidate generation is scored at
  its real width.
* A leading slice of ``Alkabeer_Export_Data_Clickflyer.csv``, preserving row
  order and therefore the real own-brand ratio and category mix.
* The registered LightGBM package from ``models/model_registry.json``, so
  prediction cost is real rather than stubbed.

Memory is reported two ways because they answer different questions:

* ``tracemalloc`` peak - Python-level allocations only. This is what the
  chunking work targets.
* Win32 ``PeakWorkingSetSize`` - true process RSS including native allocations
  inside RapidFuzz, LightGBM, pyarrow and NumPy, which ``tracemalloc`` cannot
  see. This is the number that decides whether a run fits in 16 GB.

Nothing here writes to the production learning store; all output goes to a
temporary workspace that is removed afterwards.
"""

from __future__ import annotations

import argparse
import cProfile
import ctypes
import ctypes.wintypes as wintypes
import gc
import json
import pstats
import shutil
import sys
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.shadow.pipeline import run_shadow_observation

MODEL_ID = "alkabeer-sku-matcher-v3-20260729T061802974421Z-8c636b0ac4a2"

_STAGE_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cleaning / preprocessing", ("data/preprocessing", "data/loaders")),
    ("candidate generation", ("matching/candidate_generator", "rapidfuzz")),
    (
        "commercial attribute parsing",
        ("features/commercial_attributes", "features/commercial_entities"),
    ),
    (
        "feature engineering",
        (
            "features/feature_generator",
            "features/measurement_features",
            "features/semantic_features",
            "features/text_features",
        ),
    ),
    ("embedding retrieval", ("embedding/",)),
    ("LightGBM prediction", ("shadow/predictor", "lightgbm", "sklearn")),
    ("agreement policy", ("agreement/",)),
    ("LLM review", ("llm_review/",)),
    ("competitor discovery", ("competitors/",)),
    ("serialization", ("pyarrow", "parquet", "to_csv", "json", "pickle")),
    ("DataFrame operations", ("pandas/", "numpy/")),
)


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _memory_counters() -> tuple[float, float]:
    """Return (peak, current) process working set in MiB, or (0, 0) elsewhere.

    Explicit ``argtypes``/``restype`` matter here: ``GetCurrentProcess``
    returns a HANDLE, and without the annotation ctypes truncates it on 64-bit
    Windows so the call silently fails and reports zero.
    """
    try:
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_MemoryCounters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        if not kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return 0.0, 0.0
        return (
            counters.PeakWorkingSetSize / 1048576,
            counters.WorkingSetSize / 1048576,
        )
    except Exception:
        return 0.0, 0.0


def _classify(function_key) -> str:
    filename, _, name = function_key
    haystack = f"{filename}|{name}".lower().replace("\\", "/")
    for stage, fragments in _STAGE_MATCHERS:
        for fragment in fragments:
            if fragment.lower() in haystack:
                return stage
    return "other / pipeline glue"


def _short(function_key) -> str:
    filename, line, name = function_key
    base = Path(filename).name if filename != "~" else "<builtin>"
    return f"{base}:{line}({name})"


def _config(workspace: Path, chunk_size: int, embeddings: bool):
    config = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))
    shadow = replace(
        config.shadow_mode,
        enabled=True,
        model_id=MODEL_ID,
        package_reference=None,
        output_directory=workspace / "shadow",
        review_staging_directory=workspace / "reviews",
        challenge_set_directory=workspace / "challenge",
    )
    object.__setattr__(shadow, "chunk_size", chunk_size)
    return replace(
        config,
        shadow_mode=shadow,
        embedding=replace(
            config.embedding,
            enabled=embeddings,
            backend="local_hashing",
            cache_path=workspace / "embedding.sqlite3",
        ),
        llm_review=replace(
            config.llm_review,
            enabled=False,
            cache_path=workspace / "llm.sqlite3",
        ),
        learning_store=replace(
            config.learning_store, database_path=workspace / "learning.db"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--dump", default=None)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip cProfile; timings only (cProfile adds overhead)",
    )
    arguments = parser.parse_args()

    workspace = PROJECT_ROOT / "_real_profile_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    print("=" * 74)
    print("PromoStrater real-pipeline profile")
    print("=" * 74)

    load_started = time.perf_counter()
    master = preprocess_product_master(
        pd.read_excel(PROJECT_ROOT / "Product_Master.xlsx")
    )
    raw = pd.read_csv(
        PROJECT_ROOT / "Alkabeer_Export_Data_Clickflyer.csv",
        low_memory=False,
        nrows=arguments.rows,
    )
    clean_started = time.perf_counter()
    offers = preprocess_clickflyer(raw)
    cleaning_seconds = time.perf_counter() - clean_started
    load_seconds = time.perf_counter() - load_started

    own_mask = (
        offers["is_own"].fillna(False).astype(bool)
        if "is_own" in offers
        else pd.Series(True, index=offers.index)
    )
    own_count = int(own_mask.sum())
    print(f"\nProduct Master:        {len(master):,} SKUs")
    print(f"ClickFlyer rows read:  {len(raw):,}")
    print(f"After cleaning:        {len(offers):,} rows x {len(offers.columns)} cols")
    print(
        f"Own-brand (inference): {own_count:,} "
        f"({own_count / max(len(offers), 1) * 100:.1f}%)"
    )
    print(f"Load + clean:          {load_seconds:.2f}s "
          f"(cleaning alone {cleaning_seconds:.2f}s)")
    print(f"chunk_size:            {arguments.chunk_size:,}")
    print(f"embeddings:            {'ENABLED' if arguments.embeddings else 'disabled (production default)'}")

    config = _config(workspace, arguments.chunk_size, arguments.embeddings)

    gc.collect()
    rss_before_peak, rss_before = _memory_counters()
    tracemalloc.start()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    profiler = cProfile.Profile() if not arguments.no_profile else None
    status = "OK"
    result = None
    try:
        if profiler is not None:
            profiler.enable()
        result = run_shadow_observation(
            offers,
            master,
            config=config,
            shadow_run_id="real-profile-run",
        )
    except Exception as error:  # noqa: BLE001 - profiling reports, never hides
        status = f"{type(error).__name__}: {error}"
    finally:
        if profiler is not None:
            profiler.disable()
        wall = time.perf_counter() - wall_started
        cpu = time.process_time() - cpu_started
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    peak_rss, current_rss = _memory_counters()
    rows = result.prediction_rows if result is not None else 0
    groups = result.offer_groups if result is not None else 0

    print("\n" + "=" * 74)
    print("THROUGHPUT")
    print("=" * 74)
    print(f"  status                  {status}")
    print(f"  wall clock              {wall:10.2f} s")
    print(f"  CPU time                {cpu:10.2f} s  "
          f"({cpu / wall * 100 if wall else 0:.0f}% of wall - single core bound if ~100%)")
    print(f"  candidate rows          {rows:10,}")
    print(f"  offer groups scored     {groups:10,}")
    if wall:
        print(f"  own-brand offers/sec    {own_count / wall:10.1f}")
        print(f"  candidates/sec          {rows / wall:10.1f}")
        print(f"  feature vectors/sec     {rows / wall:10.1f}  (one per candidate)")
        print(f"  ms per own-brand offer  {wall / max(own_count, 1) * 1000:10.2f}")

    print("\n" + "=" * 74)
    print("MEMORY")
    print("=" * 74)
    print(f"  RSS before run (data already loaded) {rss_before:10.1f} MiB")
    print(f"  RSS after run                        {current_rss:10.1f} MiB")
    print(f"  process PEAK working set (true RSS)  {peak_rss:10.1f} MiB")
    print(f"  peak RSS attributable to the run     "
          f"{max(peak_rss - rss_before, 0.0):10.1f} MiB")
    print(f"  tracemalloc peak (Python objects)    {traced_peak / 1048576:10.1f} MiB")
    print(f"  tracemalloc retained at exit         {traced_current / 1048576:10.1f} MiB")
    if own_count:
        print(f"  peak RSS per own-brand offer         "
              f"{max(peak_rss - rss_before, 0.0) / own_count * 1024:10.1f} KiB")

    if result is not None:
        manifest = json.loads(
            result.output_paths["run_manifest"].read_text(encoding="utf-8")
        )
        stage_runtimes = {
            key: float(value)
            for key, value in manifest.get(
                "stage_runtimes_seconds", {}
            ).items()
            if key != "total_before_manifest"
        }
        print("\n" + "=" * 74)
        print("STAGE WALL-CLOCK (instrumented spans in the pipeline)")
        print("=" * 74)
        print(f"  {'stage':<40} {'seconds':>9} {'% wall':>8}")
        print("  " + "-" * 60)
        print(f"  {'cleaning / preprocessing':<40} {cleaning_seconds:>9.2f} "
              f"{cleaning_seconds / (wall + cleaning_seconds) * 100:>7.1f}%")
        for stage, seconds in sorted(
            stage_runtimes.items(), key=lambda item: -item[1]
        ):
            print(f"  {stage:<40} {seconds:>9.2f} "
                  f"{seconds / wall * 100 if wall else 0:>7.1f}%")
        accounted = sum(stage_runtimes.values())
        print("  " + "-" * 60)
        print(f"  {'accounted':<40} {accounted:>9.2f} "
              f"{accounted / wall * 100 if wall else 0:>7.1f}%")
        print(f"  {'unaccounted (sampling/monitoring/IO)':<40} "
              f"{wall - accounted:>9.2f} "
              f"{(wall - accounted) / wall * 100 if wall else 0:>7.1f}%")

    if profiler is not None:
        stats = pstats.Stats(profiler)
        if arguments.dump:
            stats.dump_stats(arguments.dump)
            print(f"\nRaw profile written to {arguments.dump}")

        entries = [
            {
                "key": key,
                "calls": value[1],
                "tottime": value[2],
                "cumtime": value[3],
                "stage": _classify(key),
            }
            for key, value in stats.stats.items()
        ]

        print("\n" + "=" * 74)
        print(f"TOP {arguments.top} BY SELF TIME")
        print("=" * 74)
        print(f"  {'self s':>8} {'per call':>11} {'calls':>13}  "
              f"{'stage':<30} function")
        print("  " + "-" * 96)
        for entry in sorted(entries, key=lambda i: -i["tottime"])[
            : arguments.top
        ]:
            per = entry["tottime"] / entry["calls"] * 1e6 if entry["calls"] else 0
            print(f"  {entry['tottime']:>8.2f} {per:>10.1f}us "
                  f"{entry['calls']:>13,}  {entry['stage']:<30} "
                  f"{_short(entry['key'])}")

        print("\n" + "=" * 74)
        print(f"TOP {arguments.top} BY CUMULATIVE TIME")
        print("=" * 74)
        print(f"  {'cum s':>8} {'per call':>11} {'calls':>13}  "
              f"{'stage':<30} function")
        print("  " + "-" * 96)
        for entry in sorted(entries, key=lambda i: -i["cumtime"])[
            : arguments.top
        ]:
            per = entry["cumtime"] / entry["calls"] * 1e6 if entry["calls"] else 0
            print(f"  {entry['cumtime']:>8.2f} {per:>10.1f}us "
                  f"{entry['calls']:>13,}  {entry['stage']:<30} "
                  f"{_short(entry['key'])}")

        print("\n" + "=" * 74)
        print("SELF TIME GROUPED BY STAGE")
        print("=" * 74)
        buckets: dict[str, list[float]] = {}
        for entry in entries:
            bucket = buckets.setdefault(entry["stage"], [0.0, 0])
            bucket[0] += entry["tottime"]
            bucket[1] += entry["calls"]
        total_self = sum(b[0] for b in buckets.values()) or 1.0
        print(f"  {'stage':<34} {'self s':>9} {'share':>8} {'calls':>15}")
        print("  " + "-" * 68)
        for stage, (self_seconds, calls) in sorted(
            buckets.items(), key=lambda i: -i[1][0]
        ):
            print(f"  {stage:<34} {self_seconds:>9.2f} "
                  f"{self_seconds / total_self * 100:>7.1f}% {int(calls):>15,}")

    shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
