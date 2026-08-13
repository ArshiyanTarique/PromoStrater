"""Chunked inference execution must reproduce the single-pass results exactly.

The chunked path changes only *how* the shadow run is executed. Every artifact
it produces has to match the legacy all-at-once path: the same rows, in the same
order, with the same identities. These tests run both paths in one process over
the same inputs, holding the run id and the clock fixed so nothing time-derived
can mask a real difference.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.llm_review import reviewer as llm_reviewer
from sku_mapping.shadow import pipeline as shadow_pipeline
from sku_mapping.shadow.pipeline import (
    _frame_fingerprint,
    _plan_chunk_edges,
    run_shadow_observation,
)
from sku_mapping.shadow.predictor import RegisteredShadowPackage

FIXED_RUN_ID = "shadow-equivalence-fixture"
FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc)

#: Production configuration bounds chunk_size to 10k-25k. A fixture that large
#: would make these tests take minutes, so the tests write the field directly to
#: force several chunks over a handful of offers. The chunking mechanism itself
#: is size-agnostic; only the configured bound is opinionated.
#: 1 puts every offer in its own chunk, which maximises the number of chunk
#: boundaries and therefore the chance of per-chunk state leaking. 5 leaves a
#: remainder of 4, and 9 is the whole fixture in one chunk - the streaming code
#: path exercised against the same input the single-pass path sees.
TEST_CHUNK_SIZES = (1, 2, 3, 4, 5, 9)


class _FakePredictor:
    """Deterministic scores that vary by row, independent of chunking.

    The score depends on the row's *content*, not its position in the frame, so
    a chunked run and a single-pass run must agree. A position-dependent stub
    would hide exactly the bug these tests exist to catch.
    """

    def predict_raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                (abs(hash(tuple(row))) % 1000) / 250.0 - 2.0
                for row in frame.fillna(0).round(4).to_numpy().tolist()
            ]
        )

    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                (abs(hash(tuple(row))) % 997) / 1000.0
                for row in frame.fillna(0).round(4).to_numpy().tolist()
            ]
        )


def _offers(count: int, *, with_offer_ids: bool = True) -> pd.DataFrame:
    products = [
        ("Chicken Nuggets-Frozen", "Original", "400g"),
        ("Chicken Strips-Frozen", "Spicy", "400g"),
        ("Beef Burger-Frozen", "Classic", "500g"),
        ("Chicken Popcorn-Frozen", "Hot", "250g"),
    ]
    records = []
    for index in range(count):
        product, variant, size = products[index % len(products)]
        record = {
            "Offer Name": f"Al Kabeer {product.split('-')[0]} {size} #{index}",
            "Product": product,
            "Brand Name": "Al Kabeer",
            "Variant": variant,
            "Base Packsize": size,
            "Retailer Name": f"Retailer {index % 3}",
        }
        if with_offer_ids:
            record["offerid"] = f"offer-{index:04d}"
        records.append(record)
    offers = preprocess_clickflyer(pd.DataFrame(records))
    offers["ml_decision"] = [
        "AUTO_MATCH" if index % 2 else "MANUAL_REVIEW"
        for index in range(len(offers))
    ]
    offers["confidence_tier"] = [
        "high (ml)" if index % 2 else "medium (ml)"
        for index in range(len(offers))
    ]
    offers["matched_itemcode"] = [
        "001" if index % 2 else "REVIEW_REQUIRED"
        for index in range(len(offers))
    ]
    offers["suggested_itemcode"] = [
        "001" if index % 2 else "002" for index in range(len(offers))
    ]
    return offers


def _master() -> pd.DataFrame:
    return preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "001",
                    "Itemname": "CHICKEN NUGGETS",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Nuggets",
                    "Item Description": "Original chicken nuggets",
                    "Item-Spec": "400g",
                },
                {
                    "Itemcode": "002",
                    "Itemname": "CHICKEN STRIPS",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Strips",
                    "Item Description": "Spicy chicken strips",
                    "Item-Spec": "400g",
                },
                {
                    "Itemcode": "003",
                    "Itemname": "BEEF BURGER",
                    "Item-Cat-2": "Beef",
                    "Item-Cat-4": "Burger",
                    "Item Description": "Classic beef burger",
                    "Item-Spec": "500g",
                },
                {
                    "Itemcode": "004",
                    "Itemname": "CHICKEN POPCORN",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Popcorn",
                    "Item Description": "Hot chicken popcorn",
                    "Item-Spec": "250g",
                },
            ]
        )
    )


def _registered(tmp_path) -> RegisteredShadowPackage:
    path = tmp_path / "equivalence-v3.joblib"
    path.write_bytes(b"chunked-equivalence-package")
    package = {
        "model_id": "equivalence-v3",
        "predictor": _FakePredictor(),
        "feature_columns": list(shadow_pipeline.MODEL_FEATURE_COLUMNS),
        "auto_match_threshold": 0.9,
        "manual_review_threshold": 0.1,
    }
    return RegisteredShadowPackage(
        package=package,
        registry_entry={"model_id": "equivalence-v3"},
        package_path=path,
        package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _config(tmp_path, chunk_size: int, label: str):
    config = load_config("config/default.yaml")
    shadow = replace(
        config.shadow_mode,
        enabled=True,
        model_id="equivalence-v3",
        package_reference=None,
        # Each variant writes to its own tree so the run id can stay fixed;
        # the run id is stamped into every candidate row.
        output_directory=tmp_path / label / "shadow",
        challenge_set_directory=tmp_path / label / "challenge_sets",
        top_k=3,
        sampling_counts={
            key: 1 for key in config.shadow_mode.sampling_counts
        },
    )
    object.__setattr__(shadow, "chunk_size", chunk_size)
    embedding = replace(
        config.embedding,
        backend="local_hashing",
        cache_path=tmp_path / label / "embedding.sqlite3",
    )
    llm_review = replace(
        config.llm_review, cache_path=tmp_path / label / "llm.sqlite3"
    )
    return replace(
        config,
        shadow_mode=shadow,
        embedding=embedding,
        llm_review=llm_review,
    )


class _FixedClock:
    """Freeze the wall clock so timestamps cannot differ between runs."""

    @staticmethod
    def now(tz=None):
        return FIXED_NOW


def _run(tmp_path, monkeypatch, *, chunk_size: int, label: str, offers, master):
    monkeypatch.setattr(shadow_pipeline, "datetime", _FixedClock)
    # The reviewer stamps each review with the wall clock at the moment it
    # ran, which is correct for an audit field and is exactly why it has to
    # be frozen here: two runs of the same input genuinely happen at
    # different times, so a real clock makes the artifacts differ for a
    # reason that has nothing to do with chunking. Freezing one clock and not
    # the other left this path unpinned - invisible until routing began
    # sending offers to review at all.
    monkeypatch.setattr(llm_reviewer, "datetime", _FixedClock)
    monkeypatch.setattr(
        shadow_pipeline,
        "load_registered_shadow_package",
        lambda **_: _registered(tmp_path),
    )
    return run_shadow_observation(
        offers,
        master,
        config=_config(tmp_path, chunk_size, label),
        shadow_run_id=FIXED_RUN_ID,
    )


def _predictions(result) -> pd.DataFrame:
    return pd.read_parquet(
        result.output_paths["shadow_predictions_parquet"]
    )


_SORT_KEYS = ["offer_group_id", "candidate_rank", "master_itemcode"]


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(_SORT_KEYS, kind="stable").reset_index(
        drop=True
    )


@pytest.mark.parametrize("chunk_size", TEST_CHUNK_SIZES)
def test_chunked_run_matches_single_pass_predictions(
    tmp_path, monkeypatch, chunk_size
):
    offers, master = _offers(9), _master()

    legacy = _run(
        tmp_path,
        monkeypatch,
        chunk_size=0,
        label="legacy",
        offers=offers,
        master=master,
    )
    chunked = _run(
        tmp_path,
        monkeypatch,
        chunk_size=chunk_size,
        label=f"chunked{chunk_size}",
        offers=offers,
        master=master,
    )

    legacy_frame = _predictions(legacy)
    chunked_frame = _predictions(chunked)

    assert legacy.prediction_rows == chunked.prediction_rows
    assert legacy.offer_groups == chunked.offer_groups
    assert (
        legacy.failed_shadow_predictions
        == chunked.failed_shadow_predictions
    )

    # Content equality, independent of row order.
    pd.testing.assert_frame_equal(
        _sorted(legacy_frame), _sorted(chunked_frame), check_exact=True
    )
    # Row ordering is part of the contract, so compare unsorted too.
    pd.testing.assert_frame_equal(
        legacy_frame, chunked_frame, check_exact=True
    )
    assert _frame_fingerprint(legacy_frame) == _frame_fingerprint(
        chunked_frame
    )


@pytest.mark.parametrize("chunk_size", TEST_CHUNK_SIZES)
@pytest.mark.parametrize(
    "artifact",
    [
        "human_review_template",
        "agreement_results",
        "llm_review_results",
        "monitoring_report",
        "sampling_report",
    ],
)
def test_chunked_run_produces_byte_identical_text_artifacts(
    tmp_path, monkeypatch, chunk_size, artifact
):
    offers, master = _offers(9), _master()

    legacy = _run(
        tmp_path,
        monkeypatch,
        chunk_size=0,
        label="legacy",
        offers=offers,
        master=master,
    )
    chunked = _run(
        tmp_path,
        monkeypatch,
        chunk_size=chunk_size,
        label=f"chunked{chunk_size}",
        offers=offers,
        master=master,
    )

    assert (
        legacy.output_paths[artifact].read_bytes()
        == chunked.output_paths[artifact].read_bytes()
    )


def test_chunking_preserves_identity_when_offerid_is_missing(
    tmp_path, monkeypatch
):
    """Offer identity falls back to the row's global position.

    ``canonical_offer_identity`` and ``_source_row_identifier`` both hash the
    row's integer position when ``offerid`` is absent. Assigning identities per
    chunk would mint different ids for the same offer, so this is the case that
    catches a chunk-local offset regression.
    """
    offers, master = _offers(9, with_offer_ids=False), _master()

    legacy = _run(
        tmp_path,
        monkeypatch,
        chunk_size=0,
        label="legacy",
        offers=offers,
        master=master,
    )
    chunked = _run(
        tmp_path,
        monkeypatch,
        chunk_size=3,
        label="chunked",
        offers=offers,
        master=master,
    )

    legacy_frame = _predictions(legacy)
    chunked_frame = _predictions(chunked)

    assert (
        legacy_frame["offer_group_id"].tolist()
        == chunked_frame["offer_group_id"].tolist()
    )
    assert (
        legacy_frame["source_row_identifier"].tolist()
        == chunked_frame["source_row_identifier"].tolist()
    )
    pd.testing.assert_frame_equal(
        legacy_frame, chunked_frame, check_exact=True
    )


def test_chunked_execution_is_deterministic(tmp_path, monkeypatch):
    offers, master = _offers(9), _master()

    first = _run(
        tmp_path,
        monkeypatch,
        chunk_size=3,
        label="first",
        offers=offers,
        master=master,
    )
    second = _run(
        tmp_path,
        monkeypatch,
        chunk_size=3,
        label="second",
        offers=offers,
        master=master,
    )

    assert _frame_fingerprint(_predictions(first)) == _frame_fingerprint(
        _predictions(second)
    )
    for artifact in ("agreement_results", "monitoring_report"):
        assert (
            first.output_paths[artifact].read_bytes()
            == second.output_paths[artifact].read_bytes()
        )


def test_chunked_run_does_not_mutate_production_inputs(
    tmp_path, monkeypatch
):
    offers, master = _offers(9), _master()
    offers_before = offers.copy(deep=True)
    master_before = master.copy(deep=True)

    _run(
        tmp_path,
        monkeypatch,
        chunk_size=3,
        label="chunked",
        offers=offers,
        master=master,
    )

    pd.testing.assert_frame_equal(offers, offers_before, check_exact=True)
    pd.testing.assert_frame_equal(master, master_before, check_exact=True)


def test_streaming_path_spools_chunks_and_cleans_up(tmp_path, monkeypatch):
    """The streaming writers must actually run, and leave no spool behind.

    Without the call assertions this would pass even if the run silently fell
    back to the single-pass path, which would make every equality test above
    vacuous.
    """
    offers, master = _offers(9), _master()
    spooled: list[int] = []
    streamed_csv: list[int] = []
    atomic_parquet: list[int] = []

    original_stream_parquet = shadow_pipeline._stream_parquet
    original_stream_csv = shadow_pipeline._stream_csv
    original_atomic_parquet = shadow_pipeline._atomic_parquet

    def _record_parquet(sources, destination):
        spooled.append(len(sources))
        return original_stream_parquet(sources, destination)

    def _record_csv(sources, destination):
        streamed_csv.append(len(sources))
        return original_stream_csv(sources, destination)

    def _record_atomic(frame, destination):
        atomic_parquet.append(len(frame))
        return original_atomic_parquet(frame, destination)

    monkeypatch.setattr(
        shadow_pipeline, "_stream_parquet", _record_parquet
    )
    monkeypatch.setattr(shadow_pipeline, "_stream_csv", _record_csv)
    monkeypatch.setattr(
        shadow_pipeline, "_atomic_parquet", _record_atomic
    )

    result = _run(
        tmp_path,
        monkeypatch,
        chunk_size=3,
        label="chunked",
        offers=offers,
        master=master,
    )

    assert spooled == [3], "expected three spooled chunks for 9 offers"
    assert streamed_csv == [3]
    assert atomic_parquet == [], "streaming run must not write in one pass"

    run_directory = result.output_paths[
        "shadow_predictions_parquet"
    ].parent
    assert not (
        run_directory / shadow_pipeline._SPOOL_DIRECTORY_NAME
    ).exists()


def test_single_pass_path_does_not_spool(tmp_path, monkeypatch):
    offers, master = _offers(9), _master()
    streamed: list[int] = []
    monkeypatch.setattr(
        shadow_pipeline,
        "_stream_parquet",
        lambda sources, destination: streamed.append(len(sources)),
    )

    _run(
        tmp_path,
        monkeypatch,
        chunk_size=0,
        label="legacy",
        offers=offers,
        master=master,
    )

    assert streamed == []


def test_chunk_edges_never_split_a_source_offer():
    """A source offer can expand into several entity rows.

    ``finalize_unified_decisions`` aggregates by ``source_offer_id``, so a
    boundary drawn through one of those groups would change its aggregate.
    """
    own = pd.DataFrame(
        {"source_offer_id": ["a", "a", "a", "b", "c", "c", "d", "e"]}
    )
    edges = _plan_chunk_edges(own, 2)

    assert edges[0][0] == 0
    assert edges[-1][1] == len(own)
    for start, stop in edges:
        assert start < stop
    # Contiguous, non-overlapping cover.
    for earlier, later in zip(edges, edges[1:]):
        assert earlier[1] == later[0]
    # No group straddles a boundary.
    for _, stop in edges[:-1]:
        assert own["source_offer_id"].iloc[stop - 1] != own[
            "source_offer_id"
        ].iloc[stop]


def test_chunk_edges_degrade_to_a_single_slice():
    own = pd.DataFrame({"source_offer_id": ["a", "b", "c"]})

    assert _plan_chunk_edges(own, 0) == [(0, 3)]
    assert _plan_chunk_edges(own, 10) == [(0, 3)]
    assert _plan_chunk_edges(pd.DataFrame({"source_offer_id": []}), 5) == []
