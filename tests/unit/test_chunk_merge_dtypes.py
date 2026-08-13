"""Dtype and schema contracts for merging spooled chunks.

Three bugs lived here, all of the same family: a chunk that reviewed nothing
carries no rows but does carry dtypes, and those dtypes silently won. Each
produced identical values in a different type, which changed serialized bytes
without changing a single business result. These tests pin the contracts
directly rather than relying on the full-pipeline suite to notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from sku_mapping.shadow.pipeline import (
    ChunkSchemaConflictError,
    _align_to_schema,
    _merge_llm_results,
    _stream_parquet,
    _unified_spool_schema,
)


def _spool(frame: pd.DataFrame, directory: Path, index: int) -> Path:
    path = directory / f"chunk_{index:05d}.parquet"
    frame.to_parquet(path, index=False)
    return path


# -- A: mixed schemas across chunks ----------------------------------------
def test_chunk_without_reviews_merges_with_chunk_that_has_them(tmp_path):
    """Chunk 0 reviewed nothing; chunk 1 did. Both must survive."""
    unreviewed = pd.DataFrame(
        {"offer_id": ["a"], "score": [0.5], "llm_status": [None]}
    )
    reviewed = pd.DataFrame(
        {"offer_id": ["b"], "score": [0.9], "llm_status": ["ACCEPTED"]}
    )
    destination = tmp_path / "merged.parquet"
    _stream_parquet(
        [_spool(unreviewed, tmp_path, 0), _spool(reviewed, tmp_path, 1)],
        destination,
    )

    merged = pd.read_parquet(destination)
    assert list(merged["offer_id"]) == ["a", "b"]
    assert set(merged.columns) == {"offer_id", "score", "llm_status"}
    # Null stays null, value stays value: nothing dropped, nothing invented.
    assert pd.isna(merged.loc[0, "llm_status"])
    assert merged.loc[1, "llm_status"] == "ACCEPTED"


def test_column_order_follows_the_widest_chunk_not_the_first(tmp_path):
    """A column absent from chunk 0 must not be exiled to the end.

    First-seen ordering made the layout depend on which chunk introduced a
    column, so one-offer-per-chunk produced a different column order than a
    single-pass run over the same data.
    """
    narrow = pd.DataFrame({"offer_id": ["a"], "score": [0.5]})
    wide = pd.DataFrame({"offer_id": ["b"], "review": ["yes"], "score": [0.9]})
    destination = tmp_path / "merged.parquet"
    _stream_parquet(
        [_spool(narrow, tmp_path, 0), _spool(wide, tmp_path, 1)], destination
    )

    assert list(pd.read_parquet(destination).columns) == list(wide.columns)


def test_incompatible_concrete_types_raise_a_named_diagnostic(tmp_path):
    """An unpromotable conflict names the column and both chunk files."""
    first = pd.DataFrame({"offer_id": ["a"], "value": ["text"]})
    second = pd.DataFrame(
        {"offer_id": ["b"], "value": [pd.Timestamp("2026-01-01")]}
    )
    sources = [_spool(first, tmp_path, 0), _spool(second, tmp_path, 1)]

    with pytest.raises(ChunkSchemaConflictError) as error:
        _unified_spool_schema(sources)
    message = str(error.value)
    assert "value" in message
    assert "chunk_00000.parquet" in message
    assert "chunk_00001.parquet" in message


def test_missing_columns_are_added_as_typed_nulls(tmp_path):
    """Alignment fills an absent column rather than rejecting the table."""
    import pyarrow.parquet as pq

    narrow = pd.DataFrame({"offer_id": ["a"]})
    wide = pd.DataFrame({"offer_id": ["b"], "review": ["yes"]})
    sources = [_spool(narrow, tmp_path, 0), _spool(wide, tmp_path, 1)]
    schema = _unified_spool_schema(sources)

    aligned = _align_to_schema(pq.read_table(sources[0]), schema)
    assert aligned.schema.names == schema.names
    assert aligned.column("review").null_count == aligned.num_rows


# -- C: pandas metadata survives, so nullable dtypes reconstruct ------------
@pytest.mark.parametrize(
    "dtype, values",
    [("Int64", [None, None]), ("boolean", [None, None]), ("Int64", [1, None])],
)
def test_nullable_dtypes_survive_the_merge(tmp_path, dtype, values):
    """An all-null nullable column must not come back as float64.

    Arrow cannot distinguish an all-null Int64 column from a float column of
    NaNs; only the pandas metadata can. Building the unified schema without it
    silently downgraded the dtype. Parameterised because the behaviour is
    generic, not a property of one column name.
    """
    frame = pd.DataFrame(
        {"offer_id": ["a", "b"], "nullable": pd.array(values, dtype=dtype)}
    )
    single_pass = tmp_path / "single.parquet"
    frame.to_parquet(single_pass, index=False)

    merged_path = tmp_path / "merged.parquet"
    _stream_parquet(
        [
            _spool(frame.iloc[[0]], tmp_path, 0),
            _spool(frame.iloc[[1]], tmp_path, 1),
        ],
        merged_path,
    )

    expected = pd.read_parquet(single_pass)
    merged = pd.read_parquet(merged_path)
    assert str(merged["nullable"].dtype) == str(expected["nullable"].dtype)
    pd.testing.assert_frame_equal(merged, expected, check_exact=True)


# -- B: concat widening in the LLM review merge -----------------------------
@dataclass(frozen=True)
class _Result:
    """Stand-in carrying only what _merge_llm_results touches."""

    frame: pd.DataFrame
    status: str = "OK"
    error: str | None = None
    results: tuple = ()
    offers_routed: int = 0
    provider_calls: int = 0
    cache_hits: int = 0
    failures: int = 0


def test_empty_chunk_does_not_widen_integer_review_columns():
    """0 must stay 0, not become 0.0, because a chunk reviewed nothing."""
    reviewed = pd.DataFrame(
        {"offer_id": ["a"], "llm_retry_count": pd.array([0], dtype="int64")}
    )
    empty = reviewed.iloc[0:0].astype({"llm_retry_count": "float64"})

    merged = _merge_llm_results([_Result(empty), _Result(reviewed)])

    assert list(merged.frame["offer_id"]) == ["a"]
    assert (
        merged.frame["llm_retry_count"].dtype
        == reviewed["llm_retry_count"].dtype
    )
    assert merged.frame.to_csv(index=False) == reviewed.to_csv(index=False)


def test_all_empty_chunks_still_yield_the_columns():
    """An all-empty merge keeps the schema instead of collapsing."""
    empty = pd.DataFrame(
        {
            "offer_id": pd.Series(dtype="object"),
            "llm_retry_count": pd.Series(dtype="int64"),
        }
    )
    merged = _merge_llm_results([_Result(empty), _Result(empty)])
    assert list(merged.frame.columns) == ["offer_id", "llm_retry_count"]
    assert merged.frame.empty


def test_review_rows_from_several_chunks_are_all_retained():
    """Reviews spanning chunks accumulate; none is overwritten."""
    first = pd.DataFrame(
        {"offer_id": ["a"], "llm_retry_count": pd.array([0], dtype="int64")}
    )
    second = pd.DataFrame(
        {"offer_id": ["b"], "llm_retry_count": pd.array([2], dtype="int64")}
    )
    merged = _merge_llm_results(
        [_Result(first, offers_routed=1), _Result(second, offers_routed=1)]
    )
    assert list(merged.frame["offer_id"]) == ["a", "b"]
    assert merged.offers_routed == 2
