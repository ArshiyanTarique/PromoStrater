"""Cumulative offer state so a weekly load only pays for what is new.

The pipeline is expensive per offer and cross-sectional per Master SKU. Those
two facts pull in opposite directions and decide the whole design:

* Inference is **incremental**. An offer that has already been mapped, and
  whose content has not changed, does not go through candidate generation,
  scoring or LLM review again.
* Outputs are **cumulative**. Competitor discovery answers "who competes with
  this Master SKU", which is a question about every offer ever seen, not about
  this week's file. Exporting a delta would produce a competitor list that
  looks complete and is not.

So each run reassembles the full picture from stored state plus the delta it
just computed, and the export layer never learns that incremental loading
exists.

What decides whether an offer is new is its canonical identity plus a content
hash - never a date. ClickFlyer dumps carry backdated rows and corrected
prices, and a date watermark drops both silently and permanently. The latest
offer end date is still recorded, but only so an operator can be told how
current the data is.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

import os
import tempfile

#: One file per cumulative frame, kept out of SQLite because these hold the
#: entire offer history - the source dumps alone run to tens of megabytes -
#: and they are always read and written whole.
#:
#: Pickle rather than Parquet, deliberately. Prepared offers carry non-scalar
#: derived features (``offer_measures_detailed`` holds nested measure
#: objects), which Parquet cannot encode without a lossy type round-trip.
#: That round-trip would change the very column values the content hashes are
#: computed over, so an unchanged offer would read as revised on every load
#: and incremental loading would quietly degrade into full reprocessing.
#: These files are local, single-writer state, never an interchange format.
POOL_FILE = "canonical_offers.pkl"
OWN_ROWS_FILE = "own_offer_rows.pkl"
DECISIONS_FILE = "decisions.pkl"

#: Columns excluded from an offer's content hash because they describe the
#: load rather than the offer. ``source_row_count`` varies with how many
#: variant rows a dump happened to carry; the identity column is the key the
#: hash is stored against.
_NON_CONTENT_COLUMNS = frozenset(
    {"offer_group_id", "source_row_count", "run_id"}
)


@dataclass(frozen=True)
class CumulativeState:
    """Everything previous production runs established."""

    pool: pd.DataFrame | None
    own_rows: pd.DataFrame | None
    decisions: pd.DataFrame | None

    @property
    def is_empty(self) -> bool:
        return self.own_rows is None or self.own_rows.empty


@dataclass(frozen=True)
class IncrementalPlan:
    """Which offers this run must actually push through inference."""

    offers: pd.DataFrame
    content_hashes: Mapping[str, str]
    master_hash: str
    new_offer_ids: frozenset[str]
    revised_offer_ids: frozenset[str]
    remap_offer_ids: frozenset[str]
    skipped_offer_count: int

    @property
    def processed_offer_ids(self) -> frozenset[str]:
        """Identities whose stored results this run replaces."""
        return frozenset(
            self.new_offer_ids | self.revised_offer_ids | self.remap_offer_ids
        )

    def as_diagnostics(self) -> dict[str, object]:
        return {
            "incremental_new_offers": len(self.new_offer_ids),
            "incremental_revised_offers": len(self.revised_offer_ids),
            "incremental_remapped_offers": len(self.remap_offer_ids),
            "incremental_skipped_offers": self.skipped_offer_count,
            "incremental_inference_offers": len(self.offers),
        }


def _hash_frame_rows(frame: pd.DataFrame) -> pd.Series:
    """Return a stable per-row hash over the frame's content columns.

    ``hash_pandas_object`` is used rather than a per-row SHA-256 because this
    runs over the whole dump on every load. It is a 64-bit hash: across a
    million offers the chance of any collision is about three in a hundred
    million, and a collision's only effect is that one revised offer is not
    re-inferred. That is the same failure a stale row would cause, at a rate
    far below the rate at which dumps themselves disagree.
    """
    content_columns = sorted(
        column for column in frame.columns
        if column not in _NON_CONTENT_COLUMNS
    )
    if not content_columns:
        raise ValueError("Offer frame has no content columns to hash")
    subject = frame[content_columns].astype("string").fillna("")
    return (
        pd.util.hash_pandas_object(subject, index=False)
        .astype("uint64")
        .map(lambda value: format(int(value), "016x"))
    )


def offer_content_hashes(canonical_offers: pd.DataFrame) -> dict[str, str]:
    """Map each canonical offer identity to a hash of its content."""
    if canonical_offers.empty:
        return {}
    if "offer_group_id" not in canonical_offers.columns:
        raise ValueError("Canonical offers require offer_group_id")
    hashes = _hash_frame_rows(canonical_offers)
    identities = canonical_offers["offer_group_id"].astype(str)
    return dict(zip(identities, hashes, strict=True))


def master_fingerprint(product_master: pd.DataFrame) -> str:
    """Fingerprint the Product Master a run was evaluated against.

    New Master SKUs are the only reason a previously unmatched offer deserves
    another attempt. Without this, an offer that found no match would stay
    unmatched forever, however much the master list grew.
    """
    if product_master.empty:
        return "empty"
    ordered = product_master.sort_index(axis=1).astype("string").fillna("")
    row_hashes = pd.util.hash_pandas_object(ordered, index=False)
    digest = hashlib.sha256()
    # Sorted, so reordering the master file is not mistaken for changing it -
    # that would strand every unmatched offer in a permanent re-map loop.
    for value in sorted(int(item) for item in row_hashes):
        digest.update(value.to_bytes(8, "big"))
    return digest.hexdigest()[:16]


def replace_by_offer(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    *,
    key: str,
    replaced_ids: Iterable[str],
) -> pd.DataFrame:
    """Merge stored rows with fresh ones, replacing whole offers at a time.

    Replacement is by offer identity rather than row, because a revised offer
    may carry a different number of variant rows than the version already
    stored. Dropping every stored row for a replaced identity before adding
    the new ones is what keeps a shrunken variant set from leaving orphans
    behind.
    """
    replaced = {str(value) for value in replaced_ids}
    if previous is None or previous.empty:
        return current.reset_index(drop=True)
    if current.empty and not replaced:
        return previous.reset_index(drop=True)
    retained = previous.loc[~previous[key].astype(str).isin(replaced)]
    if current.empty:
        return retained.reset_index(drop=True)
    return pd.concat([retained, current], ignore_index=True, sort=False)


def plan_incremental_inference(
    *,
    canonical_offers: pd.DataFrame,
    own_offers: pd.DataFrame,
    known_content_hashes: Mapping[str, str],
    unmapped_offer_ids: Iterable[str],
    master_hash: str,
) -> IncrementalPlan:
    """Decide which own-brand offers this run must infer.

    Three reasons an offer is processed, and no others:

    * it has never been seen;
    * its content changed under an unchanged identity, which is what a
      corrected price looks like;
    * it was seen, never matched, and the Product Master has since changed.
    """
    hashes = offer_content_hashes(canonical_offers)
    own_identities = own_offers["offer_group_id"].astype(str)
    remap_pool = {str(value) for value in unmapped_offer_ids}

    new_ids: set[str] = set()
    revised_ids: set[str] = set()
    remap_ids: set[str] = set()
    for identity in own_identities:
        known = known_content_hashes.get(identity)
        if known is None:
            new_ids.add(identity)
        elif known != hashes.get(identity):
            revised_ids.add(identity)
        elif identity in remap_pool:
            remap_ids.add(identity)

    selected = new_ids | revised_ids | remap_ids
    offers = own_offers.loc[own_identities.isin(selected)].reset_index(
        drop=True
    )
    return IncrementalPlan(
        offers=offers,
        content_hashes=hashes,
        master_hash=master_hash,
        new_offer_ids=frozenset(new_ids),
        revised_offer_ids=frozenset(revised_ids),
        remap_offer_ids=frozenset(remap_ids),
        skipped_offer_count=int(len(own_identities) - len(selected)),
    )


class IncrementalStateStore:
    """Read and write the cumulative frames as one consistent set."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _read(self, name: str) -> pd.DataFrame | None:
        path = self.directory / name
        if not path.is_file():
            return None
        return pd.read_pickle(path)

    @staticmethod
    def _write_atomic(frame: pd.DataFrame, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_pickle(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load(self) -> CumulativeState:
        """Return stored state, or an empty state on a first run."""
        return CumulativeState(
            pool=self._read(POOL_FILE),
            own_rows=self._read(OWN_ROWS_FILE),
            decisions=self._read(DECISIONS_FILE),
        )

    def save(
        self,
        *,
        pool: pd.DataFrame,
        own_rows: pd.DataFrame,
        decisions: pd.DataFrame,
    ) -> dict[str, Path]:
        """Persist the three cumulative frames.

        Each file is replaced atomically, but the set of three is not written
        under a single transaction. A crash between them leaves state that is
        internally consistent per file and at worst re-infers offers whose
        decisions did not land - the safe direction, since the ledger is only
        advanced after this returns.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        return {
            "pool": self._write_atomic(pool, self.directory / POOL_FILE),
            "own_rows": self._write_atomic(
                own_rows, self.directory / OWN_ROWS_FILE
            ),
            "decisions": self._write_atomic(
                decisions, self.directory / DECISIONS_FILE
            ),
        }

    def clear(self) -> None:
        """Remove stored state so the next run reprocesses in full."""
        for name in (POOL_FILE, OWN_ROWS_FILE, DECISIONS_FILE):
            (self.directory / name).unlink(missing_ok=True)
