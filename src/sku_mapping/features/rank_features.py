"""Group-relative ranking features for the ranked-v5 model.

A similarity score means something different when it is the best available
versus when it is barely ahead of a near-tie. These features express that
context by describing each candidate RELATIVE to the other candidates
retrieved for the same offer.

This is the canonical production implementation - training (rank_lab.py)
imports _rank_features from here to guarantee that training and inference
use identical code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Columns for which relative features are computed.
RANK_SOURCE_COLUMNS: tuple[str, ...] = (
    "word_similarity",
    "character_similarity",
    "token_similarity",
    "size_ratio",
)

#: Suffix pattern for all generated rank columns.
RANK_COLUMN_SUFFIXES: tuple[str, ...] = (
    "__rank",
    "__minus_max",
    "__is_best",
    "__z",
)


def add_rank_features(block: pd.DataFrame) -> pd.DataFrame:
    """Return *block* with group-relative columns appended in-place.

    The block must contain all candidate rows for a SINGLE offer - mixing
    rows from different offers produces meaningless relative scores.

    Columns ``word_similarity``, ``character_similarity``,
    ``token_similarity``, and ``size_ratio`` are used as sources.
    Missing source columns are silently skipped so the function is safe to
    call on feature frames that predate the extra-feature columns.

    Returns the same DataFrame (columns added in-place) for efficiency.
    """
    for col in RANK_SOURCE_COLUMNS:
        if col not in block.columns:
            continue
        v = block[col].to_numpy(dtype=float)
        mx = v.max() if len(v) else 0.0
        order = (-v).argsort().argsort()          # rank 0 = best
        std = v.std()
        block[f"{col}__rank"] = order.astype(float)
        block[f"{col}__minus_max"] = v - mx
        block[f"{col}__is_best"] = (v >= mx - 1e-9).astype(float)
        block[f"{col}__z"] = (
            (v - v.mean()) / std if std > 1e-9 else np.zeros_like(v)
        )
    return block


def build_rank_feature_frame(
    base_features: list[dict[str, float | int | None]],
) -> pd.DataFrame:
    """Convert a list of per-candidate base-feature dicts to a scored frame.

    ``base_features`` is the output of ``build_feature_vector`` /
    ``build_extra_features`` for every candidate in a shortlist, in the
    same order as the shortlist.  The returned DataFrame has all base
    columns plus the ``__rank`` / ``__minus_max`` / ``__is_best`` / ``__z``
    columns ready for the ranked-v5 model.
    """
    frame = pd.DataFrame(base_features).astype(float).fillna(-1.0)
    add_rank_features(frame)
    return frame
