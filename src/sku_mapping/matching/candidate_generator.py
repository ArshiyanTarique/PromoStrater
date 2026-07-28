"""RapidFuzz batch candidate ranking, independent of ML decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from sku_mapping.features.measurement_features import (
    SimpleMeasure,
    pack_is_compatible,
    pack_structure_agrees,
)

PackStatus = bool | None


@dataclass(frozen=True)
class CandidateMatch:
    """A ranked offer/master candidate pair and its pre-ML diagnostics."""

    itemcode: str
    itemname: str
    text_score: float
    adjusted_score: float
    margin: float
    raw_margin: float
    pack_status: PackStatus
    pack_structure_status: PackStatus
    category: str
    candidate_rank: int
    all_candidates_incompatible: bool = False
    confidence_tier: str | None = None
    master_match_text: str = ""
    master_measures: tuple[SimpleMeasure, ...] = ()


REQUIRED_OFFER_COLUMNS = (
    "match_text",
    "offer_measures",
    "offer_measures_detailed",
    "category",
)
REQUIRED_MASTER_COLUMNS = (
    "Itemcode",
    "Itemname",
    "match_text",
    "master_measures",
    "master_measures_detailed",
    "category",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required preprocessed columns: {', '.join(missing)}")


def _no_match(category: str, confidence_tier: str) -> CandidateMatch:
    return CandidateMatch(
        itemcode="NO_MATCH",
        itemname="None",
        text_score=0.0,
        adjusted_score=0.0,
        margin=0.0,
        raw_margin=0.0,
        pack_status=None,
        pack_structure_status=None,
        category=category,
        candidate_rank=0,
        confidence_tier=confidence_tier,
    )


class CandidateGenerator:
    """Rank category-gated candidates with production RapidFuzz scoring rules."""

    def __init__(self, master: pd.DataFrame) -> None:
        _require_columns(master, REQUIRED_MASTER_COLUMNS, "master")
        self.master = master.reset_index(drop=True)
        self.master_by_category = {
            category: group.reset_index(drop=True)
            for category, group in self.master.groupby("category", sort=True)
        }

    def _pool_for_category(self, category: str) -> pd.DataFrame | None:
        pool = self.master_by_category.get(category)
        if pool is not None and not pool.empty:
            return pool
        if category == "Other" and not self.master.empty:
            return self.master
        return None

    @staticmethod
    def _rank_row(
        offer_row: pd.Series,
        pool: pd.DataFrame,
        scores: np.ndarray,
        top_k: int,
    ) -> list[CandidateMatch]:
        category = str(offer_row["category"])
        if len(str(offer_row["match_text"]).split()) < 2:
            return [_no_match(category, "no_match_empty_text")]

        pack_flags = np.array(
            [
                pack_is_compatible(offer_row["offer_measures"], measures)
                for measures in pool["master_measures"]
            ],
            dtype=object,
        )
        adjusted_scores = scores + np.where(
            pack_flags == True,  # noqa: E712 - preserves legacy tri-state comparison
            4,
            np.where(pack_flags == None, -3, 0),  # noqa: E711 - legacy tri-state comparison
        ).astype(float)
        known_incompatible = pack_flags == False  # noqa: E712 - legacy tri-state comparison
        all_incompatible = bool(known_incompatible.all())
        if (~known_incompatible).any():
            adjusted_scores = np.where(known_incompatible, -1e9, adjusted_scores)

        eligible = np.flatnonzero(adjusted_scores > -1e8)
        if not len(eligible):
            return [_no_match(category, "no_eligible_candidate")]

        # Preserve the legacy NumPy ordering used by the production pipeline.
        order = eligible[np.argsort(-adjusted_scores[eligible])]
        candidates: list[CandidateMatch] = []
        for rank, candidate_index in enumerate(order[:top_k], start=1):
            if rank < len(order):
                next_index = order[rank]
                margin = float(adjusted_scores[candidate_index] - adjusted_scores[next_index])
                raw_margin = float(scores[candidate_index] - scores[next_index])
            else:
                margin = 100.0
                raw_margin = 100.0
            master_row = pool.iloc[candidate_index]
            candidates.append(
                CandidateMatch(
                    itemcode=str(master_row["Itemcode"]),
                    itemname=str(master_row["Itemname"]),
                    text_score=round(float(scores[candidate_index]), 2),
                    adjusted_score=round(float(adjusted_scores[candidate_index]), 2),
                    margin=round(margin, 2),
                    raw_margin=round(raw_margin, 2),
                    pack_status=pack_flags[candidate_index],
                    pack_structure_status=pack_structure_agrees(
                        offer_row["offer_measures_detailed"],
                        master_row["master_measures_detailed"],
                    ),
                    category=category,
                    candidate_rank=rank,
                    all_candidates_incompatible=all_incompatible,
                    master_match_text=str(master_row["match_text"]),
                    master_measures=tuple(master_row["master_measures"]),
                )
            )
        return candidates

    def generate_candidates_batch(
        self,
        offers: pd.DataFrame,
        top_k: int = 5,
    ) -> list[list[CandidateMatch]]:
        """Rank candidates using one RapidFuzz score matrix per category."""
        _require_columns(offers, REQUIRED_OFFER_COLUMNS, "offers")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        results: list[list[CandidateMatch]] = [[] for _ in range(len(offers))]
        if offers.empty:
            return results

        indexed_offers = offers.reset_index(drop=True)
        for category, positions in indexed_offers.groupby("category", sort=True).groups.items():
            position_list = list(positions)
            subset = indexed_offers.iloc[position_list].reset_index(drop=True)
            pool = self._pool_for_category(str(category))
            if pool is None:
                for position in position_list:
                    results[position] = [_no_match(str(category), "no_match_category")]
                continue
            queries = subset["match_text"].tolist()
            choices = pool["match_text"].tolist()
            score_matrix = (
                process.cdist(queries, choices, scorer=fuzz.token_sort_ratio)
                + process.cdist(queries, choices, scorer=fuzz.token_set_ratio)
            ) / 2.0
            for row_position, (_, offer_row) in enumerate(subset.iterrows()):
                results[position_list[row_position]] = self._rank_row(
                    offer_row, pool, score_matrix[row_position].copy(), top_k
                )
        return results

    def generate_top_candidates(
        self,
        offer: pd.Series | dict[str, object],
        top_k: int = 5,
    ) -> list[CandidateMatch]:
        """Rank candidates for one already-preprocessed offer."""
        return self.generate_candidates_batch(pd.DataFrame([offer]), top_k=top_k)[0]

    def generate_best_candidate(
        self,
        offer: pd.Series | dict[str, object],
    ) -> CandidateMatch:
        """Return the highest-ranked candidate without applying confidence rules."""
        return self.generate_top_candidates(offer, top_k=1)[0]


def generate_candidates_batch(
    offers: pd.DataFrame,
    master: pd.DataFrame,
    top_k: int = 5,
) -> list[list[CandidateMatch]]:
    """Convenience wrapper for batch candidate ranking."""
    return CandidateGenerator(master).generate_candidates_batch(offers, top_k=top_k)


def generate_top_candidates(
    offer: pd.Series | dict[str, object],
    master: pd.DataFrame,
    top_k: int = 5,
) -> list[CandidateMatch]:
    """Convenience wrapper for one-off candidate ranking."""
    return CandidateGenerator(master).generate_top_candidates(offer, top_k=top_k)


def generate_best_candidate(
    offer: pd.Series | dict[str, object],
    master: pd.DataFrame,
) -> CandidateMatch:
    """Convenience wrapper for one-off top candidate ranking."""
    return CandidateGenerator(master).generate_best_candidate(offer)
