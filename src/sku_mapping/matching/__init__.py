"""Pre-ML candidate ranking and legacy rule evaluation."""

from sku_mapping.matching.candidate_generator import (
    CandidateGenerator,
    CandidateMatch,
    generate_best_candidate,
    generate_candidates_batch,
    generate_top_candidates,
)
from sku_mapping.matching.matcher import match_preprocessed_offers

__all__ = [
    "CandidateGenerator",
    "CandidateMatch",
    "generate_best_candidate",
    "generate_candidates_batch",
    "generate_top_candidates",
    "match_preprocessed_offers",
]
