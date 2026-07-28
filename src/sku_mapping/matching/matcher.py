"""Pre-ML matcher composing candidate ranking and legacy rule labels."""

from __future__ import annotations

import pandas as pd

from sku_mapping.matching.candidate_generator import CandidateGenerator, CandidateMatch
from sku_mapping.matching.rule_engine import CandidateRuleThresholds, apply_pre_ml_rules


def match_preprocessed_offers(
    offers: pd.DataFrame,
    master: pd.DataFrame,
    thresholds: CandidateRuleThresholds = CandidateRuleThresholds(),
) -> list[CandidateMatch]:
    """Generate and label one pre-ML candidate per offer, preserving input order."""
    ranked = CandidateGenerator(master).generate_candidates_batch(offers, top_k=1)
    return [apply_pre_ml_rules(candidates[0], thresholds) for candidates in ranked]
