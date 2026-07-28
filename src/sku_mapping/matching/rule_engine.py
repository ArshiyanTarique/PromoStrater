"""Legacy pre-ML confidence rules, separate from candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass, replace

from sku_mapping.matching.candidate_generator import CandidateMatch


@dataclass(frozen=True)
class CandidateRuleThresholds:
    """Existing Stage 2 candidate thresholds from the production script."""

    other_min_score: float = 85.0
    other_min_margin: float = 12.0
    normal_min_score: float = 55.0
    high_score: float = 80.0
    high_margin: float = 8.0
    medium_score: float = 65.0
    medium_margin: float = 6.0


def apply_pre_ml_rules(
    candidate: CandidateMatch,
    thresholds: CandidateRuleThresholds = CandidateRuleThresholds(),
) -> CandidateMatch:
    """Apply Stage 2 labels without making an ML probability decision."""
    if candidate.itemcode == "NO_MATCH":
        return candidate

    is_other = candidate.category == "Other"
    minimum_score = thresholds.other_min_score if is_other else thresholds.normal_min_score
    if candidate.text_score < minimum_score or (
        is_other and candidate.margin < thresholds.other_min_margin
    ):
        return replace(
            candidate,
            itemcode="NO_MATCH",
            itemname="None",
            confidence_tier="no_match",
            master_match_text="",
            master_measures=(),
        )

    if candidate.all_candidates_incompatible:
        confidence = "low_pack_conflict"
    elif candidate.pack_structure_status is False:
        confidence = "low_structure_conflict"
    elif candidate.pack_status is False:
        confidence = "low"
    elif (
        candidate.text_score >= thresholds.high_score
        and candidate.margin >= thresholds.high_margin
        and candidate.pack_status is True
    ):
        confidence = "high"
    elif candidate.text_score >= thresholds.medium_score and (
        candidate.pack_status is True or candidate.margin >= thresholds.medium_margin
    ):
        confidence = "medium"
    else:
        confidence = "low"
    return replace(candidate, confidence_tier=confidence)
