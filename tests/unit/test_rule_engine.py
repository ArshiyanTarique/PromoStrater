"""Tests for legacy pre-ML candidate confidence rules."""

from sku_mapping.matching.candidate_generator import CandidateMatch
from sku_mapping.matching.rule_engine import apply_pre_ml_rules


def _candidate(**overrides: object) -> CandidateMatch:
    values: dict[str, object] = {
        "itemcode": "SKU",
        "itemname": "Product",
        "text_score": 90.0,
        "adjusted_score": 94.0,
        "margin": 10.0,
        "raw_margin": 10.0,
        "pack_status": True,
        "pack_structure_status": True,
        "category": "Chicken",
        "candidate_rank": 1,
    }
    values.update(overrides)
    return CandidateMatch(**values)  # type: ignore[arg-type]


def test_high_and_low_margin_confidence_behavior() -> None:
    assert apply_pre_ml_rules(_candidate()).confidence_tier == "high"
    low_margin = apply_pre_ml_rules(_candidate(margin=0.0))
    assert low_margin.confidence_tier == "medium"


def test_other_category_requires_higher_score_and_margin() -> None:
    result = apply_pre_ml_rules(_candidate(category="Other", text_score=84.0, margin=20.0))
    assert result.itemcode == "NO_MATCH"
    assert result.confidence_tier == "no_match"
    result = apply_pre_ml_rules(_candidate(category="Other", text_score=90.0, margin=10.0))
    assert result.itemcode == "NO_MATCH"


def test_structure_and_pack_conflicts_are_preserved() -> None:
    structure = apply_pre_ml_rules(_candidate(pack_structure_status=False))
    assert structure.confidence_tier == "low_structure_conflict"
    incompatible = apply_pre_ml_rules(
        _candidate(pack_status=False, all_candidates_incompatible=True)
    )
    assert incompatible.confidence_tier == "low_pack_conflict"


def test_unknown_pack_can_be_medium_with_large_margin() -> None:
    result = apply_pre_ml_rules(_candidate(pack_status=None, margin=20.0))
    assert result.confidence_tier == "medium"


def test_existing_no_match_is_not_reclassified() -> None:
    result = apply_pre_ml_rules(
        _candidate(itemcode="NO_MATCH", itemname="None", confidence_tier="no_match_category")
    )
    assert result.confidence_tier == "no_match_category"
