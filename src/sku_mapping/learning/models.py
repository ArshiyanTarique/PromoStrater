"""Typed models for the Phase 7A learning store."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LabelQuality(str, Enum):
    """Governed trust levels for prospective labels."""

    GOLD = "GOLD"
    SILVER = "SILVER"
    PSEUDO = "PSEUDO"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ReviewQuestion:
    """One persisted True/False human-review question."""

    review_id: str
    session_id: str
    run_id: str
    question_number: int
    offer_id: str
    offer_description: str
    suggested_candidate_id: str
    suggested_candidate_description: str
    selection_category: str
    selection_reason: str
    fallback_reason: str | None
    supplied_candidates: tuple[tuple[str, str], ...]
    question_text: str = "Is this suggested SKU match correct?"


@dataclass(frozen=True)
class HumanReviewAnswer:
    """Validated reviewer response.

    A False response is complete only when exactly one corrective outcome is
    supplied: a listed candidate, none of the listed candidates, or an
    explicit inability to determine.
    """

    is_correct: bool
    reviewer_id: str | None = None
    corrected_candidate_id: str | None = None
    none_of_candidates: bool = False
    cannot_determine: bool = False
    notes: str | None = None
    review_source: str = "POST_UPLOAD_FIVE_QUESTION"
    decomposition_action: str | None = None
    corrected_entity_text: str | None = None
    corrected_attributes_json: str | None = None
