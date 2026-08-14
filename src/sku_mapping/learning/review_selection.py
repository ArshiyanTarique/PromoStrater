"""Deterministic selection of up to five post-upload review questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class SelectedReview:
    """A selected prediction and its auditable selection rationale."""

    prediction: dict[str, Any]
    category: str
    reason: str
    fallback_reason: str | None = None


def _probability(row: dict[str, Any]) -> float:
    value = row.get("lightgbm_probability")
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _has_conflict(row: dict[str, Any]) -> bool:
    flags = str(row.get("conflict_flags_json") or "").strip()
    return flags not in {"", "[]", "{}", "null", "None"}


def _same_top(row: dict[str, Any]) -> bool:
    return str(row.get("agreement_status", "")).upper() in {
        "SAFE_AGREEMENT",
        "WEAK_AGREEMENT",
    }


def _category_specs(
    threshold: float,
) -> list[
    tuple[
        str,
        Callable[[dict[str, Any]], bool],
        Callable[[dict[str, Any]], tuple[Any, ...]],
    ]
]:
    return [
        (
            "HIGH_CONFIDENCE_AUTO_ACCEPT",
            lambda row: (
                str(row.get("final_decision", "")).upper() == "AUTO_ACCEPT"
                and _probability(row) >= threshold
            ),
            lambda row: (-_probability(row), str(row["offer_id"])),
        ),
        (
            "NEAR_AUTO_ACCEPT_THRESHOLD",
            lambda row: _probability(row) >= 0,
            lambda row: (
                abs(_probability(row) - threshold),
                str(row["offer_id"]),
            ),
        ),
        (
            "LIGHTGBM_EMBEDDING_DISAGREEMENT",
            lambda row: (
                str(row.get("agreement_status", "")).upper()
                == "DISAGREEMENT"
            ),
            lambda row: (-_probability(row), str(row["offer_id"])),
        ),
        (
            "LLM_REVIEWED",
            lambda row: bool(str(row.get("llm_decision") or "").strip()),
            lambda row: (-_probability(row), str(row["offer_id"])),
        ),
        (
            "DIFFICULT_OR_CONFLICT_PRONE",
            lambda row: _has_conflict(row)
            or str(row.get("final_decision", "")).upper()
            in {"MANUAL_REVIEW", "MODEL_ERROR"},
            lambda row: (
                0 if _has_conflict(row) else 1,
                _probability(row),
                str(row["offer_id"]),
            ),
        ),
    ]


def select_five_reviews(
    offer_predictions: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.85,
    question_count: int = 5,
) -> list[SelectedReview]:
    """Choose up to ``question_count`` unique reviewable offers.

    A run with fewer than five own-brand proposals is still useful to a
    reviewer. The configured count is therefore a maximum, not a minimum.
    """
    rows = [
        dict(row)
        for row in offer_predictions
        if str(row.get("offer_id") or "").strip()
        and str(row.get("candidate_id") or "").strip()
    ]
    if question_count < 1:
        raise ValueError("question_count must be positive")
    by_offer: dict[str, dict[str, Any]] = {}
    for row in sorted(
        rows,
        key=lambda value: (
            str(value["offer_id"]),
            0
            if str(value.get("final_decision", "")).upper()
            != "CANDIDATE_NOT_SELECTED"
            else 1,
            -_probability(value),
            int(value.get("candidate_rank") or 10**9),
            str(value.get("candidate_id") or ""),
        ),
    ):
        by_offer.setdefault(str(row["offer_id"]), row)
    pool = list(by_offer.values())
    if not pool:
        return []

    selected: list[SelectedReview] = []
    used: set[str] = set()
    specs = _category_specs(threshold)
    target_count = min(question_count, len(pool))
    for slot_index, (category, predicate, sorter) in enumerate(
        specs[:target_count]
    ):
        candidates = [
            row
            for row in pool
            if str(row["offer_id"]) not in used and predicate(row)
        ]
        fallback_reason: str | None = None
        effective_category = category
        if candidates:
            chosen = sorted(candidates, key=sorter)[0]
        else:
            remaining = [
                row for row in pool if str(row["offer_id"]) not in used
            ]
            if not remaining:
                break
            chosen = None
            fallback_specs = (
                specs[slot_index + 1 :] + specs[:slot_index]
            )
            for (
                fallback_category,
                fallback_predicate,
                fallback_sorter,
            ) in fallback_specs:
                fallback = [
                    row for row in remaining if fallback_predicate(row)
                ]
                if fallback:
                    chosen = sorted(fallback, key=fallback_sorter)[0]
                    effective_category = fallback_category
                    break
            if chosen is None:
                chosen = sorted(
                    remaining, key=lambda row: str(row["offer_id"])
                )[0]
                effective_category = "DETERMINISTIC_REMAINDER"
            fallback_reason = (
                f"NO_ELIGIBLE_{category};FILLED_FROM_{effective_category}"
            )
        used.add(str(chosen["offer_id"]))
        selected.append(
            SelectedReview(
                prediction=chosen,
                category=effective_category,
                reason=(
                    f"TARGETED_SLOT_{len(selected) + 1}:"
                    f"{category}"
                ),
                fallback_reason=fallback_reason,
            )
        )
    return selected
