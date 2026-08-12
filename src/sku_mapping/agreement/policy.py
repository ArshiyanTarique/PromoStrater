"""Conservative agreement policy over LightGBM and embedding rankings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from sku_mapping.config import AgreementConfig
from sku_mapping.constants import (
    AgreementReasonCode,
    AgreementStatus,
    ReviewRoute,
)

REQUIRED_COLUMNS = frozenset(
    {
        "offer_group_id",
        "master_itemcode",
        "candidate_rank",
        "calibrated_probability",
    }
)

HARD_CONFLICT_NAMES = (
    "protein_conflict",
    "mixed_protein_ambiguity",
    "strong_family_conflict",
    "strong_size_weight_conflict",
    "strong_pack_format_conflict",
    "feature_generation_failure",
    "missing_master",
    "commercial_hard_conflict",
)


@dataclass(frozen=True)
class AgreementResult:
    """One explicit agreement and routing decision for one offer."""

    offer_id: str
    lightgbm_top_candidate: str | None
    lightgbm_calibrated_probability: float | None
    lightgbm_top_rank: int | None
    lightgbm_candidate_rank: int | None
    lightgbm_score_margin: float | None
    embedding_top_candidate: str | None
    embedding_similarity: float | None
    embedding_rank: int | None
    embedding_candidate_rank: int | None
    embedding_score_margin: float | None
    same_top_candidate: bool
    candidate_generation_margin: float | None
    candidate_generation_raw_margin: float | None
    conflict_flags: Mapping[str, bool]
    agreement_status: AgreementStatus
    routing_decision: ReviewRoute
    reason_codes: tuple[AgreementReasonCode, ...]
    routing_reason: str

    def to_record(self) -> dict[str, Any]:
        """Return a stable tabular representation."""
        return {
            "offer_id": self.offer_id,
            "lightgbm_top_candidate": self.lightgbm_top_candidate,
            "lightgbm_calibrated_probability": (
                self.lightgbm_calibrated_probability
            ),
            "lightgbm_top_rank": self.lightgbm_top_rank,
            "lightgbm_candidate_rank": self.lightgbm_candidate_rank,
            "lightgbm_score_margin": self.lightgbm_score_margin,
            "embedding_top_candidate": self.embedding_top_candidate,
            "embedding_similarity": self.embedding_similarity,
            "embedding_rank": self.embedding_rank,
            "embedding_candidate_rank": self.embedding_candidate_rank,
            "embedding_score_margin": self.embedding_score_margin,
            "same_top_candidate": self.same_top_candidate,
            "candidate_generation_margin": self.candidate_generation_margin,
            "candidate_generation_raw_margin": (
                self.candidate_generation_raw_margin
            ),
            "conflict_flags": json.dumps(
                dict(self.conflict_flags), sort_keys=True
            ),
            **dict(self.conflict_flags),
            "agreement_status": self.agreement_status.value,
            "routing_decision": self.routing_decision.value,
            "reason_codes": "|".join(
                reason.value for reason in self.reason_codes
            ),
            "routing_reason": self.routing_reason,
        }


@dataclass(frozen=True)
class AgreementEvaluationResult:
    """All offer-level agreement results for one candidate batch."""

    results: tuple[AgreementResult, ...]
    frame: pd.DataFrame


def _valid_scores(values: pd.Series, *, probability: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = np.isfinite(numeric)
    if probability:
        valid &= numeric.between(0.0, 1.0)
    else:
        valid &= numeric.between(-1.0, 1.0)
    return valid


@dataclass(frozen=True)
class _Ordered:
    """A candidate group's ranking, without materialising a ranked frame.

    The previous implementation copied and sorted the whole candidate group -
    roughly ninety columns wide for a five-row group - twice per offer, then
    materialised rows out of it repeatedly. Everything downstream only ever
    needs the top row, the runner-up's score, and the group size, so the sort
    is done on the three key columns alone and the result is carried as
    positions into the original group.

    Ranking is unchanged: the same ``sort_values`` call, on the same keys,
    with the same ``ascending`` flags, the same ``kind="stable"``, and the
    same default NaN placement. Scores are carried numeric-coerced exactly as
    the old code coerced them before sorting.
    """

    group: pd.DataFrame
    order: np.ndarray
    scores: np.ndarray
    top: pd.Series
    top_score: float

    @property
    def size(self) -> int:
        return int(len(self.order))


def _ordered(
    group: pd.DataFrame,
    score_column: str,
    mask: np.ndarray | None = None,
) -> _Ordered:
    scores = pd.to_numeric(
        group[score_column], errors="coerce"
    ).to_numpy(dtype=float)
    positions = (
        np.arange(len(group))
        if mask is None
        else np.flatnonzero(np.asarray(mask, dtype=bool))
    )
    keys = pd.DataFrame(
        {
            "_score": scores[positions],
            "_rank": group["candidate_rank"].to_numpy()[positions],
            "_itemcode": group["master_itemcode"].to_numpy()[positions],
        }
    )
    ranked_local = keys.sort_values(
        ["_score", "_rank", "_itemcode"],
        ascending=[False, True, True],
        kind="stable",
    ).index.to_numpy()
    order = positions[ranked_local]
    top_position = int(order[0])
    return _Ordered(
        group=group,
        order=order,
        scores=scores,
        top=group.iloc[top_position],
        top_score=float(scores[top_position]),
    )


def _score_margin(ordered: _Ordered | None) -> float | None:
    """Gap between the top two ranked scores, on the coerced values."""
    if ordered is None or ordered.size < 2:
        return None
    return float(
        ordered.scores[ordered.order[0]] - ordered.scores[ordered.order[1]]
    )


def _is_false(value: object) -> bool:
    return bool(pd.notna(value) and value is not None and not bool(value))


def _is_true(value: object) -> bool:
    return bool(pd.notna(value) and bool(value))


def _hard_conflicts(top: pd.Series) -> dict[str, bool]:
    pack_status = top.get("candidate_pack_status")
    structure_status = top.get("candidate_pack_structure_status")
    return {
        "protein_conflict": (
            _is_true(top.get("protein_conflict", False))
            or _is_false(top.get("protein_match", True))
        ),
        "mixed_protein_ambiguity": (
            _is_true(top.get("mixed_protein_ambiguity", False))
            or _is_true(top.get("is_mixed_protein_offer", False))
        ),
        "strong_family_conflict": (
            _is_true(top.get("strong_family_conflict", False))
            or _is_false(top.get("family_match", True))
        ),
        "strong_size_weight_conflict": (
            _is_true(top.get("strong_size_weight_conflict", False))
            or _is_false(pack_status)
            or _is_true(top.get("candidate_pack_status_conflict", False))
        ),
        "strong_pack_format_conflict": (
            _is_true(top.get("strong_pack_format_conflict", False))
            or structure_status is False
            or _is_false(top.get("pack_format_match", True))
        ),
        "feature_generation_failure": _is_true(
            top.get("feature_generation_failure", False)
        ),
        "missing_master": _is_true(top.get("missing_master", False)),
        "commercial_hard_conflict": _is_true(
            top.get("commercial_hard_conflict", False)
        ),
    }


def _master_missing(itemcode: str | None) -> bool:
    return itemcode is None or itemcode.strip() in {
        "",
        "NO_MATCH",
        "REVIEW_REQUIRED",
        "nan",
        "None",
    }


def _result(
    *,
    offer_id: str,
    lightgbm: _Ordered | None,
    embedding: _Ordered | None,
    conflicts: Mapping[str, bool],
    status: AgreementStatus,
    route: ReviewRoute,
    reasons: list[AgreementReasonCode],
) -> AgreementResult:
    lightgbm_top = lightgbm.top if lightgbm is not None else None
    embedding_top = embedding.top if embedding is not None else None
    lightgbm_itemcode = (
        str(lightgbm_top["master_itemcode"])
        if lightgbm_top is not None
        else None
    )
    embedding_itemcode = (
        str(embedding_top["master_itemcode"])
        if embedding_top is not None
        else None
    )
    same = bool(
        lightgbm_itemcode is not None
        and embedding_itemcode is not None
        and lightgbm_itemcode == embedding_itemcode
    )
    return AgreementResult(
        offer_id=offer_id,
        lightgbm_top_candidate=lightgbm_itemcode,
        lightgbm_calibrated_probability=(
            lightgbm.top_score if lightgbm is not None else None
        ),
        lightgbm_top_rank=1 if lightgbm_top is not None else None,
        lightgbm_candidate_rank=(
            int(lightgbm_top["candidate_rank"])
            if lightgbm_top is not None
            else None
        ),
        lightgbm_score_margin=_score_margin(lightgbm),
        embedding_top_candidate=embedding_itemcode,
        embedding_similarity=(
            embedding.top_score if embedding is not None else None
        ),
        embedding_rank=1 if embedding_top is not None else None,
        embedding_candidate_rank=(
            int(embedding_top["candidate_rank"])
            if embedding_top is not None
            else None
        ),
        embedding_score_margin=_score_margin(embedding),
        same_top_candidate=same,
        candidate_generation_margin=(
            float(lightgbm_top["candidate_margin"])
            if lightgbm_top is not None
            and pd.notna(lightgbm_top.get("candidate_margin"))
            else None
        ),
        candidate_generation_raw_margin=(
            float(lightgbm_top["candidate_raw_margin"])
            if lightgbm_top is not None
            and pd.notna(lightgbm_top.get("candidate_raw_margin"))
            else None
        ),
        conflict_flags=MappingProxyType(dict(conflicts)),
        agreement_status=status,
        routing_decision=route,
        reason_codes=tuple(reasons),
        routing_reason=";".join(reason.value for reason in reasons),
    )


def _evaluate_offer(
    offer_id: str,
    group: pd.DataFrame,
    config: AgreementConfig,
) -> AgreementResult:
    no_conflicts = {name: False for name in HARD_CONFLICT_NAMES}
    lightgbm_valid = _valid_scores(
        group["calibrated_probability"], probability=True
    )
    if not lightgbm_valid.all():
        return _result(
            offer_id=offer_id,
            lightgbm=None,
            embedding=None,
            conflicts=no_conflicts,
            status=AgreementStatus.MODEL_UNAVAILABLE,
            route=ReviewRoute.SAFE_FALLBACK,
            reasons=[AgreementReasonCode.LIGHTGBM_UNAVAILABLE],
        )

    # Selecting rows as a boolean mask rather than slicing out a sub-frame:
    # the preference order (exact, else adapted, else the whole group) is
    # unchanged, but a five-row slice of a ninety-column frame is no longer
    # built twice per offer.
    preferred_mask: np.ndarray | None = None
    if "commercial_outcome" in group:
        outcome = group["commercial_outcome"]
        exact_mask = outcome.eq("EXACT_MATCH").to_numpy()
        if exact_mask.any():
            preferred_mask = exact_mask
        else:
            adapted_mask = outcome.eq("ADAPTED_MATCH").to_numpy()
            if adapted_mask.any():
                preferred_mask = adapted_mask
    lightgbm = _ordered(
        group, "calibrated_probability", preferred_mask
    )
    top = lightgbm.top
    conflicts = _hard_conflicts(top)
    lightgbm_itemcode = str(top["master_itemcode"])
    master_missing = _master_missing(lightgbm_itemcode)
    if master_missing:
        conflicts["missing_master"] = True

    embedding_columns_present = {
        "embedding_similarity",
        "embedding_failure_reason",
    }.issubset(group.columns)
    embedding_valid = (
        _valid_scores(group["embedding_similarity"], probability=False)
        if embedding_columns_present
        else pd.Series(False, index=group.index)
    )
    embedding_failures = (
        group["embedding_failure_reason"]
        .astype("string")
        .fillna("")
        .str.strip()
        .ne("")
        if embedding_columns_present
        else pd.Series(True, index=group.index)
    )
    embedding_available = bool(
        embedding_columns_present
        and embedding_valid.all()
        and not embedding_failures.any()
    )
    embedding = (
        _ordered(group, "embedding_similarity", preferred_mask)
        if embedding_available
        else None
    )
    hard_conflict = any(conflicts.values())

    if not embedding_available:
        reasons = [AgreementReasonCode.EMBEDDING_UNAVAILABLE]
        if master_missing:
            reasons.append(AgreementReasonCode.MASTER_SKU_MISSING)
        if hard_conflict:
            reasons.append(AgreementReasonCode.HARD_CONFLICT)
        return _result(
            offer_id=offer_id,
            lightgbm=lightgbm,
            embedding=None,
            conflicts=conflicts,
            status=AgreementStatus.EMBEDDING_UNAVAILABLE,
            route=(
                config.hard_conflict_route
                if hard_conflict
                else ReviewRoute.SAFE_FALLBACK
            ),
            reasons=reasons,
        )

    embedding_itemcode = str(embedding.top["master_itemcode"])
    same = lightgbm_itemcode == embedding_itemcode
    reasons = [
        AgreementReasonCode.SAME_TOP_CANDIDATE
        if same
        else AgreementReasonCode.DIFFERENT_TOP_CANDIDATE
    ]
    if master_missing:
        reasons.append(AgreementReasonCode.MASTER_SKU_MISSING)
    if hard_conflict:
        reasons.append(AgreementReasonCode.HARD_CONFLICT)
        return _result(
            offer_id=offer_id,
            lightgbm=lightgbm,
            embedding=embedding,
            conflicts=conflicts,
            status=AgreementStatus.WEAK_AGREEMENT,
            route=config.hard_conflict_route,
            reasons=reasons,
        )

    if not same and config.require_same_top_candidate:
        return _result(
            offer_id=offer_id,
            lightgbm=lightgbm,
            embedding=embedding,
            conflicts=conflicts,
            status=AgreementStatus.DISAGREEMENT,
            route=config.disagreement_route,
            reasons=reasons,
        )

    weak = False
    probability = lightgbm.top_score
    if probability < config.lightgbm_auto_accept_threshold:
        reasons.append(AgreementReasonCode.LIGHTGBM_BELOW_THRESHOLD)
        weak = True
    embedding_similarity = embedding.top_score
    if (
        config.minimum_embedding_similarity is not None
        and embedding_similarity < config.minimum_embedding_similarity
    ):
        reasons.append(AgreementReasonCode.WEAK_EMBEDDING_SIMILARITY)
        weak = True
    embedding_margin = _score_margin(embedding)
    if (
        config.minimum_embedding_margin is not None
        and (
            embedding_margin is None
            or embedding_margin < config.minimum_embedding_margin
        )
    ):
        reasons.append(AgreementReasonCode.WEAK_EMBEDDING_MARGIN)
        weak = True
    if not same:
        weak = True
    if not weak and not config.allow_embedding_auto_accept:
        reasons.append(
            AgreementReasonCode.EMBEDDING_AUTO_INFLUENCE_DISABLED
        )

    return _result(
        offer_id=offer_id,
        lightgbm=lightgbm,
        embedding=embedding,
        conflicts=conflicts,
        status=(
            AgreementStatus.WEAK_AGREEMENT
            if weak
            else AgreementStatus.SAFE_AGREEMENT
        ),
        route=(
            config.weak_agreement_route
            if weak
            else (
                ReviewRoute.AUTO_ACCEPT
                if config.allow_embedding_auto_accept
                else ReviewRoute.MANUAL_REVIEW
            )
        ),
        reasons=reasons,
    )


def evaluate_candidate_agreement(
    candidates: pd.DataFrame,
    *,
    config: AgreementConfig,
) -> AgreementEvaluationResult:
    """Evaluate one route per offer without mutating candidate rows."""
    missing = sorted(REQUIRED_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(
            f"Agreement candidate frame is missing required columns: {missing}"
        )
    results = tuple(
        _evaluate_offer(str(offer_id), group.copy(deep=False), config)
        for offer_id, group in candidates.groupby(
            "offer_group_id", sort=False
        )
    )
    frame = pd.DataFrame([result.to_record() for result in results])
    return AgreementEvaluationResult(results=results, frame=frame)
