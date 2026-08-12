"""Stable constants shared by SKU-mapping training and inference."""

from enum import Enum


class MLDeploymentMode(str, Enum):
    """Explicit production deployment modes."""

    DISABLED = "disabled"
    SHADOW = "shadow"
    ASSISTED = "assisted"


class MatchDecision(str, Enum):
    """Auditable assisted-mode outcomes."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    NO_MATCH = "NO_MATCH"
    NO_CANDIDATE = "NO_CANDIDATE"
    MASTER_SKU_NOT_FOUND = "MASTER_SKU_NOT_FOUND"
    MODEL_ERROR = "MODEL_ERROR"


class FinalMatchDecision(str, Enum):
    """Unified assisted-inference outcome applied at the mapping gate."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    LLM_ACCEPT = "LLM_ACCEPT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    NO_MATCH = "NO_MATCH"
    NO_CANDIDATE = "NO_CANDIDATE"
    MASTER_SKU_NOT_FOUND = "MASTER_SKU_NOT_FOUND"
    MODEL_ERROR = "MODEL_ERROR"
    COMPETITOR_OFFER = "COMPETITOR_OFFER"


class AgreementStatus(str, Enum):
    """Evidence state for the two independent candidate scorers."""

    SAFE_AGREEMENT = "SAFE_AGREEMENT"
    WEAK_AGREEMENT = "WEAK_AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class ReviewRoute(str, Enum):
    """Routing result; LLM_REVIEW is a queue label, not an LLM call."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    LLM_REVIEW = "LLM_REVIEW"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    SAFE_FALLBACK = "SAFE_FALLBACK"


class AgreementReasonCode(str, Enum):
    """Stable diagnostic codes emitted by the agreement policy."""

    SAME_TOP_CANDIDATE = "SAME_TOP_CANDIDATE"
    DIFFERENT_TOP_CANDIDATE = "DIFFERENT_TOP_CANDIDATE"
    LIGHTGBM_BELOW_THRESHOLD = "LIGHTGBM_BELOW_THRESHOLD"
    HARD_CONFLICT = "HARD_CONFLICT"
    EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
    LIGHTGBM_UNAVAILABLE = "LIGHTGBM_UNAVAILABLE"
    MASTER_SKU_MISSING = "MASTER_SKU_MISSING"
    WEAK_EMBEDDING_MARGIN = "WEAK_EMBEDDING_MARGIN"
    WEAK_EMBEDDING_SIMILARITY = "WEAK_EMBEDDING_SIMILARITY"
    EMBEDDING_AUTO_INFLUENCE_DISABLED = (
        "EMBEDDING_AUTO_INFLUENCE_DISABLED"
    )


MODEL_FEATURE_COLUMNS: list[str] = [
    "protein_match",
    "family_match",
    "variant_match",
    "size_match",
    "pack_format_match",
    "word_similarity",
    "character_similarity",
    "token_similarity",
    "unit_pack_weight_g",
    "number_of_units",
    "bonus_weight_g",
    "total_offer_weight_g",
    "piece_count",
    "master_unit_weight_g",
    "master_units_per_carton",
    "is_mixed_protein_offer",
    "is_multi_family_offer",
    "contains_non_meat_product",
    "expected_match_count",
]

FEATURE_GENERATOR_VERSION = "1.0.0"
