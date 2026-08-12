"""Stable schemas and result models for structured LLM review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

PROMPT_VERSION = "sku-candidate-review-v1"
RESPONSE_SCHEMA_VERSION = "1.0"


class LLMDecision(str, Enum):
    """Only decisions the second-stage reviewer may return."""

    ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
    REJECT_ALL = "REJECT_ALL"
    UNCERTAIN = "UNCERTAIN"


class LLMReasonCode(str, Enum):
    """Controlled evidence vocabulary accepted from providers."""

    PROTEIN_MATCH = "PROTEIN_MATCH"
    PROTEIN_CONFLICT = "PROTEIN_CONFLICT"
    FAMILY_MATCH = "FAMILY_MATCH"
    FAMILY_CONFLICT = "FAMILY_CONFLICT"
    SIZE_MATCH = "SIZE_MATCH"
    SIZE_CONFLICT = "SIZE_CONFLICT"
    PACK_MATCH = "PACK_MATCH"
    PACK_CONFLICT = "PACK_CONFLICT"
    VARIANT_MATCH = "VARIANT_MATCH"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    MULTIPLE_PLAUSIBLE_CANDIDATES = "MULTIPLE_PLAUSIBLE_CANDIDATES"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"


class LLMReviewStatus(str, Enum):
    """Operational state of one requested review."""

    COMPLETED = "COMPLETED"
    DISABLED = "DISABLED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    TIMEOUT = "TIMEOUT"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"


class LLMReviewRoute(str, Enum):
    """Diagnostic eligibility after deterministic policy enforcement."""

    LLM_ACCEPT = "LLM_ACCEPT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    NO_MATCH = "NO_MATCH"


LLM_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "selected_candidate_id",
        "confidence",
        "reason_codes",
        "short_explanation",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": [decision.value for decision in LLMDecision],
        },
        "selected_candidate_id": {
            "type": ["string", "null"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reason_codes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [reason.value for reason in LLMReasonCode],
            },
            "uniqueItems": True,
        },
        "short_explanation": {
            "type": "string",
            "maxLength": 500,
        },
    },
}


@dataclass(frozen=True)
class ParsedLLMResponse:
    """A response that passed strict syntax and schema validation."""

    decision: LLMDecision
    selected_candidate_id: str | None
    confidence: float
    reason_codes: tuple[LLMReasonCode, ...]
    short_explanation: str


@dataclass(frozen=True)
class LLMReviewResult:
    """Full non-secret provenance and final diagnostic route for one offer."""

    offer_id: str
    provider: str
    model_name: str
    model_id: str
    prompt_version: str
    response_schema_version: str
    timestamp: str
    request_hash: str
    raw_response_hash: str | None
    review_status: LLMReviewStatus
    parsed_decision: LLMDecision | None
    confidence: float | None
    selected_candidate: str | None
    reason_codes: tuple[LLMReasonCode, ...]
    short_explanation: str | None
    validation_errors: tuple[str, ...]
    latency_seconds: float
    retry_count: int
    cache_hit: bool
    hard_conflict: bool
    final_route: LLMReviewRoute
    routing_reason: str

    def to_record(self) -> dict[str, Any]:
        """Return a stable tabular representation without the raw response."""
        return {
            "offer_id": self.offer_id,
            "llm_provider": self.provider,
            "llm_model_name": self.model_name,
            "llm_model_id": self.model_id,
            "llm_prompt_version": self.prompt_version,
            "llm_response_schema_version": self.response_schema_version,
            "llm_timestamp": self.timestamp,
            "llm_request_hash": self.request_hash,
            "llm_raw_response_hash": self.raw_response_hash,
            "llm_review_status": self.review_status.value,
            "llm_parsed_decision": (
                self.parsed_decision.value
                if self.parsed_decision is not None
                else None
            ),
            "llm_confidence": self.confidence,
            "llm_selected_candidate": self.selected_candidate,
            "llm_reason_codes": "|".join(
                reason.value for reason in self.reason_codes
            ),
            "llm_short_explanation": self.short_explanation,
            "llm_validation_errors": json.dumps(
                list(self.validation_errors), ensure_ascii=False
            ),
            "llm_latency_seconds": self.latency_seconds,
            "llm_retry_count": self.retry_count,
            "llm_cache_hit": self.cache_hit,
            "llm_hard_conflict": self.hard_conflict,
            "llm_final_route": self.final_route.value,
            "llm_routing_reason": self.routing_reason,
            "llm_production_applied": False,
        }


@dataclass(frozen=True)
class LLMReviewBatchResult:
    """All requested second-stage reviews for one candidate batch."""

    status: str
    results: tuple[LLMReviewResult, ...]
    frame: pd.DataFrame
    offers_routed: int
    provider_calls: int
    cache_hits: int
    failures: int
    error: str | None = None
