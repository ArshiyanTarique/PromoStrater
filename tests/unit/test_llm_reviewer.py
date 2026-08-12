"""Adversarial tests for the bounded structured LLM reviewer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from sku_mapping.config import load_config
from sku_mapping.llm_review.cache import PersistentLLMReviewCache
from sku_mapping.llm_review.models import (
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    LLMReviewRoute,
    LLMReviewStatus,
)
from sku_mapping.llm_review.provider import LLMProviderTimeout
from sku_mapping.llm_review.reviewer import (
    build_structured_review_request,
    normalized_structured_request,
    review_llm_routes,
    review_llm_routes_non_blocking,
)


class FakeProvider:
    def __init__(
        self,
        responses: list[str | Exception],
        *,
        model_id: str = "fake:test-model-v1",
    ) -> None:
        self.responses = responses
        self._model_id = model_id
        self.calls = 0
        self.requests: list[str] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_id.split(":", 1)[-1]

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        *,
        structured_request: str,
        system_prompt: str,
        timeout_seconds: float,
        temperature: float,
    ) -> str:
        del system_prompt, timeout_seconds, temperature
        self.requests.append(structured_request)
        index = min(self.calls, len(self.responses) - 1)
        response = self.responses[index]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


def _config(tmp_path: Path, **changes):
    default = load_config("config/default.yaml").llm_review
    values = {
        "enabled": True,
        "provider": "fake",
        "model": "test-model-v1",
        "maximum_retries": 0,
        "cache_responses": False,
        "cache_path": tmp_path / "llm-cache.sqlite3",
    }
    values.update(changes)
    return replace(default, **values)


def _candidates(*, hard_conflict: bool = False) -> pd.DataFrame:
    common = {
        "offer_group_id": "offer-1",
        "offer_text": "Al Kabeer chicken nuggets 400 g",
        "offer_brand": "Al Kabeer",
        "offer_product": "Chicken Nuggets",
        "offer_variant": "Original",
        "offer_base_packsize": "400 g",
        "product_family": "chicken nuggets",
        "protein_classification": "chicken",
        "unit_pack_weight_g": 400.0,
        "number_of_units": 1.0,
        "total_offer_weight_g": 400.0,
        "is_mixed_protein_offer": 0,
        "reason_codes": "DIFFERENT_TOP_CANDIDATE",
        "embedding_failure_reason": "",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "candidate_rank": 1,
                "master_itemcode": "SKU-A",
                "master_item_description": "Chicken Nuggets Original",
                "master_item_long_description": "400 g chicken nuggets",
                "master_item_family": "Nuggets",
                "master_item_category": "Chicken",
                "master_item_spec": "400 g",
                "calibrated_probability": 0.78,
                "embedding_similarity": 0.62,
                "embedding_rank": 2,
                "protein_match": 0 if hard_conflict else 1,
                "family_match": 1,
                "variant_match": 1,
                "size_match": 1,
                "pack_format_match": 1,
                "candidate_pack_status": True,
                "candidate_pack_structure_status": True,
                "pack_conflict": False,
                "master_unit_weight_g": 400.0,
            },
            {
                **common,
                "candidate_rank": 2,
                "master_itemcode": "SKU-B",
                "master_item_description": "Chicken Nuggets Spicy",
                "master_item_long_description": "400 g spicy nuggets",
                "master_item_family": "Nuggets",
                "master_item_category": "Chicken",
                "master_item_spec": "400 g",
                "calibrated_probability": 0.71,
                "embedding_similarity": 0.68,
                "embedding_rank": 1,
                "protein_match": 1,
                "family_match": 1,
                "variant_match": 0,
                "size_match": 1,
                "pack_format_match": 1,
                "candidate_pack_status": True,
                "candidate_pack_structure_status": True,
                "pack_conflict": False,
                "master_unit_weight_g": 400.0,
            },
        ]
    )


def _agreements() -> pd.DataFrame:
    return pd.DataFrame(
        [{"offer_id": "offer-1", "routing_decision": "LLM_REVIEW"}]
    )


def _response(
    decision: str,
    *,
    candidate: str | None,
    confidence: float,
    reasons: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "selected_candidate_id": candidate,
            "confidence": confidence,
            "reason_codes": reasons or ["FAMILY_MATCH"],
            "short_explanation": "Bounded candidate evidence supports this.",
        }
    )


def _one_result(tmp_path: Path, response: str, **config_changes):
    provider = FakeProvider([response])
    batch = review_llm_routes(
        _candidates(),
        _agreements(),
        config=_config(tmp_path, **config_changes),
        provider=provider,
    )
    return batch.results[0], provider


def test_valid_high_confidence_acceptance_is_eligible(tmp_path: Path) -> None:
    result, provider = _one_result(
        tmp_path,
        _response(
            "ACCEPT_CANDIDATE",
            candidate="SKU-B",
            confidence=0.91,
            reasons=["PROTEIN_MATCH", "FAMILY_MATCH", "SIZE_MATCH"],
        ),
    )
    assert result.review_status is LLMReviewStatus.COMPLETED
    assert result.final_route is LLMReviewRoute.LLM_ACCEPT
    assert result.selected_candidate == "SKU-B"
    assert result.raw_response_hash
    assert provider.calls == 1


def test_selected_candidate_outside_supplied_list_is_rejected(
    tmp_path: Path,
) -> None:
    result, _ = _one_result(
        tmp_path,
        _response(
            "ACCEPT_CANDIDATE",
            candidate="INVENTED-SKU",
            confidence=0.99,
        ),
    )
    assert result.review_status is LLMReviewStatus.INVALID_RESPONSE
    assert result.final_route is LLMReviewRoute.MANUAL_REVIEW
    assert "SELECTED_CANDIDATE_NOT_SUPPLIED" in result.validation_errors


def test_uncontrolled_reason_code_is_rejected(tmp_path: Path) -> None:
    result, _ = _one_result(
        tmp_path,
        _response(
            "UNCERTAIN",
            candidate=None,
            confidence=0.5,
            reasons=["MAKE_UP_A_SKU"],
        ),
    )
    assert result.review_status is LLMReviewStatus.INVALID_RESPONSE
    assert any(
        error.startswith("INVALID_REASON_CODE")
        for error in result.validation_errors
    )


def test_malformed_json_routes_to_manual_review(tmp_path: Path) -> None:
    result, _ = _one_result(tmp_path, "not-json")
    assert result.review_status is LLMReviewStatus.INVALID_RESPONSE
    assert result.final_route is LLMReviewRoute.MANUAL_REVIEW
    assert result.validation_errors[0].startswith("MALFORMED_JSON")


def test_timeout_retries_then_routes_to_manual_review(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            LLMProviderTimeout("slow"),
            LLMProviderTimeout("still slow"),
        ]
    )
    batch = review_llm_routes(
        _candidates(),
        _agreements(),
        config=_config(tmp_path, maximum_retries=1),
        provider=provider,
    )
    result = batch.results[0]
    assert result.review_status is LLMReviewStatus.TIMEOUT
    assert result.final_route is LLMReviewRoute.MANUAL_REVIEW
    assert result.retry_count == 1
    assert provider.calls == 2


def test_low_confidence_acceptance_routes_to_manual_review(
    tmp_path: Path,
) -> None:
    result, _ = _one_result(
        tmp_path,
        _response(
            "ACCEPT_CANDIDATE",
            candidate="SKU-A",
            confidence=0.84,
        ),
    )
    assert result.review_status is LLMReviewStatus.COMPLETED
    assert result.final_route is LLMReviewRoute.MANUAL_REVIEW
    assert result.routing_reason == "LLM_CONFIDENCE_BELOW_THRESHOLD"


def test_uncertain_response_routes_to_manual_review(tmp_path: Path) -> None:
    result, _ = _one_result(
        tmp_path,
        _response("UNCERTAIN", candidate=None, confidence=0.72),
    )
    assert result.final_route is LLMReviewRoute.MANUAL_REVIEW
    assert result.routing_reason == "LLM_UNCERTAIN"


def test_request_is_bounded_to_configured_candidates() -> None:
    request = build_structured_review_request(
        _candidates(), maximum_candidates=1
    )
    assert len(request["candidates"]) == 1
    assert request["candidates"][0]["candidate_id"] == "SKU-A"
    assert "Product Master" not in normalized_structured_request(request)


@pytest.mark.parametrize(
    ("configured_route", "expected"),
    [
        ("MANUAL_REVIEW", LLMReviewRoute.MANUAL_REVIEW),
        ("NO_MATCH", LLMReviewRoute.NO_MATCH),
    ],
)
def test_reject_all_uses_explicit_policy(
    tmp_path: Path,
    configured_route: str,
    expected: LLMReviewRoute,
) -> None:
    result, _ = _one_result(
        tmp_path,
        _response(
            "REJECT_ALL",
            candidate=None,
            confidence=0.95,
            reasons=["NO_VALID_CANDIDATE"],
        ),
        reject_all_route=configured_route,
    )
    assert result.final_route is expected


def test_hard_conflict_blocks_llm_accept(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _response(
                "ACCEPT_CANDIDATE",
                candidate="SKU-A",
                confidence=0.99,
                reasons=["PROTEIN_MATCH"],
            )
        ]
    )
    batch = review_llm_routes(
        _candidates(hard_conflict=True),
        _agreements(),
        config=_config(tmp_path),
        provider=provider,
    )
    result = batch.results[0]
    assert result.hard_conflict is True
    assert result.final_route is LLMReviewRoute.MANUAL_REVIEW
    assert "HARD_CONFLICT" in result.routing_reason


def test_cache_isolated_by_model_prompt_and_schema_versions(
    tmp_path: Path,
) -> None:
    response = _response(
        "ACCEPT_CANDIDATE", candidate="SKU-A", confidence=0.90
    )
    cache = PersistentLLMReviewCache(tmp_path / "cache.sqlite3")
    config = _config(
        tmp_path, cache_responses=True, maximum_retries=0
    )
    first = FakeProvider([response], model_id="fake:model-v1")
    review_llm_routes(
        _candidates(),
        _agreements(),
        config=config,
        provider=first,
        cache=cache,
    )
    again = FakeProvider([response], model_id="fake:model-v1")
    cached_batch = review_llm_routes(
        _candidates(),
        _agreements(),
        config=config,
        provider=again,
        cache=cache,
    )
    other_model = FakeProvider([response], model_id="fake:model-v2")
    review_llm_routes(
        _candidates(),
        _agreements(),
        config=config,
        provider=other_model,
        cache=cache,
    )

    payload = build_structured_review_request(
        _candidates(), maximum_candidates=5
    )
    normalized = normalized_structured_request(payload)
    request_hash = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
    assert first.calls == 1
    assert again.calls == 0
    assert cached_batch.results[0].cache_hit is True
    assert other_model.calls == 1
    assert cache.get(
        request_hash=request_hash,
        normalized_request=normalized,
        model_id="fake:model-v1",
        prompt_version=PROMPT_VERSION + "-changed",
        schema_version=RESPONSE_SCHEMA_VERSION,
    ) is None
    assert cache.get(
        request_hash=request_hash,
        normalized_request=normalized,
        model_id="fake:model-v1",
        prompt_version=PROMPT_VERSION,
        schema_version=RESPONSE_SCHEMA_VERSION + "-changed",
    ) is None


def test_disabled_mode_does_not_call_provider_or_mutate_candidates(
    tmp_path: Path,
) -> None:
    candidates = _candidates()
    before = candidates.copy(deep=True)
    provider = FakeProvider(
        [
            _response(
                "ACCEPT_CANDIDATE",
                candidate="SKU-A",
                confidence=1,
            )
        ]
    )
    config = replace(_config(tmp_path), enabled=False)
    batch = review_llm_routes(
        candidates,
        _agreements(),
        config=config,
        provider=provider,
    )
    assert provider.calls == 0
    assert batch.results[0].review_status is LLMReviewStatus.DISABLED
    assert batch.results[0].final_route is LLMReviewRoute.MANUAL_REVIEW
    pd.testing.assert_frame_equal(candidates, before, check_exact=True)


def test_provider_failure_is_non_blocking_and_routes_manual(
    tmp_path: Path,
) -> None:
    candidates = _candidates()
    before = candidates.copy(deep=True)
    provider = FakeProvider([RuntimeError("provider down")])
    batch = review_llm_routes_non_blocking(
        candidates,
        _agreements(),
        config=_config(tmp_path),
        provider=provider,
    )
    assert batch.status == "COMPLETED_WITH_FAILURES"
    assert batch.results[0].review_status is LLMReviewStatus.PROVIDER_FAILURE
    assert batch.results[0].final_route is LLMReviewRoute.MANUAL_REVIEW
    pd.testing.assert_frame_equal(candidates, before, check_exact=True)
