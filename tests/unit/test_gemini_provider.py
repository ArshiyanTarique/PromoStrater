"""Gemini provider: every failure mode must fail closed.

The reviewer's contract is that a provider either returns response text or
raises an LLMProviderError. Anything raised here becomes the configured fail
route - human review - so these tests pin that no malformed, truncated, empty
or errored Gemini reply can reach the accept path.

No network and no API key are required: a fake transport is injected.
"""

from __future__ import annotations

import json

import pytest

from sku_mapping.config import load_config
from sku_mapping.llm_review.gemini import (
    API_KEY_VARIABLES,
    DEFAULT_ENDPOINT,
    GeminiApiKeyMissingError,
    GeminiProvider,
)
from sku_mapping.llm_review.provider import (
    LLMProviderError,
    LLMProviderTimeout,
    create_llm_provider,
)


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _provider(reply, **kwargs):
    def transport(*, endpoint, payload, timeout_seconds, api_key):
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, (bytes, bytearray)):
            return reply
        return json.dumps(reply).encode("utf-8")

    return GeminiProvider(
        configured_model="gemini-2.0-flash", transport=transport, **kwargs
    )


def _ok(text="{\"decision\":\"ACCEPT\"}"):
    return {"candidates": [{"finishReason": "STOP",
                            "content": {"parts": [{"text": text}]}}]}


def _generate(provider):
    return provider.generate(
        structured_request="offer -> candidates",
        system_prompt="judge the candidate",
        timeout_seconds=5.0,
        temperature=0.0,
    )


# -- happy path -------------------------------------------------------------
def test_valid_response_returns_model_text():
    assert _generate(_provider(_ok())) == '{"decision":"ACCEPT"}'


def test_identity_is_stable_and_cache_isolating():
    p = _provider(_ok())
    assert p.provider_name == "gemini"
    assert p.model_name == "gemini-2.0-flash"
    assert p.model_id == "gemini:gemini-2.0-flash"


# -- failure modes, all must raise ------------------------------------------
def test_invalid_json_envelope_fails_closed():
    with pytest.raises(LLMProviderError):
        _generate(_provider(b"not json at all"))


def test_timeout_is_reported_as_a_timeout():
    with pytest.raises(LLMProviderTimeout):
        _generate(_provider(TimeoutError("slow")))


def test_api_error_envelope_fails_closed():
    with pytest.raises(LLMProviderError, match="API error"):
        _generate(_provider({"error": {"status": "PERMISSION_DENIED"}}))


def test_empty_response_fails_closed():
    with pytest.raises(LLMProviderError, match="empty"):
        _generate(_provider(_ok(text="   ")))


def test_missing_candidates_fails_closed():
    with pytest.raises(LLMProviderError, match="no candidates"):
        _generate(_provider({"candidates": []}))


def test_missing_content_parts_fails_closed():
    with pytest.raises(LLMProviderError, match="content parts"):
        _generate(_provider({"candidates": [{"content": {}}]}))


def test_truncated_generation_is_not_treated_as_a_judgement():
    """A safety-stopped reply is a failure, not a decision to parse."""
    with pytest.raises(LLMProviderError, match="stopped early"):
        _generate(_provider({"candidates": [
            {"finishReason": "SAFETY",
             "content": {"parts": [{"text": "{}"}]}}]}))


def test_transport_failure_fails_closed():
    with pytest.raises(LLMProviderError):
        _generate(_provider(OSError("connection reset")))


# -- key handling -----------------------------------------------------------
def test_missing_api_key_raises_rather_than_running_keyless(monkeypatch):
    for name in API_KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(GeminiApiKeyMissingError):
        _generate(_provider(_ok()))


def test_api_key_travels_in_a_header_not_the_url():
    seen = {}

    def transport(*, endpoint, payload, timeout_seconds, api_key):
        seen["endpoint"] = endpoint
        seen["api_key"] = api_key
        return json.dumps(_ok()).encode("utf-8")

    provider = GeminiProvider(
        configured_model="gemini-2.0-flash", transport=transport
    )
    _generate(provider)
    assert seen["api_key"] == "test-key-not-a-real-secret"
    assert "test-key-not-a-real-secret" not in seen["endpoint"]
    assert "key=" not in seen["endpoint"]


def test_no_api_key_is_present_in_source():
    """A committed key is the one mistake this module must never make."""
    from pathlib import Path

    source = Path("src/sku_mapping/llm_review/gemini.py").read_text(
        encoding="utf-8"
    )
    assert "AIza" not in source  # Google API keys start with this prefix
    assert "os.environ" in source


# -- endpoint validation ----------------------------------------------------
@pytest.mark.parametrize(
    "endpoint",
    ["ftp://example.com", "not-a-url", "https://user:pw@example.com",
     "https://example.com?key=abc"],
)
def test_unsafe_endpoints_are_refused(endpoint):
    with pytest.raises(ValueError):
        GeminiProvider(configured_model="m", base_endpoint=endpoint)


def test_empty_model_name_is_refused():
    with pytest.raises(ValueError):
        GeminiProvider(configured_model="  ")


# -- factory wiring ---------------------------------------------------------
def test_factory_builds_gemini_from_configuration():
    import dataclasses

    base = load_config("config/default.yaml").llm_review
    cfg = dataclasses.replace(base, provider="gemini", model="gemini-2.0-flash")
    provider = create_llm_provider(cfg)
    assert provider.provider_name == "gemini"
    # The Ollama localhost default must not follow Gemini into production.
    assert provider.base_endpoint == DEFAULT_ENDPOINT


def test_factory_still_builds_ollama():
    provider = create_llm_provider(load_config("config/default.yaml").llm_review)
    assert provider.provider_name == "ollama"
