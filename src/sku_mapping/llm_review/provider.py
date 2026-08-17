"""Replaceable provider boundary with a free local Ollama implementation."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from sku_mapping.config import LLMReviewConfig
from sku_mapping.llm_review.models import LLM_RESPONSE_JSON_SCHEMA


class LLMProviderError(RuntimeError):
    """Raised when a configured provider cannot return a usable envelope."""


class LLMProviderTimeout(LLMProviderError):
    """Raised when a provider request exceeds its configured timeout."""


class LLMProvider(Protocol):
    """Minimal replaceable generation interface used by the reviewer."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier."""

    @property
    def model_name(self) -> str:
        """Configured provider model name."""

    @property
    def model_id(self) -> str:
        """Cache-isolating model identity."""

    def generate(
        self,
        *,
        structured_request: str,
        system_prompt: str,
        timeout_seconds: float,
        temperature: float,
    ) -> str:
        """Return only the provider's model response text."""


def _validated_base_endpoint(value: str) -> str:
    parsed = urllib_parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "LLM endpoint must be an absolute http or https URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "LLM endpoint cannot contain credentials, a query, or a fragment"
        )
    return urllib_parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


@dataclass(frozen=True)
class OllamaProvider:
    """Ollama-compatible local `/api/generate` provider."""

    configured_model: str
    base_endpoint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_endpoint",
            _validated_base_endpoint(self.base_endpoint),
        )
        if not self.configured_model.strip():
            raise ValueError("Ollama model name must be non-empty")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.configured_model

    @property
    def model_id(self) -> str:
        return (
            f"ollama:{self.configured_model}@"
            f"{self.base_endpoint.lower()}"
        )

    @property
    def generate_endpoint(self) -> str:
        if self.base_endpoint.endswith("/api/generate"):
            return self.base_endpoint
        return f"{self.base_endpoint}/api/generate"

    def generate(
        self,
        *,
        structured_request: str,
        system_prompt: str,
        timeout_seconds: float,
        temperature: float,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.configured_model,
                "stream": False,
                "system": system_prompt,
                "prompt": structured_request,
                "format": LLM_RESPONSE_JSON_SCHEMA,
                "options": {"temperature": temperature},
            },
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        request = urllib_request.Request(
            self.generate_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                request, timeout=timeout_seconds
            ) as response:
                envelope_bytes = response.read()
        except (TimeoutError, socket.timeout) as error:
            raise LLMProviderTimeout(
                f"Ollama request timed out after {timeout_seconds:g}s"
            ) from error
        except urllib_error.URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise LLMProviderTimeout(
                    f"Ollama request timed out after {timeout_seconds:g}s"
                ) from error
            raise LLMProviderError(
                f"Ollama endpoint unavailable: {error.reason}"
            ) from error
        except OSError as error:
            raise LLMProviderError(
                f"Ollama request failed: {error}"
            ) from error

        try:
            envelope = json.loads(envelope_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LLMProviderError(
                "Ollama returned a malformed response envelope"
            ) from error
        if not isinstance(envelope, dict):
            raise LLMProviderError(
                "Ollama response envelope must be an object"
            )
        response_text = envelope.get("response")
        if not isinstance(response_text, str) or not response_text.strip():
            raise LLMProviderError(
                "Ollama response envelope has no non-empty response"
            )
        return response_text


def create_llm_provider(config: LLMReviewConfig) -> LLMProvider:
    """Create a configured provider without making a network request."""
    provider_name = config.provider.strip().lower()
    if provider_name == "ollama":
        return OllamaProvider(
            configured_model=config.model,
            base_endpoint=config.endpoint,
        )
    if provider_name == "gemini":
        # Imported here so the Gemini module - and its environment key lookup -
        # is only touched by a run that actually selected it.
        from sku_mapping.llm_review.gemini import DEFAULT_ENDPOINT, GeminiProvider

        endpoint = (config.endpoint or "").strip()
        return GeminiProvider(
            configured_model=config.model,
            # The Ollama default endpoint is meaningless for Gemini; a config
            # still pointing at localhost falls back to the Google host rather
            # than issuing a request that could never succeed.
            base_endpoint=(
                endpoint
                if endpoint and "localhost" not in endpoint
                else DEFAULT_ENDPOINT
            ),
        )
    raise ValueError(f"Unknown LLM review provider: {config.provider!r}")
