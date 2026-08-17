"""Google Gemini provider for the second-stage reviewer.

Gemini is a REVIEWER, never the matcher. It is handed candidates the pipeline
already produced and asked to judge them; it cannot introduce a Master SKU,
because the reviewer validates every returned candidate id against the list it
supplied and routes anything unrecognised to human review.

This module only turns a prompt into response text. Schema validation,
candidate-id checking, confidence policy, retries, caching and the fail-safe
route all live in :mod:`llm_review.reviewer` and are shared with every other
provider - Gemini inherits them rather than reimplementing them.

The API key is read from the environment, never from configuration files or
source. ``GEMINI_API_KEY`` is checked first, then ``GOOGLE_API_KEY``. A missing
key raises at construction, so a run fails closed at startup instead of
discovering it mid-review.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error as urllib_error
import urllib.parse as urllib_parse
import urllib.request as urllib_request
from dataclasses import dataclass, field

from sku_mapping.llm_review.models import LLM_RESPONSE_JSON_SCHEMA
from sku_mapping.llm_review.provider import (
    LLMProviderError,
    LLMProviderTimeout,
)

#: Checked in order. Both are conventional for Google APIs; neither is ever
#: written to a file, a log line, or a cache key.
API_KEY_VARIABLES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com"


class GeminiApiKeyMissingError(LLMProviderError):
    """Raised when no Gemini key is present in the environment."""


def _api_key() -> str:
    for name in API_KEY_VARIABLES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise GeminiApiKeyMissingError(
        "No Gemini API key found. Set "
        + " or ".join(API_KEY_VARIABLES)
        + " in the environment; keys are never read from configuration."
    )


@dataclass(frozen=True)
class GeminiProvider:
    """Google Generative Language `generateContent` provider."""

    configured_model: str
    base_endpoint: str = DEFAULT_ENDPOINT
    #: Injected only by tests. Production leaves it None and uses urllib.
    transport: object | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.configured_model.strip():
            raise ValueError("Gemini model name must be non-empty")
        parsed = urllib_parse.urlsplit(self.base_endpoint.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "Gemini endpoint must be an absolute http or https URL"
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "Gemini endpoint cannot contain credentials, a query, "
                "or a fragment"
            )
        object.__setattr__(
            self,
            "base_endpoint",
            urllib_parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
            ),
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.configured_model

    @property
    def model_id(self) -> str:
        return f"gemini:{self.configured_model}"

    @property
    def generate_endpoint(self) -> str:
        return (
            f"{self.base_endpoint}/v1beta/models/"
            f"{self.configured_model}:generateContent"
        )

    def generate(
        self,
        *,
        structured_request: str,
        system_prompt: str,
        timeout_seconds: float,
        temperature: float,
    ) -> str:
        """Return the model's response text, or raise a provider error.

        Every failure mode is converted into an ``LLMProviderError`` (or the
        timeout subclass) so the reviewer applies its configured fail route -
        human review - rather than an exception escaping into the run.
        """
        payload = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [
                    {"role": "user", "parts": [{"text": structured_request}]}
                ],
                "generationConfig": {
                    "temperature": temperature,
                    # Ask for JSON matching the same schema every provider is
                    # held to, so a well-behaved response needs no repair.
                    "responseMimeType": "application/json",
                    "responseSchema": _response_schema(),
                },
            },
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

        envelope_bytes = self._post(payload, timeout_seconds)

        try:
            envelope = json.loads(envelope_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LLMProviderError(
                "Gemini returned a malformed response envelope"
            ) from error
        if not isinstance(envelope, dict):
            raise LLMProviderError("Gemini response envelope must be an object")
        return _response_text(envelope)

    # -- transport ------------------------------------------------------
    def _post(self, payload: bytes, timeout_seconds: float) -> bytes:
        key = _api_key()
        if self.transport is not None:
            # The injected transport gets the same conversion as urllib: a
            # provider that raises anything other than an LLMProviderError
            # would escape the reviewer's fail-safe route entirely.
            try:
                return self.transport(  # type: ignore[operator]
                    endpoint=self.generate_endpoint,
                    payload=payload,
                    timeout_seconds=timeout_seconds,
                    api_key=key,
                )
            except LLMProviderError:
                raise
            except (TimeoutError, socket.timeout) as error:
                raise LLMProviderTimeout(
                    f"Gemini request timed out after {timeout_seconds:g}s"
                ) from error
            except Exception as error:
                raise LLMProviderError(
                    f"Gemini request failed: {error}"
                ) from error
        request = urllib_request.Request(
            self.generate_endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                # Header rather than a query parameter: a URL carrying the key
                # would reach access logs and error messages.
                "x-goog-api-key": key,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                request, timeout=timeout_seconds
            ) as response:
                return response.read()
        except (TimeoutError, socket.timeout) as error:
            raise LLMProviderTimeout(
                f"Gemini request timed out after {timeout_seconds:g}s"
            ) from error
        except urllib_error.HTTPError as error:
            # The body can echo request content; the status is enough to act on
            # and cannot leak the key.
            raise LLMProviderError(
                f"Gemini request rejected with HTTP {error.code}"
            ) from error
        except urllib_error.URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise LLMProviderTimeout(
                    f"Gemini request timed out after {timeout_seconds:g}s"
                ) from error
            raise LLMProviderError(
                f"Gemini endpoint unavailable: {error.reason}"
            ) from error
        except OSError as error:
            raise LLMProviderError(f"Gemini request failed: {error}") from error


def _response_schema() -> dict:
    """The shared reviewer schema, minus keys Gemini's dialect rejects."""
    def strip(node):
        if isinstance(node, dict):
            return {
                k: strip(v)
                for k, v in node.items()
                if k not in {"additionalProperties", "$schema"}
            }
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(LLM_RESPONSE_JSON_SCHEMA)


def _response_text(envelope: dict) -> str:
    """Pull the single text part out of a generateContent envelope."""
    if "error" in envelope:
        detail = envelope.get("error")
        status = (
            detail.get("status") if isinstance(detail, dict) else None
        ) or "unknown"
        raise LLMProviderError(f"Gemini returned an API error: {status}")
    candidates = envelope.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMProviderError("Gemini response contained no candidates")
    first = candidates[0]
    if not isinstance(first, dict):
        raise LLMProviderError("Gemini candidate must be an object")
    # A response cut short is not a judgement; treat it as a failure so the
    # reviewer falls back rather than parsing half an answer.
    finish = str(first.get("finishReason", "") or "").upper()
    if finish and finish not in {"STOP", "MAX_TOKENS", ""}:
        raise LLMProviderError(f"Gemini stopped early: {finish}")
    parts = (first.get("content") or {}).get("parts")
    if not isinstance(parts, list) or not parts:
        raise LLMProviderError("Gemini candidate carried no content parts")
    text = "".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict)
    )
    if not text.strip():
        raise LLMProviderError("Gemini returned an empty response")
    return text
