"""Persistent response cache isolated by request, model, prompt, and schema."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


class LLMReviewCacheError(ValueError):
    """Raised when a cache record fails integrity validation."""


@dataclass(frozen=True)
class CachedLLMResponse:
    """Integrity-checked provider response."""

    raw_response: str
    raw_response_hash: str


class PersistentLLMReviewCache:
    """SQLite cache with exact version isolation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_review_cache (
                request_hash TEXT NOT NULL,
                normalized_request TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                raw_response_hash TEXT NOT NULL,
                PRIMARY KEY (
                    request_hash,
                    model_id,
                    prompt_version,
                    schema_version
                )
            )
            """
        )
        return connection

    def get(
        self,
        *,
        request_hash: str,
        normalized_request: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
    ) -> CachedLLMResponse | None:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT normalized_request, raw_response, raw_response_hash
                    FROM llm_review_cache
                    WHERE request_hash = ?
                      AND model_id = ?
                      AND prompt_version = ?
                      AND schema_version = ?
                    """,
                    (
                        request_hash,
                        model_id,
                        prompt_version,
                        schema_version,
                    ),
                ).fetchone()
            finally:
                connection.close()
        if row is None:
            return None
        stored_request, raw_response, expected_hash = row
        if stored_request != normalized_request:
            raise LLMReviewCacheError(
                "Cached request hash collides with different request content"
            )
        actual_hash = hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest()
        if actual_hash != expected_hash:
            raise LLMReviewCacheError(
                "Cached LLM response checksum does not match"
            )
        return CachedLLMResponse(
            raw_response=raw_response,
            raw_response_hash=actual_hash,
        )

    def put(
        self,
        *,
        request_hash: str,
        normalized_request: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
        raw_response: str,
    ) -> str:
        raw_response_hash = hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO llm_review_cache (
                        request_hash,
                        normalized_request,
                        model_id,
                        prompt_version,
                        schema_version,
                        raw_response,
                        raw_response_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_hash,
                        normalized_request,
                        model_id,
                        prompt_version,
                        schema_version,
                        raw_response,
                        raw_response_hash,
                    ),
                )
                connection.commit()
            finally:
                connection.close()
        return raw_response_hash
