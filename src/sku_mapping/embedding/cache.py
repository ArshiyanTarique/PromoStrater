"""Version-isolated, checksummed persistent embedding vector cache."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np


class EmbeddingCacheError(ValueError):
    """Raised when cached vector data is invalid or corrupt."""


class PersistentEmbeddingCache:
    """SQLite cache isolated by full encoder/text identity and namespace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache_v2 (
                cache_fingerprint TEXT NOT NULL,
                text_namespace TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                vector_sha256 TEXT NOT NULL,
                PRIMARY KEY (
                    cache_fingerprint,
                    text_namespace,
                    normalized_text
                )
            )
            """
        )
        return connection

    def get_many(
        self,
        texts: list[str],
        *,
        model_id: str,
        model_version: str,
        cache_fingerprint: str,
        text_namespace: str,
    ) -> dict[str, np.ndarray]:
        if not texts:
            return {}
        found: dict[str, np.ndarray] = {}
        with self._lock:
            connection = self._connect()
            try:
                for text in texts:
                    row = connection.execute(
                        """
                        SELECT model_id, model_version, dimension,
                               vector, vector_sha256
                        FROM embedding_cache_v2
                        WHERE cache_fingerprint = ?
                          AND text_namespace = ?
                          AND normalized_text = ?
                        """,
                        (cache_fingerprint, text_namespace, text),
                    ).fetchone()
                    if row is None:
                        continue
                    (
                        stored_model_id,
                        stored_model_version,
                        dimension,
                        payload,
                        expected_sha,
                    ) = row
                    if (
                        stored_model_id != model_id
                        or stored_model_version != model_version
                    ):
                        raise EmbeddingCacheError(
                            "Cached embedding identity does not match request"
                        )
                    actual_sha = hashlib.sha256(payload).hexdigest()
                    if actual_sha != expected_sha:
                        raise EmbeddingCacheError(
                            "Cached embedding checksum does not match"
                        )
                    vector = np.frombuffer(payload, dtype=np.float32).copy()
                    if vector.shape != (int(dimension),):
                        raise EmbeddingCacheError(
                            "Cached embedding dimension is invalid"
                        )
                    if not np.isfinite(vector).all():
                        raise EmbeddingCacheError(
                            "Cached embedding contains non-finite values"
                        )
                    found[text] = vector
            finally:
                connection.close()
        return found

    def put_many(
        self,
        vectors: dict[str, np.ndarray],
        *,
        model_id: str,
        model_version: str,
        cache_fingerprint: str,
        text_namespace: str,
    ) -> None:
        if not vectors:
            return
        records: list[
            tuple[str, str, str, str, str, int, bytes, str]
        ] = []
        for text, raw_vector in vectors.items():
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise EmbeddingCacheError(
                    "Only finite one-dimensional embeddings may be cached"
                )
            payload = vector.tobytes(order="C")
            records.append(
                (
                    cache_fingerprint,
                    text_namespace,
                    text,
                    model_id,
                    model_version,
                    int(vector.size),
                    payload,
                    hashlib.sha256(payload).hexdigest(),
                )
            )
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO embedding_cache_v2 (
                        cache_fingerprint,
                        text_namespace,
                        normalized_text,
                        model_id,
                        model_version,
                        dimension,
                        vector,
                        vector_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def row_count(self) -> int:
        with self._lock:
            connection = self._connect()
            try:
                return int(
                    connection.execute(
                        "SELECT COUNT(*) FROM embedding_cache_v2"
                    ).fetchone()[0]
                )
            finally:
                connection.close()
