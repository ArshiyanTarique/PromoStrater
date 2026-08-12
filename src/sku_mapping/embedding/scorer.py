"""Independent scoring of the exact candidate rows retained upstream."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sku_mapping.config import EmbeddingConfig
from sku_mapping.embedding.backends import (
    EmbeddingBackend,
    create_embedding_backend,
)
from sku_mapping.embedding.cache import PersistentEmbeddingCache
from sku_mapping.embedding.text import (
    prepare_candidate_embedding_text,
    prepare_offer_embedding_text,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_CANDIDATE_COLUMNS = frozenset(
    {
        "offer_group_id",
        "candidate_rank",
        "master_itemcode",
        "offer_text",
        "master_item_description",
    }
)

EMBEDDING_SCORE_COLUMNS = (
    "embedding_status",
    "embedding_model_id",
    "embedding_model_version",
    "offer_text_used",
    "candidate_text_used",
    "embedding_similarity",
    "embedding_rank",
    "embedding_top_candidate",
    "embedding_failure_reason",
)


@dataclass(frozen=True)
class EmbeddingScoreResult:
    """Embedding scoring outcome with explicit availability and provenance."""

    status: str
    scores: pd.DataFrame
    candidates_scored: int
    failures: int
    runtime_seconds: float
    cache_hits: int
    cache_misses: int
    error: str | None = None
    requested: bool = False
    available: bool = False
    used: bool = False
    device: str = ""
    vector_dimension: int | None = None
    cache_fingerprint: str = ""


def embedding_cache_fingerprint(
    config: EmbeddingConfig,
    *,
    model_id: str,
    model_version: str,
) -> str:
    """Hash every setting capable of changing a cached vector."""
    payload = {
        "model_id": model_id,
        "model_version": model_version,
        "pooling_strategy": config.pooling_strategy,
        "normalize_vectors": config.normalize_vectors,
        "maximum_sequence_length": config.max_sequence_length,
        "text_construction_version": config.text_construction_version,
        "commercial_parser_version": config.commercial_parser_version,
        "similarity_metric": config.similarity_metric,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _failure_scores(
    candidates: pd.DataFrame,
    *,
    model_id: str,
    model_version: str,
    status: str,
    reason: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "embedding_status": status,
            "embedding_model_id": model_id,
            "embedding_model_version": model_version,
            "offer_text_used": [
                prepare_offer_embedding_text(row)
                for _, row in candidates.iterrows()
            ],
            "candidate_text_used": [
                prepare_candidate_embedding_text(row)
                for _, row in candidates.iterrows()
            ],
            "embedding_similarity": np.nan,
            "embedding_rank": pd.array(
                [pd.NA] * len(candidates), dtype="Int64"
            ),
            "embedding_top_candidate": False,
            "embedding_failure_reason": reason,
        },
        index=candidates.index,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    scores = np.divide(
        np.einsum("ij,ij->i", left, right),
        denominator,
        out=np.zeros(len(left), dtype=float),
        where=denominator > 0,
    )
    return np.clip(scores, -1.0, 1.0)


def _rank_scores(candidates: pd.DataFrame, scores: pd.DataFrame) -> None:
    scores["embedding_rank"] = pd.array(
        [pd.NA] * len(scores), dtype="Int64"
    )
    scores["embedding_top_candidate"] = False
    working = candidates.loc[
        :,
        ["offer_group_id", "candidate_rank", "master_itemcode"],
    ].copy()
    working["embedding_similarity"] = scores["embedding_similarity"]
    for _, positions in working.groupby(
        "offer_group_id", sort=False
    ).groups.items():
        ordered = working.loc[list(positions)].sort_values(
            [
                "embedding_similarity",
                "candidate_rank",
                "master_itemcode",
            ],
            ascending=[False, True, True],
            kind="stable",
        )
        ranks = pd.Series(
            range(1, len(ordered) + 1),
            index=ordered.index,
            dtype="Int64",
        )
        scores.loc[ordered.index, "embedding_rank"] = ranks
        scores.loc[ordered.index[0], "embedding_top_candidate"] = True


def score_candidate_frame(
    candidates: pd.DataFrame,
    *,
    config: EmbeddingConfig,
    backend: EmbeddingBackend | None = None,
    cache: PersistentEmbeddingCache | None = None,
) -> EmbeddingScoreResult:
    """Score retained candidates without generating, filtering, or deciding."""
    started = time.perf_counter()
    if not config.enabled:
        return EmbeddingScoreResult(
            status="DISABLED",
            scores=pd.DataFrame(index=candidates.index),
            candidates_scored=0,
            failures=0,
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=0,
            requested=False,
            available=False,
            used=False,
        )
    missing = sorted(REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(
            f"Embedding candidate frame is missing required columns: {missing}"
        )
    if candidates.empty:
        return EmbeddingScoreResult(
            status="SKIPPED_EMPTY",
            scores=pd.DataFrame(columns=EMBEDDING_SCORE_COLUMNS),
            candidates_scored=0,
            failures=0,
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=0,
            requested=True,
            available=False,
            used=False,
        )

    effective_backend = backend
    configured_id = f"{config.backend}:{config.model_name}"
    try:
        if effective_backend is None:
            effective_backend = create_embedding_backend(config)
        effective_backend.ensure_available()
        model_id = effective_backend.model_id
        model_version = effective_backend.model_version
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        LOGGER.warning(
            "Embedding backend unavailable backend=%s model=%s version=%s "
            "device=%s reason=%s",
            config.backend,
            config.model_name,
            config.model_version or "UNRESOLVED",
            config.device,
            reason,
        )
        scores = _failure_scores(
            candidates,
            model_id=configured_id,
            model_version=config.model_version or "UNAVAILABLE",
            status="UNAVAILABLE",
            reason=reason,
        )
        return EmbeddingScoreResult(
            status="UNAVAILABLE",
            scores=scores,
            candidates_scored=0,
            failures=len(candidates),
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=0,
            error=reason,
            requested=True,
            available=False,
            used=False,
            device=config.device,
        )

    offer_texts = [
        prepare_offer_embedding_text(row)
        for _, row in candidates.iterrows()
    ]
    candidate_texts = [
        prepare_candidate_embedding_text(row)
        for _, row in candidates.iterrows()
    ]
    if any(not text.strip() for text in [*offer_texts, *candidate_texts]):
        reason = (
            "Embedding text construction produced an empty offer or master text"
        )
        scores = _failure_scores(
            candidates,
            model_id=model_id,
            model_version=model_version,
            status="INVALID_TEXT",
            reason=reason,
        )
        return EmbeddingScoreResult(
            status="INVALID_TEXT",
            scores=scores,
            candidates_scored=0,
            failures=len(candidates),
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=0,
            error=reason,
            requested=True,
            available=True,
            used=False,
            device=str(getattr(effective_backend, "device", config.device)),
        )
    unique_offer_texts = list(dict.fromkeys(offer_texts))
    unique_candidate_texts = list(dict.fromkeys(candidate_texts))
    effective_cache = cache
    if config.cache_embeddings and effective_cache is None:
        effective_cache = PersistentEmbeddingCache(config.cache_path)
    offer_vectors_by_text: dict[str, np.ndarray] = {}
    candidate_vectors_by_text: dict[str, np.ndarray] = {}
    fingerprint = embedding_cache_fingerprint(
        config, model_id=model_id, model_version=model_version
    )
    try:
        if effective_cache is not None:
            offer_vectors_by_text.update(
                effective_cache.get_many(
                    unique_offer_texts,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="offer",
                )
            )
            candidate_vectors_by_text.update(
                effective_cache.get_many(
                    unique_candidate_texts,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="master",
                )
            )
        cache_hits = (
            len(offer_vectors_by_text) + len(candidate_vectors_by_text)
        )
        missing_offers = [
            text
            for text in unique_offer_texts
            if text not in offer_vectors_by_text
        ]
        missing_candidates = [
            text
            for text in unique_candidate_texts
            if text not in candidate_vectors_by_text
        ]
        missing_items = [
            *(("offer", text) for text in missing_offers),
            *(("master", text) for text in missing_candidates),
        ]
        if missing_items:
            encoded = np.asarray(
                effective_backend.encode(
                    [text for _, text in missing_items],
                    batch_size=config.batch_size,
                ),
                dtype=np.float32,
            )
            if (
                encoded.ndim != 2
                or encoded.shape[0] != len(missing_items)
                or encoded.shape[1] < 1
                or not np.isfinite(encoded).all()
            ):
                raise ValueError(
                    "Embedding backend returned invalid vectors"
                )
            if config.normalize_vectors:
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                if (norms <= 0).any():
                    raise ValueError(
                        "Embedding backend returned a zero-length vector"
                    )
                encoded = encoded / norms
            new_offer_vectors = {
                text: encoded[position]
                for position, (namespace, text) in enumerate(missing_items)
                if namespace == "offer"
            }
            new_candidate_vectors = {
                text: encoded[position]
                for position, (namespace, text) in enumerate(missing_items)
                if namespace == "master"
            }
            offer_vectors_by_text.update(new_offer_vectors)
            candidate_vectors_by_text.update(new_candidate_vectors)
            if effective_cache is not None:
                effective_cache.put_many(
                    new_offer_vectors,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="offer",
                )
                effective_cache.put_many(
                    new_candidate_vectors,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="master",
                )
        dimensions = {
            vector.shape
            for vector in [
                *offer_vectors_by_text.values(),
                *candidate_vectors_by_text.values(),
            ]
        }
        if len(dimensions) != 1:
            raise ValueError("Embedding dimensions are inconsistent")
        offer_vectors = np.vstack(
            [offer_vectors_by_text[text] for text in offer_texts]
        )
        candidate_vectors = np.vstack(
            [candidate_vectors_by_text[text] for text in candidate_texts]
        )
        similarities = _cosine(offer_vectors, candidate_vectors)
    except Exception as error:
        LOGGER.exception(
            "Embedding scoring failed; LightGBM decisions remain unchanged"
        )
        reason = f"{type(error).__name__}: {error}"
        scores = _failure_scores(
            candidates,
            model_id=model_id,
            model_version=model_version,
            status="UNAVAILABLE",
            reason=reason,
        )
        return EmbeddingScoreResult(
            status="UNAVAILABLE",
            scores=scores,
            candidates_scored=0,
            failures=len(candidates),
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=(
                len(unique_offer_texts) + len(unique_candidate_texts)
            ),
            error=reason,
            requested=True,
            available=False,
            used=False,
            device=str(getattr(effective_backend, "device", config.device)),
            cache_fingerprint=fingerprint,
        )

    scores = pd.DataFrame(
        {
            "embedding_status": "COMPLETED",
            "embedding_model_id": model_id,
            "embedding_model_version": model_version,
            "offer_text_used": offer_texts,
            "candidate_text_used": candidate_texts,
            "embedding_similarity": similarities,
            "embedding_failure_reason": "",
        },
        index=candidates.index,
    )
    _rank_scores(candidates, scores)
    scores = scores.loc[:, EMBEDDING_SCORE_COLUMNS]
    return EmbeddingScoreResult(
        status="COMPLETED",
        scores=scores,
        candidates_scored=len(candidates),
        failures=0,
        runtime_seconds=time.perf_counter() - started,
        cache_hits=cache_hits,
        cache_misses=len(missing_items),
        requested=True,
        available=True,
        used=True,
        device=str(getattr(effective_backend, "device", config.device)),
        vector_dimension=int(offer_vectors.shape[1]),
        cache_fingerprint=fingerprint,
    )


def score_candidate_frame_non_blocking(
    candidates: pd.DataFrame,
    *,
    config: EmbeddingConfig,
    backend: EmbeddingBackend | None = None,
    cache: PersistentEmbeddingCache | None = None,
) -> EmbeddingScoreResult:
    """Never allow embedding infrastructure to fail the caller."""
    try:
        return score_candidate_frame(
            candidates,
            config=config,
            backend=backend,
            cache=cache,
        )
    except Exception as error:
        LOGGER.exception(
            "Embedding scorer failed before scoring; continuing without it"
        )
        reason = f"{type(error).__name__}: {error}"
        scores = (
            _failure_scores(
                candidates,
                model_id=f"{config.backend}:{config.model_name}",
                model_version=config.model_version or "UNAVAILABLE",
                status="UNAVAILABLE",
                reason=reason,
            )
            if config.enabled
            else pd.DataFrame(index=candidates.index)
        )
        return EmbeddingScoreResult(
            status="UNAVAILABLE" if config.enabled else "DISABLED",
            scores=scores,
            candidates_scored=0,
            failures=len(candidates) if config.enabled else 0,
            runtime_seconds=0.0,
            cache_hits=0,
            cache_misses=0,
            error=reason,
            requested=config.enabled,
            available=False,
            used=False,
            device=config.device,
        )
