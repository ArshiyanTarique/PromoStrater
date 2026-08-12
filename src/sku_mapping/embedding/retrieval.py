"""Bounded, deterministic embedding retrieval over the Product Master."""

from __future__ import annotations

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
from sku_mapping.embedding.scorer import embedding_cache_fingerprint
from sku_mapping.embedding.text import (
    prepare_candidate_embedding_text,
    prepare_offer_embedding_text,
)
from sku_mapping.features.commercial_attributes import (
    attributes_json,
    parse_master_attributes,
    parse_source_attributes,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingRetrievalHit:
    master_itemcode: str
    similarity: float
    embedding_rank: int


@dataclass(frozen=True)
class EmbeddingRetrievalResult:
    status: str
    hits: tuple[tuple[EmbeddingRetrievalHit, ...], ...]
    requested: bool
    available: bool
    used: bool
    runtime_seconds: float
    cache_hits: int
    cache_misses: int
    offers_retrieved: int
    master_vectors: int
    device: str
    model_id: str
    model_version: str
    error: str | None = None


def _offer_record(row: pd.Series) -> dict[str, object]:
    return {
        "offer_brand": row.get("Brand Name", ""),
        "offer_product": row.get("Product", ""),
        "offer_variant": row.get("Variant", ""),
        "offer_text": row.get("Offer Name", ""),
        "offer_base_packsize": row.get("Base Packsize", ""),
        "product_family": row.get("product_family", ""),
        "entity_text": row.get("entity_text", ""),
        "entity_protein": row.get("entity_protein", ""),
        "entity_product_family": row.get(
            "entity_product_family", ""
        ),
        "entity_retail_weight_g": row.get(
            "entity_retail_weight_g", ""
        ),
        "conjunction_type": row.get("conjunction_type", ""),
        "source_commercial_attributes": attributes_json(
            parse_source_attributes(row)
        ),
    }


def _master_record(row: pd.Series) -> dict[str, object]:
    return {
        "master_brand": row.get("Brand Name", "Al Kabeer"),
        "master_item_description": row.get("Itemname", ""),
        "master_item_family": row.get("Item-Cat-4", ""),
        "master_item_category": row.get("Item-Cat-2", ""),
        "master_item_long_description": row.get(
            "Item Description", ""
        ),
        "master_item_spec": row.get("Item-Spec", ""),
        "master_commercial_attributes": attributes_json(
            parse_master_attributes(row)
        ),
    }


def _unit_normalize(vectors: np.ndarray) -> np.ndarray:
    if (
        vectors.ndim != 2
        or vectors.shape[1] < 1
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Embedding retrieval received invalid vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError(
            "Embedding retrieval received a zero-length vector"
        )
    return vectors / norms


def retrieve_embedding_candidates(
    offers: pd.DataFrame,
    master: pd.DataFrame,
    *,
    config: EmbeddingConfig,
    backend: EmbeddingBackend | None = None,
    cache: PersistentEmbeddingCache | None = None,
) -> EmbeddingRetrievalResult:
    """Return bounded master hits per offer without changing source ordering."""
    started = time.perf_counter()
    empty_hits = tuple(() for _ in range(len(offers)))
    if not config.enabled or not config.retrieval_enabled:
        return EmbeddingRetrievalResult(
            status="DISABLED",
            hits=empty_hits,
            requested=False,
            available=False,
            used=False,
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=0,
            offers_retrieved=0,
            master_vectors=0,
            device="",
            model_id="",
            model_version="",
        )
    if offers.empty or master.empty:
        return EmbeddingRetrievalResult(
            status="SKIPPED_EMPTY",
            hits=empty_hits,
            requested=True,
            available=False,
            used=False,
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=0,
            offers_retrieved=0,
            master_vectors=0,
            device=config.device,
            model_id=f"{config.backend}:{config.model_name}",
            model_version=config.model_version or "UNRESOLVED",
        )
    effective_backend = backend
    try:
        if effective_backend is None:
            effective_backend = create_embedding_backend(config)
        effective_backend.ensure_available()
        model_id = effective_backend.model_id
        model_version = effective_backend.model_version
        fingerprint = embedding_cache_fingerprint(
            config, model_id=model_id, model_version=model_version
        )
        offer_texts = [
            prepare_offer_embedding_text(_offer_record(row))
            for _, row in offers.iterrows()
        ]
        master_texts = [
            prepare_candidate_embedding_text(_master_record(row))
            for _, row in master.iterrows()
        ]
        if any(not text for text in [*offer_texts, *master_texts]):
            raise ValueError("Embedding retrieval text is empty")
        unique_offers = list(dict.fromkeys(offer_texts))
        unique_master = list(dict.fromkeys(master_texts))
        effective_cache = cache
        if config.cache_embeddings and effective_cache is None:
            effective_cache = PersistentEmbeddingCache(config.cache_path)
        offer_map: dict[str, np.ndarray] = {}
        master_map: dict[str, np.ndarray] = {}
        if effective_cache is not None:
            offer_map.update(
                effective_cache.get_many(
                    unique_offers,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="offer",
                )
            )
            master_map.update(
                effective_cache.get_many(
                    unique_master,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="master",
                )
            )
        cache_hits = len(offer_map) + len(master_map)
        missing = [
            *(("offer", text) for text in unique_offers if text not in offer_map),
            *(("master", text) for text in unique_master if text not in master_map),
        ]
        if missing:
            encoded = np.asarray(
                effective_backend.encode(
                    [text for _, text in missing],
                    batch_size=config.batch_size,
                ),
                dtype=np.float32,
            )
            encoded = _unit_normalize(encoded)
            new_offers = {
                text: encoded[position]
                for position, (namespace, text) in enumerate(missing)
                if namespace == "offer"
            }
            new_master = {
                text: encoded[position]
                for position, (namespace, text) in enumerate(missing)
                if namespace == "master"
            }
            offer_map.update(new_offers)
            master_map.update(new_master)
            if effective_cache is not None:
                effective_cache.put_many(
                    new_offers,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="offer",
                )
                effective_cache.put_many(
                    new_master,
                    model_id=model_id,
                    model_version=model_version,
                    cache_fingerprint=fingerprint,
                    text_namespace="master",
                )
        master_vectors = np.vstack([master_map[text] for text in master_texts])
        master_codes = master["Itemcode"].astype(str).tolist()
        master_categories = master["category"].astype(str).tolist()
        output: list[tuple[EmbeddingRetrievalHit, ...]] = []
        offer_vectors = np.vstack([offer_map[text] for text in offer_texts])
        batch_size = config.retrieval_offer_batch_size
        for start in range(0, len(offers), batch_size):
            stop = min(len(offers), start + batch_size)
            similarities = offer_vectors[start:stop] @ master_vectors.T
            for local_position, scores in enumerate(similarities):
                offer_position = start + local_position
                category = str(offers.iloc[offer_position].get("category", ""))
                eligible = np.array(
                    [
                        value == category or category == "Other"
                        for value in master_categories
                    ],
                    dtype=bool,
                )
                positions = np.flatnonzero(eligible)
                ordered = sorted(
                    positions.tolist(),
                    key=lambda position: (
                        -float(scores[position]),
                        master_codes[position],
                    ),
                )[: config.retrieval_top_k]
                output.append(
                    tuple(
                        EmbeddingRetrievalHit(
                            master_itemcode=master_codes[position],
                            similarity=float(scores[position]),
                            embedding_rank=rank,
                        )
                        for rank, position in enumerate(ordered, start=1)
                    )
                )
        return EmbeddingRetrievalResult(
            status="COMPLETED",
            hits=tuple(output),
            requested=True,
            available=True,
            used=True,
            runtime_seconds=time.perf_counter() - started,
            cache_hits=cache_hits,
            cache_misses=len(missing),
            offers_retrieved=len(offers),
            master_vectors=len(master),
            device=str(getattr(effective_backend, "device", config.device)),
            model_id=model_id,
            model_version=model_version,
        )
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        LOGGER.exception(
            "Embedding retrieval unavailable; fuzzy candidates remain active "
            "backend=%s model=%s",
            config.backend,
            config.model_name,
        )
        return EmbeddingRetrievalResult(
            status="UNAVAILABLE",
            hits=empty_hits,
            requested=True,
            available=False,
            used=False,
            runtime_seconds=time.perf_counter() - started,
            cache_hits=0,
            cache_misses=0,
            offers_retrieved=0,
            master_vectors=0,
            device=config.device,
            model_id=f"{config.backend}:{config.model_name}",
            model_version=config.model_version or "UNRESOLVED",
            error=reason,
        )
