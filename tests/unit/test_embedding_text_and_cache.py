from __future__ import annotations

import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.embedding.backends import LocalHashingEmbeddingBackend
from sku_mapping.embedding.cache import (
    EmbeddingCacheError,
    PersistentEmbeddingCache,
)
from sku_mapping.embedding.scorer import score_candidate_frame
from sku_mapping.embedding.text import (
    normalize_embedding_text,
    prepare_offer_embedding_text,
)


def _config(tmp_path, **changes):
    base = load_config("config/default.yaml").embedding
    return replace(
        base,
        enabled=True,
        cache_path=tmp_path / "cache.sqlite3",
        **changes,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "offer_group_id": "o1",
                "candidate_rank": 1,
                "master_itemcode": "A",
                "offer_text": "Chicken Nuggets 400 gm",
                "master_item_description": "Chicken Nuggets 400 g",
            },
            {
                "offer_group_id": "o1",
                "candidate_rank": 2,
                "master_itemcode": "B",
                "offer_text": "Chicken Nuggets 400 gm",
                "master_item_description": "Beef Nuggets 400 g",
            },
        ]
    )


def test_unit_and_punctuation_variants_normalize_identically() -> None:
    assert normalize_embedding_text("Chicken Nuggets, 400 GM.") == (
        normalize_embedding_text("chicken nuggets 400 g")
    )


def _offer_text(raw: str, **fields: object) -> str:
    return prepare_offer_embedding_text(
        {
            "offer_text": raw,
            "entity_text": raw,
            "offer_brand": "Al Kabeer",
            **fields,
        }
    )


def test_commercially_distinct_offer_texts_stay_distinct() -> None:
    assert _offer_text(
        "Chicken Nuggets 400 g", entity_protein="chicken"
    ) != _offer_text("Beef Nuggets 400 g", entity_protein="beef")
    assert _offer_text(
        "Chicken Samosas 240 g", entity_protein="chicken"
    ) != _offer_text("Mutton Samosas 240 g", entity_protein="mutton")
    assert _offer_text(
        "Spicy Chicken Wings", offer_variant="spicy"
    ) != _offer_text("Non Spicy Chicken Wings", offer_variant="non spicy")
    assert _offer_text("Chicken Nuggets 1 kg") != _offer_text(
        "Chicken Nuggets 800 g + 200 g free"
    )
    assert _offer_text("Chicken Nuggets 400 g") != _offer_text(
        "Chicken Nuggets twin pack 2 x 400 g"
    )
    assert _offer_text(
        "Chicken Nuggets", entity_product_family="nuggets"
    ) != _offer_text(
        "Chicken Nuggets Krazee line",
        entity_product_family="nuggets",
        offer_variant="krazee",
    )


def test_hashing_vectors_are_deterministic_finite_and_exercised(
    tmp_path,
) -> None:
    config = _config(tmp_path, cache_embeddings=False)
    backend = LocalHashingEmbeddingBackend(config)
    texts = ["Chicken Nuggets 400 g", "دجاج ناجتس 400 g"] * 2
    first = backend.encode(texts, batch_size=2)
    second = backend.encode(texts, batch_size=4)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 384)
    assert np.isfinite(first).all()

    result = score_candidate_frame(
        _frame(), config=config, backend=backend
    )
    assert result.requested and result.available and result.used
    assert result.device == "cpu"
    assert result.vector_dimension == 384
    assert result.scores["embedding_similarity"].notna().all()


def test_cache_isolated_by_namespace_and_text_contract(tmp_path) -> None:
    cache = PersistentEmbeddingCache(tmp_path / "cache.sqlite3")
    first_config = _config(tmp_path, text_construction_version="v1")
    first = score_candidate_frame(
        _frame(),
        config=first_config,
        backend=LocalHashingEmbeddingBackend(first_config),
        cache=cache,
    )
    assert first.cache_hits == 0
    assert cache.row_count() == 3

    repeated = score_candidate_frame(
        _frame(),
        config=first_config,
        backend=LocalHashingEmbeddingBackend(first_config),
        cache=cache,
    )
    assert repeated.cache_hits == 3
    pd.testing.assert_frame_equal(first.scores, repeated.scores)

    changed_config = _config(tmp_path, text_construction_version="v2")
    changed = score_candidate_frame(
        _frame(),
        config=changed_config,
        backend=LocalHashingEmbeddingBackend(changed_config),
        cache=cache,
    )
    assert changed.cache_hits == 0
    assert changed.cache_fingerprint != first.cache_fingerprint


def test_corrupt_cache_fails_explicitly(tmp_path) -> None:
    config = _config(tmp_path)
    cache = PersistentEmbeddingCache(config.cache_path)
    first = score_candidate_frame(
        _frame(),
        config=config,
        backend=LocalHashingEmbeddingBackend(config),
        cache=cache,
    )
    with sqlite3.connect(cache.path) as connection:
        connection.execute(
            "UPDATE embedding_cache_v2 SET vector = ? "
            "WHERE cache_fingerprint = ?",
            (b"corrupt", first.cache_fingerprint),
        )
        connection.commit()
    corrupted = score_candidate_frame(
        _frame(),
        config=config,
        backend=LocalHashingEmbeddingBackend(config),
        cache=cache,
    )
    assert corrupted.status == "UNAVAILABLE"
    assert isinstance(corrupted.error, str)
    assert "checksum" in corrupted.error


def test_batch_order_and_duplicate_identity_are_preserved(tmp_path) -> None:
    frame = pd.concat([_frame(), _frame()], ignore_index=True)
    frame["offer_group_id"] = ["a", "a", "b", "b"]
    config = _config(tmp_path, cache_embeddings=False)
    result = score_candidate_frame(
        frame,
        config=config,
        backend=LocalHashingEmbeddingBackend(config),
    )
    assert result.scores.index.tolist() == frame.index.tolist()
    assert result.scores.loc[0, "embedding_similarity"] == (
        result.scores.loc[2, "embedding_similarity"]
    )


def test_long_non_ascii_case_and_unit_variants_are_stable(tmp_path) -> None:
    config = _config(tmp_path, cache_embeddings=False)
    backend = LocalHashingEmbeddingBackend(config)
    texts = [
        "Chicken Nuggets 400 gm",
        "CHICKEN NUGGETS 400 G",
        "دجاج ناجتس 400 جم",
        "Chicken Nuggets " + ("family pack " * 2_000),
        "nan",
        "infinity",
    ]
    single = np.vstack(
        [backend.encode([text], batch_size=1)[0] for text in texts]
    )
    batch = backend.encode(texts, batch_size=3)
    np.testing.assert_array_equal(single, batch)
    assert batch.shape == (len(texts), 384)
    assert np.isfinite(batch).all()
    assert not np.shares_memory(batch, backend.encode(texts, batch_size=6))


def test_empty_or_zero_vector_input_fails_closed(tmp_path) -> None:
    frame = _frame()
    frame["offer_text"] = ""
    frame["offer_brand"] = ""
    frame["offer_product"] = ""
    frame["offer_variant"] = ""
    frame["offer_base_packsize"] = ""
    config = _config(tmp_path, cache_embeddings=False)
    result = score_candidate_frame(
        frame,
        config=config,
        backend=LocalHashingEmbeddingBackend(config),
    )
    assert result.status == "INVALID_TEXT"
    assert result.available
    assert not result.used
    assert result.scores["embedding_similarity"].isna().all()


def test_cache_model_revision_and_parser_version_never_collide(tmp_path) -> None:
    cache = PersistentEmbeddingCache(tmp_path / "isolated.sqlite3")
    first_config = _config(
        tmp_path,
        model_version="revision-a",
        commercial_parser_version="parser-a",
    )
    first = score_candidate_frame(
        _frame(),
        config=first_config,
        backend=LocalHashingEmbeddingBackend(first_config),
        cache=cache,
    )
    second_config = replace(first_config, model_version="revision-b")
    second = score_candidate_frame(
        _frame(),
        config=second_config,
        backend=LocalHashingEmbeddingBackend(second_config),
        cache=cache,
    )
    third_config = replace(
        first_config, commercial_parser_version="parser-b"
    )
    third = score_candidate_frame(
        _frame(),
        config=third_config,
        backend=LocalHashingEmbeddingBackend(third_config),
        cache=cache,
    )
    assert first.cache_fingerprint != second.cache_fingerprint
    assert first.cache_fingerprint != third.cache_fingerprint
    assert second.cache_hits == 0
    assert third.cache_hits == 0


class _PartiallyFailingBackend(LocalHashingEmbeddingBackend):
    def encode(self, texts, *, batch_size):
        if any("beef" in str(text).lower() for text in texts):
            raise RuntimeError("intentional partial-batch sentinel")
        return super().encode(texts, batch_size=batch_size)


class _NonFiniteBackend(LocalHashingEmbeddingBackend):
    def encode(self, texts, *, batch_size):
        result = super().encode(texts, batch_size=batch_size)
        result[0, 0] = np.inf
        return result


def test_partial_batch_failure_does_not_attach_partial_scores(tmp_path) -> None:
    config = _config(tmp_path, cache_embeddings=False)
    result = score_candidate_frame(
        _frame(),
        config=config,
        backend=_PartiallyFailingBackend(config),
    )
    assert result.status == "UNAVAILABLE"
    assert result.candidates_scored == 0
    assert result.scores["embedding_similarity"].isna().all()
    assert "partial-batch sentinel" in str(result.error)


def test_non_finite_vector_fails_closed(tmp_path) -> None:
    config = _config(tmp_path, cache_embeddings=False)
    result = score_candidate_frame(
        _frame(), config=config, backend=_NonFiniteBackend(config)
    )
    assert result.status == "UNAVAILABLE"
    assert result.candidates_scored == 0
    assert result.scores["embedding_similarity"].isna().all()
    assert "invalid vectors" in str(result.error)
