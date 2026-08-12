"""Independent embedding scorer, ranking, cache, and failure tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.embedding.cache import PersistentEmbeddingCache
from sku_mapping.embedding.scorer import (
    score_candidate_frame,
    score_candidate_frame_non_blocking,
)
from sku_mapping.embedding.text import (
    prepare_candidate_embedding_text,
    prepare_offer_embedding_text,
)


class _KeywordBackend:
    def __init__(
        self, *, version: str = "test-v1", fail: bool = False
    ) -> None:
        self._version = version
        self.fail = fail
        self.encoded_texts: list[str] = []

    @property
    def model_id(self) -> str:
        return "test:keyword-embedding"

    @property
    def model_version(self) -> str:
        return self._version

    def ensure_available(self) -> None:
        if self.fail:
            raise RuntimeError("backend unavailable")

    def encode(self, texts, *, batch_size: int) -> np.ndarray:
        del batch_size
        self.encoded_texts.extend(texts)
        vectors = []
        for text in texts:
            if "nuggets" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "strips" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _config(tmp_path, *, enabled: bool = True):
    config = load_config("config/default.yaml")
    return replace(
        config.embedding,
        enabled=enabled,
        backend="test",
        model_name="keyword-embedding",
        model_version="test-v1",
        cache_path=tmp_path / "embedding-cache.sqlite3",
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "offer_group_id": "offer-1",
                "candidate_rank": 1,
                "master_itemcode": "STR",
                "offer_text": "Al Kabeer Chicken Nuggets 400g",
                "offer_brand": "Al Kabeer",
                "offer_product": "Chicken Nuggets-Frozen",
                "offer_variant": "Original",
                "offer_base_packsize": "400 g",
                "product_family": "chicken nuggets",
                "master_brand": "Al Kabeer",
                "master_item_description": "CHICKEN STRIPS",
                "master_item_family": "Strips",
                "master_item_category": "Chicken",
                "master_item_long_description": "Spicy strips",
                "master_item_spec": "400 Gms x 20 Pkts",
                "calibrated_probability": 0.91,
            },
            {
                "offer_group_id": "offer-1",
                "candidate_rank": 2,
                "master_itemcode": "NUG",
                "offer_text": "Al Kabeer Chicken Nuggets 400g",
                "offer_brand": "Al Kabeer",
                "offer_product": "Chicken Nuggets-Frozen",
                "offer_variant": "Original",
                "offer_base_packsize": "400 g",
                "product_family": "chicken nuggets",
                "master_brand": "Al Kabeer",
                "master_item_description": "CHICKEN NUGGETS",
                "master_item_family": "Nuggets",
                "master_item_category": "Chicken",
                "master_item_long_description": "Original nuggets",
                "master_item_spec": "400 Gms x 20 Pkts",
                "calibrated_probability": 0.70,
            },
            {
                "offer_group_id": "offer-1",
                "candidate_rank": 3,
                "master_itemcode": "POP",
                "offer_text": "Al Kabeer Chicken Nuggets 400g",
                "offer_brand": "Al Kabeer",
                "offer_product": "Chicken Nuggets-Frozen",
                "offer_variant": "Original",
                "offer_base_packsize": "400 g",
                "product_family": "chicken nuggets",
                "master_brand": "Al Kabeer",
                "master_item_description": "CHICKEN POPCORN",
                "master_item_family": "Popcorn",
                "master_item_category": "Chicken",
                "master_item_long_description": "Original popcorn",
                "master_item_spec": "400 Gms x 20 Pkts",
                "calibrated_probability": 0.60,
            },
        ]
    )


def test_every_candidate_is_scored_and_embedding_ranking_is_independent(
    tmp_path,
) -> None:
    candidates = _candidates()
    before = candidates.copy(deep=True)
    result = score_candidate_frame(
        candidates,
        config=_config(tmp_path),
        backend=_KeywordBackend(),
    )
    assert result.status == "COMPLETED"
    assert result.candidates_scored == len(candidates)
    assert result.scores["embedding_similarity"].notna().all()
    top = candidates.loc[
        result.scores["embedding_top_candidate"], "master_itemcode"
    ].tolist()
    assert top == ["NUG"]
    assert result.scores["embedding_rank"].tolist() == [2, 1, 3]
    assert candidates.loc[
        candidates["calibrated_probability"].idxmax(), "master_itemcode"
    ] == "STR"
    pd.testing.assert_frame_equal(candidates, before, check_exact=True)


def test_ranking_is_deterministic_for_fixed_inputs(tmp_path) -> None:
    candidates = _candidates()
    first = score_candidate_frame(
        candidates,
        config=replace(_config(tmp_path), cache_embeddings=False),
        backend=_KeywordBackend(),
    ).scores
    second = score_candidate_frame(
        candidates,
        config=replace(_config(tmp_path), cache_embeddings=False),
        backend=_KeywordBackend(),
    ).scores
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_cache_key_includes_model_identity_and_version(tmp_path) -> None:
    candidates = _candidates()
    cache = PersistentEmbeddingCache(tmp_path / "vectors.sqlite3")
    version_one = _KeywordBackend(version="v1")
    first = score_candidate_frame(
        candidates,
        config=replace(_config(tmp_path), model_version="v1"),
        backend=version_one,
        cache=cache,
    )
    assert first.cache_misses > 0

    same_version = _KeywordBackend(version="v1")
    repeated = score_candidate_frame(
        candidates,
        config=replace(_config(tmp_path), model_version="v1"),
        backend=same_version,
        cache=cache,
    )
    assert repeated.cache_hits == first.cache_misses
    assert same_version.encoded_texts == []

    new_version = _KeywordBackend(version="v2")
    changed = score_candidate_frame(
        candidates,
        config=replace(_config(tmp_path), model_version="v2"),
        backend=new_version,
        cache=cache,
    )
    assert changed.cache_hits == 0
    assert new_version.encoded_texts


def test_backend_failure_is_explicit_and_nonfatal(tmp_path) -> None:
    candidates = _candidates()
    result = score_candidate_frame_non_blocking(
        candidates,
        config=_config(tmp_path),
        backend=_KeywordBackend(fail=True),
    )
    assert result.status == "UNAVAILABLE"
    assert result.failures == len(candidates)
    assert result.scores["embedding_similarity"].isna().all()
    assert result.scores["embedding_failure_reason"].str.contains(
        "backend unavailable", regex=False
    ).all()
    assert not result.scores["embedding_top_candidate"].any()


def test_disabled_embedding_mode_preserves_existing_behavior(tmp_path) -> None:
    candidates = _candidates()
    before = candidates.copy(deep=True)
    result = score_candidate_frame(
        candidates,
        config=_config(tmp_path, enabled=False),
        backend=_KeywordBackend(fail=True),
    )
    assert result.status == "DISABLED"
    assert result.scores.empty
    pd.testing.assert_frame_equal(candidates, before, check_exact=True)


def test_text_preparation_preserves_product_and_pack_information() -> None:
    candidate = _candidates().iloc[0]
    offer_text = prepare_offer_embedding_text(candidate)
    master_text = prepare_candidate_embedding_text(candidate)
    for token in (
        "al kabeer",
        "chicken",
        "nuggets",
        "original",
        "400 g",
    ):
        assert token in offer_text
    for token in ("chicken", "strips", "400 g x 20 pkts"):
        assert token in master_text
