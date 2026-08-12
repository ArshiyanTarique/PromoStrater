from __future__ import annotations

from dataclasses import replace

import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import preprocess_product_master
from sku_mapping.embedding.backends import (
    EmbeddingBackendError,
    LocalHashingEmbeddingBackend,
)
from sku_mapping.embedding.retrieval import retrieve_embedding_candidates


def _config(tmp_path):
    base = load_config("config/default.yaml").embedding
    return replace(
        base,
        enabled=True,
        retrieval_enabled=True,
        retrieval_top_k=2,
        cache_path=tmp_path / "retrieval.sqlite3",
    )


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Offer Name": "Chicken Nuggets 400 g",
                "Product": "Chicken Nuggets",
                "Brand Name": "Al Kabeer",
                "Variant": "",
                "Base Packsize": "400 g",
                "category": "Other",
                "product_family": "chicken nuggets",
            },
            {
                "Offer Name": "Beef Burger Patty 400 g",
                "Product": "Beef Burger Patty",
                "Brand Name": "Al Kabeer",
                "Variant": "",
                "Base Packsize": "400 g",
                "category": "Other",
                "product_family": "beef burger patty",
            },
        ]
    )


def _master() -> pd.DataFrame:
    return preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "CHICKEN",
                    "Itemname": "Chicken Nuggets 400 g",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Nuggets",
                    "Item Description": "Chicken Nuggets",
                    "Item-Spec": "400 g x 20",
                },
                {
                    "Itemcode": "BEEF",
                    "Itemname": "Beef Burger Patty 400 g",
                    "Item-Cat-2": "Meat",
                    "Item-Cat-4": "Burger Patty",
                    "Item Description": "Beef Burger Patty",
                    "Item-Spec": "400 g x 20",
                },
            ]
        )
    )


def test_retrieval_keeps_offer_vector_and_candidate_identity_aligned(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    first = retrieve_embedding_candidates(
        _offers(),
        _master(),
        config=config,
        backend=LocalHashingEmbeddingBackend(config),
    )
    second = retrieve_embedding_candidates(
        _offers().iloc[::-1].reset_index(drop=True),
        _master(),
        config=config,
        backend=LocalHashingEmbeddingBackend(config),
    )
    assert first.requested and first.available and first.used
    assert [hits[0].master_itemcode for hits in first.hits] == [
        "CHICKEN",
        "BEEF",
    ]
    assert [hits[0].master_itemcode for hits in second.hits] == [
        "BEEF",
        "CHICKEN",
    ]
    assert all(len(hits) == 2 for hits in first.hits)


class _UnavailableBackend:
    model_id = "unavailable:test"
    model_version = "v1"
    device = "cpu"

    def ensure_available(self) -> None:
        raise EmbeddingBackendError("intentional local model failure")

    def encode(self, texts, *, batch_size):
        raise AssertionError("encode must not run")


def test_unavailable_retrieval_fails_closed_without_fake_hits(tmp_path) -> None:
    result = retrieve_embedding_candidates(
        _offers(),
        _master(),
        config=_config(tmp_path),
        backend=_UnavailableBackend(),
    )
    assert result.status == "UNAVAILABLE"
    assert result.requested
    assert not result.available
    assert not result.used
    assert result.hits == ((), ())
    assert "intentional local model failure" in str(result.error)
