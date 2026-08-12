"""Prove embedding scoring observes but cannot alter candidate generation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.embedding.scorer import score_candidate_frame
from sku_mapping.matching.candidate_generator import CandidateGenerator


class _FixedBackend:
    model_id = "test:fixed"
    model_version = "fixed-v1"

    def ensure_available(self) -> None:
        return None

    def encode(self, texts, *, batch_size: int) -> np.ndarray:
        del batch_size
        return np.asarray(
            [
                [float(len(text)), float(sum(map(ord, text)) % 97)]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_embedding_scores_the_exact_retained_candidate_set(tmp_path) -> None:
    offers = preprocess_clickflyer(
        pd.DataFrame(
            [
                {
                    "offerid": "one",
                    "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                    "Product": "Chicken Nuggets-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "Original",
                    "Base Packsize": "400g",
                }
            ]
        )
    )
    master = preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "NUG",
                    "Itemname": "CHICKEN NUGGETS",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Nuggets",
                    "Item Description": "Original nuggets",
                    "Item-Spec": "400g",
                },
                {
                    "Itemcode": "STR",
                    "Itemname": "CHICKEN STRIPS",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Strips",
                    "Item Description": "Original strips",
                    "Item-Spec": "400g",
                },
            ]
        )
    )
    retained = CandidateGenerator(master).generate_candidates_batch(
        offers, top_k=2
    )[0]
    candidate_frame = pd.DataFrame(
        [
            {
                "offer_group_id": "one",
                "candidate_rank": candidate.candidate_rank,
                "master_itemcode": candidate.itemcode,
                "offer_text": offers.iloc[0]["Offer Name"],
                "offer_brand": offers.iloc[0]["Brand Name"],
                "offer_product": offers.iloc[0]["Product"],
                "offer_variant": offers.iloc[0]["Variant"],
                "offer_base_packsize": offers.iloc[0]["Base Packsize"],
                "product_family": offers.iloc[0]["product_family"],
                "master_brand": "Al Kabeer",
                "master_item_description": master.loc[
                    master["Itemcode"].eq(candidate.itemcode), "Itemname"
                ].iloc[0],
                "master_item_family": master.loc[
                    master["Itemcode"].eq(candidate.itemcode), "Item-Cat-4"
                ].iloc[0],
                "master_item_category": candidate.category,
                "master_item_long_description": "",
                "master_item_spec": "400g",
            }
            for candidate in retained
        ]
    )
    identity_before = candidate_frame[
        ["offer_group_id", "candidate_rank", "master_itemcode"]
    ].copy(deep=True)
    base = load_config("config/default.yaml").embedding
    config = replace(
        base,
        enabled=True,
        backend="test",
        model_name="fixed",
        model_version="fixed-v1",
        cache_embeddings=False,
        cache_path=tmp_path / "unused.sqlite3",
    )
    result = score_candidate_frame(
        candidate_frame,
        config=config,
        backend=_FixedBackend(),
    )
    assert result.candidates_scored == len(retained)
    assert result.scores["embedding_similarity"].notna().all()
    pd.testing.assert_frame_equal(
        candidate_frame[
            ["offer_group_id", "candidate_rank", "master_itemcode"]
        ],
        identity_before,
        check_exact=True,
    )
