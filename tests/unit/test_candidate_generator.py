"""Fixture-based tests for RapidFuzz candidate ranking."""

from __future__ import annotations

import pandas as pd
import numpy as np

from sku_mapping.data.preprocessing import preprocess_clickflyer, preprocess_product_master
from sku_mapping.matching import candidate_generator
from sku_mapping.matching.candidate_generator import (
    generate_best_candidate,
    generate_candidates_batch,
    generate_top_candidates,
)


def _master(rows: list[dict[str, str]]) -> pd.DataFrame:
    base = []
    for row in rows:
        base.append(
            {
                "Itemcode": row["Itemcode"],
                "Itemname": row["Itemname"],
                "Item-Cat-2": row["Item-Cat-2"],
                "Item-Cat-4": row.get("Item-Cat-4", ""),
                "Item Description": row.get("Item Description", ""),
                "Item-Spec": row.get("Item-Spec", ""),
            }
        )
    return preprocess_product_master(pd.DataFrame(base))


def _offer(
    name: str,
    product: str,
    pack: str = "",
    variant: str = "No Variant",
) -> pd.Series:
    frame = pd.DataFrame(
        [{
            "Offer Name": name,
            "Product": product,
            "Brand Name": "Al Kabeer",
            "Variant": variant,
            "Base Packsize": pack,
        }]
    )
    return preprocess_clickflyer(frame).iloc[0]


def test_exact_product_ranks_matching_sku_first() -> None:
    master = _master(
        [
            {"Itemcode": "CH-400", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g x 20"},
            {"Itemcode": "CH-750", "Itemname": "CHICKEN STRIPS", "Item-Cat-2": "Chicken", "Item-Spec": "750g x 20"},
        ]
    )
    candidate = generate_best_candidate(
        _offer("Al Kabeer Chicken Nuggets 400g", "Chicken Nuggets-Frozen", "400g"),
        master,
    )
    assert candidate.itemcode == "CH-400"
    assert candidate.pack_status is True
    assert candidate.candidate_rank == 1
    assert candidate.adjusted_score == candidate.text_score + 4.0


def test_category_gate_prevents_chicken_beef_and_fish_fillet_confusion() -> None:
    master = _master(
        [
            {"Itemcode": "CH-FILLET", "Itemname": "CHICKEN FILLET", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
            {"Itemcode": "BF-FILLET", "Itemname": "BEEF FILLET", "Item-Cat-2": "Meat", "Item-Spec": "400g"},
            {"Itemcode": "FS-FILLET", "Itemname": "FISH FILLET", "Item-Cat-2": "Seafood", "Item-Spec": "400g"},
        ]
    )
    candidate = generate_best_candidate(
        _offer("Al Kabeer Chicken Fillet 400g", "Chicken Fillet-Frozen", "400g"),
        master,
    )
    assert candidate.itemcode == "CH-FILLET"


def test_other_category_uses_full_master_pool_and_retains_ranked_candidate() -> None:
    master = _master(
        [{"Itemcode": "MYST", "Itemname": "MYSTERY SNACK", "Item-Cat-2": "Chicken", "Item-Spec": "400g"}]
    )
    candidate = generate_best_candidate(_offer("Mystery Snack", "Unknown Product", "400g"), master)
    assert candidate.category == "Other"
    assert candidate.itemcode == "MYST"


def test_top_candidates_exclude_known_pack_mismatch_when_compatible_option_exists() -> None:
    master = _master(
        [
            {"Itemcode": "BAD", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "750g"},
            {"Itemcode": "GOOD", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "500g"},
        ]
    )
    candidates = generate_top_candidates(
        _offer("Al Kabeer Chicken Nuggets 2 x 500g", "Chicken Nuggets-Frozen", "2 x 500g"),
        master,
        top_k=5,
    )
    assert [candidate.itemcode for candidate in candidates] == ["GOOD"]
    assert candidates[0].pack_structure_status is True


def test_twin_pack_and_plain_one_kilogram_have_structure_conflict() -> None:
    master = _master(
        [{"Itemcode": "ONE-KG", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "1kg"}]
    )
    candidate = generate_best_candidate(
        _offer("Al Kabeer Chicken Nuggets 2 x 500g", "Chicken Nuggets-Frozen", "2 x 500g"),
        master,
    )
    assert candidate.pack_status is True
    assert candidate.pack_structure_status is False


def test_all_incompatible_packs_remain_ranked_for_explicit_conflict_handling() -> None:
    master = _master(
        [{"Itemcode": "400", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"}]
    )
    candidate = generate_best_candidate(
        _offer("Al Kabeer Chicken Nuggets 1kg", "Chicken Nuggets-Frozen", "1kg"),
        master,
    )
    assert candidate.itemcode == "400"
    assert candidate.pack_status is False
    assert candidate.all_candidates_incompatible is True
    assert candidate.adjusted_score == candidate.text_score


def test_unknown_pack_is_retained_with_unknown_status() -> None:
    master = _master(
        [{"Itemcode": "400", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"}]
    )
    candidate = generate_best_candidate(_offer("Al Kabeer Chicken Nuggets", "Chicken Nuggets-Frozen"), master)
    assert candidate.pack_status is None
    assert candidate.adjusted_score == candidate.text_score - 3.0


def test_tied_scores_are_deterministic_in_master_order() -> None:
    master = _master(
        [
            {"Itemcode": "FIRST", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
            {"Itemcode": "SECOND", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
        ]
    )
    candidates = generate_top_candidates(
        _offer("Al Kabeer Chicken Nuggets", "Chicken Nuggets-Frozen", "400g"),
        master,
        top_k=2,
    )
    assert [candidate.itemcode for candidate in candidates] == ["FIRST", "SECOND"]
    assert candidates[0].margin == 0.0


def test_top_k_rank_one_is_the_production_best_candidate() -> None:
    master = _master(
        [
            {"Itemcode": "NUG", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
            {"Itemcode": "STR", "Itemname": "CHICKEN STRIPS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
            {"Itemcode": "FIL", "Itemname": "CHICKEN FILLET", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
        ]
    )
    offer = _offer("Al Kabeer Chicken Nuggets 400g", "Chicken Nuggets-Frozen", "400g")

    best = generate_best_candidate(offer, master)
    top = generate_top_candidates(offer, master, top_k=3)

    assert top[0] == best
    assert [candidate.candidate_rank for candidate in top] == [1, 2, 3]


def test_batch_scoring_uses_two_vectorised_cdist_calls_per_category(
    monkeypatch,
) -> None:
    master = _master(
        [
            {"Itemcode": "CH", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
            {"Itemcode": "MEAT", "Itemname": "BEEF BURGER", "Item-Cat-2": "Meat", "Item-Spec": "400g"},
        ]
    )
    offers = preprocess_clickflyer(
        pd.DataFrame(
            [
                {
                    "Offer Name": "Chicken Nuggets",
                    "Product": "Chicken Nuggets-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "No Variant",
                    "Base Packsize": "400g",
                },
                {
                    "Offer Name": "Chicken Nuggets Value Pack",
                    "Product": "Chicken Nuggets-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "No Variant",
                    "Base Packsize": "400g",
                },
                {
                    "Offer Name": "Beef Burger",
                    "Product": "Beef Burger-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "No Variant",
                    "Base Packsize": "400g",
                },
            ]
        )
    )
    original_cdist = candidate_generator.process.cdist
    calls: list[tuple[int, int]] = []

    def recording_cdist(queries, choices, **kwargs):
        calls.append((len(queries), len(choices)))
        return original_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(candidate_generator.process, "cdist", recording_cdist)
    generate_candidates_batch(offers, master)

    assert calls == [(2, 1), (2, 1), (1, 1), (1, 1)]


def test_adjusted_margin_and_raw_margin_remain_distinct_after_pack_reranking(
    monkeypatch,
) -> None:
    master = _master(
        [
            {"Itemcode": "RAW-BEST", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken"},
            {"Itemcode": "PACK-BEST", "Itemname": "CHICKEN STRIPS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
        ]
    )
    offer = _offer("Chicken Nuggets 400g", "Chicken Nuggets-Frozen", "400g")
    score_matrix = np.array([[100.0, 95.0]], dtype=float)
    monkeypatch.setattr(
        candidate_generator.process,
        "cdist",
        lambda queries, choices, scorer: score_matrix.copy(),
    )

    best = generate_best_candidate(offer, master)

    assert best.itemcode == "PACK-BEST"
    assert best.text_score == 95.0
    assert best.adjusted_score == 99.0
    assert best.margin == 2.0
    assert best.raw_margin == -5.0


def test_missing_category_pool_returns_no_match_candidate() -> None:
    master = _master(
        [{"Itemcode": "CH", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"}]
    )
    candidate = generate_best_candidate(_offer("Fish Fillet", "Fish Fillet-Frozen", "400g"), master)
    assert candidate.itemcode == "NO_MATCH"
    assert candidate.confidence_tier == "no_match_category"


def test_batch_generation_preserves_offer_order() -> None:
    master = _master(
        [
            {"Itemcode": "CH", "Itemname": "CHICKEN NUGGETS", "Item-Cat-2": "Chicken", "Item-Spec": "400g"},
            {"Itemcode": "MEAT", "Itemname": "BEEF BURGER", "Item-Cat-2": "Meat", "Item-Spec": "400g"},
        ]
    )
    offers = preprocess_clickflyer(
        pd.DataFrame(
            [
                {
                    "Offer Name": "Beef Burger",
                    "Product": "Beef Burger-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "No Variant",
                    "Base Packsize": "400g",
                },
                {
                    "Offer Name": "Chicken Nuggets",
                    "Product": "Chicken Nuggets-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "No Variant",
                    "Base Packsize": "400g",
                },
            ]
        )
    )
    ranked = generate_candidates_batch(offers, master)
    assert [candidates[0].itemcode for candidates in ranked] == ["MEAT", "CH"]
