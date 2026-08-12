from __future__ import annotations

from dataclasses import replace

import pandas as pd

from sku_mapping.agreement.policy import evaluate_candidate_agreement
from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import preprocess_product_master
from sku_mapping.embedding.backends import LocalHashingEmbeddingBackend
from sku_mapping.embedding.scorer import score_candidate_frame
from sku_mapping.exports.business_outputs import build_business_outputs
from sku_mapping.inference.pipeline import (
    _attach_final_decisions,
    _expand_inference_entities,
    finalize_unified_decisions,
)


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "offer_group_id": "source-1",
                "offerid": "source-1",
                "Offer Name": "Chicken / Beef Burger Patty 400 gm",
                "Product": "Burger Patties-Frozen",
                "Brand Name": "Al Kabeer",
                "Variant": "",
                "Base Packsize": "400 gm",
                "Retailer Name": "Retailer",
                "Flyer Name": "Flyer",
                "Offer Price": 10,
                "Regular Price": 12,
                "Country": "UAE",
                "is_own": True,
                "product_family": "burger patty",
                "category": "Meat",
                "match_text": "chicken beef burger patty 400 gm",
                "offer_measures": [(400.0, "weight")],
            }
        ]
    )


def _master() -> pd.DataFrame:
    return preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "CHICKEN-400",
                    "Itemname": "Chicken Burger Patty 400 g",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Burger Patty",
                    "Item Description": "Chicken Burger Patty",
                    "Item-Spec": "400 g x 12",
                },
                {
                    "Itemcode": "BEEF-400",
                    "Itemname": "Beef Burger Patty 400 g",
                    "Item-Cat-2": "Beef",
                    "Item-Cat-4": "Burger Patty",
                    "Item Description": "Beef Burger Patty",
                    "Item-Spec": "400 g x 12",
                },
            ]
        )
    )


def test_each_entity_has_independent_embedding_model_decision_and_export(
    tmp_path,
) -> None:
    offers = _expand_inference_entities(_offers())
    assert len(offers) == 2
    assert offers["source_offer_id"].nunique() == 1
    assert offers["offer_group_id"].nunique() == 2

    candidates: list[dict[str, object]] = []
    for offer in offers.to_dict(orient="records"):
        correct = (
            "CHICKEN-400"
            if offer["entity_protein"] == "chicken"
            else "BEEF-400"
        )
        wrong = "BEEF-400" if correct == "CHICKEN-400" else "CHICKEN-400"
        for rank, sku in enumerate((correct, wrong), start=1):
            protein = "Chicken" if sku.startswith("CHICKEN") else "Beef"
            candidates.append(
                {
                    "offer_group_id": offer["offer_group_id"],
                    "candidate_rank": rank,
                    "master_itemcode": sku,
                    "offer_text": offer["Offer Name"],
                    "offer_brand": "Al Kabeer",
                    "offer_product": offer["Product"],
                    "offer_variant": "",
                    "offer_base_packsize": offer["Base Packsize"],
                    "entity_text": offer["entity_text"],
                    "entity_protein": offer["entity_protein"],
                    "entity_product_family": "burger patty",
                    "entity_retail_weight_g": 400,
                    "conjunction_type": offer["conjunction_type"],
                    "master_brand": "Al Kabeer",
                    "master_item_description": f"{protein} Burger Patty 400 g",
                    "master_item_family": "Burger Patty",
                    "master_item_category": protein,
                    "master_item_long_description": (
                        f"{protein} Burger Patty"
                    ),
                    "master_item_spec": "400 g x 12",
                    "calibrated_probability": 0.96 if sku == correct else 0.10,
                    "commercial_outcome": (
                        "EXACT_MATCH"
                        if sku == correct
                        else "UNACCEPTABLE_MATCH"
                    ),
                    "commercial_hard_conflict": sku != correct,
                    "commercial_severity": (
                        "NONE" if sku == correct else "HARD"
                    ),
                    "commercial_exact_match_eligible": sku == correct,
                    "commercial_measurement_match": "EXACT",
                    "commercial_reason_codes": (
                        "" if sku == correct else "PROTEIN_CONFLICT"
                    ),
                    "candidate_margin": 10.0,
                    "candidate_raw_margin": 10.0,
                }
            )
    frame = pd.DataFrame(candidates)
    config = load_config("config/default.yaml")
    embedding_config = replace(
        config.embedding,
        enabled=True,
        cache_path=tmp_path / "entities.sqlite3",
    )
    embedding = score_candidate_frame(
        frame,
        config=embedding_config,
        backend=LocalHashingEmbeddingBackend(embedding_config),
    )
    assert embedding.status == "COMPLETED"
    scored = pd.concat(
        [frame.reset_index(drop=True), embedding.scores.reset_index(drop=True)],
        axis=1,
    )
    agreement = evaluate_candidate_agreement(
        scored, config=config.agreement
    ).frame.rename(
        columns={
            "embedding_top_candidate": (
                "agreement_embedding_top_candidate"
            ),
            "embedding_similarity": "agreement_embedding_similarity",
            "embedding_rank": "agreement_embedding_rank",
        }
    )
    scored = scored.merge(
        agreement,
        left_on="offer_group_id",
        right_on="offer_id",
        how="left",
        validate="many_to_one",
    )
    decisions = finalize_unified_decisions(
        offers,
        scored,
        run_id="entity-run",
        model_id="frozen-model",
    )
    assert len(decisions) == 2
    assert decisions["proposed_master_sku"].tolist() == [
        "CHICKEN-400",
        "BEEF-400",
    ]
    assert decisions["matched_master_sku"].eq("").all()
    assert decisions["final_decision"].eq("MANUAL_REVIEW").all()
    assert decisions["lightgbm_probability"].eq(0.96).all()
    assert decisions["entity_id"].is_unique
    assert decisions["source_offer_id"].eq("source-1").all()
    assert decisions["source_offer_aggregate_status"].eq(
        "AMBIGUOUS_DECOMPOSITION"
    ).all()

    applied = _attach_final_decisions(offers, decisions)
    business = build_business_outputs(
        applied,
        _master(),
        decisions,
        competitor_config=config.competitors,
        run_id="entity-run",
    )
    assert len(business.sku_mapping) == 2
    assert business.sku_mapping["source_offer_id"].eq("source-1").all()
    assert business.sku_mapping["entity_id"].is_unique
    assert set(business.sku_mapping["matched_master_sku"]) == {
        "CHICKEN-400",
        "BEEF-400",
    }
    assert set(business.competitor_export["master_sku"]) == {
        "CHICKEN-400",
        "BEEF-400",
    }
    assert all(
        value in {
            '["source-1_1"]',
            '["source-1_2"]',
        }
        for value in business.competitor_export["source_entity_ids"]
    )


def test_one_entity_can_match_while_another_remains_in_review() -> None:
    source = _offers()
    source.loc[
        0, "Offer Name"
    ] = "Chicken Nuggets 400 gm + Chicken Popcorn 500 gm"
    source.loc[0, "Product"] = "Chicken Products-Frozen"
    source.loc[0, "Base Packsize"] = ""
    offers = _expand_inference_entities(source)
    assert len(offers) == 2
    rows = []
    for position, offer in offers.reset_index(drop=True).iterrows():
        rows.append(
            {
                "offer_group_id": offer["offer_group_id"],
                "candidate_rank": 1,
                "master_itemcode": f"SKU-{position + 1}",
                "master_item_description": offer["entity_text"],
                "calibrated_probability": 0.95,
                "embedding_similarity": 0.9,
                "embedding_status": "COMPLETED",
                "embedding_failure_reason": "",
                "lightgbm_top_candidate": f"SKU-{position + 1}",
                "agreement_status": (
                    "SAFE_AGREEMENT" if position == 0 else "DISAGREEMENT"
                ),
                "routing_decision": (
                    "AUTO_ACCEPT" if position == 0 else "MANUAL_REVIEW"
                ),
                "same_top_candidate": position == 0,
                "commercial_outcome": "EXACT_MATCH",
                "commercial_exact_match_eligible": True,
                "commercial_hard_conflict": False,
                "commercial_severity": "NONE",
                "commercial_measurement_match": "EXACT",
                "commercial_reason_codes": "",
            }
        )
    decisions = finalize_unified_decisions(
        offers,
        pd.DataFrame(rows),
        run_id="partial-entity-run",
        model_id="frozen-model",
    )
    assert decisions["final_eligible_mapping"].tolist() == [True, False]
    assert decisions["source_offer_aggregate_status"].eq(
        "PARTIALLY_MATCHED"
    ).all()
