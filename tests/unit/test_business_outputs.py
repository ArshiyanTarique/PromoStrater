"""Business-contract regression tests for mapping and competitor outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import sku_mapping.competitors.discovery as competitor_discovery

from sku_mapping.competitors.discovery import (
    COMPETITOR_EXPORT_COLUMNS,
    COMPETITOR_LONG_COLUMNS,
    discover_competitors,
)
from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.exports.business_outputs import (
    SKU_MAPPING_COLUMNS,
    build_business_outputs,
    build_sku_mapping_export,
)
from sku_mapping.exports.run_outputs import write_run_outputs


def _raw_offers() -> pd.DataFrame:
    common = {
        "Country": "KSA",
        "Retailer Name": "Retailer",
        "Flyer Name": "Flyer",
        "Variant": "",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "offerid": "own-1",
                "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                "Offer Price": 10,
                "Regular Price": 12,
                "Brand Name": "Al Kabeer",
                "Product": "Chicken Nuggets",
                "Base Packsize": "400g",
            },
            {
                **common,
                "offerid": "own-2",
                "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                "Offer Price": 11,
                "Regular Price": 13,
                "Brand Name": "Al Kabeer",
                "Product": "Chicken Nuggets",
                "Base Packsize": "400g",
            },
            {
                **common,
                "Country": "UAE",
                "offerid": "comp-good",
                "Offer Name": "Chicken Nuggets 400g",
                "Offer Price": 9,
                "Regular Price": 11,
                "Brand Name": "Americana",
                # Real flyer Product fields can be generic even when the
                # specific family appears clearly in the offer name.
                "Product": "Frozen Breaded Chicken",
                "Base Packsize": "400g",
            },
            {
                **common,
                "offerid": "comp-pack-conflict",
                "Offer Name": "Seara Chicken Nuggets 1kg",
                "Offer Price": 15,
                "Regular Price": 18,
                "Brand Name": "Seara",
                "Product": "Chicken Nuggets",
                "Base Packsize": "1kg",
            },
            {
                **common,
                "offerid": "comp-family-conflict",
                "Offer Name": "Americana Chicken Strips 400g",
                "Offer Price": 8,
                "Regular Price": 10,
                "Brand Name": "Americana",
                "Product": "Chicken Strips",
                "Base Packsize": "400g",
            },
            {
                **common,
                "offerid": "comp-protein-conflict",
                "Offer Name": "Beef Nuggets 400g",
                "Offer Price": 7,
                "Regular Price": 9,
                "Brand Name": "Other",
                "Product": "Beef Nuggets",
                "Base Packsize": "400g",
            },
        ]
    )


def _prepared_offers() -> pd.DataFrame:
    prepared = preprocess_clickflyer(_raw_offers())
    prepared["offer_group_id"] = prepared["offerid"]
    # Force the adversarial row through the cheap family/category candidate
    # gate so the independent protein guard itself is exercised.
    conflict = prepared["offerid"].eq("comp-protein-conflict")
    prepared.loc[conflict, "category"] = "Chicken"
    prepared.loc[conflict, "product_family"] = "chicken nuggets"
    prepared.loc[conflict, "match_text"] = "beef nuggets"
    return prepared


def _master() -> pd.DataFrame:
    return preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "SKU-NUGGETS",
                    "Itemname": "Chicken Nuggets",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Chicken Nuggets",
                    "Item Description": "Frozen Chicken Nuggets 400g",
                    "Item-Spec": "400 Gms x 10 Pkts",
                },
                {
                    "Itemcode": "SKU-SAMOSA",
                    "Itemname": "Chicken Samosa",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Chicken Samosa",
                    "Item Description": "Frozen Chicken Samosa 240g",
                    "Item-Spec": "240 Gms x 12 Pkts",
                },
            ]
        )
    )


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "offer_id": "own-1",
                "proposed_master_sku": "SKU-NUGGETS",
                "proposed_master_description": "Chicken Nuggets",
                "matched_master_sku": "",
                "lightgbm_probability": 0.74,
                "final_decision": "MANUAL_REVIEW",
                "final_decision_reason": "BELOW_AUTO_ACCEPT_THRESHOLD",
            },
            {
                "offer_id": "own-2",
                "proposed_master_sku": "SKU-SAMOSA",
                "proposed_master_description": "Chicken Samosa",
                "matched_master_sku": "SKU-SAMOSA",
                "lightgbm_probability": 0.95,
                "final_decision": "AUTO_ACCEPT",
                "final_decision_reason": "SAFE_AGREEMENT_POLICY",
            },
            {
                "offer_id": "comp-good",
                "final_decision": "COMPETITOR_OFFER",
                "final_decision_reason": "NON_OWN_BRAND_OFFER",
            },
        ]
    )


def test_sku_mapping_contains_each_canonical_own_offer_and_keeps_proposal() -> None:
    export = build_sku_mapping_export(
        _prepared_offers(),
        _master(),
        _decisions(),
        run_id="business-run",
    )

    assert tuple(export.columns) == SKU_MAPPING_COLUMNS
    assert len(export) == 2
    assert export["source_offer_id"].tolist() == ["own-1", "own-2"]
    assert export["source_offer_name"].nunique() == 1
    assert not export["source_offer_id"].str.startswith("comp").any()
    manual = export.set_index("source_offer_id").loc["own-1"]
    assert manual["matched_master_sku"] == "SKU-NUGGETS"
    assert manual["matched_master_name"] == "Chicken Nuggets"
    assert manual["matched_master_description"] == (
        "Frozen Chicken Nuggets 400g"
    )
    assert manual["mapping_status"] == "MANUAL_REVIEW"
    assert bool(manual["requires_human_review"]) is True


def test_missing_candidate_has_explicit_reason_without_dropping_offer() -> None:
    export = build_sku_mapping_export(
        _prepared_offers(),
        _master(),
        _decisions().query("offer_id != 'own-2'"),
        run_id="business-run",
    ).set_index("source_offer_id")

    assert export.loc["own-2", "matched_master_sku"] == ""
    assert export.loc["own-2", "mapping_status"] == "NO_CANDIDATE"
    assert export.loc["own-2", "mapping_reason"] == "NO_RETAINED_CANDIDATE"


def test_competitor_aggregate_is_target_sku_centric_exact_and_aligned() -> None:
    result = build_business_outputs(
        _prepared_offers(),
        _master(),
        _decisions(),
        competitor_config=load_config("config/default.yaml").competitors,
        run_id="business-run",
    )

    assert tuple(result.competitor_export.columns) == (
        COMPETITOR_EXPORT_COLUMNS
    )
    assert tuple(result.competitor_long_format.columns) == (
        COMPETITOR_LONG_COLUMNS
    )
    assert len(result.competitor_export) == 2
    assert not result.competitor_export["master_sku"].duplicated().any()
    aggregate = result.competitor_export.set_index("master_sku")
    nuggets = aggregate.loc["SKU-NUGGETS"]
    assert nuggets["competitor_status"] == "COMPETITORS_FOUND"
    assert nuggets["competitor_count"] == 1

    aligned = [
        json.loads(nuggets[column])
        for column in (
            "competitor_brand_names",
            "competitor_offer_ids",
            "competitor_offer_names",
            "competitor_offer_prices",
        )
    ]
    assert {len(values) for values in aligned} == {1}
    assert aligned[0] == ["Americana"]
    assert aligned[1] == ["comp-good"]
    assert aligned[2] == ["Chicken Nuggets 400g"]
    assert aligned[3] == [9]

    samosa = aggregate.loc["SKU-SAMOSA"]
    assert samosa["competitor_count"] == 0
    assert samosa["competitor_status"] == "NO_COMPETITOR_FOUND"
    assert samosa["competitor_reason"]

    audit = result.competitor_long_format
    nuggets_audit = audit[audit["master_sku"].eq("SKU-NUGGETS")]
    by_id = nuggets_audit.set_index("competitor_offer_id")
    assert by_id.loc["comp-good", "competitor_match_status"] == "MATCHED"
    assert (
        by_id.loc[
            "comp-pack-conflict", "competitor_match_reason"
        ]
        == "PACK_CONFLICT"
    )
    assert (
        by_id.loc[
            "comp-family-conflict", "competitor_match_reason"
        ]
        == "PRODUCT_FAMILY_CONFLICT"
    )
    assert (
        by_id.loc[
            "comp-protein-conflict", "competitor_match_reason"
        ]
        == "PROTEIN_CONFLICT"
    )
    assert result.diagnostics["source_competitor_offer_count"] == 4
    assert result.diagnostics["matched_competitor_rows"] == 1
    assert result.diagnostics["no_match_target_count"] == 1
    assert result.diagnostics["excluded_relationship_count"] == 7


def test_scored_but_rejected_candidates_have_an_explicit_policy_reason() -> None:
    strict_config = replace(
        load_config("config/default.yaml").competitors,
        raw_score_floor=99.0,
        adjusted_score_floor=99.0,
    )
    result = build_business_outputs(
        _prepared_offers(),
        _master(),
        _decisions().query("offer_id == 'own-1'"),
        competitor_config=strict_config,
        run_id="strict-policy-run",
    )

    target = result.competitor_export.iloc[0]
    assert target["competitor_count"] == 0
    assert target["competitor_status"] == "NO_COMPETITOR_FOUND"
    assert target["competitor_reason"] == (
        "no candidate exceeded the competitor eligibility policy"
    )


def test_write_run_outputs_persists_internal_long_form(tmp_path: Path) -> None:
    result = build_business_outputs(
        _prepared_offers(),
        _master(),
        _decisions(),
        competitor_config=load_config("config/default.yaml").competitors,
        run_id="business-run",
    )
    outputs = write_run_outputs(
        run_id="business-run",
        sku_mapping_export=result.sku_mapping,
        competitor_export=result.competitor_export,
        competitor_long_format=result.competitor_long_format,
        summary=result.diagnostics,
        output_root=tmp_path,
    )

    assert set(outputs.paths) == {
        "sku_mapping",
        "competitor_offers",
        "competitor_long_form",
        "run_summary",
    }
    long_frame = pd.read_csv(outputs.paths["competitor_long_form"])
    assert tuple(long_frame.columns) == COMPETITOR_LONG_COLUMNS
    assert set(long_frame["competitor_offer_id"]) == {
        "comp-good",
        "comp-pack-conflict",
        "comp-family-conflict",
        "comp-protein-conflict",
    }


def test_equal_score_competitors_use_offer_id_tiebreak_for_every_list() -> None:
    prepared = _prepared_offers()
    duplicate = prepared[prepared["offerid"].eq("comp-good")].copy()
    duplicate["offerid"] = "comp-a"
    duplicate["offer_group_id"] = "comp-a"
    duplicate["Brand Name"] = "Brand A"
    prepared = pd.concat([prepared, duplicate], ignore_index=True)
    result = build_business_outputs(
        prepared,
        _master(),
        _decisions().query("offer_id == 'own-1'"),
        competitor_config=load_config("config/default.yaml").competitors,
        run_id="deterministic-run",
    )

    row = result.competitor_export.iloc[0]
    offer_ids = json.loads(row["competitor_offer_ids"])
    brands = json.loads(row["competitor_brand_names"])
    names = json.loads(row["competitor_offer_names"])
    assert offer_ids == ["comp-a", "comp-good"]
    assert brands == ["Brand A", "Americana"]
    assert names == ["Chicken Nuggets 400g", "Chicken Nuggets 400g"]


def test_large_competitor_audit_is_category_gated_chunked_and_file_backed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _prepared_offers()
    mapping = build_sku_mapping_export(
        prepared,
        _master(),
        _decisions().query("offer_id == 'own-1'"),
        run_id="chunked-run",
    )
    observed_chunk_sizes: list[int] = []
    original = competitor_discovery._evaluate_target

    def recording_evaluator(**kwargs):
        observed_chunk_sizes.append(len(kwargs["competitor_pool"]))
        return original(**kwargs)

    monkeypatch.setattr(
        competitor_discovery, "COMPETITOR_EVALUATION_CHUNK_SIZE", 2
    )
    monkeypatch.setattr(
        competitor_discovery, "_evaluate_target", recording_evaluator
    )
    audit_path = tmp_path / "competitor_audit.partial.csv"

    result = discover_competitors(
        prepared,
        _master(),
        mapping,
        config=load_config("config/default.yaml").competitors,
        run_id="chunked-run",
        audit_path=audit_path,
    )

    assert observed_chunk_sizes
    assert max(observed_chunk_sizes) <= 2
    assert result.long_format.empty
    assert result.long_format_path == audit_path
    persisted = pd.read_csv(audit_path)
    assert tuple(persisted.columns) == COMPETITOR_LONG_COLUMNS
    assert result.diagnostics["target_offer_relationships_evaluated"] == 4
    assert result.diagnostics["detailed_relationship_rows"] == len(persisted)


def _variant_raw_offers() -> pd.DataFrame:
    """One own-brand samosa offer and a competitor offer repeated per variant.

    The competitor's protein appears only in ``Variant``. Its offer name says
    "Mix ... Sambosa", which is exactly the shape that loses all protein
    evidence when a caller collapses the dump to one row per offer identity.
    """
    common = {
        "Country": "KSA",
        "Retailer Name": "Retailer",
        "Flyer Name": "Flyer",
        "Product": "Samosa-Frozen",
        "Base Packsize": "240 gm",
        "Offer Price": 9,
        "Regular Price": 11,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "offerid": "own-samosa",
                "Offer Name": "Al Kabeer Chicken Samosas 240g",
                "Brand Name": "Al Kabeer",
                "Variant": "Chicken",
            },
            # One ClickFlyer offer identity, three variant rows. "No Variant"
            # sorts first and is the row a collapsed frame would retain.
            {
                **common,
                "offerid": "comp-mixed",
                "Offer Name": "Rival Mix Frozen Sambosa 240gm",
                "Brand Name": "Rival",
                "Variant": "No Variant",
            },
            {
                **common,
                "offerid": "comp-mixed",
                "Offer Name": "Rival Mix Frozen Sambosa 240gm",
                "Brand Name": "Rival",
                "Variant": "Chicken",
            },
            {
                **common,
                "offerid": "comp-mixed",
                "Offer Name": "Rival Mix Frozen Sambosa 240gm",
                "Brand Name": "Rival",
                "Variant": "Other-Veg",
            },
        ]
    )


def _variant_master() -> pd.DataFrame:
    return preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "CKSA",
                    "Itemname": "12 CHICKEN SAMOSAS",
                    "Item-Cat-2": "Dough",
                    "Item-Cat-4": "SAMOSA",
                    "Item Description": "12 CHICKEN SAMOSAS 240 Gms x 20 Pkts",
                    "Item-Spec": "240 Gms x 20 Pkts",
                }
            ]
        )
    )


def _variant_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the variant-level pool, the collapsed frame, and the mapping."""
    prepared = preprocess_clickflyer(_variant_raw_offers())
    prepared["offer_group_id"] = prepared["offerid"]
    collapsed = prepared.drop_duplicates(
        "offer_group_id", keep="first"
    ).reset_index(drop=True)
    mapping = pd.DataFrame(
        [
            {
                "source_offer_id": "own-samosa",
                "source_offer_name": "Al Kabeer Chicken Samosas 240g",
                "matched_master_sku": "CKSA",
            }
        ]
    )
    return prepared, collapsed, mapping


def _supported_competitor_names(result: object) -> list[str]:
    return json.loads(result.export.iloc[0]["competitor_offer_names"])


def test_collapsed_pool_hides_a_competitor_whose_protein_is_only_a_variant() -> None:
    """Guard the defect itself: the collapsed frame keeps the No Variant row."""
    _, collapsed, mapping = _variant_frames()

    result = discover_competitors(
        collapsed,
        _variant_master(),
        mapping,
        config=load_config("config/default.yaml").competitors,
        run_id="collapsed-run",
    )

    assert result.export.iloc[0]["competitor_count"] == 0
    retained = result.long_format["competitor_variant"].tolist()
    assert retained == ["No Variant"]


def test_variant_level_pool_discovers_the_competitor_exactly_once() -> None:
    """The supplied pool restores the variant row without duplicating the offer."""
    prepared, collapsed, mapping = _variant_frames()

    result = discover_competitors(
        collapsed,
        _variant_master(),
        mapping,
        config=load_config("config/default.yaml").competitors,
        run_id="variant-run",
        competitor_offers=prepared,
    )

    aggregate = result.export.iloc[0]
    assert aggregate["competitor_count"] == 1
    assert aggregate["competitor_status"] == "COMPETITORS_FOUND"
    assert _supported_competitor_names(result) == [
        "Rival Mix Frozen Sambosa 240gm"
    ]
    # The offer must appear once even though two of its variant rows were
    # evaluated and one of them scored below the policy floors.
    offer_ids = json.loads(aggregate["competitor_offer_ids"])
    assert offer_ids == ["comp-mixed"]

    supported = result.long_format[
        result.long_format["competitor_match_status"].isin(
            {"MATCHED", "AMBIGUOUS"}
        )
    ]
    assert supported["competitor_variant"].tolist() == ["Chicken"]
    # Every variant row stays auditable rather than being silently dropped.
    assert set(result.long_format["competitor_variant"]) == {
        "No Variant",
        "Chicken",
        "Other-Veg",
    }


def test_supplied_competitor_pool_leaves_the_own_brand_mapping_unchanged() -> None:
    """The pool must not alter own-offer rows, which stay one per identity."""
    prepared, collapsed, _ = _variant_frames()
    decisions = pd.DataFrame(
        [
            {
                "offer_id": "own-samosa",
                "matched_master_sku": "CKSA",
                "final_decision": "AUTO_ACCEPT",
                "final_decision_reason": "SAFE_AGREEMENT_POLICY",
            }
        ]
    )
    config = load_config("config/default.yaml").competitors

    baseline = build_business_outputs(
        collapsed,
        _variant_master(),
        decisions,
        competitor_config=config,
        run_id="baseline-run",
    )
    widened = build_business_outputs(
        collapsed,
        _variant_master(),
        decisions,
        competitor_config=config,
        run_id="widened-run",
        competitor_offers=prepared,
    )

    assert tuple(widened.sku_mapping.columns) == SKU_MAPPING_COLUMNS
    assert len(widened.sku_mapping) == len(baseline.sku_mapping) == 1
    assert widened.sku_mapping.iloc[0]["source_offer_id"] == "own-samosa"
    assert widened.sku_mapping.iloc[0]["matched_master_sku"] == "CKSA"
    # Only competitor recall changes.
    assert baseline.competitor_export.iloc[0]["competitor_count"] == 0
    assert widened.competitor_export.iloc[0]["competitor_count"] == 1
