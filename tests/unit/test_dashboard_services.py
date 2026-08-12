"""Page-independent dashboard review, registry, and download tests."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dashboard.services.registry_service import (
    DashboardRegistryService,
    build_display_label,
)
from dashboard.services.review_service import (
    DashboardReviewService,
    ReviewAnswerError,
)
from dashboard.services.run_service import (
    DashboardRunService,
    download_filename,
)
from sku_mapping.competitors.discovery import COMPETITOR_EXPORT_COLUMNS
from sku_mapping.competitors.discovery import discover_competitors
from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.exports.run_outputs import SKU_MAPPING_COLUMNS
from sku_mapping.learning.store import LearningStore


def _config(tmp_path: Path):
    base = load_config("config/default.yaml")
    return replace(
        base,
        dashboard=replace(
            base.dashboard,
            output_directory=tmp_path / "outputs",
            input_directory=tmp_path / "uploads",
        ),
        learning_store=replace(
            base.learning_store,
            database_path=tmp_path / "learning.db",
        ),
    )


def test_registry_rejects_arbitrary_model_id(tmp_path: Path) -> None:
    service = DashboardRegistryService(_config(tmp_path))
    options = service.list_models()
    assert options
    with pytest.raises(ValueError, match="not in the safe registry"):
        service.validate_model_id("../../arbitrary.joblib")


def test_review_service_requires_explicit_false_resolution(
    tmp_path: Path,
) -> None:
    service = DashboardReviewService(LearningStore(tmp_path / "learning.db"))
    with pytest.raises(ReviewAnswerError, match="False requires"):
        service.save(review_id="missing", answer="FALSE")


def test_review_service_saves_explicit_none_as_gold(tmp_path: Path) -> None:
    store = LearningStore(tmp_path / "learning.db")
    store.upsert_pipeline_run(
        {
            "run_id": "run",
            "status": "COMPLETED_ASSISTED",
            "deployment_mode": "assisted",
        }
    )
    store.add_predictions(
        "run",
        [
            {
                "offer_id": f"offer-{number}",
                "offer_description": f"Offer {number}",
                "candidate_id": f"sku-{number}",
                "candidate_description": f"SKU {number}",
                "candidate_rank": 1,
                "lightgbm_probability": 0.9,
                "embedding_similarity": 0.8,
                "agreement_status": "SAFE_AGREEMENT",
                "final_decision": "AUTO_ACCEPT",
                "decision_source": "AGREEMENT",
            }
            for number in range(5)
        ],
    )
    session = store.create_review_session("run")
    assert session is not None
    question = store.review_questions(session)[0]
    DashboardReviewService(store).save(
        review_id=question["review_id"],
        answer="FALSE_NONE",
        reviewer_id="reviewer",
    )
    saved = store.review_questions(session)[0]
    assert saved["none_of_candidates"] == 1
    assert saved["label_quality"] == "GOLD"


def test_false_review_preserves_proposal_and_reviewer_correction(
    tmp_path: Path,
) -> None:
    store = LearningStore(tmp_path / "learning.db")
    store.upsert_pipeline_run(
        {
            "run_id": "correction-run",
            "status": "COMPLETED_ASSISTED",
            "deployment_mode": "assisted",
        }
    )
    store.add_predictions(
        "correction-run",
        [
            {
                "offer_id": "offer",
                "offer_description": "Own-brand offer",
                "candidate_id": f"candidate-{rank}",
                "candidate_description": f"Candidate {rank}",
                "candidate_rank": rank,
                "lightgbm_probability": probability,
                "final_decision": (
                    "MANUAL_REVIEW"
                    if rank == 1
                    else "CANDIDATE_NOT_SELECTED"
                ),
                "decision_source": "AGREEMENT_POLICY",
            }
            for rank, probability in ((1, 0.8), (2, 0.7))
        ],
    )
    session = store.create_review_session("correction-run")
    assert session is not None
    question = store.review_questions(session)[0]
    original_proposal = str(question["suggested_candidate_id"])
    correction = next(
        str(candidate["candidate_id"])
        for candidate in question["supplied_candidates"]
        if str(candidate["candidate_id"]) != original_proposal
    )

    DashboardReviewService(store).save(
        review_id=str(question["review_id"]),
        answer="FALSE_CANDIDATE",
        corrected_candidate_id=correction,
        reviewer_id="reviewer",
    )

    saved = store.review_questions(session)[0]
    assert saved["suggested_candidate_id"] == original_proposal
    assert saved["corrected_candidate_id"] == correction
    assert saved["human_answer"] == 0
    assert saved["label_quality"] == "GOLD"


def test_review_can_select_product_master_sku_outside_supplied_top_k(
    tmp_path: Path,
) -> None:
    store = LearningStore(tmp_path / "learning.db")
    store.upsert_pipeline_run(
        {
            "run_id": "master-correction-run",
            "status": "COMPLETED_ASSISTED",
            "deployment_mode": "assisted",
        }
    )
    store.add_predictions(
        "master-correction-run",
        [
            {
                "offer_id": "source-1_1",
                "source_offer_id": "source-1",
                "source_offer_text": "Chicken / Beef Burger Patty 400 gm",
                "entity_id": "source-1_1",
                "entity_index": 1,
                "entity_count": 2,
                "entity_text": "Chicken Burger Patty 400 g",
                "conjunction_type": "UNKNOWN_MULTI_PRODUCT",
                "attribute_inheritance_flags": (
                    "family:inherited_shared|retail_weight_g:inherited_shared"
                ),
                "entity_parse_confidence": 0.9,
                "offer_description": "Chicken Burger Patty 400 g",
                "candidate_id": "WRONG",
                "candidate_description": "Wrong candidate",
                "candidate_rank": 1,
                "lightgbm_probability": 0.7,
                "final_decision": "MANUAL_REVIEW",
                "decision_source": "AGREEMENT_POLICY",
            }
        ],
    )
    session = store.create_review_session("master-correction-run")
    assert session is not None
    master_path = tmp_path / "master.xlsx"
    pd.DataFrame(
        [
            {"Itemcode": "WRONG", "Itemname": "Wrong candidate"},
            {"Itemcode": "RIGHT", "Itemname": "Correct Product Master SKU"},
        ]
    ).to_excel(master_path, index=False)
    service = DashboardReviewService(
        store, product_master_path=master_path
    )
    question = store.review_questions(session)[0]
    service.save(
        review_id=str(question["review_id"]),
        answer="FALSE_CANDIDATE",
        corrected_candidate_id="RIGHT",
        decomposition_action="SPLIT_FURTHER",
        corrected_entity_text="Chicken Burger Patty 400 g",
        corrected_attributes_json='{"retail_weight_g": 400}',
    )
    saved = store.review_questions(session)[0]
    assert saved["suggested_candidate_id"] == "WRONG"
    assert saved["corrected_candidate_id"] == "RIGHT"
    assert saved["source_offer_id"] == "source-1"
    assert saved["entity_count"] == 2
    assert saved["decomposition_action"] == "SPLIT_FURTHER"


def test_download_service_hides_missing_and_invalid_outputs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    run_id = "dashboard-run"
    store.upsert_pipeline_run(
        {
            "run_id": run_id,
            "status": "COMPLETED_DASHBOARD_SHADOW",
            "deployment_mode": "shadow",
        }
    )
    directory = config.dashboard.output_directory / run_id
    directory.mkdir(parents=True)
    valid_mapping = directory / f"sku_mapping_{run_id}.csv"
    pd.DataFrame(columns=SKU_MAPPING_COLUMNS).to_csv(
        valid_mapping, index=False, encoding="utf-8-sig"
    )
    invalid_competitor = directory / f"competitor_offers_{run_id}.csv"
    pd.DataFrame({"wrong": []}).to_csv(invalid_competitor, index=False)
    summary = directory / f"run_summary_{run_id}.json"
    summary.write_text(json.dumps({"auto_accept_count": 0}), encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    store.update_run_outputs(
        run_id,
        {
            "sku_mapping": valid_mapping,
            "competitor_offers": invalid_competitor,
            "run_summary": summary,
            "monitoring_report": outside,
        },
    )

    artifacts = DashboardRunService(config, store).downloads(run_id)
    assert {artifact.key for artifact in artifacts} == {
        "sku_mapping",
        "run_summary",
    }
    assert all(str(tmp_path) not in artifact.filename for artifact in artifacts)


def test_model_display_labels_read_calibration_from_evidence() -> None:
    calibrated = build_display_label(
        {
            "model_version": "ranked-v5-calibrated",
            "model_id": "ranked-v5-cal-20260810T100756Z-matcher",
            "creation_timestamp": "2026-08-10T10:07:56+00:00",
        },
        {"calibration_method": "isotonic"},
    )
    uncalibrated = build_display_label(
        {
            "model_version": "ranked-v5",
            "model_id": "ranked-v5-20260806T103601Z-matcher",
            "creation_timestamp": "2026-08-06T10:36:01+00:00",
        },
        {},
    )

    assert calibrated == "Version 5 Ranked-Calibrated (2026-08-10)"
    assert uncalibrated == "Version 5 Ranked-Uncalibrated (2026-08-06)"


def test_model_display_label_never_claims_calibration_from_the_name() -> None:
    # A package named "calibrated" that never recorded a calibration method
    # must not present itself as calibrated in the operator selector.
    label = build_display_label(
        {
            "model_version": "ranked-v9-calibrated",
            "model_id": "ranked-v9",
            "creation_timestamp": "2026-09-01T00:00:00+00:00",
        },
        {},
    )
    assert label == "Version 9 Ranked-Uncalibrated (2026-09-01)"


def test_registered_models_expose_readable_labels() -> None:
    options = DashboardRegistryService(
        load_config("config/default.yaml")
    ).list_models()

    assert options
    assert all(
        option.display_label.startswith("Version ") for option in options
    )


def test_download_filenames_use_upload_stem_date_and_model() -> None:
    run = {
        "run_id": "dashboard-run",
        "source_filename": "Weekly Dump (UAE).xlsx",
        "started_at": "2026-08-10T07:08:25+00:00",
        "model_id": "ranked-v5-cal-20260810T070825Z-matcher",
    }
    assert (
        download_filename(run, "sku_mapping", Path("sku_mapping_x.csv"))
        == "Weekly_Dump_UAE_MAPPED_20260810_ranked-v5-cal.csv"
    )
    assert (
        download_filename(run, "competitor_offers", Path("c.csv"))
        == "Weekly_Dump_UAE_COMPETITORS_20260810_ranked-v5-cal.csv"
    )
    assert (
        download_filename(run, "run_summary", Path("s.json"))
        == "Weekly_Dump_UAE_RUN_SUMMARY_20260810_ranked-v5-cal.json"
    )


def test_download_filenames_survive_missing_run_metadata() -> None:
    name = download_filename(
        {"run_id": "abcdef123456789"}, "sku_mapping", Path("m.csv")
    )
    assert name == "abcdef123456_MAPPED_no-model.csv"


def test_run_summary_uses_unified_statistics_when_dashboard_export_missing(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = LearningStore(config.learning_store.database_path)
    run_id = "inference-only-run"
    directory = config.dashboard.output_directory / run_id
    directory.mkdir(parents=True)
    statistics = directory / "run_statistics.json"
    statistics.write_text(
        json.dumps(
            {
                "manual_review_count": 20_262,
                "offers_processed": 20_262,
                "total_runtime_seconds": 2161.66,
            }
        ),
        encoding="utf-8",
    )
    store.upsert_pipeline_run(
        {
            "run_id": run_id,
            "status": "COMPLETED_ASSISTED",
            "deployment_mode": "assisted",
            "source_row_count": 254_479,
            "unique_offer_count": 198_217,
            "output_paths": {"unified_statistics": statistics},
            "run_metadata": {"inference_offer_count": 20_262},
        }
    )

    summary = DashboardRunService(config, store).run_summary(run_id)

    assert summary["unique_offers"] == 198_217
    assert summary["inference_offer_count"] == 20_262
    assert summary["manual_review_count"] == 20_262
    assert summary["total_runtime_seconds"] == 2161.66
    assert summary["summary_source"] == "unified_inference_statistics"
    assert summary["dashboard_outputs_complete"] is False


def test_competitor_schema_constant_matches_business_contract() -> None:
    assert COMPETITOR_EXPORT_COLUMNS == (
        "master_sku",
        "master_name",
        "master_description",
        "source_alkabeer_offer_ids",
        "source_entity_ids",
        "source_alkabeer_offer_names",
        "competitor_count",
        "competitor_brand_names",
        "competitor_offer_ids",
        "competitor_offer_names",
        "competitor_products",
        "competitor_variants",
        "competitor_pack_sizes",
        "competitor_retailers",
        "competitor_flyers",
        "competitor_offer_prices",
        "competitor_regular_prices",
        "competitor_status",
        "competitor_reason",
        "run_id",
    )


def test_competitors_use_visible_manual_review_mapping_targets(
    tmp_path: Path,
) -> None:
    offers = preprocess_clickflyer(
        pd.DataFrame(
            [
                {
                    "Country": "KSA",
                    "Retailer Name": "R",
                    "Flyer Name": "F",
                    "offerid": "1",
                    "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                    "Offer Price": 10,
                    "Regular Price": 12,
                    "Brand Name": "Al Kabeer",
                    "Product": "Chicken Nuggets",
                    "Variant": "",
                    "Base Packsize": "400g",
                },
                {
                    "Country": "KSA",
                    "Retailer Name": "R",
                    "Flyer Name": "F",
                    "offerid": "2",
                    "Offer Name": "Other Chicken Nuggets 400g",
                    "Offer Price": 9,
                    "Regular Price": 11,
                    "Brand Name": "Other Brand",
                    "Product": "Chicken Nuggets",
                    "Variant": "",
                    "Base Packsize": "400g",
                },
            ]
        )
    )
    offers["offer_group_id"] = offers["offerid"].astype(str)
    master = preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "SKU-1",
                    "Itemname": "Chicken Nuggets",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Chicken Nuggets",
                    "Item Description": "Frozen Chicken Nuggets",
                    "Item-Spec": "400 Gms x 10 Pkts",
                }
            ]
        )
    )
    mapping = pd.DataFrame(
        [
            {
                "source_offer_id": "1",
                "source_offer_name": "Al Kabeer Chicken Nuggets 400g",
                "matched_master_sku": "SKU-1",
            }
        ]
    )
    config = load_config("config/default.yaml").competitors
    result = discover_competitors(
        offers,
        master,
        mapping,
        config=config,
        run_id="business-run",
    )
    assert result.eligible_target_count == 1
    assert len(result.export) == 1
    assert result.export.loc[0, "master_sku"] == "SKU-1"
    assert result.export.loc[0, "competitor_count"] == 1
    assert '"2"' in result.export.loc[0, "competitor_offer_ids"]
