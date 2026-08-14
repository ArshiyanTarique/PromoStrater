"""End-to-end dashboard service processing without Streamlit page logic."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from dashboard.services.processing_service import (
    DashboardProcessRequest,
    DashboardProcessingError,
    DashboardProcessingService,
    DuplicateProcessingError,
)
from dashboard.services.progress import ProcessingState
from dashboard.services.run_service import DashboardRunService
from dashboard.services.review_service import DashboardReviewService
from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.data.preprocessing import preprocess_clickflyer
from sku_mapping.failure_diagnostics import capture_exception_details
from sku_mapping.inference.pipeline import UnifiedInferenceResult
from sku_mapping.learning.store import LearningStore

from tests.registered_models import registered_model_id  # noqa: E402

MODEL_ID = registered_model_id()


def _config(tmp_path: Path):
    base = load_config("config/default.yaml")
    return replace(
        base,
        output=replace(base.output, output_dir=tmp_path / "legacy_outputs"),
        dashboard=replace(
            base.dashboard,
            input_directory=tmp_path / "uploads",
            output_directory=tmp_path / "dashboard_outputs",
        ),
        learning_store=replace(
            base.learning_store,
            database_path=tmp_path / "learning.db",
            csv_export_directory=tmp_path / "exports",
        ),
        shadow_mode=replace(
            base.shadow_mode,
            output_directory=tmp_path / "shadow",
            review_staging_directory=tmp_path / "reviews",
            challenge_set_directory=tmp_path / "challenge",
        ),
        llm_review=replace(
            base.llm_review,
            cache_path=tmp_path / "llm.sqlite3",
        ),
    )


def _five_offer_csv() -> bytes:
    base = pd.read_csv("tests/fixtures/clickflyer_valid.csv")
    rows = []
    for number, weight in enumerate((270, 400, 500, 750, 1000)):
        row = base.iloc[0].copy()
        row["offerid"] = f"dashboard-{number}"
        row["Offer Name"] = f"Al Kabeer Chicken Nuggets {weight}g"
        row["Base Packsize"] = f"{weight}g"
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _mixed_offer_csv() -> bytes:
    own = pd.read_csv(BytesIO(_five_offer_csv()))
    competitors = own.copy()
    competitors["offerid"] = [
        f"competitor-{number}" for number in range(len(competitors))
    ]
    competitors["Brand Name"] = "Other Brand"
    competitors["Offer Name"] = competitors["Offer Name"].str.replace(
        "Al Kabeer", "Other Brand", regex=False
    )
    return pd.concat([own, competitors], ignore_index=True).to_csv(
        index=False
    ).encode("utf-8")


def test_processing_persists_run_questions_and_validated_downloads(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    stages = []
    request = DashboardProcessRequest(
        filename="../unsafe dump.csv",
        content=_five_offer_csv(),
        deployment_mode=MLDeploymentMode.SHADOW,
        model_id=MODEL_ID,
    )
    result = service.process(
        request,
        progress=stages.append,
    )

    assert result.status == "COMPLETED_DASHBOARD_SHADOW"
    assert result.review_session_id is not None
    assert stages[0].state is ProcessingState.RUNNING
    assert stages[-1].state is ProcessingState.SUCCEEDED
    assert stages[-1].overall_percent == 100
    assert all(
        current.overall_percent <= following.overall_percent
        for current, following in zip(stages, stages[1:], strict=False)
    )
    store = LearningStore(config.learning_store.database_path)
    run = store.get_pipeline_run(result.run_id)
    assert run["source_filename"] == "unsafe_dump.csv"
    assert run["source_row_count"] == 5
    assert run["unique_offer_count"] == 5
    assert len(store.review_questions(result.review_session_id)) == 5
    assert store.summary()["retraining_performed"] is False

    artifacts = DashboardRunService(config, store).downloads(result.run_id)
    keys = {artifact.key for artifact in artifacts}
    assert {"sku_mapping", "competitor_offers", "run_summary"}.issubset(keys)
    competitor = next(
        artifact for artifact in artifacts if artifact.key == "competitor_offers"
    )
    mapping = next(
        artifact for artifact in artifacts if artifact.key == "sku_mapping"
    )
    mapping_frame = pd.read_csv(BytesIO(mapping.content))
    frame = pd.read_csv(BytesIO(competitor.content))
    assert len(mapping_frame) == 5
    assert len(frame) == mapping_frame["matched_master_sku"].nunique()
    assert frame["competitor_count"].eq(0).all()
    assert frame["competitor_status"].eq("NO_COMPETITOR_FOUND").all()
    assert frame["competitor_reason"].fillna("").ne("").all()

    with pytest.raises(DuplicateProcessingError):
        service.process(request)


def test_assisted_processing_propagates_offer_identity_to_competitors(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    result = service.process(
        DashboardProcessRequest(
            filename="assisted.csv",
            content=_five_offer_csv(),
            deployment_mode=MLDeploymentMode.ASSISTED,
            model_id=MODEL_ID,
        )
    )

    assert result.status == "COMPLETED_DASHBOARD_ASSISTED"
    assert result.summary["auto_accept_count"] == 0
    assert result.summary["manual_review_count"] > 0
    store = LearningStore(config.learning_store.database_path)
    artifacts = DashboardRunService(config, store).downloads(result.run_id)
    assert {"sku_mapping", "competitor_offers"}.issubset(
        artifact.key for artifact in artifacts
    )


def test_40_offer_count_and_identity_survive_browser_refresh(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    content = Path(
        "tests/fixtures/clickflyer_40_distinct_offerids.csv"
    ).read_bytes()
    result = DashboardProcessingService(config).process(
        DashboardProcessRequest(
            filename="forty-offers.csv",
            content=content,
            deployment_mode=MLDeploymentMode.SHADOW,
            model_id=MODEL_ID,
        )
    )

    assert result.summary["unique_offers"] == 40
    assert result.summary["inference_offers"] == 40
    assert result.summary["offer_identity_source"] == "offerid"

    # Simulate a refreshed browser by constructing fresh store/service objects.
    refreshed_store = LearningStore(config.learning_store.database_path)
    refreshed = DashboardRunService(config, refreshed_store).run_summary(
        result.run_id
    )
    assert refreshed["unique_offers"] == 40
    assert refreshed["inference_offer_count"] == 40
    assert refreshed["offer_identity_source"] == "offerid"
    session = refreshed_store.review_session_for_run(result.run_id)
    assert session is not None
    questions = refreshed_store.review_questions(str(session["session_id"]))
    assert len(questions) == 5
    assert len({question["offer_id"] for question in questions}) == 5

    connection = sqlite3.connect(config.learning_store.database_path)
    try:
        persisted_ids = {
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT offer_id FROM predictions
                WHERE run_id = ?
                """,
                (result.run_id,),
            )
        }
    finally:
        connection.close()
    assert len(persisted_ids) == 40
    assert all(persisted_ids)
    assert all(
        str(question["offer_id"]) in persisted_ids
        for question in questions
    )


def test_missing_offerid_uses_persisted_fallback_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    frame = pd.read_csv(BytesIO(_five_offer_csv())).drop(
        columns=["offerid"]
    )
    result = DashboardProcessingService(config).process(
        DashboardProcessRequest(
            filename="fallback-offers.csv",
            content=frame.to_csv(index=False).encode("utf-8"),
            deployment_mode=MLDeploymentMode.SHADOW,
            model_id=MODEL_ID,
        )
    )

    store = LearningStore(config.learning_store.database_path)
    persisted = store.get_pipeline_run(result.run_id)
    assert persisted is not None
    assert persisted["unique_offer_count"] == 5
    assert (
        persisted["run_metadata"]["offer_identity_source"]
        == "stable_offer_fingerprint_v1"
    )
    assert persisted["run_metadata"]["inference_offer_count"] == 5


def test_mixed_brand_run_preserves_every_terminal_offer_and_proposal(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = DashboardProcessingService(config).process(
        DashboardProcessRequest(
            filename="mixed-offers.csv",
            content=_mixed_offer_csv(),
            deployment_mode=MLDeploymentMode.ASSISTED,
            model_id=MODEL_ID,
        )
    )

    assert result.summary["unique_offers"] == 10
    assert result.summary["inference_offers"] == 5
    assert result.summary["terminal_decision_count"] == 10
    assert result.summary["decision_coverage_complete"] is True
    assert result.summary["competitor_offer_count"] == 5
    assert (
        result.summary["competitor_diagnostics"][
            "source_competitor_offer_count"
        ]
        == 5
    )

    store = LearningStore(config.learning_store.database_path)
    artifacts = DashboardRunService(config, store).downloads(result.run_id)
    mapping = next(
        artifact for artifact in artifacts if artifact.key == "sku_mapping"
    )
    business_mapping = pd.read_csv(BytesIO(mapping.content))
    assert len(business_mapping) == 5
    assert business_mapping["source_offer_id"].nunique() == 5
    assert business_mapping["source_brand"].eq("Al-Kabeer").all()
    assert business_mapping["matched_master_sku"].fillna("").ne("").all()
    assert business_mapping["mapping_status"].eq("MANUAL_REVIEW").all()
    assert business_mapping["requires_human_review"].eq(True).all()

    connection = sqlite3.connect(config.learning_store.database_path)
    try:
        decision_count, distinct_count = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT offer_id)
            FROM offer_decisions WHERE run_id = ?
            """,
            (result.run_id,),
        ).fetchone()
    finally:
        connection.close()
    assert decision_count == distinct_count == 10


def test_competitor_only_run_preserves_terminal_offer_lifecycle(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    own = pd.read_csv(BytesIO(_five_offer_csv()))
    own["Brand Name"] = "Other Brand"
    own["Offer Name"] = own["Offer Name"].str.replace(
        "Al Kabeer", "Other Brand", regex=False
    )
    result = DashboardProcessingService(config).process(
        DashboardProcessRequest(
            filename="competitor-only.csv",
            content=own.to_csv(index=False).encode("utf-8"),
            deployment_mode=MLDeploymentMode.ASSISTED,
            model_id=MODEL_ID,
        )
    )

    assert result.summary["unique_offers"] == 5
    assert result.summary["inference_offers"] == 0
    assert result.summary["terminal_decision_count"] == 5
    assert result.summary["decision_coverage_complete"] is True
    assert result.summary["competitor_offer_count"] == 5
    assert result.review_session_id is None

    store = LearningStore(config.learning_store.database_path)
    mapping = next(
        artifact
        for artifact in DashboardRunService(config, store).downloads(
            result.run_id
        )
        if artifact.key == "sku_mapping"
    )
    business_mapping = pd.read_csv(BytesIO(mapping.content))
    assert business_mapping.empty
    connection = sqlite3.connect(config.learning_store.database_path)
    try:
        competitor_lifecycle_count = connection.execute(
            """
            SELECT COUNT(*) FROM offer_decisions
            WHERE run_id = ? AND final_decision = 'COMPETITOR_OFFER'
            """,
            (result.run_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert competitor_lifecycle_count == 5


def test_business_flow_contract_on_current_40_row_fixture(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    fixture_path = Path(
        "tests/fixtures/clickflyer_business_flow_40.csv"
    )
    raw = pd.read_csv(fixture_path, dtype={"offerid": str})
    prepared = preprocess_clickflyer(raw)
    own = prepared[prepared["is_own"]].copy()
    competitors = prepared[~prepared["is_own"]].copy()

    result = DashboardProcessingService(config).process(
        DashboardProcessRequest(
            filename=fixture_path.name,
            content=fixture_path.read_bytes(),
            deployment_mode=MLDeploymentMode.ASSISTED,
            model_id=MODEL_ID,
            enable_llm_review=False,
        )
    )

    store = LearningStore(config.learning_store.database_path)
    artifacts = {
        artifact.key: artifact
        for artifact in DashboardRunService(config, store).downloads(
            result.run_id
        )
    }
    mapping = pd.read_csv(
        BytesIO(artifacts["sku_mapping"].content),
        dtype={"source_offer_id": str, "matched_master_sku": str},
    ).fillna("")
    competitor_export = pd.read_csv(
        BytesIO(artifacts["competitor_offers"].content),
        dtype={"master_sku": str},
    ).fillna("")
    persisted_run = store.get_pipeline_run(result.run_id)
    shadow_predictions = pd.read_csv(
        Path(persisted_run["output_paths"]["shadow_predictions_csv"]),
        dtype={"offer_group_id": str, "master_itemcode": str},
    ).fillna("")

    expected_own_ids = set(own["offerid"].astype(str))
    competitor_source = {
        str(row["offerid"]): str(row["Offer Name"])
        for _, row in competitors.iterrows()
    }
    assert len(mapping) == own["offerid"].nunique()
    assert set(mapping["source_offer_id"]) == expected_own_ids
    assert mapping["source_offer_id"].is_unique
    assert mapping["source_brand"].map(
        lambda value: value.lower().replace("-", " ").strip()
    ).eq("al kabeer").all()
    assert mapping["matched_master_sku"].ne("").all()
    manual = mapping["mapping_status"].eq("MANUAL_REVIEW")
    assert manual.any()
    assert mapping.loc[manual, "matched_master_sku"].ne("").all()
    assert mapping.loc[manual, "requires_human_review"].eq(True).all()

    # The selected SKU must be the candidate owning the highest probability.
    # The secondary keys make ties deterministic and mirror inference ranking.
    model_top = (
        shadow_predictions.sort_values(
            [
                "offer_group_id",
                "calibrated_probability",
                "candidate_rank",
                "master_itemcode",
            ],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("offer_group_id", keep="first")
        .set_index("offer_group_id")["master_itemcode"]
    )
    selected = mapping.set_index("source_offer_id")["matched_master_sku"]
    assert selected.to_dict() == model_top.loc[selected.index].to_dict()

    # Semantic parsing may ignore a contradictory variant for inference, but
    # the business export must retain the exact source field.
    raw_variants = (
        raw.drop_duplicates("offerid", keep="first")
        .set_index("offerid")["Variant"]
        .fillna("")
        .astype(str)
    )
    exported_variants = mapping.set_index("source_offer_id")[
        "source_variant"
    ]
    assert exported_variants.to_dict() == (
        raw_variants.loc[exported_variants.index].to_dict()
    )

    target_skus = set(mapping["matched_master_sku"]) - {""}
    assert len(competitor_export) == len(target_skus)
    assert set(competitor_export["master_sku"]) == target_skus
    assert competitor_export["master_sku"].is_unique
    assert competitor_export["competitor_reason"].ne("").all()
    for _, row in competitor_export.iterrows():
        aligned = [
            json.loads(row[column])
            for column in (
                "competitor_brand_names",
                "competitor_offer_ids",
                "competitor_offer_names",
                "competitor_offer_prices",
                "competitor_regular_prices",
            )
        ]
        assert {len(values) for values in aligned} == {
            int(row["competitor_count"])
        }
        for offer_id, offer_name in zip(
            aligned[1], aligned[2], strict=True
        ):
            assert offer_id in competitor_source
            assert offer_name == competitor_source[offer_id]

    diagnostics = result.summary["competitor_diagnostics"]
    expected_competitor_count = competitors["offerid"].nunique()
    assert diagnostics["source_competitor_offer_count"] == (
        expected_competitor_count
    )
    assert diagnostics["source_competitor_offers_evaluated"] == (
        expected_competitor_count
    )
    assert result.summary["runtime_components"] == {
        "candidate_generation": "ACTIVE",
        "feature_generation": "ACTIVE",
        "lightgbm": "ACTIVE",
        "llm": "NOT_USED",
        "competitor_discovery": "ACTIVE",
    }

    # Reopen the canonical configured database and reconstruct the page service.
    visible = DashboardReviewService(
        LearningStore(config.learning_store.database_path)
    ).runs_with_review_sessions()
    assert result.run_id in {str(run["run_id"]) for run in visible}
    session = store.review_session_for_run(result.run_id)
    assert session is not None
    assert len(
        store.review_questions(str(session["session_id"]))
    ) == len(mapping)



def test_upload_processing_source_has_no_retraining_calls() -> None:
    source = Path(
        "dashboard/services/processing_service.py"
    ).read_text(encoding="utf-8").lower()
    assert ".fit(" not in source
    assert "run_training_pipeline" not in source
    assert "train_model" not in source


def test_failed_run_persists_exact_error_and_ends_running_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    updates = []

    def failing_runner(*args, **kwargs):
        kwargs["progress"](
            "candidate_generation",
            1,
            5,
            "Generated one candidate group",
        )
        raise ValueError("representative competitor-stage prerequisite failure")

    service = DashboardProcessingService(
        config, pipeline_runner=failing_runner
    )
    request = DashboardProcessRequest(
        filename="failed.csv",
        content=_five_offer_csv(),
        deployment_mode=MLDeploymentMode.ASSISTED,
        model_id=MODEL_ID,
    )

    with pytest.raises(DashboardProcessingError) as captured:
        service.process(request, progress=updates.append)

    error = captured.value
    assert error.run_id
    assert error.failed_stage == "Mapping own-brand SKUs"
    assert error.last_completed_stage == "Preparing offers"
    assert updates[-1].state is ProcessingState.FAILED
    assert updates[-1].overall_percent < 100
    assert all(
        update.state is ProcessingState.RUNNING
        for update in updates[:-1]
    )

    store = LearningStore(config.learning_store.database_path)
    run = store.get_pipeline_run(str(error.run_id))
    assert run["status"] == "FAILED_DASHBOARD"
    assert run["error_summary"] == (
        "builtins.ValueError: "
        "representative competitor-stage prerequisite failure"
    )
    report = json.loads(
        Path(run["output_paths"]["failure_report"]).read_text(
            encoding="utf-8"
        )
    )
    assert report["failed_stage"] == "Mapping own-brand SKUs"
    assert report["last_completed_stage"] == "Preparing offers"
    assert "ValueError" in report["exact_error"]
    assert Path(run["output_paths"]["failure_log"]).is_file()


def _raise_original_application_error() -> None:
    raise TypeError("could not convert NoneType to float")


def _raise_wrapped_application_error() -> None:
    try:
        _raise_original_application_error()
    except TypeError as cause:
        raise ValueError("feature construction failed") from cause


def test_safe_fallback_report_preserves_pipeline_status_and_original_cause(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    updates = []

    def safe_fallback_runner(rows, *_args, run_id=None, **_kwargs):
        try:
            _raise_wrapped_application_error()
        except ValueError as error:
            details = capture_exception_details(error)
            return UnifiedInferenceResult(
                status="MODEL_ERROR_SAFE_FALLBACK",
                rows=rows,
                decisions=pd.DataFrame(),
                candidates=pd.DataFrame(),
                run_id=run_id,
                statistics={},
                output_paths={},
                shadow_result=None,
                error=str(error),
                original_exception=details,
                exception=error,
            )

    service = DashboardProcessingService(
        config,
        pipeline_runner=safe_fallback_runner,
    )
    request = DashboardProcessRequest(
        filename="safe-fallback.csv",
        content=_five_offer_csv(),
        deployment_mode=MLDeploymentMode.ASSISTED,
        model_id=MODEL_ID,
    )

    with pytest.raises(DashboardProcessingError) as captured:
        service.process(request, progress=updates.append)

    error = captured.value
    assert error.pipeline_status == "MODEL_ERROR_SAFE_FALLBACK"
    assert error.original_exception is not None
    assert error.original_exception["type"] == "TypeError"
    assert error.original_exception["message"] == (
        "could not convert NoneType to float"
    )
    assert error.original_exception["source_filename"] == Path(__file__).name
    assert error.original_exception["function"] == (
        "_raise_original_application_error"
    )
    assert isinstance(error.original_exception["line"], int)
    assert [
        item["type"]
        for item in error.original_exception["exception_chain"]
    ] == ["TypeError", "ValueError"]
    assert "direct cause" in error.original_exception["traceback"]
    assert error.__cause__ is not None
    assert error.__cause__.__cause__ is not None

    run = LearningStore(
        config.learning_store.database_path
    ).get_pipeline_run(str(error.run_id))
    report = json.loads(
        Path(run["output_paths"]["failure_report"]).read_text(
            encoding="utf-8"
        )
    )
    assert report["pipeline_status"] == "MODEL_ERROR_SAFE_FALLBACK"
    assert report["original_exception"] == error.original_exception
    assert report["failed_stage"] == "Mapping own-brand SKUs"
    assert report["last_completed_stage"] == "Preparing offers"
    assert "Unified inference ended with MODEL_ERROR_SAFE_FALLBACK" in (
        Path(run["output_paths"]["failure_log"]).read_text(
            encoding="utf-8"
        )
    )
    assert updates[-1].pipeline_status == "MODEL_ERROR_SAFE_FALLBACK"
    assert "inference" not in updates[-1].completed_stage_keys
