"""Incremental loading and developer/production run-mode isolation.

The property that matters most is equivalence: splitting a dump across two
incremental loads must produce the same cumulative business outputs as one
full load of the whole thing. If that ever stops holding, incremental loading
is silently shipping a different answer than the pipeline it replaced.
"""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from dashboard.services.processing_service import (
    DashboardProcessRequest,
    DashboardProcessingService,
    DuplicateProcessingError,
)
from dashboard.services.run_service import DashboardRunService
from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.learning.store import (
    DEVELOPER_RUN_MODE,
    PRODUCTION_RUN_MODE,
    LearningStore,
)

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


def _fixture_frame() -> pd.DataFrame:
    return pd.read_csv(
        "tests/fixtures/clickflyer_40_distinct_offerids.csv",
        dtype={"offerid": str},
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def _request(content: bytes, *, run_mode: str, filename: str):
    return DashboardProcessRequest(
        filename=filename,
        content=content,
        deployment_mode=MLDeploymentMode.SHADOW,
        model_id=MODEL_ID,
        run_mode=run_mode,
    )


def _download(config, store, run_id: str, key: str) -> pd.DataFrame:
    artifacts = DashboardRunService(config, store).downloads(run_id)
    artifact = next(item for item in artifacts if item.key == key)
    return pd.read_csv(BytesIO(artifact.content))


def _sorted_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values("entity_id")
        .reset_index(drop=True)
        .drop(columns=["run_id"])
    )


# ---------------------------------------------------------------------------
# The equivalence property
# ---------------------------------------------------------------------------


def test_two_incremental_loads_equal_one_full_load(tmp_path: Path) -> None:
    """Splitting a dump in half must not change the cumulative answer."""
    frame = _fixture_frame()
    first_half = frame.iloc[:20].copy()
    second_half = frame.iloc[20:].copy()

    incremental_config = _config(tmp_path / "incremental")
    service = DashboardProcessingService(incremental_config)
    service.process(
        _request(
            _csv_bytes(first_half),
            run_mode=PRODUCTION_RUN_MODE,
            filename="week1.csv",
        )
    )
    second = service.process(
        _request(
            _csv_bytes(second_half),
            run_mode=PRODUCTION_RUN_MODE,
            filename="week2.csv",
        )
    )

    whole_config = _config(tmp_path / "whole")
    whole = DashboardProcessingService(whole_config).process(
        _request(
            _csv_bytes(frame),
            run_mode=PRODUCTION_RUN_MODE,
            filename="whole.csv",
        )
    )

    incremental_store = LearningStore(
        incremental_config.learning_store.database_path
    )
    whole_store = LearningStore(whole_config.learning_store.database_path)

    incremental_mapping = _download(
        incremental_config, incremental_store, second.run_id, "sku_mapping"
    )
    whole_mapping = _download(
        whole_config, whole_store, whole.run_id, "sku_mapping"
    )

    assert len(incremental_mapping) == len(whole_mapping)
    pd.testing.assert_frame_equal(
        _sorted_mapping(incremental_mapping),
        _sorted_mapping(whole_mapping),
        check_like=True,
    )

    # Competitor discovery is the cross-sectional stage: it answers a question
    # about every offer ever seen, so a delta run must still cover every
    # Master SKU the full run covers.
    incremental_competitors = _download(
        incremental_config,
        incremental_store,
        second.run_id,
        "competitor_offers",
    )
    whole_competitors = _download(
        whole_config, whole_store, whole.run_id, "competitor_offers"
    )
    assert set(incremental_competitors["master_sku"]) == set(
        whole_competitors["master_sku"]
    )


def test_second_load_only_infers_the_new_offers(tmp_path: Path) -> None:
    """The whole point: unchanged offers must not be inferred twice."""
    frame = _fixture_frame()
    config = _config(tmp_path)
    service = DashboardProcessingService(config)

    service.process(
        _request(
            _csv_bytes(frame.iloc[:20]),
            run_mode=PRODUCTION_RUN_MODE,
            filename="week1.csv",
        )
    )
    # Week 2 repeats every week-1 offer and adds twenty more.
    second = service.process(
        _request(
            _csv_bytes(frame),
            run_mode=PRODUCTION_RUN_MODE,
            filename="week2.csv",
        )
    )

    assert second.summary["incremental_new_offers"] == 20
    assert second.summary["incremental_skipped_offers"] == 20
    assert second.summary["incremental_revised_offers"] == 0
    # Cumulative outputs still describe all forty.
    store = LearningStore(config.learning_store.database_path)
    mapping = _download(config, store, second.run_id, "sku_mapping")
    assert mapping["source_offer_id"].nunique() == 40


def test_corrected_price_is_reprocessed_not_skipped(tmp_path: Path) -> None:
    """A revision under an unchanged offerid must re-enter inference."""
    frame = _fixture_frame().iloc[:10].copy()
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    service.process(
        _request(
            _csv_bytes(frame),
            run_mode=PRODUCTION_RUN_MODE,
            filename="week1.csv",
        )
    )

    corrected = frame.copy()
    corrected.loc[corrected.index[0], "Offer Price"] = (
        float(corrected.loc[corrected.index[0], "Offer Price"]) + 3.5
    )
    second = service.process(
        _request(
            _csv_bytes(corrected),
            run_mode=PRODUCTION_RUN_MODE,
            filename="week1_corrected.csv",
        )
    )

    assert second.summary["incremental_revised_offers"] == 1
    assert second.summary["incremental_new_offers"] == 0
    assert second.summary["incremental_skipped_offers"] == 9


def test_backdated_rows_are_not_dropped(tmp_path: Path) -> None:
    """Identity, not date, decides the work.

    A dump whose offers all end earlier than what is already loaded must
    still be processed. A date watermark would discard exactly this.
    """
    frame = _fixture_frame()
    config = _config(tmp_path)
    service = DashboardProcessingService(config)

    recent = frame.iloc[:10].copy()
    if "Offer End Date" in recent.columns:
        recent["Offer End Date"] = "2026-12-31"
    service.process(
        _request(
            _csv_bytes(recent),
            run_mode=PRODUCTION_RUN_MODE,
            filename="recent.csv",
        )
    )

    backdated = frame.iloc[10:20].copy()
    if "Offer End Date" in backdated.columns:
        backdated["Offer End Date"] = "2020-01-01"
    second = service.process(
        _request(
            _csv_bytes(backdated),
            run_mode=PRODUCTION_RUN_MODE,
            filename="backdated.csv",
        )
    )

    assert second.summary["incremental_new_offers"] == 10
    assert second.summary["incremental_skipped_offers"] == 0


# ---------------------------------------------------------------------------
# Run-mode isolation
# ---------------------------------------------------------------------------


def test_developer_runs_reprocess_in_full_and_leave_no_trace(
    tmp_path: Path,
) -> None:
    frame = _fixture_frame().iloc[:10]
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    store = LearningStore(config.learning_store.database_path)

    first = service.process(
        _request(
            _csv_bytes(frame),
            run_mode=DEVELOPER_RUN_MODE,
            filename="experiment.csv",
        )
    )
    second = service.process(
        _request(
            _csv_bytes(frame),
            run_mode=DEVELOPER_RUN_MODE,
            filename="experiment.csv",
        )
    )

    # No ledger, so the identical second run does all the work again.
    assert first.summary["incremental_loading"] is False
    assert second.summary["incremental_loading"] is False
    assert store.offer_ledger_watermark()["offer_count"] == 0

    # Outputs live in the developer tree, never the production one.
    developer_root = config.dashboard.output_directory / DEVELOPER_RUN_MODE
    assert (developer_root / second.run_id).is_dir()
    assert not (
        config.dashboard.output_directory
        / PRODUCTION_RUN_MODE
        / second.run_id
    ).is_dir()

    # And they stay out of the production listings.
    assert [run["run_id"] for run in store.list_pipeline_runs()] == []
    developer_runs = {
        run["run_id"]
        for run in store.list_pipeline_runs(run_mode=DEVELOPER_RUN_MODE)
    }
    assert {first.run_id, second.run_id} == developer_runs


def test_developer_run_does_not_block_production_on_the_same_bytes(
    tmp_path: Path,
) -> None:
    """Duplicate detection is per mode, so the two loops never collide."""
    content = _csv_bytes(_fixture_frame().iloc[:10])
    config = _config(tmp_path)
    service = DashboardProcessingService(config)

    # Developer mode repeats freely; the same bytes again is not an error.
    service.process(
        _request(content, run_mode=DEVELOPER_RUN_MODE, filename="dump.csv")
    )
    service.process(
        _request(content, run_mode=DEVELOPER_RUN_MODE, filename="dump.csv")
    )
    production = service.process(
        _request(content, run_mode=PRODUCTION_RUN_MODE, filename="dump.csv")
    )
    assert production.summary["run_mode"] == PRODUCTION_RUN_MODE

    # Within one mode the guard still fires.
    with pytest.raises(DuplicateProcessingError):
        service.process(
            _request(
                content, run_mode=PRODUCTION_RUN_MODE, filename="dump.csv"
            )
        )


def test_developer_labels_stay_out_of_training(tmp_path: Path) -> None:
    """Developer mode runs the full pipeline, review staging included.

    What keeps it honest is not a blocked code path but the selection: those
    labels are not counted toward retraining and are not handed to a
    training snapshot unless a caller asks for them by name.
    """
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    store = LearningStore(config.learning_store.database_path)

    developer = service.process(
        _request(
            _csv_bytes(_fixture_frame().iloc[:10]),
            run_mode=DEVELOPER_RUN_MODE,
            filename="experiment.csv",
        )
    )
    # The full pipeline really did run: a review session exists.
    assert developer.review_session_id is not None
    assert store.review_session_for_run(developer.run_id) is not None

    assert store.count_new_gold_labels_since_last_model() == 0
    production_labels = store.governed_training_labels()
    every_label = store.governed_training_labels(run_mode=None)
    assert len(production_labels["gold"]) <= len(every_label["gold"])


def test_run_mode_is_recorded_on_the_run_and_its_summary(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    service = DashboardProcessingService(config)
    store = LearningStore(config.learning_store.database_path)

    result = service.process(
        _request(
            _csv_bytes(_fixture_frame().iloc[:5]),
            run_mode=PRODUCTION_RUN_MODE,
            filename="dump.csv",
        )
    )
    assert result.summary["run_mode"] == PRODUCTION_RUN_MODE
    assert store.get_pipeline_run(result.run_id)["run_mode"] == (
        PRODUCTION_RUN_MODE
    )
    watermark = store.offer_ledger_watermark()
    assert watermark["offer_count"] == 5
