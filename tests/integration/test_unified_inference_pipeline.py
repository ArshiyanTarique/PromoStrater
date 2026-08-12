"""End-to-end mode and artifact guards for unified assisted inference."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.inference import pipeline as unified
from sku_mapping.shadow.pipeline import ShadowRunResult


def _config(tmp_path: Path, mode: MLDeploymentMode):
    config = load_config("config/default.yaml")
    return replace(
        config,
        output=replace(config.output, output_dir=tmp_path / "outputs"),
        ml=replace(
            config.ml,
            mode=mode,
            model_id=None
            if mode is MLDeploymentMode.DISABLED
            else "registered-v3",
        ),
        shadow_mode=replace(
            config.shadow_mode,
            output_directory=tmp_path / "shadow",
        ),
        learning_store=replace(
            config.learning_store,
            database_path=tmp_path / "learning.db",
            csv_export_directory=tmp_path / "learning_exports",
        ),
    )


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "offer_group_id": ["offer-auto", "offer-review"],
            "Offer Name": ["Safe offer", "Review offer"],
            "suggested_itemcode": ["OLD-A", "OLD-B"],
            "matched_itemcode": ["OLD-A", "OLD-B"],
            "ml_decision": ["AUTO_MATCH", "AUTO_MATCH"],
            "confidence_tier": ["high (ml)", "high (ml)"],
        }
    )


def _candidate_rows() -> pd.DataFrame:
    base = {
        "candidate_rank": 1,
        "embedding_model_id": "embedding:test-v1",
        "llm_model_id": "",
        "llm_review_status": "",
        "llm_final_route": "",
        "llm_parsed_decision": "",
        "llm_selected_candidate": "",
        "llm_confidence": None,
        "llm_routing_reason": "",
        "protein_conflict": False,
        "mixed_protein_ambiguity": False,
        "strong_family_conflict": False,
        "strong_size_weight_conflict": False,
        "strong_pack_format_conflict": False,
        "feature_generation_failure": False,
        "missing_master": False,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "offer_group_id": "offer-auto",
                "master_itemcode": "SKU-A",
                "master_item_description": "Accepted SKU A",
                "calibrated_probability": 0.92,
                "embedding_similarity": 0.75,
                "agreement_status": "SAFE_AGREEMENT",
                "routing_decision": "AUTO_ACCEPT",
                "same_top_candidate": True,
                "lightgbm_top_candidate": "SKU-A",
                "routing_reason": "SAME_TOP_CANDIDATE",
            },
            {
                **base,
                "offer_group_id": "offer-review",
                "master_itemcode": "SKU-B",
                "master_item_description": "Review SKU B",
                "calibrated_probability": 0.72,
                "embedding_similarity": 0.44,
                "agreement_status": "DISAGREEMENT",
                "routing_decision": "LLM_REVIEW",
                "same_top_candidate": False,
                "lightgbm_top_candidate": "SKU-B",
                "routing_reason": "DIFFERENT_TOP_CANDIDATE",
                "llm_review_status": "TIMEOUT",
                "llm_final_route": "MANUAL_REVIEW",
                "llm_routing_reason": "TIMEOUT_TO_FAIL_ROUTE",
            },
        ]
    )


def _completed_shadow(tmp_path: Path, run_id: str) -> ShadowRunResult:
    directory = tmp_path / "shadow" / run_id
    directory.mkdir(parents=True)
    predictions = directory / "shadow_predictions.parquet"
    manifest = directory / "shadow_run_manifest.json"
    _candidate_rows().to_parquet(predictions, index=False)
    manifest.write_text(
        json.dumps(
            {
                "llm_review": {"provider_calls": 1},
                "stage_runtimes_seconds": {
                    "candidate_generation_and_features": 0.1,
                    "lightgbm_scoring": 0.2,
                    "embedding_scoring": 0.3,
                    "agreement_policy": 0.01,
                    "llm_review": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    return ShadowRunResult(
        status="COMPLETED_OBSERVATIONAL_ONLY",
        shadow_run_id=run_id,
        prediction_rows=2,
        offer_groups=2,
        failed_shadow_predictions=0,
        output_paths={
            "shadow_predictions_parquet": predictions,
            "run_manifest": manifest,
        },
    )


def test_disabled_mode_is_exactly_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    offers = _offers()
    called = False

    def _unexpected(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Disabled mode cannot run inference")

    monkeypatch.setattr(
        unified, "run_shadow_observation_non_blocking", _unexpected
    )
    result = unified.run_unified_inference(
        offers,
        pd.DataFrame(),
        config=_config(tmp_path, MLDeploymentMode.DISABLED),
    )
    assert result.status == "DISABLED"
    assert result.rows is offers
    assert result.decisions.empty
    assert called is False


def test_shadow_mode_scores_but_does_not_modify_production(
    tmp_path: Path, monkeypatch
) -> None:
    offers = _offers()
    before = offers.copy(deep=True)
    monkeypatch.setattr(
        unified,
        "run_shadow_observation_non_blocking",
        lambda *_args, shadow_run_id=None, **_kwargs: _completed_shadow(
            tmp_path, str(shadow_run_id)
        ),
    )
    result = unified.run_unified_inference(
        offers,
        pd.DataFrame(),
        config=_config(tmp_path, MLDeploymentMode.SHADOW),
        run_id="shadow-unified-test",
        persist_records=False,
    )
    assert result.status == "COMPLETED_SHADOW_OBSERVATIONAL_ONLY"
    assert result.rows is offers
    pd.testing.assert_frame_equal(offers, before, check_exact=True)
    assert not result.decisions.empty


def test_assisted_mode_applies_only_explicit_policy_and_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    offers = _offers()
    master = pd.DataFrame({"Itemcode": ["SKU-A", "SKU-B"]})
    master_before = master.copy(deep=True)
    monkeypatch.setattr(
        unified,
        "run_shadow_observation_non_blocking",
        lambda *_args, shadow_run_id=None, **_kwargs: _completed_shadow(
            tmp_path, str(shadow_run_id)
        ),
    )
    result = unified.run_unified_inference(
        offers,
        master,
        config=_config(tmp_path, MLDeploymentMode.ASSISTED),
        run_id="assisted-unified-test",
    )

    applied = result.rows.set_index("offer_group_id")
    assert applied.loc["offer-auto", "final_decision"] == "AUTO_ACCEPT"
    assert applied.loc["offer-auto", "matched_itemcode"] == "SKU-A"
    assert applied.loc["offer-auto", "ml_decision"] == "AUTO_MATCH"
    assert applied.loc["offer-review", "final_decision"] == "MANUAL_REVIEW"
    assert applied.loc["offer-review", "matched_itemcode"] == (
        "REVIEW_REQUIRED"
    )
    assert applied.loc["offer-review", "ml_decision"] == "MANUAL_REVIEW"
    assert result.statistics["offers_processed"] == 2
    assert result.statistics["candidates_generated"] == 2
    assert result.statistics["auto_accept_count"] == 1
    assert result.statistics["manual_review_count"] == 1
    assert result.statistics["llm_calls"] == 1
    assert result.statistics["stage_runtimes_seconds"][
        "candidate_generation_and_features"
    ] == 0.1
    assert result.output_paths["unified_decisions"].is_file()
    assert result.output_paths["unified_statistics"].is_file()
    assert result.decisions["run_id"].eq("assisted-unified-test").all()
    pd.testing.assert_frame_equal(master, master_before, check_exact=True)


def test_model_package_failure_is_nonfatal_and_not_eligible(
    tmp_path: Path, monkeypatch
) -> None:
    offers = _offers()
    before = offers.copy(deep=True)
    monkeypatch.setattr(
        unified,
        "run_shadow_observation_non_blocking",
        lambda *_args, shadow_run_id=None, **_kwargs: ShadowRunResult(
            status="FAILED_CLOSED",
            shadow_run_id=shadow_run_id,
            prediction_rows=0,
            offer_groups=0,
            failed_shadow_predictions=0,
            output_paths={},
            error="registered package failed compatibility validation",
        ),
    )
    result = unified.run_unified_inference(
        offers,
        pd.DataFrame(),
        config=_config(tmp_path, MLDeploymentMode.ASSISTED),
        run_id="model-failure-test",
        persist_records=False,
    )
    assert result.status == "MODEL_ERROR_SAFE_FALLBACK"
    assert result.decisions["final_decision"].eq("MODEL_ERROR").all()
    assert not result.rows["final_eligible_mapping"].any()
    assert result.rows["matched_itemcode"].eq("REVIEW_REQUIRED").all()
    assert offers.equals(before)


def test_unexpected_pipeline_failure_is_nonblocking_and_ineligible(
    tmp_path: Path, monkeypatch
) -> None:
    offers = _offers()
    before = offers.copy(deep=True)
    monkeypatch.setattr(
        unified,
        "run_unified_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected stage failure")
        ),
    )
    result = unified.run_unified_inference_non_blocking(
        offers,
        pd.DataFrame(),
        config=_config(tmp_path, MLDeploymentMode.ASSISTED),
        run_id="unexpected-failure-test",
        persist_records=False,
    )
    assert result.status == "MODEL_ERROR_SAFE_FALLBACK"
    assert result.original_exception is not None
    assert result.original_exception["type"] == "RuntimeError"
    assert result.original_exception["message"] == "unexpected stage failure"
    assert result.exception is not None
    assert result.original_exception["source_filename"] == Path(__file__).name
    assert result.original_exception["function"] == "<genexpr>"
    assert "unexpected stage failure" in (
        result.original_exception["traceback"]
    )
    assert result.decisions["final_decision"].eq("MODEL_ERROR").all()
    assert not result.rows["final_eligible_mapping"].any()
    assert result.rows["matched_itemcode"].eq("REVIEW_REQUIRED").all()
    pd.testing.assert_frame_equal(offers, before, check_exact=True)


def test_upload_inference_source_has_no_training_call() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/sku_mapping/inference/pipeline.py"
    ).read_text(encoding="utf-8").lower()
    assert ".fit(" not in source
    assert "run_training_pipeline" not in source
    assert "training_features" not in source
