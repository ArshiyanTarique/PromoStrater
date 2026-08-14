"""End-to-end proof that shadow observation cannot alter production outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sku_mapping.agreement.policy import AgreementEvaluationResult
from sku_mapping.config import load_config
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.shadow import pipeline as shadow_pipeline
from sku_mapping.shadow.pipeline import (
    run_shadow_observation,
    run_shadow_observation_non_blocking,
)
from sku_mapping.shadow.predictor import RegisteredShadowPackage


class _FakePredictor:
    def predict_raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.linspace(-2.0, 2.0, len(frame))

    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.linspace(0.01, 0.99, len(frame))


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    offers = preprocess_clickflyer(
        pd.DataFrame(
            [
                {
                    "offerid": "offer-1",
                    "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                    "Product": "Chicken Nuggets-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "Original",
                    "Base Packsize": "400g",
                    "Retailer Name": "Retailer A",
                },
                {
                    "offerid": "offer-2",
                    "Offer Name": "Al Kabeer Chicken Strips 400g",
                    "Product": "Chicken Strips-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "Spicy",
                    "Base Packsize": "400g",
                    "Retailer Name": "Retailer B",
                },
            ]
        )
    )
    offers["ml_decision"] = ["AUTO_MATCH", "MANUAL_REVIEW"]
    offers["confidence_tier"] = ["high (ml)", "medium (ml)"]
    offers["matched_itemcode"] = ["001", "REVIEW_REQUIRED"]
    offers["suggested_itemcode"] = ["001", "002"]
    master = preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "001",
                    "Itemname": "CHICKEN NUGGETS",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Nuggets",
                    "Item Description": "Original chicken nuggets",
                    "Item-Spec": "400g",
                },
                {
                    "Itemcode": "002",
                    "Itemname": "CHICKEN STRIPS",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Strips",
                    "Item Description": "Spicy chicken strips",
                    "Item-Spec": "400g",
                },
                {
                    "Itemcode": "003",
                    "Itemname": "CHICKEN POPCORN",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Popcorn",
                    "Item Description": "Chicken popcorn",
                    "Item-Spec": "400g",
                },
            ]
        )
    )
    return offers, master


def _registered(tmp_path) -> RegisteredShadowPackage:
    path = tmp_path / "explicit-v3.joblib"
    path.write_bytes(b"test-shadow-package")
    package = {
        "model_id": "explicit-v3",
        "predictor": _FakePredictor(),
        "feature_columns": list(shadow_pipeline.MODEL_FEATURE_COLUMNS),
        "auto_match_threshold": 0.9,
        "manual_review_threshold": 0.1,
    }
    return RegisteredShadowPackage(
        package=package,
        registry_entry={"model_id": "explicit-v3"},
        package_path=path,
        package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _enabled_config(tmp_path):
    config = load_config("config/default.yaml")
    shadow = replace(
        config.shadow_mode,
        enabled=True,
        model_id="explicit-v3",
        package_reference=None,
        output_directory=tmp_path / "outputs" / "shadow",
        challenge_set_directory=tmp_path / "challenge_sets",
        top_k=3,
        sampling_counts={
            key: 1 for key in config.shadow_mode.sampling_counts
        },
    )
    return replace(config, shadow_mode=shadow)


def test_shadow_enabled_and_disabled_leave_production_bytes_and_state_unchanged(
    tmp_path, monkeypatch
) -> None:
    production, master = _frames()
    production_before = production.copy(deep=True)
    master_before = master.copy(deep=True)
    competitor_inputs_before = production[
        production["confidence_tier"].eq("high (ml)")
    ].copy(deep=True)
    mapping_path = tmp_path / "production_mapping.csv"
    competitor_path = tmp_path / "production_competitors.csv"
    production.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    competitor_inputs_before.to_csv(
        competitor_path, index=False, encoding="utf-8-sig"
    )
    mapping_bytes = mapping_path.read_bytes()
    competitor_bytes = competitor_path.read_bytes()

    disabled = load_config("config/default.yaml")
    disabled_result = run_shadow_observation_non_blocking(
        production, master, config=disabled
    )
    assert disabled_result.status == "DISABLED"

    monkeypatch.setattr(
        shadow_pipeline,
        "load_registered_shadow_package",
        lambda **_: _registered(tmp_path),
    )
    result = run_shadow_observation(
        production,
        master,
        config=_enabled_config(tmp_path),
        shadow_run_id="shadow-isolation-test",
    )

    pd.testing.assert_frame_equal(production, production_before, check_exact=True)
    pd.testing.assert_frame_equal(master, master_before, check_exact=True)
    pd.testing.assert_frame_equal(
        production[production["confidence_tier"].eq("high (ml)")],
        competitor_inputs_before,
        check_exact=True,
    )
    assert mapping_path.read_bytes() == mapping_bytes
    assert competitor_path.read_bytes() == competitor_bytes
    assert result.prediction_rows == 6

    predictions = pd.read_parquet(
        result.output_paths["shadow_predictions_parquet"]
    )
    assert len(predictions) == 6
    assert predictions.groupby("offer_group_id").size().eq(3).all()
    assert set(predictions["shadow_decision_bucket"]) <= {
        "SHADOW_HIGH_SCORE",
        "SHADOW_REVIEW",
        "SHADOW_LOW_SCORE",
    }
    assert not predictions["shadow_decision_bucket"].str.contains(
        "AUTO_MATCH", regex=False
    ).any()
    review = pd.read_csv(
        result.output_paths["human_review_template"],
        dtype="string",
        keep_default_na=False,
    )
    assert review["human_label"].eq("").all()
    assert review["selected_master_itemcode"].eq("").all()


def test_shadow_failure_is_non_fatal_and_preserves_production(
    tmp_path, monkeypatch
) -> None:
    production, master = _frames()
    before = production.copy(deep=True)
    monkeypatch.setattr(
        shadow_pipeline,
        "load_registered_shadow_package",
        lambda **_: (_ for _ in ()).throw(ValueError("broken package")),
    )
    result = run_shadow_observation_non_blocking(
        production,
        master,
        config=_enabled_config(tmp_path),
    )
    assert result.status == "FAILED_CLOSED"
    assert "broken package" in result.error
    assert result.original_exception is not None
    assert result.original_exception["type"] == "ValueError"
    assert result.original_exception["message"] == "broken package"
    assert result.original_exception["source_filename"] == Path(__file__).name
    assert result.original_exception["function"] == "<genexpr>"
    assert isinstance(result.original_exception["line"], int)
    manifest = json.loads(
        result.output_paths["failure_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["status"] == "FAILED_CLOSED"
    assert manifest["original_exception"] == result.original_exception
    pd.testing.assert_frame_equal(production, before, check_exact=True)
    assert result.output_paths["failure_manifest"].is_file()


def test_authoritative_integration_hook_runs_only_after_production_exports() -> None:
    source = (
        Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    ).read_text(encoding="utf-8")
    hook_position = source.index("run_configured_shadow_sidecar(df, master)")
    assert hook_position > source.index("file1.to_csv(FINAL_CLICKFLYER_CSV")
    assert hook_position > source.index("file2.to_csv(FINAL_COMPETITOR_CSV")



def test_llm_provider_failure_does_not_crash_or_modify_production(
    tmp_path, monkeypatch
) -> None:
    production, master = _frames()
    production_before = production.copy(deep=True)
    master_before = master.copy(deep=True)
    config = _enabled_config(tmp_path)
    config = replace(
        config,
        embedding=replace(
            config.embedding,
            enabled=True,
            backend="local_hashing",
            model_name="isolation-hashing",
            model_version="isolation-v1",
            cache_path=tmp_path / "embedding-cache.sqlite3",
        ),
        agreement=replace(
            config.agreement,
            lightgbm_auto_accept_threshold=1.0,
        ),
        llm_review=replace(
            config.llm_review,
            enabled=True,
            provider="unavailable-test-provider",
            model="missing-local-model",
            maximum_retries=0,
            cache_path=tmp_path / "llm-cache.sqlite3",
        ),
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "load_registered_shadow_package",
        lambda **_: _registered(tmp_path),
    )
    monkeypatch.setattr(
        shadow_pipeline,
        "evaluate_candidate_agreement",
        lambda candidates, **_: AgreementEvaluationResult(
            results=(),
            frame=pd.DataFrame(
                {
                    "offer_id": candidates[
                        "offer_group_id"
                    ].drop_duplicates(),
                    "agreement_status": "DISAGREEMENT",
                    "routing_decision": "LLM_REVIEW",
                    "reason_codes": "DIFFERENT_TOP_CANDIDATE",
                    "routing_reason": "DIFFERENT_TOP_CANDIDATE",
                }
            ),
        ),
    )

    result = run_shadow_observation(
        production,
        master,
        config=config,
        shadow_run_id="llm-provider-failure-isolation",
    )

    assert result.status == "COMPLETED_OBSERVATIONAL_ONLY"
    llm_results = pd.read_csv(
        result.output_paths["llm_review_results"],
        keep_default_na=False,
    )
    assert not llm_results.empty
    assert llm_results["llm_review_status"].eq(
        "PROVIDER_FAILURE"
    ).all()
    assert llm_results["llm_final_route"].eq(
        "MANUAL_REVIEW"
    ).all()
    assert not llm_results["llm_production_applied"].astype(bool).any()
    pd.testing.assert_frame_equal(
        production, production_before, check_exact=True
    )
    pd.testing.assert_frame_equal(master, master_before, check_exact=True)
