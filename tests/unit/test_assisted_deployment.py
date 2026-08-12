"""Safety and provenance tests for explicit assisted deployment."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode, MatchDecision
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.ml import deployment
from sku_mapping.shadow.predictor import RegisteredShadowPackage


class _FrozenRawModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.fit_calls = 0

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), self.probability)
        return np.column_stack([1 - positive, positive])

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("Assisted inference must never fit a model")


class _FrozenPredictor:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_calibrated_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.probability)


def _config(tmp_path, *, threshold: float = 0.85):
    config = load_config("config/default.yaml")
    return replace(
        config,
        output=replace(config.output, output_dir=tmp_path / "outputs"),
        ml=replace(
            config.ml,
            mode=MLDeploymentMode.ASSISTED,
            model_id="registered-v3",
            auto_accept_threshold=threshold,
        ),
    )


def _frames(*, protein_conflict: bool = False):
    offer = preprocess_clickflyer(
        pd.DataFrame(
            [
                {
                    "offerid": "offer-1",
                    "Offer Name": "Al Kabeer Chicken Nuggets 400g",
                    "Product": "Chicken Nuggets-Frozen",
                    "Brand Name": "Al Kabeer",
                    "Variant": "Original",
                    "Base Packsize": "400g",
                }
            ]
        )
    )
    offer["suggested_itemcode"] = "001"
    offer["ml_probability"] = 0.72
    offer["ml_decision"] = "MANUAL_REVIEW"
    master_protein = "BEEF" if protein_conflict else "CHICKEN"
    master = preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "001",
                    "Itemname": f"{master_protein} NUGGETS",
                    "Item-Cat-2": (
                        "Meat" if protein_conflict else "Chicken"
                    ),
                    "Item-Cat-4": "Nuggets",
                    "Item Description": f"{master_protein} nuggets",
                    "Item-Spec": "400g",
                }
            ]
        )
    )
    return offer, master


def _registered(tmp_path, probability: float):
    path = tmp_path / "registered-v3.joblib"
    path.write_bytes(b"immutable-assisted-test-package")
    model = _FrozenRawModel(probability)
    return (
        RegisteredShadowPackage(
            package={
                "model_id": "registered-v3",
                "model": model,
                "predictor": _FrozenPredictor(probability),
            },
            registry_entry={"model_id": "registered-v3"},
            package_path=path,
            package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
        model,
    )


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.85, MatchDecision.AUTO_ACCEPT.value),
        (0.849999, MatchDecision.MANUAL_REVIEW.value),
    ],
)
def test_configured_threshold_is_inclusive_and_below_routes_to_review(
    tmp_path, monkeypatch, probability, expected
) -> None:
    registered, _ = _registered(tmp_path, probability)
    monkeypatch.setattr(deployment, "_load_package", lambda *_: registered)
    offers, master = _frames()

    result = deployment.apply_assisted_decisions(
        offers,
        master,
        config=_config(tmp_path),
        persist_records=False,
        run_id="threshold-test",
    )

    assert result.rows.loc[0, "assisted_decision"] == expected
    assert result.rows.loc[0, "assisted_threshold_source"] == "user_configured"
    assert result.rows.loc[
        0, "assisted_production_threshold_approved"
    ] == np.bool_(False)
    assert result.rows.loc[0, "assisted_decision"] != MatchDecision.NO_MATCH.value


def test_hard_conflict_blocks_high_probability_auto_accept(
    tmp_path, monkeypatch
) -> None:
    registered, _ = _registered(tmp_path, 0.99)
    monkeypatch.setattr(deployment, "_load_package", lambda *_: registered)
    offers, master = _frames(protein_conflict=True)

    result = deployment.apply_assisted_decisions(
        offers,
        master,
        config=_config(tmp_path),
        persist_records=False,
    )

    record = result.decisions.iloc[0]
    assert record["decision"] == MatchDecision.MANUAL_REVIEW.value
    assert record["protein_conflict"] == np.bool_(True)
    assert record["safety_override_applied"] == np.bool_(True)


def test_model_load_failure_is_nonfatal_and_retains_existing_decision(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        deployment,
        "_load_package",
        lambda *_: (_ for _ in ()).throw(ValueError("invalid package")),
    )
    offers, master = _frames()
    before = offers.copy(deep=True)

    result = deployment.apply_assisted_decisions(
        offers,
        master,
        config=_config(tmp_path),
        persist_records=False,
    )

    assert result.status == "MODEL_ERROR_SAFE_FALLBACK"
    assert result.rows.loc[0, "assisted_decision"] == MatchDecision.MODEL_ERROR.value
    assert result.rows.loc[0, "ml_decision"] == before.loc[0, "ml_decision"]
    pd.testing.assert_frame_equal(offers, before, check_exact=True)


def test_decision_provenance_is_complete_and_no_online_learning_occurs(
    tmp_path, monkeypatch
) -> None:
    registered, model = _registered(tmp_path, 0.91)
    package_bytes = registered.package_path.read_bytes()
    monkeypatch.setattr(deployment, "_load_package", lambda *_: registered)
    offers, master = _frames()

    result = deployment.apply_assisted_decisions(
        offers,
        master,
        config=_config(tmp_path),
        persist_records=True,
        run_id="provenance-test",
    )

    required = {
        "run_id",
        "timestamp",
        "ml_mode",
        "offer_identifier",
        "candidate_identifier",
        "candidate_rank",
        "model_id",
        "model_package_sha256",
        "raw_probability",
        "calibrated_probability",
        "auto_accept_threshold",
        "threshold_source",
        "production_threshold_approved",
        "decision",
        "decision_reason",
        "safety_override_applied",
        "conflict_flags",
    }
    assert required.issubset(result.decisions.columns)
    assert model.fit_calls == 0
    assert registered.package_path.read_bytes() == package_bytes
    assert result.output_paths["assisted_decisions"].is_file()


def test_disabled_mode_returns_original_object_without_side_effects(
    tmp_path,
) -> None:
    offers, master = _frames()
    config = load_config("config/default.yaml")
    config = replace(
        config, ml=replace(config.ml, mode=MLDeploymentMode.DISABLED)
    )
    result = deployment.apply_assisted_decisions(
        offers,
        master,
        config=config,
    )
    assert result.status == "DISABLED"
    assert result.rows is offers
    assert result.decisions.empty


def test_assisted_monitoring_receives_resulting_decisions(
    tmp_path, monkeypatch
) -> None:
    registered, _ = _registered(tmp_path, 0.9)
    monkeypatch.setattr(deployment, "_load_package", lambda *_: registered)
    offers, master = _frames()
    config = _config(tmp_path)
    assisted = deployment.apply_assisted_decisions(
        offers,
        master,
        config=config,
        persist_records=False,
    )
    captured = {}

    def _capture(rows, product_master, *, config, model_directory=None):
        captured["rows"] = rows.copy(deep=True)
        captured["config"] = config
        return deployment.ShadowRunResult(
            status="COMPLETED_OBSERVATIONAL_ONLY",
            shadow_run_id="monitor",
            prediction_rows=1,
            offer_groups=1,
            failed_shadow_predictions=0,
            output_paths={},
        )

    monkeypatch.setattr(
        deployment, "run_shadow_observation_non_blocking", _capture
    )
    result = deployment.run_assisted_monitoring_non_blocking(
        assisted.rows,
        master,
        config=config,
    )
    assert result.status == "COMPLETED_OBSERVATIONAL_ONLY"
    assert captured["config"].shadow_mode.enabled is True
    assert captured["rows"]["assisted_decision"].tolist() == ["AUTO_ACCEPT"]

