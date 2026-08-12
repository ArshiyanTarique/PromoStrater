"""Phase 6 output is observed without changing inference or training state."""

from __future__ import annotations

import sqlite3
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sku_mapping.config import load_config
from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.learning.observer import observe_unified_result
from sku_mapping.learning.store import LearningStore


def test_unified_candidate_outputs_persist_with_five_questions(
    tmp_path: Path,
) -> None:
    base = load_config("config/default.yaml")
    config = replace(
        base,
        learning_store=replace(
            base.learning_store,
            database_path=tmp_path / "learning.db",
            csv_export_directory=tmp_path / "exports",
        ),
    )
    source = tmp_path / "upload.csv"
    source.write_text("offer_id,description\n1,test\n", encoding="utf-8")
    candidates = []
    decisions = []
    for number in range(5):
        offer_id = f"offer-{number}"
        candidate = {
            "offer_group_id": offer_id,
            "offer_text": f"Chicken product {number}",
            "master_itemcode": f"sku-{number}",
            "master_item_description": f"Master SKU {number}",
            "candidate_rank": 1,
            "calibrated_probability": 0.95 - number * 0.05,
            "embedding_similarity": 0.9 - number * 0.05,
            "agreement_status": (
                "DISAGREEMENT" if number == 2 else "SAFE_AGREEMENT"
            ),
            "lightgbm_top_candidate": f"sku-{number}",
            "llm_parsed_decision": (
                "UNCERTAIN" if number == 3 else ""
            ),
            "llm_confidence": 0.5 if number == 3 else None,
            "model_package_sha256": "abc123",
            "embedding_model_id": "embedding:model",
            "llm_model_id": "ollama:model",
            "pack_conflict": number == 4,
        }
        candidate.update(
            {feature: float(number) for feature in MODEL_FEATURE_COLUMNS}
        )
        candidates.append(candidate)
        decisions.append(
            {
                "offer_id": offer_id,
                "matched_master_sku": (
                    f"sku-{number}" if number == 0 else ""
                ),
                "final_decision": (
                    "AUTO_ACCEPT"
                    if number == 0
                    else "MANUAL_REVIEW"
                ),
                "decision_source": (
                    "STRUCTURED_LLM_REVIEW"
                    if number == 3
                    else "AGREEMENT_POLICY"
                ),
            }
        )
    result = SimpleNamespace(
        run_id="unified-test-run",
        status="COMPLETED_ASSISTED",
        candidates=pd.DataFrame(candidates),
        decisions=pd.DataFrame(decisions),
        statistics={
            "offers_processed": 5,
            "stage_runtimes_seconds": {"lightgbm_scoring": 0.1},
            "training_or_online_learning_performed": False,
        },
        output_paths={},
        error=None,
    )

    session_id = observe_unified_result(
        result, config=config, source_path=source
    )

    assert session_id is not None
    store = LearningStore(config.learning_store.database_path)
    summary = store.summary()
    assert summary["counts"]["pipeline_runs"] == 1
    assert summary["counts"]["predictions"] == 5
    assert summary["counts"]["human_reviews"] == 5
    assert summary["counts"]["automated_labels"] == 5
    assert summary["retraining_performed"] is False
    connection = sqlite3.connect(store.path)
    try:
        feature_snapshot = connection.execute(
            "SELECT feature_snapshot_json FROM predictions LIMIT 1"
        ).fetchone()[0]
        pseudo_eligibility = connection.execute(
            """
            SELECT eligibility_status FROM automated_labels
            WHERE label_quality = 'PSEUDO'
            """
        ).fetchone()[0]
        source_hash = connection.execute(
            "SELECT source_file_hash FROM pipeline_runs"
        ).fetchone()[0]
    finally:
        connection.close()
    parsed_features = json.loads(feature_snapshot)
    assert list(parsed_features) == sorted(MODEL_FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in parsed_features.values())
    assert pseudo_eligibility == "NOT_TRAINING_ELIGIBLE"
    assert len(source_hash) == 64
