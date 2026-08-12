"""Static integration guards around the authoritative production pipeline."""

from pathlib import Path


def test_assisted_decisions_precede_mapping_gate_and_competitor_discovery() -> None:
    source = (
        Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    ).read_text(encoding="utf-8")
    assisted = source.index("run_unified_inference_non_blocking(")
    mapping_gate = source.index(
        '_auto_ok = own_keys["ml_decision"].eq("AUTO_MATCH")'
    )
    competitors = source.index("STAGE 3: SKU-level competitor discovery")
    assert assisted < mapping_gate < competitors


def test_assisted_monitoring_precedes_competitor_discovery() -> None:
    source = (
        Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    ).read_text(encoding="utf-8")
    monitoring = source.index(
        '_assisted_result, "shadow_result", None'
    )
    competitors = source.index("STAGE 3: SKU-level competitor discovery")
    assert monitoring < competitors


def test_no_training_or_online_learning_is_invoked_by_production_pipeline() -> None:
    source = (
        Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    ).read_text(encoding="utf-8")
    phase_6a = source[source.index("PHASE 6A: explicit assisted deployment") :]
    assert ".fit(" not in phase_6a
    assert "run_training_pipeline" not in phase_6a


def test_competitor_discovery_uses_final_eligibility_when_available() -> None:
    source = (
        Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    ).read_text(encoding="utf-8")
    eligibility = source.index("select_competitor_eligible_rows(")
    competitors = source.index("STAGE 3: SKU-level competitor discovery")
    discovery_loop = source.index("find_competitors_batch(")
    assert competitors < eligibility < discovery_loop


def test_assisted_offer_export_appends_unified_provenance() -> None:
    source = (
        Path(__file__).parents[2] / "sku_mapping_pipeline_ml.py"
    ).read_text(encoding="utf-8")
    required = (
        "FINAL_SKU_MAPPING_DECISIONS_CSV",
        '"final_decision"',
        '"decision_source"',
        '"lightgbm_probability"',
        '"embedding_similarity"',
        '"llm_decision"',
        '"llm_confidence"',
        '"embedding_model_id"',
        '"llm_model_id"',
        '"run_id"',
    )
    assert all(token in source for token in required)
