"""Static Phase 6D boundaries against learning and arbitrary SKU generation."""

from pathlib import Path


def test_llm_reviewer_has_no_training_or_production_write_path() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/sku_mapping/llm_review/reviewer.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = (
        ".fit(",
        "training_features",
        "gold_training",
        "product_master.xlsx",
        "to_parquet(",
        "to_csv(",
        "matched_itemcode",
    )
    assert not any(token in source for token in forbidden)


def test_llm_prompt_and_parser_forbid_arbitrary_candidates() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/sku_mapping/llm_review/reviewer.py"
    ).read_text(encoding="utf-8")
    assert "Never invent or emit another SKU" in source
    assert "SELECTED_CANDIDATE_NOT_SUPPLIED" in source
    assert "maximum_candidates=config.maximum_candidates" in source
