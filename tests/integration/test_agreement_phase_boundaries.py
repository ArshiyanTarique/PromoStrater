"""Phase 6C must route only; it cannot call an LLM or alter learning data."""

from pathlib import Path


def test_agreement_policy_has_no_llm_or_training_dependency() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/sku_mapping/agreement/policy.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "requests.",
        "openai",
        "ollama",
        ".fit(",
        "partial_fit",
        "training_features",
        "review_staging",
    )
    assert not any(token in source.lower() for token in forbidden)


def test_agreement_policy_does_not_generate_candidates() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/sku_mapping/agreement/policy.py"
    ).read_text(encoding="utf-8")
    assert "CandidateGenerator" not in source
    assert "generate_candidates" not in source
