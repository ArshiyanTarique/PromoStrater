"""Phase 6B must not implement the future agreement decision policy."""

from pathlib import Path


def test_assisted_decisions_do_not_consume_embedding_scores() -> None:
    deployment = (
        Path(__file__).parents[2]
        / "src/sku_mapping/ml/deployment.py"
    ).read_text(encoding="utf-8")
    assert "embedding_similarity" not in deployment
    assert "embedding_top_candidate" not in deployment


def test_embedding_scorer_does_not_generate_candidates() -> None:
    scorer = (
        Path(__file__).parents[2]
        / "src/sku_mapping/embedding/scorer.py"
    ).read_text(encoding="utf-8")
    assert "CandidateGenerator" not in scorer
    assert "generate_candidates" not in scorer
