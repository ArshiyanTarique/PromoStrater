"""Independent embedding-based candidate scoring."""

from sku_mapping.embedding.scorer import (
    EmbeddingScoreResult,
    score_candidate_frame,
    score_candidate_frame_non_blocking,
)

__all__ = [
    "EmbeddingScoreResult",
    "score_candidate_frame",
    "score_candidate_frame_non_blocking",
]
