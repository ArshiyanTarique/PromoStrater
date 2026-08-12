"""Explicit offline retraining and champion–challenger governance."""

from sku_mapping.retraining.comparison import (
    ComparisonResult,
    PromotionPolicy,
    compare_models,
)
from sku_mapping.retraining.registry import ControlledModelRegistry
from sku_mapping.retraining.snapshot import (
    InsufficientGoldLabelsError,
    SnapshotBuildResult,
    build_training_snapshot,
    load_training_snapshot,
)
from sku_mapping.retraining.trainer import (
    ChallengerTrainingResult,
    train_challenger,
)

__all__ = [
    "ChallengerTrainingResult",
    "ComparisonResult",
    "ControlledModelRegistry",
    "InsufficientGoldLabelsError",
    "PromotionPolicy",
    "SnapshotBuildResult",
    "build_training_snapshot",
    "compare_models",
    "load_training_snapshot",
    "train_challenger",
]
