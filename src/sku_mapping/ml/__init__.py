"""Reproducible model training, evaluation, and packaging APIs."""

from sku_mapping.ml.evaluator import evaluate_binary_classifier
from sku_mapping.ml.model_package import (
    REQUIRED_MODEL_PACKAGE_KEYS,
    load_model_package,
    save_model_package,
    validate_model_package,
)
from sku_mapping.ml.threshold_tuning import ThresholdTuningResult, tune_thresholds
from sku_mapping.ml.shadow_trainer import (
    ShadowTrainingConfig,
    ShadowTrainingResult,
    run_shadow_training_pipeline,
)
from sku_mapping.ml.trainer import (
    DatasetSplits,
    ModelTrainingResult,
    TrainingConfig,
    assert_no_group_leakage,
    create_group_splits,
    run_training_pipeline,
)

__all__ = [
    "DatasetSplits",
    "ModelTrainingResult",
    "REQUIRED_MODEL_PACKAGE_KEYS",
    "ThresholdTuningResult",
    "TrainingConfig",
    "ShadowTrainingConfig",
    "ShadowTrainingResult",
    "assert_no_group_leakage",
    "create_group_splits",
    "evaluate_binary_classifier",
    "load_model_package",
    "run_training_pipeline",
    "run_shadow_training_pipeline",
    "save_model_package",
    "tune_thresholds",
    "validate_model_package",
]
