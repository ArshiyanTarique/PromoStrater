"""Tests for typed YAML configuration."""

from pathlib import Path

import pytest

from sku_mapping.config import (
    ConfigurationError,
    LLMReviewConfig,
    MatchingConfig,
    RetrainingConfig,
    RuntimeConfig,
    load_config,
)
from sku_mapping.constants import MLDeploymentMode, ReviewRoute


def test_default_config_loads_with_repository_relative_paths() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.data.flyer_path.name == "Alkabeer_Export_Data_Clickflyer.csv"
    assert config.data.gold_pairs_path.name == "GOLD_TRAINING_PAIRS_v5_FINAL.csv"
    assert config.model.model_path.name == "matcher_ranked_v5_calibrated.joblib"
    assert config.runtime.random_seed == 42
    assert config.runtime.output_encoding == "utf-8-sig"
    assert config.training.train_fraction == 0.70
    assert config.training.calibration_method == "auto"
    assert config.training.provenance_weights["synthetic"] == 0.60
    assert config.training.target_auto_precision == 0.995
    assert config.retraining.minimum_new_gold_labels == 50
    assert config.retraining.recent_gold_holdout_count == 20
    assert config.retraining.include_silver is False
    assert config.retraining.gold_weight == 1.0
    assert config.retraining.silver_weight == 0.25
    assert config.retraining.pseudo_weight == 0.0
    assert config.retraining.operational_threshold == 0.85
    assert config.shadow_mode.enabled is False
    assert config.shadow_mode.model_id is None
    assert config.shadow_mode.require_package_status == "SHADOW_MODE_ONLY"
    assert config.shadow_mode.retain_all_candidates is True
    assert config.shadow_mode.top_k == 5
    assert config.ml.mode is MLDeploymentMode.ASSISTED
    # Timestamped per training run; assert the family, not the exact id.
    assert str(config.ml.model_id).startswith("ranked-v5-cal-")
    assert config.ml.auto_accept_threshold == 0.85
    assert config.ml.require_registered_model is True
    assert config.ml.apply_safety_overrides is True
    assert config.ml.continue_shadow_monitoring is True
    assert config.agreement.require_same_top_candidate is True
    assert config.agreement.lightgbm_auto_accept_threshold == 0.95
    assert config.agreement.disagreement_route is ReviewRoute.LLM_REVIEW
    assert config.agreement.weak_agreement_route is ReviewRoute.LLM_REVIEW
    assert config.agreement.hard_conflict_route is ReviewRoute.MANUAL_REVIEW
    assert config.llm_review.enabled is False
    assert config.llm_review.provider == "ollama"
    assert config.llm_review.model == "llama3.1:8b"
    assert config.llm_review.endpoint == "http://localhost:11434"
    assert config.llm_review.timeout_seconds == 60
    assert config.llm_review.maximum_candidates == 5
    assert config.llm_review.temperature == 0
    assert config.llm_review.minimum_accept_confidence == 0.85
    assert config.llm_review.maximum_retries == 1
    assert config.llm_review.fail_route == "MANUAL_REVIEW"
    assert config.llm_review.reject_all_route == "MANUAL_REVIEW"
    assert config.llm_review.cache_responses is True
    assert config.learning_store.enabled is True
    assert config.learning_store.database_path.name == "sku_learning.db"
    assert config.learning_store.questions_per_run == 5
    assert config.dashboard.max_upload_size_mb == 100
    assert config.dashboard.allowed_extensions == (
        ".csv",
        ".xlsx",
        ".xls",
    )


def test_configured_database_path_stays_absolute_after_cwd_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repository_root)
    config = load_config(Path("config/default.yaml"))
    configured_database = config.learning_store.database_path

    assert configured_database.is_absolute()
    assert configured_database == (
        repository_root / "data" / "learning" / "sku_learning.db"
    ).resolve()

    monkeypatch.chdir(tmp_path)
    assert config.learning_store.database_path == configured_database


@pytest.mark.parametrize(
    ("manual", "auto"),
    [(0.7, 0.7), (0.8, 0.7), (-0.1, 0.9), (0.2, 1.1)],
)
def test_invalid_review_thresholds_are_rejected(manual: float, auto: float) -> None:
    with pytest.raises(ConfigurationError, match="manual_review_threshold"):
        MatchingConfig(
            category_other_min_score=85,
            category_other_min_margin=12,
            normal_min_score=55,
            pack_compatible_bonus=4,
            pack_unknown_penalty=-3,
            high_score_threshold=80,
            high_margin_threshold=8,
            medium_score_threshold=65,
            medium_margin_threshold=6,
            auto_match_threshold=auto,
            manual_review_threshold=manual,
        )


def test_missing_configuration_path_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_missing_config_value_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data: {}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="model"):
        load_config(path)


def test_runtime_configuration_rejects_invalid_log_level_and_encoding() -> None:
    with pytest.raises(ConfigurationError, match="log_level"):
        RuntimeConfig(random_seed=42, log_level="LOUD", output_encoding="utf-8")
    with pytest.raises(ConfigurationError, match="output_encoding"):
        RuntimeConfig(random_seed=42, log_level="INFO", output_encoding="not-an-encoding")


def test_llm_review_configuration_rejects_unsafe_routes(
    tmp_path: Path,
) -> None:
    base = load_config("config/default.yaml").llm_review
    with pytest.raises(ConfigurationError, match="fail_route"):
        LLMReviewConfig(
            **{
                **base.__dict__,
                "fail_route": "NO_MATCH",
                "cache_path": tmp_path / "cache.sqlite3",
            }
        )
    with pytest.raises(ConfigurationError, match="maximum_candidates"):
        LLMReviewConfig(
            **{
                **base.__dict__,
                "maximum_candidates": 0,
                "cache_path": tmp_path / "cache.sqlite3",
            }
        )


def test_retraining_configuration_rejects_unsafe_label_weights() -> None:
    base = load_config("config/default.yaml").retraining
    with pytest.raises(ConfigurationError, match="lower than GOLD"):
        RetrainingConfig(**{**base.__dict__, "silver_weight": 1.0})
    with pytest.raises(ConfigurationError, match="pseudo_weight"):
        RetrainingConfig(**{**base.__dict__, "pseudo_weight": 0.1})


def test_empty_path_and_fractional_integer_settings_are_rejected(tmp_path: Path) -> None:
    default = Path("config/default.yaml").read_text(encoding="utf-8")
    empty_path = tmp_path / "empty-path.yaml"
    empty_path.write_text(
        default.replace("flyer_path: ../Alkabeer_Export_Data_Clickflyer.csv", "flyer_path: ''"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="paths"):
        load_config(empty_path)

    fractional_seed = tmp_path / "fractional-seed.yaml"
    fractional_seed.write_text(
        default.replace("random_seed: 42", "random_seed: 1.5"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="random_seed"):
        load_config(fractional_seed)
