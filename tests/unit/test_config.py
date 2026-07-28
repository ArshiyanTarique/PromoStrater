"""Tests for typed YAML configuration."""

from pathlib import Path

import pytest

from sku_mapping.config import ConfigurationError, MatchingConfig, RuntimeConfig, load_config


def test_default_config_loads_with_repository_relative_paths() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.data.flyer_path.name == "Alkabeer_Export_Data_Clickflyer.csv"
    assert config.data.gold_pairs_path.name == "GOLD_TRAINING_PAIRS_v5_FINAL.csv"
    assert config.model.model_path.name == "alkabeer_sku_matcher_v1.joblib"
    assert config.runtime.random_seed == 42
    assert config.runtime.output_encoding == "utf-8-sig"


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
