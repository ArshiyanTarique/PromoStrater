"""Typed, import-safe configuration for the SKU-mapping project."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration file is malformed or unsafe."""


@dataclass(frozen=True)
class DataPaths:
    """Input and processed-data locations."""

    flyer_path: Path
    master_path: Path
    gold_pairs_path: Path
    processed_dir: Path


@dataclass(frozen=True)
class ModelPaths:
    """Model registry locations."""

    model_dir: Path
    model_path: Path


@dataclass(frozen=True)
class OutputPaths:
    """Business-output location."""

    output_dir: Path


@dataclass(frozen=True)
class MatchingConfig:
    """Legacy candidate and review thresholds, retained without applying them."""

    category_other_min_score: float
    category_other_min_margin: float
    normal_min_score: float
    pack_compatible_bonus: float
    pack_unknown_penalty: float
    high_score_threshold: float
    high_margin_threshold: float
    medium_score_threshold: float
    medium_margin_threshold: float
    auto_match_threshold: float
    manual_review_threshold: float

    def __post_init__(self) -> None:
        if not 0 <= self.manual_review_threshold < self.auto_match_threshold <= 1:
            raise ConfigurationError(
                "matching thresholds must satisfy "
                "0 <= manual_review_threshold < auto_match_threshold <= 1"
            )


@dataclass(frozen=True)
class CompetitorConfig:
    """Thresholds retained for a later competitor-discovery phase."""

    raw_score_floor: float
    adjusted_score_floor: float
    max_per_target: int

    def __post_init__(self) -> None:
        if self.max_per_target < 1:
            raise ConfigurationError("competitors.max_per_target must be at least 1")


@dataclass(frozen=True)
class RuntimeConfig:
    """Deterministic runtime settings."""

    random_seed: int
    log_level: str
    output_encoding: str

    def __post_init__(self) -> None:
        if self.log_level.upper() not in logging.getLevelNamesMapping():
            raise ConfigurationError(f"runtime.log_level is invalid: {self.log_level!r}")
        try:
            codecs.lookup(self.output_encoding)
        except LookupError as error:
            raise ConfigurationError(
                f"runtime.output_encoding is invalid: {self.output_encoding!r}"
            ) from error


@dataclass(frozen=True)
class PipelineConfig:
    """Complete Phase 2 configuration boundary."""

    data: DataPaths
    model: ModelPaths
    output: OutputPaths
    matching: MatchingConfig
    competitors: CompetitorConfig
    runtime: RuntimeConfig


def _required_section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing or invalid '{name}' configuration section")
    return value


def _required_value(raw: Mapping[str, Any], section: str, name: str) -> Any:
    if name not in raw:
        raise ConfigurationError(f"Missing required configuration value '{section}.{name}'")
    return raw[name]


def _resolve_path(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigurationError("Configured paths must be non-empty strings")
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be numeric") from error


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if not numeric.is_integer():
        raise ConfigurationError(f"{name} must be an integer")
    return int(numeric)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file without accessing its configured inputs."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in configuration file: {config_path}") from error
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a mapping")

    data = _required_section(raw, "data")
    model = _required_section(raw, "model")
    output = _required_section(raw, "output")
    matching = _required_section(raw, "matching")
    competitors = _required_section(raw, "competitors")
    runtime = _required_section(raw, "runtime")
    base_dir = config_path.parent

    return PipelineConfig(
        data=DataPaths(
            flyer_path=_resolve_path(_required_value(data, "data", "flyer_path"), base_dir),
            master_path=_resolve_path(_required_value(data, "data", "master_path"), base_dir),
            gold_pairs_path=_resolve_path(
                _required_value(data, "data", "gold_pairs_path"), base_dir
            ),
            processed_dir=_resolve_path(
                _required_value(data, "data", "processed_dir"), base_dir
            ),
        ),
        model=ModelPaths(
            model_dir=_resolve_path(_required_value(model, "model", "model_dir"), base_dir),
            model_path=_resolve_path(_required_value(model, "model", "model_path"), base_dir),
        ),
        output=OutputPaths(
            output_dir=_resolve_path(_required_value(output, "output", "output_dir"), base_dir)
        ),
        matching=MatchingConfig(
            **{
                key: _as_float(_required_value(matching, "matching", key), f"matching.{key}")
                for key in (
                    "category_other_min_score",
                    "category_other_min_margin",
                    "normal_min_score",
                    "pack_compatible_bonus",
                    "pack_unknown_penalty",
                    "high_score_threshold",
                    "high_margin_threshold",
                    "medium_score_threshold",
                    "medium_margin_threshold",
                    "auto_match_threshold",
                    "manual_review_threshold",
                )
            }
        ),
        competitors=CompetitorConfig(
            raw_score_floor=_as_float(
                _required_value(competitors, "competitors", "raw_score_floor"),
                "competitors.raw_score_floor",
            ),
            adjusted_score_floor=_as_float(
                _required_value(competitors, "competitors", "adjusted_score_floor"),
                "competitors.adjusted_score_floor",
            ),
            max_per_target=_as_int(
                _required_value(competitors, "competitors", "max_per_target"),
                "competitors.max_per_target",
            ),
        ),
        runtime=RuntimeConfig(
            random_seed=_as_int(
                _required_value(runtime, "runtime", "random_seed"),
                "runtime.random_seed",
            ),
            log_level=str(_required_value(runtime, "runtime", "log_level")),
            output_encoding=str(_required_value(runtime, "runtime", "output_encoding")),
        ),
    )
