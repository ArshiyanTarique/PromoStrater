"""Typed, import-safe configuration for the SKU-mapping project."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from sku_mapping.constants import MLDeploymentMode, ReviewRoute


class ConfigurationError(ValueError):
    """Raised when a configuration file is malformed or unsafe."""


#: Bounds for streaming inference chunks. Below the minimum the per-chunk
#: fixed costs dominate; above the maximum the peak footprint stops fitting
#: comfortably alongside the rest of a run on a 16 GB machine.
MINIMUM_INFERENCE_CHUNK_SIZE = 10_000
MAXIMUM_INFERENCE_CHUNK_SIZE = 25_000


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
    #: Competitors kept per master SKU. ``0`` means keep every competitor that
    #: clears the score floors, which is the setting the dashboard expects: the
    #: detail lists scroll, so a long list costs nothing to display.
    max_per_target: int
    #: Order surviving candidates with the registered own-brand model instead
    #: of the fuzzy score. Ranking only - the model can never admit, reject, or
    #: override a conflict, and its score is never read as a probability.
    #: Defaults to off so the pre-ML ordering stays reachable by configuration.
    ml_reranking_enabled: bool = False
    #: Candidates per offer shortlist. ``0`` adopts the model package's own
    #: ``retrieval_k``, which is the group size it was trained against.
    ml_shortlist_top_k: int = 0
    #: Drop a competitor's own brand tokens before featurisation. Their brand
    #: can never appear in Al Kabeer master text, so leaving it in penalises a
    #: rival for naming itself.
    brand_stripping_enabled: bool = True
    #: Competitors staged per master SKU for human review. ``0`` stages
    #: nothing, which is the default: a run that writes to the learning store
    #: should be something the operator turned on. Set it to a small number to
    #: start accumulating the competitor ground truth that neither the rules
    #: nor the borrowed model have ever been measured against.
    review_staging_per_target: int = 0
    #: Decide every competitor relationship automatically: the rules and the
    #: ranker settle the clear cases, the adjudicator settles the rest, and
    #: anything unresolved rejects. Off by default so the rules-only behaviour
    #: stays reachable by configuration alone.
    automatic_decisions_enabled: bool = False
    #: PROVISIONAL. Raw-margin floor the top candidate must clear. This is an
    #: uncalibrated within-shortlist margin, NOT a probability, and it is
    #: unrelated to agreement.lightgbm_auto_accept_threshold - that number
    #: answers a different question about a different population.
    clear_margin_threshold: float = 0.0
    #: PROVISIONAL. Margin the top candidate must lead the runner-up by. The
    #: shipped 2.0 is the median measured gap; a quarter of offers measure 0.0,
    #: and those are exactly the ones worth adjudicating.
    clear_gap_threshold: float = 2.0
    #: Send ambiguous offers to the configured LLM provider. Off means every
    #: ambiguous offer rejects instead.
    llm_adjudication_enabled: bool = False
    #: Hard ceiling on candidates sent in one adjudication call.
    llm_max_candidates: int = 5
    llm_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_per_target < 0:
            raise ConfigurationError(
                "competitors.max_per_target must be 0 (no limit) or a positive "
                "integer"
            )
        if self.ml_shortlist_top_k < 0:
            raise ConfigurationError(
                "competitors.ml_shortlist_top_k must be 0 (use the model's "
                "own retrieval_k) or a positive integer"
            )
        if self.review_staging_per_target < 0:
            raise ConfigurationError(
                "competitors.review_staging_per_target must be 0 (stage "
                "nothing) or a positive integer"
            )
        if self.llm_max_candidates < 1:
            raise ConfigurationError(
                "competitors.llm_max_candidates must be at least 1; the "
                "adjudicator cannot choose from an empty shortlist"
            )
        if self.llm_timeout_seconds <= 0:
            raise ConfigurationError(
                "competitors.llm_timeout_seconds must be greater than 0"
            )
        if self.clear_gap_threshold < 0:
            raise ConfigurationError(
                "competitors.clear_gap_threshold must be 0 or greater; a "
                "negative gap would accept a candidate the model ranked second"
            )

    @property
    def is_unlimited(self) -> bool:
        return self.max_per_target == 0


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
class TrainingPolicyConfig:
    """Pre-registered leakage, calibration, threshold, and provenance policy."""

    train_fraction: float
    validation_fraction: float
    calibration_fraction: float
    split_search_candidates: int
    calibration_method: str
    isotonic_min_rows: int
    isotonic_min_positive_rows: int
    target_auto_precision: float
    min_auto_match_rows: int
    max_auto_match_false_positives: int
    min_auto_precision_lower_bound: float
    precision_confidence_level: float
    min_calibration_rows: int
    min_calibration_positive_rows: int
    auto_threshold_min: float
    auto_threshold_max: float
    auto_threshold_steps: int
    manual_threshold_min: float
    manual_threshold_max: float
    manual_threshold_steps: int
    provenance_weights: Mapping[str, float]
    python_major_minor: str
    lightgbm_major_minor: str
    sklearn_major_minor: str

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.calibration_fraction,
        )
        if any(not 0 < value < 1 for value in fractions) or abs(sum(fractions) - 1) > 1e-9:
            raise ConfigurationError("training split fractions must be positive and sum to 1")
        if self.calibration_method not in {"auto", "sigmoid", "isotonic"}:
            raise ConfigurationError(
                "training.calibration_method must be auto, sigmoid, or isotonic"
            )
        for name, value in (
            ("target_auto_precision", self.target_auto_precision),
            ("min_auto_precision_lower_bound", self.min_auto_precision_lower_bound),
            ("precision_confidence_level", self.precision_confidence_level),
            ("auto_threshold_min", self.auto_threshold_min),
            ("auto_threshold_max", self.auto_threshold_max),
            ("manual_threshold_min", self.manual_threshold_min),
            ("manual_threshold_max", self.manual_threshold_max),
        ):
            if not 0 <= value <= 1:
                raise ConfigurationError(f"training.{name} must be within [0, 1]")
        if self.auto_threshold_min >= self.auto_threshold_max:
            raise ConfigurationError("training AUTO threshold range is invalid")
        if self.manual_threshold_min >= self.manual_threshold_max:
            raise ConfigurationError("training manual threshold range is invalid")
        integer_settings = {
            "split_search_candidates": self.split_search_candidates,
            "isotonic_min_rows": self.isotonic_min_rows,
            "isotonic_min_positive_rows": self.isotonic_min_positive_rows,
            "min_auto_match_rows": self.min_auto_match_rows,
            "min_calibration_rows": self.min_calibration_rows,
            "min_calibration_positive_rows": self.min_calibration_positive_rows,
            "auto_threshold_steps": self.auto_threshold_steps,
            "manual_threshold_steps": self.manual_threshold_steps,
        }
        if any(value < 1 for value in integer_settings.values()):
            raise ConfigurationError("training count and grid settings must be positive")
        if self.max_auto_match_false_positives < 0:
            raise ConfigurationError(
                "training.max_auto_match_false_positives cannot be negative"
            )
        required_weights = {
            "human_audited",
            "human_audited_contradiction",
            "clickflyer_autolabel",
            "clickflyer_autolabel_forced",
            "synthetic",
            "rule_generated",
            "forced_or_contradiction",
            "unknown",
        }
        missing_weights = sorted(required_weights - set(self.provenance_weights))
        if missing_weights:
            raise ConfigurationError(
                f"training.provenance_weights is missing: {missing_weights}"
            )
        if any(float(value) <= 0 for value in self.provenance_weights.values()):
            raise ConfigurationError("training provenance weights must be positive")
        for name, value in (
            ("python_major_minor", self.python_major_minor),
            ("lightgbm_major_minor", self.lightgbm_major_minor),
            ("sklearn_major_minor", self.sklearn_major_minor),
        ):
            if not re.fullmatch(r"\d+\.\d+", value):
                raise ConfigurationError(
                    f"training.compatibility.{name} must use major.minor format"
                )


@dataclass(frozen=True)
class RetrainingConfig:
    """Offline retraining, evidence, and promotion policy."""

    minimum_new_gold_labels: int
    recent_gold_holdout_count: int
    include_silver: bool
    baseline_weight_multiplier: float
    gold_weight: float
    silver_weight: float
    pseudo_weight: float
    operational_threshold: float
    precision_tolerance: float
    calibration_brier_tolerance: float
    calibration_ece_tolerance: float
    subgroup_precision_tolerance: float
    coverage_tolerance: float
    minimum_subgroup_rows: int
    minimum_evaluation_rows: int
    snapshot_directory: Path
    challenger_directory: Path
    comparison_report_directory: Path

    def __post_init__(self) -> None:
        if self.minimum_new_gold_labels < 1:
            raise ConfigurationError(
                "retraining.minimum_new_gold_labels must be positive"
            )
        if self.recent_gold_holdout_count < 1:
            raise ConfigurationError(
                "retraining.recent_gold_holdout_count must be positive"
            )
        if self.recent_gold_holdout_count >= self.minimum_new_gold_labels:
            raise ConfigurationError(
                "retraining.recent_gold_holdout_count must be below "
                "minimum_new_gold_labels"
            )
        weights = {
            "baseline_weight_multiplier": self.baseline_weight_multiplier,
            "gold_weight": self.gold_weight,
            "silver_weight": self.silver_weight,
            "pseudo_weight": self.pseudo_weight,
        }
        if any(value < 0 for value in weights.values()):
            raise ConfigurationError("retraining label weights cannot be negative")
        if self.baseline_weight_multiplier <= 0 or self.gold_weight <= 0:
            raise ConfigurationError(
                "retraining baseline and GOLD weights must be positive"
            )
        if not 0 <= self.silver_weight < self.gold_weight:
            raise ConfigurationError(
                "retraining.silver_weight must be lower than GOLD weight"
            )
        if self.pseudo_weight != 0:
            raise ConfigurationError(
                "retraining.pseudo_weight must remain 0 by default policy"
            )
        unit_interval = {
            "operational_threshold": self.operational_threshold,
            "precision_tolerance": self.precision_tolerance,
            "calibration_brier_tolerance": self.calibration_brier_tolerance,
            "calibration_ece_tolerance": self.calibration_ece_tolerance,
            "subgroup_precision_tolerance": self.subgroup_precision_tolerance,
            "coverage_tolerance": self.coverage_tolerance,
        }
        if any(not 0 <= value <= 1 for value in unit_interval.values()):
            raise ConfigurationError(
                "retraining thresholds and tolerances must be within [0, 1]"
            )
        if self.minimum_subgroup_rows < 1 or self.minimum_evaluation_rows < 1:
            raise ConfigurationError(
                "retraining evaluation row minimums must be positive"
            )


@dataclass(frozen=True)
class ShadowModeConfig:
    """Strictly observational shadow-inference and review settings."""

    enabled: bool
    model_id: str | None
    package_reference: Path | None
    registry_path: Path
    output_directory: Path
    review_staging_directory: Path
    challenge_set_directory: Path
    retain_all_candidates: bool
    require_package_status: str
    top_k: int
    sampling_counts: Mapping[str, int]
    #: Own-brand offers processed per streaming chunk. ``0`` keeps the legacy
    #: all-at-once execution. This changes execution only: chunked and legacy
    #: runs produce the same decisions, features, and scores.
    chunk_size: int = 0

    def __post_init__(self) -> None:
        if self.require_package_status != "SHADOW_MODE_ONLY":
            raise ConfigurationError(
                "shadow_mode.require_package_status must be SHADOW_MODE_ONLY"
            )
        if self.top_k < 1:
            raise ConfigurationError("shadow_mode.top_k must be at least 1")
        if self.chunk_size != 0 and not (
            MINIMUM_INFERENCE_CHUNK_SIZE
            <= self.chunk_size
            <= MAXIMUM_INFERENCE_CHUNK_SIZE
        ):
            raise ConfigurationError(
                "shadow_mode.chunk_size must be 0 to disable chunking, or "
                f"between {MINIMUM_INFERENCE_CHUNK_SIZE:,} and "
                f"{MAXIMUM_INFERENCE_CHUNK_SIZE:,}"
            )
        if self.enabled and not self.model_id and self.package_reference is None:
            raise ConfigurationError(
                "Enabled shadow mode requires an explicit model_id or "
                "immutable package_reference"
            )
        if self.model_id and self.package_reference is not None:
            raise ConfigurationError(
                "Configure only one of shadow_mode.model_id or package_reference"
            )
        if any(int(value) < 0 for value in self.sampling_counts.values()):
            raise ConfigurationError(
                "shadow_mode.sampling_counts values cannot be negative"
            )


@dataclass(frozen=True)
class MLDeploymentConfig:
    """Explicit deployment policy for disabled, shadow, and assisted modes."""

    mode: MLDeploymentMode
    model_id: str | None
    auto_accept_threshold: float
    require_registered_model: bool
    apply_safety_overrides: bool
    continue_shadow_monitoring: bool

    def __post_init__(self) -> None:
        if not 0 <= self.auto_accept_threshold <= 1:
            raise ConfigurationError(
                "ml.auto_accept_threshold must be within [0, 1]"
            )
        if self.mode is not MLDeploymentMode.DISABLED and not self.model_id:
            raise ConfigurationError(
                f"ml.model_id is required when ml.mode is {self.mode.value!r}"
            )
        if self.mode is MLDeploymentMode.ASSISTED and not self.require_registered_model:
            raise ConfigurationError(
                "Assisted mode requires a registered model package"
            )


@dataclass(frozen=True)
class AgreementConfig:
    """Conservative candidate-ranker agreement and routing policy."""

    require_same_top_candidate: bool
    lightgbm_auto_accept_threshold: float
    disagreement_route: ReviewRoute
    weak_agreement_route: ReviewRoute
    hard_conflict_route: ReviewRoute

    def __post_init__(self) -> None:
        if not 0 <= self.lightgbm_auto_accept_threshold <= 1:
            raise ConfigurationError(
                "agreement.lightgbm_auto_accept_threshold must be within [0, 1]"
            )
        if self.disagreement_route is ReviewRoute.AUTO_ACCEPT:
            raise ConfigurationError(
                "agreement.disagreement_route cannot be AUTO_ACCEPT"
            )
        if self.weak_agreement_route is ReviewRoute.AUTO_ACCEPT:
            raise ConfigurationError(
                "agreement.weak_agreement_route cannot be AUTO_ACCEPT"
            )
        if self.hard_conflict_route in {
            ReviewRoute.AUTO_ACCEPT,
            ReviewRoute.LLM_REVIEW,
        }:
            raise ConfigurationError(
                "agreement.hard_conflict_route must be manual_review "
                "or safe_fallback"
            )


@dataclass(frozen=True)
class LLMReviewConfig:
    """Bounded structured second-stage reviewer configuration."""

    enabled: bool
    provider: str
    model: str
    endpoint: str
    timeout_seconds: float
    maximum_candidates: int
    temperature: float
    minimum_accept_confidence: float
    maximum_retries: int
    fail_route: str
    reject_all_route: str
    cache_responses: bool
    cache_path: Path
    #: Auto-accept model-score cut-offs for the two modes of the global toggle.
    #: MODEL SCORES, not probabilities of correctness: 0.95 does not mean "95%
    #: accurate". With a reviewer behind it the cut can afford to be strict;
    #: with no reviewer the residue is a human queue, so it relaxes.
    on_auto_accept_threshold: float = 0.95
    off_auto_accept_threshold: float = 0.85

    def __post_init__(self) -> None:
        for name in ("on_auto_accept_threshold", "off_auto_accept_threshold"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ConfigurationError(
                    f"llm_review.{name} must be within (0, 1]"
                )
        if self.off_auto_accept_threshold > self.on_auto_accept_threshold:
            raise ConfigurationError(
                "llm_review.off_auto_accept_threshold must not exceed "
                "on_auto_accept_threshold: turning the reviewer off should "
                "automate more, not less"
            )
        if not self.provider.strip():
            raise ConfigurationError("llm_review.provider must be non-empty")
        if self.enabled and not self.model.strip():
            raise ConfigurationError(
                "Enabled LLM review requires llm_review.model"
            )
        if not self.endpoint.strip():
            raise ConfigurationError("llm_review.endpoint must be non-empty")
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                "llm_review.timeout_seconds must be greater than zero"
            )
        if not 1 <= self.maximum_candidates <= 20:
            raise ConfigurationError(
                "llm_review.maximum_candidates must be within [1, 20]"
            )
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError(
                "llm_review.temperature must be within [0, 2]"
            )
        if not 0 <= self.minimum_accept_confidence <= 1:
            raise ConfigurationError(
                "llm_review.minimum_accept_confidence must be within [0, 1]"
            )
        if self.maximum_retries < 0:
            raise ConfigurationError(
                "llm_review.maximum_retries must be non-negative"
            )
        if self.fail_route != "MANUAL_REVIEW":
            raise ConfigurationError(
                "llm_review.fail_route must be manual_review"
            )
        if self.reject_all_route not in {"MANUAL_REVIEW", "NO_MATCH"}:
            raise ConfigurationError(
                "llm_review.reject_all_route must be manual_review "
                "or no_match"
            )


@dataclass(frozen=True)
class LearningStoreConfig:
    """Persistent inference observation and human-review storage."""

    enabled: bool
    database_path: Path
    csv_export_directory: Path
    questions_per_run: int

    def __post_init__(self) -> None:
        if self.questions_per_run != 5:
            raise ConfigurationError(
                "learning_store.questions_per_run must be exactly 5"
            )


@dataclass(frozen=True)
class DashboardConfig:
    """Controlled upload and output boundaries for the local dashboard."""

    input_directory: Path
    output_directory: Path
    max_upload_size_mb: int
    allowed_extensions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.max_upload_size_mb <= 1024:
            raise ConfigurationError(
                "dashboard.max_upload_size_mb must be within [1, 1024]"
            )
        allowed = {".csv", ".xlsx", ".xls"}
        if (
            not self.allowed_extensions
            or not set(self.allowed_extensions).issubset(allowed)
        ):
            raise ConfigurationError(
                "dashboard.allowed_extensions must use csv/xlsx/xls"
            )


@dataclass(frozen=True)
class PipelineConfig:
    """Complete Phase 2 configuration boundary."""

    data: DataPaths
    model: ModelPaths
    output: OutputPaths
    matching: MatchingConfig
    competitors: CompetitorConfig
    runtime: RuntimeConfig
    training: TrainingPolicyConfig
    retraining: RetrainingConfig
    ml: MLDeploymentConfig
    agreement: AgreementConfig
    llm_review: LLMReviewConfig
    learning_store: LearningStoreConfig
    dashboard: DashboardConfig
    shadow_mode: ShadowModeConfig


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
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


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


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigurationError(f"{name} must be a boolean")


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return _resolve_path(value, base_dir)


def _optional_float(value: Any, name: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    return _as_float(value, name)


def _review_route(value: Any, name: str) -> ReviewRoute:
    try:
        return ReviewRoute(str(value).strip().upper())
    except ValueError as error:
        allowed = ", ".join(route.value.lower() for route in ReviewRoute)
        raise ConfigurationError(f"{name} must be one of: {allowed}") from error


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file without accessing its configured inputs."""
    config_path = Path(path).expanduser().resolve()
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
    training = _required_section(raw, "training")
    retraining = _required_section(raw, "retraining")
    provenance_weights = _required_section(training, "provenance_weights")
    compatibility = _required_section(training, "compatibility")
    base_dir = config_path.parent
    shadow = raw.get("shadow_mode", {})
    if not isinstance(shadow, Mapping):
        raise ConfigurationError("Invalid 'shadow_mode' configuration section")
    sampling_counts = shadow.get("sampling_counts", {})
    if not isinstance(sampling_counts, Mapping):
        raise ConfigurationError("shadow_mode.sampling_counts must be a mapping")
    ml = raw.get("ml", {})
    if not isinstance(ml, Mapping):
        raise ConfigurationError("Invalid 'ml' configuration section")
    try:
        ml_mode = MLDeploymentMode(str(ml.get("mode", "disabled")).strip().lower())
    except ValueError as error:
        raise ConfigurationError(
            "ml.mode must be one of: disabled, shadow, assisted"
        ) from error
    agreement = raw.get("agreement", {})
    if not isinstance(agreement, Mapping):
        raise ConfigurationError("Invalid 'agreement' configuration section")
    llm_review = raw.get("llm_review", {})
    if not isinstance(llm_review, Mapping):
        raise ConfigurationError("Invalid 'llm_review' configuration section")
    learning_store = raw.get("learning_store", {})
    if not isinstance(learning_store, Mapping):
        raise ConfigurationError(
            "Invalid 'learning_store' configuration section"
        )
    dashboard = raw.get("dashboard", {})
    if not isinstance(dashboard, Mapping):
        raise ConfigurationError("Invalid 'dashboard' configuration section")

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
            ml_reranking_enabled=bool(
                competitors.get("ml_reranking_enabled", False)
            ),
            ml_shortlist_top_k=_as_int(
                competitors.get("ml_shortlist_top_k", 0),
                "competitors.ml_shortlist_top_k",
            ),
            brand_stripping_enabled=bool(
                competitors.get("brand_stripping_enabled", True)
            ),
            review_staging_per_target=_as_int(
                competitors.get("review_staging_per_target", 0),
                "competitors.review_staging_per_target",
            ),
            automatic_decisions_enabled=bool(
                competitors.get("automatic_decisions_enabled", False)
            ),
            clear_margin_threshold=_as_float(
                competitors.get("clear_margin_threshold", 0.0),
                "competitors.clear_margin_threshold",
            ),
            clear_gap_threshold=_as_float(
                competitors.get("clear_gap_threshold", 2.0),
                "competitors.clear_gap_threshold",
            ),
            llm_adjudication_enabled=bool(
                competitors.get("llm_adjudication_enabled", False)
            ),
            llm_max_candidates=_as_int(
                competitors.get("llm_max_candidates", 5),
                "competitors.llm_max_candidates",
            ),
            llm_timeout_seconds=_as_float(
                competitors.get("llm_timeout_seconds", 60.0),
                "competitors.llm_timeout_seconds",
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
        training=TrainingPolicyConfig(
            train_fraction=_as_float(
                _required_value(training, "training", "train_fraction"),
                "training.train_fraction",
            ),
            validation_fraction=_as_float(
                _required_value(training, "training", "validation_fraction"),
                "training.validation_fraction",
            ),
            calibration_fraction=_as_float(
                _required_value(training, "training", "calibration_fraction"),
                "training.calibration_fraction",
            ),
            split_search_candidates=_as_int(
                _required_value(training, "training", "split_search_candidates"),
                "training.split_search_candidates",
            ),
            calibration_method=str(
                _required_value(training, "training", "calibration_method")
            ),
            isotonic_min_rows=_as_int(
                _required_value(training, "training", "isotonic_min_rows"),
                "training.isotonic_min_rows",
            ),
            isotonic_min_positive_rows=_as_int(
                _required_value(
                    training, "training", "isotonic_min_positive_rows"
                ),
                "training.isotonic_min_positive_rows",
            ),
            target_auto_precision=_as_float(
                _required_value(training, "training", "target_auto_precision"),
                "training.target_auto_precision",
            ),
            min_auto_match_rows=_as_int(
                _required_value(training, "training", "min_auto_match_rows"),
                "training.min_auto_match_rows",
            ),
            max_auto_match_false_positives=_as_int(
                _required_value(
                    training, "training", "max_auto_match_false_positives"
                ),
                "training.max_auto_match_false_positives",
            ),
            min_auto_precision_lower_bound=_as_float(
                _required_value(
                    training, "training", "min_auto_precision_lower_bound"
                ),
                "training.min_auto_precision_lower_bound",
            ),
            precision_confidence_level=_as_float(
                _required_value(
                    training, "training", "precision_confidence_level"
                ),
                "training.precision_confidence_level",
            ),
            min_calibration_rows=_as_int(
                _required_value(training, "training", "min_calibration_rows"),
                "training.min_calibration_rows",
            ),
            min_calibration_positive_rows=_as_int(
                _required_value(
                    training, "training", "min_calibration_positive_rows"
                ),
                "training.min_calibration_positive_rows",
            ),
            auto_threshold_min=_as_float(
                _required_value(training, "training", "auto_threshold_min"),
                "training.auto_threshold_min",
            ),
            auto_threshold_max=_as_float(
                _required_value(training, "training", "auto_threshold_max"),
                "training.auto_threshold_max",
            ),
            auto_threshold_steps=_as_int(
                _required_value(training, "training", "auto_threshold_steps"),
                "training.auto_threshold_steps",
            ),
            manual_threshold_min=_as_float(
                _required_value(training, "training", "manual_threshold_min"),
                "training.manual_threshold_min",
            ),
            manual_threshold_max=_as_float(
                _required_value(training, "training", "manual_threshold_max"),
                "training.manual_threshold_max",
            ),
            manual_threshold_steps=_as_int(
                _required_value(training, "training", "manual_threshold_steps"),
                "training.manual_threshold_steps",
            ),
            provenance_weights={
                str(key): _as_float(value, f"training.provenance_weights.{key}")
                for key, value in provenance_weights.items()
            },
            python_major_minor=str(
                _required_value(
                    compatibility, "training.compatibility", "python_major_minor"
                )
            ),
            lightgbm_major_minor=str(
                _required_value(
                    compatibility,
                    "training.compatibility",
                    "lightgbm_major_minor",
                )
            ),
            sklearn_major_minor=str(
                _required_value(
                    compatibility,
                    "training.compatibility",
                    "sklearn_major_minor",
                )
            ),
        ),
        retraining=RetrainingConfig(
            minimum_new_gold_labels=_as_int(
                _required_value(
                    retraining, "retraining", "minimum_new_gold_labels"
                ),
                "retraining.minimum_new_gold_labels",
            ),
            recent_gold_holdout_count=_as_int(
                _required_value(
                    retraining, "retraining", "recent_gold_holdout_count"
                ),
                "retraining.recent_gold_holdout_count",
            ),
            include_silver=_as_bool(
                _required_value(retraining, "retraining", "include_silver"),
                "retraining.include_silver",
            ),
            baseline_weight_multiplier=_as_float(
                _required_value(
                    retraining, "retraining", "baseline_weight_multiplier"
                ),
                "retraining.baseline_weight_multiplier",
            ),
            gold_weight=_as_float(
                _required_value(retraining, "retraining", "gold_weight"),
                "retraining.gold_weight",
            ),
            silver_weight=_as_float(
                _required_value(retraining, "retraining", "silver_weight"),
                "retraining.silver_weight",
            ),
            pseudo_weight=_as_float(
                _required_value(retraining, "retraining", "pseudo_weight"),
                "retraining.pseudo_weight",
            ),
            operational_threshold=_as_float(
                _required_value(
                    retraining, "retraining", "operational_threshold"
                ),
                "retraining.operational_threshold",
            ),
            precision_tolerance=_as_float(
                _required_value(
                    retraining, "retraining", "precision_tolerance"
                ),
                "retraining.precision_tolerance",
            ),
            calibration_brier_tolerance=_as_float(
                _required_value(
                    retraining, "retraining", "calibration_brier_tolerance"
                ),
                "retraining.calibration_brier_tolerance",
            ),
            calibration_ece_tolerance=_as_float(
                _required_value(
                    retraining, "retraining", "calibration_ece_tolerance"
                ),
                "retraining.calibration_ece_tolerance",
            ),
            subgroup_precision_tolerance=_as_float(
                _required_value(
                    retraining, "retraining", "subgroup_precision_tolerance"
                ),
                "retraining.subgroup_precision_tolerance",
            ),
            coverage_tolerance=_as_float(
                _required_value(
                    retraining, "retraining", "coverage_tolerance"
                ),
                "retraining.coverage_tolerance",
            ),
            minimum_subgroup_rows=_as_int(
                _required_value(
                    retraining, "retraining", "minimum_subgroup_rows"
                ),
                "retraining.minimum_subgroup_rows",
            ),
            minimum_evaluation_rows=_as_int(
                _required_value(
                    retraining, "retraining", "minimum_evaluation_rows"
                ),
                "retraining.minimum_evaluation_rows",
            ),
            snapshot_directory=_resolve_path(
                _required_value(
                    retraining, "retraining", "snapshot_directory"
                ),
                base_dir,
            ),
            challenger_directory=_resolve_path(
                _required_value(
                    retraining, "retraining", "challenger_directory"
                ),
                base_dir,
            ),
            comparison_report_directory=_resolve_path(
                _required_value(
                    retraining, "retraining", "comparison_report_directory"
                ),
                base_dir,
            ),
        ),
        ml=MLDeploymentConfig(
            mode=ml_mode,
            model_id=(
                str(ml["model_id"]).strip()
                if ml.get("model_id") is not None
                and str(ml["model_id"]).strip()
                else None
            ),
            auto_accept_threshold=_as_float(
                ml.get("auto_accept_threshold", 0.85),
                "ml.auto_accept_threshold",
            ),
            require_registered_model=_as_bool(
                ml.get("require_registered_model", True),
                "ml.require_registered_model",
            ),
            apply_safety_overrides=_as_bool(
                ml.get("apply_safety_overrides", True),
                "ml.apply_safety_overrides",
            ),
            continue_shadow_monitoring=_as_bool(
                ml.get("continue_shadow_monitoring", True),
                "ml.continue_shadow_monitoring",
            ),
        ),
        agreement=AgreementConfig(
            require_same_top_candidate=_as_bool(
                agreement.get("require_same_top_candidate", True),
                "agreement.require_same_top_candidate",
            ),
            lightgbm_auto_accept_threshold=_as_float(
                agreement.get("lightgbm_auto_accept_threshold", 0.85),
                "agreement.lightgbm_auto_accept_threshold",
            ),
            disagreement_route=_review_route(
                agreement.get("disagreement_route", "llm_review"),
                "agreement.disagreement_route",
            ),
            weak_agreement_route=_review_route(
                agreement.get("weak_agreement_route", "llm_review"),
                "agreement.weak_agreement_route",
            ),
            hard_conflict_route=_review_route(
                agreement.get("hard_conflict_route", "manual_review"),
                "agreement.hard_conflict_route",
            ),
        ),
        llm_review=LLMReviewConfig(
            enabled=_as_bool(
                llm_review.get("enabled", False), "llm_review.enabled"
            ),
            provider=str(
                llm_review.get("provider", "ollama")
            ).strip(),
            model=str(
                llm_review.get("model", "llama3.1:8b")
            ).strip(),
            endpoint=str(
                llm_review.get("endpoint", "http://localhost:11434")
            ).strip(),
            timeout_seconds=_as_float(
                llm_review.get("timeout_seconds", 60),
                "llm_review.timeout_seconds",
            ),
            maximum_candidates=_as_int(
                llm_review.get("maximum_candidates", 5),
                "llm_review.maximum_candidates",
            ),
            temperature=_as_float(
                llm_review.get("temperature", 0),
                "llm_review.temperature",
            ),
            minimum_accept_confidence=_as_float(
                llm_review.get("minimum_accept_confidence", 0.85),
                "llm_review.minimum_accept_confidence",
            ),
            maximum_retries=_as_int(
                llm_review.get("maximum_retries", 1),
                "llm_review.maximum_retries",
            ),
            fail_route=str(
                llm_review.get("fail_route", "manual_review")
            ).strip().upper(),
            reject_all_route=str(
                llm_review.get("reject_all_route", "manual_review")
            ).strip().upper(),
            cache_responses=_as_bool(
                llm_review.get("cache_responses", True),
                "llm_review.cache_responses",
            ),
            on_auto_accept_threshold=_as_float(
                llm_review.get("on_auto_accept_threshold", 0.95),
                "llm_review.on_auto_accept_threshold",
            ),
            off_auto_accept_threshold=_as_float(
                llm_review.get("off_auto_accept_threshold", 0.85),
                "llm_review.off_auto_accept_threshold",
            ),
            cache_path=_resolve_path(
                llm_review.get(
                    "cache_path",
                    "../data/processed/llm_review_cache.sqlite3",
                ),
                base_dir,
            ),
        ),
        learning_store=LearningStoreConfig(
            enabled=_as_bool(
                learning_store.get("enabled", True),
                "learning_store.enabled",
            ),
            database_path=_resolve_path(
                learning_store.get(
                    "database_path",
                    "../data/learning/sku_learning.db",
                ),
                base_dir,
            ),
            csv_export_directory=_resolve_path(
                learning_store.get(
                    "csv_export_directory",
                    "../data/learning/exports",
                ),
                base_dir,
            ),
            questions_per_run=_as_int(
                learning_store.get("questions_per_run", 5),
                "learning_store.questions_per_run",
            ),
        ),
        dashboard=DashboardConfig(
            input_directory=_resolve_path(
                dashboard.get(
                    "input_directory", "../data/dashboard_uploads"
                ),
                base_dir,
            ),
            output_directory=_resolve_path(
                dashboard.get(
                    "output_directory", "../outputs/dashboard_runs"
                ),
                base_dir,
            ),
            max_upload_size_mb=_as_int(
                dashboard.get("max_upload_size_mb", 100),
                "dashboard.max_upload_size_mb",
            ),
            allowed_extensions=tuple(
                str(value).lower()
                if str(value).startswith(".")
                else f".{str(value).lower()}"
                for value in dashboard.get(
                    "allowed_extensions", ["csv", "xlsx", "xls"]
                )
            ),
        ),
        shadow_mode=ShadowModeConfig(
            enabled=_as_bool(shadow.get("enabled", False), "shadow_mode.enabled"),
            model_id=(
                str(shadow["model_id"]).strip()
                if shadow.get("model_id") is not None
                and str(shadow["model_id"]).strip()
                else None
            ),
            package_reference=_optional_path(
                shadow.get("package_reference"), base_dir
            ),
            registry_path=_resolve_path(
                shadow.get("registry_path", "../models/model_registry.json"),
                base_dir,
            ),
            output_directory=_resolve_path(
                shadow.get("output_directory", "../outputs/shadow"), base_dir
            ),
            review_staging_directory=_resolve_path(
                shadow.get(
                    "review_staging_directory", "../data/review_staging"
                ),
                base_dir,
            ),
            challenge_set_directory=_resolve_path(
                shadow.get(
                    "challenge_set_directory", "../data/challenge_sets"
                ),
                base_dir,
            ),
            retain_all_candidates=_as_bool(
                shadow.get("retain_all_candidates", True),
                "shadow_mode.retain_all_candidates",
            ),
            require_package_status=str(
                shadow.get("require_package_status", "SHADOW_MODE_ONLY")
            ),
            top_k=_as_int(shadow.get("top_k", 5), "shadow_mode.top_k"),
            sampling_counts={
                str(key): _as_int(
                    value, f"shadow_mode.sampling_counts.{key}"
                )
                for key, value in sampling_counts.items()
            },
            chunk_size=_as_int(
                shadow.get("chunk_size", 0), "shadow_mode.chunk_size"
            ),
        ),
    )
