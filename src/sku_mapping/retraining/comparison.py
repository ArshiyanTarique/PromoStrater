"""Conservative, same-data champion–challenger evaluation policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sku_mapping.config import PipelineConfig
from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.learning.store import LearningStore
from sku_mapping.ml.evaluator import evaluate_binary_classifier
from sku_mapping.ml.model_package import load_model_package, validate_model_package
from sku_mapping.retraining.artifacts import atomic_csv, atomic_json, sha256_file
from sku_mapping.retraining.registry import ControlledModelRegistry
from sku_mapping.retraining.snapshot import load_training_snapshot
from sku_mapping.shadow.challenge import assert_not_sealed_challenge_input

CRITICAL_FEATURES = (
    "protein_match",
    "family_match",
    "size_match",
    "pack_format_match",
)


@dataclass(frozen=True)
class PromotionPolicy:
    operational_threshold: float = 0.85
    precision_tolerance: float = 0.01
    calibration_brier_tolerance: float = 0.01
    calibration_ece_tolerance: float = 0.02
    subgroup_precision_tolerance: float = 0.05
    coverage_tolerance: float = 0.02
    minimum_subgroup_rows: int = 20
    minimum_evaluation_rows: int = 20

    @classmethod
    def from_pipeline_config(cls, config: PipelineConfig) -> "PromotionPolicy":
        policy = config.retraining
        return cls(
            operational_threshold=policy.operational_threshold,
            precision_tolerance=policy.precision_tolerance,
            calibration_brier_tolerance=policy.calibration_brier_tolerance,
            calibration_ece_tolerance=policy.calibration_ece_tolerance,
            subgroup_precision_tolerance=policy.subgroup_precision_tolerance,
            coverage_tolerance=policy.coverage_tolerance,
            minimum_subgroup_rows=policy.minimum_subgroup_rows,
            minimum_evaluation_rows=policy.minimum_evaluation_rows,
        )


@dataclass(frozen=True)
class ComparisonResult:
    comparison_id: str
    promotion_policy_passed: bool
    decision: str
    report_path: Path
    regression_cases_path: Path
    report: dict[str, Any]
    registered_package_path: Path | None


def _probabilities(package: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    validate_model_package(package)
    features = frame.loc[:, MODEL_FEATURE_COLUMNS].copy()
    if list(features.columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Evaluation feature order is incompatible")
    predictor = package.get("predictor")
    if callable(getattr(predictor, "predict_calibrated_proba", None)):
        result = predictor.predict_calibrated_proba(features)
    else:
        result = package["model"].predict_proba(features)[:, 1]
    probabilities = np.asarray(result, dtype=float)
    if (
        probabilities.shape != (len(frame),)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Model produced invalid evaluation probabilities")
    return probabilities


def _load_evaluation_sets(
    snapshot_manifest_path: str | Path,
    additional_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, dict[str, str]]:
    _, _, recent = load_training_snapshot(snapshot_manifest_path)
    recent = recent.copy()
    recent["evaluation_set"] = "recent_human_confirmed"
    frames = [recent]
    hashes = {
        "recent_human_confirmed": sha256_file(
            Path(snapshot_manifest_path).parent
            / "recent_gold_evaluation.parquet"
        )
    }
    for index, raw_path in enumerate(additional_paths, start=1):
        path = Path(raw_path)
        assert_not_sealed_challenge_input(path)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation table not found: {path}")
        frame = pd.read_parquet(path)
        name = f"additional_{index}_{path.stem}"
        frame = frame.copy()
        frame["evaluation_set"] = name
        frames.append(frame)
        hashes[name] = sha256_file(path)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = ["pair_label", *MODEL_FEATURE_COLUMNS]
    missing = [column for column in required if column not in combined]
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {missing}")
    labels = pd.to_numeric(combined["pair_label"], errors="coerce")
    if labels.isna().any() or set(labels.astype(int).unique()) != {0, 1}:
        raise ValueError("Evaluation data must contain both binary classes")
    combined["pair_label"] = labels.astype(int)
    if "product_class_offer" not in combined:
        combined["product_class_offer"] = "<unknown>"
    return combined, hashes


def _critical_errors(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    accepted_false_positive = (
        (frame["pair_label"].to_numpy(dtype=int) == 0)
        & (probabilities >= threshold)
    )
    errors: dict[str, int] = {}
    for column in CRITICAL_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        errors[f"critical_{column}_errors"] = int(
            (accepted_false_positive & (values == 0)).sum()
        )
    errors["critical_error_total"] = int(sum(errors.values()))
    return errors


def _model_evaluation(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    policy: PromotionPolicy,
) -> dict[str, Any]:
    metrics = evaluate_binary_classifier(
        frame,
        probabilities,
        threshold=policy.operational_threshold,
    )
    accepted = probabilities >= policy.operational_threshold
    metrics["coverage_at_operational_threshold"] = float(accepted.mean())
    metrics["accepted_rows_at_operational_threshold"] = int(accepted.sum())
    metrics["precision_at_operational_threshold"] = metrics["precision"]
    metrics["critical_errors"] = _critical_errors(
        frame, probabilities, policy.operational_threshold
    )
    recent = frame["evaluation_set"].astype(str).eq(
        "recent_human_confirmed"
    )
    metrics["recent_human_confirmed"] = evaluate_binary_classifier(
        frame.loc[recent],
        probabilities[recent.to_numpy()],
        threshold=policy.operational_threshold,
    )
    return metrics


def _subgroup_regressions(
    champion: Mapping[str, Any],
    challenger: Mapping[str, Any],
    policy: PromotionPolicy,
) -> list[dict[str, Any]]:
    champion_groups = champion.get("by_product_family", {})
    challenger_groups = challenger.get("by_product_family", {})
    regressions: list[dict[str, Any]] = []
    for group in sorted(set(champion_groups) & set(challenger_groups)):
        champion_group = champion_groups[group]
        challenger_group = challenger_groups[group]
        row_count = min(
            int(champion_group["row_count"]),
            int(challenger_group["row_count"]),
        )
        if row_count < policy.minimum_subgroup_rows:
            continue
        decline = float(champion_group["precision"]) - float(
            challenger_group["precision"]
        )
        if decline > policy.subgroup_precision_tolerance:
            regressions.append(
                {
                    "product_family": group,
                    "row_count": row_count,
                    "champion_precision": champion_group["precision"],
                    "challenger_precision": challenger_group["precision"],
                    "precision_decline": decline,
                }
            )
    return regressions


def _regression_cases(
    frame: pd.DataFrame,
    champion_probability: np.ndarray,
    challenger_probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    labels = frame["pair_label"].to_numpy(dtype=int)
    champion_prediction = (champion_probability >= threshold).astype(int)
    challenger_prediction = (challenger_probability >= threshold).astype(int)
    mask = (champion_prediction == labels) & (challenger_prediction != labels)
    identifying = [
        column
        for column in (
            "evaluation_set",
            "record_id",
            "offer_group_id",
            "offer_text",
            "master_itemcode",
            "pair_label",
            "product_class_offer",
            "row_review_id",
        )
        if column in frame
    ]
    output = frame.loc[mask, identifying].copy()
    output["champion_probability"] = champion_probability[mask]
    output["challenger_probability"] = challenger_probability[mask]
    output["champion_prediction"] = champion_prediction[mask]
    output["challenger_prediction"] = challenger_prediction[mask]
    return output


def _promotion_checks(
    champion: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    policy: PromotionPolicy,
    evaluation_rows: int,
    package_validation_passed: bool,
    regression_tests_passed: bool,
) -> dict[str, dict[str, Any]]:
    subgroup_regressions = _subgroup_regressions(
        champion, challenger, policy
    )
    champion_critical = champion["critical_errors"]["critical_error_total"]
    challenger_critical = challenger["critical_errors"]["critical_error_total"]
    checks = {
        "minimum_evaluation_rows": {
            "passed": evaluation_rows >= policy.minimum_evaluation_rows,
            "actual": evaluation_rows,
            "required": policy.minimum_evaluation_rows,
        },
        "precision_at_0_85_not_materially_worse": {
            "passed": (
                challenger["precision_at_operational_threshold"]
                + policy.precision_tolerance
                >= champion["precision_at_operational_threshold"]
            ),
            "champion": champion["precision_at_operational_threshold"],
            "challenger": challenger["precision_at_operational_threshold"],
            "tolerance": policy.precision_tolerance,
        },
        "critical_errors_not_worse": {
            "passed": challenger_critical <= champion_critical,
            "champion": champion_critical,
            "challenger": challenger_critical,
        },
        "brier_score_not_materially_worse": {
            "passed": (
                challenger["calibration"]["brier_score"]
                <= champion["calibration"]["brier_score"]
                + policy.calibration_brier_tolerance
            ),
            "champion": champion["calibration"]["brier_score"],
            "challenger": challenger["calibration"]["brier_score"],
            "tolerance": policy.calibration_brier_tolerance,
        },
        "calibration_ece_not_materially_worse": {
            "passed": (
                challenger["calibration"]["expected_calibration_error"]
                <= champion["calibration"]["expected_calibration_error"]
                + policy.calibration_ece_tolerance
            ),
            "champion": champion["calibration"][
                "expected_calibration_error"
            ],
            "challenger": challenger["calibration"][
                "expected_calibration_error"
            ],
            "tolerance": policy.calibration_ece_tolerance,
        },
        "no_unacceptable_subgroup_regression": {
            "passed": not subgroup_regressions,
            "regressions": subgroup_regressions,
            "minimum_rows": policy.minimum_subgroup_rows,
            "tolerance": policy.subgroup_precision_tolerance,
        },
        "coverage_remains_acceptable": {
            "passed": (
                challenger["coverage_at_operational_threshold"]
                + policy.coverage_tolerance
                >= champion["coverage_at_operational_threshold"]
            ),
            "champion": champion["coverage_at_operational_threshold"],
            "challenger": challenger["coverage_at_operational_threshold"],
            "tolerance": policy.coverage_tolerance,
        },
        "package_validation": {"passed": package_validation_passed},
        "regression_tests": {"passed": regression_tests_passed},
    }
    return checks


def compare_models(
    *,
    champion_model_id: str,
    challenger_package_path: str | Path,
    challenger_metadata_path: str | Path,
    snapshot_manifest_path: str | Path,
    registry: ControlledModelRegistry,
    store: LearningStore,
    config: PipelineConfig,
    additional_evaluation_paths: Sequence[str | Path] = (),
    policy: PromotionPolicy | None = None,
    regression_tests_passed: bool,
    register_if_passed: bool = True,
) -> ComparisonResult:
    """Compare both packages on identical protected evaluation rows."""
    effective = policy or PromotionPolicy.from_pipeline_config(config)
    champion, _, champion_path = registry.load_registered_package(
        champion_model_id
    )
    challenger_path = Path(challenger_package_path)
    assert_not_sealed_challenge_input(challenger_path)
    challenger = load_model_package(challenger_path)
    challenger_model_id = str(challenger["model_id"])
    if str(challenger.get("parent_model")) != champion_model_id:
        raise ValueError("Challenger parent does not match champion model ID")
    frame, evaluation_hashes = _load_evaluation_sets(
        snapshot_manifest_path,
        additional_evaluation_paths,
    )
    champion_probability = _probabilities(champion, frame)
    challenger_probability = _probabilities(challenger, frame)
    champion_metrics = _model_evaluation(
        frame, champion_probability, effective
    )
    challenger_metrics = _model_evaluation(
        frame, challenger_probability, effective
    )
    checks = _promotion_checks(
        champion_metrics,
        challenger_metrics,
        policy=effective,
        evaluation_rows=len(frame),
        package_validation_passed=True,
        regression_tests_passed=regression_tests_passed,
    )
    passed = all(bool(check["passed"]) for check in checks.values())
    timestamp = datetime.now(timezone.utc)
    identity = (
        f"{champion_model_id}|{challenger_model_id}|"
        f"{json.dumps(evaluation_hashes, sort_keys=True)}"
    )
    import hashlib

    comparison_id = (
        f"comparison-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"
    )
    directory = config.retraining.comparison_report_directory / comparison_id
    report_path = directory / "comparison_report.json"
    regression_path = directory / "regression_cases.csv"
    regressions = _regression_cases(
        frame,
        champion_probability,
        challenger_probability,
        effective.operational_threshold,
    )
    report: dict[str, Any] = {
        "comparison_schema_version": "phase-7c-v1",
        "comparison_id": comparison_id,
        "created_at": timestamp.isoformat(),
        "champion_model_id": champion_model_id,
        "challenger_model_id": challenger_model_id,
        "champion_package_sha256": sha256_file(champion_path),
        "challenger_package_sha256": sha256_file(challenger_path),
        "evaluation_sets": evaluation_hashes,
        "evaluation_rows": int(len(frame)),
        "same_rows_for_both_models": True,
        "sealed_challenge_opened": False,
        "operational_threshold": effective.operational_threshold,
        "promotion_policy": asdict(effective),
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "promotion_checks": checks,
        "regression_case_count": int(len(regressions)),
        "promotion_policy_passed": passed,
        "decision": (
            "APPROVED_FOR_ASSISTED_USE" if passed else "REJECTED"
        ),
        "automatic_production_matching_approved": False,
        "activation_performed": False,
        "active_model_before": registry.active_model_id(),
    }
    atomic_csv(regressions, regression_path)
    atomic_json(report, report_path)
    registered_path: Path | None = None
    if passed and register_if_passed:
        entry = registry.register_approved_challenger(
            package_path=challenger_path,
            metadata_path=challenger_metadata_path,
            comparison_summary=report,
        )
        registered_path = (
            registry.model_directory / str(entry["package_filename"])
        )
    elif not passed:
        store.update_model_lifecycle(
            model_id=challenger_model_id,
            status="REJECTED",
            champion_status="REJECTED_CHALLENGER",
            evaluation_summary=report,
        )
    report["active_model_after"] = registry.active_model_id()
    report["registered_for_assisted_use"] = registered_path is not None
    atomic_json(report, report_path)
    return ComparisonResult(
        comparison_id=comparison_id,
        promotion_policy_passed=passed,
        decision=str(report["decision"]),
        report_path=report_path,
        regression_cases_path=regression_path,
        report=report,
        registered_package_path=registered_path,
    )
