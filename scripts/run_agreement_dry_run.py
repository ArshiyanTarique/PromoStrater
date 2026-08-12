"""Evaluate Phase 6C routes from existing scorer outputs; call no LLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sku_mapping.agreement.policy import evaluate_candidate_agreement
from sku_mapping.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-predictions", type=Path, required=True
    )
    parser.add_argument(
        "--existing-decisions",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/assisted/assisted-20260729T073733694314Z/"
        "assisted_decisions.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/default.yaml",
    )
    return parser.parse_args()


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _choice_pattern(row: pd.Series) -> str:
    existing = str(row.get("existing_production_choice", ""))
    lightgbm = str(row.get("lightgbm_top_candidate", ""))
    embedding = str(row.get("embedding_top_candidate", ""))
    if existing == lightgbm == embedding:
        return "ALL_THREE_SAME"
    if lightgbm == embedding:
        return "SCORERS_AGREE_EXISTING_DIFFERS"
    if existing == lightgbm:
        return "EXISTING_EQUALS_LIGHTGBM_ONLY"
    if existing == embedding:
        return "EXISTING_EQUALS_EMBEDDING_ONLY"
    return "ALL_DIFFERENT"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    candidates = pd.read_parquet(args.candidate_predictions)
    evaluation = evaluate_candidate_agreement(
        candidates, config=config.agreement
    )
    offer_identity = candidates.drop_duplicates("offer_group_id")[
        ["offer_group_id", "source_row_identifier"]
    ].rename(
        columns={
            "offer_group_id": "offer_id",
            "source_row_identifier": "source_offer_id",
        }
    )
    table = evaluation.frame.merge(
        offer_identity, on="offer_id", how="left", validate="one_to_one"
    )
    if args.existing_decisions.is_file():
        existing = pd.read_csv(
            args.existing_decisions,
            dtype="string",
            keep_default_na=False,
        )[
            ["offer_identifier", "candidate_identifier", "decision"]
        ].rename(
            columns={
                "offer_identifier": "source_offer_id",
                "candidate_identifier": "existing_production_choice",
                "decision": "existing_production_decision",
            }
        )
        table["source_offer_id"] = table["source_offer_id"].astype(str)
        existing["source_offer_id"] = existing[
            "source_offer_id"
        ].astype(str)
        table["_source_occurrence"] = table.groupby(
            "source_offer_id", sort=False
        ).cumcount()
        existing["_source_occurrence"] = existing.groupby(
            "source_offer_id", sort=False
        ).cumcount()
        table = table.merge(
            existing,
            on=["source_offer_id", "_source_occurrence"],
            how="left",
            validate="one_to_one",
        )
        table.drop(columns=["_source_occurrence"], inplace=True)
    else:
        table["existing_production_choice"] = ""
        table["existing_production_decision"] = "UNAVAILABLE"
    table["choice_comparison"] = table.apply(
        _choice_pattern, axis=1
    )
    confusion_columns = [
        "source_offer_id",
        "offer_id",
        "existing_production_choice",
        "existing_production_decision",
        "lightgbm_top_candidate",
        "lightgbm_calibrated_probability",
        "embedding_top_candidate",
        "embedding_similarity",
        "same_top_candidate",
        "agreement_status",
        "routing_decision",
        "routing_reason",
        "choice_comparison",
    ]
    confusion = table.loc[:, confusion_columns]

    now = datetime.now(timezone.utc)
    run_id = "agreement-dry-run-" + now.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = config.output.output_dir / "agreement_dry_runs" / run_id
    agreement_path = output / "agreement_results.csv"
    confusion_path = output / "confusion_style_table.csv"
    summary_path = output / "agreement_dry_run_summary.json"
    _atomic_csv(evaluation.frame, agreement_path)
    _atomic_csv(confusion, confusion_path)

    route_counts = {
        str(key): int(value)
        for key, value in evaluation.frame[
            "routing_decision"
        ].value_counts().items()
    }
    status_counts = {
        str(key): int(value)
        for key, value in evaluation.frame[
            "agreement_status"
        ].value_counts().items()
    }
    comparison_counts = {
        str(key): int(value)
        for key, value in confusion[
            "choice_comparison"
        ].value_counts().items()
    }
    cross_tab = pd.crosstab(
        confusion["existing_production_decision"],
        confusion["routing_decision"],
        dropna=False,
    )
    summary = {
        "report_type": "PHASE_6C_AGREEMENT_DRY_RUN",
        "timestamp": now.isoformat(),
        "run_id": run_id,
        "candidate_predictions": str(
            args.candidate_predictions.resolve()
        ),
        "existing_decisions": str(
            args.existing_decisions.resolve()
        ),
        "offers": len(evaluation.frame),
        "agreement_status_counts": status_counts,
        "routing_decision_counts": route_counts,
        "choice_comparison_counts": comparison_counts,
        "existing_decision_by_agreement_route": {
            str(index): {
                str(column): int(value)
                for column, value in row.items()
            }
            for index, row in cross_tab.to_dict(orient="index").items()
        },
        "lightgbm_auto_accept_threshold": (
            config.agreement.lightgbm_auto_accept_threshold
        ),
        "minimum_embedding_similarity": (
            config.agreement.minimum_embedding_similarity
        ),
        "minimum_embedding_margin": (
            config.agreement.minimum_embedding_margin
        ),
        "llm_called": False,
        "learning_dataset_modified": False,
        "routes_are_diagnostic": True,
        "artifacts": {
            "agreement_results": str(agreement_path),
            "confusion_style_table": str(confusion_path),
        },
    }
    _atomic_json(summary, summary_path)
    summary["artifacts"]["summary"] = str(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
