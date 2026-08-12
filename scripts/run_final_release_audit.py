"""Run a bounded, isolated dashboard-to-review release audit.

This script deliberately writes under ``outputs/release_audit`` and uses an
isolated SQLite database. The generated GOLD answers are audit fixtures, not
human evidence and not eligible for the repository's real retraining store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard.services.processing_service import (
    DashboardProcessRequest,
    DashboardProcessingService,
)
from dashboard.services.review_service import DashboardReviewService
from dashboard.services.run_service import DashboardRunService
from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.learning.store import LearningStore

DEFAULT_MODEL_ID = (
    "alkabeer-sku-matcher-v3-20260729T061802974421Z-8c636b0ac4a2"
)


def _fixture_upload() -> bytes:
    """Build five unique, bounded offers from the repository fixture."""
    fixture = pd.read_csv(
        PROJECT_ROOT / "tests" / "fixtures" / "clickflyer_valid.csv"
    )
    base = fixture.iloc[0]
    rows: list[pd.Series] = []
    weights = (270, 400, 500, 750, 1000)
    for position, weight in enumerate(weights, start=1):
        row = base.copy()
        row["offerid"] = f"phase8-audit-{position}"
        row["Offer Name"] = f"Al Kabeer Chicken Nuggets {weight}g"
        row["Base Packsize"] = f"{weight}g"
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _audit_config(audit_root: Path):
    base = load_config(PROJECT_ROOT / "config" / "default.yaml")
    return replace(
        base,
        output=replace(
            base.output,
            output_dir=audit_root / "pipeline_outputs",
        ),
        dashboard=replace(
            base.dashboard,
            input_directory=audit_root / "uploads",
            output_directory=audit_root / "dashboard_outputs",
        ),
        learning_store=replace(
            base.learning_store,
            database_path=audit_root / "audit_learning.db",
            csv_export_directory=audit_root / "learning_exports",
        ),
        shadow_mode=replace(
            base.shadow_mode,
            output_directory=audit_root / "shadow",
            review_staging_directory=audit_root / "review_staging",
            challenge_set_directory=audit_root / "unused_challenge_path",
        ),
        embedding=replace(
            base.embedding,
            enabled=True,
            backend="local_hashing",
            model_name="sku-hashing-384",
            model_version="sku-hashing-384-v1",
            cache_path=audit_root / "embedding_cache.sqlite3",
        ),
        llm_review=replace(
            base.llm_review,
            enabled=False,
            cache_path=audit_root / "llm_cache.sqlite3",
        ),
    )


def run_audit(*, model_id: str) -> dict[str, Any]:
    """Execute and validate one isolated five-offer assisted-mode run."""
    timestamp = datetime.now(timezone.utc)
    audit_id = f"phase8-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}"
    repository_root = Path.cwd().resolve()
    audit_root = repository_root / "outputs" / "release_audit" / audit_id
    config = _audit_config(audit_root)
    content = _fixture_upload()
    stages: list[dict[str, Any]] = []

    processing = DashboardProcessingService(config)
    result = processing.process(
        DashboardProcessRequest(
            filename="../phase8 bounded fixture.csv",
            content=content,
            deployment_mode=MLDeploymentMode.ASSISTED,
            model_id=model_id,
            enable_embedding=True,
            enable_llm_review=False,
        ),
        progress=lambda update: stages.append(
            {
                "stage": update.stage_label,
                "percent": update.overall_percent,
                "state": update.state.value,
            }
        ),
    )

    store = LearningStore(config.learning_store.database_path)
    session = store.review_session_for_run(result.run_id)
    if session is None:
        raise RuntimeError("Bounded run did not create its five-question session")
    session_id = str(session["session_id"])
    questions = store.review_questions(session_id)
    if len(questions) != 5:
        raise RuntimeError(f"Expected five questions, received {len(questions)}")

    review_service = DashboardReviewService(store)
    corrected_candidate_id: str | None = None
    for index, question in enumerate(questions):
        if index == 0:
            suggested = str(question["suggested_candidate_id"])
            alternatives = [
                str(candidate["candidate_id"])
                for candidate in question["supplied_candidates"]
                if str(candidate["candidate_id"]) != suggested
            ]
            if not alternatives:
                raise RuntimeError(
                    "Audit fixture lacks a supplied alternative for correction"
                )
            corrected_candidate_id = alternatives[0]
            review_service.save(
                review_id=str(question["review_id"]),
                answer="FALSE_CANDIDATE",
                corrected_candidate_id=corrected_candidate_id,
                reviewer_id="PHASE8_AUDIT_FIXTURE",
                notes="Isolated release-audit correction; not real human evidence.",
            )
        else:
            review_service.save(
                review_id=str(question["review_id"]),
                answer="TRUE",
                reviewer_id="PHASE8_AUDIT_FIXTURE",
                notes="Isolated release-audit answer; not real human evidence.",
            )

    progress = review_service.progress(session_id)
    if (progress.answered, progress.total) != (5, 5):
        raise RuntimeError("Five-question audit session did not complete")

    run_service = DashboardRunService(config, store)
    downloads = run_service.downloads(result.run_id)
    download_keys = {artifact.key for artifact in downloads}
    required_downloads = {"sku_mapping", "competitor_offers", "run_summary"}
    if not required_downloads.issubset(download_keys):
        raise RuntimeError(
            f"Validated downloads are missing: {sorted(required_downloads - download_keys)}"
        )

    run_record = store.get_pipeline_run(result.run_id)
    if run_record is None:
        raise RuntimeError("Completed audit run is missing from SQLite")
    monitoring_path = Path(
        str(run_record["output_paths"].get("monitoring_report", ""))
    )
    if not monitoring_path.is_file():
        raise RuntimeError("Monitoring output is missing")
    monitoring = json.loads(monitoring_path.read_text(encoding="utf-8"))
    database_summary = store.summary()
    database_summary["database_path"] = _relative(
        config.learning_store.database_path, repository_root
    )
    reviewed = store.reviewed_labels()
    gold_count = sum(
        str(record.get("label_quality")) == "GOLD" for record in reviewed
    )
    corrected_gold_count = sum(
        str(record.get("label_quality")) == "GOLD"
        and bool(record.get("corrected_candidate_id"))
        for record in reviewed
    )
    if gold_count != 5 or corrected_gold_count != 1:
        raise RuntimeError("Audit answers did not persist as five governed GOLD rows")

    summary = result.summary
    report = {
        "report_type": "PHASE8_BOUNDED_RELEASE_AUDIT",
        "audit_fixture_notice": (
            "All five answers use an isolated audit reviewer identity and are "
            "not real human evidence or real retraining input."
        ),
        "audit_id": audit_id,
        "run_id": result.run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_fixture": "tests/fixtures/clickflyer_valid.csv",
        "source_bytes_sha256": _sha256(content),
        "deployment_mode": "assisted",
        "model_id": model_id,
        "operational_threshold": summary.get("operational_threshold"),
        "threshold_source": summary.get("threshold_source"),
        "production_threshold_approved": summary.get(
            "production_threshold_approved"
        ),
        "rows_processed": summary.get("input_rows"),
        "unique_offers": summary.get("unique_offers"),
        "candidate_count": summary.get("candidates_generated"),
        "auto_accept_count": summary.get("auto_accept_count"),
        "llm_accept_count": summary.get("llm_accept_count"),
        "manual_review_count": summary.get("manual_review_count"),
        "no_candidate_count": summary.get("no_candidate_count"),
        "model_failures": summary.get("model_error_count"),
        "embedding_failures": monitoring.get("embedding_scoring", {}).get(
            "failure_rows", 0
        ),
        "llm_failures": (
            monitoring.get("llm_review", {}).get("invalid_responses", 0)
            + monitoring.get("llm_review", {}).get("timeouts", 0)
            + monitoring.get("llm_review", {}).get("provider_failures", 0)
        ),
        "five_question_selection": [
            {
                "question_number": int(question["question_number"]),
                "selection_category": question["selection_category"],
                "selection_reason": question["selection_reason"],
                "fallback_reason": question["fallback_reason"],
            }
            for question in questions
        ],
        "five_question_completion": {
            "answered": progress.answered,
            "total": progress.total,
            "gold_labels": gold_count,
            "corrected_candidate_gold_labels": corrected_gold_count,
            "corrected_candidate_id": corrected_candidate_id,
            "test_only_database": True,
        },
        "output_validation": {
            "validated_download_keys": sorted(download_keys),
            "artifacts": {
                artifact.key: {
                    "filename": artifact.filename,
                    "size_bytes": len(artifact.content),
                    "sha256": _sha256(artifact.content),
                }
                for artifact in downloads
            },
        },
        "monitoring_output": {
            "path": _relative(monitoring_path, repository_root),
            "report_type": monitoring.get("report_type"),
            "not_production_accuracy": monitoring.get(
                "not_production_accuracy"
            ),
            "agreement_routing": monitoring.get("agreement_routing"),
            "llm_review": monitoring.get("llm_review"),
        },
        "database_records": database_summary,
        "progress_stages": stages,
        "stage_runtimes_seconds": summary.get("stage_runtimes_seconds", {}),
        "total_runtime_seconds": summary.get("total_runtime_seconds"),
        "training_or_online_learning_performed": summary.get(
            "training_or_online_learning_performed"
        ),
        "sealed_challenge_opened": False,
    }
    report_path = audit_root / "phase8_bounded_release_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    report["audit_report_path"] = _relative(report_path, repository_root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated Phase 8 dashboard release audit."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(model_id=args.model_id),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
