"""Page-independent upload-to-download dashboard workflow."""

from __future__ import annotations

import logging
import json
import threading
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from dashboard.services.job_state import (
    CANCELLED_RUN_STATUS,
    CancellationToken,
    RunCancelled,
)
from dashboard.services.registry_service import DashboardRegistryService
from dashboard.services.progress import (
    ProcessingState,
    ProgressCallback,
    ProgressTracker,
    inference_progress_units,
)
from dashboard.services.upload_service import DashboardUploadService
from sku_mapping.config import PipelineConfig
from sku_mapping.constants import MLDeploymentMode
from sku_mapping.data.loaders import load_product_master
from sku_mapping.data.offer_identity import (
    FALLBACK_IDENTITY_VERSION,
    assign_offer_identities,
)
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.exports.business_outputs import build_business_outputs
from sku_mapping.exports.run_outputs import write_run_outputs
from sku_mapping.failure_diagnostics import (
    capture_exception_details,
    exception_summary,
)
from sku_mapping.inference.pipeline import (
    UnifiedInferenceResult,
    run_unified_inference_non_blocking,
)
from sku_mapping.learning.store import LearningStore

LOGGER = logging.getLogger(__name__)
_PROCESS_START_LOCK = threading.Lock()


class DuplicateProcessingError(ValueError):
    """Raised when uploaded bytes already have an active/completed run."""

    def __init__(self, run_ids: list[str]) -> None:
        self.run_ids = run_ids
        super().__init__(
            "This exact file has already been processed or is processing"
        )


class ProcessingCancelledError(RuntimeError):
    """Raised when a run stopped cleanly at a cancellation checkpoint.

    This is a successful cooperative stop, not a failure: partial artifacts
    produced before the checkpoint are reported so they can be preserved.
    """

    def __init__(
        self,
        message: str = "Processing was cancelled before completion",
        *,
        run_id: str | None = None,
        cancelled_stage: str | None = None,
        last_completed_stage: str | None = None,
        partial_artifacts: tuple[str, ...] = (),
    ) -> None:
        self.run_id = run_id
        self.cancelled_stage = cancelled_stage
        self.last_completed_stage = last_completed_stage
        self.partial_artifacts = partial_artifacts
        super().__init__(message)


class DashboardProcessingError(RuntimeError):
    """Safe error returned to the UI without internal traceback details."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        failed_stage: str | None = None,
        last_completed_stage: str | None = None,
        technical_details: str | None = None,
        partial_artifacts: tuple[str, ...] = (),
        pipeline_status: str | None = None,
        original_exception: dict[str, object] | None = None,
    ) -> None:
        self.run_id = run_id
        self.failed_stage = failed_stage
        self.last_completed_stage = last_completed_stage
        self.technical_details = technical_details
        self.partial_artifacts = partial_artifacts
        self.pipeline_status = pipeline_status
        self.original_exception = original_exception
        super().__init__(message)


class UnifiedPipelineStatusError(RuntimeError):
    """Preserve a non-success pipeline status without losing its cause."""

    def __init__(
        self,
        pipeline_status: str,
        original_exception: dict[str, object] | None,
    ) -> None:
        self.pipeline_status = pipeline_status
        self.original_exception = original_exception
        super().__init__(f"Unified inference ended with {pipeline_status}")


@dataclass(frozen=True)
class DashboardProcessRequest:
    """Explicit processing request from the presentation layer."""

    filename: str
    content: bytes
    deployment_mode: MLDeploymentMode
    model_id: str
    allow_duplicate: bool = False
    enable_embedding: bool = False
    enable_llm_review: bool = False


@dataclass(frozen=True)
class DashboardProcessResult:
    """Durable run identity and safe page-level summary."""

    run_id: str
    status: str
    summary: dict[str, object]
    review_session_id: str | None


def _competitor_review_staging(store, business, config: PipelineConfig, run_id: str) -> int:
    """Stage a review queue from a finished run, never failing the run.

    The outputs are already written and correct by the time this runs, so a
    staging problem must degrade to "no queue this time" rather than losing a
    two-hour run. Returns the number of rows staged, 0 when disabled.
    """
    per_target = int(
        getattr(config.competitors, "review_staging_per_target", 0) or 0
    )
    if per_target <= 0:
        return 0
    try:
        from sku_mapping.competitors.review import stage_competitor_review_queue

        competitors = getattr(business, "competitors", None)
        export = getattr(competitors, "export", None)
        if export is None:
            return 0
        ml = (getattr(competitors, "diagnostics", None) or {}).get(
            "ml_reranking", {}
        )
        return stage_competitor_review_queue(
            store,
            export,
            run_id=run_id,
            per_target=per_target,
            model_id=(ml.get("model_id") or None),
            ranking_source="lightgbm" if ml.get("enabled") else "rules",
        )
    except Exception:
        LOGGER.exception(
            "Competitor review staging failed; the run and its outputs stand"
        )
        return 0


def _competitor_reranker(config: PipelineConfig):
    """Build the competitor re-ranker when configuration asks for one.

    Returns ``None`` whenever it is disabled or the package cannot be loaded,
    which is the same value competitor discovery receives today - the rules
    then order candidates exactly as they did before ML existed. The model is
    resolved through the shadow registry, so a package that is unregistered or
    not in ``SHADOW_MODE_ONLY`` is refused here as it would be for own-brand.
    """
    competitors = config.competitors
    if not getattr(competitors, "ml_reranking_enabled", False):
        return None
    from sku_mapping.competitors.reranker import load_competitor_reranker

    shadow = config.shadow_mode
    model_directory = (
        shadow.package_reference.parent
        if shadow.package_reference is not None
        else shadow.registry_path.parent / "registry"
    )
    reranker = load_competitor_reranker(
        registry_path=shadow.registry_path,
        model_directory=model_directory,
        model_id=shadow.model_id,
        package_reference=shadow.package_reference,
        require_package_status=shadow.require_package_status,
        strip_brand=getattr(competitors, "brand_stripping_enabled", True),
    )
    if reranker is None:
        LOGGER.warning(
            "Competitor ML re-ranking is enabled but no model package could "
            "be loaded; competitor discovery will use rule ordering"
        )
    return reranker


def _runtime_component_summary(
    request: DashboardProcessRequest,
    result: UnifiedInferenceResult,
) -> dict[str, str]:
    """Expose component participation without changing business decisions."""
    completed = result.status.startswith("COMPLETED")
    model_ok = completed and not int(
        result.statistics.get("model_error_count", 0) or 0
    )
    embedding_observed = str(
        result.statistics.get("embedding_status") or ""
    ).upper()
    embedding_requested = bool(
        result.statistics.get("embedding_requested", False)
    )
    embedding_available = bool(
        result.statistics.get("embedding_available", False)
    )
    embedding_used = bool(
        result.statistics.get("embedding_used", False)
    )
    if not request.enable_embedding or not embedding_requested:
        embedding_status = "DISABLED"
    elif embedding_available and embedding_used:
        embedding_status = "ACTIVE"
    elif embedding_observed == "UNAVAILABLE":
        embedding_status = "UNAVAILABLE"
    else:
        embedding_status = "NOT_EXERCISED"

    llm_calls = int(result.statistics.get("llm_calls", 0) or 0)
    llm_status = (
        "ACTIVE"
        if request.enable_llm_review and llm_calls > 0
        else "NOT_USED"
    )
    return {
        "candidate_generation": "ACTIVE" if completed else "FAILED",
        "feature_generation": "ACTIVE" if completed else "FAILED",
        "lightgbm": "ACTIVE" if model_ok else "FAILED",
        "embedding": embedding_status,
        "llm": llm_status,
        "competitor_discovery": "ACTIVE",
    }


class DashboardProcessingService:
    """Orchestrate existing services without embedding business logic in pages."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        pipeline_runner: Callable[..., UnifiedInferenceResult] = (
            run_unified_inference_non_blocking
        ),
    ) -> None:
        self.config = config
        self.uploads = DashboardUploadService(config.dashboard)
        self.registry = DashboardRegistryService(config)
        self.store = LearningStore(config.learning_store.database_path)
        self.pipeline_runner = pipeline_runner

    def process(
        self,
        request: DashboardProcessRequest,
        *,
        progress: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> DashboardProcessResult:
        """Validate, stage, process, export, and persist one upload."""
        tracker = ProgressTracker(progress)
        tracker.start()
        token = cancel_token or CancellationToken()

        def checkpoint(stage_key: str | None = None) -> None:
            """Stop cleanly between stages once cancellation was requested."""
            if token.cancelled:
                if tracker.state is not ProcessingState.CANCELLING:
                    tracker.cancelling(detail=None)
                raise RunCancelled(stage_key)

        if request.deployment_mode not in {
            MLDeploymentMode.SHADOW,
            MLDeploymentMode.ASSISTED,
        }:
            raise DashboardProcessingError(
                "Dashboard processing supports shadow or assisted mode"
            )
        upload = self.uploads.validate(request.filename, request.content)
        try:
            self.registry.validate_model_id(request.model_id)
        except Exception as error:
            raise DashboardProcessingError(
                "The selected model is unavailable or incompatible"
            ) from error
        tracker.update(
            "validation",
            completed=1,
            total=1,
            detail=(
                f"Validated {upload.sanitized_filename} "
                f"({upload.size_bytes:,} bytes)"
            ),
        )

        now = datetime.now(timezone.utc)
        run_id = (
            f"dashboard-{now.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{upload.source_file_hash[:12]}"
        )
        with _PROCESS_START_LOCK:
            duplicates = (
                self.store.active_or_completed_runs_for_source_hash(
                    upload.source_file_hash
                )
            )
            if duplicates and not request.allow_duplicate:
                raise DuplicateProcessingError(
                    [str(item["run_id"]) for item in duplicates]
                )
            self.store.upsert_pipeline_run(
                {
                    "run_id": run_id,
                    "started_at": now.isoformat(),
                    "source_filename": upload.sanitized_filename,
                    "source_file_hash": upload.source_file_hash,
                    "deployment_mode": request.deployment_mode.value,
                    "status": "VALIDATING",
                    "model_id": request.model_id,
                    "threshold": self.config.ml.auto_accept_threshold,
                }
            )
        try:
            checkpoint("input_loading")
            staged_path = self.uploads.stage(upload, run_id=run_id)
            tracker.update(
                "input_loading",
                detail=f"Reading {upload.size_bytes:,} bytes",
                run_id=run_id,
            )
            raw = self.uploads.read_and_validate(staged_path)
            tracker.update(
                "input_loading",
                completed=upload.size_bytes,
                total=upload.size_bytes,
                detail=f"Loaded {len(raw):,} source rows",
                run_id=run_id,
            )
            tracker.update(
                "canonicalisation",
                completed=0,
                total=len(raw),
                detail=f"Preparing {len(raw):,} rows",
                run_id=run_id,
            )
            checkpoint("canonicalisation")
            prepared = preprocess_clickflyer(raw)
            identity = assign_offer_identities(prepared)
            prepared["offer_group_id"] = identity.identities
            source_counts = prepared.groupby(
                "offer_group_id", sort=False
            ).size()
            canonical_offers = prepared.drop_duplicates(
                "offer_group_id", keep="first"
            ).reset_index(drop=True)
            canonical_offers["source_row_count"] = (
                canonical_offers["offer_group_id"].map(source_counts).astype(int)
            )
            own = canonical_offers.loc[canonical_offers["is_own"]]
            inference_offer_count = int(
                own["offer_group_id"].nunique(dropna=True)
            )
            tracker.update(
                "canonicalisation",
                completed=len(raw),
                total=len(raw),
                detail=(
                    f"Prepared {identity.unique_offer_count:,} unique offers; "
                    f"{inference_offer_count:,} own-brand offers"
                ),
                run_id=run_id,
            )
            run_metadata = {
                "offer_identity_column": (
                    "offerid" if "offerid" in prepared.columns else None
                ),
                "offer_identity_source": identity.source,
                "offer_identity_fallback_version": (
                    FALLBACK_IDENTITY_VERSION
                ),
                "valid_offer_id_rows": identity.valid_offer_id_count,
                "missing_offer_id_rows": identity.missing_offer_id_count,
                "inference_offer_count": inference_offer_count,
                "canonical_offer_count": int(len(canonical_offers)),
                "inference_scope": "supported_own_brand_offers",
            }
            checkpoint("canonicalisation")
            master = preprocess_product_master(
                load_product_master(self.config.data.master_path)
            )
            effective = replace(
                self.config,
                output=replace(
                    self.config.output,
                    output_dir=self.config.dashboard.output_directory
                    / "_pipeline",
                ),
                ml=replace(
                    self.config.ml,
                    mode=request.deployment_mode,
                    model_id=request.model_id,
                ),
                shadow_mode=replace(
                    self.config.shadow_mode,
                    output_directory=(
                        self.config.dashboard.output_directory
                        / "_shadow"
                    ),
                ),
                embedding=replace(
                    self.config.embedding,
                    enabled=request.enable_embedding,
                ),
                llm_review=replace(
                    self.config.llm_review,
                    enabled=request.enable_llm_review,
                ),
            )
            self.store.upsert_pipeline_run(
                {
                    "run_id": run_id,
                    "started_at": now.isoformat(),
                    "source_filename": upload.sanitized_filename,
                    "source_file_hash": upload.source_file_hash,
                    "source_row_count": len(raw),
                    "unique_offer_count": identity.unique_offer_count,
                    "deployment_mode": request.deployment_mode.value,
                    "status": "PROCESSING",
                    "model_id": request.model_id,
                    "threshold": effective.ml.auto_accept_threshold,
                    "run_metadata": run_metadata,
                }
            )
            tracker.update(
                "inference",
                completed=0,
                total=inference_offer_count,
                detail=(
                    f"Generating candidates for "
                    f"{inference_offer_count:,} own-brand offers"
                ),
                run_id=run_id,
            )

            checkpoint("inference")

            def update_inference(
                phase: str,
                completed: int | None,
                total: int | None,
                detail: str,
            ) -> None:
                # Every inference phase already reports here, so this is the
                # natural cancellation checkpoint inside the ML pipeline
                # without modifying the pipeline itself.
                checkpoint(f"inference:{phase}")
                units, unit_total = inference_progress_units(
                    phase, completed, total
                )
                tracker.update(
                    "inference",
                    completed=units,
                    total=unit_total,
                    detail=detail,
                    run_id=run_id,
                )

            result = self.pipeline_runner(
                canonical_offers,
                master,
                config=effective,
                run_id=run_id,
                persist_records=True,
                source_path=staged_path,
                progress=update_inference,
            )
            # The unified runner fails closed and converts any exception --
            # including a cancellation raised from the progress callback --
            # into a safe-fallback result. A requested cancellation is
            # therefore reclassified here instead of being reported as an
            # inference failure.
            checkpoint("inference")
            if not result.status.startswith("COMPLETED"):
                status_error = UnifiedPipelineStatusError(
                    result.status,
                    result.original_exception,
                )
                if result.exception is not None:
                    raise status_error from result.exception
                raise status_error
            tracker.update(
                "inference",
                completed=1000,
                total=1000,
                detail=(
                    f"Scored {int(result.statistics.get('candidates_generated', 0)):,} "
                    f"candidate pairs for {inference_offer_count:,} offers"
                ),
                run_id=run_id,
            )

            tracker.update(
                "sku_mapping",
                completed=0,
                total=inference_offer_count,
                detail="Building the own-offer SKU mapping",
                run_id=run_id,
            )
            tracker.update(
                "sku_mapping",
                completed=inference_offer_count // 2,
                total=inference_offer_count,
                detail="Processing SKU mapping decisions",
                run_id=run_id,
            )
            competitor_audit_path = (
                self.config.dashboard.output_directory
                / run_id
                / f".competitor_long_form_{run_id}.partial.csv"
            )

            def update_business_stage(
                stage: str,
                completed: int,
                total: int,
                detail: str,
            ) -> None:
                checkpoint(stage)
                tracker.update(
                    stage,
                    completed=completed,
                    total=total,
                    detail=detail,
                    run_id=run_id,
                )

            def update_competitors(
                target_completed: int,
                target_total: int,
                relationships_completed: int,
                relationships_total: int,
                target_code: str,
            ) -> None:
                # Competitor discovery reports once per target SKU, giving a
                # fine-grained cooperative stop inside the longest stage.
                checkpoint("competitor_discovery")
                target_detail = (
                    f"Target SKU {target_completed:,} of "
                    f"{target_total:,} ({target_code})"
                    if target_completed
                    else f"Preparing {target_total:,} target SKUs"
                )
                tracker.update(
                    "competitor_discovery",
                    completed=relationships_completed,
                    total=relationships_total,
                    detail=(
                        f"{target_detail}; evaluated "
                        f"{relationships_completed:,} of "
                        f"{relationships_total:,} relationships"
                    ),
                    run_id=run_id,
                )

            business = build_business_outputs(
                result.rows,
                master,
                result.decisions,
                competitor_config=effective.competitors,
                run_id=run_id,
                competitor_audit_path=competitor_audit_path,
                competitor_progress=update_competitors,
                stage_progress=update_business_stage,
                competitor_reranker=_competitor_reranker(effective),
                # ``canonical_offers`` keeps one row per offer identity, which
                # is what inference needs. Competitor discovery needs the
                # variant-level rows: a ClickFlyer offer repeats once per
                # variant, and the collapsed frame retains only the first -
                # in practice the "No Variant" row. Offers whose protein
                # appears solely in ``Variant`` (assorted/mixed competitor
                # packs) then lose the only evidence tying them to a Master
                # SKU and are scored as unrelated.
                competitor_offers=prepared,
            )
            staged = _competitor_review_staging(
                self.store, business, effective, run_id
            )
            if staged:
                LOGGER.info(
                    "Staged %s competitor relationships for review", staged
                )
            runtime_components = _runtime_component_summary(
                request, result
            )
            persistence_diagnostics = self.store.summary()
            summary = {
                **result.statistics,
                **business.diagnostics,
                "run_id": run_id,
                "status": result.status,
                "input_rows": int(len(raw)),
                "unique_offers": identity.unique_offer_count,
                "inference_offers": inference_offer_count,
                **run_metadata,
                "source_filename": upload.sanitized_filename,
                "source_file_hash": upload.source_file_hash,
                "deployment_mode": request.deployment_mode.value,
                "operational_threshold": effective.ml.auto_accept_threshold,
                "threshold_source": "user_configured",
                "production_threshold_approved": False,
                "competitor_target_count": business.diagnostics[
                    "target_master_sku_count"
                ],
                "competitor_diagnostics": business.diagnostics,
                "runtime_components": runtime_components,
                "persistence_diagnostics": persistence_diagnostics,
                "training_or_online_learning_performed": False,
            }
            # Last checkpoint before the run becomes durable. Past this point
            # the export and persistence stages are allowed to finish so a
            # cancelled run never leaves half-written outputs behind.
            checkpoint("exports")
            monitoring_path = result.output_paths.get("monitoring_report")
            tracker.update(
                "exports",
                completed=0,
                total=4,
                detail="Writing four validated run artifacts",
                run_id=run_id,
            )
            outputs = write_run_outputs(
                run_id=run_id,
                sku_mapping_export=business.sku_mapping,
                competitor_export=business.competitor_export,
                competitor_long_format=business.competitor_long_format,
                competitor_long_format_path=(
                    business.competitor_long_format_path
                ),
                summary=summary,
                output_root=self.config.dashboard.output_directory,
                monitoring_path=monitoring_path,
                progress=lambda completed, total, detail: tracker.update(
                    "exports",
                    completed=completed,
                    total=total,
                    detail=detail,
                    run_id=run_id,
                ),
            )
            review_session = self.store.review_session_for_run(run_id)
            tracker.update(
                "persistence",
                completed=0,
                total=1,
                detail="Saving output locations and final run status",
                run_id=run_id,
            )
            self.store.update_run_outputs(
                run_id,
                outputs.paths,
                status=(
                    "COMPLETED_DASHBOARD_ASSISTED"
                    if request.deployment_mode
                    is MLDeploymentMode.ASSISTED
                    else "COMPLETED_DASHBOARD_SHADOW"
                ),
            )
            tracker.update(
                "persistence",
                completed=1,
                total=1,
                detail="Run results saved",
                run_id=run_id,
            )
            tracker.succeed(
                run_id=run_id,
                detail="All processing and output validation completed",
            )
            return DashboardProcessResult(
                run_id=run_id,
                status=str(
                    self.store.get_pipeline_run(run_id)["status"]
                ),
                summary=summary,
                review_session_id=(
                    str(review_session["session_id"])
                    if review_session is not None
                    else None
                ),
            )
        except DuplicateProcessingError:
            raise
        except RunCancelled as cancellation:
            LOGGER.info(
                "Dashboard run %s cancelled during stage %s",
                run_id,
                cancellation.stage_key,
            )
            partial_artifacts = self._collect_partial_artifacts(locals())
            try:
                log_path = self._write_cancellation_log(
                    run_id,
                    stage_key=cancellation.stage_key,
                    stage_label=tracker.latest.stage_label,
                    last_completed_stage=(
                        tracker.latest.last_completed_stage
                    ),
                    elapsed_seconds=tracker.latest.elapsed_seconds,
                    partial_artifacts=tuple(partial_artifacts),
                )
                partial_artifacts.append(str(log_path))
                self.store.update_run_outputs(
                    run_id,
                    {"cancellation_log": log_path},
                    status=CANCELLED_RUN_STATUS,
                    error_summary=(
                        "Cancelled by user during "
                        f"{tracker.latest.stage_label or 'processing'}"
                    ),
                )
            except Exception:
                LOGGER.exception(
                    "Could not persist cancellation details for %s", run_id
                )
            tracker.cancelled(
                run_id=run_id,
                partial_artifacts=tuple(partial_artifacts),
            )
            raise ProcessingCancelledError(
                "Processing was cancelled before completion",
                run_id=run_id,
                cancelled_stage=tracker.latest.stage_label,
                last_completed_stage=tracker.latest.last_completed_stage,
                partial_artifacts=tuple(partial_artifacts),
            ) from cancellation
        except Exception as error:
            LOGGER.exception("Dashboard processing failed for run %s", run_id)
            technical_details = traceback.format_exc()
            pipeline_status = getattr(
                error,
                "pipeline_status",
                "DASHBOARD_PROCESSING_ERROR",
            )
            original_exception = getattr(
                error,
                "original_exception",
                None,
            ) or capture_exception_details(error)
            exact_error = exception_summary(original_exception)
            partial_artifacts = self._collect_partial_artifacts(locals())
            try:
                log_path = self._write_failure_log(
                    run_id, technical_details
                )
                partial_artifacts.append(str(log_path))
                report_path = self._write_failure_report(
                    run_id=run_id,
                    exact_error=exact_error,
                    failed_stage=tracker.latest.stage_label,
                    pipeline_status=pipeline_status,
                    original_exception=original_exception,
                    last_completed_stage=(
                        tracker.latest.last_completed_stage
                    ),
                    technical_log=log_path,
                    partial_artifacts=tuple(partial_artifacts),
                )
                partial_artifacts.append(str(report_path))
                self.store.update_run_outputs(
                    run_id,
                    {
                        "failure_log": log_path,
                        "failure_report": report_path,
                    },
                    status="FAILED_DASHBOARD",
                    error_summary=exact_error,
                )
            except Exception:
                LOGGER.exception(
                    "Could not persist dashboard failure details for %s",
                    run_id,
                )
            tracker.fail(
                run_id=run_id,
                error_summary=exact_error,
                technical_details=technical_details,
                partial_artifacts=tuple(partial_artifacts),
                pipeline_status=pipeline_status,
                original_exception=original_exception,
            )
            raise DashboardProcessingError(
                "Processing failed. The run was saved for investigation.",
                run_id=run_id,
                failed_stage=tracker.latest.stage_label,
                last_completed_stage=tracker.latest.last_completed_stage,
                technical_details=technical_details,
                partial_artifacts=tuple(partial_artifacts),
                pipeline_status=pipeline_status,
                original_exception=original_exception,
            ) from error

    @staticmethod
    def _collect_partial_artifacts(scope: Mapping[str, object]) -> list[str]:
        """List real files produced before a run stopped early."""
        artifacts: list[str] = []
        result = scope.get("result")
        output_paths = getattr(result, "output_paths", None)
        if isinstance(output_paths, Mapping):
            artifacts.extend(
                str(path)
                for path in output_paths.values()
                if Path(path).is_file()
            )
        audit_path = scope.get("competitor_audit_path")
        if isinstance(audit_path, Path) and audit_path.is_file():
            artifacts.append(str(audit_path))
        return artifacts

    def _write_cancellation_log(
        self,
        run_id: str,
        *,
        stage_key: str | None,
        stage_label: str | None,
        last_completed_stage: str | None,
        elapsed_seconds: float,
        partial_artifacts: tuple[str, ...],
    ) -> Path:
        """Preserve an auditable record of where the run stopped."""
        directory = self.config.dashboard.output_directory / run_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "cancellation_report.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "CANCELLED",
                    "pipeline_status": CANCELLED_RUN_STATUS,
                    "cancelled_at": datetime.now(timezone.utc).isoformat(),
                    "cancelled_during_stage": stage_key,
                    "cancelled_during_stage_label": stage_label,
                    "last_completed_stage": last_completed_stage,
                    "elapsed_seconds": round(float(elapsed_seconds), 3),
                    "partial_artifacts": list(partial_artifacts),
                    "cancellation_type": "cooperative_checkpoint",
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _write_failure_log(
        self, run_id: str, technical_details: str
    ) -> Path:
        directory = self.config.dashboard.output_directory / run_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "internal_failure.log"
        path.write_text(technical_details, encoding="utf-8")
        return path

    def _write_failure_report(
        self,
        *,
        run_id: str,
        exact_error: str,
        failed_stage: str | None,
        pipeline_status: str,
        original_exception: dict[str, object],
        last_completed_stage: str | None,
        technical_log: Path,
        partial_artifacts: tuple[str, ...],
    ) -> Path:
        directory = self.config.dashboard.output_directory / run_id
        path = directory / "failure_report.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "FAILED",
                    "pipeline_status": pipeline_status,
                    "exact_error": exact_error,
                    "original_exception": original_exception,
                    "failed_stage": failed_stage,
                    "last_completed_stage": last_completed_stage,
                    "technical_log": str(technical_log),
                    "partial_artifacts": list(partial_artifacts),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path
