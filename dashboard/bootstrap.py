"""Shared import-safe dashboard context."""

from __future__ import annotations

import sys
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
SRC = _BOOTSTRAP_ROOT / "src"
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard.services.job_manager import (
    DashboardJobManager,
    get_job_manager,
)
from dashboard.services.processing_service import DashboardProcessingService
from dashboard.services.registry_service import DashboardRegistryService
from dashboard.services.review_service import DashboardReviewService
from dashboard.services.run_service import DashboardRunService
from sku_mapping.config import PipelineConfig, load_config
from sku_mapping.learning.store import LearningStore
from sku_mapping.paths import PROJECT_ROOT

if PROJECT_ROOT != _BOOTSTRAP_ROOT:
    raise RuntimeError(
        "Dashboard bootstrap root does not match the installed project root: "
        f"{_BOOTSTRAP_ROOT} != {PROJECT_ROOT}"
    )


def load_dashboard_context() -> tuple[
    PipelineConfig,
    LearningStore,
    DashboardProcessingService,
    DashboardRegistryService,
    DashboardReviewService,
    DashboardRunService,
    DashboardJobManager,
]:
    """Construct lightweight services; model loading remains lazy."""
    config = load_config(PROJECT_ROOT / "config/default.yaml")
    store = LearningStore(config.learning_store.database_path)
    processing = DashboardProcessingService(config)
    return (
        config,
        store,
        processing,
        DashboardRegistryService(config),
        DashboardReviewService(
            store,
            threshold=config.ml.auto_accept_threshold,
            question_count=config.learning_store.questions_per_run,
            product_master_path=config.data.master_path,
        ),
        DashboardRunService(config, store),
        get_job_manager(store, processing),
    )
