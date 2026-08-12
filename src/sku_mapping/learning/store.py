"""SQLite repository for inference observations and governed review labels."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sku_mapping.learning.migrations import CURRENT_SCHEMA_VERSION, migrate
from sku_mapping.learning.models import (
    HumanReviewAnswer,
    LabelQuality,
    ReviewQuestion,
)
from sku_mapping.learning.review_selection import select_five_reviews
from sku_mapping.paths import portable_repository_path, resolve_portable_path


class LearningStoreError(ValueError):
    """Raised when learning-store state or an attempted transition is invalid."""


class DuplicateHumanReviewError(LearningStoreError):
    """Raised when a persisted question has already been answered."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def _stable_id(namespace: str, *parts: object) -> str:
    content = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
    return f"{namespace}-{digest}"


def _clean_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _encode_output_paths(
    output_paths: Mapping[str, str | Path],
) -> dict[str, str]:
    return {
        str(key): portable_repository_path(value)
        for key, value in output_paths.items()
    }


def _decode_output_paths(output_paths: object) -> dict[str, str]:
    if not isinstance(output_paths, Mapping):
        return {}
    return {
        str(key): str(resolve_portable_path(value))
        for key, value in output_paths.items()
        if isinstance(value, (str, Path)) and str(value).strip()
    }


class LearningStore:
    """Thread-safe repository boundary around a versioned SQLite database."""

    def __init__(self, path: str | Path) -> None:
        # Resolve once at construction so a later working-directory change
        # cannot silently redirect reads and writes to a second database.
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.RLock()
        #: Set once ``_connect`` has proven the schema is current. Until then
        #: even a read must take the migrating path, because migration writes.
        self._migrated = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        migrate(connection)
        self._migrated = True
        return connection

    def _connect_reader(self) -> sqlite3.Connection:
        """Open a connection for SELECT-only work.

        ``migrate`` is skipped deliberately: it writes, and it is the only
        reason a read ever needed the write lock.
        """
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _reading(self):
        """Yield a connection for a SELECT-only query without the write lock.

        The database runs in WAL mode, so a reader sees a consistent snapshot
        while a writer is mid-transaction. Sharing one lock between the two
        made every status poll wait for whatever bulk insert was in flight: a
        3ms read measured 1.7s behind a single 120k-row ``add_predictions``.
        The dashboard polls job state from a 2s fragment, so the live panel -
        including its elapsed clock - froze for the whole of each stage and
        only caught up once the write released the lock.

        Writers keep the lock and stay serialized against each other.
        """
        if not self._migrated:
            # Schema not yet proven current; take the migrating write path.
            with self._lock:
                connection = self._connect()
                try:
                    yield connection
                finally:
                    connection.close()
            return
        connection = self._connect_reader()
        try:
            yield connection
        finally:
            connection.close()

    @property
    def schema_version(self) -> int:
        """Return the applied schema version, initializing on first use."""
        with self._lock:
            connection = self._connect()
            try:
                return int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
            finally:
                connection.close()

    def table_names(self) -> tuple[str, ...]:
        """Return user-owned tables for inspection and tests."""
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
                return tuple(str(row["name"]) for row in rows)
            finally:
                connection.close()

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        for column in (
            "output_paths_json",
            "stage_runtimes_json",
            "run_metadata_json",
        ):
            raw = record.pop(column, "{}")
            try:
                record[column.removesuffix("_json")] = json.loads(raw or "{}")
            except json.JSONDecodeError:
                record[column.removesuffix("_json")] = {}
        record["output_paths"] = _decode_output_paths(
            record.get("output_paths")
        )
        return record

    def get_pipeline_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one durable run record with decoded JSON metadata."""
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM pipeline_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                return self._decode_run(row) if row is not None else None
            finally:
                connection.close()

    def list_pipeline_runs(
        self, *, completed_only: bool = False, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List recent runs for durable dashboard navigation."""
        if limit < 1 or limit > 1000:
            raise LearningStoreError("run list limit must be within [1, 1000]")
        where = "WHERE status LIKE 'COMPLETED%'" if completed_only else ""
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    f"""
                    SELECT * FROM pipeline_runs
                    {where}
                    ORDER BY COALESCE(completed_at, started_at) DESC, run_id
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [self._decode_run(row) for row in rows]
            finally:
                connection.close()

    def completed_runs_for_source_hash(
        self, source_file_hash: str
    ) -> list[dict[str, Any]]:
        """Return completed runs matching an exact uploaded-byte hash."""
        if len(source_file_hash) != 64:
            raise LearningStoreError("source_file_hash must be SHA-256")
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT * FROM pipeline_runs
                    WHERE source_file_hash = ? AND status LIKE 'COMPLETED%'
                    ORDER BY completed_at DESC, run_id
                    """,
                    (source_file_hash,),
                ).fetchall()
                return [self._decode_run(row) for row in rows]
            finally:
                connection.close()

    def active_or_completed_runs_for_source_hash(
        self, source_file_hash: str
    ) -> list[dict[str, Any]]:
        """Find duplicates that are processing or already completed."""
        if len(source_file_hash) != 64:
            raise LearningStoreError("source_file_hash must be SHA-256")
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT * FROM pipeline_runs
                    WHERE source_file_hash = ?
                      AND (
                        status LIKE 'COMPLETED%'
                        OR status IN (
                            'PROCESSING', 'VALIDATING', 'CANCELLING'
                        )
                      )
                    ORDER BY COALESCE(completed_at, started_at) DESC, run_id
                    """,
                    (source_file_hash,),
                ).fetchall()
                return [self._decode_run(row) for row in rows]
            finally:
                connection.close()

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        for column in (
            "completed_stage_keys_json",
            "partial_artifacts_json",
        ):
            raw = record.pop(column, "[]")
            try:
                record[column.removesuffix("_json")] = json.loads(raw or "[]")
            except json.JSONDecodeError:
                record[column.removesuffix("_json")] = []
        record["cancel_requested"] = bool(record.get("cancel_requested"))
        return record

    def create_processing_job(self, values: Mapping[str, Any]) -> str:
        """Insert one QUEUED job before any worker claims it."""
        job_id = str(values.get("job_id") or "").strip()
        if not job_id:
            raise LearningStoreError("processing job requires job_id")
        now = _now()
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO processing_jobs (
                            job_id, run_id, state, cancel_requested,
                            source_filename, source_file_hash,
                            deployment_mode, model_id, created_at, updated_at
                        ) VALUES (
                            :job_id, :run_id, :state, 0,
                            :source_filename, :source_file_hash,
                            :deployment_mode, :model_id, :created_at,
                            :updated_at
                        )
                        """,
                        {
                            "job_id": job_id,
                            "run_id": values.get("run_id"),
                            "state": str(values.get("state") or "QUEUED"),
                            "source_filename": values.get("source_filename"),
                            "source_file_hash": values.get(
                                "source_file_hash"
                            ),
                            "deployment_mode": values.get("deployment_mode"),
                            "model_id": values.get("model_id"),
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
            finally:
                connection.close()
        return job_id

    def get_processing_job(self, job_id: str) -> dict[str, Any] | None:
        """Return one durable job record with decoded JSON columns."""
        with self._reading() as connection:
            row = connection.execute(
                "SELECT * FROM processing_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return self._decode_job(row) if row is not None else None

    def claim_processing_job(
        self,
        job_id: str,
        *,
        run_id: str | None,
        worker_pid: int,
        worker_host: str,
        worker_boot_id: str,
    ) -> bool:
        """Move QUEUED -> RUNNING exactly once and record worker identity."""
        now = _now()
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    cursor = connection.execute(
                        """
                        UPDATE processing_jobs SET
                            state = 'RUNNING',
                            run_id = COALESCE(?, run_id),
                            worker_pid = ?,
                            worker_host = ?,
                            worker_boot_id = ?,
                            heartbeat_at = ?,
                            updated_at = ?
                        WHERE job_id = ? AND state = 'QUEUED'
                        """,
                        (
                            run_id,
                            int(worker_pid),
                            worker_host,
                            worker_boot_id,
                            now,
                            now,
                            job_id,
                        ),
                    )
                return cursor.rowcount == 1
            finally:
                connection.close()

    def record_job_heartbeat(
        self,
        job_id: str,
        *,
        run_id: str | None = None,
        stage_key: str | None = None,
        stage_label: str | None = None,
        stage_detail: str | None = None,
        overall_percent: float | None = None,
        completed_stage_keys: Sequence[str] | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Persist liveness and the latest observable progress snapshot.

        Progress is written by the worker only; the UI never owns it. The
        state column is deliberately untouched so a concurrent cancel request
        cannot be overwritten by an in-flight heartbeat.
        """
        percent = (
            None
            if overall_percent is None
            else max(0.0, min(100.0, float(overall_percent)))
        )
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        UPDATE processing_jobs SET
                            heartbeat_at = ?,
                            updated_at = ?,
                            run_id = COALESCE(?, run_id),
                            stage_key = COALESCE(?, stage_key),
                            stage_label = COALESCE(?, stage_label),
                            stage_detail = COALESCE(?, stage_detail),
                            overall_percent = COALESCE(?, overall_percent),
                            completed_stage_keys_json = COALESCE(
                                ?, completed_stage_keys_json
                            ),
                            elapsed_seconds = COALESCE(?, elapsed_seconds)
                        WHERE job_id = ?
                        """,
                        (
                            _now(),
                            _now(),
                            run_id,
                            stage_key,
                            stage_label,
                            stage_detail,
                            percent,
                            (
                                None
                                if completed_stage_keys is None
                                else _canonical_json(
                                    [str(key) for key in completed_stage_keys]
                                )
                            ),
                            (
                                None
                                if elapsed_seconds is None
                                else max(0.0, float(elapsed_seconds))
                            ),
                            job_id,
                        ),
                    )
            finally:
                connection.close()

    def request_job_cancellation(self, job_id: str) -> str | None:
        """Flag a cancel request and return the resulting durable state.

        A QUEUED job never started a stage, so it becomes CANCELLED at once.
        A RUNNING job becomes CANCELLING and stays owned by its worker until
        the worker observes the flag at its next stage checkpoint. Terminal
        jobs are left untouched and reported back unchanged.
        """
        now = _now()
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    row = connection.execute(
                        "SELECT state FROM processing_jobs WHERE job_id = ?",
                        (job_id,),
                    ).fetchone()
                    if row is None:
                        return None
                    state = str(row["state"])
                    if state in {"CANCELLED", "COMPLETED", "FAILED"}:
                        return state
                    target = "CANCELLED" if state == "QUEUED" else "CANCELLING"
                    connection.execute(
                        """
                        UPDATE processing_jobs SET
                            state = ?,
                            cancel_requested = 1,
                            cancel_requested_at = COALESCE(
                                cancel_requested_at, ?
                            ),
                            finished_at = CASE
                                WHEN ? = 'CANCELLED' THEN COALESCE(
                                    finished_at, ?
                                )
                                ELSE finished_at
                            END,
                            updated_at = ?
                        WHERE job_id = ?
                        """,
                        (target, now, target, now, now, job_id),
                    )
                    return target
            finally:
                connection.close()

    def finish_processing_job(
        self,
        job_id: str,
        *,
        state: str,
        run_id: str | None = None,
        error_summary: str | None = None,
        partial_artifacts: Sequence[str] | None = None,
        overall_percent: float | None = None,
        stage_detail: str | None = None,
    ) -> None:
        """Record one terminal job state and release the active-job slot."""
        if state not in {"CANCELLED", "COMPLETED", "FAILED"}:
            raise LearningStoreError(
                f"{state!r} is not a terminal processing-job state"
            )
        now = _now()
        percent = (
            None
            if overall_percent is None
            else max(0.0, min(100.0, float(overall_percent)))
        )
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        UPDATE processing_jobs SET
                            state = ?,
                            run_id = COALESCE(?, run_id),
                            error_summary = COALESCE(?, error_summary),
                            partial_artifacts_json = COALESCE(
                                ?, partial_artifacts_json
                            ),
                            overall_percent = COALESCE(?, overall_percent),
                            stage_detail = COALESCE(?, stage_detail),
                            heartbeat_at = ?,
                            finished_at = COALESCE(finished_at, ?),
                            updated_at = ?
                        WHERE job_id = ?
                        """,
                        (
                            state,
                            run_id,
                            error_summary,
                            (
                                None
                                if partial_artifacts is None
                                else _canonical_json(
                                    [str(item) for item in partial_artifacts]
                                )
                            ),
                            percent,
                            stage_detail,
                            now,
                            now,
                            now,
                            job_id,
                        ),
                    )
            finally:
                connection.close()

    def active_processing_jobs(self) -> list[dict[str, Any]]:
        """Return every job a worker is still expected to own."""
        with self._reading() as connection:
            rows = connection.execute(
                """
                SELECT * FROM processing_jobs
                WHERE state IN ('QUEUED', 'RUNNING', 'CANCELLING')
                ORDER BY created_at, job_id
                """
            ).fetchall()
            return [self._decode_job(row) for row in rows]

    def active_jobs_for_source_hash(
        self, source_file_hash: str
    ) -> list[dict[str, Any]]:
        """Find non-terminal jobs for exact uploaded bytes.

        Cancelled, completed, and failed jobs are excluded, so a cancelled
        run never blocks the next attempt at the same file.
        """
        if len(source_file_hash) != 64:
            raise LearningStoreError("source_file_hash must be SHA-256")
        with self._reading() as connection:
            rows = connection.execute(
                """
                SELECT * FROM processing_jobs
                WHERE source_file_hash = ?
                  AND state IN ('QUEUED', 'RUNNING', 'CANCELLING')
                ORDER BY created_at, job_id
                """,
                (source_file_hash,),
            ).fetchall()
            return [self._decode_job(row) for row in rows]

    def latest_processing_job(self) -> dict[str, Any] | None:
        """Return the most recently created job regardless of state.

        The dashboard uses this so a browser refresh still shows the outcome
        of the last run instead of an empty page.
        """
        with self._reading() as connection:
            row = connection.execute(
                """
                SELECT * FROM processing_jobs
                ORDER BY created_at DESC, job_id DESC
                LIMIT 1
                """
            ).fetchone()
            return self._decode_job(row) if row is not None else None

    def latest_processing_job_for_run(
        self, run_id: str
    ) -> dict[str, Any] | None:
        """Return the most recent job that produced a given run."""
        with self._reading() as connection:
            row = connection.execute(
                """
                SELECT * FROM processing_jobs
                WHERE run_id = ?
                ORDER BY created_at DESC, job_id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            return self._decode_job(row) if row is not None else None

    def update_run_outputs(
        self,
        run_id: str,
        output_paths: Mapping[str, str | Path],
        *,
        status: str | None = None,
        error_summary: str | None = None,
    ) -> None:
        """Merge validated dashboard artifacts into an existing run."""
        existing = self.get_pipeline_run(run_id)
        if existing is None:
            raise LearningStoreError(f"Unknown pipeline run {run_id!r}")
        merged = {
            **existing.get("output_paths", {}),
            **{key: str(value) for key, value in output_paths.items()},
        }
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        UPDATE pipeline_runs SET
                            output_paths_json = ?,
                            status = COALESCE(?, status),
                            error_summary = COALESCE(?, error_summary),
                            completed_at = COALESCE(completed_at, ?)
                        WHERE run_id = ?
                        """,
                        (
                            _canonical_json(_encode_output_paths(merged)),
                            status,
                            error_summary,
                            _now(),
                            run_id,
                        ),
                    )
            finally:
                connection.close()

    def review_session_for_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the durable review-session record for a run."""
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM review_sessions WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                return dict(row) if row is not None else None
            finally:
                connection.close()

    def review_questions(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Return persisted questions, answers, diagnostics, and candidates."""
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT h.*, p.offer_description,
                           p.source_offer_id, p.source_offer_text,
                           p.entity_id, p.entity_index, p.entity_count,
                           p.entity_text, p.conjunction_type,
                           p.attribute_inheritance_flags,
                           p.entity_parse_confidence,
                           p.candidate_description,
                           p.lightgbm_probability,
                           p.embedding_similarity,
                           p.agreement_status,
                           p.decision_source,
                           p.conflict_flags_json
                    FROM human_reviews h
                    JOIN predictions p
                      ON p.prediction_id = h.prediction_id
                    WHERE h.session_id = ?
                    ORDER BY h.question_number
                    """,
                    (session_id,),
                ).fetchall()
                records = []
                for row in rows:
                    record = dict(row)
                    record["supplied_candidates"] = (
                        self._supplied_candidate_details(
                            connection,
                            str(row["run_id"]),
                            str(row["offer_id"]),
                        )
                    )
                    try:
                        record["conflict_flags"] = json.loads(
                            record.pop("conflict_flags_json") or "[]"
                        )
                    except json.JSONDecodeError:
                        record["conflict_flags"] = []
                    records.append(record)
                return records
            finally:
                connection.close()

    def review_progress(self, session_id: str) -> tuple[int, int]:
        """Return answered and required question counts."""
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT
                        SUM(CASE WHEN answered_at IS NOT NULL THEN 1 ELSE 0 END),
                        COUNT(*)
                    FROM human_reviews WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                return int(row[0] or 0), int(row[1] or 0)
            finally:
                connection.close()

    def list_model_versions(self) -> list[dict[str, Any]]:
        """Return observed model-version metadata for the placeholder page."""
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT * FROM model_versions
                    ORDER BY created_at DESC, model_id
                    """
                ).fetchall()
                records = []
                for row in rows:
                    record = dict(row)
                    try:
                        record["evaluation_summary"] = json.loads(
                            record.pop("evaluation_summary_json") or "{}"
                        )
                    except json.JSONDecodeError:
                        record["evaluation_summary"] = {}
                    records.append(record)
                return records
            finally:
                connection.close()

    def upsert_pipeline_run(self, values: Mapping[str, Any]) -> None:
        """Create or update one run without deleting its dependent records."""
        run_id = str(values.get("run_id") or "").strip()
        if not run_id:
            raise LearningStoreError("pipeline run requires run_id")
        now = _now()
        record = {
            "run_id": run_id,
            "started_at": str(values.get("started_at") or now),
            "completed_at": values.get("completed_at"),
            "source_filename": values.get("source_filename"),
            "source_file_hash": values.get("source_file_hash"),
            "source_row_count": int(values.get("source_row_count") or 0),
            "unique_offer_count": int(values.get("unique_offer_count") or 0),
            "deployment_mode": str(
                values.get("deployment_mode") or "unknown"
            ),
            "status": str(values.get("status") or "STARTED"),
            "model_id": values.get("model_id"),
            "embedding_model_id": values.get("embedding_model_id"),
            "llm_model_id": values.get("llm_model_id"),
            "threshold": _clean_float(values.get("threshold")),
            "output_paths_json": _canonical_json(
                _encode_output_paths(values.get("output_paths") or {})
            ),
            "stage_runtimes_json": _canonical_json(
                values.get("stage_runtimes") or {}
            ),
            "run_metadata_json": _canonical_json(
                values.get("run_metadata") or {}
            ),
            "error_summary": values.get("error_summary"),
            "created_at": str(values.get("created_at") or now),
        }
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO pipeline_runs (
                            run_id, started_at, completed_at, source_filename,
                            source_file_hash, source_row_count,
                            unique_offer_count, deployment_mode, status,
                            model_id, embedding_model_id, llm_model_id,
                            threshold, output_paths_json,
                            stage_runtimes_json, run_metadata_json,
                            error_summary, created_at
                        ) VALUES (
                            :run_id, :started_at, :completed_at,
                            :source_filename, :source_file_hash,
                            :source_row_count, :unique_offer_count,
                            :deployment_mode, :status, :model_id,
                            :embedding_model_id, :llm_model_id, :threshold,
                            :output_paths_json, :stage_runtimes_json,
                            :run_metadata_json, :error_summary, :created_at
                        )
                        ON CONFLICT(run_id) DO UPDATE SET
                            source_filename=COALESCE(
                                excluded.source_filename,
                                pipeline_runs.source_filename
                            ),
                            source_file_hash=COALESCE(
                                excluded.source_file_hash,
                                pipeline_runs.source_file_hash
                            ),
                            source_row_count=CASE
                                WHEN pipeline_runs.source_row_count > 0
                                THEN pipeline_runs.source_row_count
                                ELSE excluded.source_row_count
                            END,
                            unique_offer_count=CASE
                                WHEN pipeline_runs.unique_offer_count > 0
                                THEN pipeline_runs.unique_offer_count
                                ELSE excluded.unique_offer_count
                            END,
                            deployment_mode=excluded.deployment_mode,
                            completed_at=excluded.completed_at,
                            status=excluded.status,
                            model_id=COALESCE(
                                excluded.model_id,
                                pipeline_runs.model_id
                            ),
                            embedding_model_id=excluded.embedding_model_id,
                            llm_model_id=excluded.llm_model_id,
                            threshold=COALESCE(
                                excluded.threshold,
                                pipeline_runs.threshold
                            ),
                            output_paths_json=excluded.output_paths_json,
                            stage_runtimes_json=excluded.stage_runtimes_json,
                            run_metadata_json=CASE
                                WHEN excluded.run_metadata_json != '{}'
                                THEN excluded.run_metadata_json
                                ELSE pipeline_runs.run_metadata_json
                            END,
                            error_summary=excluded.error_summary
                        """,
                        record,
                    )
            finally:
                connection.close()

    def register_model_version(
        self,
        *,
        model_id: str,
        model_hash: str | None,
        status: str,
        parent_model_id: str | None = None,
        training_dataset_id: str | None = None,
        evaluation_summary: Mapping[str, Any] | None = None,
        champion_status: str = "OBSERVED",
        created_at: str | None = None,
        activated_at: str | None = None,
        retired_at: str | None = None,
    ) -> None:
        """Record immutable model identity/provenance without activating it."""
        if not model_id.strip():
            raise LearningStoreError("model_id must be non-empty")
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO model_versions (
                            model_id, model_hash, status, parent_model_id,
                            created_at, training_dataset_id,
                            evaluation_summary_json, champion_status,
                            activated_at, retired_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(model_id) DO UPDATE SET
                            model_hash=COALESCE(
                                model_versions.model_hash,
                                excluded.model_hash
                            ),
                            parent_model_id=COALESCE(
                                model_versions.parent_model_id,
                                excluded.parent_model_id
                            ),
                            training_dataset_id=COALESCE(
                                model_versions.training_dataset_id,
                                excluded.training_dataset_id
                            ),
                            status=CASE
                                WHEN model_versions.champion_status IN (
                                    'CHALLENGER_NOT_ACTIVE',
                                    'APPROVED_CHALLENGER_NOT_ACTIVE',
                                    'REJECTED_CHALLENGER',
                                    'CHAMPION_ACTIVE',
                                    'PREVIOUS_CHAMPION'
                                )
                                THEN model_versions.status
                                ELSE excluded.status
                            END,
                            evaluation_summary_json=CASE
                                WHEN excluded.evaluation_summary_json = '{}'
                                THEN model_versions.evaluation_summary_json
                                ELSE excluded.evaluation_summary_json
                            END,
                            champion_status=CASE
                                WHEN model_versions.champion_status IN (
                                    'CHALLENGER_NOT_ACTIVE',
                                    'APPROVED_CHALLENGER_NOT_ACTIVE',
                                    'REJECTED_CHALLENGER',
                                    'CHAMPION_ACTIVE',
                                    'PREVIOUS_CHAMPION'
                                )
                                THEN model_versions.champion_status
                                ELSE excluded.champion_status
                            END,
                            activated_at=COALESCE(
                                model_versions.activated_at,
                                excluded.activated_at
                            ),
                            retired_at=excluded.retired_at
                        """,
                        (
                            model_id,
                            model_hash,
                            status,
                            parent_model_id,
                            created_at or _now(),
                            training_dataset_id,
                            _canonical_json(evaluation_summary or {}),
                            champion_status,
                            activated_at,
                            retired_at,
                        ),
                    )
            finally:
                connection.close()

    def add_offer_decisions(
        self,
        run_id: str,
        records: Iterable[Mapping[str, Any]],
    ) -> None:
        """Persist exactly one terminal decision per canonical offer."""
        prepared = []
        now = _now()
        seen: set[str] = set()
        for raw in records:
            offer_id = str(raw.get("offer_id") or "").strip()
            if not offer_id:
                raise LearningStoreError("offer decision requires offer_id")
            if offer_id in seen:
                raise LearningStoreError(
                    f"duplicate canonical offer decision {offer_id!r}"
                )
            seen.add(offer_id)
            prepared.append(
                {
                    "run_id": run_id,
                    "offer_id": offer_id,
                    "offer_description": str(
                        raw.get("offer_description") or ""
                    ),
                    "source_offer_id": str(
                        raw.get("source_offer_id") or offer_id
                    ),
                    "source_offer_text": str(
                        raw.get("source_offer_text")
                        or raw.get("offer_description")
                        or ""
                    ),
                    "entity_id": str(raw.get("entity_id") or offer_id),
                    "entity_index": int(raw.get("entity_index") or 1),
                    "entity_count": int(raw.get("entity_count") or 1),
                    "entity_text": str(
                        raw.get("entity_text")
                        or raw.get("offer_description")
                        or ""
                    ),
                    "conjunction_type": str(
                        raw.get("conjunction_type") or "SINGLE"
                    ),
                    "attribute_inheritance_flags": str(
                        raw.get("attribute_inheritance_flags") or ""
                    ),
                    "entity_parse_confidence": _clean_float(
                        raw.get("entity_parse_confidence")
                    ),
                    "source_row_count": max(
                        1, int(raw.get("source_row_count") or 1)
                    ),
                    "is_own_brand": int(bool(raw.get("is_own_brand", True))),
                    "proposed_master_sku": raw.get("proposed_master_sku"),
                    "proposed_master_description": raw.get(
                        "proposed_master_description"
                    ),
                    "proposed_candidate_rank": raw.get(
                        "proposed_candidate_rank"
                    ),
                    "matched_master_sku": raw.get("matched_master_sku"),
                    "final_decision": str(raw.get("final_decision") or ""),
                    "decision_source": str(
                        raw.get("decision_source") or ""
                    ),
                    "final_decision_reason": str(
                        raw.get("final_decision_reason") or ""
                    ),
                    "final_eligible_mapping": int(
                        bool(raw.get("final_eligible_mapping", False))
                    ),
                    "lightgbm_probability": _clean_float(
                        raw.get("lightgbm_probability")
                    ),
                    "embedding_similarity": _clean_float(
                        raw.get("embedding_similarity")
                    ),
                    "embedding_status": raw.get("embedding_status"),
                    "embedding_failure_reason": raw.get(
                        "embedding_failure_reason"
                    ),
                    "created_at": str(raw.get("created_at") or now),
                }
            )
        if not prepared:
            return
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.executemany(
                        """
                        INSERT INTO offer_decisions (
                            run_id, offer_id, offer_description,
                            source_row_count, is_own_brand,
                            proposed_master_sku,
                            proposed_master_description,
                            proposed_candidate_rank, matched_master_sku,
                            final_decision, decision_source,
                            final_decision_reason, final_eligible_mapping,
                            lightgbm_probability, embedding_similarity,
                            embedding_status, embedding_failure_reason,
                            created_at
                        ) VALUES (
                            :run_id, :offer_id, :offer_description,
                            :source_row_count, :is_own_brand,
                            :proposed_master_sku,
                            :proposed_master_description,
                            :proposed_candidate_rank, :matched_master_sku,
                            :final_decision, :decision_source,
                            :final_decision_reason,
                            :final_eligible_mapping,
                            :lightgbm_probability, :embedding_similarity,
                            :embedding_status, :embedding_failure_reason,
                            :created_at
                        )
                        ON CONFLICT(run_id, offer_id) DO NOTHING
                        """,
                        prepared,
                    )
            finally:
                connection.close()

    def add_predictions(
        self, run_id: str, records: Iterable[Mapping[str, Any]]
    ) -> list[str]:
        """Persist candidate-level observations idempotently."""
        created_at = _now()
        prepared: list[dict[str, Any]] = []
        for raw in records:
            offer_id = str(raw.get("offer_id") or "").strip()
            candidate_id = str(raw.get("candidate_id") or "").strip()
            rank = int(raw.get("candidate_rank") or 0)
            if not offer_id or not candidate_id or rank < 1:
                raise LearningStoreError(
                    "prediction requires offer_id, candidate_id, and rank >= 1"
                )
            prediction_id = str(
                raw.get("prediction_id")
                or _stable_id(
                    "prediction", run_id, offer_id, candidate_id, rank
                )
            )
            prepared.append(
                {
                    "prediction_id": prediction_id,
                    "run_id": run_id,
                    "offer_id": offer_id,
                    "offer_description": str(
                        raw.get("offer_description") or ""
                    ),
                    "source_offer_id": str(
                        raw.get("source_offer_id") or offer_id
                    ),
                    "source_offer_text": str(
                        raw.get("source_offer_text")
                        or raw.get("offer_description")
                        or ""
                    ),
                    "entity_id": str(raw.get("entity_id") or offer_id),
                    "entity_index": int(raw.get("entity_index") or 1),
                    "entity_count": int(raw.get("entity_count") or 1),
                    "entity_text": str(
                        raw.get("entity_text")
                        or raw.get("offer_description")
                        or ""
                    ),
                    "conjunction_type": str(
                        raw.get("conjunction_type") or "SINGLE"
                    ),
                    "attribute_inheritance_flags": str(
                        raw.get("attribute_inheritance_flags") or ""
                    ),
                    "entity_parse_confidence": _clean_float(
                        raw.get("entity_parse_confidence")
                    ),
                    "candidate_id": candidate_id,
                    "candidate_description": str(
                        raw.get("candidate_description") or ""
                    ),
                    "candidate_rank": rank,
                    "lightgbm_probability": _clean_float(
                        raw.get("lightgbm_probability")
                    ),
                    "embedding_similarity": _clean_float(
                        raw.get("embedding_similarity")
                    ),
                    "agreement_status": raw.get("agreement_status"),
                    "llm_decision": raw.get("llm_decision"),
                    "llm_confidence": _clean_float(
                        raw.get("llm_confidence")
                    ),
                    "final_decision": str(
                        raw.get("final_decision") or "UNOBSERVED"
                    ),
                    "decision_source": str(
                        raw.get("decision_source") or "UNOBSERVED"
                    ),
                    "conflict_flags_json": _canonical_json(
                        raw.get("conflict_flags") or []
                    ),
                    "feature_snapshot_json": (
                        _canonical_json(raw["feature_snapshot"])
                        if raw.get("feature_snapshot") is not None
                        else None
                    ),
                    "created_at": str(raw.get("created_at") or created_at),
                }
            )
        if not prepared:
            return []
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.executemany(
                        """
                        INSERT INTO predictions (
                            prediction_id, run_id, offer_id,
                            offer_description, source_offer_id,
                            source_offer_text, entity_id, entity_index,
                            entity_count, entity_text, conjunction_type,
                            attribute_inheritance_flags,
                            entity_parse_confidence, candidate_id,
                            candidate_description, candidate_rank,
                            lightgbm_probability, embedding_similarity,
                            agreement_status, llm_decision, llm_confidence,
                            final_decision, decision_source,
                            conflict_flags_json, feature_snapshot_json,
                            created_at
                        ) VALUES (
                            :prediction_id, :run_id, :offer_id,
                            :offer_description, :source_offer_id,
                            :source_offer_text, :entity_id, :entity_index,
                            :entity_count, :entity_text, :conjunction_type,
                            :attribute_inheritance_flags,
                            :entity_parse_confidence, :candidate_id,
                            :candidate_description, :candidate_rank,
                            :lightgbm_probability, :embedding_similarity,
                            :agreement_status, :llm_decision, :llm_confidence,
                            :final_decision, :decision_source,
                            :conflict_flags_json, :feature_snapshot_json,
                            :created_at
                        )
                        ON CONFLICT(prediction_id) DO NOTHING
                        """,
                        prepared,
                    )
            finally:
                connection.close()
        return [str(record["prediction_id"]) for record in prepared]

    def add_automated_label(
        self,
        *,
        prediction_id: str,
        source: str,
        proposed_label: str,
        selected_candidate_id: str | None,
        confidence: float | None,
        label_quality: LabelQuality,
        eligibility_status: str,
        rejection_reason: str | None = None,
    ) -> str:
        """Persist a proposed label with explicit non-training eligibility."""
        if (
            label_quality is LabelQuality.PSEUDO
            and eligibility_status.upper() in {"ELIGIBLE", "INCLUDED"}
        ):
            raise LearningStoreError(
                "PSEUDO labels cannot automatically become training data"
            )
        label_id = _stable_id("label", prediction_id, source)
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO automated_labels (
                            label_id, prediction_id, source, proposed_label,
                            selected_candidate_id, confidence, label_quality,
                            eligibility_status, rejection_reason, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(prediction_id, source) DO NOTHING
                        """,
                        (
                            label_id,
                            prediction_id,
                            source,
                            proposed_label,
                            selected_candidate_id,
                            confidence,
                            label_quality.value,
                            eligibility_status,
                            rejection_reason,
                            _now(),
                        ),
                    )
            finally:
                connection.close()
        return label_id

    def create_review_session(
        self,
        run_id: str,
        *,
        threshold: float = 0.85,
        question_count: int = 5,
    ) -> str | None:
        """Create deterministic post-run questions once per completed run.

        ``question_count`` is the maximum. Runs with fewer reviewable
        own-brand proposals receive a smaller session instead of disappearing
        from Human Validation.
        """
        with self._lock:
            connection = self._connect()
            try:
                run = connection.execute(
                    "SELECT status FROM pipeline_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if run is None:
                    raise LearningStoreError(f"Unknown pipeline run {run_id!r}")
                if not str(run["status"]).startswith("COMPLETED"):
                    return None
                existing = connection.execute(
                    "SELECT session_id FROM review_sessions WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if existing is not None:
                    return str(existing["session_id"])
                rows = connection.execute(
                    """
                    SELECT * FROM predictions
                    WHERE run_id = ?
                    ORDER BY offer_id,
                             lightgbm_probability DESC,
                             candidate_rank,
                             candidate_id
                    """,
                    (run_id,),
                ).fetchall()
                selected = select_five_reviews(
                    (dict(row) for row in rows),
                    threshold=threshold,
                    question_count=question_count,
                )
                if not selected:
                    return None
                session_id = _stable_id("review-session", run_id)
                created_at = _now()
                with connection:
                    connection.execute(
                        """
                        INSERT INTO review_sessions (
                            session_id, run_id, created_at,
                            required_question_count,
                            selected_question_count, status
                        ) VALUES (?, ?, ?, ?, ?, 'OPEN')
                        """,
                        (
                            session_id,
                            run_id,
                            created_at,
                            len(selected),
                            len(selected),
                        ),
                    )
                    for number, item in enumerate(selected, start=1):
                        row = item.prediction
                        review_id = _stable_id(
                            "review", session_id, number, row["offer_id"]
                        )
                        connection.execute(
                            """
                            INSERT INTO human_reviews (
                                review_id, session_id, prediction_id, run_id,
                                offer_id, question_number, question_text,
                                suggested_candidate_id,
                                suggested_candidate_description,
                                selection_category, selection_reason,
                                fallback_reason, review_source
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                review_id,
                                session_id,
                                row["prediction_id"],
                                run_id,
                                row["offer_id"],
                                number,
                                "Is this suggested SKU match correct?",
                                row["candidate_id"],
                                row["candidate_description"],
                                item.category,
                                item.reason,
                                item.fallback_reason,
                                "POST_UPLOAD_FIVE_QUESTION",
                            ),
                        )
                return session_id
            finally:
                connection.close()

    def next_unanswered_question(
        self, session_id: str
    ) -> ReviewQuestion | None:
        """Fetch and mark the next unanswered question as presented."""
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    row = connection.execute(
                        """
                        SELECT * FROM human_reviews
                        WHERE session_id = ? AND answered_at IS NULL
                        ORDER BY question_number
                        LIMIT 1
                        """,
                        (session_id,),
                    ).fetchone()
                    if row is None:
                        return None
                    if row["presented_at"] is None:
                        connection.execute(
                            """
                            UPDATE human_reviews SET presented_at = ?
                            WHERE review_id = ?
                            """,
                            (_now(), row["review_id"]),
                        )
                return ReviewQuestion(
                    review_id=str(row["review_id"]),
                    session_id=str(row["session_id"]),
                    run_id=str(row["run_id"]),
                    question_number=int(row["question_number"]),
                    offer_id=str(row["offer_id"]),
                    offer_description=self._prediction_description(
                        connection, str(row["prediction_id"])
                    ),
                    suggested_candidate_id=str(
                        row["suggested_candidate_id"]
                    ),
                    suggested_candidate_description=str(
                        row["suggested_candidate_description"]
                    ),
                    selection_category=str(row["selection_category"]),
                    selection_reason=str(row["selection_reason"]),
                    fallback_reason=row["fallback_reason"],
                    supplied_candidates=self._supplied_candidates(
                        connection,
                        str(row["run_id"]),
                        str(row["offer_id"]),
                    ),
                )
            finally:
                connection.close()

    @staticmethod
    def _prediction_description(
        connection: sqlite3.Connection, prediction_id: str
    ) -> str:
        row = connection.execute(
            """
            SELECT offer_description FROM predictions
            WHERE prediction_id = ?
            """,
            (prediction_id,),
        ).fetchone()
        return str(row["offer_description"]) if row else ""

    @staticmethod
    def _supplied_candidates(
        connection: sqlite3.Connection, run_id: str, offer_id: str
    ) -> tuple[tuple[str, str], ...]:
        rows = connection.execute(
            """
            SELECT candidate_id, candidate_description
            FROM predictions
            WHERE run_id = ? AND offer_id = ?
            ORDER BY candidate_rank, candidate_id
            """,
            (run_id, offer_id),
        ).fetchall()
        return tuple(
            (str(row["candidate_id"]), str(row["candidate_description"]))
            for row in rows
        )

    @staticmethod
    def _supplied_candidate_details(
        connection: sqlite3.Connection, run_id: str, offer_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT candidate_id, candidate_description, candidate_rank,
                   lightgbm_probability, embedding_similarity,
                   conflict_flags_json
            FROM predictions
            WHERE run_id = ? AND offer_id = ?
            ORDER BY candidate_rank, candidate_id
            """,
            (run_id, offer_id),
        ).fetchall()
        details = []
        for row in rows:
            record = dict(row)
            try:
                record["conflict_flags"] = json.loads(
                    record.pop("conflict_flags_json") or "[]"
                )
            except json.JSONDecodeError:
                record["conflict_flags"] = []
            details.append(record)
        return details

    def save_answer(
        self, review_id: str, answer: HumanReviewAnswer
    ) -> None:
        """Validate and atomically save a single immutable human answer."""
        reviewer = (answer.reviewer_id or "").strip() or None
        corrected = (
            (answer.corrected_candidate_id or "").strip() or None
        )
        resolutions = sum(
            (
                corrected is not None,
                bool(answer.none_of_candidates),
                bool(answer.cannot_determine),
            )
        )
        if answer.is_correct and resolutions:
            raise LearningStoreError(
                "A True answer cannot include a corrective outcome"
            )
        if not answer.is_correct and resolutions != 1:
            raise LearningStoreError(
                "A False answer requires exactly one of corrected candidate, "
                "none of candidates, or cannot determine"
            )
        with self._lock:
            connection = self._connect()
            try:
                review = connection.execute(
                    "SELECT * FROM human_reviews WHERE review_id = ?",
                    (review_id,),
                ).fetchone()
                if review is None:
                    raise LearningStoreError(
                        f"Unknown human review {review_id!r}"
                    )
                if review["answered_at"] is not None:
                    raise DuplicateHumanReviewError(
                        f"Review {review_id!r} is already answered"
                    )
                quality = (
                    LabelQuality.REJECTED
                    if answer.cannot_determine
                    else LabelQuality.GOLD
                )
                with connection:
                    updated = connection.execute(
                        """
                        UPDATE human_reviews SET
                            answered_at = ?,
                            human_answer = ?,
                            predicted_candidate_correct = ?,
                            corrected_candidate_id = ?,
                            none_of_candidates = ?,
                            cannot_determine = ?,
                            reviewer_id = ?,
                            review_source = ?,
                            notes = ?,
                            decomposition_action = ?,
                            corrected_entity_text = ?,
                            corrected_attributes_json = ?,
                            label_quality = ?
                        WHERE review_id = ? AND answered_at IS NULL
                        """,
                        (
                            _now(),
                            int(answer.is_correct),
                            int(answer.is_correct),
                            corrected,
                            int(answer.none_of_candidates),
                            int(answer.cannot_determine),
                            reviewer,
                            answer.review_source,
                            answer.notes,
                            answer.decomposition_action,
                            answer.corrected_entity_text,
                            answer.corrected_attributes_json,
                            quality.value,
                            review_id,
                        ),
                    )
                    if updated.rowcount != 1:
                        raise DuplicateHumanReviewError(
                            f"Review {review_id!r} is already answered"
                        )
                    remaining = int(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM human_reviews
                            WHERE session_id = ? AND answered_at IS NULL
                            """,
                            (review["session_id"],),
                        ).fetchone()[0]
                    )
                    if remaining == 0:
                        connection.execute(
                            """
                            UPDATE review_sessions SET status = 'COMPLETED'
                            WHERE session_id = ?
                            """,
                            (review["session_id"],),
                        )
            finally:
                connection.close()

    def review_session_complete(self, session_id: str) -> bool:
        """Return True only when every persisted question is answered."""
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT required_question_count, selected_question_count,
                           status
                    FROM review_sessions WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                return bool(
                    row
                    and row["status"] == "COMPLETED"
                    and int(row["required_question_count"])
                    == int(row["selected_question_count"])
                )
            finally:
                connection.close()

    def count_new_gold_labels_since_last_model(self) -> int:
        """Count GOLD answers newer than the latest explicit activation.

        Training or rejecting a challenger does not consume the GOLD-label
        trigger. Before the first Phase 7C activation, every GOLD answer is
        considered new.
        """
        with self._lock:
            connection = self._connect()
            try:
                latest = connection.execute(
                    "SELECT MAX(activated_at) FROM model_versions"
                ).fetchone()[0]
                if latest is None:
                    row = connection.execute(
                        """
                        SELECT COUNT(*) FROM human_reviews
                        WHERE label_quality = 'GOLD'
                        """
                    ).fetchone()
                else:
                    row = connection.execute(
                        """
                        SELECT COUNT(*) FROM human_reviews
                        WHERE label_quality = 'GOLD' AND answered_at > ?
                        """,
                        (latest,),
                    ).fetchone()
                return int(row[0])
            finally:
                connection.close()

    def reviewed_labels(self) -> list[dict[str, Any]]:
        """Return transparent human-review exports without PSEUDO labels."""
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT h.*, p.offer_description,
                           p.candidate_description
                    FROM human_reviews h
                    JOIN predictions p ON p.prediction_id = h.prediction_id
                    WHERE h.answered_at IS NOT NULL
                    ORDER BY h.answered_at, h.review_id
                    """
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                connection.close()

    def governed_training_labels(
        self,
        *,
        include_silver: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return GOLD reviews and explicitly requested SILVER proposals.

        PSEUDO and REJECTED labels are intentionally absent from this API.
        Feature snapshots are decoded but never reconstructed from model
        inputs or source identifiers.
        """
        with self._lock:
            connection = self._connect()
            try:
                gold_rows = connection.execute(
                    """
                    SELECT h.*, p.offer_description
                    FROM human_reviews h
                    JOIN predictions p ON p.prediction_id = h.prediction_id
                    WHERE h.answered_at IS NOT NULL
                      AND h.label_quality = 'GOLD'
                    ORDER BY h.answered_at, h.review_id
                    """
                ).fetchall()
                gold: list[dict[str, Any]] = []
                for row in gold_rows:
                    record = dict(row)
                    candidates = connection.execute(
                        """
                        SELECT prediction_id, candidate_id,
                               candidate_description, candidate_rank,
                               feature_snapshot_json
                        FROM predictions
                        WHERE run_id = ? AND offer_id = ?
                        ORDER BY candidate_rank, candidate_id
                        """,
                        (record["run_id"], record["offer_id"]),
                    ).fetchall()
                    record["candidates"] = []
                    for candidate in candidates:
                        item = dict(candidate)
                        try:
                            item["feature_snapshot"] = json.loads(
                                item.pop("feature_snapshot_json") or "{}"
                            )
                        except json.JSONDecodeError as error:
                            raise LearningStoreError(
                                "Stored prediction feature snapshot is invalid"
                            ) from error
                        record["candidates"].append(item)
                    gold.append(record)

                silver: list[dict[str, Any]] = []
                if include_silver:
                    silver_rows = connection.execute(
                        """
                        SELECT a.*, p.run_id, p.offer_id,
                               p.offer_description, p.candidate_id,
                               p.candidate_description,
                               p.feature_snapshot_json
                        FROM automated_labels a
                        JOIN predictions p
                          ON p.prediction_id = a.prediction_id
                        WHERE a.label_quality = 'SILVER'
                          AND a.eligibility_status =
                            'POLICY_QUALIFIED_REVIEW_REQUIRED_BEFORE_TRAINING'
                        ORDER BY a.created_at, a.label_id
                        """
                    ).fetchall()
                    for row in silver_rows:
                        record = dict(row)
                        try:
                            record["feature_snapshot"] = json.loads(
                                record.pop("feature_snapshot_json") or "{}"
                            )
                        except json.JSONDecodeError as error:
                            raise LearningStoreError(
                                "Stored SILVER feature snapshot is invalid"
                            ) from error
                        silver.append(record)
                return {"gold": gold, "silver": silver}
            finally:
                connection.close()

    def get_training_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """Return one decoded immutable training-snapshot record."""
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM training_datasets WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchone()
                if row is None:
                    return None
                record = dict(row)
                for column in (
                    "source_label_counts_json",
                    "included_review_ids_json",
                    "excluded_review_ids_json",
                    "sealed_challenge_exclusion_proof_json",
                    "included_automated_label_ids_json",
                    "inclusion_policy_json",
                    "override_record_json",
                ):
                    record[column.removesuffix("_json")] = json.loads(
                        record.pop(column) or "{}"
                    )
                return record
            finally:
                connection.close()

    def register_training_snapshot(
        self,
        *,
        dataset_id: str,
        created_at: str,
        source_label_counts: Mapping[str, int],
        row_count: int,
        included_review_ids: Sequence[str],
        excluded_review_ids: Sequence[str],
        included_automated_label_ids: Sequence[str],
        content_hash: str,
        feature_schema_version: str,
        artifact_path: str,
        artifact_sha256: str,
        manifest_path: str,
        challenge_exclusion_proof: Mapping[str, Any],
        inclusion_policy: Mapping[str, Any],
        override_record: Mapping[str, Any],
        evaluation_artifact_path: str | None,
        evaluation_artifact_sha256: str | None,
    ) -> None:
        """Insert an immutable materialized snapshot or verify exact identity."""
        payload = {
            "dataset_id": dataset_id,
            "created_at": created_at,
            "source_label_counts_json": _canonical_json(source_label_counts),
            "row_count": int(row_count),
            "included_review_ids_json": _canonical_json(
                sorted(set(included_review_ids))
            ),
            "excluded_review_ids_json": _canonical_json(
                sorted(set(excluded_review_ids))
            ),
            "content_hash": content_hash,
            "sealed_challenge_exclusion_proof_json": _canonical_json(
                challenge_exclusion_proof
            ),
            "feature_schema_version": feature_schema_version,
            "artifact_path": artifact_path,
            "artifact_sha256": artifact_sha256,
            "manifest_path": manifest_path,
            "included_automated_label_ids_json": _canonical_json(
                sorted(set(included_automated_label_ids))
            ),
            "inclusion_policy_json": _canonical_json(inclusion_policy),
            "override_record_json": _canonical_json(override_record),
            "evaluation_artifact_path": evaluation_artifact_path,
            "evaluation_artifact_sha256": evaluation_artifact_sha256,
        }
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    existing = connection.execute(
                        "SELECT * FROM training_datasets WHERE dataset_id = ?",
                        (dataset_id,),
                    ).fetchone()
                    if existing is not None:
                        for key, value in payload.items():
                            if str(existing[key]) != str(value):
                                raise LearningStoreError(
                                    "Immutable training snapshot record differs "
                                    f"for {dataset_id}: {key}"
                                )
                        return
                    duplicate = connection.execute(
                        """
                        SELECT dataset_id FROM training_datasets
                        WHERE content_hash = ?
                        """,
                        (content_hash,),
                    ).fetchone()
                    if duplicate is not None:
                        raise LearningStoreError(
                            "Training snapshot content hash already belongs to "
                            f"{duplicate['dataset_id']}"
                        )
                    connection.execute(
                        """
                        INSERT INTO training_datasets (
                            dataset_id, created_at,
                            source_label_counts_json, row_count,
                            included_review_ids_json,
                            excluded_review_ids_json, content_hash,
                            sealed_challenge_exclusion_proof_json,
                            feature_schema_version, artifact_path,
                            artifact_sha256, manifest_path,
                            included_automated_label_ids_json,
                            inclusion_policy_json, override_record_json,
                            evaluation_artifact_path,
                            evaluation_artifact_sha256
                        ) VALUES (
                            :dataset_id, :created_at,
                            :source_label_counts_json, :row_count,
                            :included_review_ids_json,
                            :excluded_review_ids_json, :content_hash,
                            :sealed_challenge_exclusion_proof_json,
                            :feature_schema_version, :artifact_path,
                            :artifact_sha256, :manifest_path,
                            :included_automated_label_ids_json,
                            :inclusion_policy_json, :override_record_json,
                            :evaluation_artifact_path,
                            :evaluation_artifact_sha256
                        )
                        """,
                        payload,
                    )
            finally:
                connection.close()

    def update_model_lifecycle(
        self,
        *,
        model_id: str,
        status: str,
        champion_status: str,
        evaluation_summary: Mapping[str, Any] | None = None,
        activated_at: str | None = None,
        retired_at: str | None = None,
    ) -> None:
        """Update lifecycle fields without changing immutable model identity."""
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    updated = connection.execute(
                        """
                        UPDATE model_versions SET
                            status = ?,
                            champion_status = ?,
                            evaluation_summary_json = ?,
                            activated_at = COALESCE(?, activated_at),
                            retired_at = ?
                        WHERE model_id = ?
                        """,
                        (
                            status,
                            champion_status,
                            _canonical_json(evaluation_summary or {}),
                            activated_at,
                            retired_at,
                            model_id,
                        ),
                    )
                    if updated.rowcount != 1:
                        raise LearningStoreError(
                            f"Unknown model version {model_id!r}"
                        )
            finally:
                connection.close()

    def export_reviewed_labels(self, destination: str | Path) -> Path:
        """Atomically export completed human reviews as UTF-8-SIG CSV."""
        return self._atomic_csv(self.reviewed_labels(), destination)

    def export_table(
        self, table: str, destination: str | Path
    ) -> Path:
        """Atomically export one allow-listed store table to CSV."""
        allowed = set(self.table_names()) - {"schema_migrations"}
        if table not in allowed:
            raise LearningStoreError(
                f"Unknown or non-exportable table {table!r}"
            )
        with self._lock:
            connection = self._connect()
            try:
                rows = [
                    dict(row)
                    for row in connection.execute(
                        f'SELECT * FROM "{table}"'
                    ).fetchall()
                ]
            finally:
                connection.close()
        return self._atomic_csv(rows, destination)

    @staticmethod
    def _atomic_csv(
        rows: Sequence[Mapping[str, Any]], destination: str | Path
    ) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8-sig", newline=""
            ) as handle:
                fields = list(rows[0]) if rows else []
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def create_training_dataset_record(
        self,
        *,
        included_review_ids: Sequence[str],
        excluded_review_ids: Sequence[str] = (),
        challenge_manifest_paths: Sequence[str | Path] = (),
    ) -> str:
        """Register a prospective dataset only after challenge-set exclusion.

        This records governance metadata; it does not materialize features,
        fit a model, or alter model registry state.
        """
        included = sorted(set(included_review_ids))
        excluded = sorted(set(excluded_review_ids))
        sealed_ids: set[str] = set()
        manifest_proof: list[dict[str, str]] = []
        for raw_path in challenge_manifest_paths:
            path = Path(raw_path)
            content = path.read_bytes()
            try:
                manifest = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise LearningStoreError(
                    f"Unreadable challenge manifest: {path}"
                ) from error
            if manifest.get("status") == "SEALED_UNOPENED":
                for key in (
                    "review_ids",
                    "included_review_ids",
                    "review_record_ids",
                ):
                    values = manifest.get(key, [])
                    if isinstance(values, list):
                        sealed_ids.update(str(value) for value in values)
                hashes = manifest.get("artifact_hashes", {}).get(
                    "review_record_sha256", []
                )
                if isinstance(hashes, list):
                    sealed_ids.update(str(value) for value in hashes)
            manifest_proof.append(
                {
                    "path": portable_repository_path(path),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "status": str(manifest.get("status") or ""),
                }
            )
        overlap = sorted(set(included) & sealed_ids)
        if overlap:
            raise LearningStoreError(
                "Training dataset includes sealed challenge review IDs: "
                f"{overlap}"
            )
        with self._lock:
            connection = self._connect()
            try:
                if included:
                    placeholders = ",".join("?" for _ in included)
                    rows = connection.execute(
                        f"""
                        SELECT review_id, label_quality FROM human_reviews
                        WHERE review_id IN ({placeholders})
                        """,
                        included,
                    ).fetchall()
                    found = {str(row["review_id"]): row for row in rows}
                    missing = sorted(set(included) - set(found))
                    if missing:
                        raise LearningStoreError(
                            f"Unknown included review IDs: {missing}"
                        )
                    non_gold = sorted(
                        review_id
                        for review_id, row in found.items()
                        if row["label_quality"] != LabelQuality.GOLD.value
                    )
                    if non_gold:
                        raise LearningStoreError(
                            "Only GOLD human reviews may be included: "
                            f"{non_gold}"
                        )
                proof = {
                    "proof_version": "phase-7a-v1",
                    "sealed_manifest_checks": manifest_proof,
                    "sealed_review_ids_checked": sorted(sealed_ids),
                    "intersection": [],
                    "challenge_rows_excluded": True,
                }
                content = {
                    "included_review_ids": included,
                    "excluded_review_ids": excluded,
                    "proof": proof,
                }
                content_hash = hashlib.sha256(
                    _canonical_json(content).encode("utf-8")
                ).hexdigest()
                dataset_id = (
                    f"training-dataset-{content_hash[:24]}"
                )
                with connection:
                    connection.execute(
                        """
                        INSERT INTO training_datasets (
                            dataset_id, created_at,
                            source_label_counts_json, row_count,
                            included_review_ids_json,
                            excluded_review_ids_json, content_hash,
                            sealed_challenge_exclusion_proof_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(content_hash) DO NOTHING
                        """,
                        (
                            dataset_id,
                            _now(),
                            _canonical_json(
                                {LabelQuality.GOLD.value: len(included)}
                            ),
                            len(included),
                            _canonical_json(included),
                            _canonical_json(excluded),
                            content_hash,
                            _canonical_json(proof),
                        ),
                    )
                return dataset_id
            finally:
                connection.close()

    def summary(self) -> dict[str, Any]:
        """Return non-sensitive row counts and open review state."""
        with self._lock:
            connection = self._connect()
            try:
                counts = {}
                for table in (
                    "pipeline_runs",
                    "offer_decisions",
                    "predictions",
                    "review_sessions",
                    "human_reviews",
                    "automated_labels",
                    "model_versions",
                    "training_datasets",
                ):
                    counts[table] = int(
                        connection.execute(
                            f'SELECT COUNT(*) FROM "{table}"'
                        ).fetchone()[0]
                    )
                counts["unanswered_reviews"] = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM human_reviews
                        WHERE answered_at IS NULL
                        """
                    ).fetchone()[0]
                )
                counts["new_gold_labels_since_last_model"] = (
                    self.count_new_gold_labels_since_last_model()
                )
                trained_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM model_versions
                        WHERE training_dataset_id IS NOT NULL
                        """
                    ).fetchone()[0]
                )
                latest_run = connection.execute(
                    """
                    SELECT run_id,
                           COALESCE(completed_at, started_at, created_at)
                               AS latest_timestamp
                    FROM pipeline_runs
                    ORDER BY
                        COALESCE(completed_at, started_at, created_at) DESC,
                        run_id DESC
                    LIMIT 1
                    """
                ).fetchone()
                return {
                    "database_path": str(self.path),
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "counts": counts,
                    "stored_run_count": counts["pipeline_runs"],
                    "stored_decision_count": counts["offer_decisions"],
                    "latest_run_id": (
                        str(latest_run["run_id"])
                        if latest_run is not None
                        else None
                    ),
                    "latest_run_timestamp": (
                        str(latest_run["latest_timestamp"])
                        if latest_run is not None
                        else None
                    ),
                    "retraining_performed": trained_count > 0,
                }
            finally:
                connection.close()
