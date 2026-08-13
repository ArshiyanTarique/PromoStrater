"""Ordered, transactional SQLite schema migrations."""

from __future__ import annotations

import sqlite3

CURRENT_SCHEMA_VERSION = 8

MIGRATIONS: dict[int, str] = {
    1: """
        CREATE TABLE pipeline_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            source_filename TEXT,
            source_file_hash TEXT,
            source_row_count INTEGER NOT NULL DEFAULT 0 CHECK(source_row_count >= 0),
            unique_offer_count INTEGER NOT NULL DEFAULT 0 CHECK(unique_offer_count >= 0),
            deployment_mode TEXT NOT NULL,
            status TEXT NOT NULL,
            model_id TEXT,
            embedding_model_id TEXT,
            llm_model_id TEXT,
            threshold REAL,
            output_paths_json TEXT NOT NULL DEFAULT '{}',
            stage_runtimes_json TEXT NOT NULL DEFAULT '{}',
            error_summary TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE predictions (
            prediction_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
            offer_id TEXT NOT NULL,
            offer_description TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_description TEXT NOT NULL,
            candidate_rank INTEGER NOT NULL CHECK(candidate_rank >= 1),
            lightgbm_probability REAL,
            embedding_similarity REAL,
            agreement_status TEXT,
            llm_decision TEXT,
            llm_confidence REAL,
            final_decision TEXT NOT NULL,
            decision_source TEXT NOT NULL,
            conflict_flags_json TEXT NOT NULL DEFAULT '[]',
            feature_snapshot_json TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, offer_id, candidate_id, candidate_rank)
        );

        CREATE TABLE review_sessions (
            session_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            required_question_count INTEGER NOT NULL,
            selected_question_count INTEGER NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE human_reviews (
            review_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES review_sessions(session_id) ON DELETE CASCADE,
            prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
            run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
            offer_id TEXT NOT NULL,
            question_number INTEGER NOT NULL CHECK(question_number >= 1),
            question_text TEXT NOT NULL,
            suggested_candidate_id TEXT NOT NULL,
            suggested_candidate_description TEXT NOT NULL,
            selection_category TEXT NOT NULL,
            selection_reason TEXT NOT NULL,
            fallback_reason TEXT,
            presented_at TEXT,
            answered_at TEXT,
            human_answer INTEGER CHECK(human_answer IN (0, 1)),
            predicted_candidate_correct INTEGER CHECK(predicted_candidate_correct IN (0, 1)),
            corrected_candidate_id TEXT,
            none_of_candidates INTEGER NOT NULL DEFAULT 0 CHECK(none_of_candidates IN (0, 1)),
            cannot_determine INTEGER NOT NULL DEFAULT 0 CHECK(cannot_determine IN (0, 1)),
            reviewer_id TEXT,
            review_source TEXT NOT NULL,
            notes TEXT,
            label_quality TEXT CHECK(label_quality IN ('GOLD', 'SILVER', 'PSEUDO', 'REJECTED')),
            UNIQUE(session_id, question_number),
            UNIQUE(session_id, offer_id)
        );

        CREATE TABLE automated_labels (
            label_id TEXT PRIMARY KEY,
            prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
            source TEXT NOT NULL,
            proposed_label TEXT NOT NULL,
            selected_candidate_id TEXT,
            confidence REAL,
            label_quality TEXT NOT NULL CHECK(label_quality IN ('GOLD', 'SILVER', 'PSEUDO', 'REJECTED')),
            eligibility_status TEXT NOT NULL,
            rejection_reason TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(prediction_id, source)
        );

        CREATE TABLE model_versions (
            model_id TEXT PRIMARY KEY,
            model_hash TEXT,
            status TEXT NOT NULL,
            parent_model_id TEXT,
            created_at TEXT NOT NULL,
            training_dataset_id TEXT,
            evaluation_summary_json TEXT NOT NULL DEFAULT '{}',
            champion_status TEXT NOT NULL,
            activated_at TEXT,
            retired_at TEXT
        );

        CREATE TABLE training_datasets (
            dataset_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            source_label_counts_json TEXT NOT NULL,
            row_count INTEGER NOT NULL CHECK(row_count >= 0),
            included_review_ids_json TEXT NOT NULL,
            excluded_review_ids_json TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            sealed_challenge_exclusion_proof_json TEXT NOT NULL
        );
    """,
    2: """
        CREATE INDEX idx_predictions_run_offer
            ON predictions(run_id, offer_id);
        CREATE INDEX idx_reviews_session_unanswered
            ON human_reviews(session_id, answered_at, question_number);
        CREATE INDEX idx_reviews_quality_answered
            ON human_reviews(label_quality, answered_at);
        CREATE INDEX idx_automated_labels_quality
            ON automated_labels(label_quality, eligibility_status);
        CREATE INDEX idx_model_versions_created
            ON model_versions(created_at);
    """,
    3: """
        ALTER TABLE training_datasets
            ADD COLUMN feature_schema_version TEXT NOT NULL DEFAULT 'unknown';
        ALTER TABLE training_datasets
            ADD COLUMN artifact_path TEXT;
        ALTER TABLE training_datasets
            ADD COLUMN artifact_sha256 TEXT;
        ALTER TABLE training_datasets
            ADD COLUMN manifest_path TEXT;
        ALTER TABLE training_datasets
            ADD COLUMN included_automated_label_ids_json TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE training_datasets
            ADD COLUMN inclusion_policy_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE training_datasets
            ADD COLUMN override_record_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE training_datasets
            ADD COLUMN evaluation_artifact_path TEXT;
        ALTER TABLE training_datasets
            ADD COLUMN evaluation_artifact_sha256 TEXT;

        CREATE INDEX idx_training_datasets_created
            ON training_datasets(created_at);
    """,
    4: """
        ALTER TABLE pipeline_runs
            ADD COLUMN run_metadata_json TEXT NOT NULL DEFAULT '{}';
    """,
    5: """
        CREATE TABLE offer_decisions (
            run_id TEXT NOT NULL
                REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
            offer_id TEXT NOT NULL,
            offer_description TEXT NOT NULL,
            source_row_count INTEGER NOT NULL DEFAULT 1
                CHECK(source_row_count >= 1),
            is_own_brand INTEGER NOT NULL CHECK(is_own_brand IN (0, 1)),
            proposed_master_sku TEXT,
            proposed_master_description TEXT,
            proposed_candidate_rank INTEGER,
            matched_master_sku TEXT,
            final_decision TEXT NOT NULL,
            decision_source TEXT NOT NULL,
            final_decision_reason TEXT NOT NULL,
            final_eligible_mapping INTEGER NOT NULL
                CHECK(final_eligible_mapping IN (0, 1)),
            lightgbm_probability REAL,
            embedding_similarity REAL,
            embedding_status TEXT,
            embedding_failure_reason TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, offer_id)
        );

        CREATE INDEX idx_offer_decisions_run_state
            ON offer_decisions(run_id, final_decision);
    """,
    6: """
        ALTER TABLE predictions ADD COLUMN source_offer_id TEXT;
        ALTER TABLE predictions ADD COLUMN source_offer_text TEXT;
        ALTER TABLE predictions ADD COLUMN entity_id TEXT;
        ALTER TABLE predictions ADD COLUMN entity_index INTEGER;
        ALTER TABLE predictions ADD COLUMN entity_count INTEGER;
        ALTER TABLE predictions ADD COLUMN entity_text TEXT;
        ALTER TABLE predictions ADD COLUMN conjunction_type TEXT;
        ALTER TABLE predictions
            ADD COLUMN attribute_inheritance_flags TEXT;
        ALTER TABLE predictions ADD COLUMN entity_parse_confidence REAL;

        ALTER TABLE human_reviews
            ADD COLUMN decomposition_action TEXT;
        ALTER TABLE human_reviews
            ADD COLUMN corrected_entity_text TEXT;
        ALTER TABLE human_reviews
            ADD COLUMN corrected_attributes_json TEXT;
    """,
    7: """
        CREATE TABLE processing_jobs (
            job_id TEXT PRIMARY KEY,
            run_id TEXT,
            state TEXT NOT NULL CHECK(state IN (
                'QUEUED', 'RUNNING', 'CANCELLING',
                'CANCELLED', 'COMPLETED', 'FAILED'
            )),
            cancel_requested INTEGER NOT NULL DEFAULT 0
                CHECK(cancel_requested IN (0, 1)),
            cancel_requested_at TEXT,
            source_filename TEXT,
            source_file_hash TEXT,
            deployment_mode TEXT,
            model_id TEXT,
            worker_pid INTEGER,
            worker_host TEXT,
            worker_boot_id TEXT,
            heartbeat_at TEXT,
            stage_key TEXT,
            stage_label TEXT,
            stage_detail TEXT,
            overall_percent REAL NOT NULL DEFAULT 0.0
                CHECK(overall_percent >= 0.0 AND overall_percent <= 100.0),
            completed_stage_keys_json TEXT NOT NULL DEFAULT '[]',
            elapsed_seconds REAL NOT NULL DEFAULT 0.0
                CHECK(elapsed_seconds >= 0.0),
            error_summary TEXT,
            partial_artifacts_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT
        );

        CREATE INDEX idx_processing_jobs_state
            ON processing_jobs(state, created_at);
        CREATE INDEX idx_processing_jobs_run
            ON processing_jobs(run_id);
        CREATE INDEX idx_processing_jobs_source_hash
            ON processing_jobs(source_file_hash, state);
    """,
    8: """
        -- Competitor relationships awaiting human judgement.
        --
        -- Competitor discovery today is rules plus a borrowed own-brand
        -- ranker, and neither has ever been measured, because no competitor
        -- pair has ever been labelled. This table is where that ground truth
        -- accumulates: the pipeline proposes, a reviewer decides, and a
        -- future competitor model or threshold is calibrated from what is
        -- recorded here. Nothing reads it for decisions yet - by design.
        --
        -- Deliberately separate from offer_decisions, which answers a
        -- different question (is this offer this SKU) on a different
        -- population. Merging them would blur two label sets that must stay
        -- distinguishable when either is used for training.
        CREATE TABLE competitor_decisions (
            run_id TEXT NOT NULL,
            master_sku TEXT NOT NULL,
            competitor_offer_id TEXT NOT NULL,
            competitor_offer_name TEXT,
            competitor_brand TEXT,
            master_name TEXT,
            -- What the pipeline proposed, and on what evidence.
            proposed_status TEXT NOT NULL,
            proposed_reason TEXT,
            rule_score REAL,
            rule_adjusted_score REAL,
            -- Raw margin, comparable only within one offer's shortlist, and
            -- never a competitor probability. Null when ML was disabled.
            lightgbm_score REAL,
            lightgbm_rank INTEGER,
            ranking_source TEXT,
            model_id TEXT,
            -- What a human said. PENDING until somebody looks at it.
            decision TEXT NOT NULL DEFAULT 'PENDING',
            reviewer TEXT,
            decided_at TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, master_sku, competitor_offer_id),
            CHECK (
                decision IN ('PENDING', 'CONFIRMED', 'REJECTED', 'UNSURE')
            )
        );

        CREATE INDEX idx_competitor_decisions_pending
            ON competitor_decisions(decision, created_at);
        CREATE INDEX idx_competitor_decisions_sku
            ON competitor_decisions(master_sku, decision);
        CREATE INDEX idx_competitor_decisions_run
            ON competitor_decisions(run_id);
    """,
}


def migrate(connection: sqlite3.Connection) -> int:
    """Apply all pending migrations and return the resulting schema version."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
        )
        """
    )
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Learning-store schema {current} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}"
        )
    for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
        script = MIGRATIONS.get(version)
        if script is None:
            raise RuntimeError(f"Missing learning-store migration {version}")
        with connection:
            connection.executescript(script)
            connection.execute(
                "INSERT INTO schema_migrations(version) VALUES (?)",
                (version,),
            )
            connection.execute(f"PRAGMA user_version = {version}")
    return CURRENT_SCHEMA_VERSION
