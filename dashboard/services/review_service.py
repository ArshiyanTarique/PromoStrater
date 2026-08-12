"""Durable bounded review workflow for Streamlit pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sku_mapping.learning.models import HumanReviewAnswer
from sku_mapping.learning.store import LearningStore


class ReviewAnswerError(ValueError):
    """Raised when the UI submits an incomplete or contradictory answer."""


@dataclass(frozen=True)
class ReviewProgress:
    """Current persisted completion state."""

    answered: int
    total: int


class DashboardReviewService:
    """Page-independent review navigation and answer persistence."""

    def __init__(
        self,
        store: LearningStore,
        *,
        threshold: float = 0.85,
        question_count: int = 5,
        product_master_path: str | Path | None = None,
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.question_count = question_count
        self.product_master_path = (
            Path(product_master_path)
            if product_master_path is not None
            else None
        )
        self._master_options: dict[str, str] | None = None

    def master_options(self) -> dict[str, str]:
        """Return every Product Master SKU for unrestricted correction."""
        if self._master_options is not None:
            return dict(self._master_options)
        if self.product_master_path is None:
            return {}
        frame = pd.read_excel(
            self.product_master_path, usecols=["Itemcode", "Itemname"]
        )
        options = {
            str(code).strip(): str(name).strip()
            for code, name in zip(
                frame["Itemcode"], frame["Itemname"], strict=True
            )
            if str(code).strip()
        }
        self._master_options = options
        return dict(options)

    def runs_with_review_sessions(self) -> list[dict[str, object]]:
        runs = self.store.list_pipeline_runs(completed_only=True)
        available = []
        for run in runs:
            run_id = str(run["run_id"])
            session = self.store.review_session_for_run(run_id)
            if session is None:
                # Backfill completed runs created before partial review
                # sessions were supported. Creation is idempotent and never
                # clears or replaces an existing session.
                session_id = self.store.create_review_session(
                    run_id,
                    threshold=self.threshold,
                    question_count=self.question_count,
                )
                if session_id is not None:
                    session = self.store.review_session_for_run(run_id)
            if session is not None:
                available.append({**run, "review_session": session})
        return available

    def questions_for_run(
        self, run_id: str
    ) -> tuple[str, list[dict[str, object]]]:
        session = self.store.review_session_for_run(run_id)
        if session is None:
            raise ReviewAnswerError(
                "This run does not have a reviewable SKU proposal session"
            )
        session_id = str(session["session_id"])
        return session_id, self.store.review_questions(session_id)

    def progress(self, session_id: str) -> ReviewProgress:
        answered, total = self.store.review_progress(session_id)
        return ReviewProgress(answered=answered, total=total)

    def save(
        self,
        *,
        review_id: str,
        answer: str,
        corrected_candidate_id: str | None = None,
        reviewer_id: str | None = None,
        notes: str | None = None,
        decomposition_action: str | None = None,
        corrected_entity_text: str | None = None,
        corrected_attributes_json: str | None = None,
    ) -> None:
        normalized = answer.strip().upper()
        if normalized == "TRUE":
            response = HumanReviewAnswer(
                is_correct=True,
                reviewer_id=reviewer_id,
                notes=notes,
                review_source="STREAMLIT_DASHBOARD",
                decomposition_action=decomposition_action,
                corrected_entity_text=corrected_entity_text,
                corrected_attributes_json=corrected_attributes_json,
            )
        elif normalized == "FALSE_CANDIDATE":
            if not corrected_candidate_id:
                raise ReviewAnswerError(
                    "Select a corrected Product Master SKU"
                )
            master_options = self.master_options()
            if (
                self.product_master_path is not None
                and corrected_candidate_id not in master_options
            ):
                raise ReviewAnswerError(
                    "Corrected SKU is absent from Product Master"
                )
            response = HumanReviewAnswer(
                is_correct=False,
                corrected_candidate_id=corrected_candidate_id,
                reviewer_id=reviewer_id,
                notes=notes,
                review_source="STREAMLIT_DASHBOARD",
                decomposition_action=decomposition_action,
                corrected_entity_text=corrected_entity_text,
                corrected_attributes_json=corrected_attributes_json,
            )
        elif normalized == "FALSE_NONE":
            response = HumanReviewAnswer(
                is_correct=False,
                none_of_candidates=True,
                reviewer_id=reviewer_id,
                notes=notes,
                review_source="STREAMLIT_DASHBOARD",
                decomposition_action=decomposition_action,
                corrected_entity_text=corrected_entity_text,
                corrected_attributes_json=corrected_attributes_json,
            )
        elif normalized == "FALSE_CANNOT_DETERMINE":
            response = HumanReviewAnswer(
                is_correct=False,
                cannot_determine=True,
                reviewer_id=reviewer_id,
                notes=notes,
                review_source="STREAMLIT_DASHBOARD",
                decomposition_action=decomposition_action,
                corrected_entity_text=corrected_entity_text,
                corrected_attributes_json=corrected_attributes_json,
            )
        else:
            raise ReviewAnswerError(
                "False requires candidate, none, or cannot-determine state"
            )
        self.store.save_answer(review_id, response)
